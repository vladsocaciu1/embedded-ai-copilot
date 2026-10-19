import logging
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import errors

from app.code_graph import CodeGraph
from app.config import load_settings
from app.context_selector import pack_files, rank_files
from app.index_cache import IndexCache
from app.models import Answer, AnswerContext, ProjectFile, ProjectIndex
from app.project_loader import load_project_files
from app.prompts import (
    assistant_prompt,
    build_assistant_prompt,
    build_focused_map,
    build_partial_prompt,
    build_project_map,
)

settings = load_settings()

client = genai.Client(api_key=settings.gemini_api_key)

index_cache = IndexCache()

logger = logging.getLogger(__name__)

MAX_CHARS_PER_TOKEN = 10
MAP_BUDGET_SHARE = 0.20
MAP_CHARS_PER_TOKEN = 3
PARTIAL_RESERVE_CHARS = 2_000


class AIServiceError(Exception):
    """Base error raised while generating an answer."""


class ProjectLoadError(AIServiceError):
    """The project path cannot be loaded."""


class ProjectTooLargeError(AIServiceError):
    """The final prompt exceeds MAX_PROMPT_TOKENS."""

    def __init__(self, token_count: int, limit: int, estimated: bool = False):
        self.token_count = token_count
        self.limit = limit
        self.estimated = estimated
        super().__init__(f"Prompt is {token_count} tokens, limit {limit}.")


class UpstreamRejectedError(AIServiceError):
    """Gemini rejected the request (4xx)."""


class EmptyUpstreamResponseError(UpstreamRejectedError):
    """Gemini returned no text or no token count."""


class UpstreamUnavailableError(AIServiceError):
    """Gemini is unavailable (5xx or another API error)."""


@contextmanager
def _upstream_errors():
    try:
        yield
    except errors.ClientError as e:
        # 4xx errors
        logger.exception("Client error.")
        raise UpstreamRejectedError("AI provider rejected the request.") from e
    except errors.ServerError as e:
        # 5xx errors (retry recommended)
        logger.exception("Server error.")
        raise UpstreamUnavailableError("AI provider is unavailable.") from e
    except errors.APIError as e:
        # Any other API error
        logger.exception("API error.")
        raise UpstreamUnavailableError("AI provider is unavailable.") from e


def _count_tokens(prompt: str) -> int:
    with _upstream_errors():
        response = client.models.count_tokens(model=settings.gemini_model, contents=prompt)

    count = response.total_tokens
    if count is None:
        logger.error("Gemini returned no token count.")
        raise EmptyUpstreamResponseError("AI provider returned no token count.")
    return count


def generate_answer(question: str) -> Answer:
    """
    Generate an answer to the given question using Google's Gemini API.

    Args:
        question (str): The question to answer.

    Returns:
        Answer: The generated answer and the context that was sent.
    """
    # Load project path
    project_path = settings.project_path

    # Load project files from the specified project path
    try:
        project_files = load_project_files(settings.project_path, settings.exclude_dirs)
    except ValueError as e:
        logger.exception(f"Invalid project path: {project_path}")
        raise ProjectLoadError("Project files could not be loaded.") from e

    logger.info(f"Loaded {len(project_files)} project files from {project_path}")

    limit = settings.max_prompt_tokens
    cap = int(limit * MAP_BUDGET_SHARE * MAP_CHARS_PER_TOKEN)
    total_chars = sum(len(project_file.content) for project_file in project_files)
    index, graph = index_cache.get(settings.project_path, project_files)

    if total_chars <= limit * MAX_CHARS_PER_TOKEN:
        base_prompt = build_assistant_prompt(project_files, question)
        project_map = build_project_map(index, graph, max_chars=cap)
        logger.info(
            f"Project map: tier {project_map.tier}, {len(project_map.text)} chars (cap {cap})."
        )
        prompt = build_assistant_prompt(
            project_files, question, project_map=project_map.text
        )
        count = _count_tokens(prompt)
        map_tier: str | None = project_map.tier
        if count > limit:
            logger.info("Prompt with the map is over budget; retrying without the map.")
            prompt = base_prompt
            count = _count_tokens(prompt)
            map_tier = None
        if count <= limit:
            return _send(
                prompt,
                count,
                mode="full",
                map_tier=map_tier,
                project_files=project_files,
                included=tuple(project_file.path for project_file in project_files),
                limit=limit,
            )
        logger.info(
            f"Full prompt is {count} tokens (limit {limit}); using partial context."
        )
    else:
        logger.info(
            f"Project is {total_chars} chars, over the "
            f"{limit * MAX_CHARS_PER_TOKEN}-char pre-check; using partial context."
        )

    prompt, count, map_tier, included = _partial_prompt(
        question, project_files, index, graph, limit, cap
    )
    return _send(
        prompt,
        count,
        mode="partial",
        map_tier=map_tier,
        project_files=project_files,
        included=included,
        limit=limit,
    )


def warm_up() -> None:
    started = time.perf_counter()
    try:
        project_files = load_project_files(settings.project_path, settings.exclude_dirs)
    except ValueError:
        logger.warning(
            f"Index warm-up skipped: invalid project path {settings.project_path}"
        )
        return
    index_cache.get(settings.project_path, project_files)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        f"Index warm-up done: {len(project_files)} files in {elapsed_ms} ms."
    )


def _partial_prompt(
    question: str,
    project_files: list[ProjectFile],
    index: ProjectIndex,
    graph: CodeGraph,
    limit: int,
    cap: int,
) -> tuple[str, int, str, tuple[Path, ...]]:
    started = time.perf_counter()
    ranked = rank_files(question, index, graph)
    ranked_ms = round((time.perf_counter() - started) * 1000)
    budget = (
        int(limit * MAP_CHARS_PER_TOKEN)
        - cap
        - len(assistant_prompt)
        - len(question)
        - PARTIAL_RESERVE_CHARS
    )
    files_by_path = {project_file.path: project_file for project_file in project_files}
    included = pack_files(ranked, files_by_path, budget)
    if not included:
        first = ranked[0]
        block = _block(first, files_by_path[first].content)
        estimated = (
            len(assistant_prompt) + len(question) + cap + block
        ) // MAX_CHARS_PER_TOKEN
        logger.warning(
            f"Prompt is at least {estimated} tokens (estimated, limit {limit}); "
            "no selected file fits."
        )
        raise ProjectTooLargeError(estimated, limit, estimated=True)

    project_map, prompt = _build_partial(
        question, project_files, index, graph, ranked, included, limit, cap
    )
    count = _count_tokens(prompt)
    if count > limit:
        need = math.ceil((count - limit) * len(prompt) / count * 1.1)
        dropped = 0
        drop_count = 0
        for path in reversed(included):
            dropped += _block(path, files_by_path[path].content)
            drop_count += 1
            if dropped >= need:
                break
        included = included[:-drop_count]
        if not included:
            logger.warning(f"Prompt is {count} tokens (limit {limit}).")
            raise ProjectTooLargeError(count, limit)
        project_map, prompt = _build_partial(
            question, project_files, index, graph, ranked, included, limit, cap
        )
        count = _count_tokens(prompt)
        if count > limit:
            logger.warning(f"Prompt is {count} tokens (limit {limit}).")
            raise ProjectTooLargeError(count, limit)

    file_chars = sum(len(files_by_path[path].content) for path in included)
    map_files = _focused_file_count(project_map.text)
    logger.info(
        f"Partial context: {len(included)} of {len(project_files)} files, "
        f"{file_chars} chars of files, map {map_files} files "
        f"({len(project_map.text)} chars), {len(ranked)} candidates, "
        f"ranked in {ranked_ms} ms."
    )
    return prompt, count, project_map.tier, included


def _build_partial(
    question: str,
    project_files: list[ProjectFile],
    index: ProjectIndex,
    graph: CodeGraph,
    ranked: tuple[Path, ...],
    included: tuple[Path, ...],
    limit: int,
    cap: int,
):
    project_map = build_focused_map(index, graph, ranked, included, cap)
    prompt = build_partial_prompt(
        project_files, included, question, project_map.text, limit
    )
    return project_map, prompt


def _focused_file_count(text: str) -> int:
    lines = text.splitlines()
    return sum(1 for line in lines[1:] if line and not line.startswith(" "))


def _block(path: Path, content: str) -> int:
    return len(content) + len(path.as_posix()) + 64


def _send(
    prompt: str,
    count: int,
    mode: Literal["full", "partial"],
    map_tier: str | None,
    project_files: list[ProjectFile],
    included: tuple[Path, ...],
    limit: int,
) -> Answer:
    logger.info(f"Prompt is {count} tokens (limit {limit}).")
    logger.info("Sending request to Gemini API.")
    start_time = time.perf_counter()

    try:
        with _upstream_errors():
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )

        logger.info("Received response from Gemini API.")
        # Checked outside the google-genai handlers so an empty body is not
        # remapped to UpstreamUnavailableError or swallowed.
        answer = response.text
        if answer is None or answer == "":
            logger.error("Gemini returned an empty response.")
            raise EmptyUpstreamResponseError("AI provider returned an empty response.")
        if mode == "partial":
            included_posix = tuple(path.as_posix() for path in included)
            answer += (
                "\n\n---\n"
                f"Partial context: this answer is based on {len(included)} of "
                f"{len(project_files)} project files, because the whole project does "
                f"not fit the {limit}-token prompt budget. Files included:\n"
                + "\n".join(f"- {path}" for path in included_posix)
            )
        return Answer(
            text=answer,
            context=AnswerContext(
                mode=mode,
                files_total=len(project_files),
                files_included=tuple(path.as_posix() for path in included),
                prompt_tokens=count,
                map_tier=map_tier,
            ),
        )

    finally:
        elapsed_time = (time.perf_counter() - start_time)*1000  # Convert to milliseconds
        logger.info(f"Gemini API request completed in {elapsed_time:.0f} milliseconds.")
