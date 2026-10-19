import logging
import math
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import errors

from app.code_graph import CodeGraph
from app.config import load_settings
from app.context_selector import pack_files, rank_files_detailed
from app.index_cache import IndexCache
from app.models import Answer, AnswerContext, PreparedPrompt, ProjectFile, ProjectIndex
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


def _estimate_tokens(prompt: str) -> int:
    return math.ceil(len(prompt) / MAP_CHARS_PER_TOKEN)


def _prepare(
    question: str,
    counter: Callable[[str], int],
    token_source: Literal["count", "estimate"],
) -> PreparedPrompt:
    project_path = settings.project_path

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
    files_total = len(project_files)

    if total_chars <= limit * MAX_CHARS_PER_TOKEN:
        base_prompt = build_assistant_prompt(project_files, question)
        project_map = build_project_map(index, graph, max_chars=cap)
        logger.info(
            f"Project map: tier {project_map.tier}, {len(project_map.text)} chars (cap {cap})."
        )
        prompt = build_assistant_prompt(
            project_files, question, project_map=project_map.text
        )
        count = counter(prompt)
        map_tier: str | None = project_map.tier
        map_files = project_map.file_count
        if count > limit:
            logger.info("Prompt with the map is over budget; retrying without the map.")
            prompt = base_prompt
            count = counter(prompt)
            map_tier = None
            map_files = 0
        if count <= limit:
            return PreparedPrompt(
                prompt=prompt,
                mode="full",
                limit=limit,
                token_count=count,
                token_source=token_source,
                files_total=files_total,
                included=tuple(project_file.path for project_file in project_files),
                map_tier=map_tier,
                map_files=map_files,
                ranked=(),
            )
        logger.info(
            f"Full prompt is {count} tokens (limit {limit}); using partial context."
        )
    else:
        logger.info(
            f"Project is {total_chars} chars, over the "
            f"{limit * MAX_CHARS_PER_TOKEN}-char pre-check; using partial context."
        )

    started = time.perf_counter()
    ranked = rank_files_detailed(question, index, graph)
    ranked_ms = round((time.perf_counter() - started) * 1000)
    budget = (
        int(limit * MAP_CHARS_PER_TOKEN)
        - cap
        - len(assistant_prompt)
        - len(question)
        - PARTIAL_RESERVE_CHARS
    )
    files_by_path = {project_file.path: project_file for project_file in project_files}
    ranked_paths = tuple(item.path for item in ranked)
    included = pack_files(ranked_paths, files_by_path, budget)
    if not included:
        first = ranked_paths[0]
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
        question, project_files, index, graph, ranked_paths, included, limit, cap
    )
    count = counter(prompt)
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
            question, project_files, index, graph, ranked_paths, included, limit, cap
        )
        count = counter(prompt)
        if count > limit:
            logger.warning(f"Prompt is {count} tokens (limit {limit}).")
            raise ProjectTooLargeError(count, limit)

    file_chars = sum(len(files_by_path[path].content) for path in included)
    logger.info(
        f"Partial context: {len(included)} of {len(project_files)} files, "
        f"{file_chars} chars of files, map {project_map.file_count} files "
        f"({len(project_map.text)} chars), {len(ranked)} candidates, "
        f"ranked in {ranked_ms} ms."
    )
    return PreparedPrompt(
        prompt=prompt,
        mode="partial",
        limit=limit,
        token_count=count,
        token_source=token_source,
        files_total=files_total,
        included=included,
        map_tier=project_map.tier,
        map_files=project_map.file_count,
        ranked=ranked,
    )


def generate_answer(question: str) -> Answer:
    """
    Generate an answer to the given question using Google's Gemini API.

    Args:
        question (str): The question to answer.

    Returns:
        Answer: The generated answer and the context that was sent.
    """
    return _send(_prepare(question, _count_tokens, "count"))


def dry_run(question: str, exact: bool = False) -> PreparedPrompt:
    counter = _count_tokens if exact else _estimate_tokens
    token_source: Literal["count", "estimate"] = "count" if exact else "estimate"
    prepared = _prepare(question, counter, token_source)
    logger.info(
        f"Dry run: mode {prepared.mode}, {prepared.token_count} tokens "
        f"({prepared.token_source}), limit {prepared.limit}, "
        f"{len(prepared.included)} of {prepared.files_total} files."
    )
    return prepared


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


def _block(path: Path, content: str) -> int:
    return len(content) + len(path.as_posix()) + 64


def _send(prepared: PreparedPrompt) -> Answer:
    logger.info(f"Prompt is {prepared.token_count} tokens (limit {prepared.limit}).")
    logger.info("Sending request to Gemini API.")
    start_time = time.perf_counter()

    try:
        with _upstream_errors():
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prepared.prompt,
            )

        logger.info("Received response from Gemini API.")
        # Checked outside the google-genai handlers so an empty body is not
        # remapped to UpstreamUnavailableError or swallowed.
        answer = response.text
        if answer is None or answer == "":
            logger.error("Gemini returned an empty response.")
            raise EmptyUpstreamResponseError("AI provider returned an empty response.")
        if prepared.mode == "partial":
            k = len(prepared.included)
            n = prepared.files_total
            limit = prepared.limit
            answer += (
                "\n\n---\n"
                f"Partial context: this answer is based on {k} of {n} project files, "
                f"because the whole project does not fit the {limit}-token prompt budget. "
                "The included files are listed in context.files_included."
            )
        return Answer(
            text=answer,
            context=AnswerContext(
                mode=prepared.mode,
                files_total=prepared.files_total,
                files_included=tuple(path.as_posix() for path in prepared.included),
                prompt_tokens=prepared.token_count,
                map_tier=prepared.map_tier,
            ),
        )

    finally:
        elapsed_time = (time.perf_counter() - start_time)*1000  # Convert to milliseconds
        logger.info(f"Gemini API request completed in {elapsed_time:.0f} milliseconds.")
