import logging
import time
from contextlib import contextmanager

from google import genai
from google.genai import errors

from app.config import load_settings
from app.project_loader import load_project_files
from app.prompts import build_assistant_prompt

settings = load_settings()

client = genai.Client(api_key=settings.gemini_api_key)

logger = logging.getLogger(__name__)

MAX_CHARS_PER_TOKEN = 10


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


def _check_prompt_budget(prompt: str) -> None:
    limit = settings.max_prompt_tokens
    if len(prompt) > limit * MAX_CHARS_PER_TOKEN:
        raise ProjectTooLargeError(len(prompt) // MAX_CHARS_PER_TOKEN, limit, estimated=True)

    with _upstream_errors():
        response = client.models.count_tokens(model=settings.gemini_model, contents=prompt)

    count = response.total_tokens
    if count is None:
        logger.error("Gemini returned no token count.")
        raise EmptyUpstreamResponseError("AI provider returned no token count.")
    if count > limit:
        logger.warning(f"Prompt is {count} tokens (limit {limit}).")
        raise ProjectTooLargeError(count, limit, estimated=False)
    logger.info(f"Prompt is {count} tokens (limit {limit}).")


def generate_answer(question: str) -> str:
    """
    Generate an answer to the given question using Google's Gemini API.

    Args:
        question (str): The question to answer.

    Returns:
        str: The generated answer.
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

    prompt = build_assistant_prompt(project_files, question)

    _check_prompt_budget(prompt)

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
        return answer

    finally:
        elapsed_time = (time.perf_counter() - start_time)*1000  # Convert to milliseconds
        logger.info(f"Gemini API request completed in {elapsed_time:.0f} milliseconds.")
