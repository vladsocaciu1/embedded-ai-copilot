import logging
import time

from google import genai
from google.genai import errors

from app.config import load_settings
from app.project_loader import load_project_files
from app.prompts import build_assistant_prompt

settings = load_settings()

client = genai.Client(api_key=settings.gemini_api_key)

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """Base error raised while generating an answer."""


class ProjectLoadError(AIServiceError):
    """The project path cannot be loaded."""


class UpstreamRejectedError(AIServiceError):
    """Gemini rejected the request (4xx)."""


class EmptyUpstreamResponseError(UpstreamRejectedError):
    """Gemini returned no text (blocked or empty)."""


class UpstreamUnavailableError(AIServiceError):
    """Gemini is unavailable (5xx or another API error)."""


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

    logger.info("Sending request to Gemini API.")
    start_time = time.perf_counter()

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
       
        logger.info("Received response from Gemini API.")
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
    else:
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
