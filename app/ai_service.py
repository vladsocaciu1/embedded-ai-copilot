import logging
import time

from pathlib import Path
from google import genai
from google.genai import errors

from app.config import load_settings
from app.project_loader import load_project_files
from app.prompts import build_assistant_prompt

settings = load_settings()

client = genai.Client(api_key=settings.gemini_api_key)

logger = logging.getLogger(__name__)

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
        project_files = load_project_files(project_path)
    except ValueError:
        logger.exception(f"Invalid project path: {project_path}")
        return "Invalid project path."

    logger.info(f"Loaded {len(project_files)} project files from {project_path}")

    prompt = build_assistant_prompt(project_files, question)

    logger.info(f"Sending request to Gemini API.")
    start_time = time.perf_counter()

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
       
        logger.info("Received response from Gemini API.")
        return response.text
    except errors.ClientError:
        # 4xx errors
        logger.exception(f"Client error.")
        return "Client error occurred."
    except errors.ServerError:
        # 5xx errors (retry recommended)
        logger.exception(f"Server error.")
        return "Server error occurred."
    except errors.APIError:
        # Any other API error
        logger.exception(f"API error.")
        return "API error occurred."
    except Exception:
        # Catch-all for any other exceptions
        logger.exception(f"Unexpected error.")
        return "An unexpected error occurred."
    
    finally:
        elapsed_time = (time.perf_counter() - start_time)*1000  # Convert to milliseconds
        logger.info(f"Gemini API request completed in {elapsed_time:.0f} milliseconds.")