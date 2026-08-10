import logging

from pathlib import Path
from google import genai
from google.genai import errors

from app.config import PROJECT_PATH, GEMINI_KEY, GEMINI_MODEL
from app.project_loader import load_project_files
from app.prompts import build_assistant_prompt

client = genai.Client(api_key=GEMINI_KEY)

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
    project_path = PROJECT_PATH

    # Load project files from the specified project path
    project_files = load_project_files(project_path)

    # Build the prompt using the loaded project files and the user's question
    prompt = build_assistant_prompt(project_files, question)

    #Prompt output for testing
    output = Path("test/generated_prompt_test.md")
    output.write_text(prompt, encoding="utf-8")

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
       
        return response.text
    except errors.ClientError as e:
        # 4xx errors
        logger.error(f"Client error: {e}")
        return "Client error occurred."
    except errors.ServerError as e:
        # 5xx errors (retry recommended)
        logger.error(f"Server error: {e}")
        return "Server error occurred."
    except errors.APIError as e:
        # Any other API error
        logger.error(f"API error: {e}")
        return "API error occurred."
    except Exception as e:
        # Catch-all for any other exceptions
        logger.error(f"Unexpected error: {e}")
        return "An unexpected error occurred."