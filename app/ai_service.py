import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=gemini_key)

def generate_answer(question: str) -> str:
    """
    Generate an answer to the given question using Google's Gemini API.

    Args:
        question (str): The question to answer.

    Returns:
        str: The generated answer.
    """
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=question,
    )
   
    return response.text