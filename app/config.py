
import os

from dotenv import load_dotenv
from pathlib import Path

# Path configurations
PROJECT_NAME = "sample_project"
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_PATH_CFG = BASE_DIR / PROJECT_NAME

# Load environment variables from .env file
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
if not GEMINI_MODEL:
    raise RuntimeError("GEMINI_MODEL is not set.")