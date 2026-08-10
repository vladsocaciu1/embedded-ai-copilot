import os
import logging

from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)

# Path configurations
PROJECT_NAME = "sample_project"
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_PATH = BASE_DIR / PROJECT_NAME
if not PROJECT_PATH.exists():
    logger.error(f"Project path {PROJECT_PATH} does not exist.")
    raise FileNotFoundError(f"Project path {PROJECT_PATH} does not exist.")

# Load environment variables from .env file
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    logger.error("GEMINI_API_KEY is not set.")
    raise RuntimeError("GEMINI_API_KEY is not set.")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
if not GEMINI_MODEL:
    logger.error("GEMINI_MODEL is not set.")
    raise RuntimeError("GEMINI_MODEL is not set.")