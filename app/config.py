import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT_PATH = BASE_DIR / "sample_project"


@dataclass(frozen=True)
class Settings:
    project_path: Path
    gemini_api_key: str
    gemini_model: str


def load_settings(project_path: Path = DEFAULT_PROJECT_PATH) -> Settings:
    if not project_path.exists():
        logger.error(f"Project path {project_path} does not exist.")
        raise FileNotFoundError(f"Project path {project_path} does not exist.")

    # Plain load_dotenv() searches from this file, not the process cwd.
    load_dotenv(find_dotenv(usecwd=True))

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY is not set.")
        raise RuntimeError("GEMINI_API_KEY is not set.")

    gemini_model = os.getenv("GEMINI_MODEL")
    if not gemini_model:
        logger.error("GEMINI_MODEL is not set.")
        raise RuntimeError("GEMINI_MODEL is not set.")

    return Settings(
        project_path=project_path,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
    )
