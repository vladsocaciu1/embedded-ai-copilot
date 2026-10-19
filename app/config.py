import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT_PATH = BASE_DIR / "sample_project"
DEFAULT_EXCLUDE_DIRS = (
    "CMSIS",
    "*_HAL_Driver",
    "Middlewares",
    "third_party",
    "vendor",
    "external",
    "GenData",
)


@dataclass(frozen=True)
class Settings:
    project_path: Path
    gemini_api_key: str
    gemini_model: str
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS


def _resolve_project_path(raw: str | Path) -> Path:
    # expanduser() first. Relative paths anchor at BASE_DIR (repo root), not cwd.
    # BASE_DIR is read here so tests can monkeypatch the module global.
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _project_path_from_env() -> Path:
    configured = os.getenv("PROJECT_PATH", "").strip()
    if not configured:
        return DEFAULT_PROJECT_PATH
    return _resolve_project_path(configured)


def _exclude_dirs_from_env() -> tuple[str, ...]:
    raw = os.getenv("EXCLUDE_DIRS")
    if raw is None or raw.strip() == "":
        return DEFAULT_EXCLUDE_DIRS
    entries = tuple(part.strip() for part in raw.split(",") if part.strip())
    for entry in entries:
        if "/" in entry or "\\" in entry:
            logger.error(f"EXCLUDE_DIRS entry {entry} must be a directory name.")
            raise ValueError(f"EXCLUDE_DIRS entry {entry} must be a directory name.")
    return entries


def _require_project_directory(project_path: Path) -> None:
    if not project_path.exists():
        logger.error(f"Project path {project_path} does not exist.")
        raise FileNotFoundError(f"Project path {project_path} does not exist.")
    if not project_path.is_dir():
        logger.error(f"Project path {project_path} is not a directory.")
        raise NotADirectoryError(f"Project path {project_path} is not a directory.")


def load_settings(project_path: Path | None = None) -> Settings:
    # Plain load_dotenv() searches from this file, not the process cwd.
    # PROJECT_PATH and EXCLUDE_DIRS are read after this so they can live in .env.
    load_dotenv(find_dotenv(usecwd=True))

    # Explicit argument, then PROJECT_PATH, then DEFAULT_PROJECT_PATH.
    if project_path is None:
        project_path = _project_path_from_env()
    else:
        project_path = _resolve_project_path(project_path)

    _require_project_directory(project_path)

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY is not set.")
        raise RuntimeError("GEMINI_API_KEY is not set.")

    gemini_model = os.getenv("GEMINI_MODEL")
    if not gemini_model:
        logger.error("GEMINI_MODEL is not set.")
        raise RuntimeError("GEMINI_MODEL is not set.")

    exclude_dirs = _exclude_dirs_from_env()

    logger.info(f"Project path {project_path}.")

    return Settings(
        project_path=project_path,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        exclude_dirs=exclude_dirs,
    )
