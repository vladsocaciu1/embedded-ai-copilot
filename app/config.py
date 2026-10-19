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


def _require_project_directory(project_path: Path) -> None:
    if not project_path.exists():
        logger.error(f"Project path {project_path} does not exist.")
        raise FileNotFoundError(f"Project path {project_path} does not exist.")
    if not project_path.is_dir():
        logger.error(f"Project path {project_path} is not a directory.")
        raise NotADirectoryError(f"Project path {project_path} is not a directory.")


def _logger_has_handlers(log: logging.Logger) -> bool:
    current: logging.Logger | None = log
    while current is not None:
        if current.handlers:
            return True
        if not current.propagate:
            return False
        current = current.parent
    return False


def _log_project_path(project_path: Path) -> None:
    # load_settings() runs at ai_service import, before setup_logging().
    # With no handlers, INFO is dropped (lastResort is WARNING). Emit once so
    # the startup console and logs/app.log show the chosen directory.
    message = f"Project path {project_path}."
    if _logger_has_handlers(logger):
        logger.info(message)
        return

    # Root stays at WARNING until setup_logging(), so INFO is disabled until
    # this logger's own level allows it.
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | "
        "%(name)s | %(filename)s:%(lineno)d | %(message)s"
    )
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_dir = BASE_DIR / "logs"
    if log_dir.is_dir():
        handlers.append(logging.FileHandler(log_dir / "app.log", encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    previous_propagate = logger.propagate
    logger.propagate = False
    try:
        logger.info(message)
    finally:
        logger.propagate = previous_propagate
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def load_settings(project_path: Path | None = None) -> Settings:
    # Plain load_dotenv() searches from this file, not the process cwd.
    # PROJECT_PATH is read after this so it can live in .env.
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

    _log_project_path(project_path)

    return Settings(
        project_path=project_path,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
    )
