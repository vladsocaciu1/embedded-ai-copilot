import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
DEFAULT_MAX_PROMPT_TOKENS = 200_000
DEFAULT_SEMANTIC_SEARCH: Literal["off", "local"] = "local"
SEMANTIC_SEARCH_MODES = ("off", "local")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_CACHE_DIR = BASE_DIR / ".cache"


@dataclass(frozen=True)
class EmbeddingSettings:
    semantic_search: Literal["off", "local"] = DEFAULT_SEMANTIC_SEARCH
    model: str = DEFAULT_EMBEDDING_MODEL
    cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR


@dataclass(frozen=True)
class Settings:
    project_path: Path
    gemini_api_key: str
    gemini_model: str
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS
    embedding: EmbeddingSettings = EmbeddingSettings()


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


def _max_prompt_tokens_from_env() -> int:
    raw = os.getenv("MAX_PROMPT_TOKENS")
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_PROMPT_TOKENS
    value = raw.strip()
    if re.fullmatch(r"[0-9]+", value) is None or int(value) <= 0:
        logger.error("MAX_PROMPT_TOKENS must be a positive integer.")
        raise ValueError("MAX_PROMPT_TOKENS must be a positive integer.")
    return int(value)


def _semantic_search_from_env() -> Literal["off", "local"]:
    raw = os.getenv("SEMANTIC_SEARCH")
    if raw is None or raw.strip() == "":
        return DEFAULT_SEMANTIC_SEARCH
    mode = raw.strip().lower()
    if mode not in SEMANTIC_SEARCH_MODES:
        logger.error("SEMANTIC_SEARCH must be one of: off, local.")
        raise ValueError("SEMANTIC_SEARCH must be one of: off, local.")
    if mode == "local":
        return "local"
    return "off"


def _embedding_model_from_env() -> str:
    raw = os.getenv("EMBEDDING_MODEL")
    if raw is None or raw.strip() == "":
        return DEFAULT_EMBEDDING_MODEL
    return raw.strip()


def _embedding_cache_dir_from_env() -> Path:
    raw = os.getenv("EMBEDDING_CACHE_DIR")
    if raw is None or raw.strip() == "":
        return DEFAULT_EMBEDDING_CACHE_DIR
    path = _resolve_project_path(raw)
    if path.exists() and not path.is_dir():
        logger.error(f"EMBEDDING_CACHE_DIR {path} is not a directory.")
        raise NotADirectoryError(f"EMBEDDING_CACHE_DIR {path} is not a directory.")
    return path


def load_embedding_settings() -> EmbeddingSettings:
    # Same dotenv lookup as load_settings. Scripts call this without GEMINI_* or a project path.
    load_dotenv(find_dotenv(usecwd=True))
    return EmbeddingSettings(
        semantic_search=_semantic_search_from_env(),
        model=_embedding_model_from_env(),
        cache_dir=_embedding_cache_dir_from_env(),
    )


def _require_project_directory(project_path: Path) -> None:
    if not project_path.exists():
        logger.error(f"Project path {project_path} does not exist.")
        raise FileNotFoundError(f"Project path {project_path} does not exist.")
    if not project_path.is_dir():
        logger.error(f"Project path {project_path} is not a directory.")
        raise NotADirectoryError(f"Project path {project_path} is not a directory.")


def load_settings(project_path: Path | None = None) -> Settings:
    # Plain load_dotenv() searches from this file, not the process cwd.
    # PROJECT_PATH, EXCLUDE_DIRS, and MAX_PROMPT_TOKENS are read after this so they can live in .env.
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
    max_prompt_tokens = _max_prompt_tokens_from_env()
    embedding = load_embedding_settings()

    logger.info(f"Project path {project_path}.")
    if embedding.semantic_search == "local":
        logger.info(
            f"Semantic search local (model {embedding.model}); used for partial context."
        )

    return Settings(
        project_path=project_path,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        exclude_dirs=exclude_dirs,
        max_prompt_tokens=max_prompt_tokens,
        embedding=embedding,
    )
