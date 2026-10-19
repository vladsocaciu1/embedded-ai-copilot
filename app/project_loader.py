import logging
from collections.abc import Collection
from fnmatch import fnmatchcase
from pathlib import Path

from app.models import ProjectFile

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".cxx",
    ".hh",
    ".hxx",
    ".inl",
    ".s",
    ".asm",
    ".ld",
}

MAX_FILE_BYTES = 512 * 1024


def _dir_is_excluded(name: str, exclude_dirs: Collection[str]) -> bool:
    if name.startswith("."):
        return True
    lowered = name.lower()
    for pattern in ("build", "out", "cmake-build-*"):
        if fnmatchcase(lowered, pattern.lower()):
            return True
    for pattern in exclude_dirs:
        if fnmatchcase(lowered, pattern.lower()):
            return True
    return False


def load_project_files(
    project_path: Path, exclude_dirs: Collection[str] = ()
) -> list[ProjectFile]:
    """
    Load source files from the specified project path.

    Args:
        project_path (Path): The path to the project directory.
        exclude_dirs: Directory-name globs to skip below the root.

    Returns:
        list[ProjectFile]: A list of ProjectFile objects containing the path and content of each file.
    """
    if not project_path.exists() or not project_path.is_dir():
        logger.error(f"Invalid project path: {project_path}")
        raise ValueError(f"Invalid project path: {project_path}")

    project_files = []
    for dirpath, dirnames, filenames in project_path.walk(
        top_down=True, follow_symlinks=False
    ):
        dirnames[:] = [
            name for name in dirnames if not _dir_is_excluded(name, exclude_dirs)
        ]
        for filename in filenames:
            file = dirpath / filename
            if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            size = file.stat().st_size
            if size > MAX_FILE_BYTES:
                logger.warning(f"Skipping file over size cap: {file} ({size} bytes)")
                continue
            try:
                with file.open("r", encoding="utf-8-sig") as f:
                    content = f.read()
            except UnicodeDecodeError:
                logger.warning(f"Skipping non-UTF-8 file: {file}")
                continue

            project_files.append(ProjectFile(path=file, content=content))

    return project_files
