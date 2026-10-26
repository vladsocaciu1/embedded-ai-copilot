import logging
from collections.abc import Collection
from fnmatch import fnmatchcase
from pathlib import Path

from app.models import ProjectFile
from app.text_files import text_kind

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
    cp1252_paths: list[Path] = []
    oversize_paths: list[Path] = []
    undecodable_paths: list[Path] = []
    for dirpath, dirnames, filenames in project_path.walk(
        top_down=True, follow_symlinks=False
    ):
        dirnames[:] = [
            name for name in dirnames if not _dir_is_excluded(name, exclude_dirs)
        ]
        for filename in filenames:
            file = dirpath / filename
            if file.suffix.lower() not in SUPPORTED_EXTENSIONS and text_kind(file) is None:
                continue
            relative = file.relative_to(project_path)
            relative_posix = relative.as_posix()
            size = file.stat().st_size
            if size > MAX_FILE_BYTES:
                logger.debug(
                    f"Skipping file over size cap: {relative_posix} ({size} bytes)"
                )
                oversize_paths.append(relative)
                continue
            try:
                with file.open("r", encoding="utf-8-sig") as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with file.open("r", encoding="cp1252") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    logger.debug(
                        "Skipping file that is neither UTF-8 nor cp1252: "
                        f"{relative_posix}"
                    )
                    undecodable_paths.append(relative)
                    continue
                logger.debug(
                    f"Decoded {relative_posix} as cp1252 (not valid UTF-8)."
                )
                cp1252_paths.append(relative)

            project_files.append(ProjectFile(path=relative, content=content))

    project_files.sort(key=lambda f: f.path.parts)
    _log_summary(
        logging.INFO,
        (
            "Decoded {n} {files} as cp1252 (not valid UTF-8); "
            "e.g. {examples}. Per-file list: DEBUG, logs/app.log."
        ),
        cp1252_paths,
    )
    _log_summary(
        logging.WARNING,
        (
            "Skipped {n} {files} over the {kib} KiB size cap; "
            "e.g. {examples}. Per-file list: DEBUG, logs/app.log."
        ),
        oversize_paths,
    )
    _log_summary(
        logging.WARNING,
        (
            "Skipped {n} {files} that are neither UTF-8 nor cp1252; "
            "e.g. {examples}. Per-file list: DEBUG, logs/app.log."
        ),
        undecodable_paths,
    )
    return project_files


def _log_summary(level: int, template: str, paths: list[Path]) -> None:
    count = len(paths)
    if count == 0:
        return
    files = "file" if count == 1 else "files"
    ordered = sorted(paths, key=lambda path: path.parts)
    examples = ", ".join(path.as_posix() for path in ordered[:3])
    extra = count - 3
    if extra > 0:
        examples = f"{examples}, and {extra} more"
    logger.log(
        level,
        template.format(
            n=count,
            files=files,
            examples=examples,
            kib=MAX_FILE_BYTES // 1024,
        ),
    )
