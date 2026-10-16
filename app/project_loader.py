import logging

from pathlib import Path
from app.models import ProjectFile

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".c",
    ".h",
    ".cpp",
    ".hpp",
}

def load_project_files(project_path: Path) -> list[ProjectFile]:
    """
    Load all files from the specified project path.

    Args:
        project_path (Path): The path to the project directory.
    
    Returns:
        list[ProjectFile]: A list of ProjectFile objects containing the path and content of each file.
    """
    if not project_path.exists() or not project_path.is_dir():
        logger.error(f"Invalid project path: {project_path}")
        raise ValueError(f"Invalid project path: {project_path}")
    
    project_files = []
    for file in project_path.rglob('*'):
        if file.is_file() and file.suffix in SUPPORTED_EXTENSIONS:
            try:
                with file.open('r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                logger.warning(f"Skipping non-UTF-8 file: {file}")
                continue

            project_files.append(ProjectFile(path=file, content=content))

    return project_files
