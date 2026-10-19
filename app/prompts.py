import logging

from pathlib import Path
from app.models import ProjectFile

logger = logging.getLogger(__name__)

assistant_prompt_path = Path(__file__).resolve().parent / "prompts" / "assistant.md"

with assistant_prompt_path.open('r', encoding='utf-8') as f:
    assistant_prompt = f.read()

def build_assistant_prompt(project_files:list[ProjectFile], question: str) -> str:
    """
    Build a prompt that includes the assistant prompt, context, and the user's question.

    Args:
        project_files (list[ProjectFile]): A list of ProjectFile objects containing the path and content of each file.
        question (str): The user's question.

    Returns:
        str: The complete prompt to send to the AI model.
    """

    header = _format_project_header(project_files)
    project_files_text = _format_project_context(project_files)
    parts = [assistant_prompt, header]
    if project_files_text:
        parts.append(project_files_text)
    parts.append(f"Question:\n{question}")
    return "\n\n".join(parts)

def _format_project_header(project_files: list[ProjectFile]) -> str:
    file_count = len(project_files)
    line_count = sum(
        len(project_file.content.splitlines()) for project_file in project_files
    )
    return (
        f"Project context: {file_count} files, {line_count} lines. "
        "Paths are relative to the project root."
    )

def _format_project_context(project_files: list[ProjectFile]) -> str:
    """
    Format the project files into a string that can be included in the prompt.

    Args:
        project_files (list[ProjectFile]): A list of ProjectFile objects.

    Returns:
        str: A formatted string representing the project files contents.
    """
    sections = []
    
    for project_file in project_files:
        line_count = len(project_file.content.splitlines())
        sections.append(f"""========================
File: {project_file.path.as_posix()} ({line_count} lines)
========================
{project_file.content}""")
    
    return "\n".join(sections)
