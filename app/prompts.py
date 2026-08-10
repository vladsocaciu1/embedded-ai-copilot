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

    project_files_text = _format_project_context(project_files)
    
    return f"{assistant_prompt}\n\n{project_files_text}\n\nQuestion:\n{question}"

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
        sections.append(f"""========================
File: {project_file.path}
========================
{project_file.content}""")
    
    return"\n".join(sections)
