import pytest

from pathlib import Path
from app.prompts import build_assistant_prompt
from app.models import ProjectFile

@pytest.fixture 
def project_files():
    return [
        ProjectFile(path=Path("file1.c"), content="int main() { return 0; }"),
        ProjectFile(path=Path("file2.h"), content="#define MAX 100"),
    ]

def test_build_assistant_prompt_file_paths(project_files):

    question = "What does this code do?"

    # Build the prompt
    prompt = build_assistant_prompt(project_files, question)

    # Check that the prompt contains the project files, porject file contents and question
    assert "file1.c" in prompt
    assert "file2.h" in prompt

def test_build_assistant_prompt_file_content(project_files):
    question = "What does this code do?"

    # Build the prompt
    prompt = build_assistant_prompt(project_files, question)

    # Check that the prompt contains the project file contents
    assert "int main() { return 0; }" in prompt
    assert "#define MAX 100" in prompt

def test_build_assistant_prompt_question(project_files):
    question = "What does this code do?"

    # Build the prompt
    prompt = build_assistant_prompt(project_files, question)

    # Check that the prompt contains the user's question
    assert "Question:" in prompt
    assert question in prompt

def test_build_assistant_prompt_assistant_prompt_inclusion(project_files):
    question = "What does this code do?"

    # Build the prompt
    prompt = build_assistant_prompt(project_files, question)

    # Check that the prompt contains the assistant prompt
    assert "You are an expert embedded software engineer specializing in automotive C development." in prompt 

def test_build_assistant_prompt_empty_files():
    # Test with no project files
    project_files = []
    question = "What does this code do?"

    # Build the prompt
    prompt = build_assistant_prompt(project_files, question)

    # Check that the prompt contains the question and no file content
    assert "Question:" in prompt
    assert question in prompt

