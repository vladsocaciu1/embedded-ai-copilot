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

def test_build_assistant_prompt_file_count(project_files):
    question = "What does this code do?"

    prompt = build_assistant_prompt(project_files, question)

    assert "2 files" in prompt

def test_build_assistant_prompt_line_count():
    project_files = [
        ProjectFile(path=Path("a.c"), content="one\ntwo\nthree\n"),
        ProjectFile(path=Path("b.h"), content="one\ntwo\n"),
    ]
    question = "What does this code do?"

    prompt = build_assistant_prompt(project_files, question)

    assert "5 lines" in prompt

def test_build_assistant_prompt_per_file_line_count():
    project_files = [
        ProjectFile(path=Path("file1.c"), content="one\ntwo\nthree\n"),
    ]
    question = "What does this code do?"

    prompt = build_assistant_prompt(project_files, question)

    assert "file1.c (3 lines)" in prompt

def test_build_assistant_prompt_posix_path():
    project_files = [
        ProjectFile(path=Path("src") / "can.c", content="int x;\n"),
    ]
    question = "What does this code do?"

    prompt = build_assistant_prompt(project_files, question)

    assert "src/can.c" in prompt

def test_build_assistant_prompt_empty_list_shows_zero_files():
    question = "What does this code do?"

    prompt = build_assistant_prompt([], question)

    assert "0 files" in prompt
    assert question in prompt

