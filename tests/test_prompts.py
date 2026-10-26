import pytest

from pathlib import Path
from app.code_graph import build_graph
from app.code_index import build_index
from app.models import ProjectFile, SymbolKind
from app.prompts import (
    assistant_prompt,
    build_assistant_prompt,
    build_focused_map,
    build_partial_prompt,
    build_project_map,
)

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
    assert (
        "You are an expert embedded software engineer who knows C and C++ "
        "for microcontrollers and embedded systems: drivers, HALs, RTOS code, "
        "interrupts, memory-mapped registers and linker scripts."
    ) in prompt
    assert "project map" in prompt.lower() 

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


def _indexed(files):
    index = build_index(files)
    return index, build_graph(index)


@pytest.fixture
def mapped_files():
    return [
        ProjectFile(
            path=Path("src/unit.c"),
            content=(
                '#include <stdint.h>\n'
                '#include "stm32f4xx_hal.h"\n'
                '#include "can.h"\n'
                "typedef struct\n"
                "{\n"
                "    int id;\n"
                "} FrameType;\n"
                "#define MACRO_ONLY_NAME 1\n"
                "int global_counter = 1;\n"
                "static int static_var = 0;\n"
                "static void static_fn(void)\n"
                "{\n"
                "    return;\n"
                "}\n"
                "void exported_fn(void)\n"
                "{\n"
                "    return;\n"
                "}\n"
            ),
        ),
        ProjectFile(path=Path("inc/can.h"), content="void Can_Init(void);\n"),
        ProjectFile(path=Path("boot.s"), content="nop\n"),
        ProjectFile(
            path=Path("src/bad.c"),
            content="int broken(int x){ return x + ; }\n",
        ),
    ]


def test_build_project_map_full_tier_contains_paths_and_symbols(mapped_files):
    index, graph = _indexed(mapped_files)

    project_map = build_project_map(index, graph, 10**9)

    assert project_map.tier == "full"
    text = project_map.text
    for project_file in mapped_files:
        assert project_file.path.as_posix() in text
    function = next(
        symbol
        for file in index.files
        for symbol in file.symbols
        if symbol.kind == SymbolKind.FUNCTION
        and not symbol.is_static
        and symbol.name == "exported_fn"
        and symbol.start_line != symbol.end_line
    )
    assert f"{function.name} {function.start_line}-{function.end_line}" in text
    assert "FrameType" in text
    assert "MACRO_ONLY_NAME" in text
    assert "global_counter" in text
    assert "Can_Init" in text


def test_build_project_map_full_tier_contains_static_symbols(mapped_files):
    index, graph = _indexed(mapped_files)

    text = build_project_map(index, graph, 10**9).text

    static_line = next(line for line in text.splitlines() if line.startswith("  static:"))
    assert "static_fn" in static_line
    assert "static_var" in static_line


def test_build_project_map_external_headers_skip_resolved_includes(mapped_files):
    index, graph = _indexed(mapped_files)

    text = build_project_map(index, graph, 10**9).text
    external = next(
        line for line in text.splitlines() if line.startswith("External headers")
    )

    assert "<stdint.h>" in external
    assert "stm32f4xx_hal.h" in external
    assert "<stm32f4xx_hal.h>" not in external
    assert "can.h" not in external


def test_build_project_map_markers_for_unparsed_and_errors(mapped_files):
    index, graph = _indexed(mapped_files)

    text = build_project_map(index, graph, 10**9).text

    assert "boot.s (not parsed)" in text
    assert "src/bad.c (parse errors; may be incomplete)" in text


def test_build_project_map_tiers_shrink_to_fit(mapped_files):
    index, graph = _indexed(mapped_files)
    full = build_project_map(index, graph, 10**9)
    full_len = len(full.text)

    assert build_project_map(index, graph, full_len).tier == "full"
    shrunk = build_project_map(index, graph, full_len - 1)
    assert shrunk.tier != "full"
    assert "MACRO_ONLY_NAME" not in shrunk.text
    assert len(shrunk.text) <= full_len - 1

    current = shrunk
    while current.tier != "no_statics":
        current = build_project_map(index, graph, len(current.text) - 1)
        assert current.tier != "omitted"
    assert "static_fn" not in current.text
    assert "static_var" not in current.text

    while current.tier != "counts":
        current = build_project_map(index, graph, len(current.text) - 1)
        assert current.tier != "omitted"
    assert "exported_fn" not in current.text
    assert "src/unit.c" in current.text
    assert "inc/can.h" in current.text

    omitted = build_project_map(index, graph, 0)
    assert omitted.tier == "omitted"
    assert "omitted" in omitted.text


def test_build_project_map_is_deterministic(mapped_files):
    index, graph = _indexed(mapped_files)
    first = build_project_map(index, graph, 10**9).text
    second = build_project_map(index, graph, 10**9).text
    shuffled_index, shuffled_graph = _indexed(list(reversed(mapped_files)))
    shuffled = build_project_map(shuffled_index, shuffled_graph, 10**9).text

    assert first == second == shuffled


def test_build_assistant_prompt_places_map_or_omits_it(project_files):
    question = "What does this code do?"
    map_text = "Project map (full): 1 files, 1 functions. marker-for-order"
    prompt = build_assistant_prompt(project_files, question, project_map=map_text)

    assert prompt.find("\n") < prompt.find(map_text) < prompt.find("File:")
    assert prompt.find(map_text) < prompt.find("Question:")

    without_map = build_assistant_prompt(project_files, question)
    assert "Project map" not in without_map
    assert build_assistant_prompt(project_files, question, project_map=None) == without_map
    assert build_assistant_prompt(project_files, question, project_map="") == without_map


def _focused_files():
    return [
        ProjectFile(
            path=Path("src/a.c"),
            content="void alpha(void)\n{\n    return;\n}\n#define MACRO_A 1\n",
        ),
        ProjectFile(
            path=Path("src/b.c"),
            content="void beta(void)\n{\n    return;\n}\n#define MACRO_B 1\n",
        ),
        ProjectFile(
            path=Path("src/c.c"),
            content="void gamma_only(void)\n{\n    return;\n}\n#define MACRO_C 1\n",
        ),
    ]


def test_focused_map_marks_included_file_and_skips_macros():
    files = _focused_files()
    index, graph = _indexed(files)
    ranked = (Path("src/a.c"), Path("src/b.c"))
    included = (Path("src/a.c"),)

    project_map = build_focused_map(index, graph, ranked, included, 10**9)

    assert "Project map (focused): 2 of 3 project files" in project_map.text
    assert project_map.tier == "focused"
    included_line = next(
        line for line in project_map.text.splitlines() if line.startswith("src/a.c")
    )
    other_line = next(
        line for line in project_map.text.splitlines() if line.startswith("src/b.c")
    )
    assert "(included below)" in included_line
    assert "(included below)" not in other_line
    assert "macros:" not in project_map.text
    assert "src/c.c" not in project_map.text


def test_focused_map_cap_keeps_the_included_file():
    files = _focused_files()
    index, graph = _indexed(files)
    included = (Path("src/b.c"),)
    one = build_focused_map(index, graph, included, included, 10**9)
    ranked = (Path("src/a.c"), Path("src/b.c"), Path("src/c.c"))

    project_map = build_focused_map(index, graph, ranked, included, len(one.text))

    assert len(project_map.text) <= len(one.text)
    assert "src/b.c" in project_map.text
    assert "(included below)" in project_map.text
    assert "src/a.c" not in project_map.text


def test_partial_prompt_order_and_rank():
    files = _focused_files()
    question = "What does beta do?"
    map_text = "Project map (focused): 2 of 3 project files. marker-for-order"
    included = (Path("src/b.c"), Path("src/a.c"))

    prompt = build_partial_prompt(files, included, question, map_text, 1000)

    assert (
        prompt.find("You are an expert")
        < prompt.find(map_text)
        < prompt.find("Partial context:")
        < prompt.find("File: src/b.c")
        < prompt.find("Question:")
    )
    assert prompt.find("File: src/b.c") < prompt.find("File: src/a.c")
    assert "gamma_only" not in prompt
    assert "MACRO_C" not in prompt


def test_assistant_prompt_mentions_partial():
    assert "partial" in assistant_prompt.lower()


def test_build_project_map_file_count_full_and_omitted(mapped_files):
    index, graph = _indexed(mapped_files)

    full = build_project_map(index, graph, 10**9)
    omitted = build_project_map(index, graph, 0)

    assert full.tier == "full"
    assert full.file_count == len(index.files)
    assert omitted.tier == "omitted"
    assert omitted.file_count == 0


def test_focused_map_file_count_matches_header():
    files = _focused_files()
    index, graph = _indexed(files)
    ranked = (Path("src/a.c"), Path("src/b.c"))
    included = (Path("src/a.c"),)

    project_map = build_focused_map(index, graph, ranked, included, 10**9)
    header = project_map.text.splitlines()[0]
    shown = int(header.split(":", 1)[1].split(" of ", 1)[0].strip())

    assert f"{shown} of {len(files)} project files" in header
    assert project_map.file_count == shown


def test_p1_text_files_show_as_not_parsed_without_fields():
    files = [
        ProjectFile(path=Path("main.c"), content="void main(void) { return; }\n"),
        ProjectFile(path=Path("Kconfig"), content="config FOO\n\tbool\n"),
        ProjectFile(path=Path("CMakeLists.txt"), content="project(app)\n"),
        ProjectFile(path=Path("app.overlay"), content="led0: led_0 {\n};\n"),
        ProjectFile(path=Path("doc.rst"), content="Title\n=====\n\nbody\n"),
    ]
    index = build_index(files)
    text = build_project_map(index, build_graph(index), 10**9).text
    lines = text.splitlines()

    for name in ("Kconfig", "CMakeLists.txt", "app.overlay", "doc.rst"):
        marker = f"{name} (not parsed)"
        assert marker in lines
        follow = lines.index(marker) + 1
        if follow < len(lines):
            assert not lines[follow].startswith("  ")
    assert "main.c (not parsed)" not in text
    assert any(line.startswith("  ") for line in lines)

