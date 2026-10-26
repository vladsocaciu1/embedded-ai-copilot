import logging
import posixpath
import random
import re
import textwrap
from pathlib import Path

import pytest

from app.code_graph import (
    CodeGraph,
    GraphUpdateTooLarge,
    _include_key,
    build_graph,
    update_graph,
)
from app.code_index import build_index
from app.index_cache import IndexCache
from app.models import (
    Call,
    CallRef,
    FileSymbols,
    Include,
    ProjectFile,
    ProjectIndex,
    SymbolKind,
)
from app.project_loader import load_project_files
from tests.test_index_cache import _install_spies
from tests.test_selection_eval import eval_files


def _graph(*items: tuple[str, str]) -> CodeGraph:
    project_files = [
        ProjectFile(path=Path(path), content=textwrap.dedent(content))
        for path, content in items
    ]
    return build_graph(build_index(project_files))


def _sample_graph() -> CodeGraph:
    repo = Path(__file__).resolve().parent.parent
    return build_graph(build_index(load_project_files(repo / "sample_project")))


def _candidates(graph: CodeGraph, path: str, target: str) -> tuple[Path, ...]:
    found = [
        resolved.candidates
        for resolved in graph.includes(Path(path))
        if resolved.include.target == target
    ]
    assert len(found) == 1
    return found[0]


def test_r1_quoted_include_prefers_the_relative_file():
    graph = _graph(
        ("src/a.c", '#include "b.h"\n'),
        ("src/b.h", "\n"),
        ("inc/b.h", "\n"),
    )

    assert _candidates(graph, "src/a.c", "b.h") == (Path("src/b.h"),)


def test_r2_multi_part_and_parent_includes():
    graph = _graph(
        (
            "src/y.c",
            """\
            #include "hal/can.h"
            #include "../inc/x.h"
            #include "..\\inc\\x.h"
            """,
        ),
        ("drivers/hal/can.h", "\n"),
        ("inc/x.h", "\n"),
    )

    assert _candidates(graph, "src/y.c", "hal/can.h") == (Path("drivers/hal/can.h"),)
    assert _candidates(graph, "src/y.c", "../inc/x.h") == (Path("inc/x.h"),)
    assert _candidates(graph, "src/y.c", "..\\inc\\x.h") == (Path("inc/x.h"),)


def test_r3_nearest_directory_picks_the_same_project():
    graph = _graph(
        ("projA/Src/main.c", '#include "main.h"\n'),
        ("projA/Inc/main.h", "\n"),
        ("projB/Inc/main.h", "\n"),
    )

    assert _candidates(graph, "projA/Src/main.c", "main.h") == (
        Path("projA/Inc/main.h"),
    )


def test_r4_equal_distance_includes_stay_ambiguous():
    graph = _graph(
        ("lib/x.c", '#include "conf.h"\n'),
        ("p1/conf.h", "\n"),
        ("p2/conf.h", "\n"),
    )

    assert _candidates(graph, "lib/x.c", "conf.h") == (
        Path("p1/conf.h"),
        Path("p2/conf.h"),
    )


def test_r5_system_includes_resolve_only_inside_the_project():
    graph = _graph(
        (
            "src/a.c",
            """\
            #include <stdint.h>
            #include <board.h>
            """,
        ),
        ("bsp/board.h", "\n"),
    )

    assert _candidates(graph, "src/a.c", "stdint.h") == ()
    assert _candidates(graph, "src/a.c", "board.h") == (Path("bsp/board.h"),)


def test_r6_include_match_is_case_insensitive():
    graph = _graph(
        ("src/main.c", '#include "CAN.h"\n'),
        ("inc/can.h", "\n"),
    )

    assert _candidates(graph, "src/main.c", "CAN.h") == (Path("inc/can.h"),)


def test_r7_included_by_lists_the_including_files():
    graph = _graph(
        ("lib/x.c", '#include "conf.h"\n'),
        ("p1/conf.h", "\n"),
        ("p2/conf.h", "\n"),
    )
    assert Path("lib/x.c") in graph.included_by(Path("p1/conf.h"))

    sample = _sample_graph()
    assert sample.included_by(Path("inc/can.h")) == (
        Path("src/app.c"),
        Path("src/can.c"),
    )


def test_d1_statics_in_other_sources_are_hidden():
    pair = _graph(
        ("a.c", "static int helper(void) { return 1; }\n"),
        ("b.c", "static int helper(void) { return 2; }\n"),
    )
    assert tuple(ref.path for ref in pair.definitions("helper")) == (
        Path("a.c"),
        Path("b.c"),
    )
    assert tuple(
        ref.path for ref in pair.definitions("helper", from_path=Path("a.c"))
    ) == (Path("a.c"),)

    with_header = _graph(
        ("a.c", "static int helper(void) { return 1; }\n"),
        ("b.c", "static int helper(void) { return 2; }\n"),
        ("util.h", "static inline int helper(void) { return 3; }\n"),
    )
    assert tuple(
        ref.path
        for ref in with_header.definitions("helper", from_path=Path("a.c"))
    ) == (Path("a.c"), Path("util.h"))


def test_d2_definitions_skip_prototypes_and_use_qualified_names():
    graph = _graph(
        (
            "src/controller.cpp",
            """\
            namespace drivers {
            class Controller {
            public:
                void send();
            };
            void Controller::send() {}
            }

            namespace other {
            class Other {
            public:
                void send();
            };
            }

            void only_proto(void);
            """,
        ),
    )

    found = graph.definitions("Controller::send")
    assert tuple(ref.symbol.name for ref in found) == (
        "drivers::Controller::send",
    )
    assert tuple(ref.symbol.name for ref in graph.definitions("send")) == (
        "drivers::Controller::send",
    )
    assert graph.definitions("Other::send") == ()
    assert graph.definitions("only_proto") == ()
    assert all(ref.symbol.kind != SymbolKind.PROTOTYPE for ref in found)


def test_k1_sample_project_callers_and_callees():
    graph = _sample_graph()

    assert graph.callers("Can_SendFrame") == (
        CallRef(
            path=Path("src/app.c"),
            call=Call(
                caller="App_Run",
                callee="Can_SendFrame",
                line=20,
                is_member=False,
            ),
        ),
    )
    assert {ref.call.callee for ref in graph.callees("App_Init")} == {
        "Gpio_Init",
        "Can_Init",
    }


def test_p1_pairs_follow_stem_and_nearest_directory():
    sample = _sample_graph()
    assert sample.pair(Path("src/can.c")) == (Path("inc/can.h"),)
    assert sample.pair(Path("inc/can.h")) == (Path("src/can.c"),)
    assert sample.pair(Path("src/main.c")) == ()

    graph = _graph(
        ("projA/Src/uart.c", "void uart(void) {}\n"),
        ("projA/Inc/uart.h", "void uart(void);\n"),
        ("projB/Inc/uart.h", "void uart(void);\n"),
        ("start.s", "nop\n"),
        ("start.h", "\n"),
    )
    assert graph.pair(Path("projA/Src/uart.c")) == (Path("projA/Inc/uart.h"),)
    assert graph.pair(Path("start.s")) == ()


def test_u1_unknown_paths_and_names_are_empty():
    graph = _graph(
        ("src/a.c", '#include "a.h"\nvoid f(void) { g(); }\n'),
        ("inc/a.h", "void f(void);\n"),
    )
    missing = Path("missing.c")

    assert graph.definitions("no_such_symbol") == ()
    assert graph.definitions("f", from_path=missing) == ()
    assert graph.callers("no_such_symbol") == ()
    assert graph.callees("no_such_symbol") == ()
    assert graph.includes(missing) == ()
    assert graph.included_by(missing) == ()
    assert graph.pair(missing) == ()


def test_l1_one_info_line_with_the_call_count(caplog):
    caplog.set_level(logging.INFO)
    source = "void f(void) { g(); }\n/* MARKER_GRAPH_7f3a */\n"
    build_graph(build_index([ProjectFile(Path("src/a.c"), source)]))

    records = [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and record.name == "app.code_graph"
    ]
    assert len(records) == 1
    assert "1 calls" in records[0].getMessage()
    assert "MARKER_GRAPH_7f3a" not in caplog.text


_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
_HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx"}
_CPP_SUFFIXES = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}


def _sample_files() -> list[ProjectFile]:
    repo = Path(__file__).resolve().parent.parent
    return load_project_files(repo / "sample_project")


def _graph_fixture_files() -> list[ProjectFile]:
    groups = (
        (
            "r1",
            (
                (
                    "src/a.c",
                    """\
                    #include "b.h"

                    void a_fn(void)
                    {
                        a_fn();
                    }
                    """,
                ),
                ("src/b.h", "\n"),
                ("inc/b.h", "\n"),
                ("notes.txt", "not parsed\n"),
            ),
        ),
        (
            "r2",
            (
                (
                    "src/y.c",
                    """\
                    #include "hal/can.h"
                    #include "../inc/x.h"
                    #include "..\\inc\\x.h"
                    """,
                ),
                ("drivers/hal/can.h", "\n"),
                ("inc/x.h", "\n"),
            ),
        ),
        (
            "r3",
            (
                ("projA/Src/main.c", '#include "main.h"\n'),
                ("projA/Inc/main.h", "\n"),
                ("projB/Inc/main.h", "\n"),
            ),
        ),
        (
            "r4",
            (
                ("lib/x.c", '#include "conf.h"\n'),
                ("p1/conf.h", "\n"),
                ("p2/conf.h", "\n"),
            ),
        ),
        (
            "r5",
            (
                (
                    "src/a.c",
                    """\
                    #include <stdint.h>
                    #include <board.h>
                    """,
                ),
                ("bsp/board.h", "\n"),
            ),
        ),
        (
            "r6",
            (
                ("src/main.c", '#include "CAN.h"\n'),
                ("inc/can.h", "\n"),
            ),
        ),
        (
            "r7",
            (
                ("lib/x.c", '#include "conf.h"\n'),
                ("p1/conf.h", "\n"),
                ("p2/conf.h", "\n"),
            ),
        ),
        (
            "d1",
            (
                ("a.c", "static int helper(void) { return 1; }\n"),
                ("b.c", "static int helper(void) { return 2; }\n"),
                ("util.h", "static inline int helper(void) { return 3; }\n"),
            ),
        ),
        (
            "d2",
            (
                (
                    "src/controller.cpp",
                    """\
                    namespace drivers {
                    class Controller {
                    public:
                        void send();
                    };
                    void Controller::send() {}
                    }

                    namespace other {
                    class Other {
                    public:
                        void send();
                    };
                    }

                    void only_proto(void);
                    """,
                ),
            ),
        ),
        (
            "p1",
            (
                ("projA/Src/uart.c", "void uart(void) {}\n"),
                ("projA/Inc/uart.h", "void uart(void);\n"),
                ("projB/Inc/uart.h", "void uart(void);\n"),
                ("start.s", "nop\n"),
                ("start.h", "\n"),
            ),
        ),
    )
    files: list[ProjectFile] = []
    for top, items in groups:
        for path, content in items:
            files.append(ProjectFile(Path(top) / path, textwrap.dedent(content)))
    return files


def _fixture_files(name: str) -> list[ProjectFile]:
    if name == "sample_project":
        return _sample_files()
    if name == "eval":
        return eval_files()
    if name == "graph":
        return _graph_fixture_files()
    raise AssertionError(name)


def _delete_lines(content: str, start: int, end: int) -> str:
    lines = content.splitlines(keepends=True)
    del lines[start - 1 : end]
    return "".join(lines)


def _replace_line(content: str, line_no: int, replacer) -> str:
    lines = content.splitlines(keepends=True)
    lines[line_no - 1] = replacer(lines[line_no - 1])
    return "".join(lines)


def _first_function(symbols: FileSymbols):
    return next(symbol for symbol in symbols.symbols if symbol.kind == SymbolKind.FUNCTION)


def _first_source(
    files: list[ProjectFile], index: ProjectIndex
) -> tuple[ProjectFile, FileSymbols]:
    by_path = {project_file.path: project_file for project_file in files}
    for file_symbols in index.files:
        if file_symbols.path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        has_function = any(
            symbol.kind == SymbolKind.FUNCTION for symbol in file_symbols.symbols
        )
        if has_function and file_symbols.includes:
            return by_path[file_symbols.path], file_symbols
    raise AssertionError("no source file with a function and an include")


def _most_included_header(files: list[ProjectFile], graph: CodeGraph) -> ProjectFile:
    headers = [
        project_file
        for project_file in files
        if project_file.path.suffix.lower() in _HEADER_SUFFIXES
    ]
    headers.sort(
        key=lambda project_file: (
            -len(graph.included_by(project_file.path)),
            project_file.path.parts,
        )
    )
    if not headers or not graph.included_by(headers[0].path):
        raise AssertionError("no included header")
    return headers[0]


def _included_partner(
    files: list[ProjectFile], graph: CodeGraph, source: ProjectFile
) -> ProjectFile:
    by_path = {project_file.path: project_file for project_file in files}
    for resolved in graph.includes(source.path):
        if resolved.candidates:
            return by_path[resolved.candidates[0]]
    raise AssertionError("source does not include another project file")


def _relative_include(source: Path, target: Path) -> str:
    parent = source.parent.as_posix()
    if parent == "":
        parent = "."
    return posixpath.relpath(target.as_posix(), start=parent)


def _apply_edit(
    files: list[ProjectFile], index: ProjectIndex, graph: CodeGraph, edit: str
) -> tuple[list[ProjectFile], list[Path]]:
    if edit == "E10":
        header = _most_included_header(files, graph)
        content = header.content + "\nvoid zz_header_edit(void) { }\n"
        return _replaced(files, {header.path: content}), [header.path]

    source, symbols = _first_source(files, index)
    if edit == "E12":
        partner = _included_partner(files, graph, source)
        relative = _relative_include(source.path, partner.path)
        source_content = (
            f'#include "{relative}"\n'
            + source.content
            + "\nvoid zz_e12_user(void) { zz_e12(); }\n"
        )
        partner_content = partner.content + "\nvoid zz_e12(void) { }\n"
        updates = {source.path: source_content, partner.path: partner_content}
        return _replaced(files, updates), [source.path, partner.path]

    content = _edit_source(edit, source, symbols, files)
    if content == source.content:
        raise AssertionError(f"{edit} did not change {source.path}")
    return _replaced(files, {source.path: content}), [source.path]


def _edit_source(
    edit: str,
    source: ProjectFile,
    symbols: FileSymbols,
    files: list[ProjectFile],
) -> str:
    if edit == "E1":
        function = _first_function(symbols)
        name = function.name.rsplit("::", 1)[-1]
        return source.content + f"\nvoid zz_added(void) {{ {name}(); }}\n"
    if edit == "E2":
        function = _first_function(symbols)
        return _delete_lines(source.content, function.start_line, function.end_line)
    if edit == "E3":
        if not symbols.calls:
            raise AssertionError(f"{source.path} has no call")
        call = symbols.calls[0]
        return _replace_line(
            source.content,
            call.line,
            lambda line: re.sub(
                rf"\b{re.escape(call.callee)}\b",
                call.callee + "_renamed",
                line,
                count=1,
            ),
        )
    if edit == "E4":
        headers = sorted(
            (
                project_file.path
                for project_file in files
                if project_file.path != source.path
                and project_file.path.suffix.lower() in _HEADER_SUFFIXES
            ),
            key=lambda path: path.parts,
        )
        if not headers:
            raise AssertionError("no header")
        relative = _relative_include(source.path, headers[0])
        return f'#include "{relative}"\n' + source.content
    if edit == "E5":
        include = symbols.includes[0]
        return _delete_lines(source.content, include.line, include.line)
    if edit == "E6":
        include = symbols.includes[0]

        def replace_target(line: str) -> str:
            if include.is_system:
                return line.replace(f"<{include.target}>", "<zz_missing.h>", 1)
            return line.replace(f'"{include.target}"', '"zz_missing.h"', 1)

        return _replace_line(source.content, include.line, replace_target)
    if edit == "E7":
        function = _first_function(symbols)

        def toggle_static(line: str) -> str:
            if function.is_static:
                return line.replace("static ", "", 1)
            return "static " + line

        return _replace_line(source.content, function.start_line, toggle_static)
    if edit == "E8":
        return source.content + "\nint broken( {\n"
    if edit == "E9":
        return ""
    if edit == "E11":
        return source.content + "\n/* zz_comment_only */\n"
    raise AssertionError(edit)


def _replaced(
    files: list[ProjectFile], updates: dict[Path, str]
) -> list[ProjectFile]:
    return [
        ProjectFile(project_file.path, updates.get(project_file.path, project_file.content))
        for project_file in files
    ]


def test_g1_snapshot_matches_a_rebuild_and_is_detached():
    files = [ProjectFile(Path("src/a.c"), "int a(void) { return 1; }\n")]
    index = build_index(files)
    graph = build_graph(index)
    assert graph.snapshot() == build_graph(index).snapshot()

    snap = graph.snapshot()
    snap["definitions"].clear()
    snap["paths"] = ()
    assert graph.snapshot() == build_graph(index).snapshot()
    assert graph.definitions("a")

    edited = [
        ProjectFile(
            Path("src/a.c"),
            "int a(void) { return 1; }\nint added(void) { return 2; }\n",
        )
    ]
    assert graph.snapshot() != build_graph(build_index(edited)).snapshot()


def test_g2_update_graph_rejects_a_different_path_set():
    files = [ProjectFile(Path("a.c"), "int a(void) { return 1; }\n")]
    index = build_index(files)
    graph = build_graph(index)
    added = build_index(
        files + [ProjectFile(Path("b.c"), "int b(void) { return 2; }\n")]
    )
    with pytest.raises(ValueError):
        update_graph(graph, index, added, [])
    with pytest.raises(ValueError):
        update_graph(graph, index, index, [Path("missing.c")])


def test_g3_empty_changed_matches_the_graph():
    files = [ProjectFile(Path("a.c"), "int a(void) { return 1; }\n")]
    index = build_index(files)
    graph = build_graph(index)
    assert update_graph(graph, index, index, []).snapshot() == graph.snapshot()


def test_g4_update_does_not_mutate_the_input_graph():
    files = [
        ProjectFile(Path("src/a.c"), '#include "a.h"\nint a(void) { b(); return 1; }\n'),
        ProjectFile(Path("inc/a.h"), "void b(void);\n"),
    ]
    index = build_index(files)
    graph = build_graph(index)
    before = graph.snapshot()
    edited = [
        ProjectFile(
            Path("src/a.c"),
            '#include "a.h"\nint a(void) { b(); return 2; }\nint c(void) { return 3; }\n',
        ),
        files[1],
    ]
    updated = update_graph(graph, index, build_index(edited), [Path("src/a.c")])
    assert updated is not graph
    assert graph.snapshot() == before
    assert updated._definitions is not graph._definitions
    assert updated._calls_by_callee is not graph._calls_by_callee
    assert updated._calls_by_caller is not graph._calls_by_caller
    assert updated._includes is not graph._includes
    assert updated._included_by is not graph._included_by
    assert updated._pairs is graph._pairs
    assert updated._paths is graph._paths
    assert updated._by_parts is graph._by_parts
    assert updated._by_basename is graph._by_basename
    assert updated._lower_parts is graph._lower_parts
    assert updated._parent_parts is graph._parent_parts
    assert updated._by_stem_group is graph._by_stem_group


def test_g5_log_line_has_no_source_text(caplog):
    source = "int a(void) { return 1; }\n/* SOURCE_MARKER_zz20 */\n"
    index = build_index([ProjectFile(Path("a.c"), source)])
    graph = build_graph(index)
    edited = build_index(
        [ProjectFile(Path("a.c"), source + "int b(void) { return 2; }\n")]
    )
    caplog.clear()
    caplog.set_level(logging.INFO)
    update_graph(graph, index, edited, [Path("a.c")])

    records = [record for record in caplog.records if record.levelno == logging.INFO]
    assert len(records) == 1
    message = records[0].getMessage()
    assert message.startswith("Graph updated: 1 files")
    assert "SOURCE_MARKER_zz20" not in message


@pytest.mark.parametrize("fixture_name", ["sample_project", "eval", "graph"])
@pytest.mark.parametrize("edit", [f"E{number}" for number in range(1, 13)])
def test_content_edit_matches_full_rebuild(fixture_name, edit):
    files = _fixture_files(fixture_name)
    old = build_index(files)
    graph = build_graph(old)
    edited, changed = _apply_edit(files, old, graph, edit)
    new = build_index(edited)
    assert (
        update_graph(graph, old, new, changed).snapshot()
        == build_graph(new).snapshot()
    )


_X1_PATHS = (
    "src/main.c",
    "src/can.c",
    "inc/can.h",
    "drivers/can/can.h",
    "drivers/can/can.c",
    "app/app.cpp",
    "app/app.hpp",
    "app/sub/ui.cc",
    "inc/util.h",
    "src/util.c",
    "lib/x/util.h",
    "README.txt",
)
_X1_INCLUDES = (
    '"can.h"',
    '"../inc/can.h"',
    '"can/can.h"',
    "<util.h>",
    '"util.h"',
    '"app.hpp"',
    '"missing.h"',
    '"..\\inc\\util.h"',
    '"./ui.cc"',
)
_X1_NAMES = (
    "init",
    "can_send",
    "can_recv",
    "util_max",
    "main",
    "tick",
    "run",
    "helper",
    "isr",
    "HAL_Delay",
)


def _x1_file(rng: random.Random, path: Path) -> str:
    if path.name == "README.txt":
        return "plain text\n"
    blank = rng.random() < 0.05
    broken = rng.random() < 0.10
    if blank:
        return "" if rng.random() < 0.5 else "plain text\n"
    lines: list[str] = []
    for _ in range(rng.randint(0, 4)):
        lines.append(f"#include {rng.choice(_X1_INCLUDES)}")
    for _ in range(rng.randint(0, 5)):
        kind = rng.randrange(5)
        if kind == 0:
            lines.append(f"#define ZZ_{rng.randrange(100)} 1")
        elif kind == 1:
            static = "static " if rng.random() < 0.5 else ""
            lines.append(f"{static}int v_{rng.randrange(100)} = 1;")
        elif kind == 2:
            lines.append(f"int {rng.choice(_X1_NAMES)}(void);")
        elif kind == 3:
            lines.append(f"typedef int t_{rng.randrange(100)};")
        else:
            static = "static " if rng.random() < 0.5 else ""
            name = rng.choice(_X1_NAMES)
            calls = " ".join(
                f"{rng.choice(_X1_NAMES)}();" for _ in range(rng.randint(0, 3))
            )
            body = f"{static}int {name}(void) {{ {calls} return 0; }}"
            if path.suffix.lower() in _CPP_SUFFIXES and rng.random() < 0.5:
                body = f"namespace ns {{ {body} }}"
            lines.append(body)
    if broken:
        lines.append("int broken( {")
    return "\n".join(lines) + "\n"


def test_x1_seeded_random_edits_match_a_rebuild(monkeypatch):
    rng = random.Random(20261019)
    paths = [Path(path) for path in _X1_PATHS]
    contents = {path: _x1_file(rng, path) for path in paths}

    def project_files() -> list[ProjectFile]:
        return [ProjectFile(path, contents[path]) for path in paths]

    cache = IndexCache()
    _index, graph = cache.get(Path("/x1"), project_files())
    spies = _install_spies(monkeypatch)
    for step in range(200):
        before = graph.snapshot()
        previous = graph
        for path in rng.sample(paths, rng.randint(1, 3)):
            generated = _x1_file(rng, path)
            if generated == contents[path]:
                generated += f"\n/* {step} */\n"
            contents[path] = generated
        spies.graph_calls.clear()
        spies.update_calls.clear()
        _index, graph = cache.get(Path("/x1"), project_files())
        assert spies.graph_calls == [], step
        assert len(spies.update_calls) == 1, step
        assert previous.snapshot() == before, step
        rebuilt = build_graph(build_index(project_files()))
        assert graph.snapshot() == rebuilt.snapshot(), step


def test_g6_update_graph_rejects_inconsistent_paths():
    files = [
        ProjectFile(Path("a.c"), "int a(void) { return 1; }\n"),
        ProjectFile(Path("b.c"), "int b(void) { return 2; }\n"),
    ]
    index = build_index(files)
    graph = build_graph(index)
    only_a = build_index(files[:1])

    with pytest.raises(ValueError):
        update_graph(graph, only_a, index, [Path("b.c")])
    with pytest.raises(ValueError):
        update_graph(graph, index, only_a, [])
    with pytest.raises(ValueError):
        update_graph(graph, index, index, [Path("missing.c")])


def test_g7_path_change_copies_tables_and_keeps_the_input():
    files = [
        ProjectFile(Path("src/a.c"), '#include "a.h"\nint a(void) { return 1; }\n'),
        ProjectFile(Path("inc/a.h"), "void b(void);\n"),
    ]
    index = build_index(files)
    graph = build_graph(index)
    before = graph.snapshot()
    new_files = [
        files[0],
        ProjectFile(Path("inc/b.h"), "void c(void);\n"),
    ]
    new_index = build_index(new_files)
    updated = update_graph(
        graph, index, new_index, [Path("inc/a.h"), Path("inc/b.h")]
    )

    assert graph.snapshot() == before
    assert updated.snapshot() == build_graph(new_index).snapshot()
    assert updated._by_parts is not graph._by_parts
    assert updated._by_basename is not graph._by_basename
    assert updated._lower_parts is not graph._lower_parts
    assert updated._parent_parts is not graph._parent_parts
    assert updated._by_stem_group is not graph._by_stem_group
    assert updated._pairs is not graph._pairs
    assert updated._paths is not graph._paths
    assert updated._includers_by_key is not graph._includers_by_key


def test_g8_rename_logs_added_and_removed(caplog):
    index = build_index([ProjectFile(Path("a.c"), "int a(void) { return 1; }\n")])
    graph = build_graph(index)
    renamed = build_index([ProjectFile(Path("b.c"), "int a(void) { return 1; }\n")])
    caplog.clear()
    caplog.set_level(logging.INFO)
    update_graph(graph, index, renamed, [Path("a.c"), Path("b.c")])

    records = [record for record in caplog.records if record.levelno == logging.INFO]
    assert len(records) == 1
    assert records[0].getMessage().startswith(
        "Graph updated: 2 files (1 added, 1 removed)"
    )


def test_g9_max_includes_rejects_before_patching():
    files = [ProjectFile(Path("a.c"), "int a(void) { return 1; }\n")]
    index = build_index(files)
    graph = build_graph(index)
    before = graph.snapshot()
    new_index = build_index(files + [ProjectFile(Path("b.h"), '#include "a.h"\n')])

    with pytest.raises(GraphUpdateTooLarge) as caught:
        update_graph(graph, index, new_index, [Path("b.h")], max_includes=0)
    assert caught.value.includes == 1
    assert graph.snapshot() == before

    updated = update_graph(graph, index, new_index, [Path("b.h")], max_includes=None)
    assert updated.snapshot() == build_graph(new_index).snapshot()


def test_g10_include_key_lists_each_file_once_per_key():
    assert _include_key(Include("a/B.h", False, 1)) == "b.h"
    assert _include_key(Include("x.h", True, 1)) == "x.h"
    assert _include_key(Include("..\\inc\\U.H", False, 1)) == "u.h"
    assert _include_key(Include("x/..", False, 1)) == ""
    assert _include_key(Include("./", False, 1)) == ""

    graph = _graph(("a.c", '#include "b.h"\n#include "B.H"\n#include "c.h"\n'))
    assert graph._includers_by_key["b.h"] == frozenset({Path("a.c")})
    assert graph._includers_by_key["c.h"] == frozenset({Path("a.c")})
    assert set(graph._includers_by_key) == {"b.h", "c.h"}


def _assert_path_update(
    old_files: list[ProjectFile],
    new_files: list[ProjectFile],
    changed: list[Path],
) -> CodeGraph:
    old = build_index(old_files)
    graph = build_graph(old)
    before = graph.snapshot()
    new = build_index(new_files)
    updated = update_graph(graph, old, new, changed)
    assert updated.snapshot() == build_graph(new).snapshot()
    assert graph.snapshot() == before
    return updated


def _assert_add_and_remove(base: list[ProjectFile], extra: ProjectFile) -> None:
    _assert_path_update(base, base + [extra], [extra.path])
    _assert_path_update(base + [extra], base, [extra.path])


def test_w1_wildcard_include_resolves_to_the_normalized_file():
    base = [ProjectFile(Path("d/a.c"), '#include "x/.."\n')]
    extra = ProjectFile(Path("D"), "int d;\n")
    full = build_graph(build_index(base + [extra]))
    assert _candidates(full, "d/a.c", "x/..") == (Path("D"),)
    _assert_add_and_remove(base, extra)


def test_w2_relative_hit_beats_a_nearer_suffix_match():
    base = [
        ProjectFile(Path("p/q/src/x.c"), '#include "../../h.h"\n'),
        ProjectFile(Path("p/q/src/h.h"), "\n"),
    ]
    extra = ProjectFile(Path("p/h.h"), "\n")
    full = build_graph(build_index(base + [extra]))
    assert _candidates(full, "p/q/src/x.c", "../../h.h") == (Path("p/h.h"),)
    _assert_add_and_remove(base, extra)


def test_w3_case_variant_joins_a_relative_hit():
    base = [
        ProjectFile(Path("src/a.c"), '#include "b.h"\n'),
        ProjectFile(Path("src/b.h"), "\n"),
    ]
    extra = ProjectFile(Path("src/B.h"), "\n")
    full = build_graph(build_index(base + [extra]))
    assert _candidates(full, "src/a.c", "b.h") == (Path("src/B.h"), Path("src/b.h"))
    _assert_add_and_remove(base, extra)


def test_w4_equal_score_header_is_ambiguous_and_a_farther_one_is_not(caplog):
    base = [
        ProjectFile(Path("lib/x.c"), '#include "conf.h"\n'),
        ProjectFile(Path("p1/conf.h"), "\n"),
    ]
    extra = ProjectFile(Path("p2/conf.h"), "\n")
    full = build_graph(build_index(base + [extra]))
    assert _candidates(full, "lib/x.c", "conf.h") == (
        Path("p1/conf.h"),
        Path("p2/conf.h"),
    )
    _assert_add_and_remove(base, extra)

    near_base = [
        ProjectFile(Path("proj/src/a.c"), '#include "h.h"\n'),
        ProjectFile(Path("proj/inc/h.h"), "\n"),
    ]
    farther = ProjectFile(Path("other/h.h"), "\n")
    near = build_graph(build_index(near_base))
    assert _candidates(near, "proj/src/a.c", "h.h") == (Path("proj/inc/h.h"),)
    old = build_index(near_base)
    graph = build_graph(old)
    caplog.clear()
    caplog.set_level(logging.INFO)
    update_graph(graph, old, build_index(near_base + [farther]), [farther.path])
    assert any(
        "0 includes re-resolved" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    )
    _assert_add_and_remove(near_base, farther)
    assert _candidates(
        build_graph(build_index(near_base + [farther])), "proj/src/a.c", "h.h"
    ) == (Path("proj/inc/h.h"),)


def test_w5_removing_the_nearest_candidate_falls_back():
    source = ProjectFile(Path("proj/src/a.c"), '#include "h.h"\n')
    near = ProjectFile(Path("proj/inc/h.h"), "\n")
    far = ProjectFile(Path("other/h.h"), "\n")
    full = build_graph(build_index([source, near, far]))
    assert _candidates(full, "proj/src/a.c", "h.h") == (Path("proj/inc/h.h"),)
    fallen = build_graph(build_index([source, far]))
    assert _candidates(fallen, "proj/src/a.c", "h.h") == (Path("other/h.h"),)
    _assert_add_and_remove([source, far], near)

    kept = build_graph(build_index([source, near]))
    assert _candidates(kept, "proj/src/a.c", "h.h") == (Path("proj/inc/h.h"),)
    _assert_add_and_remove([source, near], far)


def test_w6_pairs_follow_nearer_equal_and_fallback_headers():
    source = ProjectFile(Path("proj/Src/uart.c"), "void uart(void) {}\n")
    near = ProjectFile(Path("proj/Inc/uart.h"), "void uart(void);\n")
    far = ProjectFile(Path("other/uart.h"), "void uart(void);\n")
    equal = ProjectFile(Path("zzz/uart.h"), "void uart(void);\n")

    with_far = build_graph(build_index([source, far]))
    assert with_far.pair(source.path) == (far.path,)
    with_near = build_graph(build_index([source, far, near]))
    assert with_near.pair(source.path) == (near.path,)
    _assert_add_and_remove([source, far], near)

    with_equal = build_graph(build_index([source, far, equal]))
    assert with_equal.pair(source.path) == (far.path, equal.path)
    _assert_add_and_remove([source, far], equal)

    fallen = build_graph(build_index([source, far]))
    assert fallen.pair(source.path) == (far.path,)


def _header_and_includer(files: list[ProjectFile]) -> tuple[ProjectFile, Path]:
    graph = build_graph(build_index(files))
    header = _most_included_header(files, graph)
    includer = sorted(graph.included_by(header.path), key=lambda path: path.parts)[0]
    return header, includer


def _first_source_file(files: list[ProjectFile]) -> ProjectFile:
    sources = [
        project_file
        for project_file in files
        if project_file.path.suffix.lower() in _SOURCE_SUFFIXES
    ]
    sources.sort(key=lambda project_file: project_file.path.parts)
    return sources[0]


def _replaced_path(
    files: list[ProjectFile], old: Path, new: Path
) -> list[ProjectFile]:
    return [
        ProjectFile(new if project_file.path == old else project_file.path, project_file.content)
        for project_file in files
    ]


@pytest.mark.parametrize("fixture_name", ["sample_project", "eval", "graph"])
def test_p1_add_unique_file_matches_rebuild(fixture_name):
    files = _fixture_files(fixture_name)
    added = ProjectFile(
        Path("zz_new/zz_unique.c"), "int zz_unique(void) { return 1; }\n"
    )
    _assert_path_update(files, files + [added], [added.path])


@pytest.mark.parametrize("fixture_name", ["sample_project", "eval", "graph"])
def test_p2_remove_most_included_header_matches_rebuild(fixture_name):
    files = _fixture_files(fixture_name)
    header, _includer = _header_and_includer(files)
    kept = [project_file for project_file in files if project_file.path != header.path]
    _assert_path_update(files, kept, [header.path])


@pytest.mark.parametrize("fixture_name", ["sample_project", "eval", "graph"])
def test_p3_add_header_name_beside_the_includer_matches_rebuild(fixture_name):
    files = _fixture_files(fixture_name)
    header, includer = _header_and_includer(files)
    added = ProjectFile(
        includer.parent / "zz_other" / header.path.name,
        "int zz_other(void);\n",
    )
    _assert_path_update(files, files + [added], [added.path])


@pytest.mark.parametrize("fixture_name", ["sample_project", "eval", "graph"])
def test_p4_add_header_in_the_includer_directory_matches_rebuild(fixture_name):
    files = _fixture_files(fixture_name)
    header, includer = _header_and_includer(files)
    path = includer.parent / header.path.name
    if any(project_file.path == path for project_file in files):
        pytest.skip("header name already exists in the includer directory")
    added = ProjectFile(path, "int zz_relative(void);\n")
    _assert_path_update(files, files + [added], [added.path])


@pytest.mark.parametrize("fixture_name", ["sample_project", "eval", "graph"])
def test_p5_rename_header_in_place_matches_rebuild(fixture_name):
    files = _fixture_files(fixture_name)
    header, _includer = _header_and_includer(files)
    renamed = header.path.with_name("zz_" + header.path.name)
    _assert_path_update(files, _replaced_path(files, header.path, renamed), [header.path, renamed])


@pytest.mark.parametrize("fixture_name", ["sample_project", "eval", "graph"])
def test_p6_move_header_matches_rebuild(fixture_name):
    files = _fixture_files(fixture_name)
    header, _includer = _header_and_includer(files)
    moved = Path("zz_moved") / header.path.name
    _assert_path_update(files, _replaced_path(files, header.path, moved), [header.path, moved])


@pytest.mark.parametrize("fixture_name", ["sample_project", "eval", "graph"])
def test_p7_add_file_for_an_unresolved_include_matches_rebuild(fixture_name):
    files = _fixture_files(fixture_name)
    graph = build_graph(build_index(files))
    unresolved = []
    for project_file in files:
        for item in graph.includes(project_file.path):
            if item.candidates:
                continue
            parts = tuple(
                part
                for part in item.include.target.replace("\\", "/").split("/")
                if part and part not in {".", ".."}
            )
            if not parts:
                continue
            unresolved.append((project_file.path, item.include.target, parts[-1]))
    if not unresolved:
        pytest.skip("no unresolved include")
    unresolved.sort()
    name = unresolved[0][2]
    path = Path(name)
    if any(project_file.path == path for project_file in files):
        path = Path("zz_resolved") / name
    added = ProjectFile(path, "int zz_resolved(void);\n")
    _assert_path_update(files, files + [added], [added.path])


@pytest.mark.parametrize("fixture_name", ["sample_project", "eval", "graph"])
def test_p8_add_pair_header_and_remove_source_matches_rebuild(fixture_name):
    files = _fixture_files(fixture_name)
    source = _first_source_file(files)
    added = ProjectFile(Path("zz_pair") / f"{source.path.stem}.h", "int zz_pair(void);\n")
    kept = [
        project_file for project_file in files if project_file.path != source.path
    ]
    _assert_path_update(files, kept + [added], [source.path, added.path])


@pytest.mark.parametrize("fixture_name", ["sample_project", "eval", "graph"])
def test_p9_add_upper_case_header_name_matches_rebuild(fixture_name):
    files = _fixture_files(fixture_name)
    header, _includer = _header_and_includer(files)
    added = ProjectFile(
        Path("ZZ_case") / header.path.name.upper(),
        "int zz_case(void);\n",
    )
    _assert_path_update(files, files + [added], [added.path])


@pytest.mark.parametrize("fixture_name", ["sample_project", "eval", "graph"])
def test_p10_edit_remove_and_add_match_rebuild(fixture_name):
    files = _fixture_files(fixture_name)
    source = _first_source_file(files)
    header, _includer = _header_and_includer(files)
    added = ProjectFile(Path("zz_p10/zz_added.c"), "int zz_added(void) { return 1; }\n")
    new_files = []
    for project_file in files:
        if project_file.path == header.path:
            continue
        if project_file.path == source.path:
            new_files.append(
                ProjectFile(
                    project_file.path,
                    project_file.content + "\nvoid zz_p10(void) { }\n",
                )
            )
        else:
            new_files.append(project_file)
    new_files.append(added)
    _assert_path_update(
        files, new_files, [source.path, header.path, added.path]
    )


_X2_EXTRA = (
    "src/new.h",
    "inc/main.h",
    "lib/x/can.h",
    "app/ui.hpp",
    "drivers/can/util.c",
    "Can.H",
    "src/sub/app.hpp",
    "inc/missing.h",
    "app/sub/ui.h",
    "zz/README.txt",
)


def test_x2_seeded_random_path_changes_match_a_rebuild(monkeypatch):
    rng = random.Random(20261020)
    pool = [Path(path) for path in (*_X1_PATHS, *_X2_EXTRA)]
    present = {}
    for path in pool:
        if rng.random() < 0.6:
            present[path] = _x1_file(rng, path)

    def project_files() -> list[ProjectFile]:
        return [ProjectFile(path, present[path]) for path in pool if path in present]

    cache = IndexCache()
    _index, graph = cache.get(Path("/x2"), project_files())
    spies = _install_spies(monkeypatch)
    for step in range(300):
        before = graph.snapshot()
        previous = graph
        before_present = dict(present)
        for _op in range(rng.randint(1, 3)):
            roll = rng.random()
            if roll < 0.40:
                path = rng.choice(pool)
                generated = _x1_file(rng, path)
                if path in present and generated == present[path]:
                    generated += "/* step */"
                present[path] = generated
            elif roll < 0.70:
                if len(present) <= 1:
                    continue
                del present[rng.choice(list(present))]
            else:
                absent = [path for path in pool if path not in present]
                if not present or not absent:
                    continue
                src = rng.choice(list(present))
                dst = rng.choice(absent)
                present[dst] = present.pop(src)
        spies.graph_calls.clear()
        spies.update_calls.clear()
        _index, graph = cache.get(Path("/x2"), project_files())
        assert spies.graph_calls == [], step
        if present == before_present:
            assert spies.update_calls == [], step
        else:
            assert len(spies.update_calls) == 1, step
        assert previous.snapshot() == before, step
        rebuilt = build_graph(build_index(project_files()))
        assert graph.snapshot() == rebuilt.snapshot(), step


def test_i2_config_is_a_definition_section_and_node_are_not():
    files = [
        ProjectFile(Path("Kconfig"), "config X\n\tbool\n"),
        ProjectFile(Path("prj.conf"), "CONFIG_X=y\n"),
        ProjectFile(Path("doc.rst"), "Title\n=====\n\nbody\n"),
        ProjectFile(Path("app.overlay"), "led0: led_0 {\n};\n"),
    ]
    graph = build_graph(build_index(files))

    assert {ref.path for ref in graph.definitions("CONFIG_X")} == {
        Path("Kconfig"),
        Path("prj.conf"),
    }
    assert graph.definitions("Title") == ()
    assert graph.definitions("led0") == ()
