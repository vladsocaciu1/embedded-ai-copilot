import logging
import textwrap
from pathlib import Path

from app.code_graph import CodeGraph, build_graph
from app.code_index import build_index
from app.models import Call, CallRef, ProjectFile, SymbolKind
from app.project_loader import load_project_files


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
