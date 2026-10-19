import logging
from pathlib import Path

import app.index_cache as index_cache_module
from app.code_graph import CodeGraph, build_graph
from app.code_index import build_index
from app.index_cache import IndexCache
from app.models import ProjectFile

PROJECT = Path("/p")
MARKER = "zz_spec15_marker_9f3c"


def assert_same_graph(left: CodeGraph, right: CodeGraph) -> None:
    assert left._paths == right._paths
    for path in left._paths:
        assert left.includes(path) == right.includes(path)
        assert left.included_by(path) == right.included_by(path)
        assert left.pair(path) == right.pair(path)
    names = set(left._definitions) | set(right._definitions)
    for name in names:
        assert left.definitions(name) == right.definitions(name)
        assert left.callers(name) == right.callers(name)
        assert left.callees(name) == right.callees(name)


def _files(*items: tuple[str, str]) -> list[ProjectFile]:
    return [ProjectFile(path=Path(path), content=content) for path, content in items]


def _symbol_names(index) -> set[str]:
    return {symbol.name for file_symbols in index.files for symbol in file_symbols.symbols}


class _Spies:
    def __init__(self) -> None:
        self.index_calls: list[tuple[ProjectFile, ...]] = []
        self.graph_calls: list[object] = []


def _install_spies(monkeypatch) -> _Spies:
    spies = _Spies()

    def spy_index(project_files):
        spies.index_calls.append(tuple(project_files))
        return build_index(project_files)

    def spy_graph(index):
        spies.graph_calls.append(index)
        return build_graph(index)

    monkeypatch.setattr(index_cache_module, "build_index", spy_index)
    monkeypatch.setattr(index_cache_module, "build_graph", spy_graph)
    return spies


def _equals_uncached(index, graph, files) -> None:
    expected = build_index(files)
    assert index == expected
    assert_same_graph(graph, build_graph(expected))


def test_c1_cold_equals_uncached():
    files = _files(
        ("a.c", "int a(void) { return 1; }\n"),
        ("b.c", "int b(void) { return 2; }\n"),
    )

    index, graph = IndexCache().get(PROJECT, files)

    _equals_uncached(index, graph, files)


def test_c2_warm_hit_returns_same_objects(monkeypatch):
    files = _files(
        ("a.c", "int a(void) { return 1; }\n"),
        ("b.c", "int b(void) { return 2; }\n"),
    )
    cache = IndexCache()
    index, graph = cache.get(PROJECT, files)
    spies = _install_spies(monkeypatch)
    again = [ProjectFile(path=item.path, content=item.content) for item in files]

    warm_index, warm_graph = cache.get(PROJECT, again)

    assert warm_index is index
    assert warm_graph is graph
    assert spies.index_calls == []
    assert spies.graph_calls == []


def test_c3_one_edit_reparses_that_file_and_reuses_symbols(monkeypatch):
    original = _files(
        ("a.c", "int a(void) { return 1; }\n"),
        ("b.c", "int b(void) { return 2; }\n"),
        ("c.c", "int c(void) { return 3; }\n"),
    )
    cache = IndexCache()
    first, _graph = cache.get(PROJECT, original)
    spies = _install_spies(monkeypatch)
    edited = ProjectFile(path=Path("b.c"), content="int b(void) { return 9; }\n")
    files = [original[0], edited, original[2]]

    index, graph = cache.get(PROJECT, files)

    assert spies.index_calls == [(edited,)]
    assert len(spies.graph_calls) == 1
    _equals_uncached(index, graph, files)
    by_path = {file_symbols.path: file_symbols for file_symbols in first.files}
    second = {file_symbols.path: file_symbols for file_symbols in index.files}
    assert second[Path("a.c")] is by_path[Path("a.c")]
    assert second[Path("c.c")] is by_path[Path("c.c")]


def test_c4_same_length_new_content_is_detected():
    cache = IndexCache()
    cache.get(PROJECT, _files(("a.c", "int a = 1;")))

    index, _graph = cache.get(PROJECT, _files(("a.c", "int b = 1;")))

    assert "b" in _symbol_names(index)
    assert "a" not in _symbol_names(index)


def test_c5_add_parses_only_the_new_file(monkeypatch):
    main = ProjectFile(path=Path("main.c"), content='#include "new.h"\n')
    cache = IndexCache()
    _index, graph = cache.get(PROJECT, [main])
    assert graph.includes(Path("main.c"))[0].candidates == ()
    spies = _install_spies(monkeypatch)
    added = ProjectFile(path=Path("inc/new.h"), content="int added(void);\n")

    _index, graph = cache.get(PROJECT, [main, added])

    assert spies.index_calls == [(added,)]
    assert graph.includes(Path("main.c"))[0].candidates == (Path("inc/new.h"),)


def test_c6_delete_rebuilds_graph_without_parsing(monkeypatch, caplog):
    main = ProjectFile(path=Path("main.c"), content='#include "helper.h"\n')
    helper = ProjectFile(path=Path("helper.h"), content="int helper(void);\n")
    cache = IndexCache()
    cache.get(PROJECT, [main, helper])
    spies = _install_spies(monkeypatch)
    caplog.set_level(logging.INFO)

    index, graph = cache.get(PROJECT, [main])

    assert spies.index_calls == []
    assert len(spies.graph_calls) == 1
    assert [file_symbols.path for file_symbols in index.files] == [Path("main.c")]
    assert graph.includes(Path("main.c"))[0].candidates == ()
    assert any("1 dropped" in record.message for record in caplog.records)


def test_c7_rename_parses_again_and_drops_the_old_path(monkeypatch):
    content = "int renamed(void) { return 1; }\n"
    cache = IndexCache()
    cache.get(PROJECT, _files(("old.c", content)))
    spies = _install_spies(monkeypatch)
    renamed = ProjectFile(path=Path("new.c"), content=content)

    index, _graph = cache.get(PROJECT, [renamed])

    assert spies.index_calls == [(renamed,)]
    assert [file_symbols.path for file_symbols in index.files] == [Path("new.c")]


def test_c8_index_order_follows_input_across_reused_and_fresh_files(monkeypatch):
    original = _files(
        ("a.c", "int a;\n"),
        ("b.c", "int b;\n"),
        ("c.c", "int c;\n"),
    )
    cache = IndexCache()
    cache.get(PROJECT, original)
    spies = _install_spies(monkeypatch)
    fresh = ProjectFile(path=Path("a.c"), content="int a2;\n")
    files = [original[2], fresh, original[1]]

    index, _graph = cache.get(PROJECT, files)

    assert [file_symbols.path for file_symbols in index.files] == [
        Path("c.c"),
        Path("a.c"),
        Path("b.c"),
    ]
    assert spies.index_calls == [(fresh,)]


def test_c9_project_change_reparses_every_file(monkeypatch):
    files = _files(
        ("a.c", "int a(void) { return 1; }\n"),
        ("b.c", "int b(void) { return 2; }\n"),
    )
    cache = IndexCache()
    cache.get(PROJECT, files)
    spies = _install_spies(monkeypatch)

    cache.get(Path("/q"), files)

    assert spies.index_calls == [tuple(files)]


def test_c10_log_line_omits_source(caplog):
    files = _files(
        ("a.c", "int a;\n"),
        ("b.c", f"int b; /* {MARKER} */\n"),
        ("c.c", "int c;\n"),
    )
    cache = IndexCache()
    cache.get(PROJECT, files)
    caplog.set_level(logging.INFO)
    edited = ProjectFile(path=Path("b.c"), content=f"int b2; /* {MARKER} */\n")

    cache.get(PROJECT, [files[0], edited, files[2]])
    cache.get(PROJECT, [files[0], edited, files[2]])

    messages = [record.message for record in caplog.records]
    assert any(
        record.levelno == logging.INFO
        and "Index cache: 3 files, 2 reused, 1 parsed, 0 dropped, graph rebuilt" in record.message
        for record in caplog.records
    )
    assert any("graph reused" in message for message in messages)
    assert all(MARKER not in record.message for record in caplog.records)
