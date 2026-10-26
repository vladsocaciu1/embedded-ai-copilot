from pathlib import Path

from app.code_graph import build_graph, update_graph
from app.code_index import build_index, index_file
from app.context_selector import _syscall_variants, rank_files_detailed
from app.models import ProjectFile
from app.text_files import config_refs


def _project(files: dict[str, str]):
    items = [ProjectFile(Path(p), c) for p, c in files.items()]
    index = build_index(items)
    return items, index, build_graph(index)


def test_n1_syscall_variants():
    assert _syscall_variants("k_sem_take") == ("z_impl_k_sem_take", "z_vrfy_k_sem_take")
    assert _syscall_variants("z_impl_k_sem_take") == ("k_sem_take", "z_vrfy_k_sem_take")
    assert _syscall_variants("z_vrfy_k_sem_take") == ("k_sem_take", "z_impl_k_sem_take")
    assert _syscall_variants("z_impl_") == ("z_impl_z_impl_", "z_vrfy_z_impl_")


def test_n2_syscall_name_ranks_the_implementation_over_its_declaration():
    _, index, graph = _project({
        "include/kernel.h": "__syscall int k_foo_take(int x);\nstatic inline void use(void) { k_foo_take(1); }\n",
        "kernel/foo.c": "int z_impl_k_foo_take(int x) { return x; }\n",
        "kernel/bar.c": "void bar(void) { k_foo_take(2); }\n",
    })
    ranked = rank_files_detailed("How does k_foo_take work?", index, graph)
    assert ranked[0].path == Path("kernel/foo.c")
    assert (ranked[0].tier, ranked[0].reason) == (1, "defines")


def test_n3_config_refs_per_line_and_name():
    text = "#if CONFIG_A && defined(CONFIG_B)\nint x = IS_ENABLED(CONFIG_A) + CONFIG_A;\n// none\n"
    refs = config_refs(text)
    assert [(c.callee, c.line) for c in refs] == [("CONFIG_A", 1), ("CONFIG_B", 1), ("CONFIG_A", 2)]
    assert all(c.caller == "" and not c.is_member for c in refs)
    assert config_refs("no options here\n") == ()
    assert config_refs("CONFIG_lower and XCONFIG_B\n") == ()


def test_n4_config_refs_by_file_kind():
    c = index_file(ProjectFile(Path("a.c"), "#ifdef CONFIG_X\nvoid f(void) { g(); }\n#endif\n"))
    assert [(x.caller, x.callee) for x in c.calls] == [("f", "g"), ("", "CONFIG_X")]
    asm = index_file(ProjectFile(Path("a.S"), "#if defined(CONFIG_X)\nnop\n#endif\n"))
    assert [x.callee for x in asm.calls] == ["CONFIG_X"]
    cmake = index_file(ProjectFile(Path("CMakeLists.txt"), "zephyr_sources_ifdef(CONFIG_X a.c)\n"))
    assert [x.callee for x in cmake.calls] == ["CONFIG_X"]
    for name, text in (("prj.conf", "CONFIG_X=y\n"), ("Kconfig", "config X\n\tbool\n\tdepends on CONFIG_Y\n")):
        assert index_file(ProjectFile(Path(name), text)).calls == ()


def test_n5_config_question_ranks_definition_first_then_users_by_count():
    _, index, graph = _project({
        "kernel/Kconfig": "config FOO\n\tbool \"foo\"\n",
        "kernel/a.c": "#ifdef CONFIG_FOO\nint a;\n#endif\n#if CONFIG_FOO\nint b;\n#endif\n",
        "kernel/b.c": "#ifdef CONFIG_FOO\nint c;\n#endif\n",
        "kernel/c.c": "int unrelated;\n",
    })
    ranked = rank_files_detailed("What does CONFIG_FOO change?", index, graph)
    assert [(r.path.name, r.reason) for r in ranked[:3]] == [
        ("Kconfig", "defines"), ("a.c", "call"), ("b.c", "call"),
    ]
    assert ranked[1].hits == 2 and ranked[2].hits == 1
    assert "c.c" not in [r.path.name for r in ranked]


def test_n6_config_refs_survive_an_incremental_graph_update():
    items, index, graph = _project({"a.c": "#if CONFIG_Q\nint a;\n#endif\n", "b.c": "int b;\n"})
    changed = [items[0], ProjectFile(Path("b.c"), "#if CONFIG_Q\nint b;\n#endif\n")]
    new_index = build_index(changed)
    updated = update_graph(graph, index, new_index, frozenset({Path("b.c")}))
    assert sorted(r.path.name for r in updated.callers("CONFIG_Q")) == ["a.c", "b.c"]
    assert sorted(r.path.name for r in build_graph(new_index).callers("CONFIG_Q")) == ["a.c", "b.c"]
