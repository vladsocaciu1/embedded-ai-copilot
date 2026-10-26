import textwrap
from pathlib import Path

from app.chunker import chunk_file, chunk_index, unique_texts
from app.code_graph import build_graph
from app.code_index import build_index, index_file
from app.context_selector import rank_files_detailed
from app.models import ProjectFile, SymbolKind

SWAP = textwrap.dedent(
    """\
    #include <zephyr/arch/cpu.h>

    _ASM_FILE_PROLOGUE

    GTEXT(arm_m_pendsv)
    GTEXT(arm_m_svc)

    /**
     * @brief PendSV exception handler, handling context switches
     */
    SECTION_FUNC(TEXT, arm_m_pendsv)
    \tldr r0, =_kernel
    #if defined(CONFIG_FPU)
    \tvpush {s16-s31}
    #endif /* CONFIG_FPU */
    _thread_irq_disabled:
    \tbx lr

    /* SVC handler */
    SECTION_FUNC(TEXT, arm_m_svc)
    \tsvc 0
    \tbx lr

    GTEXT(not_a_function)
    """
)


def _fs(path: str, content: str):
    return index_file(ProjectFile(path=Path(path), content=content))


def _functions(path: str, content: str):
    return [
        (s.name, s.start_line, s.end_line)
        for s in _fs(path, content).symbols
        if s.kind is SymbolKind.FUNCTION
    ]


def test_a1_section_func_symbols_and_ends():
    fs = _fs("arch/swap_helper.S", SWAP)
    assert fs.grammar is None and fs.includes == ()
    assert _functions("arch/swap_helper.S", SWAP) == [
        ("arm_m_pendsv", 11, 17),
        ("arm_m_svc", 20, 24),
    ]


def test_a2_gtext_label_and_type_function():
    gtext = "GTEXT(entry)\nGTEXT(helper)\nentry:\n\tnop\nhelper:\n\tnop\n"
    assert _functions("a.S", gtext) == [("entry", 3, 4), ("helper", 5, 6)]
    typed = ".global f\n.type f, %function\nf:\n\tnop\ng:\n\tnop\n"
    assert _functions("b.s", typed) == [("f", 3, 6)]


def test_a3_markers_and_undeclared_labels_are_ignored():
    text = "start:\n\tnop\nSECTION_FUNC(TEXT, f)\n\tnop\ninner:\n\tnop\n"
    assert _functions("c.S", text) == [("f", 3, 6)]


def test_a4_macros_are_symbols_and_no_functions_falls_back_to_pieces():
    fs = _fs("m.S", ".macro push_regs\n\tnop\n.endm\n")
    assert [(s.name, s.kind) for s in fs.symbols] == [("push_regs", SymbolKind.MACRO)]
    chunks = chunk_file(fs, ".macro push_regs\n\tnop\n.endm\n")
    assert [c.kind for c in chunks] == ["file", "text"]


def test_a5_function_chunk_has_strict_lead_and_card_lists_functions():
    chunks = chunk_file(_fs("arch/swap_helper.S", SWAP), SWAP)
    assert [(c.kind, c.name) for c in chunks] == [
        ("file", ""), ("function", "arm_m_pendsv"), ("function", "arm_m_svc"),
    ]
    assert "symbols: arm_m_pendsv arm_m_svc" in chunks[0].text
    pendsv = chunks[1].text
    assert pendsv.startswith("function arm_m_pendsv in ")
    assert "PendSV exception handler" in pendsv
    assert "bx lr" in pendsv and "SVC handler" not in pendsv
    svc = chunks[2].text
    assert "SVC handler" in svc and "#endif" not in svc


def test_a6_keys_are_unique_across_mixed_project():
    files = [
        ProjectFile(Path("src/main.c"), "void main(void) { return; }\n"),
        ProjectFile(Path("arch/swap_helper.S"), SWAP),
        ProjectFile(Path("arch/other.S"), SWAP.replace("arm_m_", "other_")),
        ProjectFile(Path("link.ld"), "SECTIONS {}\n"),
    ]
    chunks = chunk_index(build_index(files), files)
    assert len({c.key for c in chunks}) == len(chunks) == len(unique_texts(chunks))
    assert all(c.path.suffix != ".ld" for c in chunks)


def test_a7_pendsv_question_finds_the_assembly_file():
    files = [
        ProjectFile(Path("arch/swap_helper.S"), SWAP),
        ProjectFile(Path("kernel/sched.c"), "void z_swap(void) { return; }\n"),
        ProjectFile(Path("kernel/thread.c"), "void z_thread(void) { return; }\n"),
    ]
    index = build_index(files)
    graph = build_graph(index)
    question = "What does PendSV do during a context switch?"
    assert rank_files_detailed(question, index, graph)[0].path == Path("arch/swap_helper.S")
    named = rank_files_detailed("What does arm_m_pendsv do?", index, graph)[0]
    assert (named.path, named.reason) == (Path("arch/swap_helper.S"), "defines")
