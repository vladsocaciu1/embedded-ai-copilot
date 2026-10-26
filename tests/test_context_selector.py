import random
from pathlib import Path

from app.code_graph import build_graph
from app.code_index import build_index
from app.context_selector import fuse_rankings, pack_files, rank_files, rank_files_detailed
from app.models import ProjectFile, RankedFile


def _rank(question: str, files: list[ProjectFile]) -> tuple[Path, ...]:
    index = build_index(files)
    return rank_files(question, index, build_graph(index))


def _block(path: Path, content: str) -> int:
    return len(content) + len(path.as_posix()) + 64


def test_named_file_tail_ranks_exact_path_first():
    src = Path("src/can.c")
    other_dir = Path("drivers/can.c")
    same_dir = Path("src/other.c")
    ranked = _rank(
        "Explain src/can.c",
        [
            ProjectFile(other_dir, "int from_drivers;\n"),
            ProjectFile(same_dir, "int from_other;\n"),
            ProjectFile(src, "int from_src;\n"),
        ],
    )

    assert ranked[0] == src
    assert ranked.index(src) < ranked.index(same_dir)


def test_defining_file_ranks_before_its_pair():
    source = Path("src/can.c")
    header = Path("inc/can.h")
    ranked = _rank(
        "What does Can_Init do?",
        [
            ProjectFile(
                header,
                "void Can_Init(void);\n",
            ),
            ProjectFile(
                source,
                '#include "can.h"\nvoid Can_Init(void)\n{\n    return;\n}\n',
            ),
        ],
    )

    assert ranked[0] == source
    assert ranked[1] == header


def test_external_seed_ranks_more_call_sites_first():
    few = Path("a.c")
    many = Path("z.c")
    ranked = _rank(
        "Where is HAL_GPIO_Init called?",
        [
            ProjectFile(
                few,
                "void g(void)\n{\n    HAL_GPIO_Init();\n}\n",
            ),
            ProjectFile(
                many,
                "void f(void)\n{\n    HAL_GPIO_Init();\n    HAL_GPIO_Init();\n"
                "    HAL_GPIO_Init();\n}\n",
            ),
        ],
    )

    assert few in ranked
    assert many in ranked
    assert ranked.index(many) < ranked.index(few)


def test_callee_definitions_rank_before_unrelated_file():
    caller = Path("app.c")
    callee = Path("motor.c")
    unrelated = Path("other.c")
    ranked = _rank(
        "What does App_Run call?",
        [
            ProjectFile(unrelated, "void app_helper(void)\n{\n    return;\n}\n"),
            ProjectFile(callee, "void Motor_Start(void)\n{\n    return;\n}\n"),
            ProjectFile(caller, "void App_Run(void)\n{\n    Motor_Start();\n}\n"),
        ],
    )

    assert ranked.index(callee) < ranked.index(unrelated)


def test_resolved_include_ranks_before_lexical_only_file():
    source = Path("src/app.c")
    header = Path("inc/util.h")
    lexical = Path("lexical.c")
    ranked = _rank(
        "What does App_Run call?",
        [
            ProjectFile(lexical, "void app_note(void)\n{\n    return;\n}\n"),
            ProjectFile(header, "void Util_Ready(void);\n"),
            ProjectFile(
                source,
                '#include "util.h"\nvoid App_Run(void)\n{\n    return;\n}\n',
            ),
        ],
    )

    assert ranked.index(header) < ranked.index(lexical)


def test_focus_dir_puts_named_project_first():
    files = []
    for project in ("Proj_A", "Proj_B"):
        for name in ("src/main.c", "src/extra.c"):
            files.append(
                ProjectFile(
                    Path(project) / name,
                    "int main(void)\n{\n    return 0;\n}\n"
                    if name.endswith("main.c")
                    else "void extra(void)\n{\n    return;\n}\n",
                )
            )
    ranked = _rank("How does Proj_B start?", files)
    proj_b = [path for path in ranked if path.parts[0] == "Proj_B"]
    proj_a = [path for path in ranked if path.parts[0] == "Proj_A"]

    assert proj_b
    assert proj_a
    assert max(ranked.index(path) for path in proj_b) < min(
        ranked.index(path) for path in proj_a
    )


def test_plain_word_is_not_a_seed_and_ranks_by_path():
    data_var = Path("x.c")
    data_init = Path("y.c")
    ranked = _rank(
        "What is data?",
        [
            ProjectFile(data_init, "void Data_Init(void)\n{\n    return;\n}\n"),
            ProjectFile(data_var, "int data;\n"),
        ],
    )

    assert ranked == (data_var, data_init)


def test_prefix_word_ranks_configured_symbol_first():
    clock = Path("src/clock.c")
    other = Path("src/other.c")
    ranked = _rank(
        "How is the clock configured?",
        [
            ProjectFile(other, "void idle(void)\n{\n    return;\n}\n"),
            ProjectFile(
                clock,
                "void SystemClock_Config(void)\n{\n    return;\n}\n",
            ),
        ],
    )

    assert ranked[0] == clock
    assert other not in ranked or ranked.index(clock) < ranked.index(other)


def test_no_candidates_falls_back_to_entry_files_by_path():
    main = Path("src/main.c")
    startup = Path("startup_x.s")
    linker = Path("link.ld")
    other = Path("other.c")
    ranked = _rank(
        "Hello?",
        [
            ProjectFile(other, "int value;\n"),
            ProjectFile(startup, "nop\n"),
            ProjectFile(main, "int main(void)\n{\n    return 0;\n}\n"),
            ProjectFile(linker, "SECTIONS\n{\n}\n"),
        ],
    )

    assert ranked == (linker, main, startup)


def test_equal_signals_rank_parsed_file_before_unparsed():
    parsed = Path("z_buffer.c")
    unparsed = Path("a_buffer.s")
    ranked = _rank(
        "Where is buffer?",
        [
            ProjectFile(unparsed, "nop\n"),
            ProjectFile(parsed, "int buffer;\n"),
        ],
    )

    assert ranked[0] == parsed
    assert ranked.index(parsed) < ranked.index(unparsed)


def test_shuffled_input_ranks_the_same():
    files = [
        ProjectFile(Path("src/can.c"), "void Can_Init(void)\n{\n    return;\n}\n"),
        ProjectFile(Path("inc/can.h"), "void Can_Init(void);\n"),
        ProjectFile(Path("src/app.c"), "void App_Run(void)\n{\n    Can_Init();\n}\n"),
        ProjectFile(Path("src/other.c"), "void app_helper(void)\n{\n    return;\n}\n"),
    ]
    question = "What does Can_Init do?"
    first = _rank(question, files)
    reversed_files = list(reversed(files))
    shuffled = files.copy()
    random.Random(0).shuffle(shuffled)

    assert first == _rank(question, reversed_files) == _rank(question, shuffled)
    assert len(first) > 1


def test_pack_skips_oversized_file_and_keeps_rank_order():
    big = Path("big.c")
    later_a = Path("a.c")
    later_b = Path("b.c")
    files = {
        big: ProjectFile(big, "x" * 1000),
        later_a: ProjectFile(later_a, "aa"),
        later_b: ProjectFile(later_b, "bbb"),
    }
    ranked = (big, later_a, later_b)
    max_chars = _block(later_a, "aa") + _block(later_b, "bbb")
    packed = pack_files(ranked, files, max_chars)

    assert packed == (later_a, later_b)
    assert _block(big, files[big].content) > max_chars
    assert sum(_block(path, files[path].content) for path in packed) <= max_chars


def test_pack_zero_budget_returns_empty():
    path = Path("a.c")
    files = {path: ProjectFile(path, "int x;\n")}

    assert pack_files((path,), files, 0) == ()


def _detailed(question: str, files: list[ProjectFile]):
    index = build_index(files)
    return rank_files_detailed(question, index, build_graph(index))


def test_rank_files_detailed_paths_match_rank_files():
    files = [
        ProjectFile(Path("src/can.c"), "void Can_Init(void)\n{\n    return;\n}\n"),
        ProjectFile(Path("inc/can.h"), "void Can_Init(void);\n"),
        ProjectFile(Path("src/app.c"), "void App_Run(void)\n{\n    Can_Init();\n}\n"),
        ProjectFile(Path("src/other.c"), "void app_helper(void)\n{\n    return;\n}\n"),
    ]
    question = "What does Can_Init do?"
    index = build_index(files)
    graph = build_graph(index)

    detailed = rank_files_detailed(question, index, graph)

    assert tuple(item.path for item in detailed) == rank_files(question, index, graph)


def test_rank_files_detailed_reasons():
    named = Path("src/can.c")
    named_ranked = _detailed(
        "Explain src/can.c",
        [
            ProjectFile(named, "int from_src;\n"),
            ProjectFile(Path("src/other.c"), "int from_other;\n"),
        ],
    )
    named_hit = next(item for item in named_ranked if item.path == named)
    assert named_hit.reason == "named"

    source = Path("src/can.c")
    header = Path("inc/can.h")
    defined = _detailed(
        "What does Can_Init do?",
        [
            ProjectFile(header, "void Can_Init(void);\n"),
            ProjectFile(
                source,
                '#include "can.h"\nvoid Can_Init(void)\n{\n    return;\n}\n',
            ),
        ],
    )
    definer = next(item for item in defined if item.path == source)
    pair = next(item for item in defined if item.path == header)
    assert definer.reason == "defines"
    assert definer.tier == 1
    assert definer.hits >= 1
    assert pair.reason == "pair"

    lexical = Path("src/clock.c")
    lexical_ranked = _detailed(
        "How is the clock configured?",
        [
            ProjectFile(Path("src/other.c"), "void idle(void)\n{\n    return;\n}\n"),
            ProjectFile(
                lexical,
                "void SystemClock_Config(void)\n{\n    return;\n}\n",
            ),
        ],
    )
    lexical_hit = next(item for item in lexical_ranked if item.path == lexical)
    assert lexical_hit.reason == "lexical"
    assert lexical_hit.tier == 9

    focus = Path("ab1/foo.c")
    focus_ranked = _detailed(
        "How does ab1 work?",
        [ProjectFile(focus, "int x;\n")],
    )
    assert focus_ranked[0].path == focus
    assert focus_ranked[0].reason == "focus"


def test_rank_files_detailed_entry_reason_when_no_candidates():
    files = [
        ProjectFile(Path("other.c"), "int value;\n"),
        ProjectFile(Path("startup_x.s"), "nop\n"),
        ProjectFile(Path("src/main.c"), "int main(void)\n{\n    return 0;\n}\n"),
        ProjectFile(Path("link.ld"), "SECTIONS\n{\n}\n"),
    ]

    detailed = _detailed("Hello?", files)

    assert detailed
    assert all(
        item.reason == "entry"
        and item.focus == 0
        and item.tier == 9
        and item.hits == 0
        and item.lexical == 0
        for item in detailed
    )


def _ranked(
    path: str,
    *,
    reason: str = "focus",
    focus: int = 1,
    tier: int = 9,
    hits: int = 0,
    lexical: int = 0,
) -> RankedFile:
    return RankedFile(Path(path), reason, focus, tier, hits, lexical)


# F1
def test_f1_focus_fusion_follows_rrf_and_keeps_non_focus_order():
    ranked = (
        _ranked("a.c", reason="named", tier=0, hits=2, lexical=1),
        _ranked("b.c", reason="focus", lexical=4),
        _ranked("c.c", reason="lexical", focus=1, tier=9, hits=1, lexical=3),
        _ranked("z.c", reason="lexical", focus=0, lexical=2),
        _ranked("m.c", reason="entry", focus=0, tier=9),
    )
    fused = fuse_rankings(ranked, [Path("c.c"), Path("a.c")])
    k = 60
    sem = {Path("c.c"): 1, Path("a.c"): 2}
    lex = {Path("a.c"): 1, Path("b.c"): 2, Path("c.c"): 3}

    def score(path: Path) -> float:
        total = 1 / (k + lex[path])
        if path in sem:
            total += 1 / (k + sem[path])
        return total

    focus_order = tuple(sorted(lex, key=lambda path: (-score(path), path.parts)))
    assert tuple(item.path for item in fused) == focus_order + (Path("z.c"), Path("m.c"))
    assert [item.semantic for item in fused[:3]] == [
        sem.get(path) for path in focus_order
    ]
    assert [item.semantic for item in fused[3:]] == [None, None]
    by_path = {item.path: item for item in ranked}
    for item in fused:
        source = by_path[item.path]
        assert item.reason == source.reason
        assert item.focus == source.focus
        assert item.tier == source.tier
        assert item.hits == source.hits
        assert item.lexical == source.lexical


# F2
def test_f2_semantic_paths_outside_focus_do_not_shift_ranks():
    ranked = (
        _ranked("a.c"),
        _ranked("b.c"),
        _ranked("c.c"),
        _ranked("n.c", reason="lexical", focus=0),
    )
    inside = fuse_rankings(ranked, [Path("c.c"), Path("a.c")])
    outside = fuse_rankings(
        ranked,
        [Path("outside.c"), Path("n.c"), Path("c.c"), Path("a.c")],
    )

    assert outside == inside
    semantic = {item.path: item.semantic for item in outside}
    assert semantic == {
        Path("a.c"): 2,
        Path("b.c"): None,
        Path("c.c"): 1,
        Path("n.c"): None,
    }


# F3
def test_f3_empty_semantic_order_matches_input():
    ranked = (
        _ranked("b.c", reason="named", tier=0, hits=1, lexical=2),
        _ranked("a.c", reason="lexical", focus=0, lexical=1),
    )

    assert fuse_rankings(ranked, ()) == ranked
    assert fuse_rankings(ranked, []) == ranked


# F4
def test_f4_symmetric_tie_breaks_by_path_parts():
    later = _ranked("b.c", hits=1, lexical=3)
    earlier = _ranked("a.c", hits=1, lexical=3)
    ranked = (later, earlier)
    order = [Path("a.c"), Path("b.c")]

    first = fuse_rankings(ranked, order)
    second = fuse_rankings(ranked, order)

    assert [item.path for item in first] == order
    assert first == second


# F5
def test_f5_without_focus_semantic_only_path_uses_semantic_reason():
    entry = RankedFile(Path("src/main.c"), "entry", 0, 9, 0, 0)
    only = Path("only.c")

    fused = fuse_rankings((entry,), [only, entry.path])

    assert fused[0] == RankedFile(entry.path, "entry", 0, 9, 0, 0, 2)
    assert fused[1] == RankedFile(only, "semantic", 0, 9, 0, 0, 1)


# F6
def test_f6_duplicates_are_ignored_and_k_changes_order():
    first = Path("a.c")
    second = Path("b.c")
    ranked = (_ranked("a.c"), _ranked("b.c"))
    fused = fuse_rankings(ranked, [first, first, second])

    assert [item.semantic for item in fused] == [1, 2]

    alone = Path("s.c")
    fillers = [Path(f"f{index}.c") for index in range(3)]
    both = Path("m.c")
    extra = Path("x.c")
    rows = [_ranked("s.c")]
    rows.extend(_ranked(path.as_posix()) for path in fillers)
    rows.append(_ranked("m.c"))
    rows.append(_ranked("x.c"))
    semantic = [*fillers, extra, both]
    order_k0 = [item.path for item in fuse_rankings(rows, semantic, k=0)]
    order_k60 = [item.path for item in fuse_rankings(rows, semantic, k=60)]

    assert order_k0.index(alone) < order_k0.index(both)
    assert order_k60.index(both) < order_k60.index(alone)


# F7
def test_f7_rank_files_detailed_semantic_is_none():
    detailed = _detailed(
        "What does Can_Init do?",
        [
            ProjectFile(Path("inc/can.h"), "void Can_Init(void);\n"),
            ProjectFile(
                Path("src/can.c"),
                '#include "can.h"\nvoid Can_Init(void)\n{\n    return;\n}\n',
            ),
        ],
    )

    assert detailed
    assert all(item.semantic is None for item in detailed)


def test_x1_config_option_terms_rank_the_defining_kconfig_first():
    from app.context_selector import _file_terms

    foo = ProjectFile(Path("kernel/Kconfig"), "config FOO\n\tbool\n")
    bar = ProjectFile(Path("subsys/Kconfig"), "config BAR\n\tbool\n")
    other = ProjectFile(Path("kernel/other.c"), "void unrelated(void) {}\n")
    index = build_index([other, foo, bar])
    terms = _file_terms(next(file for file in index.files if file.path == foo.path))

    assert "foo" in terms
    assert "config" in terms
    ranked = _rank("How is CONFIG_FOO used?", [other, bar, foo])
    assert ranked[0] == foo.path


def test_x2_named_kconfig_and_cmakelists_rank_first():
    kconfig = Path("kernel/Kconfig")
    cmake = Path("CMakeLists.txt")
    main = Path("src/main.c")
    other = Path("src/other.c")
    files = [
        ProjectFile(other, "void other(void) {}\n"),
        ProjectFile(cmake, "project(app)\n"),
        ProjectFile(kconfig, "config A\n\tbool\n"),
        ProjectFile(main, "void main(void) {}\n"),
    ]

    assert _rank("Explain the Kconfig options", files)[0] == kconfig
    assert _rank("What is in CMakeLists.txt?", files)[0] == cmake
    assert _rank("Explain src/main.c", files)[0] == main


def test_x3_rst_section_title_is_a_file_term():
    from app.context_selector import _file_terms

    rst = ProjectFile(
        Path("doc/index.rst"),
        "Thread states\n=============\n\nA thread can be unready.\n",
    )
    index = build_index([rst])

    terms = _file_terms(index.files[0])
    assert "thread" in terms
    assert "states" in terms
