import random
from pathlib import Path

from app.code_graph import build_graph
from app.code_index import build_index
from app.context_selector import pack_files, rank_files
from app.models import ProjectFile


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
