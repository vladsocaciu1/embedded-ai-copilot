import json
import re
from pathlib import Path

import pytest

from app.chunker import chunk_index, unique_texts
from app.code_graph import build_graph
from app.code_index import build_index
from app.config import DEFAULT_EXCLUDE_DIRS
from app.context_selector import rank_files
from app.models import ProjectFile
from app.project_loader import load_project_files
from scripts.embed_check import select_scope
from scripts.eval_selection import CaseResult, case_rank, evaluate_gate, main, summarize
from tests.fake_embedder import FakeEmbedder


BOARDS = ("STM32F4-Discovery", "STM32F429I-Discovery")
EXAMPLES = {
    "GPIO_EXTI": (
        "EXTI0_IRQHandler",
        "HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_0);",
        "BSP_LED_Toggle(LED3);",
    ),
    "UART_TwoBoards_ComIT": (
        "USART2_IRQHandler",
        "HAL_UART_IRQHandler(&UartHandle);",
        "HAL_UART_Transmit_IT(&UartHandle, aTxBuffer, TXBUFFERSIZE);",
    ),
    "ADC_RegularConversion": (
        "ADC_IRQHandler",
        "HAL_ADC_IRQHandler(&AdcHandle);",
        "HAL_ADC_Start_IT(&AdcHandle);",
    ),
}


def eval_files() -> list[ProjectFile]:
    files = []
    for board in BOARDS:
        for example, (irq, irq_body, main_call) in EXAMPLES.items():
            root = Path("Projects", board, "Examples", example.split("_")[0], example)
            files += [
                ProjectFile(
                    root / "Inc/main.h",
                    '#include "stm32f4xx_hal.h"\nvoid Error_Handler(void);\n',
                ),
                ProjectFile(
                    root / "Inc/stm32f4xx_it.h",
                    f"void SysTick_Handler(void);\nvoid {irq}(void);\n",
                ),
                ProjectFile(
                    root / "Src/main.c",
                    '#include "main.h"\nstatic void SystemClock_Config(void);\n'
                    "int main(void)\n{\n    HAL_Init();\n    SystemClock_Config();\n"
                    f"    BSP_LED_Init(LED3);\n    {main_call}\n    while (1) {{ }}\n}}\n"
                    "static void SystemClock_Config(void)\n{\n    HAL_RCC_OscConfig(&RCC_OscInitStruct);\n"
                    "    HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5);\n}\n"
                    "void Error_Handler(void)\n{\n    while (1) { }\n}\n",
                ),
                ProjectFile(
                    root / "Src/stm32f4xx_it.c",
                    '#include "main.h"\n#include "stm32f4xx_it.h"\n'
                    "void SysTick_Handler(void)\n{\n    HAL_IncTick();\n}\n"
                    f"void {irq}(void)\n{{\n    {irq_body}\n}}\n",
                ),
                ProjectFile(
                    root / "Src/system_stm32f4xx.c",
                    "uint32_t SystemCoreClock = 16000000;\nvoid SystemInit(void)\n{\n}\n"
                    "void SystemCoreClockUpdate(void)\n{\n}\n",
                ),
                ProjectFile(
                    root / "EWARM/startup_stm32f407xx.s",
                    "Reset_Handler\n  BL SystemInit\n  BL main\n",
                ),
            ]
    bsp = Path("Drivers/BSP/STM32F4-Discovery")
    files += [
        ProjectFile(
            bsp / "stm32f4_discovery.h",
            "void BSP_LED_Init(Led_TypeDef Led);\nvoid BSP_LED_Toggle(Led_TypeDef Led);\n",
        ),
        ProjectFile(
            bsp / "stm32f4_discovery.c",
            '#include "stm32f4_discovery.h"\n'
            "void BSP_LED_Init(Led_TypeDef Led)\n{\n    HAL_GPIO_Init(GPIO_PORT[Led], &GPIO_InitStruct);\n}\n"
            "void BSP_LED_Toggle(Led_TypeDef Led)\n{\n    HAL_GPIO_TogglePin(GPIO_PORT[Led], GPIO_PIN[Led]);\n}\n",
        ),
    ]
    return files


_FILES = eval_files()
_INDEX = build_index(_FILES)
_GRAPH = build_graph(_INDEX)

_CASES = [
    pytest.param(
        "In STM32F4-Discovery GPIO_EXTI, what does EXTI0_IRQHandler call?",
        ["Projects/STM32F4-Discovery/Examples/GPIO/GPIO_EXTI/Src/stm32f4xx_it.c"],
        1,
        id="E1",
    ),
    pytest.param(
        "How is the system clock configured in the GPIO_EXTI example for STM32F4-Discovery?",
        ["Projects/STM32F4-Discovery/Examples/GPIO/GPIO_EXTI/Src/main.c"],
        2,
        id="E2",
    ),
    pytest.param(
        "What does BSP_LED_Init do?",
        [
            "Drivers/BSP/STM32F4-Discovery/stm32f4_discovery.c",
            "Drivers/BSP/STM32F4-Discovery/stm32f4_discovery.h",
        ],
        2,
        id="E3",
    ),
    pytest.param(
        "Explain Projects/STM32F429I-Discovery/Examples/UART/UART_TwoBoards_ComIT/Src/main.c",
        ["Projects/STM32F429I-Discovery/Examples/UART/UART_TwoBoards_ComIT/Src/main.c"],
        1,
        id="E4",
    ),
    pytest.param(
        "How does UART_TwoBoards_ComIT send data on STM32F4-Discovery?",
        ["Projects/STM32F4-Discovery/Examples/UART/UART_TwoBoards_ComIT/Src/main.c"],
        3,
        id="E5",
    ),
    pytest.param(
        "Where is HAL_UART_Transmit_IT called?",
        [
            "Projects/STM32F4-Discovery/Examples/UART/UART_TwoBoards_ComIT/Src/main.c",
            "Projects/STM32F429I-Discovery/Examples/UART/UART_TwoBoards_ComIT/Src/main.c",
        ],
        2,
        id="E6",
    ),
    # weak spot: no "interrupt"->IRQ synonym
    pytest.param(
        "Which function handles the ADC interrupt in ADC_RegularConversion on STM32F429I-Discovery?",
        ["Projects/STM32F429I-Discovery/Examples/ADC/ADC_RegularConversion/Src/stm32f4xx_it.c"],
        4,
        id="E7",
    ),
    pytest.param(
        "What does SystemInit do in STM32F4-Discovery GPIO_EXTI?",
        ["Projects/STM32F4-Discovery/Examples/GPIO/GPIO_EXTI/Src/system_stm32f4xx.c"],
        1,
        id="E8",
    ),
    pytest.param(
        "Where is EXTI0_IRQHandler declared in STM32F429I-Discovery GPIO_EXTI?",
        ["Projects/STM32F429I-Discovery/Examples/GPIO/GPIO_EXTI/Inc/stm32f4xx_it.h"],
        2,
        id="E9",
    ),
    # weak spot: the definition ranks before callers
    pytest.param(
        "Who calls BSP_LED_Toggle?",
        [
            "Projects/STM32F4-Discovery/Examples/GPIO/GPIO_EXTI/Src/main.c",
            "Projects/STM32F429I-Discovery/Examples/GPIO/GPIO_EXTI/Src/main.c",
        ],
        4,
        id="E10",
    ),
    # weak spot: plain language, no EXTI/KEY name
    pytest.param(
        "What happens when the user button is pressed on STM32F4-Discovery?",
        ["Projects/STM32F4-Discovery/Examples/GPIO/GPIO_EXTI/Src/main.c"],
        10,
        id="P1",
    ),
    pytest.param(
        "How does STM32F429I-Discovery read an analog input?",
        ["Projects/STM32F429I-Discovery/Examples/ADC/ADC_RegularConversion/Src/main.c"],
        3,
        id="P2",
    ),
    pytest.param(
        "How are bytes sent between two boards over the serial port on STM32F4-Discovery?",
        ["Projects/STM32F4-Discovery/Examples/UART/UART_TwoBoards_ComIT/Src/main.c"],
        3,
        id="P3",
    ),
    pytest.param(
        "Which file switches an LED on and off?",
        ["Drivers/BSP/STM32F4-Discovery/stm32f4_discovery.c"],
        1,
        id="P4",
    ),
]


@pytest.mark.parametrize("question,expected,k", _CASES)
def test_selection_eval(question, expected, k):
    ranked = [path.as_posix() for path in rank_files(question, _INDEX, _GRAPH)]
    if all(path in ranked[:k] for path in expected):
        return
    lines = [f"top 5: {ranked[:5]}"]
    for path in expected:
        if path in ranked:
            rank = ranked.index(path) + 1
        else:
            rank = "absent"
        lines.append(f"{path} rank {rank}")
    pytest.fail("\n".join(lines))


def _case_result(rank: int | None, k: int) -> CaseResult:
    return CaseResult(
        question="q",
        expected=("a",),
        k=k,
        ranks=(rank,),
        rank=rank,
        passed=rank is not None and rank <= k,
    )


# M1
def test_case_rank_worst_expected_path():
    assert case_rank(["a", "b", "c"], ["a"]) == 1
    assert case_rank(["a", "b", "c"], ["c", "a"]) == 3


# M2
def test_case_rank_absent_or_empty_ranking():
    assert case_rank(["a", "b"], ["a", "missing"]) is None
    assert case_rank([], ["a"]) is None


# M3
def test_summarize_pass_hit5_and_mrr():
    ranks = [1, 2, None, 5, 6]
    ks = [1, 1, 5, 5, 5]
    summary = summarize([_case_result(rank, k) for rank, k in zip(ranks, ks)])
    assert summary.passed == 2
    assert summary.hit5 == 3
    assert summary.mrr == pytest.approx((1 + 0.5 + 0 + 0.2 + 1 / 6) / 5)


# M4
def test_summarize_empty_results():
    summary = summarize([])
    assert summary.cases == 0
    assert summary.passed == 0
    assert summary.hit5 == 0
    assert summary.mrr == 0.0


# M5
def test_cli_prints_summary_and_writes_json(tmp_path, capsys):
    (tmp_path / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
    (tmp_path / "led.c").write_text(
        "void led_on(void) {}\nvoid led_off(void) {}\n", encoding="utf-8"
    )
    (tmp_path / "button.c").write_text("void button(void) {}\n", encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "question": "Where is main defined?",
                    "expected": ["main.c"],
                    "k": 5,
                }
            ]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "out.json"
    main([str(tmp_path), str(cases_path), "--json", str(out_path)])
    captured = capsys.readouterr()
    assert f"== {cases_path.name} ==" in captured.out
    assert "summary: pass" in captured.out
    assert re.search(r"\d+/\d+ passed", captured.out)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(payload) == {"project_files", "files"}
    assert payload["project_files"] == 3
    assert len(payload["files"]) == 1
    entry = payload["files"][0]
    assert set(entry) == {"cases_file", "summary", "results"}
    assert entry["cases_file"] == cases_path.name
    assert set(entry["summary"]) == {"cases", "passed", "hit5", "mrr"}
    assert entry["results"]
    for result in entry["results"]:
        assert set(result) == {"question", "expected", "k", "ranks", "rank", "passed"}


# M6
def test_semantic_case_files_schema():
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    dev = json.loads((scripts / "cube_semantic_dev.json").read_text(encoding="utf-8"))
    heldout = json.loads(
        (scripts / "cube_semantic_heldout.json").read_text(encoding="utf-8")
    )
    assert len(dev) == 12
    assert len(heldout) == 8
    for cases in (dev, heldout):
        for case in cases:
            assert isinstance(case["question"], str) and case["question"]
            assert isinstance(case["expected"], list) and case["expected"]
            assert all(isinstance(path, str) and path for path in case["expected"])
            assert case["k"] == 5
    assert {case["question"] for case in dev}.isdisjoint(
        case["question"] for case in heldout
    )


def _normalize_timing(text: str) -> str:
    return re.sub(r"\d+ ms", "N ms", text)


def _tiny_project(root: Path) -> None:
    (root / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
    (root / "led.c").write_text(
        "void led_on(void) {}\nvoid led_off(void) {}\n", encoding="utf-8"
    )


def _one_case(root: Path) -> Path:
    cases_path = root / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "question": "Where is main defined?",
                    "expected": ["main.c"],
                    "k": 5,
                }
            ]
        ),
        encoding="utf-8",
    )
    return cases_path


# H1
def test_h1_lexical_ranker_matches_default(tmp_path, capsys):
    _tiny_project(tmp_path)
    cases_path = _one_case(tmp_path)
    default_json = tmp_path / "default.json"
    lexical_json = tmp_path / "lexical.json"

    main([str(tmp_path), str(cases_path), "--json", str(default_json)])
    default_out = capsys.readouterr().out
    main(
        [
            str(tmp_path),
            str(cases_path),
            "--ranker",
            "lexical",
            "--json",
            str(lexical_json),
        ]
    )
    lexical_out = capsys.readouterr().out

    assert _normalize_timing(default_out) == _normalize_timing(lexical_out)
    assert json.loads(default_json.read_text(encoding="utf-8")) == json.loads(
        lexical_json.read_text(encoding="utf-8")
    )
    payload = json.loads(default_json.read_text(encoding="utf-8"))
    assert set(payload) == {"project_files", "files"}
    assert set(payload["files"][0]) == {"cases_file", "summary", "results"}


# H2
def test_h2_hybrid_prints_scope_and_embeds_only_in_scope(tmp_path, capsys):
    board = tmp_path / "Board"
    other = tmp_path / "Other"
    board.mkdir()
    other.mkdir()
    (board / "main.c").write_text(
        "void board_marker(void) { return; }\nint main(void) { return 0; }\n",
        encoding="utf-8",
    )
    (other / "main.c").write_text(
        "void other_marker(void) { return; }\n",
        encoding="utf-8",
    )
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "question": "Where is board_marker defined?",
                    "expected": ["Board/main.c"],
                    "k": 5,
                }
            ]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "out.json"
    fake = FakeEmbedder()
    main(
        [
            str(tmp_path),
            str(cases_path),
            "--ranker",
            "hybrid",
            "--scope",
            "Board",
            "--json",
            str(out_path),
        ],
        embedder=fake,
    )
    captured = capsys.readouterr().out
    assert re.search(r"scope Board files \d+ chunks \d+ unique \d+", captured)
    assert re.search(
        r"embed \d+ chunks in \d+\.\d s \(\d+\.\d chunks/s\) model fake-bow",
        captured,
    )
    assert re.search(
        r"(PASS|FAIL) k=\d+ ranks=\[.*\] lexical=\[.*\] \d+ ms ",
        captured,
    )
    assert re.search(
        r"summary: pass \d+/\d+, hit@5 \d+/\d+, MRR \d+\.\d\d",
        captured,
    )
    assert re.search(
        r"lexical: pass \d+/\d+, hit@5 \d+/\d+, MRR \d+\.\d\d",
        captured,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["ranker"] == "hybrid"
    assert payload["scope"] == ["Board"]
    assert payload["model"] == "fake-bow"
    assert set(payload["embed"]) == {"unique", "seconds"}
    entry = payload["files"][0]
    assert set(entry["lexical_summary"]) == {"cases", "passed", "hit5", "mrr"}
    assert entry["lexical_results"]
    assert set(entry["lexical_results"][0]) == {
        "question",
        "expected",
        "k",
        "ranks",
        "rank",
        "passed",
    }
    files = load_project_files(tmp_path, DEFAULT_EXCLUDE_DIRS)
    expected = unique_texts(select_scope(chunk_index(build_index(files), files), ["Board"]))
    assert fake.document_calls == [list(expected.values())]
    joined = "\n".join(fake.document_calls[0])
    assert "other_marker" not in joined
    assert "board_marker" in joined


# H3
def test_h3_hybrid_scope_and_gate_errors(tmp_path, capsys):
    _tiny_project(tmp_path)
    fake = FakeEmbedder()

    with pytest.raises(SystemExit) as missing_scope:
        main([str(tmp_path), "--ranker", "hybrid"], embedder=fake)
    assert missing_scope.value.code == 2

    with pytest.raises(SystemExit) as lexical_gate:
        main([str(tmp_path), "--gate"], embedder=fake)
    assert lexical_gate.value.code == 2
    assert fake.document_calls == []
    assert fake.query_calls == []

    fake = FakeEmbedder()
    with pytest.raises(SystemExit) as unknown:
        main(
            [str(tmp_path), "--ranker", "hybrid", "--scope", "missing"],
            embedder=fake,
        )
    assert unknown.value.code == 2
    assert "missing" in capsys.readouterr().err
    assert fake.document_calls == []

    src = tmp_path / "src"
    src.mkdir()
    (src / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
    (src / "led.c").write_text("void led_on(void) {}\n", encoding="utf-8")
    fake = FakeEmbedder()
    with pytest.raises(SystemExit) as refused:
        main(
            [
                str(tmp_path),
                "--ranker",
                "hybrid",
                "--scope",
                "src",
                "--max-chunks",
                "1",
            ],
            embedder=fake,
        )
    assert refused.value.code == 2
    refused_out = capsys.readouterr().out
    assert "refusing to embed" in refused_out
    assert "narrow --scope" in refused_out
    assert fake.document_calls == []
    assert fake.query_calls == []


def _gate_results(n: int, rank: int | None = 1, k: int = 5) -> list[dict]:
    results = []
    for index in range(n):
        passed = rank is not None and rank <= k
        results.append(
            {
                "question": f"q{index}",
                "expected": ["a.c"],
                "k": k,
                "ranks": [rank],
                "rank": rank,
                "passed": passed,
            }
        )
    return results


def _gate_entry(
    name: str,
    n: int,
    *,
    hit5: int,
    mrr: float,
    passed: int,
    lexical_hit5: int,
    lexical_mrr: float,
    hybrid_rank: int | None = 1,
    lexical_rank: int | None = 1,
) -> dict:
    return {
        "cases_file": name,
        "summary": {"cases": n, "passed": passed, "hit5": hit5, "mrr": mrr},
        "results": _gate_results(n, hybrid_rank),
        "lexical_summary": {
            "cases": n,
            "passed": n,
            "hit5": lexical_hit5,
            "mrr": lexical_mrr,
        },
        "lexical_results": _gate_results(n, lexical_rank),
    }


def _passing_gate_reports() -> dict[str, dict]:
    return {
        "cube_semantic_dev.json": _gate_entry(
            "cube_semantic_dev.json",
            12,
            hit5=8,
            mrr=0.60,
            passed=8,
            lexical_hit5=7,
            lexical_mrr=0.45,
        ),
        "cube_semantic_heldout.json": _gate_entry(
            "cube_semantic_heldout.json",
            8,
            hit5=4,
            mrr=0.45,
            passed=4,
            lexical_hit5=3,
            lexical_mrr=0.40,
        ),
        "cube_eval_cases.json": _gate_entry(
            "cube_eval_cases.json",
            8,
            hit5=8,
            mrr=1.0,
            passed=8,
            lexical_hit5=8,
            lexical_mrr=1.0,
        ),
    }


def _failed_names(result) -> list[str]:
    return [check.name for check in result.checks if not check.passed]


# H4
def test_h4_evaluate_gate_thresholds_and_cases():
    passing = evaluate_gate(_passing_gate_reports())
    assert passing.passed
    assert _failed_names(passing) == []
    assert [check.name for check in passing.checks] == [
        "dev_hit5",
        "dev_mrr",
        "heldout_hit5",
        "heldout_mrr",
        "structural_pass",
        "structural_ranks",
    ]

    threshold = _passing_gate_reports()
    threshold["cube_semantic_dev.json"]["summary"]["hit5"] = 8
    threshold["cube_semantic_dev.json"]["summary"]["mrr"] = 0.55
    threshold["cube_semantic_heldout.json"]["summary"]["hit5"] = 4
    threshold["cube_semantic_heldout.json"]["lexical_summary"]["hit5"] = 3
    threshold["cube_semantic_heldout.json"]["summary"]["mrr"] = 0.45
    threshold["cube_semantic_heldout.json"]["lexical_summary"]["mrr"] = 0.40
    threshold["cube_eval_cases.json"]["results"] = _gate_results(8, rank=2)
    threshold["cube_eval_cases.json"]["lexical_results"] = _gate_results(8, rank=1)
    threshold["cube_eval_cases.json"]["summary"]["passed"] = 8
    assert evaluate_gate(threshold).passed

    failures = {
        "dev_hit5": ("cube_semantic_dev.json", "summary", "hit5", 7),
        "dev_mrr": ("cube_semantic_dev.json", "summary", "mrr", 0.549),
        "heldout_hit5": ("cube_semantic_heldout.json", "summary", "hit5", 3),
        "heldout_mrr": ("cube_semantic_heldout.json", "summary", "mrr", 0.44),
        "structural_pass": ("cube_eval_cases.json", "summary", "passed", 7),
    }
    for name, (filename, section, field, value) in failures.items():
        reports = _passing_gate_reports()
        reports[filename][section][field] = value
        result = evaluate_gate(reports)
        assert not result.passed
        assert _failed_names(result) == [name]

    none_rank = _passing_gate_reports()
    none_rank["cube_eval_cases.json"]["results"] = _gate_results(8, rank=None)
    none_rank["cube_eval_cases.json"]["summary"]["passed"] = 8
    none_result = evaluate_gate(none_rank)
    assert not none_result.passed
    assert _failed_names(none_result) == ["structural_ranks"]

    missing = _passing_gate_reports()
    del missing["cube_semantic_heldout.json"]
    missing_result = evaluate_gate(missing)
    assert not missing_result.passed
    assert _failed_names(missing_result) == ["cases"]
    assert "cube_semantic_heldout.json" in missing_result.checks[0].detail

    short = _passing_gate_reports()
    short["cube_semantic_dev.json"]["summary"]["cases"] = 11
    short["cube_semantic_dev.json"]["results"] = _gate_results(11)
    short["cube_semantic_dev.json"]["lexical_summary"]["cases"] = 11
    short["cube_semantic_dev.json"]["lexical_results"] = _gate_results(11)
    short_result = evaluate_gate(short)
    assert not short_result.passed
    assert _failed_names(short_result) == ["cases"]
    assert "cube_semantic_dev.json" in short_result.checks[0].detail


def _write_role_cases(directory: Path) -> dict[str, Path]:
    counts = {
        "cube_eval_cases.json": 8,
        "cube_semantic_dev.json": 12,
        "cube_semantic_heldout.json": 8,
    }
    paths = {}
    for name, count in counts.items():
        path = directory / name
        path.write_text(
            json.dumps(
                [
                    {
                        "question": f"{name} question {index}",
                        "expected": ["Board/main.c"],
                        "k": 5,
                    }
                    for index in range(count)
                ]
            ),
            encoding="utf-8",
        )
        paths[name] = path
    return paths


def _gate_block(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[lines.index("== gate ==") :])


# H5
def test_h5_gate_block_and_gate_from(tmp_path, capsys):
    board = tmp_path / "Board"
    board.mkdir()
    (board / "main.c").write_text(
        "int main(void) { return 0; }\n",
        encoding="utf-8",
    )
    cases = _write_role_cases(tmp_path)
    combined_json = tmp_path / "combined.json"
    structural_dev_json = tmp_path / "structural-dev.json"
    heldout_json = tmp_path / "heldout.json"
    argv_prefix = [str(tmp_path), "--ranker", "hybrid", "--scope", "Board"]

    main(
        [
            *argv_prefix,
            str(cases["cube_eval_cases.json"]),
            str(cases["cube_semantic_dev.json"]),
            str(cases["cube_semantic_heldout.json"]),
            "--gate",
            "--json",
            str(combined_json),
        ],
        embedder=FakeEmbedder(),
    )
    combined_out = capsys.readouterr().out
    block = _gate_block(combined_out)
    block_lines = block.splitlines()
    assert block_lines[0] == "== gate =="
    assert sum(line.startswith("structural ") for line in block_lines) == 8
    assert sum(line.startswith("dev ") for line in block_lines) == 12
    assert sum(line.startswith("held-out ") for line in block_lines) == 8
    check_lines = [line for line in block_lines if line.startswith("check ")]
    assert len(check_lines) == 6
    assert [line.split()[1] for line in check_lines] == [
        "dev_hit5:",
        "dev_mrr:",
        "heldout_hit5:",
        "heldout_mrr:",
        "structural_pass:",
        "structural_ranks:",
    ]
    assert block_lines[-1].startswith("gate: ")
    payload = json.loads(combined_json.read_text(encoding="utf-8"))
    assert set(payload["gate"]) == {"passed", "checks"}
    assert len(payload["gate"]["checks"]) == 6

    main(
        [
            *argv_prefix,
            str(cases["cube_eval_cases.json"]),
            str(cases["cube_semantic_dev.json"]),
            "--json",
            str(structural_dev_json),
        ],
        embedder=FakeEmbedder(),
    )
    capsys.readouterr()
    main(
        [
            *argv_prefix,
            str(cases["cube_semantic_heldout.json"]),
            "--json",
            str(heldout_json),
        ],
        embedder=FakeEmbedder(),
    )
    capsys.readouterr()
    main(["--gate-from", str(structural_dev_json), str(heldout_json)])
    split_out = capsys.readouterr().out
    assert _gate_block(split_out) == block

    with pytest.raises(SystemExit) as duplicate:
        main(["--gate-from", str(structural_dev_json), str(structural_dev_json)])
    assert duplicate.value.code == 2
    assert "duplicate" in capsys.readouterr().err


def test_e1_scope_dot_selects_every_chunk(tmp_path, capsys):
    (tmp_path / "a.c").write_text("int a(void) { return 1; }\n", encoding="utf-8")
    (tmp_path / "b.c").write_text("int b(void) { return 2; }\n", encoding="utf-8")
    files = load_project_files(tmp_path, DEFAULT_EXCLUDE_DIRS)
    index = build_index(files)
    chunks = chunk_index(index, files)
    assert select_scope(chunks, ["."]) == tuple(chunks)

    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "question": "Where is a defined?",
                    "expected": ["a.c"],
                    "k": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    main(
        [
            str(tmp_path),
            str(cases_path),
            "--ranker",
            "hybrid",
            "--scope",
            ".",
            "--max-chunks",
            "100",
        ],
        embedder=FakeEmbedder(),
    )
    out = capsys.readouterr().out
    assert any(line.startswith("scope . files 2 chunks") for line in out.splitlines())
