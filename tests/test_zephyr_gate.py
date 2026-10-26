import json
from pathlib import Path

import pytest

from scripts.zephyr_gate import evaluate, main, tag_table


def _result(question: str, passed: bool = True, rank: int | None = 1) -> dict:
    return {"question": question, "passed": passed, "rank": rank}


def _summary(cases: int, passed: int, hit5: int, mrr: float) -> dict:
    return {"cases": cases, "passed": passed, "hit5": hit5, "mrr": mrr}


def _dev(results: list[dict], summary: dict | None = None) -> dict:
    if summary is None:
        summary = _summary(18, 18, 18, 1.0)
    return {
        "cases_file": "zephyr_semantic_dev.json",
        "summary": summary,
        "results": results,
    }


def _structural(passed: int = 8, cases: int = 8) -> dict:
    return {
        "cases_file": "zephyr_eval_cases.json",
        "summary": _summary(cases, passed, passed, 1.0),
        "results": [],
    }


def _heldout(hit5: int, lexical: int) -> dict:
    return {
        "scope": ["."],
        "embed": {"unique": 1225},
        "files": [
            {
                "cases_file": "zephyr_semantic_heldout.json",
                "summary": _summary(8, hit5, hit5, 0.5),
                "lexical_summary": _summary(8, lexical, lexical, 0.4),
                "results": [],
            }
        ],
    }


def _questions() -> list[dict]:
    return [_result(f"q{index}") for index in range(18)]


def _baseline(results: list[dict] | None = None, mrr: float = 0.40) -> dict:
    rows = _questions() if results is None else results
    return {
        "scope": ["."],
        "embed": {"unique": 928},
        "files": [_dev(rows, _summary(len(rows), len(rows), len(rows), mrr))],
    }


def _current(
    results: list[dict] | None = None,
    *,
    hit5: int = 18,
    mrr: float = 1.0,
    structural_passed: int = 8,
    structural_cases: int = 8,
    unique: int = 1225,
    scope: list[str] | None = None,
    include_dev: bool = True,
) -> dict:
    rows = _questions() if results is None else results
    files = [_structural(structural_passed, structural_cases)]
    if include_dev:
        files.append(_dev(rows, _summary(18, 18, hit5, mrr)))
    return {
        "scope": ["."] if scope is None else scope,
        "embed": {"unique": unique},
        "files": files,
    }


def _write(directory: Path, name: str, payload: dict) -> str:
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_z1_go(tmp_path, capsys):
    baseline = _baseline()
    current = _current()
    heldout = _heldout(6, 6)
    rows = evaluate(baseline, current, heldout)

    assert rows
    assert all(ok for _name, ok, _detail in rows)
    code = main(
        [
            "--baseline",
            _write(tmp_path, "baseline.json", baseline),
            "--current",
            _write(tmp_path, "current.json", current),
            "--heldout",
            _write(tmp_path, "heldout.json", heldout),
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert output.splitlines()[-1] == "GATE: GO"
    assert all(line.startswith("PASS  ") for line in output.splitlines() if line[:4] in {"PASS", "FAIL"})


def test_z2_each_rule_failing_alone(tmp_path, capsys):
    baseline = _baseline()
    lost = _questions()
    lost[0] = _result("q0", passed=False, rank=None)
    cases = [
        _current(hit5=14),
        _current(mrr=0.47),
        _current(lost),
        _current(structural_passed=7),
        _current(unique=1501),
        _current(scope=["kernel"]),
    ]
    for index, current in enumerate(cases):
        rows = evaluate(baseline, current)
        assert sum(not ok for _name, ok, _detail in rows) == 1
        code = main(
            [
                "--baseline",
                _write(tmp_path, f"base-{index}.json", baseline),
                "--current",
                _write(tmp_path, f"current-{index}.json", current),
            ]
        )
        output = capsys.readouterr().out
        assert code == 1
        assert output.splitlines()[-1] == "DEV GATE: NO-GO"


def test_z3_matching_by_question_text():
    baseline = _baseline(
        [_result("kept"), _result("dropped")],
        mrr=0.40,
    )
    # Keep the dev summary at the GO thresholds; only the result rows move.
    failed = _current([_result("dropped", passed=False, rank=None), _result("kept")])
    reordered = _current([_result("dropped"), _result("kept")])

    failed_rows = {name: (ok, detail) for name, ok, detail in evaluate(baseline, failed)}
    passed_rows = {name: (ok, detail) for name, ok, detail in evaluate(baseline, reordered)}

    assert failed_rows["no dev case lost"] == (False, "1 lost")
    assert passed_rows["no dev case lost"] == (True, "0 lost")


def test_z4_dev_gate_without_heldout(tmp_path, capsys):
    code = main(
        [
            "--baseline",
            _write(tmp_path, "baseline.json", _baseline()),
            "--current",
            _write(tmp_path, "current.json", _current()),
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "DEV GATE: GO (held-out not run)" in output.splitlines()


@pytest.mark.parametrize(("hit5", "lexical"), [(5, 5), (6, 7)])
def test_z5_heldout_below_threshold_is_no_go(tmp_path, capsys, hit5, lexical):
    baseline = _baseline()
    current = _current()
    heldout = _heldout(hit5, lexical)
    rows = {name: ok for name, ok, _detail in evaluate(baseline, current, heldout)}

    assert rows["held-out hit@5"] is False
    assert all(ok for name, ok in rows.items() if name != "held-out hit@5")
    code = main(
        [
            "--baseline",
            _write(tmp_path, "baseline.json", baseline),
            "--current",
            _write(tmp_path, "current.json", current),
            "--heldout",
            _write(tmp_path, "heldout.json", heldout),
        ]
    )
    output = capsys.readouterr().out

    assert code == 1
    assert output.splitlines()[-1] == "GATE: NO-GO"


def test_z6_tag_table_lines_and_missing_entry(tmp_path, capsys):
    tags = {"alpha": "code", "beta": "kconfig"}
    baseline = _baseline(
        [_result("alpha", rank=2), _result("beta", rank=1)],
        mrr=0.75,
    )
    current = _current(
        [_result("alpha", passed=False, rank=None), _result("beta", rank=1)],
        hit5=1,
        mrr=0.5,
    )

    assert tag_table(baseline, current, tags) == [
        "code       baseline 1/1 MRR 0.50     current 0/1 MRR 0.00",
        "kconfig    baseline 1/1 MRR 1.00     current 1/1 MRR 1.00",
    ]

    with pytest.raises(SystemExit, match="report has no results for zephyr_semantic_dev.json"):
        main(
            [
                "--baseline",
                _write(tmp_path, "baseline.json", baseline),
                "--current",
                _write(tmp_path, "current.json", _current(include_dev=False)),
            ]
        )
    capsys.readouterr()
