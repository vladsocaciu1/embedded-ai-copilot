import json

import pytest

from scripts.zephyr_stage_gate import STAGES, evaluate, main


def _res(i: int, passed: bool, rank: int | None = 1) -> dict:
    return {"question": f"q{i}", "passed": passed, "rank": rank}


def _entry(name: str, passes: list[bool], mrr: float, lex: int = 0) -> dict:
    hits = sum(passes)
    summary = {"cases": len(passes), "passed": hits, "hit5": hits, "mrr": mrr}
    return {
        "cases_file": name,
        "summary": summary,
        "lexical_summary": {**summary, "hit5": lex},
        "results": [_res(i, p) for i, p in enumerate(passes)],
    }


def _report(dev, dev_mrr, structural=8, unique=1250):
    files = [_entry("zephyr_eval_cases.json", [True] * structural + [False] * (8 - structural), 1.0)]
    files.append(_entry("zephyr_semantic_dev.json", dev, dev_mrr))
    return {"scope": ["."], "embed": {"unique": unique}, "files": files}


def _verdict(rows):
    return {name: ok for name, ok, _ in rows}


BASE_DEV = [True] * 16 + [False] * 2


def test_g1_stage_31a_go():
    rows = evaluate(STAGES["31a"], _report(BASE_DEV, 0.5613), _report([True] * 17 + [False], 0.5741))
    assert all(ok for _, ok, _ in rows)


@pytest.mark.parametrize(
    "dev, mrr, name",
    [
        (BASE_DEV, 0.5741, "dev hit@5"),
        ([True] * 17 + [False], 0.5650, "dev MRR"),
        ([False] + [True] * 16 + [True], 0.5741, "no dev case lost"),
    ],
)
def test_g2_stage_31a_no_go(dev, mrr, name):
    rows = evaluate(STAGES["31a"], _report(BASE_DEV, 0.5613), _report(dev, mrr))
    assert _verdict(rows)[name] is False


def test_g3_structural_and_size_fail():
    base = _report(BASE_DEV, 0.5613)
    cur = _report([True] * 17 + [False], 0.5741, structural=7, unique=1501)
    got = _verdict(evaluate(STAGES["31a"], base, cur))
    assert got["structural"] is False and got["check size"] is False


def test_g6_cli_exit_codes(tmp_path, capsys):
    base, good = tmp_path / "b.json", tmp_path / "c.json"
    base.write_text(json.dumps(_report(BASE_DEV, 0.5613)))
    good.write_text(json.dumps(_report([True] * 17 + [False], 0.5741)))
    assert main(["--stage", "31a", "--baseline", str(base), "--current", str(good)]) == 0
    assert "GATE 31a: GO" in capsys.readouterr().out
    assert main(["--stage", "31a", "--baseline", str(good), "--current", str(base)]) == 1
    assert "NO-GO" in capsys.readouterr().out
