import json
from pathlib import Path

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


def _report(dev, dev_mrr, dev2=None, dev2_mrr=0.5, structural=8, unique=1250, held=None):
    files = [_entry("zephyr_eval_cases.json", [True] * structural + [False] * (8 - structural), 1.0)]
    files.append(_entry("zephyr_semantic_dev.json", dev, dev_mrr))
    if dev2 is not None:
        files.append(_entry("zephyr_semantic_dev2.json", dev2, dev2_mrr))
    if held is not None:
        files.append(_entry("zephyr_semantic_heldout2.json", [True] * held[0] + [False] * (8 - held[0]), 0.5, held[1]))
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


TAGS = {f"q{i}": t for i, t in enumerate(["syscall-name"] * 5 + ["CONFIG"] * 4 + ["devicetree"])}


def test_g4_stage_32_rules():
    base2 = [True] * 7 + [False] * 2 + [True]
    base = _report([True] * 17 + [False], 0.5741, base2, 0.63)
    good = _report([True] * 17 + [False], 0.64, [True] * 8 + [False] + [True], 0.72)
    assert all(ok for _, ok, _ in evaluate(STAGES["32"], base, good, tags=TAGS))
    worse_tag = _report([True] * 17 + [False], 0.64, [True] * 4 + [False] + [True] * 4 + [False, True], 0.72)
    got = _verdict(evaluate(STAGES["32"], base, worse_tag, tags=TAGS))
    assert got["no dev2 tag worse"] is False


def test_g5_heldout2_rules():
    base = _report([True] * 17 + [False], 0.5741, [True] * 8 + [False] * 2, 0.6)
    cur = _report([True] * 17 + [False], 0.64, [True] * 9 + [False], 0.72, held=(6, 5))
    before = _report([True] * 17 + [False], 0.5741, [True] * 8 + [False] * 2, 0.6, held=(5, 5))
    got = evaluate(STAGES["32"], base, cur, cur, before, TAGS)
    assert _verdict(got)["held-out2 hit@5"] is True
    low = _report([True] * 17 + [False], 0.64, [True] * 9 + [False], 0.72, held=(5, 4))
    assert _verdict(evaluate(STAGES["32"], base, cur, low, before, TAGS))["held-out2 hit@5"] is False
    lexical_better = _report([True] * 17 + [False], 0.64, [True] * 9 + [False], 0.72, held=(6, 7))
    assert _verdict(evaluate(STAGES["32"], base, cur, lexical_better, before, TAGS))["held-out2 hit@5"] is False


def test_g6_cli_exit_codes(tmp_path, capsys):
    base, good = tmp_path / "b.json", tmp_path / "c.json"
    base.write_text(json.dumps(_report(BASE_DEV, 0.5613)))
    good.write_text(json.dumps(_report([True] * 17 + [False], 0.5741)))
    assert main(["--stage", "31a", "--baseline", str(base), "--current", str(good)]) == 0
    assert "GATE 31a: GO" in capsys.readouterr().out
    assert main(["--stage", "31a", "--baseline", str(good), "--current", str(base)]) == 1
    assert "NO-GO" in capsys.readouterr().out


def test_g7_case_files_have_the_documented_shape():
    scripts = Path(__file__).resolve().parent.parent / "scripts"

    def load(name):
        return json.loads((scripts / name).read_text(encoding="utf-8"))

    dev2, held2 = load("zephyr_semantic_dev2.json"), load("zephyr_semantic_heldout2.json")
    assert (len(dev2), len(held2)) == (10, 8)
    assert {c["tag"] for c in dev2} == {"syscall-name", "CONFIG", "devicetree"}
    assert {c["tag"] for c in held2} == {"syscall-name", "CONFIG"}
    others = [c["question"] for n in ("zephyr_semantic_dev.json", "zephyr_semantic_heldout.json") for c in load(n)]
    questions = [c["question"] for c in dev2 + held2] + others
    assert len(set(questions)) == len(questions)
    for case in dev2 + held2:
        assert case["k"] == 5 and case["expected"] and all("/" in p for p in case["expected"])
