"""Phase 6 stage gates; the thresholds of each stage are fixed in its spec."""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.zephyr_gate import DEV, MAX_UNIQUE, STRUCTURAL, _entry


@dataclass(frozen=True)
class Stage:
    dev_hits: int
    dev_gain: float


STAGES = {"31a": Stage(dev_hits=17, dev_gain=0.009)}


def _lost(base: dict, cur: dict) -> list[str]:
    was = {r["question"] for r in base["results"] if r["passed"]}
    return [r["question"] for r in cur["results"] if r["question"] in was and not r["passed"]]


def _suite(rows: list, label: str, base: dict, cur: dict, hits: int, gain: float) -> None:
    s = cur["summary"]
    need = base["summary"]["mrr"] + gain
    rows.append((f"{label} hit@5", s["hit5"] >= hits, f"{s['hit5']}/{s['cases']} (need {hits})"))
    rows.append((f"{label} MRR", s["mrr"] >= need, f"{s['mrr']:.4f} (need {need:.4f})"))
    lost = _lost(base, cur)
    rows.append((f"no {label} case lost", not lost, f"{len(lost)} lost"))


def evaluate(stage: Stage, baseline: dict, current: dict) -> list[tuple[str, bool, str]]:
    rows: list[tuple[str, bool, str]] = []
    _suite(rows, "dev", _entry(baseline, DEV), _entry(current, DEV), stage.dev_hits, stage.dev_gain)
    s = _entry(current, STRUCTURAL)["summary"]
    rows.append(("structural", s["passed"] == s["cases"], f"{s['passed']}/{s['cases']}"))
    ok = current["embed"]["unique"] <= MAX_UNIQUE and current["scope"] == ["."]
    rows.append(("check size", ok, f"unique {current['embed']['unique']}, scope {current['scope']}"))
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    args = parser.parse_args(argv)
    load = lambda path: json.loads(Path(path).read_text())  # noqa: E731
    rows = evaluate(STAGES[args.stage], load(args.baseline), load(args.current))
    for name, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    go = all(ok for _, ok, _ in rows)
    note = " (held-out v1 is consumed and is not re-run)" if go else ""
    print(f"GATE {args.stage}: {'GO' if go else 'NO-GO'}{note}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
