"""Phase 6 stage gates for specs 31a and 32; thresholds are fixed in the spec."""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.zephyr_gate import CASES_DIR, DEV, MAX_UNIQUE, STRUCTURAL, _entry

DEV2, HELDOUT2 = "zephyr_semantic_dev2.json", "zephyr_semantic_heldout2.json"


@dataclass(frozen=True)
class Stage:
    dev_hits: int
    dev_gain: float
    dev2_hits: int | None = None
    dev2_gain: float | None = None
    heldout_hits: int | None = None


STAGES = {
    "31a": Stage(dev_hits=17, dev_gain=0.009),
    "32": Stage(dev_hits=17, dev_gain=0.03, dev2_hits=9, dev2_gain=0.05, heldout_hits=6),
}


def _lost(base: dict, cur: dict) -> list[str]:
    was = {r["question"] for r in base["results"] if r["passed"]}
    return [r["question"] for r in cur["results"] if r["question"] in was and not r["passed"]]


def _tag_hits(entry: dict, tags: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in entry["results"]:
        tag = tags.get(r["question"], "")
        out[tag] = out.get(tag, 0) + bool(r["passed"])
    return out


def _suite(rows: list, label: str, base: dict, cur: dict, hits: int, gain: float) -> None:
    s = cur["summary"]
    need = base["summary"]["mrr"] + gain
    rows.append((f"{label} hit@5", s["hit5"] >= hits, f"{s['hit5']}/{s['cases']} (need {hits})"))
    rows.append((f"{label} MRR", s["mrr"] >= need, f"{s['mrr']:.4f} (need {need:.4f})"))
    lost = _lost(base, cur)
    rows.append((f"no {label} case lost", not lost, f"{len(lost)} lost"))


def evaluate(stage: Stage, baseline: dict, current: dict, heldout: dict | None = None,
             heldout_baseline: dict | None = None,
             tags: dict[str, str] | None = None) -> list[tuple[str, bool, str]]:
    rows: list[tuple[str, bool, str]] = []
    _suite(rows, "dev", _entry(baseline, DEV), _entry(current, DEV), stage.dev_hits, stage.dev_gain)
    if stage.dev2_hits is not None:
        base2, cur2 = _entry(baseline, DEV2), _entry(current, DEV2)
        _suite(rows, "dev2", base2, cur2, stage.dev2_hits, stage.dev2_gain)
        was, now = _tag_hits(base2, tags or {}), _tag_hits(cur2, tags or {})
        worse = sorted(t for t in was if now.get(t, 0) < was[t])
        rows.append(("no dev2 tag worse", not worse, f"worse: {worse or 'none'}"))
    s = _entry(current, STRUCTURAL)["summary"]
    rows.append(("structural", s["passed"] == s["cases"], f"{s['passed']}/{s['cases']}"))
    ok = current["embed"]["unique"] <= MAX_UNIQUE and current["scope"] == ["."]
    rows.append(("check size", ok, f"unique {current['embed']['unique']}, scope {current['scope']}"))
    if stage.heldout_hits is not None and heldout is not None and heldout_baseline is not None:
        now, was = _entry(heldout, HELDOUT2), _entry(heldout_baseline, HELDOUT2)
        h, lex, b = now["summary"]["hit5"], now["lexical_summary"]["hit5"], was["summary"]["hit5"]
        ok = h >= stage.heldout_hits and h >= lex and h >= b
        rows.append(("held-out2 hit@5", ok, f"hybrid {h}, lexical {lex}, before-change {b} (need >= {stage.heldout_hits}, >= lexical, >= before-change)"))
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--heldout")
    parser.add_argument("--heldout-baseline")
    args = parser.parse_args(argv)
    if bool(args.heldout) != bool(args.heldout_baseline):
        parser.error("--heldout and --heldout-baseline go together")
    load = lambda path: json.loads(Path(path).read_text())  # noqa: E731
    stage = STAGES[args.stage]
    heldout = load(args.heldout) if args.heldout else None
    base_held = load(args.heldout_baseline) if args.heldout_baseline else None
    tags = {c["question"]: c["tag"] for c in load(CASES_DIR / DEV2)} if stage.dev2_hits else None
    rows = evaluate(stage, load(args.baseline), load(args.current), heldout, base_held, tags)
    for name, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    go = all(ok for _, ok, _ in rows)
    full = stage.heldout_hits is None or heldout is not None
    label = f"GATE {args.stage}" if full else f"DEV GATE {args.stage}"
    note = "" if full or not go else " (held-out2 not run yet)"
    if args.stage == "31a":
        note = " (held-out v1 is consumed and is not re-run)" if go else ""
    print(f"{label}: {'GO' if go else 'NO-GO'}{note}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
