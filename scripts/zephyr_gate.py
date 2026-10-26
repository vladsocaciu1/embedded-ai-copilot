"""Phase 6 gate: compare two eval_selection.py --json reports (spec 31)."""
import argparse
import json
import sys
from pathlib import Path

MIN_DEV_HITS = 15
MIN_MRR_GAIN = 0.08
MIN_HELDOUT_HITS = 6
MAX_UNIQUE = 1500
CASES_DIR = Path(__file__).resolve().parent
STRUCTURAL, DEV, HELDOUT = "zephyr_eval_cases.json", "zephyr_semantic_dev.json", "zephyr_semantic_heldout.json"


def _entry(report: dict, name: str) -> dict:
    for entry in report["files"]:
        if Path(entry["cases_file"]).name == name:
            return entry
    raise SystemExit(f"report has no results for {name}")


def evaluate(baseline: dict, current: dict, heldout: dict | None = None) -> list[tuple[str, bool, str]]:
    rows = []
    dev, base_dev = _entry(current, DEV), _entry(baseline, DEV)
    hits, mrr = dev["summary"]["hit5"], dev["summary"]["mrr"]
    need = base_dev["summary"]["mrr"] + MIN_MRR_GAIN
    rows.append(("dev hit@5", hits >= MIN_DEV_HITS, f"{hits}/{dev['summary']['cases']} (need {MIN_DEV_HITS})"))
    rows.append(("dev MRR", mrr >= need, f"{mrr:.4f} (need {need:.4f})"))
    was = {r["question"] for r in base_dev["results"] if r["passed"]}
    lost = [r["question"] for r in dev["results"] if r["question"] in was and not r["passed"]]
    rows.append(("no dev case lost", not lost, f"{len(lost)} lost"))
    s = _entry(current, STRUCTURAL)["summary"]
    rows.append(("structural", s["passed"] == s["cases"], f"{s['passed']}/{s['cases']}"))
    ok = current["embed"]["unique"] <= MAX_UNIQUE and current["scope"] == ["."]
    rows.append(("check size", ok, f"unique {current['embed']['unique']}, scope {current['scope']}"))
    if heldout is not None:
        entry = _entry(heldout, HELDOUT)
        h, lex = entry["summary"]["hit5"], entry["lexical_summary"]["hit5"]
        rows.append(("held-out hit@5", h >= MIN_HELDOUT_HITS and h >= lex, f"hybrid {h}, lexical {lex} (need >= {MIN_HELDOUT_HITS} and >= lexical)"))
    return rows


def tag_table(baseline: dict, current: dict, tags: dict[str, str]) -> list[str]:
    lines = []
    for tag in sorted(set(tags.values())):
        cells = []
        for report in (baseline, current):
            results = _entry(report, DEV)["results"]
            mine = [r for r in results if tags.get(r["question"]) == tag]
            passed = sum(r["passed"] for r in mine)
            mrr = sum(1 / r["rank"] if r["rank"] else 0 for r in mine) / max(1, len(mine))
            cells.append(f"{passed}/{len(mine)} MRR {mrr:.2f}")
        lines.append(f"{tag:<10} baseline {cells[0]:<16} current {cells[1]}")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--heldout")
    args = parser.parse_args(argv)
    load = lambda path: json.loads(Path(path).read_text())  # noqa: E731
    baseline, current = load(args.baseline), load(args.current)
    heldout = load(args.heldout) if args.heldout else None
    tags = {c["question"]: c.get("tag", "") for c in load(CASES_DIR / DEV)}
    rows = evaluate(baseline, current, heldout)
    print("\n".join(tag_table(baseline, current, tags)))
    for name, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    go = all(ok for _, ok, _ in rows)
    label = "GATE" if heldout else "DEV GATE"
    note = " (held-out not run)" if go and not heldout else ""
    print(f"{label}: {'GO' if go else 'NO-GO'}{note}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
