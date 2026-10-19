import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# This file lives in scripts/, so that directory is sys.path[0]. The repo root
# has to be importable for `uv run python scripts/eval_selection.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.code_graph import build_graph
from app.code_index import build_index
from app.config import DEFAULT_EXCLUDE_DIRS
from app.context_selector import rank_files
from app.project_loader import load_project_files

_USAGE = (
    "usage: python scripts/eval_selection.py <project_dir> [cases_json ...] [--json OUT]"
)


@dataclass(frozen=True)
class CaseResult:
    question: str
    expected: tuple[str, ...]
    k: int
    ranks: tuple[int | None, ...]
    rank: int | None
    passed: bool


@dataclass(frozen=True)
class Summary:
    cases: int
    passed: int
    hit5: int
    mrr: float


def case_rank(ranked: Sequence[str], expected: Sequence[str]) -> int | None:
    positions: dict[str, int] = {}
    for index, path in enumerate(ranked, start=1):
        positions.setdefault(path, index)
    worst: int | None = None
    for path in expected:
        rank = positions.get(path)
        if rank is None:
            return None
        if worst is None or rank > worst:
            worst = rank
    return worst


def summarize(results: Sequence[CaseResult]) -> Summary:
    total = len(results)
    if total == 0:
        return Summary(cases=0, passed=0, hit5=0, mrr=0.0)
    passed = 0
    hit5 = 0
    reciprocal = 0.0
    for result in results:
        if result.rank is None:
            continue
        reciprocal += 1 / result.rank
        if result.rank <= result.k:
            passed += 1
        if result.rank <= 5:
            hit5 += 1
    return Summary(cases=total, passed=passed, hit5=hit5, mrr=reciprocal / total)


def _default_cases_path() -> Path:
    return Path(__file__).resolve().parent / "cube_eval_cases.json"


def _evaluate_cases(cases_path: Path, index, graph) -> tuple[list[CaseResult], Summary]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    print(f"== {cases_path.name} ==")
    results: list[CaseResult] = []
    for case in cases:
        question = case["question"]
        expected = case["expected"]
        k = case["k"]
        started = time.perf_counter()
        ranked = [path.as_posix() for path in rank_files(question, index, graph)]
        elapsed_ms = (time.perf_counter() - started) * 1000
        ranks: list[int | None] = []
        for path in expected:
            if path in ranked:
                ranks.append(ranked.index(path) + 1)
            else:
                ranks.append(None)
        rank = case_rank(ranked, expected)
        ok = rank is not None and rank <= k
        print(
            f"{'PASS' if ok else 'FAIL'} k={k} ranks={ranks} {elapsed_ms:.0f} ms {question}"
        )
        results.append(
            CaseResult(
                question=question,
                expected=tuple(expected),
                k=k,
                ranks=tuple(ranks),
                rank=rank,
                passed=ok,
            )
        )
    summary = summarize(results)
    print(f"{summary.passed}/{summary.cases} passed")
    print(
        f"summary: pass {summary.passed}/{summary.cases}, "
        f"hit@5 {summary.hit5}/{summary.cases}, MRR {summary.mrr:.2f}"
    )
    return results, summary


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 0:
        print(_USAGE, file=sys.stderr)
        raise SystemExit(0)

    parser = argparse.ArgumentParser(prog="python scripts/eval_selection.py", usage=_USAGE)
    parser.add_argument("project_dir")
    parser.add_argument("cases_json", nargs="*")
    parser.add_argument("--json", dest="json_out", metavar="OUT")
    args = parser.parse_args(list(argv))

    root = Path(args.project_dir)
    cases_paths = (
        [Path(path) for path in args.cases_json]
        if args.cases_json
        else [_default_cases_path()]
    )
    files = load_project_files(root, DEFAULT_EXCLUDE_DIRS)
    started = time.perf_counter()
    index = build_index(files)
    graph = build_graph(index)
    build_ms = (time.perf_counter() - started) * 1000
    print(f"index+graph {len(files)} files {build_ms:.0f} ms")

    reports = []
    for cases_path in cases_paths:
        results, summary = _evaluate_cases(cases_path, index, graph)
        reports.append(
            {
                "cases_file": cases_path.name,
                "summary": asdict(summary),
                "results": [asdict(result) for result in results],
            }
        )

    if args.json_out:
        payload = {"project_files": len(files), "files": reports}
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
