import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# This file lives in scripts/, so that directory is sys.path[0]. The repo root
# has to be importable for `uv run python scripts/eval_selection.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunker import chunk_index, unique_texts
from app.code_graph import build_graph
from app.code_index import build_index
from app.config import DEFAULT_EXCLUDE_DIRS, load_embedding_settings
from app.context_selector import fuse_rankings, rank_files, rank_files_detailed
from app.embedder import Embedder
from app.project_loader import load_project_files
from app.semantic_search import embed_chunks, semantic_rank
from scripts.embed_check import select_scope

_USAGE = (
    "usage: python scripts/eval_selection.py <project_dir> [cases_json ...] "
    "[--json OUT] [--ranker lexical|hybrid] [--scope DIR ...] "
    "[--model NAME] [--max-chunks N] [--gate]\n"
    "       python scripts/eval_selection.py --gate-from A.json [B.json ...]"
)

_GATE_ROLES = (
    ("cube_eval_cases.json", "structural", 8),
    ("cube_semantic_dev.json", "dev", 12),
    ("cube_semantic_heldout.json", "held-out", 8),
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


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: tuple[GateCheck, ...]


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


def evaluate_gate(reports: Mapping[str, Mapping]) -> GateResult:
    problems: list[str] = []
    entries: dict[str, Mapping] = {}
    for filename, role, count in _GATE_ROLES:
        entry = reports.get(filename)
        if entry is None:
            problems.append(f"missing {role} ({filename})")
            continue
        hybrid_n = len(entry.get("results", ()))
        lexical_n = len(entry.get("lexical_results", ()))
        hybrid_cases = entry.get("summary", {}).get("cases")
        lexical_cases = entry.get("lexical_summary", {}).get("cases")
        if (
            hybrid_n != count
            or lexical_n != count
            or hybrid_cases != count
            or lexical_cases != count
        ):
            problems.append(
                f"{filename} has {hybrid_n} hybrid cases and "
                f"{lexical_n} lexical cases, expected {count}"
            )
            continue
        entries[filename] = entry
    if problems:
        return GateResult(
            passed=False,
            checks=(GateCheck("cases", False, "; ".join(problems)),),
        )

    dev = entries["cube_semantic_dev.json"]["summary"]
    held = entries["cube_semantic_heldout.json"]
    held_hybrid = held["summary"]
    held_lexical = held["lexical_summary"]
    structural = entries["cube_eval_cases.json"]
    structural_summary = structural["summary"]

    dev_hit = dev["hit5"]
    dev_mrr = float(dev["mrr"])
    held_hit_hybrid = held_hybrid["hit5"]
    held_hit_lexical = held_lexical["hit5"]
    held_mrr_hybrid = float(held_hybrid["mrr"])
    held_mrr_lexical = float(held_lexical["mrr"])
    structural_passed = structural_summary["passed"]

    rank_problems: list[str] = []
    worst_delta: int | None = None
    for index, hybrid in enumerate(structural["results"]):
        lexical = structural["lexical_results"][index]
        hybrid_rank = hybrid.get("rank")
        lexical_rank = lexical.get("rank")
        if (
            hybrid_rank is None
            or lexical_rank is None
            or hybrid_rank > lexical_rank + 1
        ):
            rank_problems.append(
                f"{hybrid.get('question')} hybrid {hybrid_rank} lexical {lexical_rank}"
            )
            continue
        delta = hybrid_rank - lexical_rank
        if worst_delta is None or delta > worst_delta:
            worst_delta = delta

    checks = (
        GateCheck("dev_hit5", dev_hit >= 8, f"{dev_hit}/12"),
        GateCheck("dev_mrr", dev_mrr + 1e-9 >= 0.55, f"{dev_mrr:.6f}"),
        GateCheck(
            "heldout_hit5",
            held_hit_hybrid >= held_hit_lexical + 1,
            f"hybrid {held_hit_hybrid} lexical {held_hit_lexical}",
        ),
        GateCheck(
            "heldout_mrr",
            held_mrr_hybrid + 1e-9 >= held_mrr_lexical + 0.05,
            f"hybrid {held_mrr_hybrid:.6f} lexical {held_mrr_lexical:.6f}",
        ),
        GateCheck(
            "structural_pass",
            structural_passed == 8,
            f"{structural_passed}/8",
        ),
        GateCheck(
            "structural_ranks",
            not rank_problems,
            "; ".join(rank_problems)
            if rank_problems
            else f"max hybrid-lexical {worst_delta}",
        ),
    )
    return GateResult(
        passed=all(check.passed for check in checks),
        checks=checks,
    )


def _default_cases_path() -> Path:
    return Path(__file__).resolve().parent / "cube_eval_cases.json"


def _result_for(
    question: str, expected: Sequence[str], k: int, ranked_paths: Sequence[str]
) -> tuple[CaseResult, list[int | None]]:
    ranks: list[int | None] = []
    for path in expected:
        if path in ranked_paths:
            ranks.append(ranked_paths.index(path) + 1)
        else:
            ranks.append(None)
    rank = case_rank(ranked_paths, expected)
    ok = rank is not None and rank <= k
    return (
        CaseResult(
            question=question,
            expected=tuple(expected),
            k=k,
            ranks=tuple(ranks),
            rank=rank,
            passed=ok,
        ),
        ranks,
    )


def _print_summary(summary: Summary, label: str | None = None) -> None:
    prefix = "summary: " if label is None else f"{label}: "
    print(
        f"{prefix}pass {summary.passed}/{summary.cases}, "
        f"hit@5 {summary.hit5}/{summary.cases}, MRR {summary.mrr:.2f}"
    )


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
        result, ranks = _result_for(question, expected, k, ranked)
        print(
            f"{'PASS' if result.passed else 'FAIL'} k={k} ranks={ranks} "
            f"{elapsed_ms:.0f} ms {question}"
        )
        results.append(result)
    summary = summarize(results)
    print(f"{summary.passed}/{summary.cases} passed")
    _print_summary(summary)
    return results, summary


def _evaluate_hybrid_cases(
    cases_path: Path, index, graph, embedded, embedder
) -> tuple[list[CaseResult], Summary, list[CaseResult], Summary]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    print(f"== {cases_path.name} ==")
    hybrid_results: list[CaseResult] = []
    lexical_results: list[CaseResult] = []
    for case in cases:
        question = case["question"]
        expected = case["expected"]
        k = case["k"]
        started = time.perf_counter()
        lex = rank_files_detailed(question, index, graph)
        hyb = fuse_rankings(lex, semantic_rank(question, embedded, embedder))
        elapsed_ms = (time.perf_counter() - started) * 1000
        lex_paths = [item.path.as_posix() for item in lex]
        hyb_paths = [item.path.as_posix() for item in hyb]
        hybrid_result, hybrid_ranks = _result_for(question, expected, k, hyb_paths)
        lexical_result, lexical_ranks = _result_for(question, expected, k, lex_paths)
        print(
            f"{'PASS' if hybrid_result.passed else 'FAIL'} k={k} "
            f"ranks={hybrid_ranks} lexical={lexical_ranks} "
            f"{elapsed_ms:.0f} ms {question}"
        )
        hybrid_results.append(hybrid_result)
        lexical_results.append(lexical_result)
    hybrid_summary = summarize(hybrid_results)
    lexical_summary = summarize(lexical_results)
    print(f"{hybrid_summary.passed}/{hybrid_summary.cases} passed")
    _print_summary(hybrid_summary)
    _print_summary(lexical_summary, label="lexical")
    return hybrid_results, hybrid_summary, lexical_results, lexical_summary


def _fmt_rank(rank: int | None) -> str:
    if rank is None:
        return "None"
    return str(rank)


def _print_gate(reports: Mapping[str, Mapping]) -> GateResult:
    result = evaluate_gate(reports)
    print("== gate ==")
    print("role lexical hybrid k question")
    for filename, role, _count in _GATE_ROLES:
        entry = reports.get(filename)
        if not isinstance(entry, Mapping):
            continue
        hybrid_results = entry.get("results", ())
        lexical_results = entry.get("lexical_results", ())
        for index, hybrid in enumerate(hybrid_results):
            lexical = lexical_results[index] if index < len(lexical_results) else {}
            print(
                f"{role} lexical={_fmt_rank(lexical.get('rank'))} "
                f"hybrid={_fmt_rank(hybrid.get('rank'))} "
                f"k={hybrid.get('k')} {hybrid.get('question')}"
            )
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"check {check.name}: {status} ({check.detail})")
    if result.passed:
        print("gate: GO")
    else:
        failed = ", ".join(check.name for check in result.checks if not check.passed)
        print(f"gate: NO-GO (failed: {failed})")
    return result


def _entry_dict(cases_path: Path, results: Sequence[CaseResult], summary: Summary) -> dict:
    return {
        "cases_file": cases_path.name,
        "summary": asdict(summary),
        "results": [asdict(result) for result in results],
    }


def _load_gate_reports(paths: Sequence[str]) -> dict[str, Mapping]:
    merged: dict[str, Mapping] = {}
    for raw in paths:
        path = Path(raw)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"cannot read {path}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        files = payload.get("files")
        if not isinstance(files, list):
            print(f"{path} has no files entries", file=sys.stderr)
            raise SystemExit(2)
        for entry in files:
            name = entry.get("cases_file")
            if name in merged:
                print(f"duplicate cases file {name}", file=sys.stderr)
                raise SystemExit(2)
            merged[name] = entry
    return merged


def _prepare_hybrid(index, files, scopes: Sequence[str], max_chunks: int):
    chunks = chunk_index(index, files)
    try:
        scoped = select_scope(chunks, scopes)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc
    texts = unique_texts(scoped)
    unique = len(texts)
    scoped_files = len({chunk.path for chunk in scoped})
    print(
        f"scope {','.join(scopes)} files {scoped_files} "
        f"chunks {len(scoped)} unique {unique}"
    )
    if unique > max_chunks:
        print(
            f"refusing to embed {unique} unique chunks (max {max_chunks}); narrow --scope"
        )
        raise SystemExit(2)
    return scoped, unique


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/eval_selection.py",
        usage=_USAGE,
    )
    parser.add_argument("project_dir", nargs="?")
    parser.add_argument("cases_json", nargs="*")
    parser.add_argument("--json", dest="json_out", metavar="OUT")
    parser.add_argument("--ranker", choices=("lexical", "hybrid"), default="lexical")
    parser.add_argument("--scope", action="append", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-chunks", type=int, default=5000)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--gate-from", nargs="+", dest="gate_from")
    return parser


def main(argv: Sequence[str] | None = None, embedder=None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 0:
        print(_USAGE, file=sys.stderr)
        raise SystemExit(0)

    parser = _parser()
    args = parser.parse_args(list(argv))
    if args.gate and args.ranker != "hybrid":
        parser.error("--gate requires --ranker hybrid")
    if args.ranker == "hybrid" and not args.scope and not args.gate_from:
        parser.error("--ranker hybrid requires --scope")
    if args.gate_from:
        if args.project_dir is not None:
            parser.error("--gate-from does not take a project directory")
        _print_gate(_load_gate_reports(args.gate_from))
        return

    if args.project_dir is None:
        parser.error("the following arguments are required: project_dir")

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

    embedded = None
    unique = None
    embed_seconds = None
    model_name = None
    if args.ranker == "hybrid":
        _scoped, unique = _prepare_hybrid(index, files, args.scope, args.max_chunks)
        if embedder is None:
            settings = load_embedding_settings()
            model_name = settings.model if args.model is None else args.model
            embedder = Embedder(model_name, cache_dir=settings.cache_dir)
        elif args.model is not None:
            model_name = args.model
        else:
            model_name = getattr(embedder, "model", "")
        started = time.perf_counter()
        embedded = embed_chunks(_scoped, embedder)
        embed_seconds = time.perf_counter() - started
        rate = unique / embed_seconds if embed_seconds else 0.0
        print(
            f"embed {unique} chunks in {embed_seconds:.1f} s "
            f"({rate:.1f} chunks/s) model {model_name}"
        )

    reports = []
    report_by_name: dict[str, dict] = {}
    for cases_path in cases_paths:
        if args.ranker == "hybrid":
            hybrid_results, hybrid_summary, lexical_results, lexical_summary = (
                _evaluate_hybrid_cases(cases_path, index, graph, embedded, embedder)
            )
            entry = _entry_dict(cases_path, hybrid_results, hybrid_summary)
            entry["lexical_summary"] = asdict(lexical_summary)
            entry["lexical_results"] = [asdict(result) for result in lexical_results]
        else:
            results, summary = _evaluate_cases(cases_path, index, graph)
            entry = _entry_dict(cases_path, results, summary)
        reports.append(entry)
        report_by_name[cases_path.name] = entry

    gate_result = None
    if args.gate:
        gate_result = _print_gate(report_by_name)

    if args.json_out:
        payload: dict = {"project_files": len(files), "files": reports}
        if args.ranker == "hybrid":
            payload = {
                "project_files": len(files),
                "ranker": "hybrid",
                "scope": list(args.scope),
                "model": model_name,
                "embed": {"unique": unique, "seconds": embed_seconds},
                "files": reports,
            }
        if gate_result is not None:
            payload["gate"] = {
                "passed": gate_result.passed,
                "checks": [asdict(check) for check in gate_result.checks],
            }
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
