import json
import sys
import time
from pathlib import Path

# This file lives in scripts/, so that directory is sys.path[0]. The repo root
# has to be importable for `uv run python scripts/eval_selection.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.code_graph import build_graph
from app.code_index import build_index
from app.config import DEFAULT_EXCLUDE_DIRS
from app.context_selector import rank_files
from app.project_loader import load_project_files


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print(
            "usage: python scripts/eval_selection.py <project_dir> [cases_json]",
            file=sys.stderr,
        )
        raise SystemExit(0)
    root = Path(sys.argv[1])
    cases_path = (
        Path(sys.argv[2])
        if len(sys.argv) == 3
        else Path(__file__).resolve().parent / "cube_eval_cases.json"
    )
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    files = load_project_files(root, DEFAULT_EXCLUDE_DIRS)
    started = time.perf_counter()
    index = build_index(files)
    graph = build_graph(index)
    build_ms = (time.perf_counter() - started) * 1000
    print(f"index+graph {len(files)} files {build_ms:.0f} ms")

    passed = 0
    for case in cases:
        question = case["question"]
        expected = case["expected"]
        k = case["k"]
        started = time.perf_counter()
        ranked = [path.as_posix() for path in rank_files(question, index, graph)]
        elapsed_ms = (time.perf_counter() - started) * 1000
        ranks = []
        ok = True
        for path in expected:
            if path in ranked:
                rank = ranked.index(path) + 1
            else:
                rank = None
            ranks.append(rank)
            if rank is None or rank > k:
                ok = False
        if ok:
            passed += 1
        print(
            f"{'PASS' if ok else 'FAIL'} k={k} ranks={ranks} {elapsed_ms:.0f} ms {question}"
        )
    print(f"{passed}/{len(cases)} passed")


if __name__ == "__main__":
    main()
