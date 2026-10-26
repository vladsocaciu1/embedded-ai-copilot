import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# This file lives in scripts/, so that directory is sys.path[0]. The repo root
# has to be importable for `uv run python scripts/vector_check.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.code_index import build_index
from app.chunker import chunk_index, unique_texts
from app.config import DEFAULT_EXCLUDE_DIRS, load_embedding_settings
from app.embedder import Embedder
from app.project_loader import load_project_files
from app.vector_index import VectorIndex
from scripts.embed_check import _peak_rss_mb, select_scope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vector_check.py")
    parser.add_argument("project_dir")
    parser.add_argument("--scope", action="append", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-chunks", type=int, default=3000)
    return parser


def _index_size_mb(directory: Path) -> str:
    total = 0
    for name in ("vectors.npy", "keys.json", "meta.json"):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            continue
        total += path.stat().st_size
    return f"{total / 1_000_000:.1f}"


def main(argv: Sequence[str] | None = None, embedder=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        args = _parser().parse_args(list(argv))
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 2

    settings = load_embedding_settings()
    cache_dir = settings.cache_dir if args.cache_dir is None else Path(args.cache_dir)
    model_name = settings.model if args.model is None else args.model

    files = load_project_files(Path(args.project_dir), DEFAULT_EXCLUDE_DIRS)
    index = build_index(files)
    chunks = chunk_index(index, files)
    try:
        scoped = select_scope(chunks, args.scope)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    unique = len(unique_texts(scoped))
    if unique > args.max_chunks:
        print(
            f"refusing to embed {unique} unique chunks (max {args.max_chunks}); narrow --scope"
        )
        return 2

    if embedder is None:
        embedder = Embedder(model=model_name, cache_dir=cache_dir)

    store = VectorIndex(cache_dir, model_name)
    result = store.ensure(scoped, embedder)
    scoped_files = len({chunk.path for chunk in scoped})
    compacted = "yes" if result.compacted else "no"
    write_failed = "yes" if result.write_failed else "no"
    print(
        f"scope {','.join(args.scope)} files {scoped_files} chunks unique {unique}"
    )
    print(
        f"ensure live {result.live} reused {result.reused} embedded {result.embedded} "
        f"stale {result.stale} compacted {compacted} write_failed {write_failed} "
        f"in {result.seconds:.1f} s"
    )
    empty = store.lookup(())
    dims = int(empty.matrix.shape[1]) if empty is not None else 0
    print(
        f"index {len(store)} rows dims {dims} dir {store.directory} "
        f"size {_index_size_mb(store.directory)} MB"
    )
    found = store.lookup(scoped)
    if found is None:
        print(f"lookup missing {store.missing(scoped)}")
    else:
        print("lookup ok")
    print(f"rss {_peak_rss_mb()} MB peak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
