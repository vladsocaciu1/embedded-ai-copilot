import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

# This file lives in scripts/, so that directory is sys.path[0]. The repo root
# has to be importable for `uv run python scripts/embed_check.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunker import chunk_index, unique_texts
from app.code_index import build_index
from app.config import DEFAULT_EXCLUDE_DIRS, load_embedding_settings
from app.embedder import Embedder
from app.models import Chunk
from app.project_loader import load_project_files

_USAGE = (
    "usage: embed_check.py <project_dir> --scope DIR "
    "[--scope DIR ...] [--model NAME] [--max-chunks N]"
)


def select_scope(chunks: Sequence[Chunk], scopes: Sequence[str]) -> tuple[Chunk, ...]:
    scope_parts = [(scope, Path(scope).parts) for scope in scopes]
    for scope, parts in scope_parts:
        if not any(chunk.path.parts[: len(parts)] == parts for chunk in chunks):
            raise ValueError(f"scope {scope} matches no parsed file")
    selected: list[Chunk] = []
    for chunk in chunks:
        for _scope, parts in scope_parts:
            if chunk.path.parts[: len(parts)] == parts:
                selected.append(chunk)
                break
    return tuple(selected)


def _model_files_mb(models_dir: Path) -> str:
    total = 0
    if models_dir.exists():
        for dirpath, _dirnames, filenames in os.walk(models_dir, followlinks=False):
            for name in filenames:
                path = Path(dirpath) / name
                if path.is_symlink():
                    continue
                if path.is_file():
                    total += path.stat().st_size
    return f"{total / 1_000_000:.1f}"


def _peak_rss_mb() -> str:
    try:
        import resource
    except ImportError:
        return "n/a"
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return str(rss // (1024 * 1024))
    return str(rss // 1024)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embed_check.py")
    parser.add_argument("project_dir")
    parser.add_argument("--scope", action="append", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-chunks", type=int, default=5000)
    return parser


def main(argv: Sequence[str] | None = None, embedder=None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 0:
        print(_USAGE)
        return

    args = _parser().parse_args(list(argv))
    settings = load_embedding_settings()
    model_name = settings.model if args.model is None else args.model
    models_dir = settings.cache_dir / "models"

    files = load_project_files(Path(args.project_dir), DEFAULT_EXCLUDE_DIRS)
    index = build_index(files)
    chunks = chunk_index(index, files)
    try:
        scoped = select_scope(chunks, args.scope)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc

    texts = unique_texts(scoped)
    unique = len(texts)
    parsed = sum(file_symbols.grammar is not None for file_symbols in index.files)
    scoped_files = len({chunk.path for chunk in scoped})

    print(f"files {len(files)} parsed {parsed}")
    print(
        f"scope {','.join(args.scope)} files {scoped_files} "
        f"chunks {len(scoped)} unique {unique}"
    )
    if unique > args.max_chunks:
        print(
            f"refusing to embed {unique} unique chunks (max {args.max_chunks}); narrow --scope"
        )
        raise SystemExit(2)

    if embedder is None:
        embedder = Embedder(model=model_name, cache_dir=settings.cache_dir)

    state = "present" if models_dir.exists() else "absent"
    print(f"model {model_name} cache {models_dir} ({state} before load)")

    started = time.perf_counter()
    embedder.embed_query("warm up")
    load_s = time.perf_counter() - started
    print(f"model load {load_s:.1f} s")
    print(f"model files {_model_files_mb(models_dir)} MB")

    started = time.perf_counter()
    embedder.embed_documents(list(texts.values()))
    embed_s = time.perf_counter() - started
    rate = unique / embed_s if embed_s else 0.0
    print(f"embed {unique} chunks in {embed_s:.1f} s ({rate:.1f} chunks/s)")
    print(f"peak_rss_mb {_peak_rss_mb()}")


if __name__ == "__main__":
    main()
