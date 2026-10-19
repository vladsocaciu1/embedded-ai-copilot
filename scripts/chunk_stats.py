import math
import sys
import time
from pathlib import Path

# This file lives in scripts/, so that directory is sys.path[0]. The repo root
# has to be importable for `uv run python scripts/chunk_stats.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunker import chunk_index, unique_texts
from app.code_index import build_index
from app.config import DEFAULT_EXCLUDE_DIRS
from app.project_loader import load_project_files

# phase-5-plan §4 measurements
LOCAL_CHUNKS_PER_SEC = 16
GEMINI_TEXTS_PER_DAY = 1000

_USAGE = "usage: python scripts/chunk_stats.py <project_dir>"


def main(argv: list[str]) -> None:
    if not argv:
        print(_USAGE)
        return

    root = Path(argv[0])
    files = load_project_files(root, DEFAULT_EXCLUDE_DIRS)
    index = build_index(files)
    started = time.perf_counter()
    chunks = chunk_index(index, files)
    chunk_ms = round((time.perf_counter() - started) * 1000)

    parsed = sum(file_symbols.grammar is not None for file_symbols in index.files)
    file_chunks = sum(chunk.kind == "file" for chunk in chunks)
    function_chunks = sum(chunk.kind == "function" for chunk in chunks)
    texts = unique_texts(chunks)
    unique = len(texts)
    chars = sum(len(text) for text in texts.values())
    est_tokens = chars // 4
    minutes = math.ceil(unique / LOCAL_CHUNKS_PER_SEC / 60)
    days = math.ceil(unique / GEMINI_TEXTS_PER_DAY)

    print(f"files {len(files)} parsed {parsed}")
    print(
        f"chunks {len(chunks)} (file {file_chunks}, function {function_chunks}) "
        f"unique {unique}"
    )
    print(f"chars {chars} est_tokens {est_tokens}")
    print(f"chunk_ms {chunk_ms}")
    print(f"local embed at {LOCAL_CHUNKS_PER_SEC} chunks/s: ~{minutes} min")
    print(f"gemini free tier at {GEMINI_TEXTS_PER_DAY} texts/day: ~{days} days")


if __name__ == "__main__":
    main(sys.argv[1:])
