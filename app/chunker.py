import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

from app.models import (
    Chunk,
    FileSymbols,
    ProjectFile,
    ProjectIndex,
    Symbol,
    SymbolKind,
)

CHUNK_FORMAT = "chunk-v1"
FILE_COMMENT_CAP = 600
LEAD_COMMENT_CAP = 400
BODY_CAP = 1500
CARD_SYMBOL_KINDS = (
    SymbolKind.FUNCTION,
    SymbolKind.TYPEDEF,
    SymbolKind.STRUCT,
    SymbolKind.ENUM,
)
CARD_SYMBOL_LIMIT = 40
LEAD_COMMENT_MAX_LINES = 25
TAIL_PARTS = 4

_NOISE = re.compile(
    r"copyright|license|all rights reserved|as-is|@attention|@author|^[\s*/=\-]*$",
    re.I,
)
_FILE_COMMENT = re.compile(r"\s*/\*(.*?)\*/", re.S)


def clean(block: str) -> str:
    kept: list[str] = []
    for line in block.splitlines():
        stripped = line.strip(" */\t")
        if not stripped or _NOISE.search(stripped):
            continue
        kept.append(stripped)
    return " ".join(kept)


def tail(path: Path) -> str:
    return "/".join(path.parts[-TAIL_PARTS:])


def _key(text: str) -> str:
    return hashlib.sha256((CHUNK_FORMAT + "\n" + text).encode("utf-8")).hexdigest()


def _lead(lines: list[str], start_line: int) -> str:
    collected: list[str] = []
    index = start_line - 2
    while index >= 0 and len(collected) < LEAD_COMMENT_MAX_LINES:
        stripped = lines[index].strip()
        qualifies = stripped.startswith(("/*", "*", "//")) or stripped.endswith("*/")
        if not qualifies:
            break
        collected.append(lines[index])
        index -= 1
    if not collected:
        return ""
    collected.reverse()
    return clean("\n".join(collected))[:LEAD_COMMENT_CAP]


def _body(lines: list[str], start_line: int, end_line: int) -> str:
    start = max(start_line - 1, 0)
    return "\n".join(lines[start:end_line])[:BODY_CAP]


def _card(fs: FileSymbols, content: str) -> Chunk:
    match = _FILE_COMMENT.match(content)
    comment = clean(match.group(1))[:FILE_COMMENT_CAP] if match else ""
    names = [
        symbol.name
        for symbol in fs.symbols
        if symbol.kind in CARD_SYMBOL_KINDS
    ][:CARD_SYMBOL_LIMIT]
    text = f"file {tail(fs.path)}\n{comment}\nsymbols: {' '.join(names)}"
    line_count = len(content.splitlines())
    return Chunk(
        path=fs.path,
        kind="file",
        name="",
        start_line=1,
        end_line=max(1, line_count),
        text=text,
        key=_key(text),
    )


def _function_chunk(fs: FileSymbols, content: str, symbol: Symbol) -> Chunk:
    lines = content.splitlines()
    lead = _lead(lines, symbol.start_line)
    body = _body(lines, symbol.start_line, symbol.end_line)
    text = f"function {symbol.name} in {tail(fs.path)}\n{lead}\n{body}"
    return Chunk(
        path=fs.path,
        kind="function",
        name=symbol.name,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        text=text,
        key=_key(text),
    )


def chunk_file(fs: FileSymbols, content: str) -> tuple[Chunk, ...]:
    if fs.grammar is None:
        return ()
    chunks = [_card(fs, content)]
    for symbol in fs.symbols:
        if symbol.kind is SymbolKind.FUNCTION:
            chunks.append(_function_chunk(fs, content, symbol))
    return tuple(chunks)


def chunk_index(
    index: ProjectIndex, files: Sequence[ProjectFile]
) -> tuple[Chunk, ...]:
    contents = {project_file.path: project_file.content for project_file in files}
    chunks: list[Chunk] = []
    for file_symbols in index.files:
        if file_symbols.grammar is None:
            continue
        content = contents.get(file_symbols.path)
        if content is None:
            raise ValueError(f"no ProjectFile for {file_symbols.path}")
        chunks.extend(chunk_file(file_symbols, content))
    return tuple(chunks)


def unique_texts(chunks: Sequence[Chunk]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for chunk in chunks:
        if chunk.key not in texts:
            texts[chunk.key] = chunk.text
    return texts
