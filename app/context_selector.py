import dataclasses
import re
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from pathlib import Path

from app.code_graph import CodeGraph
from app.models import FileSymbols, ProjectFile, ProjectIndex, RankedFile, SymbolKind
from app.project_loader import SUPPORTED_EXTENSIONS
from app.text_files import text_kind

RRF_K = 60

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-/\\:]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_SEED_RE = re.compile(r"^[A-Za-z_]\w*(::[A-Za-z_]\w*)*$")
_EDGE_PUNCT = ".,;:!?()[]{}'\"`"
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "how", "does", "what", "where", "which", "who",
    "why", "when", "this", "that", "from", "into", "are", "was", "were", "its",
    "can", "not", "all", "any", "use", "used", "uses", "using", "show", "explain",
    "file", "files", "code", "project", "function", "functions", "call", "called",
    "calls", "about", "there", "their", "have", "has", "example", "examples",
    "work", "works",
})
_TERM_KINDS = frozenset({
    SymbolKind.FUNCTION,
    SymbolKind.PROTOTYPE,
    SymbolKind.TYPEDEF,
    SymbolKind.STRUCT,
    SymbolKind.UNION,
    SymbolKind.ENUM,
    SymbolKind.CLASS,
    SymbolKind.VARIABLE,
    SymbolKind.CONFIG,
    SymbolKind.SECTION,
    SymbolKind.NODE,
})


def rank_files_detailed(
    question: str, index: ProjectIndex, graph: CodeGraph
) -> tuple[RankedFile, ...]:
    tokens = _tokens(question)
    code_like = [token for token in tokens if _is_code_like(token)]
    seeds = [token for token in code_like if _SEED_RE.fullmatch(token)]
    named_tails = [
        tail for token in tokens if (tail := _named_file_tail(token)) is not None
    ]
    words = _question_words(tokens)
    word_forms = tuple(_word_forms(word) for word in words)
    by_path = {file.path: file for file in index.files}
    directory_names = {
        part.lower()
        for file in index.files
        for part in file.path.parts[:-1]
    }
    focus_dirs = {
        token.lower() for token in code_like if token.lower() in directory_names
    }

    tiers: dict[Path, int] = {}
    hits: dict[Path, int] = {}

    def add(path: Path, tier: int) -> None:
        if path not in by_path:
            return
        current = tiers.get(path)
        if current is None or tier < current:
            tiers[path] = tier
        hits[path] = hits.get(path, 0) + 1

    for file in index.files:
        parts = tuple(part.lower() for part in file.path.parts)
        for tail in named_tails:
            if _ends_with(parts, tail):
                add(file.path, 0)
    for seed in seeds:
        for ref in graph.definitions(seed):
            add(ref.path, 1)

    tier01 = sorted(
        (path for path, tier in tiers.items() if tier <= 1),
        key=lambda path: path.parts,
    )
    for path in tier01:
        for pair in graph.pair(path):
            add(pair, 2)
    for seed in seeds:
        for ref in graph.callers(seed):
            add(ref.path, 3)
        for ref in graph.callees(seed):
            for defined in graph.definitions(ref.call.callee, from_path=ref.path):
                add(defined.path, 3)
    for path in tier01:
        for resolved in graph.includes(path):
            if len(resolved.candidates) == 1:
                add(resolved.candidates[0], 4)

    focus_of: dict[Path, int] = {}
    lexical_of: dict[Path, int] = {}
    for file in index.files:
        focus_of[file.path] = _focus_count(file.path, focus_dirs)
        lexical_of[file.path] = _lexical_score(file, word_forms)

    candidates = [
        file.path
        for file in index.files
        if tiers.get(file.path, 9) < 9
        or focus_of[file.path] > 0
        or lexical_of[file.path] > 0
    ]
    if not candidates:
        return tuple(
            RankedFile(
                path=path, reason="entry", focus=0, tier=9, hits=0, lexical=0
            )
            for path in _entry_fallback(index, graph)
        )
    candidates.sort(
        key=lambda path: (
            -focus_of[path],
            tiers.get(path, 9),
            -hits.get(path, 0),
            -lexical_of[path],
            by_path[path].grammar is None,
            path.parts,
        )
    )
    ranked: list[RankedFile] = []
    for path in candidates:
        tier = tiers.get(path, 9)
        focus = focus_of[path]
        if tier == 0:
            reason = "named"
        elif tier == 1:
            reason = "defines"
        elif tier == 2:
            reason = "pair"
        elif tier == 3:
            reason = "call"
        elif tier == 4:
            reason = "include"
        elif focus > 0:
            reason = "focus"
        else:
            reason = "lexical"
        ranked.append(
            RankedFile(
                path=path,
                reason=reason,
                focus=focus,
                tier=tier,
                hits=hits.get(path, 0),
                lexical=lexical_of[path],
            )
        )
    return tuple(ranked)


def rank_files(
    question: str, index: ProjectIndex, graph: CodeGraph
) -> tuple[Path, ...]:
    return tuple(
        ranked.path for ranked in rank_files_detailed(question, index, graph)
    )


def fuse_rankings(
    ranked: Sequence[RankedFile],
    semantic_order: Sequence[Path],
    k: int = RRF_K,
) -> tuple[RankedFile, ...]:
    ranked_items = tuple(ranked)
    focus = {item.path for item in ranked_items if item.focus > 0}
    sem_paths: list[Path] = []
    seen_sem: set[Path] = set()
    for path in semantic_order:
        if path in seen_sem:
            continue
        seen_sem.add(path)
        if focus and path not in focus:
            continue
        sem_paths.append(path)
    sem_rank = {path: index for index, path in enumerate(sem_paths, start=1)}
    lex_rank: dict[Path, int] = {}
    by_path: dict[Path, RankedFile] = {}
    for index, item in enumerate(ranked_items, start=1):
        lex_rank.setdefault(item.path, index)
        by_path.setdefault(item.path, item)

    def score(path: Path) -> float:
        total = 0.0
        if path in lex_rank:
            total += 1.0 / (k + lex_rank[path])
        if path in sem_rank:
            total += 1.0 / (k + sem_rank[path])
        return total

    if focus:
        focus_paths = [item.path for item in ranked_items if item.path in focus]
        rest = [item for item in ranked_items if item.path not in focus]
        focus_paths.sort(key=lambda path: (-score(path), path.parts))
        ordered = [
            dataclasses.replace(by_path[path], semantic=sem_rank.get(path))
            for path in focus_paths
        ]
        ordered.extend(
            dataclasses.replace(item, semantic=None) for item in rest
        )
        return tuple(ordered)

    pool: list[Path] = []
    seen_pool: set[Path] = set()
    for item in ranked_items:
        if item.path not in seen_pool:
            pool.append(item.path)
            seen_pool.add(item.path)
    for path in sem_paths:
        if path not in seen_pool:
            pool.append(path)
            seen_pool.add(path)
    pool.sort(key=lambda path: (-score(path), path.parts))
    fused: list[RankedFile] = []
    for path in pool:
        rank = sem_rank.get(path)
        item = by_path.get(path)
        if item is None:
            fused.append(
                RankedFile(
                    path=path,
                    reason="semantic",
                    focus=0,
                    tier=9,
                    hits=0,
                    lexical=0,
                    semantic=rank,
                )
            )
        else:
            fused.append(dataclasses.replace(item, semantic=rank))
    return tuple(fused)


def pack_files(
    ranked: Sequence[Path], files: Mapping[Path, ProjectFile], max_chars: int
) -> tuple[Path, ...]:
    if max_chars <= 0:
        return ()
    used = 0
    included: list[Path] = []
    for path in ranked:
        project_file = files[path]
        block = len(project_file.content) + len(path.as_posix()) + 64
        if used + block <= max_chars:
            included.append(path)
            used += block
    return tuple(included)


def _tokens(question: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(question):
        token = raw.strip(_EDGE_PUNCT)
        if token:
            tokens.append(token)
    return tokens


def _is_code_like(token: str) -> bool:
    if "::" in token or "-" in token or any(char.isdigit() for char in token):
        return True
    if any(char.isupper() for char in token[1:]):
        return True
    return "_" in token.strip("_")


def _named_file_tail(token: str) -> tuple[str, ...] | None:
    lowered = token.lower()
    segment = lowered.rsplit("/", 1)[-1]
    named = (
        any(lowered.endswith(suffix) for suffix in SUPPORTED_EXTENSIONS)
        or text_kind(Path(lowered)) is not None
        or segment == "cmakelists.txt"
        or segment.startswith("kconfig")
    )
    if not named:
        return None
    return tuple(part.lower() for part in re.split(r"[/\\]", token) if part)


def _question_words(tokens: list[str]) -> tuple[str, ...]:
    words: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for part in _split_ident(token):
            if part in seen:
                continue
            seen.add(part)
            words.append(part)
    return tuple(words)


def _word_forms(word: str) -> frozenset[str]:
    return frozenset((word, *(word[:k] for k in range(5, len(word)))))


def _split_ident(text: str) -> list[str]:
    parts: list[str] = []
    for chunk in re.split(r"[_:\-./\\]+", text):
        if not chunk:
            continue
        pieces = _CAMEL_RE.findall(chunk)
        for piece in pieces:
            lowered = piece.lower()
            if len(lowered) >= 3 and lowered not in _STOPWORDS:
                parts.append(lowered)
        if _has_short_camel_piece(chunk, pieces):
            parts.append(chunk.lower())
    return parts


def _has_short_camel_piece(chunk: str, pieces: list[str]) -> bool:
    if len(pieces) < 2 or len(chunk) < 5 or not chunk.isalnum():
        return False
    return any(piece.isalpha() and len(piece) < 3 for piece in pieces)


def _ends_with(parts: tuple[str, ...], tail: tuple[str, ...]) -> bool:
    if len(tail) > len(parts):
        return False
    return parts[-len(tail):] == tail


def _focus_count(path: Path, focus_dirs: set[str]) -> int:
    matched: set[str] = set()
    for part in path.parts[:-1]:
        lowered = part.lower()
        if lowered in focus_dirs:
            matched.add(lowered)
    return len(matched)


def _file_terms(file: FileSymbols) -> frozenset[str]:
    terms: set[str] = set()
    for component in file.path.parts:
        terms.update(_split_ident(component))
    for symbol in file.symbols:
        if symbol.kind in _TERM_KINDS:
            terms.update(_split_ident(symbol.name))
    return frozenset(terms)


def _lexical_score(file: FileSymbols, word_forms: tuple[frozenset[str], ...]) -> int:
    terms = _file_terms(file)
    return sum(1 for forms in word_forms if terms & forms)


def _entry_fallback(index: ProjectIndex, graph: CodeGraph) -> tuple[Path, ...]:
    chosen: list[Path] = []
    seen: set[Path] = set()

    def take(path: Path) -> None:
        if path in seen:
            return
        seen.add(path)
        chosen.append(path)

    for ref in graph.definitions("main"):
        simple = ref.symbol.name.rsplit("::", 1)[-1]
        if ref.symbol.kind == SymbolKind.FUNCTION and simple == "main":
            take(ref.path)
    for file in index.files:
        name = file.path.name.lower()
        if fnmatchcase(name, "startup*.s") or file.path.suffix.lower() == ".ld":
            take(file.path)
    if not chosen:
        return tuple(sorted((file.path for file in index.files), key=lambda path: path.parts))
    chosen.sort(key=lambda path: path.parts)
    return tuple(chosen)
