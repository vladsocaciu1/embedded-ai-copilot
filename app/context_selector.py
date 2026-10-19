import re
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from pathlib import Path

from app.code_graph import CodeGraph
from app.models import FileSymbols, ProjectFile, ProjectIndex, SymbolKind
from app.project_loader import SUPPORTED_EXTENSIONS

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
})


def rank_files(
    question: str, index: ProjectIndex, graph: CodeGraph
) -> tuple[Path, ...]:
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
        return _entry_fallback(index, graph)
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
    return tuple(candidates)


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
    if not any(lowered.endswith(suffix) for suffix in SUPPORTED_EXTENSIONS):
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
        for piece in _CAMEL_RE.findall(chunk):
            lowered = piece.lower()
            if len(lowered) >= 3 and lowered not in _STOPWORDS:
                parts.append(lowered)
    return parts


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
