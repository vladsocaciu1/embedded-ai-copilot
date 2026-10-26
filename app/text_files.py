import re
from collections.abc import Iterator
from pathlib import Path

from app.models import ProjectFile, Symbol, SymbolKind

KCONFIG_NAMES_PREFIX = "Kconfig"
PIECE_CAP = 1500
FILE_MAX_PIECES = 8

_KCONFIG_KEYWORD = re.compile(
    r"^(config|menuconfig|menu|endmenu|choice|endchoice|if|endif|source|rsource|"
    r"osource|orsource|comment|mainmenu)\b"
)
_KCONFIG_SYMBOL = re.compile(r"^(?:menuconfig|config)\s+([A-Za-z0-9_]+)\s*$")
_RST_UNDERLINE = re.compile(r"^([=\-#*~^\"'`+:.<>_])\1{2,}\s*$")
_DT_LABEL = re.compile(r"^\s*([A-Za-z_]\w*):\s*[\w@,\-]+\s*\{", re.M)
_DT_ALIAS = re.compile(r"^\s*([A-Za-z_][\w\-]*)\s*=\s*&", re.M)
_CONF_OPTION = re.compile(r"^\s*(CONFIG_[A-Za-z0-9_]+)\s*=", re.M)


def text_kind(path: Path) -> str | None:
    name = path.name
    suffix = path.suffix.lower()
    if name == "Kconfig" or name.startswith("Kconfig."):
        return "kconfig"
    if name == "CMakeLists.txt" or suffix == ".cmake":
        return "cmake"
    if suffix in (".overlay", ".dts", ".dtsi"):
        return "devicetree"
    if suffix == ".conf":
        return "conf"
    if suffix in (".yaml", ".yml"):
        return "yaml"
    if suffix == ".rst":
        return "rst"
    return None


def index_text(project_file: ProjectFile, kind: str) -> tuple[Symbol, ...]:
    content = project_file.content
    if kind == "kconfig":
        return tuple(_kconfig_symbols(content))
    if kind == "rst":
        return tuple(_rst_symbols(content))
    if kind == "devicetree":
        names = [m.group(1) for m in _DT_LABEL.finditer(content)]
        names += [m.group(1) for m in _DT_ALIAS.finditer(content)]
        return tuple(Symbol(n, SymbolKind.NODE, 1, 1) for n in dict.fromkeys(names))
    if kind == "conf":
        names = [m.group(1) for m in _CONF_OPTION.finditer(content)]
        return tuple(Symbol(n, SymbolKind.CONFIG, 1, 1) for n in dict.fromkeys(names))
    return ()


def _kconfig_symbols(content: str) -> Iterator[Symbol]:
    lines = content.splitlines()
    name = None
    start = 0
    for number, line in enumerate(lines, 1):
        if not _KCONFIG_KEYWORD.match(line):
            continue
        if name is not None:
            yield Symbol(name, SymbolKind.CONFIG, start, number - 1)
            name = None
        match = _KCONFIG_SYMBOL.match(line)
        if match:
            name, start = "CONFIG_" + match.group(1), number
    if name is not None:
        yield Symbol(name, SymbolKind.CONFIG, start, len(lines))


def _rst_symbols(content: str) -> Iterator[Symbol]:
    lines = content.splitlines()
    heads: list[tuple[str, int]] = []
    i = 0
    while i < len(lines) - 1:
        title, under = lines[i].strip(), lines[i + 1].strip()
        if (
            title
            and not lines[i].startswith((" ", "\t", ".."))
            and not _RST_UNDERLINE.match(title)
            and _RST_UNDERLINE.match(under)
            and len(under) >= len(title)
        ):
            heads.append((title, i + 1))
            i += 2
        else:
            i += 1
    for index, (title, line) in enumerate(heads):
        end = heads[index + 1][1] - 1 if index + 1 < len(heads) else len(lines)
        yield Symbol(title, SymbolKind.SECTION, line, end)


def pieces(text: str, cap: int, limit: int) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        line = line[:cap]
        if size + len(line) + 1 > cap and current:
            out.append("\n".join(current))
            current, size = [], 0
            if len(out) == limit:
                return out
        current.append(line)
        size += len(line) + 1
    if current and len(out) < limit:
        out.append("\n".join(current))
    return out
