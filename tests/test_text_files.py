from pathlib import Path

import pytest

from app.models import ProjectFile, Symbol, SymbolKind
from app.text_files import index_text, pieces, text_kind


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Kconfig", "kconfig"),
        ("Kconfig.zephyr", "kconfig"),
        ("kconfig", None),
        ("CMakeLists.txt", "cmake"),
        ("cmakelists.txt", None),
        ("x.CMAKE", "cmake"),
        ("board.overlay", "devicetree"),
        ("a.dtsi", "devicetree"),
        ("prj.conf", "conf"),
        ("x.yml", "yaml"),
        ("index.rst", "rst"),
        ("README.md", None),
        ("notes.txt", None),
        ("startup.S", None),
        ("Makefile", None),
        ("Kconfig-x", None),
    ],
)
def test_t1_text_kind(name, expected):
    assert text_kind(Path(name)) == expected


def test_t2_kconfig_symbols():
    content = (
        "config SIMPLE\n"
        '\tbool "s"\n'
        "\thelp\n"
        "\t  stays in the block\n"
        "\n"
        "menuconfig MENU\n"
        '\tbool "m"\n'
        "\n"
        "choice\n"
        '\tprompt "pick"\n'
        "\n"
        "config INSIDE\n"
        '\tbool "in"\n'
        "\n"
        "endchoice\n"
        "\n"
        "config $(X)\n"
        '\tbool "skip"\n'
        "\n"
        "config LAST\n"
        '\tint "end"\n'
    )
    symbols = index_text(ProjectFile(Path("Kconfig"), content), "kconfig")

    assert symbols == (
        Symbol("CONFIG_SIMPLE", SymbolKind.CONFIG, 1, 5),
        Symbol("CONFIG_MENU", SymbolKind.CONFIG, 6, 8),
        Symbol("CONFIG_INSIDE", SymbolKind.CONFIG, 12, 14),
        Symbol("CONFIG_LAST", SymbolKind.CONFIG, 20, 21),
    )
    assert symbols[0].end_line >= 4
    assert all("$(X)" not in symbol.name for symbol in symbols)
    assert symbols[-1].end_line == len(content.splitlines())
    assert index_text(ProjectFile(Path("Kconfig"), ""), "kconfig") == ()


def test_t3_rst_headings_and_section_ranges():
    content = (
        "Equal\n"
        "=====\n"
        "\n"
        "body\n"
        "\n"
        "Longer title\n"
        "----------------\n"
        "more\n"
        "\n"
        "Short\n"
        "----\n"
        "\n"
        "    indented\n"
        "    =========\n"
        "\n"
        ".. comment\n"
        "==========\n"
        "\n"
        "-----\n"
        "\n"
        "Tail\n"
        "====\n"
        "end\n"
    )
    symbols = index_text(ProjectFile(Path("index.rst"), content), "rst")

    assert symbols == (
        Symbol("Equal", SymbolKind.SECTION, 1, 5),
        Symbol("Longer title", SymbolKind.SECTION, 6, 20),
        Symbol("Tail", SymbolKind.SECTION, 21, 23),
    )
    names = {symbol.name for symbol in symbols}
    assert "Short" not in names
    assert "indented" not in names
    assert "comment" not in names
    assert "-----" not in names


def test_t4_devicetree_conf_and_pieces():
    overlay = (
        "/ {\n"
        "    leds: gpio-leds {\n"
        "        led0: led_0 {\n"
        "        };\n"
        "    };\n"
        "    aliases {\n"
        "        led0 = &led0;\n"
        "        pwm-led = &pwm;\n"
        "    };\n"
        "};\n"
    )
    nodes = index_text(ProjectFile(Path("board.overlay"), overlay), "devicetree")
    assert [symbol.name for symbol in nodes] == ["leds", "led0", "pwm-led"]
    assert all(symbol.kind is SymbolKind.NODE for symbol in nodes)

    conf = "CONFIG_FOO=y\nCONFIG_FOO=n\n  CONFIG_BAR=1\n# CONFIG_BAZ is not set\n"
    options = index_text(ProjectFile(Path("prj.conf"), conf), "conf")
    assert [symbol.name for symbol in options] == ["CONFIG_FOO", "CONFIG_BAR"]
    assert all(symbol.kind is SymbolKind.CONFIG for symbol in options)

    assert index_text(ProjectFile(Path("app.yaml"), "name: app\n"), "yaml") == ()
    assert index_text(ProjectFile(Path("CMakeLists.txt"), "project(app)\n"), "cmake") == ()

    assert pieces("", 1500, 8) == []
    assert pieces("z" * 5000, 1500, 8) == ["z" * 1500]
    assert len(pieces(("y" * 1500 + "\n") * 20, 1500, 8)) == 8
    line = "a" * 750
    assert pieces(f"{line}\n{line}\n", 1500, 8) == [line, line]
