import logging
import os
from pathlib import Path

import pytest

from app.project_loader import MAX_FILE_BYTES, load_project_files

def test_load_project_files_loads_each_supported_extension(tmp_path):

    # Arrange
    (tmp_path / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
    (tmp_path / "app.h").write_text("#pragma once\n", encoding="utf-8")
    (tmp_path / "driver.cpp").write_text("void init() {}\n", encoding="utf-8")
    (tmp_path / "driver.hpp").write_text("class Driver {};\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path.name for project_file in project_files} == {
        "main.c",
        "app.h",
        "driver.cpp",
        "driver.hpp",
    }

def test_load_project_files_content_matches_file_exactly(tmp_path):

    # Arrange
    content = "int value = 1;\n\n    return value;   \n"
    (tmp_path / "main.c").write_text(content, encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert len(project_files) == 1
    assert project_files[0].content == content

def test_load_project_files_paths_equal_source_paths(tmp_path):

    # Arrange
    expected_paths = {
        tmp_path / "main.c",
        tmp_path / "app.h",
        tmp_path / "driver.cpp",
        tmp_path / "driver.hpp",
    }
    for path in expected_paths:
        path.write_text("int x;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path for project_file in project_files} == {
        path.relative_to(tmp_path) for path in expected_paths
    }

def test_load_project_files_ignores_unsupported_files(tmp_path):

    # Arrange
    (tmp_path / "notes.txt").write_text("notes\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")
    (tmp_path / "script.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "main.o").write_text("object\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("all:\n", encoding="utf-8")
    (tmp_path / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path.name for project_file in project_files} == {"main.c"}

def test_load_project_files_finds_files_in_nested_directories(tmp_path):

    # Arrange
    nested_paths = {
        tmp_path / "src" / "main.c",
        tmp_path / "src" / "drivers" / "gpio.c",
        tmp_path / "inc" / "gpio.h",
    }
    for path in nested_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("void gpio_init(void) {}\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path for project_file in project_files} == {
        path.relative_to(tmp_path) for path in nested_paths
    }

def test_load_project_files_skips_directory_named_like_source_file(tmp_path):

    # Arrange
    fake_dir = tmp_path / "fake.c"
    inner_file = fake_dir / "kept.h"
    fake_dir.mkdir()
    inner_file.write_text("#pragma once\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path for project_file in project_files} == {
        inner_file.relative_to(tmp_path)
    }

def test_load_project_files_empty_directory_returns_empty_list(tmp_path):

    # Arrange
    # tmp_path starts empty

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert project_files == []

def test_load_project_files_only_unsupported_files_returns_empty_list(tmp_path):

    # Arrange
    (tmp_path / "notes.txt").write_text("notes\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert project_files == []

def test_load_project_files_nonexistent_path_raises_value_error(tmp_path):

    # Arrange
    missing_path = tmp_path / "does_not_exist"

    # Act / Assert
    with pytest.raises(ValueError):
        load_project_files(missing_path)

def test_load_project_files_file_path_raises_value_error(tmp_path):

    # Arrange
    file_path = tmp_path / "main.c"
    file_path.write_text("int main(void) { return 0; }\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError):
        load_project_files(file_path)

def test_load_project_files_reads_non_ascii_utf8(tmp_path):

    # Arrange
    content = "// Größe °C"
    (tmp_path / "sensor.c").write_text(content, encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert len(project_files) == 1
    assert project_files[0].content == content

def test_load_project_files_skips_non_utf8_file(tmp_path):

    # Arrange
    (tmp_path / "good.c").write_text("int good = 1;\n", encoding="utf-8")
    (tmp_path / "bad.c").write_bytes(b"\x81\x8d\x8f\x90\x9d")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path.name for project_file in project_files} == {"good.c"}

def test_load_project_files_warns_on_non_utf8_file(tmp_path, caplog):

    # Arrange
    (tmp_path / "good.c").write_text("int good = 1;\n", encoding="utf-8")
    (tmp_path / "bad.c").write_bytes(b"\x81\x8d\x8f\x90\x9d")
    caplog.set_level(logging.WARNING)

    # Act
    load_project_files(tmp_path)

    # Assert
    assert any(
        record.levelno == logging.WARNING and "bad.c" in record.getMessage()
        for record in caplog.records
    )

def test_load_project_files_loads_added_extensions(tmp_path):

    # Arrange
    names = [
        "unit.cc",
        "unit.cxx",
        "unit.hh",
        "unit.hxx",
        "unit.inl",
        "startup.s",
        "startup.asm",
        "linker.ld",
    ]
    for name in names:
        (tmp_path / name).write_text("int x;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path.name for project_file in project_files} == set(names)

def test_load_project_files_matches_extensions_case_insensitively(tmp_path):

    # Arrange
    names = ["MAIN.C", "APP.H", "startup.S", "Driver.CPP"]
    for name in names:
        (tmp_path / name).write_text("int x;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path.name for project_file in project_files} == set(names)

def test_load_project_files_skips_hidden_directories(tmp_path):

    # Arrange
    skipped = [tmp_path / ".git" / "x.c", tmp_path / "src" / ".cache" / "y.c"]
    kept = tmp_path / "src" / "z.c"
    for path in [*skipped, kept]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("int x;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path for project_file in project_files} == {
        kept.relative_to(tmp_path)
    }

def test_load_project_files_skips_build_directories(tmp_path):

    # Arrange
    skipped = [
        tmp_path / "build" / "a.c",
        tmp_path / "out" / "b.c",
        tmp_path / "cmake-build-debug" / "c.c",
        tmp_path / "fw" / "build" / "d.c",
    ]
    kept = tmp_path / "main.c"
    for path in [*skipped, kept]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("int x;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path for project_file in project_files} == {
        kept.relative_to(tmp_path)
    }

def test_load_project_files_loads_debug_directory(tmp_path):

    # Arrange
    kept = tmp_path / "Debug" / "trace.c"
    kept.parent.mkdir()
    kept.write_text("int trace;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path for project_file in project_files} == {
        kept.relative_to(tmp_path)
    }

def test_load_project_files_applies_exclude_dir_patterns(tmp_path):

    # Arrange
    skipped = [
        tmp_path / "Drivers" / "CMSIS" / "core.h",
        tmp_path / "Drivers" / "STM32F4xx_HAL_Driver" / "Src" / "hal.c",
        tmp_path / "Middlewares" / "Third_Party" / "x.c",
    ]
    kept = tmp_path / "Core" / "Src" / "main.c"
    for path in [*skipped, kept]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("int x;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(
        tmp_path, ("CMSIS", "*_HAL_Driver", "third_party")
    )

    # Assert
    assert {project_file.path for project_file in project_files} == {
        kept.relative_to(tmp_path)
    }

def test_load_project_files_does_not_exclude_root_by_name(tmp_path):

    # Arrange
    root = tmp_path / "vendor"
    kept = root / "a.c"
    root.mkdir()
    kept.write_text("int a;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(root, ("vendor",))

    # Assert
    assert {project_file.path for project_file in project_files} == {
        kept.relative_to(root)
    }

def test_load_project_files_default_exclude_dirs_loads_vendor(tmp_path):

    # Arrange
    kept = tmp_path / "vendor" / "a.c"
    kept.parent.mkdir()
    kept.write_text("int a;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path for project_file in project_files} == {
        kept.relative_to(tmp_path)
    }

def test_load_project_files_loads_generated_style_names(tmp_path):

    # Arrange
    names = ["Rte_Can.c", "Can_Cfg.h"]
    for name in names:
        (tmp_path / name).write_text("int x;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path.name for project_file in project_files} == set(names)

def test_load_project_files_skips_file_over_size_cap(tmp_path, caplog):

    # Arrange
    (tmp_path / "big.c").write_bytes(b"b" * (MAX_FILE_BYTES + 1))
    (tmp_path / "exact.c").write_bytes(b"a" * MAX_FILE_BYTES)
    caplog.set_level(logging.WARNING)

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert len(project_files) == 1
    assert project_files[0].path.name == "exact.c"
    assert project_files[0].content == "a" * MAX_FILE_BYTES
    assert any(
        record.levelno == logging.WARNING and "big.c" in record.getMessage()
        for record in caplog.records
    )

def test_load_project_files_strips_utf8_bom(tmp_path):

    # Arrange
    (tmp_path / "main.c").write_bytes(b"\xef\xbb\xbfint x;\n")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert len(project_files) == 1
    assert project_files[0].content == "int x;\n"

def test_load_project_files_does_not_touch_pruned_directory(tmp_path, monkeypatch):

    # Arrange
    excluded = tmp_path / "build"
    secret = excluded / "secret.c"
    excluded.mkdir()
    secret.write_text("int secret;\n", encoding="utf-8")
    secret.chmod(0)
    (tmp_path / "main.c").write_text("int kept;\n", encoding="utf-8")
    real_stat = os.stat
    real_open = open

    def guarded_stat(path, *args, **kwargs):
        if excluded in Path(path).parents:
            raise OSError(f"stat under excluded dir: {path}")
        return real_stat(path, *args, **kwargs)

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, int):
            return real_open(file, *args, **kwargs)
        if excluded in Path(file).parents:
            raise OSError(f"open under excluded dir: {file}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(os, "stat", guarded_stat)
    monkeypatch.setattr("builtins.open", guarded_open)

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path.name for project_file in project_files} == {"main.c"}

def test_load_project_files_path_is_relative_to_project_root(tmp_path):

    # Arrange
    source = tmp_path / "src" / "main.c"
    source.parent.mkdir()
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert len(project_files) == 1
    assert project_files[0].path == Path("src/main.c")
    assert not project_files[0].path.is_absolute()

def test_load_project_files_sorts_by_relative_path_parts(tmp_path):

    # Arrange
    names = [
        "src/z.c",
        "Core/a.c",
        "src/drivers/x.c",
        "src/a.c",
        "inc/a.h",
        "src-x/b.c",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("int x;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert [project_file.path.as_posix() for project_file in project_files] == [
        "Core/a.c",
        "inc/a.h",
        "src/a.c",
        "src/drivers/x.c",
        "src/z.c",
        "src-x/b.c",
    ]

def test_load_project_files_decodes_cp1252_apostrophe(tmp_path):

    # Arrange
    (tmp_path / "note.c").write_bytes(b"// it\x92s\nint x;\n")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert len(project_files) == 1
    assert project_files[0].content == "// it\u2019s\nint x;\n"

def test_load_project_files_logs_info_summary_on_cp1252_fallback(tmp_path, caplog):

    # Arrange
    (tmp_path / "note.c").write_bytes(b"// it\x92s\nint x;\n")
    caplog.set_level(logging.INFO)

    # Act
    load_project_files(tmp_path)

    # Assert
    assert any(
        record.name == "app.project_loader"
        and record.levelno == logging.INFO
        and "note.c" in record.getMessage()
        for record in caplog.records
    )

def test_load_project_files_cp1252_crlf_becomes_lf(tmp_path):

    # Arrange
    (tmp_path / "note.c").write_bytes(b"// it\x92s\r\nint x;\r\n")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert len(project_files) == 1
    assert project_files[0].content == "// it\u2019s\nint x;\n"

def test_load_project_files_utf8_logs_no_warning(tmp_path, caplog):

    # Arrange
    (tmp_path / "main.c").write_text("int x;\n", encoding="utf-8")
    caplog.set_level(logging.WARNING)

    # Act
    load_project_files(tmp_path)

    # Assert
    assert not any(record.levelno == logging.WARNING for record in caplog.records)


_CP1252_BYTES = b"// it\x92s\nint x;\n"
_CP1252_TEXT = "// it\u2019s\nint x;\n"
_UNDECODABLE_BYTES = b"\x81\x8d\x8f\x90\x9d"


def _loader_records(caplog):
    return [record for record in caplog.records if record.name == "app.project_loader"]


def _write_cp1252(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_CP1252_BYTES)


def _write_mixed_project(root: Path) -> None:
    for name in ("a.c", "b.c", "c.c", "d.c", "e.c"):
        _write_cp1252(root / name)
    (root / "big.c").write_bytes(b"b" * (MAX_FILE_BYTES + 1))
    (root / "bad.c").write_bytes(_UNDECODABLE_BYTES)
    (root / "ok.c").write_text("int ok;\n", encoding="utf-8")


def _enable_loader_debug(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="app.project_loader")


def test_load_project_files_logs_one_summary_per_category(tmp_path, caplog):

    # Arrange
    _write_mixed_project(tmp_path)
    _enable_loader_debug(caplog)

    # Act
    load_project_files(tmp_path)

    # Assert
    high = [
        record
        for record in _loader_records(caplog)
        if record.levelno >= logging.INFO
    ]
    assert [(record.levelno, record.getMessage()) for record in high] == [
        (
            logging.INFO,
            "Decoded 5 files as cp1252 (not valid UTF-8); "
            "e.g. a.c, b.c, c.c, and 2 more. "
            "Per-file list: DEBUG, logs/app.log.",
        ),
        (
            logging.WARNING,
            "Skipped 1 file over the 512 KiB size cap; "
            "e.g. big.c. Per-file list: DEBUG, logs/app.log.",
        ),
        (
            logging.WARNING,
            "Skipped 1 file that are neither UTF-8 nor cp1252; "
            "e.g. bad.c. Per-file list: DEBUG, logs/app.log.",
        ),
    ]
    assert all(str(tmp_path) not in record.getMessage() for record in _loader_records(caplog))


def test_load_project_files_logs_per_file_lines_at_debug(tmp_path, caplog):

    # Arrange
    _write_mixed_project(tmp_path)
    _enable_loader_debug(caplog)

    # Act
    load_project_files(tmp_path)

    # Assert
    records = _loader_records(caplog)
    debug = [record.getMessage() for record in records if record.levelno == logging.DEBUG]
    assert debug.count("Decoded a.c as cp1252 (not valid UTF-8).") == 1
    assert debug.count("Decoded b.c as cp1252 (not valid UTF-8).") == 1
    assert debug.count("Decoded c.c as cp1252 (not valid UTF-8).") == 1
    assert debug.count("Decoded d.c as cp1252 (not valid UTF-8).") == 1
    assert debug.count("Decoded e.c as cp1252 (not valid UTF-8).") == 1
    assert (
        f"Skipping file over size cap: big.c ({MAX_FILE_BYTES + 1} bytes)" in debug
    )
    assert "Skipping file that is neither UTF-8 nor cp1252: bad.c" in debug
    assert len(debug) == 7
    for record in records:
        if record.levelno < logging.INFO:
            continue
        message = record.getMessage()
        assert not message.startswith("Skipping file")
        assert "; e.g. " in message


def test_load_project_files_clean_utf8_logs_no_loader_record(tmp_path, caplog):

    # Arrange
    (tmp_path / "main.c").write_text("int x;\n", encoding="utf-8")
    _enable_loader_debug(caplog)

    # Act
    load_project_files(tmp_path)

    # Assert
    assert _loader_records(caplog) == []


def test_load_project_files_summary_singular_plural_and_more(tmp_path, caplog):

    # Arrange
    _enable_loader_debug(caplog)

    def summary_for(names: list[str]) -> str:
        root = tmp_path / f"n{len(names)}"
        root.mkdir()
        for name in names:
            _write_cp1252(root / name)
        caplog.clear()
        load_project_files(root)
        high = [
            record
            for record in _loader_records(caplog)
            if record.levelno >= logging.INFO
        ]
        assert len(high) == 1
        return high[0].getMessage()

    # Act
    one = summary_for(["a.c"])
    three = summary_for(["a.c", "b.c", "c.c"])
    four = summary_for(["a.c", "b.c", "c.c", "d.c"])

    # Assert
    assert one == (
        "Decoded 1 file as cp1252 (not valid UTF-8); e.g. a.c. "
        "Per-file list: DEBUG, logs/app.log."
    )
    assert "and " not in one
    assert "more" not in one
    assert three == (
        "Decoded 3 files as cp1252 (not valid UTF-8); "
        "e.g. a.c, b.c, c.c. Per-file list: DEBUG, logs/app.log."
    )
    assert "more" not in three
    assert four == (
        "Decoded 4 files as cp1252 (not valid UTF-8); "
        "e.g. a.c, b.c, c.c, and 1 more. Per-file list: DEBUG, logs/app.log."
    )


def test_load_project_files_summary_examples_sorted_posix(tmp_path, caplog, monkeypatch):

    # Arrange
    for name in (
        "src-x/b.c",
        "src/z.c",
        "src/a.c",
        "Examples/I2C/Src/main.c",
    ):
        _write_cp1252(tmp_path / name)

    def walk_unsorted(self, top_down=True, on_error=None, follow_symlinks=False):
        yield self, ["src-x", "src", "Examples"], []
        yield self / "src-x", [], ["b.c"]
        yield self / "src", [], ["z.c", "a.c"]
        yield self / "Examples", ["I2C"], []
        yield self / "Examples" / "I2C", ["Src"], []
        yield self / "Examples" / "I2C" / "Src", [], ["main.c"]

    monkeypatch.setattr(Path, "walk", walk_unsorted)
    _enable_loader_debug(caplog)

    # Act
    load_project_files(tmp_path)

    # Assert
    high = [
        record
        for record in _loader_records(caplog)
        if record.levelno >= logging.INFO
    ]
    assert len(high) == 1
    assert high[0].getMessage() == (
        "Decoded 4 files as cp1252 (not valid UTF-8); "
        "e.g. Examples/I2C/Src/main.c, src/a.c, src/z.c, and 1 more. "
        "Per-file list: DEBUG, logs/app.log."
    )


def test_load_project_files_logging_does_not_change_loaded_files(tmp_path, caplog):

    # Arrange
    _write_mixed_project(tmp_path)
    _enable_loader_debug(caplog)
    expected = [
        (Path("a.c"), _CP1252_TEXT),
        (Path("b.c"), _CP1252_TEXT),
        (Path("c.c"), _CP1252_TEXT),
        (Path("d.c"), _CP1252_TEXT),
        (Path("e.c"), _CP1252_TEXT),
        (Path("ok.c"), "int ok;\n"),
    ]

    def summaries():
        return [
            (record.levelno, record.getMessage())
            for record in _loader_records(caplog)
            if record.levelno >= logging.INFO
        ]

    # Act
    first = load_project_files(tmp_path)
    first_summaries = summaries()
    caplog.clear()
    second = load_project_files(tmp_path)

    # Assert
    assert [(item.path, item.content) for item in first] == expected
    assert [(item.path, item.content) for item in second] == expected
    assert summaries() == first_summaries
    assert len(first_summaries) == 3


def test_load_project_files_invalid_path_logs_error_without_summary(tmp_path, caplog):

    # Arrange
    missing_path = tmp_path / "does_not_exist"
    file_path = tmp_path / "main.c"
    file_path.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    _enable_loader_debug(caplog)

    # Act / Assert
    for invalid in (missing_path, file_path):
        caplog.clear()
        with pytest.raises(ValueError, match="Invalid project path:"):
            load_project_files(invalid)
        records = _loader_records(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.ERROR
        assert records[0].getMessage() == f"Invalid project path: {invalid}"


def test_l1_loads_one_file_of_each_text_kind(tmp_path):
    names = {
        "main.c": "int main(void) { return 0; }\n",
        "Kconfig": "config FOO\n\tbool\n",
        "CMakeLists.txt": "project(app)\n",
        "board.overlay": "/ {};\n",
        "prj.conf": "CONFIG_FOO=y\n",
        "app.yaml": "name: app\n",
        "doc.rst": "Title\n=====\n",
        "README.md": "hi\n",
        "notes.txt": "hi\n",
        "x.bin": "hi\n",
    }
    for name, content in names.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    loaded = load_project_files(tmp_path)

    assert [item.path.name for item in loaded] == [
        "CMakeLists.txt",
        "Kconfig",
        "app.yaml",
        "board.overlay",
        "doc.rst",
        "main.c",
        "prj.conf",
    ]


def test_l2_skips_kconfig_over_size_cap(tmp_path, caplog):
    (tmp_path / "Kconfig").write_bytes(b"a" * (600 * 1024))
    (tmp_path / "main.c").write_text("int x;\n", encoding="utf-8")
    caplog.set_level(logging.WARNING)

    loaded = load_project_files(tmp_path)

    assert [item.path.name for item in loaded] == ["main.c"]
    assert any(
        record.levelno == logging.WARNING and "Kconfig" in record.getMessage()
        for record in caplog.records
    )
