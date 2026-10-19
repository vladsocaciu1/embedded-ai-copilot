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
    assert {project_file.path for project_file in project_files} == expected_paths

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
    assert {project_file.path for project_file in project_files} == nested_paths

def test_load_project_files_skips_directory_named_like_source_file(tmp_path):

    # Arrange
    fake_dir = tmp_path / "fake.c"
    inner_file = fake_dir / "kept.h"
    fake_dir.mkdir()
    inner_file.write_text("#pragma once\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path for project_file in project_files} == {inner_file}

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
    (tmp_path / "bad.c").write_bytes(b"\xff\xfe\xfa")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path.name for project_file in project_files} == {"good.c"}

def test_load_project_files_warns_on_non_utf8_file(tmp_path, caplog):

    # Arrange
    (tmp_path / "good.c").write_text("int good = 1;\n", encoding="utf-8")
    (tmp_path / "bad.c").write_bytes(b"\xff\xfe\xfa")
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
    assert {project_file.path for project_file in project_files} == {kept}

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
    assert {project_file.path for project_file in project_files} == {kept}

def test_load_project_files_loads_debug_directory(tmp_path):

    # Arrange
    kept = tmp_path / "Debug" / "trace.c"
    kept.parent.mkdir()
    kept.write_text("int trace;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path for project_file in project_files} == {kept}

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
    assert {project_file.path for project_file in project_files} == {kept}

def test_load_project_files_does_not_exclude_root_by_name(tmp_path):

    # Arrange
    root = tmp_path / "vendor"
    kept = root / "a.c"
    root.mkdir()
    kept.write_text("int a;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(root, ("vendor",))

    # Assert
    assert {project_file.path for project_file in project_files} == {kept}

def test_load_project_files_default_exclude_dirs_loads_vendor(tmp_path):

    # Arrange
    kept = tmp_path / "vendor" / "a.c"
    kept.parent.mkdir()
    kept.write_text("int a;\n", encoding="utf-8")

    # Act
    project_files = load_project_files(tmp_path)

    # Assert
    assert {project_file.path for project_file in project_files} == {kept}

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
