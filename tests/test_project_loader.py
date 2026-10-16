import logging

import pytest

from app.project_loader import load_project_files

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
