import logging

import pytest
from google.genai import errors

from app.code_graph import build_graph
from app.code_index import build_index
from app.config import Settings

QUESTION = "Where is the answer initialized?"


def test_generate_answer_returns_response_text(ai_service):

    # Arrange
    module, client = ai_service
    client.models.generate_content.return_value.text = "The answer is 42."

    # Act
    answer = module.generate_answer(QUESTION)

    # Assert
    assert answer.text == "The answer is 42."


def test_generate_answer_calls_generate_content_once_with_model_and_contents(ai_service, tmp_path):

    # Arrange
    module, client = ai_service
    file_content = (tmp_path / "main.c").read_text(encoding="utf-8")
    client.models.generate_content.return_value.text = "The answer is 42."

    # Act
    module.generate_answer(QUESTION)

    # Assert
    client.models.generate_content.assert_called_once()
    assert client.models.generate_content.call_args.kwargs["model"] == "test-model"
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert QUESTION in contents
    assert file_content in contents


def test_generate_answer_invalid_project_path_does_not_call_gemini(ai_service, monkeypatch, tmp_path):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(
        module,
        "settings",
        Settings(
            project_path=tmp_path / "does_not_exist",
            gemini_api_key="test-key-DO-NOT-LOG",
            gemini_model="test-model",
        ),
    )

    # Act
    with pytest.raises(module.ProjectLoadError):
        module.generate_answer(QUESTION)

    # Assert
    client.models.generate_content.assert_not_called()


def test_generate_answer_logs_do_not_contain_api_key(ai_service, caplog, monkeypatch, tmp_path):

    # Arrange
    module, client = ai_service
    secret = "test-key-DO-NOT-LOG"
    valid_settings = module.settings
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="app.ai_service")

    def assert_secret_absent():
        assert caplog.records
        assert secret not in caplog.text

    # Act / Assert: success
    caplog.clear()
    client.models.generate_content.side_effect = None
    client.models.generate_content.return_value.text = "The answer is 42."
    module.generate_answer(QUESTION)
    assert_secret_absent()

    # Act / Assert: invalid project path
    caplog.clear()
    monkeypatch.setattr(
        module,
        "settings",
        Settings(
            project_path=tmp_path / "does_not_exist",
            gemini_api_key=secret,
            gemini_model="test-model",
        ),
    )
    with pytest.raises(module.ProjectLoadError):
        module.generate_answer(QUESTION)
    assert_secret_absent()

    monkeypatch.setattr(module, "settings", valid_settings)

    # Act / Assert: ClientError
    caplog.clear()
    client.models.generate_content.side_effect = errors.ClientError(
        400,
        {"message": "bad request", "status": "INVALID_ARGUMENT"},
    )
    with pytest.raises(module.UpstreamRejectedError):
        module.generate_answer(QUESTION)
    assert_secret_absent()

    # Act / Assert: ServerError
    caplog.clear()
    client.models.generate_content.side_effect = errors.ServerError(
        500,
        {"message": "unavailable", "status": "UNAVAILABLE"},
    )
    with pytest.raises(module.UpstreamUnavailableError):
        module.generate_answer(QUESTION)
    assert_secret_absent()

    # Act / Assert: other APIError
    caplog.clear()
    client.models.generate_content.side_effect = errors.APIError(
        302,
        {"message": "redirected", "status": "FOUND"},
    )
    with pytest.raises(module.UpstreamUnavailableError):
        module.generate_answer(QUESTION)
    assert_secret_absent()


def test_generate_answer_invalid_project_path_raises_project_load_error(ai_service, monkeypatch, tmp_path):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(
        module,
        "settings",
        Settings(
            project_path=tmp_path / "does_not_exist",
            gemini_api_key="test-key-DO-NOT-LOG",
            gemini_model="test-model",
        ),
    )

    # Act / Assert
    with pytest.raises(module.ProjectLoadError) as exc_info:
        module.generate_answer(QUESTION)

    assert isinstance(exc_info.value.__cause__, ValueError)
    client.models.generate_content.assert_not_called()


def test_generate_answer_client_error_raises_upstream_rejected(ai_service):

    # Arrange
    module, client = ai_service
    client.models.generate_content.side_effect = errors.ClientError(
        400,
        {"message": "bad request", "status": "INVALID_ARGUMENT"},
    )

    # Act / Assert
    with pytest.raises(module.UpstreamRejectedError) as exc_info:
        module.generate_answer(QUESTION)

    assert isinstance(exc_info.value.__cause__, errors.ClientError)


def test_generate_answer_server_error_raises_upstream_unavailable(ai_service):

    # Arrange
    module, client = ai_service
    client.models.generate_content.side_effect = errors.ServerError(
        500,
        {"message": "unavailable", "status": "UNAVAILABLE"},
    )

    # Act / Assert
    with pytest.raises(module.UpstreamUnavailableError) as exc_info:
        module.generate_answer(QUESTION)

    assert isinstance(exc_info.value.__cause__, errors.ServerError)


def test_generate_answer_api_error_raises_upstream_unavailable(ai_service):

    # Arrange
    module, client = ai_service
    client.models.generate_content.side_effect = errors.APIError(
        302,
        {"message": "redirected", "status": "FOUND"},
    )

    # Act / Assert
    with pytest.raises(module.UpstreamUnavailableError) as exc_info:
        module.generate_answer(QUESTION)

    assert type(exc_info.value.__cause__) is errors.APIError


def test_generate_answer_runtime_error_propagates(ai_service):

    # Arrange
    module, client = ai_service
    boom = RuntimeError("boom")
    client.models.generate_content.side_effect = boom

    # Act / Assert
    with pytest.raises(RuntimeError) as exc_info:
        module.generate_answer(QUESTION)

    assert exc_info.value is boom


def test_generate_answer_none_text_raises_empty_upstream_response(ai_service):

    # Arrange
    module, client = ai_service
    client.models.generate_content.return_value.text = None

    # Act / Assert
    with pytest.raises(module.EmptyUpstreamResponseError):
        module.generate_answer(QUESTION)


def test_generate_answer_empty_text_raises_empty_upstream_response(ai_service):

    # Arrange
    module, client = ai_service
    client.models.generate_content.return_value.text = ""

    # Act / Assert
    with pytest.raises(module.EmptyUpstreamResponseError):
        module.generate_answer(QUESTION)


def test_generate_answer_exclude_dirs_omits_vendor_file(ai_service, monkeypatch, tmp_path):

    # Arrange
    module, client = ai_service
    main_content = (tmp_path / "main.c").read_text(encoding="utf-8")
    marker = "VENDOR_ONLY_MARKER_9f3a"
    vendor_file = tmp_path / "vendor" / "v.c"
    vendor_file.parent.mkdir()
    vendor_file.write_text(f"int {marker} = 1;\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "settings",
        Settings(
            project_path=tmp_path,
            gemini_api_key="test-key-DO-NOT-LOG",
            gemini_model="test-model",
            exclude_dirs=("vendor",),
        ),
    )
    client.models.generate_content.return_value.text = "ok"

    # Act
    module.generate_answer(QUESTION)

    # Assert
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert marker not in contents
    assert main_content in contents


def test_generate_answer_contents_omit_absolute_project_path(ai_service, tmp_path):

    # Arrange
    module, client = ai_service
    client.models.generate_content.return_value.text = "ok"

    # Act
    module.generate_answer(QUESTION)

    # Assert
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert str(tmp_path) not in contents
    assert "main.c" in contents


def test_generate_answer_repeated_calls_send_identical_contents(ai_service, tmp_path):

    # Arrange
    module, client = ai_service
    for relative in ("src/z.c", "Core/a.c", "src/drivers/x.c", "src/a.c"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("int x;\n", encoding="utf-8")
    client.models.generate_content.return_value.text = "ok"

    # Act
    module.generate_answer(QUESTION)
    first = client.models.generate_content.call_args.kwargs["contents"]
    module.generate_answer(QUESTION)
    second = client.models.generate_content.call_args.kwargs["contents"]

    # Assert
    assert first == second


def test_empty_upstream_response_error_subclasses_upstream_rejected(ai_service):

    # Arrange
    module, _client = ai_service

    # Act / Assert
    assert issubclass(module.EmptyUpstreamResponseError, module.UpstreamRejectedError)


def _settings_with_limit(module, limit: int) -> Settings:
    current = module.settings
    return Settings(
        project_path=current.project_path,
        gemini_api_key=current.gemini_api_key,
        gemini_model=current.gemini_model,
        exclude_dirs=current.exclude_dirs,
        max_prompt_tokens=limit,
    )


def test_generate_answer_counts_tokens_then_returns_answer(ai_service):

    # Arrange
    module, client = ai_service
    client.models.count_tokens.return_value.total_tokens = 100
    client.models.generate_content.return_value.text = "The answer is 42."

    # Act
    answer = module.generate_answer(QUESTION)

    # Assert
    assert answer.text == "The answer is 42."
    client.models.count_tokens.assert_called_once()
    count_kwargs = client.models.count_tokens.call_args.kwargs
    generate_kwargs = client.models.generate_content.call_args.kwargs
    assert count_kwargs["model"] == "test-model"
    assert count_kwargs["contents"] is generate_kwargs["contents"]


def test_generate_answer_token_count_equal_to_limit_calls_generate(ai_service, monkeypatch):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 100_000))
    client.models.count_tokens.return_value.total_tokens = 100_000
    client.models.generate_content.return_value.text = "The answer is 42."

    # Act
    answer = module.generate_answer(QUESTION)

    # Assert
    assert answer.text == "The answer is 42."
    client.models.generate_content.assert_called_once()


def test_generate_answer_token_count_over_limit_raises_project_too_large(ai_service, monkeypatch):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 100_000))
    client.models.count_tokens.return_value.total_tokens = 100_001

    # Act / Assert
    with pytest.raises(module.ProjectTooLargeError) as exc_info:
        module.generate_answer(QUESTION)

    assert exc_info.value.token_count == 100_001
    assert exc_info.value.limit == 100_000
    assert exc_info.value.estimated is False
    client.models.generate_content.assert_not_called()


def test_generate_answer_precheck_over_budget_raises_estimated(ai_service, monkeypatch):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 1))

    # Act / Assert
    with pytest.raises(module.ProjectTooLargeError) as exc_info:
        module.generate_answer(QUESTION)

    assert exc_info.value.estimated is True
    client.models.count_tokens.assert_not_called()
    client.models.generate_content.assert_not_called()


def test_generate_answer_count_tokens_client_error_raises_upstream_rejected(ai_service):

    # Arrange
    module, client = ai_service
    client.models.count_tokens.side_effect = errors.ClientError(
        400,
        {"message": "bad request", "status": "INVALID_ARGUMENT"},
    )

    # Act / Assert
    with pytest.raises(module.UpstreamRejectedError) as exc_info:
        module.generate_answer(QUESTION)

    assert isinstance(exc_info.value.__cause__, errors.ClientError)
    client.models.generate_content.assert_not_called()


def test_generate_answer_count_tokens_server_error_raises_upstream_unavailable(ai_service):

    # Arrange
    module, client = ai_service
    client.models.count_tokens.side_effect = errors.ServerError(
        500,
        {"message": "unavailable", "status": "UNAVAILABLE"},
    )

    # Act / Assert
    with pytest.raises(module.UpstreamUnavailableError) as exc_info:
        module.generate_answer(QUESTION)

    assert isinstance(exc_info.value.__cause__, errors.ServerError)
    client.models.generate_content.assert_not_called()


def test_generate_answer_count_tokens_api_error_raises_upstream_unavailable(ai_service):

    # Arrange
    module, client = ai_service
    client.models.count_tokens.side_effect = errors.APIError(
        302,
        {"message": "redirected", "status": "FOUND"},
    )

    # Act / Assert
    with pytest.raises(module.UpstreamUnavailableError) as exc_info:
        module.generate_answer(QUESTION)

    assert type(exc_info.value.__cause__) is errors.APIError
    client.models.generate_content.assert_not_called()


def test_generate_answer_count_tokens_none_raises_empty_upstream_response(ai_service):

    # Arrange
    module, client = ai_service
    client.models.count_tokens.return_value.total_tokens = None

    # Act / Assert
    with pytest.raises(module.EmptyUpstreamResponseError):
        module.generate_answer(QUESTION)

    client.models.generate_content.assert_not_called()


def test_generate_answer_token_log_omits_prompt_and_api_key(ai_service, caplog, tmp_path):

    # Arrange
    module, client = ai_service
    file_marker = "FILE_MARKER_DO_NOT_LOG_7c2e"
    (tmp_path / "main.c").write_text(f"int {file_marker} = 1;\n", encoding="utf-8")
    client.models.count_tokens.return_value.total_tokens = 100
    client.models.generate_content.return_value.text = "ok"
    caplog.set_level(logging.INFO)

    # Act
    module.generate_answer(QUESTION)

    # Assert
    limit = str(module.settings.max_prompt_tokens)
    assert any("100" in record.message and limit in record.message for record in caplog.records)
    assert file_marker not in caplog.text
    assert module.settings.gemini_api_key not in caplog.text


def test_generate_answer_prompt_contains_project_map(ai_service, tmp_path):

    # Arrange
    module, client = ai_service
    (tmp_path / "main.c").write_text(
        "int f(void)\n{\n    return 1;\n}\n",
        encoding="utf-8",
    )
    client.models.generate_content.return_value.text = "ok"

    # Act
    module.generate_answer(QUESTION)

    # Assert
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert "Project map (full)" in contents
    assert "f 1-4" in contents


def test_generate_answer_char_route_indexes_then_estimates(ai_service, monkeypatch):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 1))
    calls = []
    real_get = module.index_cache.get

    def spy(project_path, files):
        calls.append((project_path, files))
        return real_get(project_path, files)

    monkeypatch.setattr(module.index_cache, "get", spy)

    # Act / Assert
    with pytest.raises(module.ProjectTooLargeError) as exc_info:
        module.generate_answer(QUESTION)

    assert exc_info.value.estimated is True
    assert len(calls) == 1
    client.models.count_tokens.assert_not_called()


def _token_response(total):
    return type("TokenResponse", (), {"total_tokens": total})()


def test_generate_answer_drops_map_when_over_budget(ai_service, monkeypatch):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 100_000))
    client.models.count_tokens.side_effect = [
        _token_response(100_001),
        _token_response(100),
    ]
    client.models.generate_content.return_value.text = "ok"

    # Act
    module.generate_answer(QUESTION)

    # Assert
    assert client.models.count_tokens.call_count == 2
    client.models.generate_content.assert_called_once()
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert "Project map" not in contents


def test_generate_answer_full_counts_over_uses_partial(ai_service, monkeypatch):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 100_000))
    client.models.count_tokens.side_effect = [
        _token_response(100_001),
        _token_response(100_001),
        _token_response(100),
    ]
    client.models.generate_content.return_value.text = "ok"

    # Act
    answer = module.generate_answer(QUESTION)

    # Assert
    assert client.models.count_tokens.call_count == 3
    client.models.generate_content.assert_called_once()
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert "Partial context:" in contents
    assert answer.context.mode == "partial"


def test_generate_answer_compacts_map_within_char_cap(ai_service, monkeypatch, tmp_path):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 2_000))
    lines = [f"#define M_{number} {number}" for number in range(200)]
    lines.append("void keep_me(void)\n{\n    return;\n}")
    (tmp_path / "main.c").write_text("\n".join(lines) + "\n", encoding="utf-8")
    client.models.generate_content.return_value.text = "ok"

    # Act
    module.generate_answer(QUESTION)

    # Assert
    contents = client.models.generate_content.call_args.kwargs["contents"]
    # The source file still contains every #define. The map section is the
    # part before the first file header, and that section drops the macros.
    map_section = contents.split("========================", 1)[0]
    assert "M_150" not in map_section
    assert "keep_me" in map_section


def test_generate_answer_logs_map_tier_without_map_text(ai_service, caplog, tmp_path):

    # Arrange
    module, client = ai_service
    (tmp_path / "main.c").write_text(
        "int UNIQUE_MAP_TEXT_zz9 = 1;\n",
        encoding="utf-8",
    )
    client.models.generate_content.return_value.text = "ok"
    caplog.set_level(logging.INFO)

    # Act
    module.generate_answer(QUESTION)

    # Assert
    contents = client.models.generate_content.call_args.kwargs["contents"]
    start = contents.index("Project map (")
    end = contents.index("\n\nProject context:")
    map_text = contents[start:end]
    assert any(
        record.levelno == logging.INFO and "Project map: tier" in record.message
        for record in caplog.records
    )
    assert all(map_text not in record.message for record in caplog.records)
    assert "UNIQUE_MAP_TEXT_zz9" not in caplog.text


def test_generate_answer_unchanged_project_builds_index_once(ai_service, monkeypatch):

    # Arrange
    module, client = ai_service
    index_calls = []
    graph_calls = []

    def spy_index(project_files):
        index_calls.append(tuple(project_files))
        return build_index(project_files)

    def spy_graph(index):
        graph_calls.append(index)
        return build_graph(index)

    monkeypatch.setattr("app.index_cache.build_index", spy_index)
    monkeypatch.setattr("app.index_cache.build_graph", spy_graph)
    client.models.generate_content.return_value.text = "ok"

    # Act
    module.generate_answer(QUESTION)
    first = client.models.generate_content.call_args.kwargs["contents"]
    module.generate_answer(QUESTION)
    second = client.models.generate_content.call_args.kwargs["contents"]

    # Assert
    assert len(index_calls) == 1
    assert len(graph_calls) == 1
    assert first == second


def test_generate_answer_edited_file_updates_map(ai_service, tmp_path):

    # Arrange
    module, client = ai_service
    client.models.generate_content.return_value.text = "ok"
    module.generate_answer(QUESTION)
    (tmp_path / "main.c").write_text(
        "int g(void)\n{\n    return 2;\n}\n",
        encoding="utf-8",
    )

    # Act
    module.generate_answer(QUESTION)

    # Assert
    contents = client.models.generate_content.call_args.kwargs["contents"]
    map_section = contents.split("========================", 1)[0]
    assert "g 1-4" in map_section


def test_generate_answer_over_budget_logs_warning(ai_service, monkeypatch, caplog):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 100_000))
    client.models.count_tokens.return_value.total_tokens = 100_001
    caplog.set_level(logging.WARNING)

    # Act / Assert
    with pytest.raises(module.ProjectTooLargeError):
        module.generate_answer(QUESTION)

    assert any(
        record.levelno == logging.WARNING
        and record.name == "app.ai_service"
        and record.message == "Prompt is 100001 tokens (limit 100000)."
        for record in caplog.records
    )


def _write_partial_project(tmp_path, *, header: bool = False) -> str:
    main = tmp_path / "main.c"
    if main.exists():
        main.unlink()
    source = tmp_path / "src"
    source.mkdir(parents=True)
    can_text = "void Can_Init(void)\n{\n    return;\n}\n"
    (source / "can.c").write_text(can_text, encoding="utf-8")
    if header:
        include = tmp_path / "inc"
        include.mkdir()
        (include / "can.h").write_text(
            "/* HEADER_ONLY_CAN_H */\nvoid Can_Init(void);\n",
            encoding="utf-8",
        )
    marker = "FILLER_TEXT_SHOULD_NOT_APPEAR_zz"
    filler = ""
    while len(filler) < 60_000:
        filler += f"/* {marker} */\n"
    (source / "pad_a.c").write_text(filler, encoding="utf-8")
    (source / "pad_b.c").write_text(filler, encoding="utf-8")
    return can_text


def test_generate_answer_char_route_uses_partial_context(ai_service, monkeypatch, tmp_path):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 10_000))
    can_text = _write_partial_project(tmp_path)
    client.models.generate_content.return_value.text = "stub answer"

    # Act
    answer = module.generate_answer("What does Can_Init do?")

    # Assert
    client.models.count_tokens.assert_called_once()
    client.models.generate_content.assert_called_once()
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert "Partial context:" in contents
    assert can_text in contents
    assert "FILLER_TEXT_SHOULD_NOT_APPEAR_zz" not in contents
    assert answer.context.mode == "partial"
    assert answer.context.files_included == ("src/can.c",)
    assert answer.context.files_total == 3
    assert answer.context.map_tier == "focused"
    assert answer.text.rstrip().endswith("- src/can.c")


def test_generate_answer_full_mode_context_matches_model_text(ai_service):

    # Arrange
    module, client = ai_service
    client.models.count_tokens.return_value.total_tokens = 100
    client.models.generate_content.return_value.text = "The answer is 42."

    # Act
    answer = module.generate_answer(QUESTION)

    # Assert
    assert answer.context.mode == "full"
    assert answer.context.files_included == ("main.c",)
    assert answer.text == "The answer is 42."
    assert answer.context.prompt_tokens == 100


def test_generate_answer_shrink_drops_lowest_ranked_file(ai_service, monkeypatch, tmp_path):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 10_000))
    can_text = _write_partial_project(tmp_path, header=True)
    client.models.count_tokens.side_effect = [
        _token_response(10_001),
        _token_response(100),
    ]
    client.models.generate_content.return_value.text = "stub answer"

    # Act
    answer = module.generate_answer("What does Can_Init do?")

    # Assert
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert can_text in contents
    assert "HEADER_ONLY_CAN_H" not in contents
    assert answer.context.files_included == ("src/can.c",)


def test_generate_answer_shrink_still_over_raises(ai_service, monkeypatch, tmp_path):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 10_000))
    _write_partial_project(tmp_path, header=True)
    client.models.count_tokens.side_effect = [
        _token_response(10_001),
        _token_response(10_001),
    ]

    # Act / Assert
    with pytest.raises(module.ProjectTooLargeError) as exc_info:
        module.generate_answer("What does Can_Init do?")

    assert exc_info.value.token_count == 10_001
    assert exc_info.value.estimated is False
    client.models.generate_content.assert_not_called()


def test_generate_answer_partial_log_omits_question_and_source(ai_service, monkeypatch, tmp_path, caplog):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 10_000))
    question = "What does Can_Init do?"
    can_text = _write_partial_project(tmp_path)
    client.models.generate_content.return_value.text = "stub answer"
    caplog.set_level(logging.INFO)

    # Act
    module.generate_answer(question)

    # Assert
    assert any(
        record.levelno == logging.INFO and "Partial context:" in record.message
        for record in caplog.records
    )
    assert all(question not in record.message for record in caplog.records)
    assert all(can_text not in record.message for record in caplog.records)


def test_generate_answer_partial_empty_text_raises(ai_service, monkeypatch, tmp_path):

    # Arrange
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 10_000))
    _write_partial_project(tmp_path)
    client.models.generate_content.return_value.text = ""

    # Act / Assert
    with pytest.raises(module.EmptyUpstreamResponseError):
        module.generate_answer("What does Can_Init do?")


def test_warm_up_indexes_loaded_files_once(ai_service, monkeypatch):

    # Arrange
    from app.project_loader import load_project_files

    module, _client = ai_service
    calls = []
    real_get = module.index_cache.get

    def spy(project_path, files):
        calls.append((project_path, list(files)))
        return real_get(project_path, files)

    monkeypatch.setattr(module.index_cache, "get", spy)

    # Act
    module.warm_up()

    # Assert
    loaded = load_project_files(module.settings.project_path, module.settings.exclude_dirs)
    assert len(calls) == 1
    assert calls[0][0] == module.settings.project_path
    assert [(item.path, item.content) for item in calls[0][1]] == [
        (item.path, item.content) for item in loaded
    ]


def test_warm_up_invalid_path_logs_warning(ai_service, monkeypatch, caplog, tmp_path):

    # Arrange
    module, _client = ai_service
    missing = tmp_path / "does_not_exist"
    monkeypatch.setattr(
        module,
        "settings",
        Settings(
            project_path=missing,
            gemini_api_key="test-key-DO-NOT-LOG",
            gemini_model="test-model",
        ),
    )
    caplog.set_level(logging.WARNING)

    # Act
    module.warm_up()

    # Assert
    assert any(
        record.levelno == logging.WARNING
        and record.message == f"Index warm-up skipped: invalid project path {missing}"
        for record in caplog.records
    )
