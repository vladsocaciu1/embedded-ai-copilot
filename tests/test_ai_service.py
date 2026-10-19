import logging

import pytest
from google.genai import errors

from app.config import Settings

QUESTION = "Where is the answer initialized?"


def test_generate_answer_returns_response_text(ai_service):

    # Arrange
    module, client = ai_service
    client.models.generate_content.return_value.text = "The answer is 42."

    # Act
    answer = module.generate_answer(QUESTION)

    # Assert
    assert answer == "The answer is 42."


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
    assert answer == "The answer is 42."
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
    assert answer == "The answer is 42."
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
