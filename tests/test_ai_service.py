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
