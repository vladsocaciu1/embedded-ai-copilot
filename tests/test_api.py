from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _answer(text: str):
    from app.models import Answer, AnswerContext

    return Answer(
        text=text,
        context=AnswerContext(
            mode="full",
            files_total=1,
            files_included=("main.c",),
            prompt_tokens=100,
            map_tier="full",
        ),
    )


@pytest.fixture
def api_client(ai_service, monkeypatch):
    from app.main import app

    monkeypatch.setattr("app.main.ai_service.warm_up", lambda: None)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_ask_returns_answer(api_client, monkeypatch):

    # Arrange
    monkeypatch.setattr("app.main.ai_service.generate_answer", lambda question: _answer("hi"))

    # Act
    response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

    # Assert
    assert response.status_code == 200
    assert response.json()["answer"] == "hi"


def test_ask_project_load_error_returns_500(api_client, monkeypatch):

    # Arrange
    from app.ai_service import ProjectLoadError

    monkeypatch.setattr(
        "app.main.ai_service.generate_answer",
        MagicMock(side_effect=ProjectLoadError("project-load-marker")),
    )

    # Act
    response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

    # Assert
    assert response.status_code == 500
    assert response.json()["detail"]


def test_ask_upstream_rejected_returns_502(api_client, monkeypatch):

    # Arrange
    from app.ai_service import UpstreamRejectedError

    monkeypatch.setattr(
        "app.main.ai_service.generate_answer",
        MagicMock(side_effect=UpstreamRejectedError("rejected-marker")),
    )

    # Act
    response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

    # Assert
    assert response.status_code == 502
    assert response.json()["detail"]


def test_ask_upstream_unavailable_returns_503(api_client, monkeypatch):

    # Arrange
    from app.ai_service import UpstreamUnavailableError

    monkeypatch.setattr(
        "app.main.ai_service.generate_answer",
        MagicMock(side_effect=UpstreamUnavailableError("unavailable-marker")),
    )

    # Act
    response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

    # Assert
    assert response.status_code == 503
    assert response.json()["detail"]


def test_ask_runtime_error_returns_500(api_client, monkeypatch):

    # Arrange
    monkeypatch.setattr(
        "app.main.ai_service.generate_answer",
        MagicMock(side_effect=RuntimeError("boom-marker")),
    )

    # Act
    response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

    # Assert
    assert response.status_code == 500


def test_ask_missing_question_returns_422(api_client, monkeypatch):

    # Arrange
    generate_answer = MagicMock(return_value=_answer("hi"))
    monkeypatch.setattr("app.main.ai_service.generate_answer", generate_answer)

    # Act
    response = api_client.post("/ask", json={})

    # Assert
    assert response.status_code == 422
    generate_answer.assert_not_called()


def test_ask_empty_question_returns_422(api_client, monkeypatch):

    # Arrange
    generate_answer = MagicMock(return_value=_answer("hi"))
    monkeypatch.setattr("app.main.ai_service.generate_answer", generate_answer)

    # Act
    response = api_client.post("/ask", json={"question": ""})

    # Assert
    assert response.status_code == 422
    generate_answer.assert_not_called()


def test_ask_whitespace_only_question_returns_422(api_client, monkeypatch):

    # Arrange
    generate_answer = MagicMock(return_value=_answer("hi"))
    monkeypatch.setattr("app.main.ai_service.generate_answer", generate_answer)

    # Act
    response = api_client.post("/ask", json={"question": "   "})

    # Assert
    assert response.status_code == 422
    generate_answer.assert_not_called()


def test_ask_whitespace_newline_tab_question_returns_422(api_client, monkeypatch):

    # Arrange
    generate_answer = MagicMock(return_value=_answer("hi"))
    monkeypatch.setattr("app.main.ai_service.generate_answer", generate_answer)

    # Act
    response = api_client.post("/ask", json={"question": "\n\t "})

    # Assert
    assert response.status_code == 422
    generate_answer.assert_not_called()


def test_ask_padded_question_returns_200_unchanged(api_client, monkeypatch):

    # Arrange
    question = "  What does can.c do?  "
    generate_answer = MagicMock(return_value=_answer("hi"))
    monkeypatch.setattr("app.main.ai_service.generate_answer", generate_answer)

    # Act
    response = api_client.post("/ask", json={"question": question})

    # Assert
    assert response.status_code == 200
    generate_answer.assert_called_once_with(question)


def test_ask_multiline_question_returns_200(api_client, monkeypatch):

    # Arrange
    question = "Line one\nLine two"
    generate_answer = MagicMock(return_value=_answer("hi"))
    monkeypatch.setattr("app.main.ai_service.generate_answer", generate_answer)

    # Act
    response = api_client.post("/ask", json={"question": question})

    # Assert
    assert response.status_code == 200
    generate_answer.assert_called_once_with(question)


def test_ask_error_bodies_omit_exception_message_and_traceback(api_client, monkeypatch):

    # Arrange
    from app.ai_service import ProjectLoadError, UpstreamRejectedError, UpstreamUnavailableError

    errors = [
        ProjectLoadError("project-load-marker"),
        UpstreamRejectedError("rejected-marker"),
        UpstreamUnavailableError("unavailable-marker"),
        RuntimeError("boom-marker"),
    ]

    for error in errors:
        monkeypatch.setattr(
            "app.main.ai_service.generate_answer",
            MagicMock(side_effect=error),
        )

        # Act
        response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

        # Assert
        assert str(error) not in response.text
        assert "Traceback" not in response.text


def test_ask_empty_upstream_response_error_returns_502_without_message(api_client, monkeypatch):

    # Arrange
    from app.ai_service import EmptyUpstreamResponseError

    marker = "empty-upstream-marker"
    monkeypatch.setattr(
        "app.main.ai_service.generate_answer",
        MagicMock(side_effect=EmptyUpstreamResponseError(marker)),
    )

    # Act
    response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

    # Assert
    assert response.status_code == 502
    assert marker not in response.text
    assert "Traceback" not in response.text


def test_ask_project_too_large_returns_413(api_client, monkeypatch):

    # Arrange
    from app.ai_service import ProjectTooLargeError

    monkeypatch.setattr(
        "app.main.ai_service.generate_answer",
        MagicMock(side_effect=ProjectTooLargeError(412345, 200000)),
    )

    # Act
    response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

    # Assert
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "412345" in detail
    assert "200000" in detail


def test_ask_project_too_large_estimated_returns_413(api_client, monkeypatch):

    # Arrange
    from app.ai_service import ProjectTooLargeError

    monkeypatch.setattr(
        "app.main.ai_service.generate_answer",
        MagicMock(side_effect=ProjectTooLargeError(2_500_000, 200000, estimated=True)),
    )

    # Act
    response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

    # Assert
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "2500000" in detail
    assert "200000" in detail


def test_ask_project_too_large_body_omits_traceback(api_client, monkeypatch):

    # Arrange
    from app.ai_service import ProjectTooLargeError

    monkeypatch.setattr(
        "app.main.ai_service.generate_answer",
        MagicMock(side_effect=ProjectTooLargeError(412345, 200000)),
    )

    # Act
    response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

    # Assert
    assert response.status_code == 413
    assert "Traceback" not in response.text


def test_ask_none_text_returns_502_not_null_answer(api_client, ai_service):

    # Arrange
    _module, client = ai_service
    client.models.generate_content.return_value.text = None

    # Act
    response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

    # Assert
    assert response.status_code == 502
    assert response.json() != {"answer": None}


def test_ask_returns_context(api_client, monkeypatch):

    # Arrange
    monkeypatch.setattr(
        "app.main.ai_service.generate_answer",
        lambda question: _answer("hi"),
    )

    # Act
    response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

    # Assert
    assert response.status_code == 200
    assert response.json() == {
        "answer": "hi",
        "context": {
            "mode": "full",
            "files_total": 1,
            "files_included": ["main.c"],
            "prompt_tokens": 100,
            "map_tier": "full",
        },
    }


_CONTEXT_KEYS = {
    "mode",
    "limit",
    "prompt_tokens",
    "token_source",
    "prompt_chars",
    "files_total",
    "files_included",
    "map_tier",
    "map_files",
    "ranked_total",
    "ranked",
}


def test_context_returns_ranked_preview_without_generate(api_client, ai_service, monkeypatch, tmp_path):

    # Arrange
    from tests.test_ai_service import _settings_with_limit, _write_partial_project

    module, client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 10_000))
    _write_partial_project(tmp_path)
    default = tmp_path / "main.c"
    if default.exists():
        default.unlink()

    # Act
    response = api_client.post("/context", json={"question": "What does Can_Init do?"})

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert set(body) == _CONTEXT_KEYS
    assert body["mode"] == "partial"
    assert body["token_source"] == "estimate"
    assert body["ranked_total"] >= len(body["ranked"])
    assert body["ranked"]
    assert len(body["ranked"]) <= 50
    for item in body["ranked"]:
        assert set(item) == {"path", "reason", "included", "focus", "tier", "hits", "lexical"}
        assert isinstance(item["included"], bool)
    client.models.generate_content.assert_not_called()
    client.models.count_tokens.assert_not_called()


def test_context_exact_client_error_matches_ask_502(api_client, ai_service):

    # Arrange
    from google.genai import errors

    _module, client = ai_service
    client.models.count_tokens.side_effect = errors.ClientError(
        400,
        {"message": "bad request", "status": "INVALID_ARGUMENT"},
    )
    question = {"question": "Where is the answer initialized?"}

    # Act
    context_response = api_client.post("/context", params={"exact": True}, json=question)
    ask_response = api_client.post("/ask", json=question)

    # Assert
    assert context_response.status_code == 502
    assert ask_response.status_code == 502
    assert context_response.json()["detail"] == ask_response.json()["detail"]
    assert context_response.json()["detail"] == "AI provider rejected the request."


def test_context_estimated_too_large_matches_ask_413(api_client, ai_service, monkeypatch):

    # Arrange
    from tests.test_ai_service import _settings_with_limit

    module, _client = ai_service
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 1))
    question = {"question": "Where is the answer initialized?"}

    # Act
    context_response = api_client.post("/context", json=question)
    ask_response = api_client.post("/ask", json=question)

    # Assert
    assert context_response.status_code == 413
    assert ask_response.status_code == 413
    detail = context_response.json()["detail"]
    assert detail == ask_response.json()["detail"]
    assert detail.startswith("Project is too large for the prompt budget: at least ")
    assert "tokens (estimated), limit 1." in detail


def test_context_empty_question_returns_422(api_client, ai_service):

    # Arrange
    _module, client = ai_service

    # Act
    response = api_client.post("/context", json={"question": ""})

    # Assert
    assert response.status_code == 422
    client.models.generate_content.assert_not_called()


def test_lifespan_warms_index_in_a_thread(ai_service, monkeypatch):

    # Arrange
    import threading

    from app.main import app

    started = threading.Event()

    def spy() -> None:
        started.set()

    monkeypatch.setattr("app.main.ai_service.warm_up", spy)

    # Act / Assert
    with TestClient(app, raise_server_exceptions=False):
        assert started.wait(5)
