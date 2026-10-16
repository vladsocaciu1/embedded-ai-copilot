from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(ai_service):
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_ask_returns_answer(api_client, monkeypatch):

    # Arrange
    monkeypatch.setattr("app.main.ai_service.generate_answer", lambda question: "hi")

    # Act
    response = api_client.post("/ask", json={"question": "Where is the answer initialized?"})

    # Assert
    assert response.status_code == 200
    assert response.json() == {"answer": "hi"}


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
    generate_answer = MagicMock(return_value="hi")
    monkeypatch.setattr("app.main.ai_service.generate_answer", generate_answer)

    # Act
    response = api_client.post("/ask", json={})

    # Assert
    assert response.status_code == 422
    generate_answer.assert_not_called()


def test_ask_empty_question_returns_422(api_client, monkeypatch):

    # Arrange
    generate_answer = MagicMock(return_value="hi")
    monkeypatch.setattr("app.main.ai_service.generate_answer", generate_answer)

    # Act
    response = api_client.post("/ask", json={"question": ""})

    # Assert
    assert response.status_code == 422
    generate_answer.assert_not_called()


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
