import json
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from google import genai
from google.genai import errors, types

from app import gemini_retry

_RETRY_INFO = "type.googleapis.com/google.rpc.RetryInfo"


@pytest.fixture
def recorded(monkeypatch):
    delays: list[float] = []
    monkeypatch.setattr(gemini_retry, "sleep", lambda delay: delays.append(delay))
    return delays


def _api_error(code: int, body: dict, response: httpx.Response | None = None):
    if 400 <= code < 500:
        return errors.ClientError(code, body, response)
    if 500 <= code < 600:
        return errors.ServerError(code, body, response)
    return errors.APIError(code, body, response)


def _flat(delay: object | None, *, include_delay: bool = True) -> errors.ClientError:
    entry: dict[str, object] = {"@type": _RETRY_INFO}
    if include_delay:
        entry["retryDelay"] = delay
    body: dict[str, object] = {
        "message": "rate limit",
        "status": "RESOURCE_EXHAUSTED",
        "details": [entry],
    }
    return errors.ClientError(429, body)


def _nested(delay: object | None, *, include_delay: bool = True) -> errors.APIError:
    entry: dict[str, object] = {"@type": _RETRY_INFO}
    if include_delay:
        entry["retryDelay"] = delay
    response = httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "Resource exhausted",
                "status": "RESOURCE_EXHAUSTED",
                "details": [entry],
            }
        },
    )
    try:
        errors.APIError.raise_for_response(response)
    except errors.APIError as error:
        return error
    raise AssertionError("expected an APIError")


def _with_header(value: str, *, delay: str | None = None) -> errors.ClientError:
    details: list[dict[str, str]] = []
    if delay is not None:
        details.append({"@type": _RETRY_INFO, "retryDelay": delay})
    return errors.ClientError(
        429,
        {"message": "rate limit", "status": "RESOURCE_EXHAUSTED", "details": details},
        httpx.Response(429, headers={"Retry-After": value}),
    )


def _run(outcomes: list[object], budget: gemini_retry.RetryBudget, what: str = "count_tokens"):
    calls: list[object] = []

    def fn():
        item = outcomes[len(calls)]
        calls.append(item)
        if isinstance(item, BaseException):
            raise item
        return item

    return gemini_retry.call_with_retries(fn, budget, what), calls


def test_r1_success_on_first_try(recorded):
    # Arrange
    budget = gemini_retry.RetryBudget()

    # Act
    result, calls = _run(["ok"], budget)

    # Assert
    assert result == "ok"
    assert len(calls) == 1
    assert recorded == []
    assert budget.remaining == gemini_retry.REQUEST_SLEEP_BUDGET_S


def test_r2_two_429s_then_success(recorded, monkeypatch):
    # Arrange
    monkeypatch.setattr(gemini_retry, "rand", lambda: 0.5)
    budget = gemini_retry.RetryBudget()
    outcomes = [
        _api_error(429, {"message": "rate limit", "status": "RESOURCE_EXHAUSTED"}),
        _api_error(429, {"message": "rate limit", "status": "RESOURCE_EXHAUSTED"}),
        "ok",
    ]

    # Act
    result, calls = _run(outcomes, budget)

    # Assert
    assert result == "ok"
    assert len(calls) == 3
    assert recorded == [0.5, 1.0]
    assert budget.remaining == 18.5


def test_r3_backoff_cap(recorded, monkeypatch):
    # Arrange
    monkeypatch.setattr(gemini_retry, "MAX_ATTEMPTS", 6)
    monkeypatch.setattr(gemini_retry, "rand", lambda: 1.0)
    budget = gemini_retry.RetryBudget(100)
    error = _api_error(503, {"message": "unavailable", "status": "UNAVAILABLE"})

    # Act / Assert
    with pytest.raises(errors.ServerError) as exc_info:
        _run([error, error, error, error, error, error], budget)

    assert exc_info.value is error
    assert recorded == [1, 2, 4, 8, 8]


def test_r4_full_jitter(monkeypatch):
    # Arrange
    outcomes = [
        _api_error(503, {"message": "unavailable", "status": "UNAVAILABLE"}),
        _api_error(503, {"message": "unavailable", "status": "UNAVAILABLE"}),
        "ok",
    ]
    zero: list[float] = []
    monkeypatch.setattr(gemini_retry, "sleep", lambda delay: zero.append(delay))
    monkeypatch.setattr(gemini_retry, "rand", lambda: 0.0)

    # Act
    _run(outcomes, gemini_retry.RetryBudget())

    # Assert
    assert zero == [0.0, 0.0]

    # Arrange
    almost: list[float] = []
    monkeypatch.setattr(gemini_retry, "sleep", lambda delay: almost.append(delay))
    monkeypatch.setattr(gemini_retry, "rand", lambda: 0.999)

    # Act
    _run(outcomes, gemini_retry.RetryBudget())

    # Assert
    assert almost[0] < 1.0
    assert almost[1] < 2.0
    assert almost == [1.0 * 0.999, 2.0 * 0.999]


def test_r5_retry_info_replaces_backoff(recorded, monkeypatch):
    # Arrange
    monkeypatch.setattr(gemini_retry, "rand", lambda: 0.5)
    budget = gemini_retry.RetryBudget()

    # Act
    result, calls = _run([_flat("17s"), "ok"], budget)

    # Assert
    assert result == "ok"
    assert len(calls) == 2
    assert recorded == [17.0]


def test_r6_retry_info_parsing():
    # Assert
    assert gemini_retry.retry_hint(_flat("17s")) == 17
    assert gemini_retry.retry_hint(_nested("17s")) == 17
    assert gemini_retry.retry_hint(_flat("0.5s")) == 0.5
    assert gemini_retry.retry_hint(_nested("0.5s")) == 0.5
    for builder in (_flat, _nested):
        assert gemini_retry.retry_hint(builder(None, include_delay=False)) is None
        assert gemini_retry.retry_hint(builder("abc")) is None
        assert gemini_retry.retry_hint(builder("17")) is None
        assert gemini_retry.retry_hint(builder("-1s")) is None
        assert gemini_retry.retry_hint(builder(17)) is None
    flat_not_list = errors.ClientError(
        429,
        {"message": "rate limit", "status": "RESOURCE_EXHAUSTED", "details": "nope"},
    )
    nested_response = httpx.Response(
        429,
        json={"error": {"code": 429, "message": "nope", "status": "RESOURCE_EXHAUSTED", "details": {}}},
    )
    try:
        errors.APIError.raise_for_response(nested_response)
    except errors.APIError as nested_not_list:
        assert gemini_retry.retry_hint(flat_not_list) is None
        assert gemini_retry.retry_hint(nested_not_list) is None
    else:
        raise AssertionError("expected an APIError")


def test_r7_retry_after_parsing(monkeypatch):
    # Arrange
    fixed = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(gemini_retry, "now", lambda: fixed)
    later = format_datetime(fixed + timedelta(seconds=30), usegmt=True)
    past = format_datetime(fixed - timedelta(seconds=10), usegmt=True)

    # Assert
    assert gemini_retry.retry_hint(_with_header("5")) == 5
    assert gemini_retry.retry_hint(_with_header(later)) == 30
    assert gemini_retry.retry_hint(_with_header(past)) == 0
    assert gemini_retry.retry_hint(_with_header("not-a-date")) is None
    assert gemini_retry.retry_hint(_with_header("5", delay="17s")) == 17


def test_r8_hint_over_budget(recorded):
    # Arrange
    budget = gemini_retry.RetryBudget()
    error = _flat("3600s")

    # Act / Assert
    with pytest.raises(errors.ClientError) as exc_info:
        _run([error], budget)

    assert exc_info.value is error
    assert recorded == []
    assert gemini_retry.retry_after_seconds(exc_info.value) == 3600
    assert budget.remaining == gemini_retry.REQUEST_SLEEP_BUDGET_S


def test_r9_budget_exhaustion(recorded, monkeypatch):
    # Arrange
    monkeypatch.setattr(gemini_retry, "rand", lambda: 0.5)
    budget = gemini_retry.RetryBudget(0.4)
    error = _api_error(503, {"message": "unavailable", "status": "UNAVAILABLE"})

    # Act / Assert
    with pytest.raises(errors.ServerError) as exc_info:
        _run([error], budget)

    assert exc_info.value is error
    assert recorded == []
    assert budget.remaining == 0.4


def test_r10_not_retried(recorded):
    # Arrange
    cases = [
        errors.ClientError(400, {"message": "bad request", "status": "INVALID_ARGUMENT"}),
        errors.ClientError(401, {"message": "unauthorized", "status": "UNAUTHENTICATED"}),
        errors.ClientError(403, {"message": "forbidden", "status": "PERMISSION_DENIED"}),
        errors.ClientError(404, {"message": "missing", "status": "NOT_FOUND"}),
        errors.APIError(302, {"message": "redirected", "status": "FOUND"}),
    ]

    # Act / Assert
    for error in cases:
        budget = gemini_retry.RetryBudget()
        with pytest.raises(type(error)) as exc_info:
            _run([error], budget)
        assert exc_info.value is error
    assert recorded == []

    boom = RuntimeError("boom")
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise boom

    with pytest.raises(RuntimeError) as exc_info:
        gemini_retry.call_with_retries(fn, gemini_retry.RetryBudget(), "count_tokens")
    assert exc_info.value is boom
    assert calls["n"] == 1


def test_r11_retried_codes_and_transport(recorded):
    # Arrange
    factories = [
        lambda: errors.ServerError(500, {"message": "down", "status": "INTERNAL"}),
        lambda: errors.ServerError(502, {"message": "down", "status": "BAD_GATEWAY"}),
        lambda: errors.ServerError(503, {"message": "down", "status": "UNAVAILABLE"}),
        lambda: errors.ServerError(504, {"message": "down", "status": "DEADLINE_EXCEEDED"}),
        lambda: httpx.ConnectError("connection refused"),
        lambda: httpx.ReadTimeout("read timed out"),
    ]

    # Act / Assert
    for factory in factories:
        raised: list[BaseException] = []

        def fn(factory=factory, raised=raised):
            error = factory()
            raised.append(error)
            raise error

        with pytest.raises(type(factory())) as exc_info:
            gemini_retry.call_with_retries(
                fn, gemini_retry.RetryBudget(), "generate_content"
            )
        assert len(raised) == 3
        assert exc_info.value is raised[-1]
        assert type(exc_info.value) is type(raised[-1])


def test_r12_retry_after_seconds():
    # Assert
    assert gemini_retry.retry_after_seconds(_flat("1.2s")) == 2
    assert gemini_retry.retry_after_seconds(_flat("0.5s")) == 1
    assert gemini_retry.retry_after_seconds(_with_header("0")) == 1
    assert (
        gemini_retry.retry_after_seconds(
            errors.ClientError(429, {"message": "rate limit", "status": "RESOURCE_EXHAUSTED"})
        )
        == 60
    )


def test_r13_logs_one_warning_per_retry(recorded, caplog):
    # Arrange
    prompt = "PROMPT_TEXT_DO_NOT_LOG_9f3a"
    secret = "test-key-DO-NOT-LOG"
    caplog.set_level(logging.DEBUG)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise errors.ClientError(
                429,
                {
                    "message": prompt,
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [{"secret": secret}],
                },
            )
        return prompt

    # Act
    result = gemini_retry.call_with_retries(
        fn, gemini_retry.RetryBudget(), "count_tokens"
    )

    # Assert
    assert result == prompt
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == "app.gemini_retry"
    ]
    assert len(warnings) == 1
    assert "retry 1 of 2" in warnings[0].message
    assert "429" in warnings[0].message
    assert prompt not in caplog.text
    assert secret not in caplog.text


def test_r14_production_defaults():
    # Assert
    assert gemini_retry.sleep is time.sleep
    assert gemini_retry.MAX_ATTEMPTS == 3
    assert gemini_retry.REQUEST_SLEEP_BUDGET_S == 20.0


def test_r15_real_sdk_local_stub(monkeypatch):
    # Arrange
    hits: list[int] = []
    sleeps: list[float] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length:
                self.rfile.read(length)
            hits.append(1)
            body = json.dumps(
                {
                    "error": {
                        "code": 429,
                        "message": "Resource exhausted",
                        "status": "RESOURCE_EXHAUSTED",
                        "details": [
                            {"@type": _RETRY_INFO, "retryDelay": "1.2s"}
                        ],
                    }
                }
            ).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.setattr(gemini_retry, "sleep", lambda delay: sleeps.append(delay))
    try:
        client = genai.Client(
            api_key="k",
            http_options=types.HttpOptions(base_url=f"http://{host}:{port}"),
        )
        with pytest.raises(errors.ClientError) as exc_info:
            gemini_retry.call_with_retries(
                lambda: client.models.count_tokens(model="gemini-2.0-flash", contents="hi"),
                gemini_retry.RetryBudget(),
                "count_tokens",
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    # Assert
    assert len(hits) == 3
    assert sleeps == [1.2, 1.2]
    assert gemini_retry.retry_after_seconds(exc_info.value) == 2
