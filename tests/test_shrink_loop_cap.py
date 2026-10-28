import logging
from unittest.mock import MagicMock

import pytest

from tests.test_shrink_loop import (
    LIMIT,
    QUESTION,
    ScriptedCounter,
    _baseline,
    _prepare,
)


def test_s33a_5_the_loop_stops_at_the_round_cap(ai_service, monkeypatch, tmp_path, caplog):
    module, _client = ai_service
    assert module.MAX_SHRINK_ROUNDS == 8
    baseline = _baseline(module, monkeypatch, tmp_path, 20)
    assert len(baseline.included) > module.MAX_SHRINK_ROUNDS + 1
    caplog.set_level(logging.INFO)

    counter = ScriptedCounter(LIMIT + 1)
    with pytest.raises(module.ProjectTooLargeError) as exc_info:
        _prepare(module, monkeypatch, tmp_path, 20, counter)

    assert exc_info.value.token_count == LIMIT + 1
    assert counter.calls == module.MAX_SHRINK_ROUNDS + 1
    rounds = [r for r in caplog.records if "dropped" in r.message and "round" in r.message]
    assert len(rounds) == module.MAX_SHRINK_ROUNDS
    assert any(
        record.levelno == logging.WARNING
        and record.message == f"Prompt is {LIMIT + 1} tokens (limit {LIMIT})."
        for record in caplog.records
    )


def test_s33a_6_the_last_allowed_round_may_still_fit(ai_service, monkeypatch, tmp_path):
    module, _client = ai_service
    baseline = _baseline(module, monkeypatch, tmp_path, 20)

    counter = ScriptedCounter(*([LIMIT + 1] * module.MAX_SHRINK_ROUNDS), LIMIT)
    prepared = _prepare(module, monkeypatch, tmp_path, 20, counter)

    assert counter.calls == module.MAX_SHRINK_ROUNDS + 1
    assert prepared.included == baseline.included[: -module.MAX_SHRINK_ROUNDS]


def test_s33a_7_ask_and_context_paths_use_the_loop(ai_service, monkeypatch, tmp_path):
    module, client = ai_service
    baseline = _baseline(module, monkeypatch, tmp_path, 6)
    client.models.count_tokens.side_effect = [
        _response(LIMIT + 1), _response(LIMIT + 1), _response(LIMIT),
    ]
    client.models.generate_content.return_value.text = "stub answer"

    answer = module.generate_answer(QUESTION)

    assert client.models.count_tokens.call_count == 3
    assert answer.context.files_included == tuple(
        path.as_posix() for path in baseline.included[:-2]
    )
    client.models.generate_content.assert_called_once()

    client.models.count_tokens.reset_mock()
    client.models.count_tokens.side_effect = [
        _response(LIMIT + 1), _response(LIMIT + 1), _response(LIMIT),
    ]
    prepared = module.dry_run(QUESTION, exact=True)

    assert client.models.count_tokens.call_count == 3
    assert prepared.included == baseline.included[:-2]


def _response(total: int):
    response = MagicMock()
    response.total_tokens = total
    return response
