import logging

import pytest

from tests.test_ai_service import _settings_with_limit

LIMIT = 10_000
QUESTION = "What does Widget_Init do?"


def _write_many_files(tmp_path, count: int) -> None:
    (tmp_path / "main.c").unlink(missing_ok=True)
    source = tmp_path / "src"
    source.mkdir(parents=True, exist_ok=True)
    for number in range(count):
        body = f"void Widget_Init_{number}(void)\n{{\n    return;\n}}\n"
        filler = "".join(f"/* pad {number} {line} */\n" for line in range(40))
        (source / f"widget_{number:02d}.c").write_text(body + filler, encoding="utf-8")
    pad = "/* BIG_PAD_FILE */\n" * 6_000
    (source / "zz_big.c").write_text(pad, encoding="utf-8")


class ScriptedCounter:
    """Token counter that returns the next scripted value; the last one repeats."""

    def __init__(self, *counts: int):
        self.counts = counts
        self.calls = 0

    def __call__(self, prompt: str) -> int:
        value = self.counts[min(self.calls, len(self.counts) - 1)]
        self.calls += 1
        return value


def _prepare(module, monkeypatch, tmp_path, count: int, counter):
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, LIMIT))
    _write_many_files(tmp_path, count)
    return module._prepare(QUESTION, counter, "count")


def _baseline(module, monkeypatch, tmp_path, count: int):
    counter = ScriptedCounter(LIMIT)
    prepared = _prepare(module, monkeypatch, tmp_path, count, counter)
    assert prepared.mode == "partial"
    assert counter.calls == 1
    return prepared


def test_s33a_1_one_step_is_enough_and_behaves_as_before(ai_service, monkeypatch, tmp_path):
    module, _client = ai_service
    baseline = _baseline(module, monkeypatch, tmp_path, 6)
    assert len(baseline.included) >= 4

    counter = ScriptedCounter(LIMIT + 1, LIMIT)
    prepared = _prepare(module, monkeypatch, tmp_path, 6, counter)

    assert counter.calls == 2
    assert prepared.mode == "partial"
    assert prepared.included == baseline.included[:-1]
    assert prepared.token_count == LIMIT


def test_s33a_2_two_or_three_rounds_drop_one_file_each_and_fit(ai_service, monkeypatch, tmp_path):
    module, _client = ai_service
    baseline = _baseline(module, monkeypatch, tmp_path, 6)
    assert len(baseline.included) >= 4

    counter = ScriptedCounter(LIMIT + 1, LIMIT + 1, LIMIT + 1, LIMIT)
    prepared = _prepare(module, monkeypatch, tmp_path, 6, counter)

    assert counter.calls == 4
    assert prepared.mode == "partial"
    assert prepared.included == baseline.included[:-3]
    assert prepared.token_count == LIMIT
    assert len(prepared.included) < len(baseline.included)


def test_s33a_3_each_round_works_from_the_latest_count(ai_service, monkeypatch, tmp_path):
    module, _client = ai_service
    baseline = _baseline(module, monkeypatch, tmp_path, 6)

    counter = ScriptedCounter(LIMIT + 1_000, LIMIT + 1, LIMIT)
    prepared = _prepare(module, monkeypatch, tmp_path, 6, counter)

    assert counter.calls == 3
    assert len(prepared.included) < len(baseline.included) - 1
    assert prepared.included == baseline.included[: len(prepared.included)]


def test_s33a_4_never_fits_raises_when_the_files_run_out(ai_service, monkeypatch, tmp_path, caplog):
    module, _client = ai_service
    baseline = _baseline(module, monkeypatch, tmp_path, 3)
    assert len(baseline.included) == 3
    caplog.set_level(logging.INFO)

    counter = ScriptedCounter(LIMIT + 5)
    with pytest.raises(module.ProjectTooLargeError) as exc_info:
        _prepare(module, monkeypatch, tmp_path, 3, counter)

    assert exc_info.value.token_count == LIMIT + 5
    assert exc_info.value.limit == LIMIT
    assert exc_info.value.estimated is False
    assert counter.calls == 3
    assert any(
        record.levelno == logging.WARNING
        and record.message == f"Prompt is {LIMIT + 5} tokens (limit {LIMIT})."
        for record in caplog.records
    )
