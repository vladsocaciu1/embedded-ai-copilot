import dataclasses

from app.prompts import assistant_prompt, learning_instructions, learning_prompt
from tests.test_ai_service import _settings_with_limit, _write_partial_project
from tests.test_main import _partial, api_client  # noqa: F401

FOOTER_TAIL = "The included files are listed in context.files_included."


def _learning(module, limit=None):
    base = module.settings if limit is None else _settings_with_limit(module, limit)
    return dataclasses.replace(base, learning_mode=True)


def test_l6_default_sends_the_unchanged_instructions(ai_service):
    module, client = ai_service
    client.models.generate_content.return_value.text = "ok"
    module.generate_answer("What is answer?")
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert contents.startswith(assistant_prompt + "\n\n")
    assert learning_prompt not in contents
    assert "Learning mode." not in contents


def test_l7_full_mode_learning_sends_the_learning_rules_first(ai_service, monkeypatch):
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _learning(module))
    client.models.generate_content.return_value.text = "ok"
    answer = module.generate_answer("What is answer?")
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert contents.startswith(learning_instructions + "\n\n")
    assert contents.count(learning_prompt) == 1
    assert answer.context.mode == "full"
    assert answer.context.files_included == ("main.c",)
    assert not hasattr(answer.context, "learning")


def test_l8_partial_learning_keeps_notice_and_footer_and_paths_out_of_the_answer(
    ai_service, monkeypatch, tmp_path
):
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _learning(module, 10_000))
    _write_partial_project(tmp_path)
    client.models.generate_content.return_value.text = "stub answer"
    answer = module.generate_answer("What does Can_Init do?")
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert contents.startswith(learning_instructions + "\n\n")
    assert "Partial context: the project has 3 files" in contents
    assert answer.context.mode == "partial"
    assert answer.context.files_included == ("src/can.c",)
    assert answer.text.endswith(
        "\n\n---\n"
        "Partial context: this answer is based on 1 of 3 project files, because the "
        "whole project does not fit the 10000-token prompt budget. " + FOOTER_TAIL
    )
    assert "Files included:" not in answer.text
    assert "\n- " not in answer.text


def test_l9_partial_budget_subtracts_the_active_instructions(ai_service, monkeypatch, tmp_path):
    module, client = ai_service
    _write_partial_project(tmp_path)
    client.models.generate_content.return_value.text = "ok"
    budgets = []
    real_pack = module.pack_files

    def spy(ranked, files, budget):
        budgets.append(budget)
        return real_pack(ranked, files, budget)

    monkeypatch.setattr(module, "pack_files", spy)
    monkeypatch.setattr(module, "settings", _settings_with_limit(module, 10_000))
    module.generate_answer("What does Can_Init do?")
    monkeypatch.setattr(module, "settings", _learning(module, 10_000))
    module.generate_answer("What does Can_Init do?")
    assert len(budgets) == 2
    assert budgets[0] - budgets[1] == len(learning_prompt) + 2


def test_l10_context_reports_learning_only_when_on(api_client, ai_service, monkeypatch, tmp_path):
    module = _partial(ai_service, monkeypatch, tmp_path)
    off = api_client.post("/context", json={"question": "What does Can_Init do?"}).json()
    assert "learning" not in off
    monkeypatch.setattr(module, "settings", _learning(module, 10_000))
    on = api_client.post("/context", json={"question": "What does Can_Init do?"}).json()
    assert on["learning"] is True
    assert on["files_included"] == off["files_included"] == ["src/can.c"]
    assert on["mode"] == "partial"
    assert on["prompt_chars"] - off["prompt_chars"] == len(learning_prompt) + 2


def test_l11_ask_response_has_no_new_field(api_client, ai_service, monkeypatch):
    module, client = ai_service
    monkeypatch.setattr(module, "settings", _learning(module))
    client.models.generate_content.return_value.text = "ok"
    body = api_client.post("/ask", json={"question": "What is answer?"}).json()
    assert set(body) == {"answer", "context"}
    assert set(body["context"]) == {
        "mode", "files_total", "files_included", "prompt_tokens", "map_tier",
    }
