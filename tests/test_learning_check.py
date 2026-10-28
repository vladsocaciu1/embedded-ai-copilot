import json
import re
from pathlib import Path

import pytest

from scripts import learning_check
from scripts.learning_check import (
    DEFAULT_CASES,
    check_docs,
    check_names,
    check_next,
    read_next_items,
    score,
    summarize,
)

CASE = {
    "id": 1,
    "question": "What is a mutex?",
    "files": ["kernel/mutex.c"],
    "symbols": ["k_mutex_lock"],
    "docs": ["doc/kernel/services/synchronization/mutexes.rst"],
}

GOOD = """Short answer: a mutex is a lock owned by one thread.

Where it lives: `k_mutex_lock` in kernel/mutex.c:120-180.

Documentation: doc/kernel/services/synchronization/mutexes.rst, section "Locking".

Read next:
1. kernel/mutex.c:200 - unlock.
2. doc/kernel/services/threads/index.rst - threads.

---
Partial context: this answer is based on 4 of 129 project files.
"""


@pytest.fixture
def root(tmp_path):
    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel/mutex.c").write_text("x\n" * 300, encoding="utf-8")
    docs = tmp_path / "doc/kernel/services"
    (docs / "synchronization").mkdir(parents=True)
    (docs / "threads").mkdir()
    (docs / "synchronization/mutexes.rst").write_text("Mutexes\n", encoding="utf-8")
    (docs / "threads/index.rst").write_text("Threads\n", encoding="utf-8")
    return tmp_path


def test_p1_good_answer_passes_a_b_c(root):
    assert check_names(GOOD, CASE, root)
    assert check_docs(GOOD, CASE, root)
    assert check_next(GOOD, root)


def test_p2_names_fails_on_missing_citation_symbol_invented_path_or_late_line(root):
    no_citation = re.sub(r"kernel/mutex\.c:[\d-]+", "kernel/mutex.c", GOOD)
    assert not check_names(no_citation, CASE, root)
    no_symbol = GOOD.replace("k_mutex_lock", "the lock call")
    assert not check_names(no_symbol, CASE, root)
    invented = GOOD + "\nSee kernel/mutex_fast.c:10.\n"
    assert not check_names(invented, CASE, root)
    late = GOOD.replace("120-180", "120-999")
    assert not check_names(late, CASE, root)


def test_p3_docs_fails_without_page_or_with_invented_page(root):
    assert not check_docs(GOOD.replace("mutexes.rst", "locks.rst"), CASE, root)
    other = GOOD.replace("synchronization/mutexes.rst", "synchronization/other.rst")
    assert not check_docs(other, CASE, root)
    none_expected = dict(CASE, docs=[])
    assert check_docs("No documentation page for this was included.", none_expected, root)
    assert not check_docs("See doc/kernel/made_up.rst", none_expected, root)


def test_p4_next_needs_heading_two_or_three_items_and_real_paths(root):
    assert read_next_items(GOOD) == [
        "1. kernel/mutex.c:200 - unlock.",
        "2. doc/kernel/services/threads/index.rst - threads.",
    ]
    assert not check_next("Short answer.", root)
    one = GOOD.replace("2. doc/kernel/services/threads/index.rst - threads.\n", "")
    assert not check_next(one, root)
    four = GOOD.replace(
        "\n---", "3. kernel/mutex.c:1 - a\n4. kernel/mutex.c:2 - b\n\n---"
    )
    assert not check_next(four, root)
    invented = GOOD.replace("kernel/mutex.c:200", "kernel/nothing.c:200")
    assert not check_next(invented, root)
    bullets = GOOD.replace("1. ", "- ").replace("2. ", "* ")
    assert check_next(bullets, root)
    assert check_next(GOOD.replace("Read next:", "5. **Read next**"), root)


def test_p5_score_and_summary_need_human_answers_and_four_of_five(root):
    cases = [dict(CASE, id=number) for number in range(1, 6)]
    answers = [{"id": number, "answer": GOOD} for number in range(1, 6)]
    answers[4]["answer"] = "I do not know."
    rows = score(answers, cases, root, None)
    assert [row["A"] for row in rows] == [True] * 4 + [False]
    line, code = summarize(rows)
    assert code == 2
    assert "human check D still needed" in line
    rows = score(answers, cases, root, [True] * 5)
    line, code = summarize(rows)
    assert code == 0
    assert line.startswith("summary: PASS 4/5 (need 4)")
    rows = score(answers, cases, root, [True, True, True, False, True])
    line, code = summarize(rows)
    assert code == 1
    assert line.startswith("summary: FAIL 3/5")


def test_p6_cases_file_has_five_questions_with_fields():
    cases = json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))
    assert [case["id"] for case in cases] == [1, 2, 3, 4, 5]
    for case in cases:
        assert set(case) == {"id", "question", "files", "symbols", "docs"}
        assert case["files"] and case["symbols"]
    assert [bool(case["docs"]) for case in cases] == [True, True, True, True, False]


def test_p7_run_posts_each_question_and_saves_answers(tmp_path, monkeypatch, capsys):
    asked = []

    def fake_ask(url, question):
        asked.append((url, question))
        return {"answer": "a", "context": {"mode": "partial", "prompt_tokens": 7}}

    monkeypatch.setattr(learning_check, "ask", fake_ask)
    out = tmp_path / "out.json"
    code = learning_check.main(["run", "--url", "http://x:1", "--out", str(out)])
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert [item["id"] for item in saved] == [1, 2, 3, 4, 5]
    assert {url for url, _question in asked} == {"http://x:1"}
    assert len(asked) == 5
    assert "ok 1 partial 7 tokens" in capsys.readouterr().out


def test_p8_run_records_errors_and_exits_one(tmp_path, monkeypatch):
    def fake_ask(url, question):
        raise OSError("refused")

    monkeypatch.setattr(learning_check, "ask", fake_ask)
    out = tmp_path / "out.json"
    code = learning_check.main(["run", "--out", str(out)])
    assert code == 1
    assert json.loads(out.read_text(encoding="utf-8"))[0]["error"] == "refused"


def test_p9_check_command_prints_rows_and_exit_codes(tmp_path, root, capsys):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([CASE]), encoding="utf-8")
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps([{"id": 1, "answer": GOOD}]), encoding="utf-8")
    base = ["check", str(answers), "--root", str(root), "--cases", str(cases)]
    assert learning_check.main(base) == 2
    assert "1 A=ok B=ok C=ok D=-" in capsys.readouterr().out
    assert learning_check.main([*base, "--human", "yes"]) == 0
    with pytest.raises(SystemExit):
        learning_check.main([*base, "--human", "maybe"])
