"""Spec 33 manual check. `run` asks the five questions of a running server;
`check` scores the saved answers with no network. Pass = A, B, C and D on 80% of them.
A names an expected file (path:line) and symbol; every cited path exists under
--root and every cited line is inside its file. B names the expected doc page;
every doc/...rst path in the answer exists (no page expected: passes). C has a
"Read next" heading with 2 or 3 items, each with an existing path. D (human,
--human yes,no,...) says what the thing is for, not only what the code does."""
import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PASS_SHARE = 0.8
TIMEOUT_SECONDS = 300
DEFAULT_CASES = Path(__file__).resolve().parent / "zephyr_learning_cases.json"
CITATION = re.compile(r"([\w+.-]+(?:/[\w+.-]+)*\.\w+):(\d+)(?:-(\d+))?")
PATH_TOKEN = re.compile(r"[\w+-]+(?:/[\w.+-]+)+")
DOC_TOKEN = re.compile(r"doc/[\w+./-]*\.rst")
HEADING = re.compile(r"^\W*(?:\d+\W+)?read next\b", re.IGNORECASE)
ITEM = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")


def _line_count(root: Path, rel: str) -> int | None:
    path = root / rel
    if not path.is_file():
        return None
    return len(path.read_text(errors="replace").splitlines())


def check_names(answer: str, case: dict, root: Path) -> bool:
    cited = CITATION.findall(answer)
    for rel, first, last in cited:
        lines = _line_count(root, rel)
        if lines is None or int(last or first) > lines:
            return False
    named = any(rel in case["files"] for rel, _first, _last in cited)
    return named and any(symbol in answer for symbol in case["symbols"])


def check_docs(answer: str, case: dict, root: Path) -> bool:
    mentioned = set(DOC_TOKEN.findall(answer))
    if any(not (root / rel).is_file() for rel in mentioned):
        return False
    return not case["docs"] or any(rel in mentioned for rel in case["docs"])


def read_next_items(answer: str) -> list[str]:
    lines = answer.splitlines()
    for number, line in enumerate(lines):
        if HEADING.match(line):
            items: list[str] = []
            for rest in lines[number + 1:]:
                if rest.strip() == "---":
                    break
                if ITEM.match(rest):
                    items.append(rest)
            return items
    return []


def check_next(answer: str, root: Path) -> bool:
    items = read_next_items(answer)
    return 2 <= len(items) <= 3 and all(
        any((root / token).is_file() for token in PATH_TOKEN.findall(item))
        for item in items
    )


def score(results: list[dict], cases: list[dict], root: Path, human: list[bool] | None) -> list[dict]:
    rows = []
    for index, (result, case) in enumerate(zip(results, cases)):
        answer = result.get("answer") or ""
        rows.append({
            "id": case["id"],
            "question": case["question"],
            "A": check_names(answer, case, root),
            "B": check_docs(answer, case, root),
            "C": check_next(answer, root),
            "D": None if human is None else human[index],
        })
    return rows


def summarize(rows: list[dict]) -> tuple[str, int]:
    passed = sum(all(row[key] is True for key in "ABCD") for row in rows)
    counts = " ".join(
        f"{key} {sum(row[key] is True for row in rows)}/{len(rows)}" for key in "ABCD"
    )
    if any(row["D"] is None for row in rows):
        return f"summary: automated only ({counts}); human check D still needed", 2
    needed = math.ceil(len(rows) * PASS_SHARE)
    ok = passed >= needed
    return f"summary: {'PASS' if ok else 'FAIL'} {passed}/{len(rows)} (need {needed}) ({counts})", int(not ok)


def ask(url: str, question: str) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/ask",
        data=json.dumps({"question": question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.load(response)


def run(args: argparse.Namespace) -> int:
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    results = []
    failed = 0
    for case in cases:
        entry = {"id": case["id"], "question": case["question"]}
        try:
            body = ask(args.url, case["question"])
            entry["answer"] = body["answer"]
            entry["context"] = body["context"]
            print(f"ok {case['id']} {body['context']['mode']} {body['context']['prompt_tokens']} tokens")
        except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
            entry["error"] = str(error)
            failed += 1
            print(f"error {case['id']} {error}")
        results.append(entry)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 1 if failed else 0


def check(args: argparse.Namespace) -> int:
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    results = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    if [r["id"] for r in results] != [c["id"] for c in cases]:
        raise SystemExit("answers do not match the cases file")
    human = None
    if args.human:
        words = [word.strip().lower() for word in args.human.split(",")]
        if len(words) != len(results) or any(w not in ("yes", "no") for w in words):
            raise SystemExit("--human needs yes or no for each question, comma-separated")
        human = [word == "yes" for word in words]
    rows = score(results, cases, Path(args.root), human)
    for row in rows:
        shown = {None: "-", True: "ok", False: "no"}
        marks = " ".join(f"{key}={shown[row[key]]}" for key in "ABCD")
        print(f"{row['id']} {marks} {row['question']}")
    line, code = summarize(rows)
    print(line)
    return code


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--url", default="http://127.0.0.1:8000")
    runner.add_argument("--cases", default=str(DEFAULT_CASES))
    runner.add_argument("--out", required=True)
    checker = commands.add_parser("check")
    checker.add_argument("answers")
    checker.add_argument("--root", required=True)
    checker.add_argument("--cases", default=str(DEFAULT_CASES))
    checker.add_argument("--human")
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else check(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
