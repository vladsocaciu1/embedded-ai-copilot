import json
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath

_CASE_FILES = {
    "scripts/zephyr_eval_cases.json": 8,
    "scripts/zephyr_semantic_dev.json": 18,
    "scripts/zephyr_semantic_heldout.json": 8,
}
_KEYS = {"question", "expected", "k", "tag"}


def _load(name: str) -> list[dict]:
    return json.loads(Path(name).read_text(encoding="utf-8"))


def test_c1_case_shape():
    for name, count in _CASE_FILES.items():
        cases = _load(name)
        assert isinstance(cases, list)
        assert len(cases) == count
        for case in cases:
            assert set(case) == _KEYS
            assert isinstance(case["question"], str) and case["question"]
            assert isinstance(case["expected"], list) and case["expected"]
            for path in case["expected"]:
                assert isinstance(path, str) and path
                assert not path.startswith("/")
                assert "\\" not in path
                assert PurePosixPath(path).as_posix() == path
            assert isinstance(case["k"], int) and not isinstance(case["k"], bool)
            assert 1 <= case["k"] <= 5
            assert isinstance(case["tag"], str)


def test_c2_tag_sets():
    structural = Counter(case["tag"] for case in _load("scripts/zephyr_eval_cases.json"))
    dev = Counter(case["tag"] for case in _load("scripts/zephyr_semantic_dev.json"))
    heldout = Counter(case["tag"] for case in _load("scripts/zephyr_semantic_heldout.json"))
    assert set(structural) == {"structural"}
    assert dev == Counter(
        {"code": 12, "asm": 1, "kconfig": 2, "doc": 1, "conf": 1, "devicetree": 1}
    )
    assert heldout == Counter({"code": 5, "cmake": 1, "doc": 1, "kconfig": 1})


def test_c3_questions_are_unique():
    questions = [
        case["question"]
        for name in _CASE_FILES
        for case in _load(name)
    ]
    assert len(questions) == len(set(questions))


def test_c4_baseline_is_hybrid_structural_and_dev():
    baseline = json.loads(Path("scripts/zephyr_baseline.json").read_text(encoding="utf-8"))
    assert baseline["ranker"] == "hybrid"
    assert baseline["scope"] == ["."]
    assert baseline["embed"]["unique"] <= 1500
    files = baseline["files"]
    assert [Path(entry["cases_file"]).name for entry in files] == [
        "zephyr_eval_cases.json",
        "zephyr_semantic_dev.json",
    ]
    for entry in files:
        cases = json.loads(
            Path("scripts", Path(entry["cases_file"]).name).read_text(encoding="utf-8")
        )
        assert [result["question"] for result in entry["results"]] == [
            case["question"] for case in cases
        ]


def test_g1_gitignore_and_pytest_testpaths():
    gitignore = Path(".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/zephyr-core/" in gitignore
    assert "/zephyr-core-check/" in gitignore
    with Path("pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    assert pyproject["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
