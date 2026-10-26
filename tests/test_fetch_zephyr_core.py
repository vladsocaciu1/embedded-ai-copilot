import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.fetch_zephyr_core import SHA, TAG, URL, PATTERNS, fetch

_git_missing = pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")
_PATTERNS = ("/kernel/", "/include/zephyr/", "!/include/zephyr/net/")
_FORBIDDEN = ("/boards/", "/drivers/", "/dts/", "/soc/", "/modules/")


def test_f1_url_tag_sha_and_patterns():
    assert URL.startswith("https://")
    assert TAG == "v4.4.2"
    assert re.fullmatch(r"[0-9a-f]{40}", SHA)
    for pattern in PATTERNS:
        for prefix in _FORBIDDEN:
            assert pattern != prefix
            assert not pattern.startswith(prefix)


def _run(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise AssertionError(f"{args} failed:\n{result.stderr}")
    return result.stdout.strip()


def _worktree(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).parts[0] != ".git"
    )


@pytest.fixture
def remote(tmp_path: Path) -> tuple[str, str]:
    repo = tmp_path / "remote"
    _run(["git", "init", "-b", "main", str(repo)])
    (repo / "kernel").mkdir()
    (repo / "kernel" / "a.c").write_bytes(b"a\n")
    (repo / "include" / "zephyr" / "net").mkdir(parents=True)
    (repo / "include" / "zephyr" / "x.h").write_bytes(b"b\n")
    (repo / "include" / "zephyr" / "net" / "n.h").write_bytes(b"n\n")
    (repo / "drivers").mkdir()
    (repo / "drivers" / "d.c").write_bytes(b"d\n")
    _run(["git", "-C", str(repo), "add", "."])
    _run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=a@b",
            "-c",
            "user.name=n",
            "commit",
            "-m",
            "init",
        ]
    )
    _run(["git", "-C", str(repo), "tag", "v1"])
    _run(["git", "-C", str(repo), "config", "uploadpack.allowFilter", "true"])
    sha = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    return repo.as_uri(), sha


@_git_missing
def test_f2_sparse_checkout_keeps_kernel_and_include(tmp_path: Path, remote: tuple[str, str]):
    url, sha = remote
    dest = tmp_path / "dest"
    assert fetch(dest, url=url, tag="v1", sha=sha, patterns=_PATTERNS) == (2, 4)
    assert _worktree(dest) == ["include/zephyr/x.h", "kernel/a.c"]


@_git_missing
def test_f3_second_fetch_keeps_the_same_files(tmp_path: Path, remote: tuple[str, str]):
    url, sha = remote
    dest = tmp_path / "dest"
    fetch(dest, url=url, tag="v1", sha=sha, patterns=_PATTERNS)
    first = _worktree(dest)
    fetch(dest, url=url, tag="v1", sha=sha, patterns=_PATTERNS)
    assert _worktree(dest) == first == ["include/zephyr/x.h", "kernel/a.c"]


@_git_missing
def test_f4_wrong_sha_names_both_shas(tmp_path: Path, remote: tuple[str, str]):
    url, sha = remote
    dest = tmp_path / "dest"
    wrong = "0" * 40
    with pytest.raises(SystemExit) as raised:
        fetch(dest, url=url, tag="v1", sha=wrong, patterns=_PATTERNS)
    message = str(raised.value)
    assert sha in message
    assert wrong in message


@_git_missing
def test_f5_non_git_dest_is_unchanged(tmp_path: Path, remote: tuple[str, str]):
    url, sha = remote
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "keep.txt").write_bytes(b"stay\n")
    before = {path.relative_to(dest).as_posix(): path.read_bytes() for path in dest.rglob("*") if path.is_file()}
    with pytest.raises(SystemExit):
        fetch(dest, url=url, tag="v1", sha=sha, patterns=_PATTERNS)
    after = {path.relative_to(dest).as_posix(): path.read_bytes() for path in dest.rglob("*") if path.is_file()}
    assert after == before
