"""Fetch the Zephyr RTOS core (no boards, drivers, dts, soc or modules) at a pinned release."""
import argparse
import subprocess
import sys
from pathlib import Path

URL = "https://github.com/zephyrproject-rtos/zephyr"
TAG = "v4.4.2"
SHA = "dccb09599635bdff17633fa7e9dab014b91dce90"
PATTERNS = (
    "/CMakeLists.txt", "/Kconfig", "/Kconfig.zephyr", "/Kconfig.constants", "/VERSION",
    "/kernel/", "/lib/", "/arch/CMakeLists.txt", "/arch/Kconfig", "/arch/archs.yml",
    "/arch/arm/", "/arch/arm64/", "/arch/common/", "/include/zephyr/",
    "!/include/zephyr/dt-bindings/", "!/include/zephyr/bluetooth/", "!/include/zephyr/net/",
    "!/include/zephyr/usb/", "!/include/zephyr/usb_c/", "!/include/zephyr/mgmt/",
    "!/include/zephyr/modem/", "!/include/zephyr/xen/",
    "/subsys/logging/", "/subsys/shell/", "/subsys/testsuite/", "/subsys/zbus/",
    "/subsys/pm/", "/subsys/rtio/", "/subsys/tracing/",
    "/samples/basic/", "/samples/synchronization/", "/samples/philosophers/", "/doc/kernel/",
)


def _git(dest: Path, *args: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(dest), *args], input=stdin, text=True, capture_output=True, check=False
    )
    if result.returncode:
        sys.exit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def fetch(dest: Path, *, url: str = URL, tag: str = TAG, sha: str = SHA,
          patterns: tuple[str, ...] = PATTERNS) -> tuple[int, int]:
    if not (dest / ".git").is_dir():
        if dest.exists() and any(dest.iterdir()):
            sys.exit(f"{dest} exists and is not a git checkout; remove it or pick another --dest")
        clone = subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
             "--branch", tag, url, str(dest)],
            check=False,
        )
        if clone.returncode:
            sys.exit("git clone failed")
    head = _git(dest, "rev-parse", "HEAD")
    if head != sha:
        sys.exit(f"{dest} is at {head}, expected {sha} ({tag}); remove it and run again")
    _git(dest, "sparse-checkout", "set", "--no-cone", "--stdin", stdin="\n".join(patterns) + "\n")
    _git(dest, "checkout", "--quiet", "HEAD")
    files = [p for p in dest.rglob("*") if p.is_file() and p.relative_to(dest).parts[0] != ".git"]
    return len(files), sum(p.stat().st_size for p in files)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="zephyr-core")
    args = parser.parse_args(argv)
    count, size = fetch(Path(args.dest))
    print(f"{args.dest}: Zephyr {TAG} ({SHA[:12]}), {count} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
