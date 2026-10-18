#!/usr/bin/env python3
"""Check or move the bridge jar pin (bridge_jar_sha256 in hytalegym/server_versions/v0_7.py).

--check  hash build/libs/HytaleRLBridge072-0.1.0.jar (SHA-256, upper case) against the pin. Exit 0 when equal;
         exit 1 and print both hashes when not.
--build  run ./gradlew jar --offline --no-build-cache in this plugin, hash the jar, rewrite only the pin line
         in v0_7.py, run tests/test_server_versions.py and print the old and new hash.

Intermediate builds go to a scratch copy. Only the final build runs --build, because the native reset compares the
hash a running bridge reports with the pin (worldgen/region/stability.py, require_current_native_runtime_bridge), and
other sessions run bridges from copies of build/libs.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

PLUGIN = Path(__file__).resolve().parents[1]
REPO = PLUGIN.parents[1]
JAR = PLUGIN / "build/libs/HytaleRLBridge072-0.1.0.jar"
PIN_FILE = REPO / "HytaleRL/hytalegym/hytalegym/server_versions/v0_7.py"
PIN = re.compile(r'bridge_jar_sha256=\(\s*"([0-9A-Fa-f]{64})"')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _pinned(text: str) -> str:
    match = PIN.search(text)
    if match is None:
        raise SystemExit(f"bridge_jar_sha256 not found in {PIN_FILE}")
    return match.group(1).upper()


def _head_pin() -> str | None:
    shown = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{PIN_FILE.relative_to(REPO).as_posix()}"],
                           capture_output=True, text=True)
    return _pinned(shown.stdout) if shown.returncode == 0 else None


def _check() -> int:
    if not JAR.is_file():
        print(f"missing jar: {JAR}")
        return 1
    pinned, actual = _pinned(PIN_FILE.read_text(encoding="utf-8")), _sha256(JAR)
    committed = _head_pin()
    if committed not in (None, pinned):
        print(f"note: the working-tree pin differs from HEAD's pin {committed} (uncommitted change)")
    if pinned == actual:
        print(f"pin matches: {actual} (working-tree pin in {PIN_FILE.relative_to(REPO)}; jar {JAR.relative_to(PLUGIN)})")
        return 0
    print(f"pin mismatch\n  pinned: {pinned}\n  jar:    {actual} ({JAR.relative_to(PLUGIN)})")
    return 1


def _build() -> int:
    text = PIN_FILE.read_text(encoding="utf-8")
    old = _pinned(text)
    if subprocess.run(["./gradlew", "jar", "--offline", "--no-build-cache"], cwd=PLUGIN).returncode:
        print("gradlew jar failed; the pin is unchanged")
        return 1
    new = _sha256(JAR)
    updated = PIN.sub(lambda m: m.group(0).replace(m.group(1), new), text, count=1)
    changed = [i for i, (a, b) in enumerate(zip(text.splitlines(), updated.splitlines())) if a != b]
    if len(changed) != 1 or old == new:
        print(f"pin not written: {len(changed)} changed lines, old {old}, new {new}")
        return 1
    PIN_FILE.write_text(updated, encoding="utf-8")
    print(f"pin line {changed[0] + 1} of {PIN_FILE.name}: {old} -> {new}")
    tests = subprocess.run([sys.executable, "-m", "pytest", "HytaleRL/hytalegym/tests/test_server_versions.py", "-q"],
                           cwd=REPO)
    print(f"old hash {old}\nnew hash {new}\ntests/test_server_versions.py exit {tests.returncode}")
    return tests.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    return _check() if args.check else _build()


if __name__ == "__main__":
    raise SystemExit(main())
