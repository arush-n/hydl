"""Named checks the console can launch and remember.

The console could launch a rollout and a training run but not the one thing you
do before trusting either: run the tests. So a result was always "green on my
machine, some time ago, probably".

A check is a command plus how to read its result. Runs are one at a time and in
the background -- the ADK suite takes over an hour, which no HTTP request will
survive -- and the last result of each is kept so the dashboard can show it
without re-running.

**Gradle is parsed from its JUnit XML, never from its exit code.** `BUILD
SUCCESSFUL` with exit 0 is what you get when zero tests ran, and this repo has
been burned by exactly that: a green build that had executed nothing.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from console.core import storage

_ROOT = Path(__file__).resolve().parents[3]

#: Resolved through `storage` rather than computed here, so this directory moves
#: when the `checks` area moves instead of being an eleventh independent answer
#: to "where does the console put things". Identical today -- `write_root`
#: returns the legacy path until the area is migrated -- so this is a rewiring,
#: not a relocation.
RESULT_ROOT = storage.write_root("checks")


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    command: list[str]
    cwd: Path
    description: str = ""
    #: Directory of JUnit XML, when the runner produces it. Present means the
    #: exit code is not trusted on its own.
    junit: Path | None = None


CHECKS: dict[str, Check] = {
    "console": Check(
        id="console", title="Console tests",
        command=["python", "-m", "pytest", "console/tests", "-q"],
        cwd=_ROOT,
        description="The console's own suite. Seconds, safe to run any time.",
    ),
    "adk-zones": Check(
        id="adk-zones", title="Zones and environments",
        command=["python", "-m", "pytest", "adk/tests/test_zones.py",
                 "adk/tests/test_validation_region.py", "-q"],
        cwd=_ROOT,
        description="Navigable-zone cut and Region fixture identity.",
    ),
    "jvm-agent": Check(
        id="jvm-agent", title="Java agent tests",
        command=["cmd", "/c", "gradlew.bat", "test", "--console=plain"],
        cwd=_ROOT / "experimental" / "jvm-agent",
        description=(
            "The Java policy port: decode, recurrent parity against JAX, and "
            "the bundle contract gate. Read from JUnit XML, because a Gradle "
            "exit code of 0 is also what zero tests executed looks like."
        ),
        junit=_ROOT / "experimental" / "jvm-agent" / "build" / "test-results" / "test",
    ),
}


@dataclass
class Run:
    check_id: str
    started_at: float
    state: str = "running"
    finished_at: float | None = None
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    exit_code: int | None = None
    tail: list[str] = field(default_factory=list)
    note: str = ""

    def view(self) -> dict[str, Any]:
        return {
            "check": self.check_id, "state": self.state,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "passed": self.passed, "failed": self.failed,
            "skipped": self.skipped, "exit_code": self.exit_code,
            "seconds": round((self.finished_at or time.time()) - self.started_at, 1),
            "tail": self.tail[-24:], "note": self.note,
        }


_LOCK = threading.Lock()
_ACTIVE: str | None = None
_LAST: dict[str, Run] = {}


def _result_path(check_id: str) -> Path:
    return RESULT_ROOT / f"{check_id}.json"


def _persist(run: Run) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _result_path(run.check_id).write_text(
        json.dumps(run.view(), indent=2) + "\n", encoding="utf-8")


def _load(check_id: str) -> dict[str, Any] | None:
    path = _result_path(check_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


_PYTEST = re.compile(
    r"(?:(\d+) failed)?[, ]*(?:(\d+) passed)?[, ]*(?:(\d+) skipped)?")


def _parse_pytest(text: str) -> tuple[int, int, int]:
    passed = failed = skipped = 0
    for line in text.splitlines():
        if " passed" in line or " failed" in line or " skipped" in line:
            for count, word in re.findall(r"(\d+) (passed|failed|skipped|error[s]?)", line):
                if word == "passed":
                    passed = int(count)
                elif word == "failed":
                    failed = int(count)
                elif word.startswith("error"):
                    failed += int(count)
                else:
                    skipped = int(count)
    return passed, failed, skipped


_JUNIT_CACHE: dict[Path, tuple[int, int, dict[str, Any]]] = {}


def junit_records(roots: Path | Iterable[Path]) -> list[dict[str, Any]]:
    """Every JUnit XML record below one or more roots, with provenance.

    ``TEST-*.xml`` is only Gradle's spelling.  The workspace also writes
    ``*.junit.xml`` and named receipts such as ``combat-contract-v1.xml``;
    accepting all XML and validating the root element covers all three without
    teaching the parser lane-specific filenames.  Failures and errors remain
    separate here so the receipt UI can show all four counters.
    """

    if isinstance(roots, (str, Path)):
        requested = [Path(roots)]
    else:
        requested = [Path(root) for root in roots]

    found: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for receipt_root in requested:
        if receipt_root.is_file():
            paths = [receipt_root] if receipt_root.suffix.lower() == ".xml" else []
            root_dir = receipt_root.parent
        elif receipt_root.is_dir():
            paths = receipt_root.rglob("*.xml")
            root_dir = receipt_root
        else:
            continue
        for path in sorted(paths):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            record = _junit_record(path)
            if record is None:
                continue
            try:
                relative = path.relative_to(root_dir).as_posix()
            except ValueError:
                relative = path.name
            found.append({
                **record,
                "root": str(root_dir),
                "relative_path": relative,
            })
    return found


def _parse_junit(roots: Path | Iterable[Path]) -> tuple[int, int, int]:
    """Counts from XML, because a Gradle exit code cannot tell you 0 ran.

    Errors intentionally fold into ``failed`` in this compatibility return
    shape.  That accounting is load-bearing; :func:`junit_records` exposes the
    separate counters for richer callers without changing check execution.
    """

    passed = failed = skipped = 0
    for record in junit_records(roots):
        if record.get("error"):
            continue
        failures = int(record["failures"])
        errors = int(record["errors"])
        skipped_count = int(record["skipped"])
        tests = int(record["tests"])
        f = failures + errors
        failed += f
        skipped += skipped_count
        passed += max(0, tests - f - skipped_count)
    return passed, failed, skipped


def _junit_record(path: Path) -> dict[str, Any] | None:
    """Parse one XML file if it is a JUnit testsuite/testsuites document."""

    try:
        stat = path.stat()
    except OSError:
        return None
    resolved = path.resolve()
    cached = _JUNIT_CACHE.get(resolved)
    identity = (stat.st_mtime_ns, stat.st_size)
    if cached and cached[:2] == identity:
        return dict(cached[2])

    try:
        xml_root = ElementTree.parse(path).getroot()
    except (ElementTree.ParseError, OSError) as exc:
        record = {
            "path": str(path),
            "file_mtime": stat.st_mtime,
            "file_size": stat.st_size,
            "timestamp": None,
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _JUNIT_CACHE[resolved] = (*identity, dict(record))
        return record

    tag = xml_root.tag.rsplit("}", 1)[-1]
    if tag not in {"testsuite", "testsuites"}:
        return None

    suites = [xml_root]
    if tag == "testsuites" and xml_root.get("tests") is None:
        suites = list(xml_root.findall("./testsuite"))
    tests = sum(_xml_count(suite, "tests") for suite in suites)
    failures = sum(_xml_count(suite, "failures") for suite in suites)
    errors = sum(_xml_count(suite, "errors") for suite in suites)
    skipped = sum(_xml_count(suite, "skipped") for suite in suites)
    timestamps = [
        str(suite.get("timestamp")) for suite in suites if suite.get("timestamp")
    ]
    record = {
        "path": str(path),
        "file_mtime": stat.st_mtime,
        "file_size": stat.st_size,
        "timestamp": min(timestamps) if timestamps else None,
        "timestamps": sorted(set(timestamps)),
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "error": "",
    }
    _JUNIT_CACHE[resolved] = (*identity, dict(record))
    return record


def _xml_count(element: ElementTree.Element, name: str) -> int:
    try:
        return int(float(element.get(name, "0")))
    except (TypeError, ValueError):
        return 0


def _execute(check: Check, run: Run) -> None:
    global _ACTIVE
    try:
        proc = subprocess.run(
            check.command, cwd=str(check.cwd), capture_output=True,
            text=True, timeout=7200,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        run.exit_code = proc.returncode
        run.tail = output.strip().splitlines()[-40:]

        if check.junit is not None:
            passed, failed, skipped = _parse_junit(check.junit)
            run.passed, run.failed, run.skipped = passed, failed, skipped
            if passed + failed + skipped == 0:
                # The trap this repo has hit: exit 0 and BUILD SUCCESSFUL with
                # nothing executed. That is a failure to run, not a pass.
                run.state = "failed"
                run.note = (
                    "No JUnit XML found — zero tests executed. A Gradle exit "
                    "code of 0 also looks like this, which is why it is not "
                    "trusted on its own."
                )
                return
        else:
            run.passed, run.failed, run.skipped = _parse_pytest(output)

        run.state = "passed" if (run.failed == 0 and run.exit_code == 0) else "failed"
    except subprocess.TimeoutExpired:
        run.state = "failed"
        run.note = "timed out after 2 hours"
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        run.state = "failed"
        run.note = f"{type(exc).__name__}: {exc}"
    finally:
        run.finished_at = time.time()
        _persist(run)
        with _LOCK:
            _ACTIVE = None


def launch(check_id: str) -> dict[str, Any]:
    """Start one check in the background. One at a time."""

    global _ACTIVE
    check = CHECKS.get(check_id)
    if check is None:
        raise ValueError(f"unknown check {check_id!r}; have {sorted(CHECKS)}")
    with _LOCK:
        if _ACTIVE is not None:
            raise RuntimeError(
                f"{_ACTIVE!r} is still running; checks run one at a time so "
                "they do not contend for the same build directory")
        _ACTIVE = check_id
        run = Run(check_id=check_id, started_at=time.time())
        _LAST[check_id] = run
    threading.Thread(target=_execute, args=(check, run), daemon=True).start()
    return run.view()


def index() -> dict[str, Any]:
    """Every check, with its last result — from memory, then from disk."""

    out = []
    for check in CHECKS.values():
        live = _LAST.get(check.id)
        last = live.view() if live else _load(check.id)
        out.append({
            "id": check.id, "title": check.title,
            "description": check.description,
            "command": " ".join(check.command),
            "last": last,
        })
    return {"checks": out, "active": _ACTIVE}
