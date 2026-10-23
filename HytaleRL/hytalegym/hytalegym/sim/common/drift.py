"""Drift checker: re-reads every declared source of one generation's overlay.

A release port is mechanical only if every overlay value can be traced to a
machine-checkable source. This module reads the declarations in
``hytalegym.sim.<gen>.sources`` (one per overlay row id, optional) and re-reads
each source from a new release's ``Assets.zip`` and ``HytaleServer.jar``:

* ``AssetSource``: a JSON value in an archive member, at an RFC 6901 pointer,
  compared with the value the row was written against.
* ``ClassSource``: a method's bytecode digest (``javap -c -p``, constant-pool
  indexes stripped) or a compile-time constant (``javap -constants``) in a
  server-jar class.
* ``Derived``: a value that is not read from the release, with its reason
  (an owner ruling, a construct gated by a differential, a content table that
  the content re-extraction checks).

Each declaration may also be a tuple, which every part must satisfy.

Statuses per row: ``unchanged``, ``changed`` (old and new values), ``missing``
(member, pointer, method or constant gone), ``derived`` (not re-read),
``undeclared`` (an overlay row without a declaration). ``stale`` lists a
declaration whose row no longer exists.

Stdlib only at import time (the overlay is imported lazily by the CLI), so it
can run before ``hytalegym`` is imported.

Command::

    python -m hytalegym.sim.common.drift --assets ZIP --jar JAR [--generation gen7] [--json OUT]

Exit status: 0 when no declared row changed, is missing, or is undeclared; 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

#: Compile-time constants and method digests are read with this JDK tool.
JAVAP_ENV = "JAVAP"
#: The game archive's JSON is JSONC: comments and trailing commas (same rule as content_extract).
_JSONC_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|,(?=\s*[}\]])', re.S)
_CP_INDEX = re.compile(r"#\d+")
STATUSES = ("unchanged", "changed", "missing", "derived", "undeclared")


@dataclass(frozen=True)
class AssetSource:
    """``path`` inside Assets.zip; ``pointer`` an RFC 6901 pointer ("" = the document)."""

    path: str
    pointer: str
    expected: object


@dataclass(frozen=True)
class ClassSource:
    """A method (``member`` = ``name`` or ``name/N``, N parameters) or a constant field.

    ``mode="digest"``: ``expected`` is the sha256 of the method's normalised ``javap -c -p`` block.
    ``mode="constant"``: ``expected`` is the value javap prints for the field (``javap -constants``).
    """

    cls: str
    member: str
    expected: str
    mode: str = "digest"


@dataclass(frozen=True)
class Derived:
    """A value this checker does not re-read; ``reason`` says what checks it instead."""

    reason: str


Source = AssetSource | ClassSource | Derived


class Missing(Exception):
    """A declared member, pointer, method or constant is absent in the release."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def parse_jsonc(text: str):
    text = re.sub(
        _JSONC_TOKEN, lambda m: m.group(0) if m.group(0).startswith('"') else "", text
    )
    return json.loads(text)


def resolve_pointer(document, pointer: str):
    """RFC 6901: ``/A/0/B`` walks keys and indexes; ``~1`` is ``/``, ``~0`` is ``~``."""

    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"not a JSON pointer: {pointer!r}")
    node = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            raise Missing(f"pointer {pointer} has no {token!r}")
    return node


class AssetReader:
    """Reads archive members once each, as parsed JSON."""

    def __init__(self, zip_path: Path):
        self.zip_path = Path(zip_path)
        self._zip = zipfile.ZipFile(self.zip_path)
        self._docs: dict[str, object] = {}

    def document(self, member: str):
        if member not in self._docs:
            try:
                raw = self._zip.read(member)
            except KeyError as error:
                raise Missing(f"member {member} not in archive") from error
            self._docs[member] = parse_jsonc(raw.decode("utf-8-sig"))
        return self._docs[member]


class JarReader:
    """``javap`` over one server jar; each class is disassembled once."""

    def __init__(self, jar: Path, javap: str | None = None):
        self.jar = Path(jar)
        self.javap = javap or shutil.which("javap") or os.environ.get(JAVAP_ENV)
        if not self.javap:
            raise RuntimeError(
                "class sources need javap (PATH or the JAVAP environment variable)"
            )
        self._cache: dict[tuple[str, str], str] = {}

    def listing(self, cls: str, flags: tuple[str, ...]) -> str:
        key = (cls, " ".join(flags))
        if key not in self._cache:
            run = subprocess.run(
                [self.javap, *flags, "-classpath", str(self.jar), cls],
                capture_output=True,
                text=True,
                check=False,
            )
            if run.returncode != 0 or not run.stdout:
                raise Missing(
                    f"class {cls} not in {self.jar.name}: {run.stderr.strip()[:200]}"
                )
            self._cache[key] = run.stdout
        return self._cache[key]


def _param_count(params: str) -> int:
    if not params.strip():
        return 0
    depth, count = 0, 1
    for char in params:
        depth += char in "<("
        depth -= char in ">)"
        count += char == "," and depth == 0
    return count


def method_block(javap_text: str, member: str) -> str:
    """The normalised ``javap -c -p`` block of one method (``name`` or ``name/N``)."""

    name, _, count = member.partition("/")
    lines = javap_text.splitlines()
    starts = []
    for index, line in enumerate(lines):
        if (
            not line.startswith("  ")
            or line.startswith("   ")
            or not line.rstrip().endswith(");")
        ):
            continue
        header = line.strip()[:-1]
        head, paren, params = header.partition("(")
        if not paren or head.split()[-1] != name:
            continue
        if count and _param_count(params[:-1]) != int(count):
            continue
        starts.append(index)
    if not starts:
        raise Missing(f"method {member} not in class")
    if len(starts) > 1:
        raise Missing(f"method {member} is overloaded; declare name/N")
    start = starts[0]
    end = start + 1
    while end < len(lines) and lines[end].strip():
        end += 1
    return "\n".join(_CP_INDEX.sub("#", line) for line in lines[start:end])


def constant_value(javap_text: str, field: str) -> str:
    pattern = re.compile(rf"\b{re.escape(field)} = (.+);$")
    for line in javap_text.splitlines():
        match = pattern.search(line.strip())
        if match:
            return match.group(1)
    raise Missing(f"constant {field} not in class")


@dataclass(frozen=True)
class PartResult:
    status: str
    expected: object
    actual: object
    detail: str


@dataclass(frozen=True)
class RowResult:
    row: str
    row_kind: str  # redirect | literal redirect | patch | construct
    status: str
    parts: tuple[PartResult, ...]
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "row": self.row,
            "row_kind": self.row_kind,
            "status": self.status,
            "reason": self.reason,
            "parts": [part.__dict__ for part in self.parts],
        }


def _check_part(
    source: Source, assets: AssetReader | None, jars: JarReader | None
) -> PartResult:
    if isinstance(source, Derived):
        return PartResult("derived", None, None, source.reason)
    try:
        if isinstance(source, AssetSource):
            actual = resolve_pointer(assets.document(source.path), source.pointer)
            detail = f"{source.path}{source.pointer}"
            status = "unchanged" if actual == source.expected else "changed"
            return PartResult(status, source.expected, actual, detail)
        if source.mode == "digest":
            block = method_block(jars.listing(source.cls, ("-c", "-p")), source.member)
            actual = _digest(block.encode())
        elif source.mode == "constant":
            actual = constant_value(
                jars.listing(source.cls, ("-constants", "-p")), source.member
            )
        else:
            raise ValueError(f"unknown class mode {source.mode!r}")
        detail = f"{source.cls}.{source.member} ({source.mode})"
        return PartResult(
            "unchanged" if actual == source.expected else "changed",
            source.expected,
            actual,
            detail,
        )
    except Missing as error:
        return PartResult(
            "missing", getattr(source, "expected", None), None, str(error)
        )


def check(
    rows,
    declarations: dict,
    assets: AssetReader | None = None,
    jars: JarReader | None = None,
) -> tuple[list[RowResult], list[str]]:
    """Check every row. ``rows`` is ``[(row_id, row_kind), ...]``; returns (results, stale ids)."""

    ids = {row_id for row_id, _ in rows}
    results = []
    for row_id, row_kind in rows:
        if row_id not in declarations:
            results.append(RowResult(row_id, row_kind, "undeclared", ()))
            continue
        declared = declarations[row_id]
        parts = declared if isinstance(declared, tuple) else (declared,)
        checked = tuple(_check_part(part, assets, jars) for part in parts)
        statuses = {part.status for part in checked}
        if statuses == {"derived"}:
            status, reason = "derived", checked[0].detail
        else:
            status = next(
                (s for s in ("missing", "changed") if s in statuses), "unchanged"
            )
            reason = ""
        results.append(RowResult(row_id, row_kind, status, checked, reason))
    stale = sorted(set(declarations) - ids)
    return results, stale


def summary(results: list[RowResult], stale: list[str]) -> dict:
    counts = {status: 0 for status in STATUSES}
    for result in results:
        counts[result.status] += 1
    counts["stale"] = len(stale)
    return counts


def report_text(results: list[RowResult], stale: list[str]) -> str:
    lines = [f"{'row':<46} {'kind':<15} {'status':<10} detail"]
    for result in results:
        if result.status == "undeclared":
            detail = "no declaration"
        else:
            changed = [
                f"...{p.detail[-70:]} {p.expected!r} -> {p.actual!r}"
                for p in result.parts
                if p.status == "changed"
            ]
            missing = [p.detail for p in result.parts if p.status == "missing"]
            first = result.parts[0].detail if result.parts else ""
            more = (
                f" (+{len(result.parts) - 1} more parts)"
                if len(result.parts) > 1
                else ""
            )
            detail = "; ".join(changed or missing) or result.reason or first + more
        lines.append(
            f"{result.row[:46]:<46} {result.row_kind:<15} {result.status:<10} {detail[:120]}"
        )
    for row_id in stale:
        lines.append(
            f"{row_id[:46]:<46} {'-':<15} {'stale':<10} declared but not in the overlay"
        )
    counts = summary(results, stale)
    lines.append("")
    lines.append("summary: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return "\n".join(lines)


def overlay_rows(overlay) -> list[tuple[str, str]]:
    """Every overlay row with its kind: literal redirect (profile), content redirect, patch, construct."""

    rows = []
    for redirect in overlay.redirects:
        kind = (
            "literal redirect" if redirect.family == "profile" else "content redirect"
        )
        rows.append((redirect.id, kind))
    rows += [(patch.id, "patch") for patch in overlay.patches]
    rows += [(hook.id, "construct") for hook in overlay.hooks]
    return rows


def load_generation(generation: str):
    overlay = importlib.import_module(f"hytalegym.sim.{generation}.overlay").OVERLAY
    try:
        declarations = importlib.import_module(
            f"hytalegym.sim.{generation}.sources"
        ).DECLARATIONS
    except ModuleNotFoundError:
        declarations = {}
    return overlay, declarations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--assets", required=True, type=Path, help="Assets.zip of the release"
    )
    parser.add_argument(
        "--jar", required=True, type=Path, help="Server/HytaleServer.jar of the release"
    )
    parser.add_argument("--generation", default="gen7")
    parser.add_argument(
        "--javap", default=None, help="javap binary (default: PATH, then $JAVAP)"
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="also write the full report as JSON"
    )
    args = parser.parse_args(argv)

    overlay, declarations = load_generation(args.generation)
    assets = AssetReader(args.assets)
    jars = JarReader(args.jar, args.javap)
    results, stale = check(overlay_rows(overlay), declarations, assets, jars)
    print(
        f"drift: {args.generation}, assets {args.assets} ({file_sha256(args.assets)[:16]}), "
        f"jar {args.jar} ({file_sha256(args.jar)[:16]})"
    )
    print(report_text(results, stale))
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "generation": args.generation,
                    "summary": summary(results, stale),
                    "rows": [r.as_dict() for r in results],
                    "stale": stale,
                },
                indent=2,
            )
        )
    counts = summary(results, stale)
    return 1 if counts["changed"] or counts["missing"] or counts["undeclared"] else 0


if __name__ == "__main__":
    # Dispatch through the package module: under ``python -m`` this file is ``__main__``, a second copy of
    # the dataclasses that sources.py (importing the package copy) would not match with isinstance.
    from hytalegym.sim.common import drift as _drift

    sys.exit(_drift.main())
