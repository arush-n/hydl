"""Read-only access to one Hytale ``Assets.zip`` with per-file provenance.

Every file an extractor reads goes through :class:`GameArchive`, which records
the SHA-256 of the raw bytes. A table's provenance is therefore exactly the set
of files that produced it. Nothing is ever extracted to disk or written back.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

# Strips // and /* */ comments and trailing commas, leaving strings intact
# (same rule as hytale-plugin-0.7.2/tools/diff_assets_072.py).
_JSONC_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|,(?=\s*[}\]])', re.S)


class ArchiveError(ValueError):
    """The archive does not contain what an extractor requires."""


def file_sha256(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest().upper()


def jar_implementation_version(server_jar: str | Path) -> str:
    """Return ``Implementation-Version`` from a server jar's manifest."""

    with ZipFile(server_jar) as jar:
        manifest = jar.read("META-INF/MANIFEST.MF").decode("utf-8", "replace")
    for line in manifest.splitlines():
        if line.startswith("Implementation-Version:"):
            return line.split(":", 1)[1].strip()
    raise ArchiveError(f"{server_jar} has no Implementation-Version")


def parse_json_bytes(data: bytes) -> tuple[Any, bool]:
    """Parse strict JSON, falling back to JSONC; return ``(value, lenient)``."""

    text = data.decode("utf-8-sig", errors="strict")
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        cleaned = _JSONC_TOKEN.sub(
            lambda m: m.group(0) if m.group(0).startswith('"') else "", text
        )
        return json.loads(cleaned), True


class GameArchive:
    """One version's ``Assets.zip`` plus the game version it belongs to."""

    def __init__(
        self,
        assets_zip: str | Path,
        *,
        game_version: str | None = None,
        server_jar: str | Path | None = None,
        assets_sha256: str | None = None,
    ):
        self.path = Path(assets_zip)
        if not self.path.is_file():
            raise FileNotFoundError(f"Assets.zip not found: {self.path}")
        if game_version is None:
            jar = (
                Path(server_jar)
                if server_jar is not None
                else self.path.parent / "Server" / "HytaleServer.jar"
            )
            game_version = jar_implementation_version(jar)
        self.game_version = game_version
        self._assets_sha256 = assets_sha256.upper() if assets_sha256 else None
        self._zip = ZipFile(self.path)
        self.names: tuple[str, ...] = tuple(
            sorted(info.filename for info in self._zip.infolist() if not info.is_dir())
        )
        self._name_set = frozenset(self.names)
        self._json: dict[str, Any] = {}
        self.lenient: set[str] = set()
        self._recording: list[dict[str, str]] = []
        self._duplicate_log: list[set[str]] = []

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "GameArchive":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def assets_sha256(self) -> str:
        if self._assets_sha256 is None:
            self._assets_sha256 = file_sha256(self.path)
        return self._assets_sha256

    # -- provenance -----------------------------------------------------
    def begin_recording(self) -> dict[str, str]:
        sources: dict[str, str] = {}
        self._recording.append(sources)
        self._duplicate_log.append(set())
        return sources

    def end_recording(self, sources: dict[str, str]) -> dict[str, str]:
        index = self._recording.index(sources)
        self._recording.pop(index)
        self.last_duplicates = tuple(sorted(self._duplicate_log.pop(index)))
        return dict(sorted(sources.items()))

    def _note(self, path: str, sha: str) -> None:
        for sources in self._recording:
            sources[path] = sha

    # -- reads ------------------------------------------------------------
    def exists(self, path: str) -> bool:
        return path in self._name_set

    def read_bytes(self, path: str) -> bytes:
        if path not in self._name_set:
            raise ArchiveError(f"missing asset {path}")
        data = self._zip.read(path)
        self._note(path, hashlib.sha256(data).hexdigest().upper())
        return data

    def read_json(self, path: str) -> Any:
        if path in self._json:
            data_sha = self._json[path][1]
            self._note(path, data_sha)
            return self._json[path][0]
        data = self.read_bytes(path)
        value, lenient = parse_json_bytes(data)
        if lenient:
            self.lenient.add(path)
        self._json[path] = (value, hashlib.sha256(data).hexdigest().upper())
        return value

    def list(self, prefix: str, suffix: str = ".json") -> tuple[str, ...]:
        return tuple(n for n in self.names if n.startswith(prefix) and n.endswith(suffix))

    def index(self, prefix: str, *, strict: bool = False) -> dict[str, str]:
        """Map file stem -> path under ``prefix``.

        Stems are the runtime asset ids. With ``strict`` a duplicated stem is
        an error (the combat catalog's rule for items and interactions);
        otherwise the first path in sorted order wins and the duplicates are
        reported by :meth:`duplicates`.
        """

        result: dict[str, str] = {}
        for path in self.list(prefix):
            stem = PurePosixPath(path).stem
            previous = result.setdefault(stem, path)
            if previous != path:
                if strict:
                    raise ArchiveError(f"duplicate asset id {stem}: {previous}, {path}")
                for log in self._duplicate_log:
                    log.add(f"duplicate asset id {stem!r}: using {previous}, ignoring {path}")
        return result

    def duplicates(self, prefix: str) -> dict[str, tuple[str, ...]]:
        seen: dict[str, list[str]] = {}
        for path in self.list(prefix):
            seen.setdefault(PurePosixPath(path).stem, []).append(path)
        return {stem: tuple(paths) for stem, paths in seen.items() if len(paths) > 1}
