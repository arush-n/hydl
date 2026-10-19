"""Version-keyed pins for the World source-pin audits.

Each audit keeps its 0.5.7 pins as module constants; those are the 0.5 line and are used unchanged.
For the 0.7 line the pins come from ``source_pin_baselines_072.json`` (observed on the installed
0.7.2 build). A pin with no recorded 0.7.2 value is compared as ``UNRECORDED`` (never equal), and a
semantic baseline is recorded as ``null`` while the audit has open 0.7.2 gaps, so the audit fails
closed. Claim text an audit hashes (its SOURCE_CONTRACT, or a payload claim constant) is re-worded per
line under the record's ``claim_texts`` object, keyed by the module constant's name; a line with no entry
hashes the module text. A 0.7 semantic pin for an audit listed in STALE_CLAIM_TEXT_072 is refused without
its claim_texts override. The release line comes from the audited server JAR's own manifest, not from the
environment.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping
from zipfile import ZipFile

from hytalegym.server_versions import release_line

ABSENT = "absent"
UNRECORDED = "unrecorded"
BASELINE_072_PATH = Path(__file__).with_name("source_pin_baselines_072.json")
_MANIFEST = "META-INF/MANIFEST.MF"


def archive_digest(archive: ZipFile, name: str) -> str:
    """SHA-256 of one archive member, or ``ABSENT`` when the member is not in the archive."""

    try:
        data = archive.read(name)
    except KeyError:
        return ABSENT
    return hashlib.sha256(data).hexdigest()


def server_jar_release_line(server_jar: str | Path) -> tuple[int, int]:
    """Return the (major, minor) line of a server JAR's Implementation-Version."""

    with ZipFile(server_jar) as archive:
        manifest = archive.read(_MANIFEST).decode("utf-8")
    for line in manifest.splitlines():
        if line.startswith("Implementation-Version: "):
            parsed = release_line(line.split(": ", 1)[1].strip())
            if parsed is None:
                break
            return parsed
    raise RuntimeError(f"installed server JAR has no usable Implementation-Version: {server_jar}")


@lru_cache(maxsize=1)
def _recorded_072() -> dict[str, dict[str, object]]:
    return json.loads(BASELINE_072_PATH.read_text(encoding="utf-8"))["audits"]


# Audits whose module claim text states a claim that is false on 0.7.2 (tools/source_pins/README.md, "Claim text
# per release line"), with the claim constants to re-word. A 0.7 record may carry a semantic pin for one of
# them only with a claim_texts override for each listed constant; pins_for refuses the record otherwise.
STALE_CLAIM_TEXT_072: Mapping[str, tuple[str, ...]] = {
    "locomotion": ("SOURCE_CONTRACT",),
    "crafting": ("SOURCE_CONTRACT",),
    "explosion_mutation": ("SOURCE_CONTRACT",),
    "mutable_blocks": ("SOURCE_CONTRACT",),
    "inventory": ("SOURCE_CONTRACT",),
    "entity_geometry": ("SOURCE_CONTRACT",),
    "perception": ("PERCEPTION_CLAIMS",),
}


def pins_for(
    audit: str,
    server_jar: str | Path,
    defaults: Mapping[str, object],
    *,
    keyed: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Return the pins an audit compares against, for the release line of ``server_jar``.

    ``defaults`` maps each pin name to its 0.5.7 value. ``keyed`` names the pins that are
    dictionaries keyed by archive member; their unrecorded value is a sentinel dictionary that no
    observed mapping equals. On the 0.7 line the record's ``claim_texts`` entries are added under their
    constant names (for example ``SOURCE_CONTRACT``); the 0.5 line never receives them.
    """

    line = server_jar_release_line(server_jar)
    if line == (0, 5):
        return dict(defaults)
    if line != (0, 7):
        raise RuntimeError(
            f"no source-pin baseline for Hytale {line[0]}.{line[1]} ({audit}); "
            "record one in source_pin_baselines_072.json or add a release line"
        )
    recorded = _recorded_072().get(audit, {})
    claims = recorded.get("claim_texts", {})
    if recorded.get("EXPECTED_SEMANTIC_SHA256") is not None:
        missing = [name for name in STALE_CLAIM_TEXT_072.get(audit, ()) if name not in claims]
        if missing:
            raise RuntimeError(
                f"{audit}: refusing a 0.7 semantic pin without claim_texts for its stale claim "
                f"constants {missing}; record the 0.7.2 wording first"
            )
    pins = {
        name: recorded[name]
        if name in recorded
        else ({UNRECORDED: UNRECORDED} if name in keyed else UNRECORDED)
        for name in defaults
    }
    pins.update(claims)
    return pins


__all__ = [
    "ABSENT",
    "BASELINE_072_PATH",
    "STALE_CLAIM_TEXT_072",
    "UNRECORDED",
    "archive_digest",
    "pins_for",
    "server_jar_release_line",
]
