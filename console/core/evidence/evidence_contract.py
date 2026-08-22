"""Validated, atomic writes for the shared six-field evidence contract."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EVIDENCE_CONTRACT_FIELDS = (
    "schema",
    "status",
    "claim",
    "gates",
    "limitations",
    "next_required_evidence",
)


def validate_evidence_contract(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("evidence artifact must be an object")
    missing = [field for field in EVIDENCE_CONTRACT_FIELDS if field not in payload]
    if missing:
        raise ValueError("missing evidence contract field(s): " + ", ".join(missing))


def write_evidence_contract(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Validate before touching the destination, then replace it atomically."""

    validate_evidence_contract(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


__all__ = [
    "EVIDENCE_CONTRACT_FIELDS",
    "validate_evidence_contract",
    "write_evidence_contract",
]
