"""ADK wrapper for the shared, non-blocking native-evidence lease."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from hytalegym.worldgen.native_lease import (
    NativeEvidenceLease,
    acquire_native_evidence_lease,
)

from adk.runtime.backends import BackendSelection


_OWNER_PATTERN = re.compile(
    r"owner_pid=(?P<pid>\d+|unknown)\s+purpose=(?P<purpose>.*)$"
)


class LeaseUnavailable(RuntimeError):
    """Contention outcome: the requested evidence gate was not evaluated."""

    def __init__(
        self,
        message: str,
        *,
        host: str,
        port: int,
        owner_pid: int | None,
        owner_purpose: str | None,
    ) -> None:
        super().__init__(message)
        self.host = host
        self.port = port
        self.owner_pid = owner_pid
        self.owner_purpose = owner_purpose


def acquire_native_lease(
    selection: BackendSelection,
    *,
    purpose: str,
    stage: str | Path | None = None,
    evidence_kinds: Sequence[str] = (),
) -> NativeEvidenceLease:
    """Acquire immediately or raise structured :class:`LeaseUnavailable`."""

    if not isinstance(selection, BackendSelection):
        raise TypeError("selection must be a BackendSelection")
    if selection.backend != "native":
        raise RuntimeError("only the native evidence backend may acquire this lease")
    if selection.is_training:
        raise RuntimeError("training entry points must never acquire the native lease")
    try:
        return acquire_native_evidence_lease(
            host=selection.host,
            port=selection.port,
            purpose=purpose,
            stage=stage,
            evidence_kinds=evidence_kinds,
        )
    except RuntimeError as error:
        message = str(error)
        match = _OWNER_PATTERN.search(message)
        if "native evidence already active" not in message:
            raise
        owner_pid = None
        owner_purpose = None
        if match is not None:
            raw_pid = match.group("pid")
            owner_pid = None if raw_pid == "unknown" else int(raw_pid)
            owner_purpose = match.group("purpose") or None
        raise LeaseUnavailable(
            message,
            host=selection.host,
            port=selection.port,
            owner_pid=owner_pid,
            owner_purpose=owner_purpose,
        ) from error


__all__ = [
    "LeaseUnavailable",
    "NativeEvidenceLease",
    "acquire_native_lease",
]
