"""Combat-owned coordination for native evidence that spans listeners."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

from hytalegym.worldgen.native_lease import (
    NativeEvidenceLease,
    acquire_native_evidence_lease,
)


NativeEndpoint = tuple[str, int]


@contextmanager
def acquire_combat_native_evidence_leases(
    endpoints: Sequence[NativeEndpoint],
    *,
    purpose: str,
    repository_root: str | Path | None = None,
    stage: str | Path | None = None,
    evidence_kinds: Sequence[str] = (),
) -> Iterator[tuple[NativeEvidenceLease, ...]]:
    """Acquire all endpoint leases in one deterministic order.

    Deterministic ordering prevents two multi-listener Combat probes from
    deadlocking when they name the same endpoints in different orders. The
    shared World-owned lease remains the only lock implementation.
    """

    ordered = tuple(
        sorted(
            set(endpoints),
            key=lambda endpoint: (
                endpoint[0].casefold().rstrip("."),
                endpoint[1],
            ),
        )
    )
    if not ordered:
        raise ValueError("combat native evidence requires at least one endpoint")

    with ExitStack() as stack:
        leases = tuple(
            stack.enter_context(
                acquire_native_evidence_lease(
                    repository_root,
                    host=host,
                    port=port,
                    purpose=purpose,
                    stage=stage,
                    evidence_kinds=evidence_kinds,
                )
            )
            for host, port in ordered
        )
        yield leases


__all__ = [
    "NativeEndpoint",
    "acquire_combat_native_evidence_leases",
]
