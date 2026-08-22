"""Turn gate results into a verdict a person can act on.

A validator that reports numbers but never says "ready" or "not ready" pushes
the judgement back onto whoever reads it, and that judgement is exactly where
this project has gone wrong: a zero gets read as a defect, a green suite gets
read as fidelity.

So every run ends in one of three verdicts, and the rules that produce them are
in code rather than in prose:

* **NOT_READY** - at least one confirmed mismatch. Something is wrong.
* **INCONCLUSIVE** - no confirmed mismatch, but some mechanic was never proven
  reachable, so its silence means nothing. This is the honest default.
* **READY** - every checked mechanic was reachable and none mismatched.

The ordering is deliberate: **INCONCLUSIVE outranks READY**. A check that could
not run is not a check that passed, and treating it as one is how a false green
reaches training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Finding(Enum):
    """Classification of a single result. Mirrors the register's taxonomy."""

    MISMATCH = 1          # confirmed JAX/native divergence
    MISSING_CAPABILITY = 2  # Gym or bridge does not implement it
    INVALID_SCENE = 3     # ADK scene lacks a provider, loadout or geometry
    STALE_IDENTITY = 4    # jar, pin, contract or listener drift
    INCONCLUSIVE = 5      # reachability not established


class Verdict(Enum):
    READY = "ready"
    NOT_READY = "not ready"
    INCONCLUSIVE = "inconclusive"


#: Classes that are ADK-side or infrastructural.  They invalidate a run without
#: implying anything about fidelity, and must never read as a mismatch.
_OUR_FAULT = frozenset({Finding.INVALID_SCENE, Finding.STALE_IDENTITY})


@dataclass(frozen=True, slots=True)
class Check:
    """One mechanic's result.

    ``reachable`` is the load-bearing field.  A check that never proved the
    mechanic ran cannot contribute a pass, whatever its outcome looked like.
    """

    name: str
    reachable: bool
    finding: Finding | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a check must be named")
        if self.finding is Finding.MISMATCH and not self.reachable:
            raise ValueError(
                f"check {self.name!r} claims a mismatch without proving the "
                "mechanic ran; an unreachable mechanic cannot mismatch"
            )


@dataclass(frozen=True, slots=True)
class Report:
    """The verdict plus the reasoning that produced it."""

    verdict: Verdict
    checks: tuple[Check, ...] = field(default_factory=tuple)

    @property
    def mismatches(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.finding is Finding.MISMATCH)

    @property
    def unreachable(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.reachable)

    @property
    def our_fault(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.finding in _OUR_FAULT)

    def summary(self) -> str:
        lines = [f"VERDICT: {self.verdict.value.upper()}  ({len(self.checks)} checks)"]
        if self.mismatches:
            lines.append(f"  confirmed mismatches ({len(self.mismatches)}):")
            lines += [f"    - {c.name}: {c.detail}" for c in self.mismatches]
        if self.unreachable:
            lines.append(f"  never proven reachable ({len(self.unreachable)}):")
            lines += [f"    - {c.name}: {c.detail}" for c in self.unreachable]
        if self.our_fault:
            lines.append(f"  ADK-side, not fidelity ({len(self.our_fault)}):")
            lines += [f"    - {c.name}: class {c.finding.value}, {c.detail}"
                      for c in self.our_fault]
        return "\n".join(lines)


def decide(checks) -> Report:
    """Reduce checks to a verdict.

    A run with no checks is INCONCLUSIVE, not READY -- vacuous success is the
    failure mode this whole module exists to prevent.
    """

    checks = tuple(checks)
    if any(c.finding is Finding.MISMATCH for c in checks):
        return Report(Verdict.NOT_READY, checks)
    if not checks or any(not c.reachable for c in checks):
        return Report(Verdict.INCONCLUSIVE, checks)
    return Report(Verdict.READY, checks)
