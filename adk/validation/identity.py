"""Run identity — what was true when a measurement was taken.

A fidelity number is meaningless without the identity of the thing measured.
This project has already spent real time on results that were void because a
jar, a pin or a head layout moved underneath them: 17 of 24 native failures
once traced to a single stale build artifact rather than any fidelity gap.

So every probe stamps identity **before and after**.  If it moved, the run is
class 4 (stale identity), not class 1 (mismatch), and its numbers are discarded
rather than reported.

    before = capture()
    ... run the probe ...
    assert_stable(before, capture())

Everything here is derived from the code as it currently is.  Nothing is
hard-coded, because a hard-coded expectation is the drift it is meant to catch.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

GYM_ROOT = Path(__file__).resolve().parents[2] / "HytaleRL" / "hytalegym"


class IdentityDrift(RuntimeError):
    """Raised when a run's identity moved mid-measurement."""


@dataclass(frozen=True, slots=True)
class Identity:
    """The identity tuple a measurement is only valid against."""

    gym_git_sha: str
    gym_source_digest: str
    #: What the Gym *expects* -- an imported constant, not a file on disk.
    bridge_jar_sha256: str
    #: What is actually deployed, hashed from the jar bytes. ``"unknown"``
    #: where no jar is reachable. See :func:`_deployed_jar_sha256` for why
    #: this exists separately from the pin above.
    deployed_jar_sha256: str
    action_head_sizes: tuple[int, ...]
    action_logits: int
    region_contract: tuple[int, ...]

    @property
    def jar_pin_matches_deployment(self) -> bool | None:
        """Whether the pin describes the jar actually on disk.

        ``None`` when no jar was reachable, which is not the same as a
        mismatch and must not be reported as one.
        """

        if self.deployed_jar_sha256 == "unknown":
            return None
        return self.bridge_jar_sha256 == self.deployed_jar_sha256

    def summary(self) -> str:
        agrees = self.jar_pin_matches_deployment
        jar = (
            f"jar={self.bridge_jar_sha256[:8]}"
            if agrees is not False
            else f"jar=PIN:{self.bridge_jar_sha256[:8]}"
               f"!=RUNNING:{self.deployed_jar_sha256[:8]}"
        )
        return (
            f"gym={self.gym_git_sha[:12]} src={self.gym_source_digest[:8]} "
            f"{jar} "
            f"heads={len(self.action_head_sizes)} logits={self.action_logits}"
        )


def _git_sha() -> str:
    """The Gym's commit, or ``"unknown"`` where git cannot answer.

    ``~/hydl`` itself is not a repository, so this deliberately asks the Gym
    checkout rather than the working directory.
    """

    try:
        result = subprocess.run(
            ["git", "-C", str(GYM_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _deployed_jar_sha256() -> str:
    """SHA-256 of the bridge jar actually on disk, or ``"unknown"``.

    **The pin is not the deployment.** ``bridge_jar_sha256`` above is
    ``CURRENT_NATIVE_EVIDENCE_JAR_SHA256`` -- a constant imported from the Gym.
    Every run stamped it as though it identified the jar being executed, and
    ``assert_stable`` could never catch the difference because both captures
    read the same constant. On 2026-08-09 the pin was ``A81720ED`` while the
    shared installed bridge was still ``D954BD79``: every ADK native run was
    recording a jar hash it was not running.

    This is the same mistake ``_source_digest`` exists to fix one level up --
    "a commit identifies what was committed, not what is imported" -- repeated
    for the jar. Hashing the artifact identifies what is loaded.

    Degrades rather than raising: no ``APPDATA``, no file, or an unreadable
    file all return ``"unknown"``, because a machine with no deployed bridge is
    a normal state and not a drift.
    """

    try:
        from adk.probes.crosslang import deployed_jar_path

        path = deployed_jar_path()
    except Exception:
        return "unknown"
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
    except OSError:
        return "unknown"


def _source_digest() -> str:
    """SHA-256 over every ``.py`` the Gym will actually execute.

    **The git SHA is not enough.** On 2026-08-04 the Gym's HEAD read
    ``bd4ac74`` for a whole session while **45 tracked source files were
    modified in the working tree** -- including ``arsenal/ability.py``,
    ``arsenal/environment.py``, ``combat/env.py`` and
    ``observation/encoder.py``. Every run reported "identity stable" while the
    code changed underneath. A commit identifies what was committed, not what
    is imported.

    Hashing file contents identifies the latter.
    """

    package = GYM_ROOT / "hytalegym"
    if not package.is_dir():
        return "unknown"

    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        try:
            body = path.read_bytes()
        except OSError:
            continue
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(hashlib.sha256(body).digest())
    return digest.hexdigest().upper()


def capture() -> Identity:
    """Read the current identity out of live code."""

    from hytalegym.jax.combat import ARSENAL_POLICY_ACTION_HEAD_SIZES
    from hytalegym.worldgen.region import contract as region_contract
    from hytalegym.worldgen.region.stability import (
        CURRENT_NATIVE_EVIDENCE_JAR_SHA256,
    )

    sizes = tuple(int(size) for size in ARSENAL_POLICY_ACTION_HEAD_SIZES)
    return Identity(
        gym_git_sha=_git_sha(),
        gym_source_digest=_source_digest(),
        bridge_jar_sha256=CURRENT_NATIVE_EVIDENCE_JAR_SHA256,
        deployed_jar_sha256=_deployed_jar_sha256(),
        action_head_sizes=sizes,
        action_logits=sum(sizes),
        region_contract=(
            int(region_contract.CAPTURE_CHUNK_COUNT),
            int(region_contract.CAPTURE_CHUNKS_PER_AXIS),
            int(region_contract.CHUNK_SIZE),
            int(region_contract.HEIGHT_SECTIONS),
            int(region_contract.MIN_Y),
        ),
    )


def assert_stable(before: Identity, after: Identity) -> None:
    """Void the run if any identity field moved while it was measuring."""

    if before == after:
        return
    moved = [
        f"  {field}: {getattr(before, field)!r} -> {getattr(after, field)!r}"
        for field in Identity.__dataclass_fields__
        if getattr(before, field) != getattr(after, field)
    ]
    raise IdentityDrift(
        "identity moved during the run; its measurements are void (class 4, "
        "stale identity -- NOT a fidelity mismatch):\n" + "\n".join(moved)
    )
