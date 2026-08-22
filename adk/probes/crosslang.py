"""Fidelity between the Java bridge's declared constants and the JAX contract.

The Gym's differential suite (``hytalegym/tests/fidelity/``) compares *running*
native behaviour against JAX. That is the strong oracle, and it needs a live
server plus the native lease on ``127.0.0.1:5556``.

This module covers a different and much cheaper axis: the bridge's Java source
declares constants -- authored dodge directions, interaction ids, direction
numbering -- and the JAX contract declares the same facts independently. If one
side is edited and the other is not, the two drift apart silently. A runtime
test only catches that if the drifted path happens to be exercised; a source
comparison catches it always, in milliseconds, with no lease and no server.

Deliberately narrow. It reads ``public static final`` declarations, which are
unambiguous, and refuses to guess at anything else. When the Java tree is
absent it reports "not evaluated" rather than passing, because a fidelity check
that silently succeeds when it cannot see one side is worse than no check.
"""

from __future__ import annotations

import hashlib
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping, NamedTuple

#: Where the bridge's Java sources live relative to the repository root.
BRIDGE_SOURCE_ROOT = Path("HytaleRL/hytale-plugin/src/main/java/com/hytalerlbridge")

#: Rulesets JAX loads.  ``hytale-plugin/build.gradle`` copies this same
#: directory into the jar, so the two sides share one source by construction --
#: which is exactly why the interesting question is whether the *deployed* jar
#: is current, not whether the trees agree.
GYM_RULESET_ROOT = Path("HytaleRL/hytalegym/hytalegym/rulesets")

#: Prefix of the ruleset entries inside the packaged jar.
JAR_RULESET_PREFIX = "hytale_0_5_7/"

_DECLARATION = re.compile(
    r"public\s+static\s+final\s+"
    r"(?P<type>int|long|float|double|boolean|String)\s+"
    r"(?P<name>[A-Z_][A-Z0-9_]*)\s*=\s*"
    r"(?P<value>\"[^\"]*\"|[-+0-9.eEfdDL]+|true|false)\s*;",
    re.MULTILINE,
)


class MissingSource(RuntimeError):
    """The Java side could not be read, so nothing was compared."""


def repository_root(start: Path | None = None) -> Path:
    """Find the tree containing both ``agent/`` and ``HytaleRL/``."""

    here = (start or Path(__file__)).resolve()
    for candidate in here.parents:
        if (candidate / "HytaleRL").is_dir() and (candidate / "adk").is_dir():
            return candidate
    raise MissingSource(
        f"no repository root containing both agent/ and HytaleRL/ above {here}"
    )


def _coerce(kind: str, raw: str) -> Any:
    if kind == "String":
        return raw[1:-1]
    if kind == "boolean":
        return raw == "true"
    if kind in ("int", "long"):
        return int(raw.rstrip("lL"))
    return float(raw.rstrip("fdFD"))


def java_constants(relative_path: str, *, root: Path | None = None) -> dict[str, Any]:
    """Every ``public static final`` primitive declared in one Java file.

    Raises :class:`MissingSource` when the file is absent, and raises when the
    file exists but declares nothing -- an empty result almost always means the
    regex stopped matching after a refactor, not that the constants went away.
    """

    base = root or repository_root()
    path = base / BRIDGE_SOURCE_ROOT / relative_path
    if not path.is_file():
        raise MissingSource(f"bridge source absent: {path}")

    text = path.read_text(encoding="utf-8")
    found = {
        match.group("name"): _coerce(match.group("type"), match.group("value"))
        for match in _DECLARATION.finditer(text)
    }
    if not found:
        raise MissingSource(
            f"no 'public static final' constants parsed from {path}; the "
            "declaration style changed and this reader needs updating"
        )
    return found


class Comparison(NamedTuple):
    """One cross-language fact and whether the two sides agree."""

    name: str
    java: Any
    jax: Any

    @property
    def agrees(self) -> bool:
        return self.java == self.jax


def compare(pairs: Mapping[str, tuple[Any, Any]]) -> tuple[Comparison, ...]:
    """Build comparisons from ``{name: (java_value, jax_value)}``."""

    return tuple(
        Comparison(name=name, java=java, jax=jax)
        for name, (java, jax) in pairs.items()
    )


def disagreements(comparisons: tuple[Comparison, ...]) -> tuple[Comparison, ...]:
    """Only the facts where the two sides differ."""

    return tuple(item for item in comparisons if not item.agrees)


def describe(comparisons: tuple[Comparison, ...]) -> str:
    """Host-side summary, aligned for reading."""

    bad = disagreements(comparisons)
    lines = [
        f"{len(comparisons)} cross-language fact(s), {len(bad)} disagreement(s)"
    ]
    for item in comparisons:
        marker = "  " if item.agrees else "!!"
        lines.append(
            f" {marker} {item.name:34s} java={item.java!r:24s} jax={item.jax!r}"
        )
    return "\n".join(lines)


def dodge_comparisons(*, root: Path | None = None) -> tuple[Comparison, ...]:
    """Compare the authored-dodge contract across Java and JAX.

    ``NativeDodgeProgram.actionMask`` returns ``[false, false, left, right]``
    and documents that order as ``[forward, back, left, right]``; JAX declares
    the same shape as ``DODGE_AUTHORED_ACTION_MASK``. Both sides also number
    the directions, and both must agree that only left and right are authored.
    """

    from hytalegym.jax.combat.mechanics import (
        DODGE_AUTHORED_ACTION_MASK,
        DODGE_DIRECTION_COUNT,
        DODGE_LEFT,
        DODGE_RIGHT,
    )

    java = java_constants("combat/dodge/NativeDodgeProgram.java", root=root)
    jax_authored = tuple(bool(x) for x in DODGE_AUTHORED_ACTION_MASK)

    return compare(
        {
            # Java counts real directions; JAX includes DODGE_NONE, so the JAX
            # count is one larger by construction.
            "dodge_direction_count": (
                java["DIRECTION_COUNT"],
                DODGE_DIRECTION_COUNT - 1,
            ),
            "dodge_left_index": (java["LEFT_DIRECTION"], DODGE_LEFT),
            "dodge_right_index": (java["RIGHT_DIRECTION"], DODGE_RIGHT),
            # Only left and right carry a payload interaction on either side.
            "authored_directions": (
                (java["LEFT_DIRECTION"], java["RIGHT_DIRECTION"]),
                tuple(
                    index
                    for index, authored in enumerate(jax_authored, start=1)
                    if authored
                ),
            ),
            "authored_mask": (
                (False, False, True, True),
                jax_authored,
            ),
        }
    )


def geometry_comparisons(*, root: Path | None = None) -> tuple[Comparison, ...]:
    """Compare the geometry contract across Java and JAX.

    Both sides declare the sampling cube independently and both derive side and
    cell count from the radius, so a radius change on one side alone shows up
    here as three disagreements rather than one.
    """

    from hytalegym.geometry.contract import (
        CELL_COUNT,
        CELL_RADIUS,
        CELL_SIDE,
        GEOMETRY_SCHEMA,
        GEOMETRY_VERSION,
        MAX_DETAIL_BOXES,
    )

    java = java_constants("geometry/GeometryContract.java", root=root)
    radius = java["RADIUS"]
    side = radius * 2 + 1

    return compare(
        {
            "geometry_schema": (java["SCHEMA"], GEOMETRY_SCHEMA),
            "geometry_version": (java["VERSION"], GEOMETRY_VERSION),
            "geometry_radius": (radius, CELL_RADIUS),
            # SIDE and CELL_COUNT are derived expressions in the Java source,
            # so they are recomputed here rather than parsed.
            "geometry_side": (side, CELL_SIDE),
            "geometry_cell_count": (side**3, CELL_COUNT),
            # Sets the world-token width: 8 + edge_capacity*5 + boxes*7.
            "max_detail_boxes": (java["MAX_DETAIL_BOXES_0_5_7"], MAX_DETAIL_BOXES),
        }
    )


def world_verb_comparisons(*, root: Path | None = None) -> tuple[Comparison, ...]:
    """Compare the native world-verb wire contract across Java and JAX.

    The strongest check here: Java **hardcodes** ``CONTRACT_SHA256`` while JAX
    **computes** it from its own schema definition. They agreeing means the two
    sides independently derived the same wire contract; they disagreeing means
    requests would be rejected or silently mis-parsed.
    """

    from hytalegym.worldgen.native_world_actions import (
        NATIVE_WORLD_VERB_TRANSPORT_SCHEMA,
        NATIVE_WORLD_VERB_TRANSPORT_VERSION,
        native_world_verb_transport_contract_sha256,
    )

    java = java_constants("action/NativeWorldVerbRequest.java", root=root)

    return compare(
        {
            "world_verb_schema": (
                java["SCHEMA"],
                NATIVE_WORLD_VERB_TRANSPORT_SCHEMA,
            ),
            "world_verb_version": (
                java["VERSION"],
                NATIVE_WORLD_VERB_TRANSPORT_VERSION,
            ),
            "world_verb_contract_sha256": (
                java["CONTRACT_SHA256"].upper(),
                native_world_verb_transport_contract_sha256().upper(),
            ),
        }
    )


def combat_ruleset_comparisons(*, root: Path | None = None) -> tuple[Comparison, ...]:
    """Compare the combat-ruleset identity across Java and JAX."""

    from hytalegym.rulesets.loader import (
        COMBAT_RULESET_RESOURCE,
        _COMBAT_SCHEMA,
        _COMBAT_VERSION,
    )

    java = java_constants("combat/CombatRuleset.java", root=root)
    # The resource path is declared with a leading slash on the Java side
    # because it is a classpath lookup; JAX names the same file relatively.
    java_resource = java["RESOURCE"].lstrip("/")

    return compare(
        {
            "combat_ruleset_schema": (java["SCHEMA"], _COMBAT_SCHEMA),
            "combat_ruleset_version": (java["VERSION"], _COMBAT_VERSION),
            "combat_ruleset_resource": (java_resource, COMBAT_RULESET_RESOURCE),
        }
    )


def all_comparisons(*, root: Path | None = None) -> tuple[Comparison, ...]:
    """Every cross-language fact this module knows how to check."""

    return (
        dodge_comparisons(root=root)
        + geometry_comparisons(root=root)
        + world_verb_comparisons(root=root)
        + combat_ruleset_comparisons(root=root)
    )


def _select_pinned_bridge_jar(
    candidates: tuple[Path, ...], expected_sha256: str
) -> Path:
    """Select deterministically by content, never by an assumed server path."""

    expected = expected_sha256.strip().upper()
    if len(expected) != 64 or any(ch not in "0123456789ABCDEF" for ch in expected):
        raise ValueError("expected bridge SHA-256 must be 64 uppercase hex characters")
    examined: list[str] = []
    matches: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        key = str(path).casefold()
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        examined.append(f"{path}={digest}")
        if digest == expected:
            matches.append(path)
    if not matches:
        detail = "; ".join(examined) if examined else "no candidate files found"
        raise MissingSource(
            f"no bridge JAR matches canonical pin {expected}; examined {detail}"
        )
    # Candidate order encodes ownership: an explicit exact override first,
    # then sealed artifacts, and mutable runtime/install/build paths last.
    return matches[0]


def _bridge_jar_candidates(root: Path) -> tuple[Path, ...]:
    override = os.environ.get("HYTALERL_NATIVE_BRIDGE_JAR")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.extend(
        sorted(
            root.glob("HytaleRL/artifacts/**/HytaleRLBridge-0.1.0*.jar"),
            key=lambda path: str(path).casefold(),
        )
    )
    candidates.extend(
        sorted(
            root.glob(
                "artifacts/console-hytale/*/mods/HytaleRLBridge-0.1.0.jar"
            ),
            key=lambda path: str(path).casefold(),
        )
    )
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(
            Path(appdata)
            / "Hytale/install/release/package/game/latest/Server/mods"
            / "HytaleRLBridge-0.1.0.jar"
        )
    candidates.append(
        root / "HytaleRL/hytale-plugin/build/libs/HytaleRLBridge-0.1.0.jar"
    )
    return tuple(candidates)


def deployed_jar_path(*, root: Path | None = None) -> Path:
    """Find bytes matching the canonical live bridge pin.

    A path alone cannot establish what an isolated server loaded. The native
    identity gate proves the listener reports the canonical hash; this helper
    then locates any local artifact with those exact bytes for offline class
    and ruleset inspection. ``HYTALERL_NATIVE_BRIDGE_JAR`` may add an explicit
    candidate, but a mismatched override never bypasses the content check.
    """

    workspace = root or repository_root()
    try:
        from hytalegym.worldgen.region.stability import (
            current_native_evidence_jar_sha256,
        )
    except ImportError as error:
        raise MissingSource("installed hytalegym lacks the canonical bridge pin") from error
    return _select_pinned_bridge_jar(
        _bridge_jar_candidates(workspace), current_native_evidence_jar_sha256()
    )


class RulesetParity(NamedTuple):
    """One ruleset, as the pinned JAR carries it and as JAX loads it."""

    entry: str
    jar_sha256: str
    gym_sha256: str

    @property
    def agrees(self) -> bool:
        return self.jar_sha256 == self.gym_sha256


class JarConstantParity(NamedTuple):
    """One compiled bridge constant and the Java source that declares it."""

    name: str
    jar: str
    source: str

    @property
    def agrees(self) -> bool:
        return self.jar == self.source


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def jar_constant_parity(
    *, jar: Path | None = None, root: Path | None = None
) -> tuple[JarConstantParity, ...]:
    """Compare release-critical constants in a bridge JAR with Java source.

    Source-to-JAX comparisons cannot detect older pinned bytecode. This check
    opens content-addressed bytes and compares compiled constants against the
    source tree that a rebuild would use. A separate live reset gate proves
    that the listener actually reports the same canonical content hash.
    """

    archive = jar or deployed_jar_path(root=root)
    if not archive.is_file():
        raise MissingSource(f"pinned bridge JAR absent: {archive}")
    try:
        from hytalegym.jax.combat.contracts.publication import (
            bridge_actor_evidence_contract_sha256,
            bridge_privileged_npc_component_filter,
        )
    except ImportError as error:
        raise MissingSource(
            "installed hytalegym lacks bridge constant-pool accessors"
        ) from error

    actor_source = java_constants(
        "observation/NativeActorEvidenceFrame.java", root=root
    )
    npc_source = java_constants("entity/PrivilegedNpcSnapshot.java", root=root)
    return (
        JarConstantParity(
            name="NativeActorEvidenceFrame.CONTRACT_SHA256",
            jar=bridge_actor_evidence_contract_sha256(archive),
            source=str(actor_source["CONTRACT_SHA256"]).upper(),
        ),
        JarConstantParity(
            name="PrivilegedNpcSnapshot.COMPONENT_FILTER",
            jar=bridge_privileged_npc_component_filter(archive),
            source=str(npc_source["COMPONENT_FILTER"]),
        ),
    )


def jar_ruleset_parity(
    *, jar: Path | None = None, root: Path | None = None
) -> tuple[RulesetParity, ...]:
    """Compare rulesets inside the pinned bridge bytes against what JAX loads.

    This is the check the jar hash cannot make. ``test_bridge_fidelity``
    verifies the listener reports the hash the ADK stamps. This function finds
    bytes with that exact hash, but then asks a different question: whether
    their embedded rules match the ones JAX simulates.

    Raises :class:`MissingSource` when the jar or the gym tree is absent, so a
    caller can report "not evaluated" rather than passing blindly.
    """

    archive = jar or deployed_jar_path(root=root)
    if not archive.is_file():
        raise MissingSource(f"pinned bridge JAR absent: {archive}")
    gym = (root or repository_root()) / GYM_RULESET_ROOT
    if not gym.is_dir():
        raise MissingSource(f"gym ruleset tree absent: {gym}")

    results: list[RulesetParity] = []
    with zipfile.ZipFile(archive) as bundle:
        entries = sorted(
            name
            for name in bundle.namelist()
            if name.startswith(JAR_RULESET_PREFIX) and name.endswith(".json")
        )
        if not entries:
            raise MissingSource(
                f"no {JAR_RULESET_PREFIX}*.json entries in {archive}; the jar "
                "layout changed and this reader needs updating"
            )
        for entry in entries:
            source = gym / entry
            results.append(
                RulesetParity(
                    entry=entry,
                    jar_sha256=_sha256(bundle.read(entry)),
                    gym_sha256=_sha256(source.read_bytes())
                    if source.is_file()
                    else "",
                )
            )
    return tuple(results)


def describe_parity(results: tuple[RulesetParity, ...]) -> str:
    """Host-side summary of :func:`jar_ruleset_parity`."""

    stale = [item for item in results if not item.agrees]
    lines = [f"{len(results)} ruleset(s) in the pinned JAR, {len(stale)} stale"]
    for item in results:
        marker = "  " if item.agrees else "!!"
        detail = item.jar_sha256[:24] if item.agrees else (
            f"jar={item.jar_sha256[:16]} gym={item.gym_sha256[:16] or '<absent>'}"
        )
        lines.append(f" {marker} {item.entry:52s} {detail}")
    return "\n".join(lines)


def describe_constant_parity(results: tuple[JarConstantParity, ...]) -> str:
    """Host-side summary of :func:`jar_constant_parity`."""

    stale = [item for item in results if not item.agrees]
    lines = [
        f"{len(results)} compiled bridge constant(s), {len(stale)} stale"
    ]
    for item in results:
        marker = "  " if item.agrees else "!!"
        detail = item.jar if item.agrees else (
            f"jar={item.jar!r} source={item.source!r}"
        )
        lines.append(f" {marker} {item.name:48s} {detail}")
    return "\n".join(lines)


__all__ = [
    "BRIDGE_SOURCE_ROOT",
    "GYM_RULESET_ROOT",
    "JAR_RULESET_PREFIX",
    "Comparison",
    "JarConstantParity",
    "MissingSource",
    "RulesetParity",
    "compare",
    "deployed_jar_path",
    "describe",
    "describe_constant_parity",
    "describe_parity",
    "disagreements",
    "dodge_comparisons",
    "jar_ruleset_parity",
    "jar_constant_parity",
    "java_constants",
    "repository_root",
]
