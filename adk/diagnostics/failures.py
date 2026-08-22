"""Decode the environment's failure bitfields into named flags.

The observation carries three ``uint32`` bitfields -- ``failure_bits``,
``mechanics_failure_bits``, ``arsenal_failure_bits`` -- and every bit has a
published name in a Gym contract module.  Without a decoder those fields are
opaque integers, so a silent misbehaviour (an invalid loadout, an overflowed
ability buffer, an unsupported world) looks like "the agent just isn't
learning".

Two paths, deliberately separated:

* :func:`failure_flags` is traced.  Each flag is one bitwise-AND against a
  constant, computed for the whole batch at once, and nothing synchronizes with
  the device.  Safe inside ``jit``/``scan``; use it to record per-step
  diagnostics alongside reward.
* :func:`explain_failures` is host-side and says so: it reduces to Python
  strings and therefore stalls.

The bit -> name tables are read from the Gym contracts at import, so a new
upstream failure mode appears here automatically rather than silently widening
an integer nobody reads.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from hytalegym.jax.combat import mechanics as _mechanics_module
try:
    # The Gym's restructure moved this to `arsenal.schema.contract`. The parent
    # package star-exports its *names*, which is enough for `from ... import
    # ARSENAL_FAILURE_X` but not for importing the module object -- and
    # `_bit_table` below needs the module to enumerate its constants. Both
    # spellings are kept so this imports against the live tree and the frozen
    # snapshot alike.
    from hytalegym.jax.combat.arsenal.schema import contract as _arsenal_contract
except ImportError:  # pragma: no cover - depends on which Gym tree is on the path
    from hytalegym.jax.combat.arsenal import contract as _arsenal_contract
try:
    from hytalegym.jax.combat.mechanics.schema import contract as _mechanics_contract
except ImportError:  # pragma: no cover - depends on the Gym tree on the path
    from hytalegym.jax.combat.mechanics import contract as _mechanics_contract
try:
    from hytalegym.jax.combat.observation.v1.schema import contract as _capacity_contract
except ImportError:  # pragma: no cover - depends on the Gym tree on the path
    from hytalegym.jax.combat.observation import contract as _capacity_contract
try:
    from hytalegym.jax.combat.observation.v3.schema import contract as _observation_contract
except ImportError:  # pragma: no cover - depends on the Gym tree on the path
    from hytalegym.jax.combat.observation.v3 import contract as _observation_contract


def _bit_table(module: Any, prefix: str) -> Mapping[str, int]:
    """Collect ``PREFIX_NAME = 1 << n`` constants into ``{name: mask}``."""

    table = {}
    for attribute in dir(module):
        if not attribute.startswith(prefix):
            continue
        value = getattr(module, attribute)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            # Only single-bit masks are individual failure modes.
            if value & (value - 1) == 0:
                table[attribute[len(prefix) :].lower()] = value
    if not table:
        raise RuntimeError(f"no {prefix}* bit constants found in {module.__name__}")
    return MappingProxyType(dict(sorted(table.items(), key=lambda item: item[1])))


#: ``failure_bits`` -- observation-level causes.
OBSERVATION_FAILURES = _bit_table(_observation_contract, "OBSERVATION_FAILURE_")
#: ``mechanics_failure_bits`` -- status/dodge/command level.
MECHANICS_FAILURES = _bit_table(_mechanics_contract, "MECHANICS_FAILURE_")
#: ``arsenal_failure_bits`` -- loadout/ability/projectile level.
ARSENAL_FAILURES = _bit_table(_arsenal_contract, "ARSENAL_FAILURE_")
#: ``base.overflow_bits`` -- which padded set ran out of slots.
#:
#: Not a malfunction: it means more of something existed than the observation
#: can carry, so the agent is deciding on a *truncated* view of the world.  A
#: policy that never sees the third hazard cannot learn to avoid it, and
#: nothing else in the observation reveals the loss.
CAPACITY_OVERFLOWS = _bit_table(_capacity_contract, "OVERFLOW_")

#: Observation field name -> ``(reporting prefix, decode table)``.  The pairing
#: is fixed by ``observation/v3/encoder.py:462-463``, where mechanics and
#: arsenal bits are copied straight off their subsystem states.  Dotted paths
#: are resolved attribute by attribute, because overflow lives on ``base``.
FAILURE_FIELDS: Mapping[str, tuple[str, Mapping[str, int]]] = MappingProxyType(
    {
        "failure_bits": ("observation", OBSERVATION_FAILURES),
        "mechanics_failure_bits": ("mechanics", MECHANICS_FAILURES),
        "arsenal_failure_bits": ("arsenal", ARSENAL_FAILURES),
    }
)

#: Reported alongside failures but deliberately *not* part of them: a crowded
#: scene overflows every step by design, so folding it into
#: :func:`any_failure` would make that gate useless.
OVERFLOW_FIELDS: Mapping[str, tuple[str, Mapping[str, int]]] = MappingProxyType(
    {"base.overflow_bits": ("overflow", CAPACITY_OVERFLOWS)}
)

#: Everything decodable, for reporting.
DIAGNOSTIC_FIELDS: Mapping[str, tuple[str, Mapping[str, int]]] = MappingProxyType(
    {**FAILURE_FIELDS, **OVERFLOW_FIELDS}
)

del _mechanics_module


def _resolve(observation: Any, path: str) -> Any:
    """Follow a dotted attribute path, returning ``None`` if any hop is absent."""

    node = observation
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def failure_flags(bits: Any, table: Mapping[str, int]) -> dict[str, jax.Array]:
    """Decode one bitfield into ``{name: bool array}``.  Traced, no sync.

    Each entry is a single bitwise-AND over the whole batch, so decoding every
    flag costs one pass regardless of how many bits the table holds.
    """

    values = jnp.asarray(bits)
    return {
        name: (values & jnp.asarray(mask, dtype=values.dtype)) != 0
        for name, mask in table.items()
    }


def observation_failures(legal_observation: Any) -> dict[str, jax.Array]:
    """Every failure flag from every bitfield, prefixed by its field.

    Traced and allocation-light: one AND per bit, no reductions, no transfers.
    Record this next to reward and the causes of a bad run become visible
    without a second rollout.
    """

    flags: dict[str, jax.Array] = {}
    for field, (prefix, table) in DIAGNOSTIC_FIELDS.items():
        bits = _resolve(legal_observation, field)
        if bits is None:
            continue
        for name, value in failure_flags(bits, table).items():
            flags[f"{prefix}.{name}"] = value
    return flags


def any_failure(legal_observation: Any) -> jax.Array:
    """A single per-row boolean: did anything fail this step?  Traced.

    Cheaper than :func:`observation_failures` when you only want a gate --
    it ORs the raw bitfields instead of decoding them.
    """

    total = None
    for field in FAILURE_FIELDS:
        bits = _resolve(legal_observation, field)
        if bits is None:
            continue
        value = jnp.asarray(bits)
        total = value if total is None else (total | value)
    if total is None:
        raise ValueError("observation carries no failure bitfields")
    return total != 0


def explain_failures(legal_observation: Any, row: int | None = None) -> list[str]:
    """Names of the failures currently set, as strings.

    **Host-side.**  Reduces arrays to Python values and therefore synchronizes;
    call it when something looks wrong, not every step.  ``row`` selects one
    environment; omitting it reports any row.
    """

    named: list[str] = []
    for name, flag in observation_failures(legal_observation).items():
        value = flag if row is None else flag[row]
        if bool(jnp.any(value)):
            named.append(name)
    return named


def failure_names() -> dict[str, tuple[str, ...]]:
    """Every decodable name, keyed by reporting prefix.  Pure Python."""

    return {prefix: tuple(table) for prefix, table in DIAGNOSTIC_FIELDS.values()}


def capacity_overflows(legal_observation: Any) -> dict[str, jax.Array]:
    """Which padded sets ran out of slots this step.  Traced, no sync.

    ``True`` means the observation dropped real world content, so any policy
    reading it is deciding on partial information.  Worth recording when a
    scene's entity or hazard count approaches its capacity.
    """

    bits = _resolve(legal_observation, "base.overflow_bits")
    if bits is None:
        raise ValueError("observation carries no overflow bitfield")
    return failure_flags(bits, CAPACITY_OVERFLOWS)


__all__ = [
    "ARSENAL_FAILURES",
    "CAPACITY_OVERFLOWS",
    "DIAGNOSTIC_FIELDS",
    "FAILURE_FIELDS",
    "MECHANICS_FAILURES",
    "OBSERVATION_FAILURES",
    "OVERFLOW_FIELDS",
    "any_failure",
    "capacity_overflows",
    "explain_failures",
    "failure_flags",
    "failure_names",
    "observation_failures",
]
