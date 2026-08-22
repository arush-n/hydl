"""Shared primitives for the arena skill contracts.

Every contract module in this package is named after its lesson (``guard_duel``,
``evasion``, ...) and publishes exactly four things under that name:

``<Lesson>Config``
    A frozen, slot-based dataclass of tunable weights. It validates itself in
    ``__post_init__``, publishes a ``manifest()`` built by
    :func:`normalized_manifest`, and content-addresses that manifest as
    ``contract_sha256`` so a run can record exactly which reward law produced
    it.

``<Lesson>Signals``
    A ``NamedTuple`` whose first field is ``reward``. The remaining fields are
    the evidence behind that number, so a rollout can log *why* a lane scored
    rather than only what it scored. A two-sided lesson leads with one reward
    per role instead: ``GuardDuelSignals`` is
    ``(attacker_reward, blocker_reward, ...)``.

``<lesson>_signals(config, *, ...)``
    A pure, JIT-safe function over lane-shaped ``[B]`` arrays. It takes explicit
    evidence rather than reading state, so it can be unit-tested without an
    arena and reused by any collector.

``<lesson>_action_scope(...)``
    The heads the lesson trains. Every other head is pinned to its no-op, so a
    contract cannot accidentally train a mechanic it does not score. A
    two-sided lesson publishes one scope per role, still prefixed with the
    module name (``guard_duel_attacker_action_scope``).

Deliberate deviation: :mod:`arena.training.contracts.pursuit` predates this
convention and keeps its original ``PursuitStageConfig`` /
``pursuit_lesson_signals`` / ``pursuit_learner_action_scope`` names, because
they are already referenced outside this package. It is the only module here
that also owns transition transforms and a collector binding.

Reward conventions used throughout, chosen to resist farming:

* **Clip every progress term.** An unbounded potential difference lets one
  lucky tick dominate an episode.
* **Deadband before paying.** Sub-threshold jitter must earn nothing, or a
  policy learns to vibrate.
* **Charge for activation, pay for outcome.** Abilities, guards and dodges cost
  something to press and only pay when engine evidence says they worked.
* **Charge per tick.** Without it, a safe idle policy is optimal.
* **Never pay for a state the opponent produced.** Credit follows the learner's
  own contribution wherever a counterfactual is available.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Iterable, Mapping

import jax
import jax.numpy as jnp

from arena.training.skills.program import FactoredActionScope
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)


#: Heads whose physical no-op is the middle bin rather than index zero.
_CENTRED_HEADS = frozenset(
    {"yaw_delta_bins", "body_yaw_delta_bins", "pitch_delta_bins"}
)

#: Heads every contract leaves open so a lane can always steer and reposition.
#: The body steer belongs here as much as the camera does: travel rides the body,
#: so a lane that cannot turn its chest cannot reposition at all.
LOCOMOTION_HEADS = (
    "locomotion_gait_compass",
    "yaw_delta_bins",
    "body_yaw_delta_bins",
    "pitch_delta_bins",
)


def contract_hash(value: Any) -> str:
    """Content-address a manifest the same way every other arena stage does."""

    return (
        hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
        .hexdigest()
        .upper()
    )


def normalized_manifest(
    *,
    schema: str,
    objective: str,
    reward: Mapping[str, Any],
    cost: Mapping[str, Any],
    anti_farm: str,
    horizon: "HorizonConfig",
    **sections: Any,
) -> dict[str, object]:
    """Assemble the manifest skeleton every contract in this package publishes.

    Keeping the skeleton in one place is what makes the contracts comparable:
    two lessons can be diffed key-by-key, and ``reward`` versus ``cost`` always
    means "terms that add" versus "terms that subtract" rather than whichever
    grouping each module happened to pick. ``sections`` carries the parts that
    are genuinely lesson-specific (a distance band, ability slots, an opponent
    description) and is merged in at the top level.

    A two-sided lesson passes role-keyed mappings, e.g.
    ``reward={"attacker": {...}, "blocker": {...}}``.
    """

    return {
        "schema": schema,
        "objective": objective,
        "reward": dict(reward),
        "cost": dict(cost),
        "anti_farm": anti_farm,
        **sections,
        **horizon.manifest(),
    }


def head_neutral(name: str, size: int) -> int:
    """Return the index that means "do nothing" for one action head."""

    return size // 2 if name in _CENTRED_HEADS else 0


def head_scope(*open_heads: str, **explicit: Iterable[int]) -> FactoredActionScope:
    """Open the named heads fully and pin every other head to its no-op.

    ``head_scope("jump_off_on", ability_none_plus_slots=(0, 3))`` opens jump
    across its whole range and restricts abilities to "none" and slot 3. Any
    head not mentioned is locked to the neutral choice, so a contract cannot
    accidentally train a mechanic it does not score.
    """

    requested = set(open_heads) | set(explicit)
    published = set(ARSENAL_POLICY_ACTION_HEAD_NAMES)
    unknown = sorted(requested - published)
    if unknown:
        raise ValueError(f"unknown action heads: {unknown}")

    choices: list[tuple[int, ...]] = []
    for name, size in zip(
        ARSENAL_POLICY_ACTION_HEAD_NAMES,
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
        strict=True,
    ):
        if name in explicit:
            row = tuple(int(value) for value in explicit[name])
            if not row:
                raise ValueError(f"explicit scope for {name} cannot be empty")
            choices.append(row)
        elif name in open_heads:
            choices.append(tuple(range(size)))
        else:
            choices.append((head_neutral(name, size),))
    return FactoredActionScope(tuple(choices))


def require_positive_int(value: Any, name: str) -> int:
    """Reject bools and non-integers before they become silent array casts."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if int(value) < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


def require_finite_nonnegative(values: Mapping[str, Any], label: str) -> None:
    """Reward weights are magnitudes; the sign lives in the reward expression."""

    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} {name} must be a real number")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{label} {name} must be finite and nonnegative")


def require_idle_is_costly(
    *,
    passive_rewards: Mapping[str, float],
    per_tick_costs: Mapping[str, float],
    label: str,
) -> None:
    """Refuse a config where doing nothing is profitable.

    "Charge per tick" is listed as a convention above, but a convention in a
    docstring is not a guard. `pursuit` learned this the expensive way: its
    line-of-sight, facing and aim-level terms are all payable while parked, and
    the lesson only stopped rewarding a stationary policy once
    `stationary_cost > line_of_sight + facing + aim_level` became a hard
    invariant rather than a note. Measured 2026-08-19 across this package, the
    same hole was open in `evasion`, where holding the band paid `+0.028`/tick
    with no threat present.

    ``passive_rewards`` is every term a lane can collect **without doing the
    thing the lesson teaches** -- not every positive term. Event payouts
    (landing a hit, evading a resolved threat) are outcomes and belong in
    neither mapping.

    The comparison is strict: equality means idling is free, which is still an
    absorbing optimum for a policy that has not yet found the objective.
    """

    income = float(sum(passive_rewards.values()))
    outgo = float(sum(per_tick_costs.values()))
    if income >= outgo:
        earned = " + ".join(sorted(passive_rewards)) or "0"
        charged = " + ".join(sorted(per_tick_costs)) or "0"
        raise ValueError(
            f"{label}: a do-nothing lane earns {income:+.4f} per tick against "
            f"{outgo:.4f} charged, so idling is optimal. Require "
            f"{charged} > {earned}, either by raising the per-tick charge or by "
            f"gating the passive term on the behaviour the lesson pays for."
        )


def band_fraction(
    value: jax.Array,
    lower: float,
    upper: float,
) -> jax.Array:
    """Return 1.0 inside ``[lower, upper]`` and fall off linearly outside it.

    Used wherever a contract wants a *held* quantity rather than a maximised
    one -- a distance band, a reach margin. The linear skirt keeps a gradient
    outside the band so a policy starting far away still learns which way to
    move, while the flat top removes any incentive to crowd one edge.
    """

    span = jnp.maximum(jnp.float32(upper - lower), jnp.float32(1.0e-6))
    below = (jnp.float32(lower) - value) / span
    above = (value - jnp.float32(upper)) / span
    outside = jnp.maximum(jnp.maximum(below, above), jnp.float32(0.0))
    return jnp.clip(jnp.float32(1.0) - outside, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class HorizonConfig:
    """Bounded episode length shared by every contract in this package."""

    maximum_ticks: int = 512
    minimum_ticks: int = 1

    def __post_init__(self) -> None:
        maximum = require_positive_int(self.maximum_ticks, "maximum_ticks")
        minimum = require_positive_int(self.minimum_ticks, "minimum_ticks")
        if minimum >= maximum:
            raise ValueError("minimum_ticks must be below maximum_ticks")

    def manifest(self) -> dict[str, object]:
        return {
            "maximum_ticks": self.maximum_ticks,
            "minimum_ticks": self.minimum_ticks,
            "horizon": "time_limit_truncation_with_bootstrap",
        }


def lane_shapes_match(*values: jax.Array) -> None:
    """Fail loudly when evidence arrays disagree, rather than broadcasting."""

    if not values:
        return
    shape = jnp.asarray(values[0]).shape
    if any(jnp.asarray(value).shape != shape for value in values[1:]):
        raise ValueError("contract evidence must share one lane shape [B]")


__all__ = [
    "HorizonConfig",
    "LOCOMOTION_HEADS",
    "band_fraction",
    "contract_hash",
    "head_neutral",
    "head_scope",
    "lane_shapes_match",
    "require_finite_nonnegative",
    "require_idle_is_costly",
    "require_positive_int",
]
