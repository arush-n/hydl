"""Hold a distance band under threat, and dodge the attack when it lands.

The defensive counterpart to :mod:`arena.training.contracts.punish_window`.
Staying alive is not the same skill as staying away: a policy that simply runs
is safe but useless, because it can never counter. What this lesson buys is
*holding a reachable gap while an attack is incoming* and getting out of the
way on the tick that matters.

Two things are scored, and both are gated on threat:

``band``
    Distance inside ``[hold_minimum_blocks, hold_maximum_blocks]``, paid only
    while ``incoming_attack_active``. **Band pay used to be unconditional, and
    the arithmetic inverted the lesson:** parking mid-band collected
    +0.028/tick forever, so standing still beat every evasive policy. The band
    term is now multiplied by the threat flag, which makes idling strictly
    negative -- it collects nothing and still pays ``tick_cost``.

``dodge``
    A dodge input on the tick the attack resolves. Distance alone does not
    avoid a hit; the dodge does, which is what separates this lesson from a
    retreat.

Damage taken is charged whether or not the policy tried to avoid it, so a
mistimed dodge is not laundered into a free attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from arena.training.contracts.common import (
    LOCOMOTION_HEADS,
    HorizonConfig,
    band_fraction,
    contract_hash,
    head_scope,
    lane_shapes_match,
    normalized_manifest,
    require_finite_nonnegative,
    require_idle_is_costly,
)
from arena.training.skills.program import FactoredActionScope


EVASION_SCHEMA = "arena-contract-evasion-v1"

#: The dodge head's no-op choice. Named so callers compare against a constant
#: rather than a bare zero, which reads as "no dodge" only by convention.
DODGE_NONE = 0


@dataclass(frozen=True, slots=True)
class EvasionConfig:
    """Tunable weights for threatened band-holding and timed dodges."""

    horizon: HorizonConfig = HorizonConfig()

    #: The gap worth holding. Inside it the policy is close enough to threaten
    #: back but far enough to react, which is the whole point of the band.
    hold_minimum_blocks: float = 3.0
    hold_maximum_blocks: float = 6.0

    band_reward: float = 0.03
    dodge_reward: float = 1.0
    damage_scale: float = 0.5
    missed_dodge_cost: float = 0.25
    tick_cost: float = 0.002

    schema: str = EVASION_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, HorizonConfig):
            raise TypeError("evasion horizon must be a HorizonConfig")
        require_finite_nonnegative(
            {
                "hold_minimum_blocks": self.hold_minimum_blocks,
                "hold_maximum_blocks": self.hold_maximum_blocks,
                "band_reward": self.band_reward,
                "dodge_reward": self.dodge_reward,
                "damage_scale": self.damage_scale,
                "missed_dodge_cost": self.missed_dodge_cost,
                "tick_cost": self.tick_cost,
            },
            "evasion",
        )
        if not self.hold_minimum_blocks < self.hold_maximum_blocks:
            raise ValueError("hold_minimum_blocks must be below hold_maximum_blocks")
        # The band pays nothing without threat, so an idle policy has no
        # passive income at all -- this states that rather than assuming it.
        require_idle_is_costly(
            passive_rewards={},
            per_tick_costs={"tick_cost": self.tick_cost},
            label="evasion",
        )
        if self.schema != EVASION_SCHEMA:
            raise ValueError("evasion schema is not current")

    def manifest(self) -> dict[str, object]:
        return normalized_manifest(
            schema=self.schema,
            objective="hold_the_distance_band_under_threat_and_dodge_on_resolve",
            reward={
                "band_reward": self.band_reward,
                "dodge_reward": self.dodge_reward,
            },
            cost={
                "damage_scale": self.damage_scale,
                "missed_dodge_cost": self.missed_dodge_cost,
                "tick_cost": self.tick_cost,
            },
            anti_farm=(
                "band_pay_is_gated_on_an_incoming_attack_so_parking_in_the_band_"
                "earns_nothing_and_still_pays_the_tick_cost"
            ),
            horizon=self.horizon,
            band={
                "hold_minimum_blocks": self.hold_minimum_blocks,
                "hold_maximum_blocks": self.hold_maximum_blocks,
                "shape": "one_inside_the_band_linear_skirt_outside",
            },
        )

    @property
    def contract_sha256(self) -> str:
        return contract_hash(self.manifest())


class EvasionSignals(NamedTuple):
    """Evasion credit plus the evidence behind it."""

    reward: jax.Array
    held_band: jax.Array
    dodged: jax.Array
    hit: jax.Array


def evasion_signals(
    config: EvasionConfig,
    *,
    distance: jax.Array,
    incoming_attack_active: jax.Array,
    incoming_attack_resolved: jax.Array,
    damage_taken: jax.Array,
    dodge_choice: jax.Array,
    valid: jax.Array,
) -> EvasionSignals:
    """Score one defensive decision against the threat state it was made in.

    ``incoming_attack_resolved`` marks the tick the attack lands, which is the
    only tick a dodge can pay on. Rewarding a dodge at any other time would buy
    a policy that dodges constantly rather than one that reads the attack.
    """

    gap = jnp.asarray(distance, dtype=jnp.float32)
    threatened = jnp.asarray(incoming_attack_active, dtype=jnp.bool_)
    resolved = jnp.asarray(incoming_attack_resolved, dtype=jnp.bool_)
    damage = jnp.asarray(damage_taken, dtype=jnp.float32)
    dodge = jnp.asarray(dodge_choice, dtype=jnp.int32)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    lane_shapes_match(gap, threatened, resolved, damage, dodge, evidence)

    dodging = dodge != jnp.int32(DODGE_NONE)
    in_band = band_fraction(
        gap, config.hold_minimum_blocks, config.hold_maximum_blocks
    )
    # Gated on threat: an unthreatened policy sitting in the band collects
    # nothing, which is what keeps idling strictly negative.
    held_band = evidence & threatened
    dodged = evidence & resolved & dodging
    hit = evidence & (damage > 0.0)
    missed_dodge = evidence & resolved & ~dodging

    reward = (
        jnp.float32(config.band_reward) * in_band * held_band
        + jnp.float32(config.dodge_reward) * dodged
        - jnp.float32(config.damage_scale) * damage
        - jnp.float32(config.missed_dodge_cost) * missed_dodge
        - jnp.float32(config.tick_cost)
    )
    return EvasionSignals(
        jnp.where(evidence, reward, jnp.float32(0.0)),
        held_band,
        dodged,
        hit,
    )


def evasion_action_scope() -> FactoredActionScope:
    """Movement and aim only; no attacks, no abilities, no guard.

    Dodges ride `locomotion_gait_compass` rather than a head of their own --
    the locomotion choice encodes gait, compass and dodge together -- so the
    locomotion heads are all a dodge lesson needs opened.
    """

    return head_scope(*LOCOMOTION_HEADS)


__all__ = [
    "DODGE_NONE",
    "EVASION_SCHEMA",
    "EvasionConfig",
    "EvasionSignals",
    "evasion_action_scope",
    "evasion_signals",
]
