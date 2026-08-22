"""1v1 ability aiming against a deterministic mover.

The opponent here is a *moving target*, not an evader. It follows a scripted
motion program and never reacts, so the only thing separating a hit from a miss
is the learner's own aim and timing. Giving it double health keeps the episode
alive long enough to produce many attempts per reset instead of ending on the
first lucky connection.

Abilities are expensive to fire and cheap to spam, so the reward is built the
other way round from a naive "reward damage" law:

* every activation is charged, whether or not it connects;
* an activation fired while badly misaligned or out of range is charged *more*,
  because that is the spray behaviour we want to extinguish;
* only engine-attributed damage pays.

A second ability unlocks partway through the episode. It is worth more than the
primary when it lands, so a policy that has learned to fire only the primary has
a reason to try it, but using it before the unlock tick is charged as a wasted
press rather than silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from arena.training.contracts.common import (
    LOCOMOTION_HEADS,
    HorizonConfig,
    contract_hash,
    head_scope,
    lane_shapes_match,
    normalized_manifest,
    require_finite_nonnegative,
    require_idle_is_costly,
    require_positive_int,
)
from arena.training.skills.program import FactoredActionScope


ABILITY_LANDING_SCHEMA = "arena-contract-ability-landing-v1"

#: Index 0 of ``ability_none_plus_slots`` is "fire nothing".
ABILITY_NONE = 0


@dataclass(frozen=True, slots=True)
class AbilityLandingConfig:
    """Tunable weights for aimed ability usage against a scripted mover."""

    horizon: HorizonConfig = HorizonConfig()

    #: Slots inside ``ability_none_plus_slots`` (1..16; 0 means fire nothing).
    primary_slot: int = 1
    secondary_slot: int = 2
    #: Tick at which the secondary becomes worth using.
    secondary_unlock_tick: int = 128

    #: The target is a punching bag, not a duellist -- keep it alive.
    target_health_multiplier: float = 2.0

    #: Landing rewards, by which slot connected.
    primary_land_reward: float = 1.0
    secondary_land_reward: float = 1.6
    damage_scale: float = 0.5

    #: Costs. Firing is never free; firing blind costs more.
    activation_cost: float = 0.05
    wasted_activation_cost: float = 0.25
    locked_ability_cost: float = 0.2
    rejected_activation_cost: float = 0.1
    tick_cost: float = 0.002

    #: An activation counts as aimed inside this cone and range.
    aim_tolerance_degrees: float = 12.0
    effective_range_blocks: float = 8.0

    schema: str = ABILITY_LANDING_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, HorizonConfig):
            raise TypeError("ability landing horizon must be a HorizonConfig")
        primary = require_positive_int(self.primary_slot, "primary_slot")
        secondary = require_positive_int(self.secondary_slot, "secondary_slot")
        if primary == secondary:
            raise ValueError("primary_slot and secondary_slot must differ")
        require_positive_int(self.secondary_unlock_tick, "secondary_unlock_tick")
        if self.secondary_unlock_tick >= self.horizon.maximum_ticks:
            raise ValueError("secondary never unlocks inside the horizon")
        require_finite_nonnegative(
            {
                "target_health_multiplier": self.target_health_multiplier,
                "primary_land_reward": self.primary_land_reward,
                "secondary_land_reward": self.secondary_land_reward,
                "damage_scale": self.damage_scale,
                "activation_cost": self.activation_cost,
                "wasted_activation_cost": self.wasted_activation_cost,
                "locked_ability_cost": self.locked_ability_cost,
                "rejected_activation_cost": self.rejected_activation_cost,
                "tick_cost": self.tick_cost,
                "aim_tolerance_degrees": self.aim_tolerance_degrees,
                "effective_range_blocks": self.effective_range_blocks,
            },
            "ability landing",
        )
        if self.target_health_multiplier < 1.0:
            raise ValueError("target_health_multiplier must be at least 1.0")
        if self.aim_tolerance_degrees <= 0.0 or self.effective_range_blocks <= 0.0:
            raise ValueError("aim tolerance and effective range must be positive")
        if self.activation_cost <= 0.0:
            raise ValueError("activation_cost must be positive or spam is free")
        require_idle_is_costly(
            passive_rewards={},
            per_tick_costs={"tick_cost": self.tick_cost},
            label="ability_landing",
        )
        if self.schema != ABILITY_LANDING_SCHEMA:
            raise ValueError("ability landing schema is not current")

    def manifest(self) -> dict[str, object]:
        return normalized_manifest(
            schema=self.schema,
            objective="land_abilities_on_a_deterministic_mover_by_aiming",
            reward={
                "primary_land_reward": self.primary_land_reward,
                "secondary_land_reward": self.secondary_land_reward,
                "damage_scale": self.damage_scale,
            },
            cost={
                "activation_cost": self.activation_cost,
                "wasted_activation_cost": self.wasted_activation_cost,
                "locked_ability_cost": self.locked_ability_cost,
                "rejected_activation_cost": self.rejected_activation_cost,
                "tick_cost": self.tick_cost,
            },
            anti_farm=(
                "every_activation_is_charged_and_unaimed_activations_are_charged_"
                "more_so_only_aimed_attempts_are_worth_making"
            ),
            horizon=self.horizon,
            opponent="deterministic_scripted_mover_not_an_evader",
            target_health_multiplier=self.target_health_multiplier,
            slots={
                "primary": self.primary_slot,
                "secondary": self.secondary_slot,
                "secondary_unlock_tick": self.secondary_unlock_tick,
            },
            aimed_window={
                "aim_tolerance_degrees": self.aim_tolerance_degrees,
                "effective_range_blocks": self.effective_range_blocks,
            },
        )

    @property
    def contract_sha256(self) -> str:
        return contract_hash(self.manifest())


class AbilityLandingSignals(NamedTuple):
    """Credit for one ability decision, plus the evidence behind it."""

    reward: jax.Array
    activated: jax.Array
    aimed_activation: jax.Array
    wasted_activation: jax.Array
    landed: jax.Array
    used_secondary: jax.Array


def ability_landing_signals(
    config: AbilityLandingConfig,
    *,
    ability_choice: jax.Array,
    ability_accepted: jax.Array,
    attributed_damage: jax.Array,
    aim_error_degrees: jax.Array,
    distance: jax.Array,
    current_tick: jax.Array,
    valid: jax.Array,
) -> AbilityLandingSignals:
    """Score one ability decision from the slot fired and what it caused.

    ``ability_choice`` is the raw head index: ``ABILITY_NONE`` for a tick that
    fired nothing. ``attributed_damage`` must be engine-attributed damage on
    this lane, not a geometric guess -- geometry decides whether an attempt was
    *reasonable*, never whether it *worked*.
    """

    choice = jnp.asarray(ability_choice, dtype=jnp.int32)
    accepted = jnp.asarray(ability_accepted, dtype=jnp.bool_)
    damage = jnp.asarray(attributed_damage, dtype=jnp.float32)
    aim_error = jnp.asarray(aim_error_degrees, dtype=jnp.float32)
    range_to_target = jnp.asarray(distance, dtype=jnp.float32)
    tick = jnp.asarray(current_tick, dtype=jnp.int32)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    lane_shapes_match(
        choice, accepted, damage, aim_error, range_to_target, tick, evidence
    )

    activated = evidence & (choice != jnp.int32(ABILITY_NONE))
    used_primary = activated & (choice == jnp.int32(config.primary_slot))
    used_secondary = activated & (choice == jnp.int32(config.secondary_slot))
    unlocked = tick >= jnp.int32(config.secondary_unlock_tick)

    aimed = (
        (aim_error <= jnp.float32(config.aim_tolerance_degrees))
        & (range_to_target <= jnp.float32(config.effective_range_blocks))
    )
    aimed_activation = activated & aimed
    wasted_activation = activated & ~aimed
    rejected = activated & ~accepted
    locked_use = used_secondary & ~unlocked

    landed = evidence & (damage > 0.0)
    land_reward = jnp.where(
        used_secondary & unlocked,
        jnp.float32(config.secondary_land_reward),
        jnp.float32(config.primary_land_reward),
    )

    reward = (
        land_reward * landed
        + jnp.float32(config.damage_scale) * jnp.where(landed, damage, 0.0)
        - jnp.float32(config.activation_cost) * activated
        - jnp.float32(config.wasted_activation_cost) * wasted_activation
        - jnp.float32(config.locked_ability_cost) * locked_use
        - jnp.float32(config.rejected_activation_cost) * rejected
        - jnp.float32(config.tick_cost)
    )
    # `used_primary` is not scored separately; it is the default land reward.
    del used_primary

    return AbilityLandingSignals(
        jnp.where(evidence, reward, jnp.float32(0.0)),
        activated,
        aimed_activation,
        wasted_activation,
        landed,
        used_secondary & unlocked,
    )


def ability_landing_action_scope(config: AbilityLandingConfig) -> FactoredActionScope:
    """Only the two configured slots, plus movement and aim.

    Restricting the ability head to ``{none, primary, secondary}`` keeps the
    credit assignment readable: any activation in the rollout is one of two
    known mechanics, so a landing rate is directly interpretable.
    """

    return head_scope(
        *LOCOMOTION_HEADS,
        ability_none_plus_slots=(
            ABILITY_NONE,
            config.primary_slot,
            config.secondary_slot,
        ),
    )


__all__ = [
    "ABILITY_LANDING_SCHEMA",
    "ABILITY_NONE",
    "AbilityLandingConfig",
    "AbilityLandingSignals",
    "ability_landing_signals",
    "ability_landing_action_scope",
]
