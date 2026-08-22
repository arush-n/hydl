"""Counter-attack inside the opponent's recovery window.

This is the contract that turns defence into wins, and it is the natural
partner to :mod:`arena.training.contracts.guard_duel`. Blocking teaches a
policy when *not* to be hit; on its own it produces a turtle. What converts a
survived attack into damage is hitting back during the window where the
opponent is committed and cannot answer.

Three timing regions, scored differently on purpose:

``recovery``
    The opponent's attack has ended and it is still committed. A hit here is
    the behaviour being trained and pays ``punish_reward``.

``opponent_active``
    The opponent's attack is live. Attacking into it is a trade, and a trade is
    not a punish -- it is charged, so the policy learns to wait rather than
    mash.

``neutral``
    Neither. **Damage here pays nothing at all.** This is a punish-timing
    lesson, not a general attack lesson -- general attack timing is what
    :mod:`arena.training.contracts.guard_duel` teaches. Crediting neutral
    damage was the first version of this contract and a deterministic probe
    showed it inverted the objective: the neutral region is far longer than the
    recovery window, so ``damage_scale`` over neutral ticks outscored the
    timing bonus and "attack whenever the opponent is not swinging" beat
    "attack only in recovery" by more than 2x. Damage is now gated on the
    window.

Attack requests are charged whether or not they connect, which is what stops a
policy from carpeting the episode with attacks to catch windows by accident:
mashing lands the same in-window hits as patient play but pays request cost on
every other tick as well.
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
)
from arena.training.skills.program import FactoredActionScope


PUNISH_WINDOW_SCHEMA = "arena-contract-punish-window-v1"


@dataclass(frozen=True, slots=True)
class PunishWindowConfig:
    """Tunable weights for recovery-window counter-attacking."""

    horizon: HorizonConfig = HorizonConfig()

    punish_reward: float = 1.2
    damage_scale: float = 0.5
    trade_cost: float = 0.4
    request_cost: float = 0.05
    missed_window_cost: float = 0.15
    tick_cost: float = 0.002

    schema: str = PUNISH_WINDOW_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, HorizonConfig):
            raise TypeError("punish window horizon must be a HorizonConfig")
        require_finite_nonnegative(
            {
                "punish_reward": self.punish_reward,
                "damage_scale": self.damage_scale,
                "trade_cost": self.trade_cost,
                "request_cost": self.request_cost,
                "missed_window_cost": self.missed_window_cost,
                "tick_cost": self.tick_cost,
            },
            "punish window",
        )
        if self.request_cost <= 0.0:
            raise ValueError("request_cost must be positive or mashing is free")
        require_idle_is_costly(
            passive_rewards={},
            per_tick_costs={"tick_cost": self.tick_cost},
            label="punish_window",
        )
        if self.schema != PUNISH_WINDOW_SCHEMA:
            raise ValueError("punish window schema is not current")

    def manifest(self) -> dict[str, object]:
        return normalized_manifest(
            schema=self.schema,
            objective="land_damage_inside_the_opponent_recovery_window",
            reward={
                "punish_reward": self.punish_reward,
                "damage_scale": self.damage_scale,
            },
            cost={
                "trade_cost": self.trade_cost,
                "request_cost": self.request_cost,
                "missed_window_cost": self.missed_window_cost,
                "tick_cost": self.tick_cost,
            },
            anti_farm=(
                "damage_and_timing_bonus_are_both_gated_on_the_window_and_every_"
                "request_is_charged_so_mashing_pays_more_for_the_same_hits"
            ),
            horizon=self.horizon,
            damage_credit="recovery_window_only_neutral_damage_pays_nothing",
        )

    @property
    def contract_sha256(self) -> str:
        return contract_hash(self.manifest())


class PunishWindowSignals(NamedTuple):
    """Counter-attack credit plus the timing evidence behind it."""

    reward: jax.Array
    punished: jax.Array
    traded: jax.Array
    missed_window: jax.Array


def punish_window_signals(
    config: PunishWindowConfig,
    *,
    attack_requested: jax.Array,
    damage_dealt: jax.Array,
    recovery_window_active: jax.Array,
    opponent_attack_active: jax.Array,
    window_closing: jax.Array,
    valid: jax.Array,
) -> PunishWindowSignals:
    """Score one counter-attack decision by which timing region it fell in.

    ``window_closing`` marks the last tick of a recovery window, which is where
    a window that produced no damage is charged. Without it a policy could
    simply never attack and pay nothing beyond the tick cost.
    """

    requested = jnp.asarray(attack_requested, dtype=jnp.bool_)
    damage = jnp.asarray(damage_dealt, dtype=jnp.float32)
    recovery = jnp.asarray(recovery_window_active, dtype=jnp.bool_)
    opponent_active = jnp.asarray(opponent_attack_active, dtype=jnp.bool_)
    closing = jnp.asarray(window_closing, dtype=jnp.bool_)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    lane_shapes_match(
        requested, damage, recovery, opponent_active, closing, evidence
    )

    landed = evidence & (damage > 0.0)
    punished = landed & recovery
    traded = evidence & requested & opponent_active
    missed_window = evidence & closing & ~punished

    # Damage is credited only inside the recovery window. Paying for neutral
    # damage makes the long neutral region outscore the short window and
    # inverts the whole lesson -- see the module docstring.
    reward = (
        jnp.float32(config.punish_reward) * punished
        + jnp.float32(config.damage_scale) * jnp.where(punished, damage, 0.0)
        - jnp.float32(config.trade_cost) * traded
        - jnp.float32(config.request_cost) * (evidence & requested)
        - jnp.float32(config.missed_window_cost) * missed_window
        - jnp.float32(config.tick_cost)
    )
    return PunishWindowSignals(
        jnp.where(evidence, reward, jnp.float32(0.0)),
        punished,
        traded,
        missed_window,
    )


def punish_window_action_scope() -> FactoredActionScope:
    """Basic attack plus movement and aim; no guard, no abilities."""

    return head_scope("base_action", *LOCOMOTION_HEADS)


__all__ = [
    "PUNISH_WINDOW_SCHEMA",
    "PunishWindowConfig",
    "PunishWindowSignals",
    "punish_window_action_scope",
    "punish_window_signals",
]
