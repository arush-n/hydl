"""1v1 attack-versus-block timing duel.

Both actors learn. The attacker is looking for the moment its opponent is not
guarding; the blocker is looking for the moment the attack actually lands. The
contract is deliberately symmetric so neither side can settle into a degenerate
policy that the other never punishes.

The hack this contract exists to prevent is **holding guard forever**. Guarding
is free in the engine, so a naive "reward damage prevented" law is maximised by
raising guard on tick zero and never lowering it. ``guard_uptime_cost`` charges
every guarded tick, which makes the tuning rule explicit:

    block_reward > guard_uptime_cost * attack_window_ticks     (guarding pays)
    guard_uptime_cost > 0                                      (timing beats always-on)

The second inequality is what matters. For any positive uptime cost, guarding
only during the threat window strictly dominates guarding always, by
``guard_uptime_cost * (attack_period - attack_window)`` per exchange. The first
inequality just keeps guarding worth doing at all.

The attacker faces the mirror hazard -- spamming attacks into a raised guard --
and pays ``false_activation_cost`` for every request that does not become an
accepted, damaging start.
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


GUARD_DUEL_SCHEMA = "arena-contract-guard-duel-v1"


@dataclass(frozen=True, slots=True)
class GuardDuelConfig:
    """Tunable weights for the paired attack/block timing lesson."""

    horizon: HorizonConfig = HorizonConfig()

    # Attacker side.
    damage_dealt_scale: float = 1.0
    clean_hit_reward: float = 0.5
    false_activation_cost: float = 0.15
    attacker_tick_cost: float = 0.002

    # Blocker side.
    block_reward: float = 0.6
    damage_taken_scale: float = 1.0
    guard_uptime_cost: float = 0.02
    late_guard_cost: float = 0.05
    blocker_tick_cost: float = 0.002

    #: Damage at or below this counts the incoming attack as blocked.
    blocked_damage_epsilon: float = 1.0e-3
    #: Consecutive guarded ticks after which uptime cost doubles.
    guard_patience_ticks: int = 24

    schema: str = GUARD_DUEL_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, HorizonConfig):
            raise TypeError("guard duel horizon must be a HorizonConfig")
        require_positive_int(self.guard_patience_ticks, "guard_patience_ticks")
        require_finite_nonnegative(
            {
                "damage_dealt_scale": self.damage_dealt_scale,
                "clean_hit_reward": self.clean_hit_reward,
                "false_activation_cost": self.false_activation_cost,
                "attacker_tick_cost": self.attacker_tick_cost,
                "block_reward": self.block_reward,
                "damage_taken_scale": self.damage_taken_scale,
                "guard_uptime_cost": self.guard_uptime_cost,
                "late_guard_cost": self.late_guard_cost,
                "blocker_tick_cost": self.blocker_tick_cost,
                "blocked_damage_epsilon": self.blocked_damage_epsilon,
            },
            "guard duel",
        )
        if self.guard_uptime_cost <= 0.0:
            raise ValueError(
                "guard_uptime_cost must be positive or always-guard is optimal"
            )
        # One check per role: a two-sided lesson can be farmable on one side
        # only, and a single combined check would hide that.
        for role, tick_cost in (
            ("guard_duel attacker", self.attacker_tick_cost),
            ("guard_duel blocker", self.blocker_tick_cost),
        ):
            require_idle_is_costly(
                passive_rewards={},
                per_tick_costs={"tick_cost": tick_cost},
                label=role,
            )
        if self.schema != GUARD_DUEL_SCHEMA:
            raise ValueError("guard duel schema is not current")

    def manifest(self) -> dict[str, object]:
        return normalized_manifest(
            schema=self.schema,
            objective="win_the_exchange_by_timing_attacks_and_guards",
            reward={
                "attacker": {
                    "damage_dealt_scale": self.damage_dealt_scale,
                    "clean_hit_reward": self.clean_hit_reward,
                },
                "blocker": {"block_reward": self.block_reward},
            },
            cost={
                "attacker": {
                    "false_activation_cost": self.false_activation_cost,
                    "tick_cost": self.attacker_tick_cost,
                },
                "blocker": {
                    "damage_taken_scale": self.damage_taken_scale,
                    "guard_uptime_cost": self.guard_uptime_cost,
                    "late_guard_cost": self.late_guard_cost,
                    "tick_cost": self.blocker_tick_cost,
                },
            },
            anti_farm=(
                "guard_uptime_is_charged_every_tick_so_timed_guard_strictly_"
                "dominates_always_guard"
            ),
            horizon=self.horizon,
            roles="attacker_and_blocker_both_trainable",
            terminal="natural_combat_outcome_or_horizon",
            guard_patience_ticks=self.guard_patience_ticks,
        )

    @property
    def contract_sha256(self) -> str:
        return contract_hash(self.manifest())


class GuardDuelSignals(NamedTuple):
    """Paired credit for one exchange, plus the evidence behind it."""

    attacker_reward: jax.Array
    blocker_reward: jax.Array
    clean_hit: jax.Array
    blocked_hit: jax.Array
    false_activation: jax.Array
    guard_pressure: jax.Array


def guard_duel_signals(
    config: GuardDuelConfig,
    *,
    attack_requested: jax.Array,
    attack_accepted: jax.Array,
    damage_dealt: jax.Array,
    incoming_attack_active: jax.Array,
    incoming_attack_resolved: jax.Array,
    guard_active: jax.Array,
    guard_streak: jax.Array,
    valid: jax.Array,
) -> GuardDuelSignals:
    """Score one tick of the duel from engine evidence only.

    ``incoming_attack_active`` is the threat window -- the attacker has an
    accepted attack in flight. ``incoming_attack_resolved`` is the tick that
    window closes, which is the only tick a block can be credited: guarding
    while nothing is incoming earns nothing and still costs uptime.
    """

    requested = jnp.asarray(attack_requested, dtype=jnp.bool_)
    accepted = jnp.asarray(attack_accepted, dtype=jnp.bool_)
    damage = jnp.asarray(damage_dealt, dtype=jnp.float32)
    threat = jnp.asarray(incoming_attack_active, dtype=jnp.bool_)
    resolved = jnp.asarray(incoming_attack_resolved, dtype=jnp.bool_)
    guarding = jnp.asarray(guard_active, dtype=jnp.bool_)
    streak = jnp.asarray(guard_streak, dtype=jnp.int32)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    lane_shapes_match(
        requested, accepted, damage, threat, resolved, guarding, streak, evidence
    )

    landed = evidence & (damage > jnp.float32(config.blocked_damage_epsilon))
    clean_hit = landed & ~guarding
    blocked_hit = evidence & resolved & guarding & ~landed
    false_activation = evidence & requested & ~accepted

    attacker_reward = (
        jnp.float32(config.damage_dealt_scale) * jnp.where(landed, damage, 0.0)
        + jnp.float32(config.clean_hit_reward) * clean_hit
        - jnp.float32(config.false_activation_cost) * false_activation
        - jnp.float32(config.attacker_tick_cost)
    )

    # Uptime is charged always, and charged double once the guard has been held
    # past the patience window. That is what turns "hold guard" into a losing
    # policy without ever making guarding itself unattractive.
    impatient = streak >= jnp.int32(config.guard_patience_ticks)
    uptime = guarding & evidence
    guard_pressure = jnp.where(
        uptime, jnp.where(impatient, jnp.float32(2.0), jnp.float32(1.0)), 0.0
    )
    # Raising guard only after the hit has already landed is the classic
    # too-late reflex; charge it separately so it is visible in telemetry.
    late_guard = landed & guarding & ~threat

    blocker_reward = (
        jnp.float32(config.block_reward) * blocked_hit
        - jnp.float32(config.damage_taken_scale) * jnp.where(landed, damage, 0.0)
        - jnp.float32(config.guard_uptime_cost) * guard_pressure
        - jnp.float32(config.late_guard_cost) * late_guard
        - jnp.float32(config.blocker_tick_cost)
    )

    zero = jnp.float32(0.0)
    return GuardDuelSignals(
        jnp.where(evidence, attacker_reward, zero),
        jnp.where(evidence, blocker_reward, zero),
        clean_hit,
        blocked_hit,
        false_activation,
        jnp.where(evidence, guard_pressure, zero),
    )


def advance_guard_streak(
    streak: jax.Array,
    guard_active: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    """Count consecutive guarded ticks; any unguarded tick resets to zero."""

    current = jnp.asarray(streak, dtype=jnp.int32)
    guarding = jnp.asarray(guard_active, dtype=jnp.bool_)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    lane_shapes_match(current, guarding, evidence)
    return jnp.where(evidence & guarding, current + 1, jnp.int32(0))


def guard_duel_attacker_action_scope() -> FactoredActionScope:
    """Basic attacks plus movement; no guard, no abilities."""

    return head_scope("base_action", *LOCOMOTION_HEADS)


def guard_duel_blocker_action_scope() -> FactoredActionScope:
    """Guard plus movement; the blocker cannot answer with its own attack."""

    return head_scope("guard_off_on", *LOCOMOTION_HEADS)


__all__ = [
    "GUARD_DUEL_SCHEMA",
    "GuardDuelConfig",
    "GuardDuelSignals",
    "advance_guard_streak",
    "guard_duel_attacker_action_scope",
    "guard_duel_blocker_action_scope",
    "guard_duel_signals",
]
