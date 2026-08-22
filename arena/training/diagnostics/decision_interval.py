"""Pure-JAX execution of one policy decision over physical combat ticks.

The live Dawn boundary does not ask its recurrent policy for another action
while an accepted native ability request is still queued or executing.  This
module provides the corresponding simulator primitive: issue the learner row
once, advance the physical arena with neutral learner rows until the request
has authoritative terminal evidence, and return one semi-Markov interval.

The callbacks are static Python callables traced into the surrounding JAX
program.  They must be pure, lane-independent functions over fixed-shape
PyTrees; this module never performs host I/O or a device transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp


DECISION_INTERVAL_PROGRAM_SCHEMA = "arena-jax-decision-interval-v2"


def _hash(value: Any) -> str:
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


@dataclass(frozen=True, slots=True)
class DecisionIntervalProgram:
    """Static semi-Markov timing, discount, and request-end contract."""

    maximum_physical_ticks: int
    discount: float = 0.99
    trace_decay: float = 0.95
    tick_rate_hz: float = 30.0
    schema: str = DECISION_INTERVAL_PROGRAM_SCHEMA

    def __post_init__(self) -> None:
        if isinstance(self.maximum_physical_ticks, bool) or not isinstance(
            self.maximum_physical_ticks, Integral
        ):
            raise TypeError("maximum physical ticks must be an integer")
        maximum = int(self.maximum_physical_ticks)
        if maximum < 1:
            raise ValueError("maximum physical ticks must be positive")
        object.__setattr__(self, "maximum_physical_ticks", maximum)
        for name in ("discount", "trace_decay", "tick_rate_hz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name.replace('_', ' ')} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{name.replace('_', ' ')} must be finite")
            object.__setattr__(self, name, normalized)
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0,1]")
        if not 0.0 <= self.trace_decay <= 1.0:
            raise ValueError("trace decay must be in [0,1]")
        if self.tick_rate_hz <= 0.0:
            raise ValueError("tick rate must be positive")
        if self.schema != DECISION_INTERVAL_PROGRAM_SCHEMA:
            raise ValueError("decision interval schema is not current")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "maximum_physical_ticks": self.maximum_physical_ticks,
            "discount": self.discount,
            "trace_decay": self.trace_decay,
            "tick_rate_hz": self.tick_rate_hz,
            "policy_cadence": "one_recurrent_advance_per_terminal_request",
            "learner_issue_law": (
                "issued_factors_on_first_physical_tick_then_exact_neutral"
            ),
            "target_cadence": "target_callback_advances_on_every_physical_tick",
            "target_context": (
                "one_opaque_lane_major_pytree_computed_by_target_callback_per_"
                "physical_tick"
            ),
            "physical_context_reuse": (
                "exact_target_context_forwarded_to_physical_callback_without_"
                "recomputation"
            ),
            "ability_terminal": (
                "accepted_request_and_observable_active_seen_then_cleared"
            ),
            "non_ability_terminal": "one_valid_physical_tick",
            "rejection_terminal": "requested_ability_not_accepted",
            "episode_end": (
                "natural_terminal_or_bootstrapped_task_truncation_or_support_invalid"
                "_or_physical_tick_horizon"
            ),
            "support_failure": "zero_issue_zero_mutation_zero_reward",
            "reward_accumulation": "sum_t_discount_power_t_times_reward_t",
            "smdp_discount": "discount_power_elapsed_physical_ticks",
            "smdp_trace_discount": (
                "(discount_times_trace_decay)_power_elapsed_physical_ticks"
            ),
            "bootstrap": "zero_on_natural_or_invalid_else_smdp_discount",
            "execution": "pure_jax_lax_while_loop_no_host_callbacks",
        }

    @property
    def contract_sha256(self) -> str:
        return _hash(self.manifest())


class DecisionIntervalTargetStep(NamedTuple):
    """One target decision, next recurrent state, and shared tick context."""

    factors: jax.Array
    state: Any
    valid: jax.Array
    context: Any


class DecisionIntervalPhysicalStep(NamedTuple):
    """Physical evidence returned by one arena microstep."""

    state: Any
    reward: jax.Array
    valid: jax.Array
    natural_terminal: jax.Array
    episode_truncated: jax.Array
    ability_requested: jax.Array
    ability_accepted: jax.Array
    active_ability: jax.Array


class DecisionIntervalLifecycle(NamedTuple):
    """Per-lane request and episode-end evidence for one interval."""

    issued: jax.Array
    accepted: jax.Array
    completed: jax.Array
    rejected: jax.Array
    interrupted: jax.Array
    active_seen: jax.Array
    invalid: jax.Array
    natural_terminal: jax.Array
    episode_truncated: jax.Array
    horizon_reached: jax.Array


class DecisionIntervalResult(NamedTuple):
    """One semi-Markov transition produced without leaving the device."""

    state: Any
    target_state: Any
    discounted_reward: jax.Array
    undiscounted_reward: jax.Array
    elapsed_ticks: jax.Array
    delta_time: jax.Array
    smdp_discount: jax.Array
    smdp_trace_discount: jax.Array
    bootstrap_discount: jax.Array
    next_world_tick: jax.Array
    learner_issue_count: jax.Array
    ability_request_count: jax.Array
    target_tick_count: jax.Array
    lifecycle: DecisionIntervalLifecycle


TargetStep = Callable[[Any, Any, jax.Array, jax.Array], DecisionIntervalTargetStep]
AdvancePhysicalStep = Callable[
    [Any, jax.Array, jax.Array, Any, jax.Array], DecisionIntervalPhysicalStep
]


class _DecisionIntervalCarry(NamedTuple):
    state: Any
    target_state: Any
    running: jax.Array
    elapsed_ticks: jax.Array
    world_tick: jax.Array
    discount_power: jax.Array
    trace_discount_power: jax.Array
    discounted_reward: jax.Array
    undiscounted_reward: jax.Array
    issued: jax.Array
    accepted: jax.Array
    completed: jax.Array
    rejected: jax.Array
    active_seen: jax.Array
    invalid: jax.Array
    natural_terminal: jax.Array
    episode_truncated: jax.Array
    horizon_reached: jax.Array
    learner_issue_count: jax.Array
    ability_request_count: jax.Array
    target_tick_count: jax.Array


def _bool_row(value: Any, batch: int, name: str) -> jax.Array:
    row = jnp.asarray(value)
    if row.dtype != jnp.dtype(jnp.bool_) or row.shape != (batch,):
        raise TypeError(f"{name} must be bool[B]")
    return row


def _int_row(value: Any, batch: int, name: str) -> jax.Array:
    row = jnp.asarray(value)
    if row.dtype != jnp.dtype(jnp.int32) or row.shape != (batch,):
        raise TypeError(f"{name} must be int32[B]")
    return row


def _factor_rows(value: Any, batch: int, width: int, name: str) -> jax.Array:
    rows = jnp.asarray(value)
    if rows.dtype != jnp.dtype(jnp.int32) or rows.shape != (batch, width):
        raise TypeError(f"{name} must be int32[B,H]")
    return rows


def _select_lanes(mask: jax.Array, candidate: Any, previous: Any) -> Any:
    """Select lane-major PyTree leaves without allowing cross-lane mutation."""

    batch = mask.shape[0]

    def select(candidate_leaf: Any, previous_leaf: Any) -> jax.Array:
        candidate_value = jnp.asarray(candidate_leaf)
        previous_value = jnp.asarray(previous_leaf)
        if candidate_value.shape != previous_value.shape:
            raise ValueError("decision interval callback changed a state shape")
        if not candidate_value.shape or candidate_value.shape[0] != batch:
            raise ValueError("decision interval state leaves must be lane-major")
        lane_mask = jnp.reshape(mask, (batch,) + (1,) * (candidate_value.ndim - 1))
        return jnp.where(lane_mask, candidate_value, previous_value)

    return jax.tree_util.tree_map(select, candidate, previous)


def _validate_lane_major_context(context: Any, batch: int) -> None:
    """Reject context PyTrees that cannot preserve independent lane semantics."""

    leaves = jax.tree_util.tree_leaves(context)
    if not leaves:
        raise ValueError("decision interval target context must not be empty")
    for leaf in leaves:
        value = jnp.asarray(leaf)
        if not value.shape or value.shape[0] != batch:
            raise ValueError(
                "decision interval target context leaves must be lane-major"
            )


def _lane_keys(keys: jax.Array, ticks: jax.Array, domain: int) -> jax.Array:
    return jax.vmap(
        lambda key, tick: jax.random.fold_in(
            jax.random.fold_in(key, jnp.uint32(domain)), tick
        )
    )(keys, ticks)


def run_decision_interval(
    program: DecisionIntervalProgram,
    state: Any,
    target_state: Any,
    issued_learner_factors: jax.Array,
    neutral_learner_factors: jax.Array,
    neutral_target_factors: jax.Array,
    learner_action_valid: jax.Array,
    ability_lifecycle_expected: jax.Array,
    world_tick: jax.Array,
    remaining_physical_ticks: jax.Array,
    keys: jax.Array,
    *,
    target_step: TargetStep,
    advance_physical_step: AdvancePhysicalStep,
) -> DecisionIntervalResult:
    """Execute one learner decision until its request or episode ends.

    ``target_step`` and ``advance_physical_step`` are traced as static pure
    callbacks.  Both state PyTrees must have a leading lane axis on every
    array leaf. ``ability_lifecycle_expected`` is derived by the caller from
    the issued learner factors; the physical callback must echo request and
    acceptance evidence from the engine.
    """

    if not isinstance(program, DecisionIntervalProgram):
        raise TypeError("decision interval needs a DecisionIntervalProgram")
    issued_rows = jnp.asarray(issued_learner_factors)
    if issued_rows.ndim != 2:
        raise TypeError("issued learner factors must be int32[B,H]")
    batch, width = issued_rows.shape
    issued_rows = _factor_rows(issued_rows, batch, width, "issued learner factors")
    learner_neutral = _factor_rows(
        neutral_learner_factors, batch, width, "neutral learner factors"
    )
    target_neutral = _factor_rows(
        neutral_target_factors, batch, width, "neutral target factors"
    )
    learner_valid = _bool_row(learner_action_valid, batch, "learner action valid")
    expects_ability = _bool_row(
        ability_lifecycle_expected, batch, "ability lifecycle expected"
    )
    ticks = _int_row(world_tick, batch, "world tick")
    remaining = _int_row(remaining_physical_ticks, batch, "remaining physical ticks")
    key_data = jax.random.key_data(keys)
    if key_data.ndim != 2 or key_data.shape[0] != batch:
        raise ValueError("decision interval keys must be one JAX key per lane")

    zero_bool = jnp.zeros((batch,), dtype=jnp.bool_)
    zero_int = jnp.zeros((batch,), dtype=jnp.int32)
    zero_float = jnp.zeros((batch,), dtype=jnp.float32)
    negative_budget = remaining < 0
    bounded_remaining = jnp.clip(
        remaining, jnp.int32(0), jnp.int32(program.maximum_physical_ticks)
    )
    initial_invalid = ~learner_valid | negative_budget
    initial_horizon = learner_valid & ~negative_budget & (bounded_remaining == 0)
    initial = _DecisionIntervalCarry(
        state=state,
        target_state=target_state,
        running=learner_valid & ~negative_budget & (bounded_remaining > 0),
        elapsed_ticks=zero_int,
        world_tick=ticks,
        discount_power=jnp.ones((batch,), dtype=jnp.float32),
        trace_discount_power=jnp.ones((batch,), dtype=jnp.float32),
        discounted_reward=zero_float,
        undiscounted_reward=zero_float,
        issued=zero_bool,
        accepted=zero_bool,
        completed=zero_bool,
        rejected=zero_bool,
        active_seen=zero_bool,
        invalid=initial_invalid,
        natural_terminal=zero_bool,
        episode_truncated=zero_bool,
        horizon_reached=initial_horizon,
        learner_issue_count=zero_int,
        ability_request_count=zero_int,
        target_tick_count=zero_int,
    )

    def condition(carry: _DecisionIntervalCarry) -> jax.Array:
        return jnp.any(carry.running)

    def body(carry: _DecisionIntervalCarry) -> _DecisionIntervalCarry:
        first_tick = carry.elapsed_ticks == 0
        learner_factors = jnp.where(
            (carry.running & first_tick)[:, None], issued_rows, learner_neutral
        )
        target = target_step(
            carry.state,
            carry.target_state,
            carry.world_tick,
            _lane_keys(keys, carry.world_tick, 0x54415247),
        )
        target_factors = _factor_rows(
            target.factors, batch, width, "target callback factors"
        )
        target_valid = _bool_row(target.valid, batch, "target callback valid")
        _validate_lane_major_context(target.context, batch)
        pre_step_invalid = carry.running & ~target_valid
        execute = carry.running & target_valid
        learner_factors = jnp.where(execute[:, None], learner_factors, learner_neutral)
        target_factors = jnp.where(execute[:, None], target_factors, target_neutral)

        def advance(_: None) -> DecisionIntervalPhysicalStep:
            return advance_physical_step(
                carry.state,
                learner_factors,
                target_factors,
                target.context,
                _lane_keys(keys, carry.world_tick, 0x50485953),
            )

        def no_advance(_: None) -> DecisionIntervalPhysicalStep:
            return DecisionIntervalPhysicalStep(
                carry.state,
                zero_float,
                zero_bool,
                zero_bool,
                zero_bool,
                zero_bool,
                zero_bool,
                zero_bool,
            )

        physical = jax.lax.cond(jnp.any(execute), advance, no_advance, operand=None)
        step_valid = _bool_row(physical.valid, batch, "physical callback valid")
        natural = _bool_row(
            physical.natural_terminal, batch, "physical natural terminal"
        )
        episode_truncated = _bool_row(
            physical.episode_truncated, batch, "physical episode truncated"
        )
        ability_requested = _bool_row(
            physical.ability_requested, batch, "physical ability requested"
        )
        ability_accepted = _bool_row(
            physical.ability_accepted, batch, "physical ability accepted"
        )
        active_ability = _bool_row(
            physical.active_ability, batch, "physical active ability"
        )
        reward = jnp.asarray(physical.reward, dtype=jnp.float32)
        if reward.shape != (batch,):
            raise ValueError("physical callback reward must have shape [B]")

        first_execution = execute & first_tick
        continuation = execute & ~first_tick
        echo_invalid = execute & (
            (first_tick & (ability_requested != expects_ability))
            | (ability_accepted & ~ability_requested)
            | (continuation & (ability_requested | ability_accepted))
        )
        committed = execute & step_valid & ~echo_invalid
        invalid = (
            carry.invalid | pre_step_invalid | (execute & ~step_valid) | echo_invalid
        )
        elapsed = carry.elapsed_ticks + committed.astype(jnp.int32)
        next_world_tick = carry.world_tick + committed.astype(jnp.int32)
        requested = first_execution & committed
        accepted_now = requested & (~expects_ability | ability_accepted)
        issued = carry.issued | requested
        accepted = carry.accepted | accepted_now
        rejected = carry.rejected | (
            requested & expects_ability & ability_requested & ~ability_accepted
        )
        active_seen = carry.active_seen | (committed & active_ability)
        ability_completed = (
            committed & accepted & expects_ability & active_seen & ~active_ability
        )
        non_ability_completed = committed & first_tick & ~expects_ability
        completed = carry.completed | ability_completed | non_ability_completed
        natural_terminal = carry.natural_terminal | (committed & natural)
        task_truncated = carry.episode_truncated | (committed & episode_truncated)
        horizon_reached = carry.horizon_reached | (
            committed & (elapsed >= bounded_remaining)
        )
        terminal = (
            completed
            | rejected
            | invalid
            | natural_terminal
            | task_truncated
            | horizon_reached
        )
        next_running = carry.running & ~terminal
        committed_reward = jnp.where(committed, reward, 0.0)
        discount = jnp.float32(program.discount)
        trace_discount = discount * jnp.float32(program.trace_decay)
        return _DecisionIntervalCarry(
            state=_select_lanes(committed, physical.state, carry.state),
            target_state=_select_lanes(committed, target.state, carry.target_state),
            running=next_running,
            elapsed_ticks=elapsed,
            world_tick=next_world_tick,
            discount_power=jnp.where(
                committed, carry.discount_power * discount, carry.discount_power
            ),
            trace_discount_power=jnp.where(
                committed,
                carry.trace_discount_power * trace_discount,
                carry.trace_discount_power,
            ),
            discounted_reward=(
                carry.discounted_reward + carry.discount_power * committed_reward
            ),
            undiscounted_reward=carry.undiscounted_reward + committed_reward,
            issued=issued,
            accepted=accepted,
            completed=completed,
            rejected=rejected,
            active_seen=active_seen,
            invalid=invalid,
            natural_terminal=natural_terminal,
            episode_truncated=task_truncated,
            horizon_reached=horizon_reached,
            learner_issue_count=(
                carry.learner_issue_count + requested.astype(jnp.int32)
            ),
            ability_request_count=(
                carry.ability_request_count
                + (committed & ability_requested).astype(jnp.int32)
            ),
            target_tick_count=(carry.target_tick_count + committed.astype(jnp.int32)),
        )

    final = jax.lax.while_loop(condition, body, initial)
    interrupted = (
        final.issued
        & final.accepted
        & ~final.completed
        & (
            final.invalid
            | final.natural_terminal
            | final.episode_truncated
            | final.horizon_reached
        )
    )
    bootstrap = jnp.where(
        final.invalid | final.natural_terminal,
        0.0,
        final.discount_power,
    )
    return DecisionIntervalResult(
        state=final.state,
        target_state=final.target_state,
        discounted_reward=final.discounted_reward,
        undiscounted_reward=final.undiscounted_reward,
        elapsed_ticks=final.elapsed_ticks,
        delta_time=(
            final.elapsed_ticks.astype(jnp.float32) / jnp.float32(program.tick_rate_hz)
        ),
        smdp_discount=final.discount_power,
        smdp_trace_discount=final.trace_discount_power,
        bootstrap_discount=bootstrap,
        next_world_tick=final.world_tick,
        learner_issue_count=final.learner_issue_count,
        ability_request_count=final.ability_request_count,
        target_tick_count=final.target_tick_count,
        lifecycle=DecisionIntervalLifecycle(
            issued=final.issued,
            accepted=final.accepted,
            completed=final.completed,
            rejected=final.rejected,
            interrupted=interrupted,
            active_seen=final.active_seen,
            invalid=final.invalid,
            natural_terminal=final.natural_terminal,
            episode_truncated=final.episode_truncated,
            horizon_reached=final.horizon_reached,
        ),
    )


__all__ = [
    "DECISION_INTERVAL_PROGRAM_SCHEMA",
    "AdvancePhysicalStep",
    "DecisionIntervalLifecycle",
    "DecisionIntervalPhysicalStep",
    "DecisionIntervalProgram",
    "DecisionIntervalResult",
    "DecisionIntervalTargetStep",
    "TargetStep",
    "run_decision_interval",
]
