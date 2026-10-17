"""SharedBelief: the ADK ``BeliefCore`` of experimental/neural_agent (ADR-001 D6).

Each update reads only the legal ``ActorPolicyInput``. It takes the previous
belief carry, the previous action factors and the elapsed time, and returns a
``BeliefUpdate`` whose ``belief`` is a ``SharedBeliefState``. Formulas and column
sources are recorded in SKILL.md.

Per update, with B lanes and H = ``BELIEF_LATENT_WIDTH`` = 64:

1. ``features.observation_readings`` gives the eight history-free features.
2. Two saturating tick counters are advanced by the elapsed ticks and reset
   by observed events. Their features are ``counter / TICKS_SATURATION``.
3. ``drive = [features, known, one_hot(previous_action), elapsed_ticks]``, width 145.
4. ``x = tanh(drive @ W_p + b_p)``, then ``h = GRU(h_prev, x)`` (Cho et al. 2014).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from adk.architecture.components import BeliefUpdate
from adk.architecture.inputs import ActorPolicyInput
from experimental.neural_agent.belief.features import observation_readings, positive_scale
from experimental.neural_agent.contracts import (
    ACTION_HEADS,
    BELIEF_FEATURE_COUNT,
    BELIEF_FEATURE_NAMES,
    BELIEF_LATENT_WIDTH,
    TICKS_SATURATION,
    SharedBeliefState,
)
from hytalegym.framework.composition import ProviderContract
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)

#: Cardinality of each factor head, in factor order (the 12-head surface).
ACTION_HEAD_SIZES: tuple[int, ...] = tuple(
    int(size) for size in ARSENAL_POLICY_ACTION_HEAD_SIZES
)
if len(ACTION_HEAD_SIZES) != ACTION_HEADS:
    raise RuntimeError(
        f"action surface has {len(ACTION_HEAD_SIZES)} heads; contract says {ACTION_HEADS}"
    )
#: Width of the one-hot previous-action encoding. The sum of the head sizes.
PREVIOUS_ACTION_WIDTH: int = sum(ACTION_HEAD_SIZES)
#: Width of the GRU drive: features, known flags, previous action, elapsed ticks.
DRIVE_WIDTH: int = 2 * BELIEF_FEATURE_COUNT + PREVIOUS_ACTION_WIDTH + 1

#: Provider contract for the ADK ``BeliefCore`` protocol. No schema refs are declared.
SHARED_BELIEF_CONTRACT = ProviderContract(
    provider_id="experimental.neural_agent.belief.shared",
)



STEP_SECONDS: float = 0.045
"""Seconds per environment decision step (loaded_dt); one belief tick."""

class SharedBeliefCarry(NamedTuple):
    """Belief carry, the ``belief_state`` leaf of ``HierarchicalCarry``.

    Shapes, with B lanes and H = 64:
        latent: float32[B, H], the recurrent latent.
        previous_self_health: float32[B], the last valid
            ``self_f32.health_fraction``, or -1 before the first valid reading.
        ticks_since_damage: float32[B], ticks since the last observed health
            drop. Starts at TICKS_SATURATION and is saturated at that value.
        ticks_since_seen: float32[B], ticks since the target was last visible.
            Starts at TICKS_SATURATION and is saturated at that value.
    """

    latent: jax.Array
    previous_self_health: jax.Array
    ticks_since_damage: jax.Array
    ticks_since_seen: jax.Array


class BeliefParams(NamedTuple):
    """Trainable parameters of the belief, a batched-free PyTree of arrays.

    Shapes, with D = DRIVE_WIDTH = 145 and H = 64. The three GRU blocks are
    ordered update (z), reset (r) and candidate (h~) along the last axis:
        projection_weight: float32[D, H].
        projection_bias: float32[H].
        gru_input_weight: float32[H, 3H], the input weights W_z|W_r|W_h.
        gru_recurrent_weight: float32[H, 3H], the recurrent weights U_z|U_r|U_h.
        gru_bias: float32[3H].
    """

    projection_weight: jax.Array
    projection_bias: jax.Array
    gru_input_weight: jax.Array
    gru_recurrent_weight: jax.Array
    gru_bias: jax.Array


def init_belief_params(key: jax.Array) -> BeliefParams:
    """Return randomly initialized belief parameters.

    Weights are normal with variance 1/fan_in (LeCun scaling). Biases are zero.
    The same key gives the same parameters.

    Args:
        key: A JAX PRNG key.

    Returns:
        BeliefParams with the shapes documented on :class:`BeliefParams`.
    """

    projection_key, input_key, recurrent_key = jax.random.split(key, 3)
    width = BELIEF_LATENT_WIDTH
    return BeliefParams(
        projection_weight=jax.random.normal(
            projection_key, (DRIVE_WIDTH, width), dtype=jnp.float32
        )
        / math.sqrt(DRIVE_WIDTH),
        projection_bias=jnp.zeros((width,), dtype=jnp.float32),
        gru_input_weight=jax.random.normal(
            input_key, (width, 3 * width), dtype=jnp.float32
        )
        / math.sqrt(width),
        gru_recurrent_weight=jax.random.normal(
            recurrent_key, (width, 3 * width), dtype=jnp.float32
        )
        / math.sqrt(width),
        gru_bias=jnp.zeros((3 * width,), dtype=jnp.float32),
    )


def previous_action_one_hot(previous_action: jax.Array) -> jax.Array:
    """Encode the previous factor indices as a concatenated one-hot row.

    Head ``h`` with cardinality ``n_h`` contributes ``n_h`` columns. An index
    outside ``[0, n_h)``, for example -1, contributes a zero block.

    Args:
        previous_action: int32[B, ACTION_HEADS] factor indices.

    Returns:
        float32[B, PREVIOUS_ACTION_WIDTH].
    """

    blocks = [
        jax.nn.one_hot(previous_action[:, head], size, dtype=jnp.float32)
        for head, size in enumerate(ACTION_HEAD_SIZES)
    ]
    return jnp.concatenate(blocks, axis=-1)


def _gru_step(params: BeliefParams, hidden: jax.Array, drive: jax.Array) -> jax.Array:
    """Advance a GRU by one step.

    Formulas, with sigma the logistic function, ``*`` elementwise, and
    ``W_z, W_r, W_h`` and ``U_z, U_r, U_h`` the column blocks of the two weights:

        z = sigma(x W_z + h U_z + b_z)
        r = sigma(x W_r + h U_r + b_r)
        h~ = tanh(x W_h + (r * h) U_h + b_h)
        h' = (1 - z) * h + z * h~

    Args:
        params: The belief parameters.
        hidden: float32[B, H], the previous latent h.
        drive: float32[B, H], the projected input x.

    Returns:
        float32[B, H], the next latent h'. It stays in [-1, 1] when h starts there.
    """

    gates = drive @ params.gru_input_weight + params.gru_bias
    update_in, reset_in, candidate_in = jnp.split(gates, 3, axis=-1)
    update_recurrent, reset_recurrent, candidate_recurrent = jnp.split(
        params.gru_recurrent_weight, 3, axis=-1
    )
    update = jax.nn.sigmoid(update_in + hidden @ update_recurrent)
    reset = jax.nn.sigmoid(reset_in + hidden @ reset_recurrent)
    candidate = jnp.tanh(candidate_in + (reset * hidden) @ candidate_recurrent)
    return (1.0 - update) * hidden + update * candidate


def _require_params(params: Any) -> None:
    """Raise TypeError or ValueError unless ``params`` has the documented shapes."""

    if not isinstance(params, BeliefParams):
        raise TypeError("params must be a BeliefParams")
    width = BELIEF_LATENT_WIDTH
    expected = {
        "projection_weight": (DRIVE_WIDTH, width),
        "projection_bias": (width,),
        "gru_input_weight": (width, 3 * width),
        "gru_recurrent_weight": (width, 3 * width),
        "gru_bias": (3 * width,),
    }
    for name, shape in expected.items():
        actual = tuple(getattr(params, name).shape)
        if actual != shape:
            raise ValueError(f"params.{name} must have shape {shape}, got {actual}")


def _positive_batch(batch: Any) -> int:
    """Return ``batch`` as a positive int. Bools are rejected."""

    if isinstance(batch, bool) or not isinstance(batch, int):
        raise TypeError(f"batch must be an integer, got {type(batch).__name__}")
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    return batch


def _require_carry(state: Any, lanes: int) -> None:
    """Raise unless ``state`` is a SharedBeliefCarry with shapes for ``lanes``."""

    if not isinstance(state, SharedBeliefCarry):
        raise TypeError("state must be a SharedBeliefCarry")
    expected = {
        "latent": (lanes, BELIEF_LATENT_WIDTH),
        "previous_self_health": (lanes,),
        "ticks_since_damage": (lanes,),
        "ticks_since_seen": (lanes,),
    }
    for name, shape in expected.items():
        actual = tuple(getattr(state, name).shape)
        if actual != shape:
            raise ValueError(f"state.{name} must have shape {shape}, got {actual}")


def _require_previous_action(previous_action: Any, lanes: int) -> jax.Array:
    """Return ``previous_action`` as an integer array of shape (lanes, ACTION_HEADS)."""

    value = jnp.asarray(previous_action)
    if not jnp.issubdtype(value.dtype, jnp.integer):
        raise TypeError(f"previous_action must be integer, got {value.dtype}")
    if value.shape != (lanes, ACTION_HEADS):
        raise ValueError(
            f"previous_action must have shape ({lanes}, {ACTION_HEADS}), got {value.shape}"
        )
    return value


def _require_lane_vector(value: Any, lanes: int, name: str) -> jax.Array:
    """Return ``value`` as float32 with shape (lanes,)."""

    array = jnp.asarray(value, dtype=jnp.float32)
    if array.shape != (lanes,):
        raise ValueError(f"{name} must have shape ({lanes},), got {array.shape}")
    return array


@dataclass(frozen=True, eq=False)
class SharedBelief:
    """The shared belief ``BeliefCore``, built from legal observation only.

    Attributes:
        params: The :class:`BeliefParams`, for example from
            :func:`init_belief_params`.
        maximum_gait_speed_blocks_per_second: ``agent_maximum_gait_speed`` of the
            environment, in blocks per second. Without it the self-speed feature
            is marked unknown. The observation does not publish this scale.
        chase_speed_blocks_per_second: ``target_chase_speed``, in blocks per
            second. Without it the closing-speed feature is marked unknown.
    """

    params: BeliefParams
    maximum_gait_speed_blocks_per_second: float | None = None
    chase_speed_blocks_per_second: float | None = None

    def __post_init__(self) -> None:
        """Validate the parameters and the optional speed scales."""

        _require_params(self.params)
        object.__setattr__(
            self,
            "maximum_gait_speed_blocks_per_second",
            positive_scale(
                "maximum_gait_speed_blocks_per_second",
                self.maximum_gait_speed_blocks_per_second,
            ),
        )
        object.__setattr__(
            self,
            "chase_speed_blocks_per_second",
            positive_scale(
                "chase_speed_blocks_per_second",
                self.chase_speed_blocks_per_second,
            ),
        )

    @property
    def contract(self) -> ProviderContract:
        """The provider contract that the ADK ``BeliefCore`` protocol requires."""

        return SHARED_BELIEF_CONTRACT

    def initial_state(self, batch: int) -> SharedBeliefCarry:
        """Return the episode-start carry for ``batch`` lanes.

        Args:
            batch: Number of lanes B, a positive int.

        Returns:
            SharedBeliefCarry with zero latent, previous health -1 (no reading),
            and both tick counters at TICKS_SATURATION.

        Raises:
            TypeError: If ``batch`` is not an integer.
            ValueError: If ``batch`` is not positive.
        """

        lanes = _positive_batch(batch)
        return SharedBeliefCarry(
            latent=jnp.zeros((lanes, BELIEF_LATENT_WIDTH), dtype=jnp.float32),
            previous_self_health=jnp.full((lanes,), -1.0, dtype=jnp.float32),
            ticks_since_damage=jnp.full((lanes,), TICKS_SATURATION, dtype=jnp.float32),
            ticks_since_seen=jnp.full((lanes,), TICKS_SATURATION, dtype=jnp.float32),
        )

    def __call__(
        self,
        state: SharedBeliefCarry,
        actor_input: ActorPolicyInput,
        previous_action: Any,
        previous_goal: Any,
        delta_time: jax.Array,
        key: jax.Array,
        /,
    ) -> BeliefUpdate:
        """Advance the belief by one decision.

        ``previous_goal`` and ``key`` match the ADK ``BeliefCore`` signature.
        This revision reads neither: SKILL.md records why, and the open items.

        Args:
            state: The SharedBeliefCarry from the previous update or
                :meth:`initial_state`.
            actor_input: The legal ``ActorPolicyInput``. Its
                ``legal_observation`` is the only input read.
            previous_action: int32[B, ACTION_HEADS], the previous factors.
            previous_goal: Unused.
            delta_time: float32[B], seconds since the previous decision. Zero
                when unknown, so the counters do not advance. Negative values
                are clamped to zero.
            key: Unused. The belief is deterministic.

        Returns:
            BeliefUpdate with ``state`` the next SharedBeliefCarry, ``belief`` a
            SharedBeliefState, ``prior`` the previous latent, ``posterior`` the
            new latent, ``uncertainty`` zeros float32[B, 10], and ``diagnostics``
            with ``features``, ``feature_known`` and ``legal_valid``.

        Raises:
            TypeError: If ``actor_input`` is not an ActorPolicyInput, or
                ``previous_action`` is not an integer array, or ``state`` is
                not a SharedBeliefCarry.
            ValueError: If a shape does not match the batch size.
        """

        if not isinstance(actor_input, ActorPolicyInput):
            raise TypeError("actor_input must be an ActorPolicyInput")
        readings = observation_readings(
            actor_input.legal_observation,
            maximum_gait_speed=self.maximum_gait_speed_blocks_per_second,
            chase_speed=self.chase_speed_blocks_per_second,
        )
        lanes = readings.valid.shape[0]
        _require_carry(state, lanes)
        previous = _require_previous_action(previous_action, lanes)
        # Unknown time arrives as zero. Negative time is clamped to zero so
        # that no counter can move backwards (the ADK core clamps the same way).
        delta = jnp.maximum(_require_lane_vector(delta_time, lanes, "delta_time"), 0.0)
        # Ticks here are environment decision steps of STEP_SECONDS (the gym's
        # loaded_dt, 0.045 s), the same unit as J0's 40-tick window, the
        # composition task's trigger window and the session clock. The server's
        # 30 ticks per second is a different clock and must not be used here.
        elapsed_ticks = delta / STEP_SECONDS

        has_health_reference = state.previous_self_health >= 0.0
        dropped = (
            readings.valid
            & has_health_reference
            & (readings.self_health < state.previous_self_health)
        )
        ticks_since_damage = jnp.where(
            dropped,
            0.0,
            jnp.minimum(state.ticks_since_damage + elapsed_ticks, TICKS_SATURATION),
        )
        seen_now = readings.valid & readings.visible
        ticks_since_seen = jnp.where(
            seen_now,
            0.0,
            jnp.minimum(state.ticks_since_seen + elapsed_ticks, TICKS_SATURATION),
        )
        next_previous_health = jnp.where(
            readings.valid,
            jnp.maximum(readings.self_health, 0.0),
            state.previous_self_health,
        )

        values = dict(readings.values)
        known = dict(readings.known)
        values["ticks_since_damage"] = ticks_since_damage / TICKS_SATURATION
        known["ticks_since_damage"] = readings.valid & has_health_reference
        values["ticks_since_target_seen"] = ticks_since_seen / TICKS_SATURATION
        known["ticks_since_target_seen"] = readings.valid
        features = jnp.stack(
            [jnp.where(known[name], values[name], 0.0) for name in BELIEF_FEATURE_NAMES],
            axis=-1,
        ).astype(jnp.float32)
        feature_known = jnp.stack([known[name] for name in BELIEF_FEATURE_NAMES], axis=-1)

        drive = jnp.concatenate(
            [
                features,
                feature_known.astype(jnp.float32),
                previous_action_one_hot(previous),
                elapsed_ticks[:, None],
            ],
            axis=-1,
        )
        projected = jnp.tanh(
            drive @ self.params.projection_weight + self.params.projection_bias
        )
        latent = _gru_step(self.params, state.latent, projected)

        next_state = SharedBeliefCarry(
            latent=latent,
            previous_self_health=next_previous_health,
            ticks_since_damage=ticks_since_damage,
            ticks_since_seen=ticks_since_seen,
        )
        return BeliefUpdate(
            state=next_state,
            belief=SharedBeliefState(
                features=features,
                feature_known=feature_known,
                latent=latent,
            ),
            prior=state.latent,
            posterior=latent,
            uncertainty=jnp.zeros((lanes, BELIEF_FEATURE_COUNT), dtype=jnp.float32),
            diagnostics={
                "features": features,
                "feature_known": feature_known,
                "legal_valid": readings.valid,
            },
        )


__all__ = [
    "ACTION_HEAD_SIZES",
    "BeliefParams",
    "DRIVE_WIDTH",
    "PREVIOUS_ACTION_WIDTH",
    "SHARED_BELIEF_CONTRACT",
    "SharedBelief",
    "SharedBeliefCarry",
    "init_belief_params",
    "previous_action_one_hot",
]
