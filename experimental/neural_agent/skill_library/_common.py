"""Shared helpers for the Level 1 skills (ADR-001 D1).

A factor vector is ``int32[B, ACTION_HEADS]``: one choice per published arsenal
head. The head sizes come from ``ARSENAL_POLICY_ACTION_HEAD_SIZES``, so choice
``c`` on head ``h`` is legal for lane ``b`` exactly when
``action_mask[b, HEAD_OFFSETS[h] + c]`` is True.

Failure policy. Wrong Python types raise ``TypeError`` and wrong shapes raise
``ValueError`` at call time. A data-dependent illegality is raised eagerly by
:func:`require_concrete_true` whenever the arrays are concrete. Under
``jax.jit`` the arrays are traced and cannot be inspected, so the skill returns
its factors and the dispatcher reports legality in ``diagnostics["legal"]``.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from adk.architecture.inputs import ActorPolicyInput
from arena.training.skills.program import FactoredActionScope
from experimental.neural_agent.contracts import (
    ACTION_HEADS,
    BELIEF_FEATURE_COUNT,
    BELIEF_LATENT_WIDTH,
    PARAM_WIDTH,
    SKILL_STATE_WIDTH,
    SharedBeliefState,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)

HEAD_NAMES: tuple[str, ...] = tuple(ARSENAL_POLICY_ACTION_HEAD_NAMES)
HEAD_SIZES: tuple[int, ...] = tuple(ARSENAL_POLICY_ACTION_HEAD_SIZES)
if len(HEAD_SIZES) != ACTION_HEADS:
    raise RuntimeError(
        f"arsenal policy publishes {len(HEAD_SIZES)} heads; contracts expects {ACTION_HEADS}"
    )
# Start column of each head inside the flat action mask, for example
# HEAD_OFFSETS[1] == 12 because base_action has 12 choices.
HEAD_OFFSETS: tuple[int, ...] = tuple(
    sum(HEAD_SIZES[:head]) for head in range(ACTION_HEADS)
)
MASK_WIDTH: int = sum(HEAD_SIZES)

# Signed look heads centre at size // 2. Every other head neutralises at 0.
SIGNED_LOOK_HEADS: frozenset[str] = frozenset(
    {"yaw_delta_bins", "body_yaw_delta_bins", "pitch_delta_bins"}
)


def neutral_choice(name: str, size: int) -> int:
    """Return the physical no-op choice of one head.

    Formula: ``size // 2`` for a signed look head, otherwise ``0``. This is the
    rule in ``arena/training/skills/program.py`` (``_NEUTRAL_FACTOR_CHOICES``)
    and ``neutral_arsenal_policy_action_factors``. A test pins the agreement.
    """

    return size // 2 if name in SIGNED_LOOK_HEADS else 0


def validate_batch(batch: int) -> int:
    """Return ``batch`` as a positive int. Bools and floats are rejected."""

    if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)):
        raise TypeError(f"batch must be an integer, got {type(batch).__name__}")
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    return int(batch)


def require_shape(value: object, expected: tuple[int, ...], name: str) -> None:
    """Raise ValueError unless ``value`` has exactly the ``expected`` shape."""

    shape = tuple(jnp.shape(value))
    if shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {shape}")


def require_dtype(value: object, dtype: object, name: str) -> None:
    """Raise TypeError unless ``value`` has exactly ``dtype``."""

    actual = getattr(value, "dtype", None)
    if actual is None or np.dtype(actual) != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}, got {actual}")


def validate_skill_inputs(
    state: object,
    belief: object,
    actor_input: object,
    params: object,
    *,
    skill_id: str,
) -> int:
    """Check one skill call and return the lane count B.

    Args:
        state: float32[B, SKILL_STATE_WIDTH], skill-local state.
        belief: SharedBeliefState with features float32[B, 10], feature_known
            bool[B, 10] and latent float32[B, 64].
        actor_input: ActorPolicyInput. ``JaxTrainingInput`` is refused.
        params: float32[B, PARAM_WIDTH], goal parameters.
        skill_id: Name used in error messages.

    Returns:
        B, taken from ``actor_input.action_mask``.

    Raises:
        TypeError: If a value has the wrong Python type or dtype.
        ValueError: If a shape is wrong.
    """

    if not isinstance(actor_input, ActorPolicyInput):
        raise TypeError(
            f"{skill_id}: actor_input must be an ActorPolicyInput, "
            f"got {type(actor_input).__name__}"
        )
    mask = actor_input.action_mask
    require_dtype(mask, jnp.bool_, f"{skill_id}: action_mask")
    shape = tuple(jnp.shape(mask))
    if len(shape) != 2 or shape[0] < 1 or shape[1] != MASK_WIDTH:
        raise ValueError(
            f"{skill_id}: action_mask must have shape [B,{MASK_WIDTH}], got {shape}"
        )
    lanes = int(shape[0])
    require_dtype(state, jnp.float32, f"{skill_id}: state")
    require_shape(state, (lanes, SKILL_STATE_WIDTH), f"{skill_id}: state")
    if not isinstance(belief, SharedBeliefState):
        raise TypeError(
            f"{skill_id}: belief must be a SharedBeliefState, got {type(belief).__name__}"
        )
    require_dtype(belief.features, jnp.float32, f"{skill_id}: belief.features")
    require_shape(
        belief.features, (lanes, BELIEF_FEATURE_COUNT), f"{skill_id}: belief.features"
    )
    require_dtype(belief.feature_known, jnp.bool_, f"{skill_id}: belief.feature_known")
    require_shape(
        belief.feature_known,
        (lanes, BELIEF_FEATURE_COUNT),
        f"{skill_id}: belief.feature_known",
    )
    require_dtype(belief.latent, jnp.float32, f"{skill_id}: belief.latent")
    require_shape(
        belief.latent, (lanes, BELIEF_LATENT_WIDTH), f"{skill_id}: belief.latent"
    )
    require_dtype(params, jnp.float32, f"{skill_id}: params")
    require_shape(params, (lanes, PARAM_WIDTH), f"{skill_id}: params")
    return lanes


def legal_choice(head_mask: jax.Array, choice: jax.Array) -> jax.Array:
    """Return bool[B]: True where ``choice`` (int32[B]) is an allowed column.

    ``head_mask`` is bool[B, size] for one head. An out-of-range choice is False.
    """

    size = head_mask.shape[1]
    in_range = (choice >= 0) & (choice < size)
    safe = jnp.clip(choice, 0, size - 1)
    return in_range & jnp.take_along_axis(head_mask, safe[:, None], axis=1)[:, 0]


def factors_legal(factors: jax.Array, action_mask: jax.Array) -> jax.Array:
    """Return bool[B]: True where every head's choice is legal under the mask.

    Args:
        factors: int32[B, ACTION_HEADS].
        action_mask: bool[B, MASK_WIDTH].
    """

    lanes = action_mask.shape[0]
    legal = jnp.ones((lanes,), dtype=jnp.bool_)
    for head, (offset, size) in enumerate(zip(HEAD_OFFSETS, HEAD_SIZES, strict=True)):
        legal = legal & legal_choice(
            action_mask[:, offset : offset + size], factors[:, head]
        )
    return legal


def scope_legal(factors: jax.Array, scope: FactoredActionScope) -> jax.Array:
    """Return bool[B]: True where every head's choice is inside ``scope``.

    ``scope.allowed_choices[h]`` lists the choices a skill may emit on head ``h``.
    This check is independent of the runtime mask.
    """

    lanes = factors.shape[0]
    legal = jnp.ones((lanes,), dtype=jnp.bool_)
    for head, (size, choices) in enumerate(zip(HEAD_SIZES, scope.allowed_choices, strict=True)):
        allowed = jnp.zeros((size,), dtype=jnp.bool_).at[
            jnp.asarray(choices, dtype=jnp.int32)
        ].set(True)
        table = jnp.broadcast_to(allowed, (lanes, size))
        legal = legal & legal_choice(table, factors[:, head])
    return legal


def first_legal(
    preferences: Sequence[jax.Array],
    head_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pick, per lane, the first preferred choice that the head mask allows.

    Args:
        preferences: Candidate choices, most preferred first. Each is int32[B].
        head_mask: bool[B, size] for one head.

    Returns:
        ``(choice, feasible)``: int32[B] and bool[B]. A lane with no legal
        candidate keeps the first preference and is ``feasible=False``. The caller
        must fail closed on that lane.
    """

    chosen = jnp.asarray(preferences[0], dtype=jnp.int32)
    feasible = jnp.zeros(chosen.shape, dtype=jnp.bool_)
    for candidate in preferences:
        candidate = jnp.asarray(candidate, dtype=jnp.int32)
        ok = legal_choice(head_mask, candidate)
        take = ok & ~feasible
        chosen = jnp.where(take, candidate, chosen)
        feasible = feasible | ok
    return chosen, feasible


def require_scoped_factors(
    factors: jax.Array,
    action_mask: jax.Array,
    scope: FactoredActionScope,
    label: str,
) -> None:
    """Raise ValueError when a concrete factor is outside ``scope`` or masked.

    The contract requires both: factors lie in the skill scope and in the runtime
    mask (``contracts.Skill``). Traced arrays are skipped, as in
    :func:`require_concrete_true`.
    """

    require_concrete_true(
        factors_legal(factors, action_mask) & scope_legal(factors, scope),
        f"{label}: a factor is outside the skill scope or masked by the runtime mask",
    )


def require_concrete_true(ok: jax.Array, message: str) -> None:
    """Raise ValueError when ``ok`` is concrete and any lane is False.

    Traced arrays (inside ``jax.jit``) cannot be inspected. The check is skipped
    for them, and the caller reports the same fact through diagnostics.
    """

    try:
        values = np.asarray(ok)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        return
    if not bool(values.all()):
        lanes = np.flatnonzero(~values).tolist()
        raise ValueError(f"{message}; failing lanes {lanes}")


__all__ = [
    "ACTION_HEADS",
    "HEAD_NAMES",
    "HEAD_OFFSETS",
    "HEAD_SIZES",
    "MASK_WIDTH",
    "SIGNED_LOOK_HEADS",
    "factors_legal",
    "first_legal",
    "legal_choice",
    "neutral_choice",
    "require_concrete_true",
    "require_scoped_factors",
    "scope_legal",
    "require_dtype",
    "require_shape",
    "validate_batch",
    "validate_skill_inputs",
]
