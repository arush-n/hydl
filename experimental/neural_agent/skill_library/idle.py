"""Idle skill: the physical neutral on every head, restricted to the runtime mask.

Scope. ``IDLE_SCOPE`` allows exactly one choice per head, the neutral of
:func:`neutral_choice`. Signed look heads centre at ``size // 2``, so a bin of 0
is a hard turn and not a no-op. Every other head neutralises at 0.

Restriction. For lane ``b`` and head ``h`` the factor is the scope choice when
``action_mask[b, HEAD_OFFSETS[h] + choice]`` is True. Otherwise the lane has no
legal idle action. A concrete call then raises ``ValueError``.

Skill-local state. None. ``initial_state`` and every returned state are zeros,
float32[B, SKILL_STATE_WIDTH].
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from adk.architecture.inputs import ActorPolicyInput
from arena.training.skills.program import FactoredActionScope
from experimental.neural_agent.contracts import (
    SKILL_IDLE,
    SKILL_STATE_WIDTH,
    SharedBeliefState,
    SkillContract,
)
from experimental.neural_agent.skill_library._common import (
    HEAD_NAMES,
    HEAD_OFFSETS,
    HEAD_SIZES,
    first_legal,
    neutral_choice,
    require_concrete_true,
    require_scoped_factors,
    validate_batch,
    validate_skill_inputs,
)

IDLE_VERSION: str = "v0001"

#: The one-choice-per-head scope of the idle skill.
IDLE_SCOPE: FactoredActionScope = FactoredActionScope(
    tuple(
        (neutral_choice(name, size),)
        for name, size in zip(HEAD_NAMES, HEAD_SIZES, strict=True)
    )
)


class IdleSkill:
    """Emit the neutral choice on every head, if the runtime mask allows it.

    Implements the ``Skill`` protocol from ``experimental/neural_agent/contracts.py``.
    """

    contract: SkillContract = SkillContract(
        skill_id=SKILL_IDLE,
        version=IDLE_VERSION,
        observation_groups=("action_mask",),
        action_scope_sha256=IDLE_SCOPE.contract_sha256,
        param_names=(),
        termination=("goal_replaced",),
        success_signals=("neutral_emitted",),
        failure_signals=("neutral_masked",),
        checkpoint_sha256=None,
    )

    def initial_state(self, batch: int) -> jax.Array:
        """Return the zero skill state, float32[batch, SKILL_STATE_WIDTH].

        Args:
            batch: Number of lanes B, a positive int.
        """

        lanes = validate_batch(batch)
        return jnp.zeros((lanes, SKILL_STATE_WIDTH), dtype=jnp.float32)

    def __call__(
        self,
        state: jax.Array,
        belief: SharedBeliefState,
        actor_input: ActorPolicyInput,
        params: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return ``(next_state, factors)`` with ``factors`` int32[B, 12].

        ``belief``, ``params`` and ``key`` are validated or ignored. The idle
        skill reads only ``actor_input.action_mask``.
        """

        lanes = validate_skill_inputs(
            state, belief, actor_input, params, skill_id=SKILL_IDLE
        )
        mask = actor_input.action_mask
        columns = []
        feasible = jnp.ones((lanes,), dtype=jnp.bool_)
        for head, (offset, size) in enumerate(zip(HEAD_OFFSETS, HEAD_SIZES, strict=True)):
            (scoped_choice,) = IDLE_SCOPE.allowed_choices[head]
            preference = jnp.full((lanes,), scoped_choice, dtype=jnp.int32)
            choice, ok = first_legal(
                [preference], mask[:, offset : offset + size]
            )
            columns.append(choice)
            feasible = feasible & ok
        factors = jnp.stack(columns, axis=-1)
        require_concrete_true(
            feasible,
            "idle: the neutral choice is masked on at least one head",
        )
        require_scoped_factors(factors, mask, IDLE_SCOPE, "idle")
        return self.initial_state(lanes), factors


__all__ = ["IDLE_SCOPE", "IDLE_VERSION", "IdleSkill"]
