"""Pursuit skill: the pinned pursuit_contact candidate v0001 behind its scoped loader.

The weights, the manifest and the scope belong to
``agent_skills/policies/pursuit_contact/`` (gitignored in this repository, present
on disk). This module does not retrain, re-export or copy them. It calls that
package's ``load(batch)``, which checks the checkpoint bytes, the observation
width, the head sizes and the scope hash before returning. It then adds the
``SkillContract``.

Facts the contract records:

* Checkpoint sha256 = ``policy.json["checkpoint_sha256"]``. ``load`` checks it.
* Scope sha256 = ``pursuit_learner_action_scope(allow_jump=False).contract_sha256``.
  ``load`` checks it against the checkpoint metadata. This module also checks it
  against ``policy.json["action_scope_sha256"]``.
* Recurrent memory width 128 = SKILL_STATE_WIDTH. Heads (12, 17, 2, 2, 37, 9, 9,
  5, 10, 2, 2, 17), total 124.

Policy call. ``policy(memory, actor_input, key) -> (memory, factors)``. Inside
``load``, the runtime mask is intersected with the scope before sampling. The
memory is the GRU carry, float32[B, 128].

Heavy imports (the agent_skills package, the arena contracts) happen inside
``load_pursuit_skill``, so importing this module stays cheap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp

from adk.architecture.inputs import ActorPolicyInput
from arena.training.skills.program import FactoredActionScope
from experimental.neural_agent.contracts import (
    ACTION_HEADS,
    SKILL_STATE_WIDTH,
    SharedBeliefState,
    SkillContract,
)
from experimental.neural_agent.skill_library._common import (
    require_scoped_factors,
    require_shape,
    validate_batch,
    validate_skill_inputs,
)

PURSUIT_SKILL_ID: str = "pursuit"
PURSUIT_VERSION: str = "v0001"


class PursuitSkill:
    """Skill wrapper around the scoped pursuit_contact policy function.

    Implements the ``Skill`` protocol. ``contract`` is set by
    :func:`load_pursuit_skill` from the pinned manifest.
    """

    def __init__(
        self,
        contract: SkillContract,
        policy: Callable[..., tuple[jax.Array, jax.Array]],
        scope: FactoredActionScope,
    ) -> None:
        """Bind the contract, the ``(memory, actor_input, key)`` policy and its scope.

        Raises:
            TypeError: If an argument has the wrong type.
        """

        if not isinstance(contract, SkillContract):
            raise TypeError("contract must be a SkillContract")
        if not callable(policy):
            raise TypeError("policy must be callable")
        if not isinstance(scope, FactoredActionScope):
            raise TypeError("scope must be a FactoredActionScope")
        self.contract = contract
        self._policy = policy
        self._scope = scope

    def initial_state(self, batch: int) -> jax.Array:
        """Return the zero GRU memory, float32[batch, SKILL_STATE_WIDTH].

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
        """Advance the policy one step and return ``(memory, factors)``.

        ``belief`` and ``params`` are validated and not read. The policy reads the
        dense ``actor_input.observation`` and the scoped mask.

        Returns:
            ``(next_state, factors)``: float32[B, 128] and int32[B, 12].

        Raises:
            ValueError: If the policy returns a wrong shape, or a factor is masked
                on a concrete call.
        """

        lanes = validate_skill_inputs(
            state, belief, actor_input, params, skill_id=PURSUIT_SKILL_ID
        )
        next_state, factors = self._policy(state, actor_input, key)
        next_state = jnp.asarray(next_state, dtype=jnp.float32)
        factors = jnp.asarray(factors, dtype=jnp.int32)
        require_shape(next_state, (lanes, SKILL_STATE_WIDTH), "pursuit: next_state")
        require_shape(factors, (lanes, ACTION_HEADS), "pursuit: factors")
        require_scoped_factors(factors, actor_input.action_mask, self._scope, "pursuit")
        return next_state, factors


def load_pursuit_skill(batch: int) -> PursuitSkill:
    """Load the pinned candidate and return it as a :class:`PursuitSkill`.

    Args:
        batch: Number of lanes B used for the initial memory check, a positive int.

    Returns:
        A PursuitSkill whose contract records the pinned checkpoint and scope hashes.

    Raises:
        TypeError: If ``batch`` is not an integer.
        ValueError: If the pinned files fail the loader's checks, or the manifest
            disagrees with the current scope.
    """

    from agent_skills.policies.pursuit_contact import policy as pursuit_policy
    from arena.training.contracts.pursuit import pursuit_learner_action_scope

    lanes = validate_batch(batch)
    policy_function, memory = pursuit_policy.load(lanes)
    if tuple(memory.shape) != (lanes, SKILL_STATE_WIDTH):
        raise ValueError(
            f"pursuit memory must have shape {(lanes, SKILL_STATE_WIDTH)}, got {tuple(memory.shape)}"
        )
    record_path = Path(pursuit_policy.__file__).parent / PURSUIT_VERSION / "policy.json"
    record = json.loads(record_path.read_text())
    scope = pursuit_learner_action_scope(allow_jump=False)
    scope_sha256 = scope.contract_sha256
    if record["action_scope_sha256"].upper() != scope_sha256.upper():
        raise ValueError(
            "pursuit manifest action_scope_sha256 differs from the current scope: "
            f"{record['action_scope_sha256']} != {scope_sha256}"
        )
    contract = SkillContract(
        skill_id=PURSUIT_SKILL_ID,
        version=PURSUIT_VERSION,
        observation_groups=("observation", "action_mask"),
        action_scope_sha256=scope_sha256,
        param_names=(),
        termination=("distance_below_one_block", "horizon_256_ticks"),
        success_signals=("distance_below_one_block",),
        failure_signals=("horizon_256_ticks",),
        checkpoint_sha256=record["checkpoint_sha256"],
    )
    return PursuitSkill(contract, policy_function, scope)


__all__ = ["PURSUIT_SKILL_ID", "PURSUIT_VERSION", "PursuitSkill", "load_pursuit_skill"]
