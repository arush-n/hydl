"""Exact-scene JAX evaluation tools with mandatory control-suite support.

The horizon is not a free parameter (GAP-14)
--------------------------------------------
Completions out of 4 lanes, measured 2026-08-03 on ``combat/open_flat_control``,
batch 4, key 7, untrained parameters:

==========  =======  =============  ==============  ===========  ==============
max_steps   trained  uniform_legal  untrained_init  always_idle  verdict
==========  =======  =============  ==============  ===========  ==============
32          0        0              0               0            **vacuous**
128         0        0              0               0            **vacuous**
256         0        1              0               4            idle separates
384         0        4              2               4            policy arms split
512         1        4              3               4            all arms live
1024        3        4              4               4            fully resolved
==========  =======  =============  ==============  ===========  ==============

Below :data:`EVALUATION_HORIZON_FLOOR` **no arm completes an episode**, so the
suite cannot distinguish a trained policy from one that does nothing -- not a
subtle bias, no discriminating power at all. A run there produces a ranking
that looks real and means nothing, which is why a short horizon warns rather
than passing quietly.

At the floor itself only ``always_idle`` separates; ``trained`` and
``untrained_init`` are still identical at 0/4. The comparison most callers
actually want needs more, which is why the default is
:data:`EVALUATION_HORIZON_DEFAULT` rather than the floor.

**There is no upper limit.** An earlier version of this module published a
"ceiling" of 384 on the strength of one ``max_steps=512`` run that died with
exit 255 and no traceback. It does not reproduce: 512 and 1024 both run clean,
and resident memory is flat across the whole range (~3.6 GB at 384, 512 and
1024) because the ``lax.scan`` here accumulates nothing per step. The kill was
almost certainly memory pressure from a concurrent JAX suite, not a property of
the horizon. Do not re-add a ceiling without a reproduction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v3.policy import (
    neutral_arsenal_policy_action_factors,
)
from hytalegym.jax.training.policy import (
    apply_policy,
    initialize_policy,
    sample_actions as gym_sample_actions,
)
from hytalegym.jax.training.ppo import combat_episode_outcome
from hytalegym.jax.training.types import PPOConfig, RecurrentPolicyParams

from adk.architecture.inputs import actor_policy_input
from adk.contracts.stamp import require_current
from adk.policy import Policy
from adk.policy.random_policy import uniform_legal
from adk.runtime.env_adapter import BuiltAgent
from adk.runtime.tree import (
    ENVIRONMENT_STATE_SELECTOR_SCHEMA,
    select_environment_rows,
)
from arena.evaluation import (
    ArmResult,
    EvaluationReplicate,
    PromotionGate,
    assess_promotion,
)

#: Below this **no arm completes an episode** and the suite measures nothing.
#: A hard minimum, not a recommendation -- at 256 only ``always_idle``
#: separates and the trained/untrained comparison is still 0/4 against 0/4.
EVALUATION_HORIZON_FLOOR: int = 256

#: Default horizon: the shortest measured value at which every arm completes
#: at least one episode, so all four are actually being compared.
EVALUATION_HORIZON_DEFAULT: int = 512


class EvaluationResult(NamedTuple):
    completed: jax.Array
    success: jax.Array
    death: jax.Array
    simultaneous: jax.Array
    other_terminal: jax.Array
    timed_out: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array

    @property
    def completions(self) -> int:
        """Lanes that reached a terminal state. Concrete results only."""

        return int(jnp.sum(self.completed))

    @property
    def timeouts(self) -> int:
        return int(jnp.sum(self.timed_out))


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    trained: EvaluationResult
    uniform_legal: EvaluationResult
    untrained_init: EvaluationResult
    always_idle: EvaluationResult
    trained_decode_mode: str
    stamp_sha256: str
    scene: dict[str, object]
    environment_state_selector_schema: str

    @property
    def arms(self) -> dict[str, EvaluationResult]:
        return {
            "trained": self.trained,
            "uniform_legal": self.uniform_legal,
            "untrained_init": self.untrained_init,
            "always_idle": self.always_idle,
        }

    @property
    def vacuous(self) -> bool:
        """True when no arm completed a single episode.

        The signature of an evaluation that measured nothing. Every arm timing
        out means the ranking below it is noise, so this is exposed as one
        boolean rather than left to be reconstructed from four timeout counts
        nobody reads.
        """

        return all(arm.completions == 0 for arm in self.arms.values())

    def diagnosis(self) -> str:
        """One line naming what the run can and cannot support."""

        counts = ", ".join(
            f"{name} {arm.completions} completed / {arm.timeouts} timed out"
            for name, arm in self.arms.items()
        )
        if self.vacuous:
            return (
                "VACUOUS: no arm completed an episode, so this suite cannot "
                f"distinguish any policy from any other -- {counts}. Raise "
                f"max_steps to at least {EVALUATION_HORIZON_FLOOR}."
            )
        return counts


def check_horizon(max_steps: int, *, allow_short_horizon: bool = False) -> None:
    """Warn when the horizon is too short to measure anything.

    A warning rather than an error: the run still produces numbers, and a
    caller doing a deliberate plumbing check has a legitimate reason to be
    below the floor. There is deliberately **no upper check** -- see the module
    docstring; the ceiling this module once published did not reproduce.
    """

    if max_steps < EVALUATION_HORIZON_FLOOR and not allow_short_horizon:
        warnings.warn(
            f"max_steps={max_steps} is below the measured evaluation floor of "
            f"{EVALUATION_HORIZON_FLOOR}: on combat/open_flat_control every "
            "control arm times out with zero completions there, so the result "
            "cannot distinguish a trained policy from one that does nothing. "
            f"Raise max_steps to at least {EVALUATION_HORIZON_DEFAULT} for a "
            "comparison that separates all four arms, or pass "
            "allow_short_horizon=True if this is a plumbing check rather than "
            "a measurement.",
            RuntimeWarning,
            stacklevel=3,
        )


def recurrent_policy(
    params: RecurrentPolicyParams,
    head_sizes: tuple[int, ...],
    *,
    decode_mode: str = "factored_argmax",
) -> tuple[Policy, jax.Array]:
    """Adapt upstream recurrent policy weights to the ADK policy protocol."""

    if decode_mode not in {"factored_argmax", "stochastic_sampling"}:
        raise ValueError(
            "decode_mode must be 'factored_argmax' or 'stochastic_sampling'"
        )
    recurrent_size = int(params.gru.recurrent_kernel.shape[0])

    def policy(carry, policy_input, key):
        next_carry, logits, _ = apply_policy(
            params,
            policy_input.observation,
            carry,
            policy_input.action_mask,
        )
        if decode_mode == "factored_argmax":
            factors = _factored_argmax(logits, head_sizes)
        else:
            factors, _ = gym_sample_actions(
                key,
                logits,
                head_sizes,
                transport="factors",
            )
        return next_carry, factors

    return policy, jnp.zeros((1, recurrent_size), dtype=jnp.float32)


def evaluate(
    built: BuiltAgent,
    policy: Policy,
    key: jax.Array,
    *,
    max_steps: int = EVALUATION_HORIZON_DEFAULT,
    initial_carry: Any = None,
    compile: bool = True,
    allow_short_horizon: bool = False,
) -> EvaluationResult:
    """Evaluate one episode per lane on the exact environment already built.

    ``allow_short_horizon`` silences the below-floor warning. Pass it when the
    run is deliberately a plumbing check rather than a measurement, so the
    intent is visible at the call site instead of the warning being filtered
    somewhere else.
    """

    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an integer")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    check_horizon(max_steps, allow_short_horizon=allow_short_horizon)
    require_current(built.stamp, built.world_geometry_config)

    def run(root_key):
        reset_key, rollout_key = jax.random.split(root_key)
        state, policy_input = built.reset(
            jax.random.split(reset_key, built.batch)
        )
        carry = initial_carry
        if carry is not None:
            carry = jax.tree.map(
                lambda value: (
                    jnp.broadcast_to(value, (built.batch,) + value.shape[1:])
                    if value.shape[0] == 1 and built.batch != 1
                    else value
                ),
                carry,
            )
        carry_structure = jax.tree.structure(carry)
        initial = (
            state,
            policy_input,
            carry,
            jnp.zeros((built.batch,), dtype=jnp.bool_),
            jnp.zeros((built.batch,), dtype=jnp.bool_),
            jnp.zeros((built.batch,), dtype=jnp.bool_),
            jnp.zeros((built.batch,), dtype=jnp.bool_),
            jnp.zeros((built.batch,), dtype=jnp.bool_),
            jnp.zeros((built.batch,), dtype=jnp.float32),
            jnp.zeros((built.batch,), dtype=jnp.int32),
        )

        def body(loop, step_key):
            (
                state,
                policy_input,
                carry,
                finished,
                success,
                death,
                simultaneous,
                other,
                episode_return,
                episode_length,
            ) = loop
            policy_key, environment_key = jax.random.split(step_key)
            candidate_carry, factors = policy(
                carry,
                actor_policy_input(
                    state,
                    policy_input,
                ),
                policy_key,
            )
            if jax.tree.structure(candidate_carry) != carry_structure:
                raise TypeError(
                    "policy carry PyTree must match initial_carry; pass an "
                    "explicit initial carry for every stateful policy"
                )
            (
                candidate_state,
                candidate_policy_input,
                reward,
                done,
                _info,
            ) = built.step(
                state,
                factors,
                jax.random.split(environment_key, built.batch),
            )
            outcome = combat_episode_outcome(
                candidate_state.environment.runtime.combat,
                done,
            )
            active = ~finished
            newly_done = active & done
            state = select_environment_rows(
                active,
                candidate_state,
                state,
            )
            policy_input = select_environment_rows(
                active,
                candidate_policy_input,
                policy_input,
            )
            if carry is not None:
                carry = select_environment_rows(
                    active,
                    candidate_carry,
                    carry,
                )
            episode_return += jnp.where(active, reward, jnp.float32(0.0))
            episode_length += active.astype(jnp.int32)
            success |= newly_done & outcome.success
            death |= newly_done & outcome.death
            simultaneous |= newly_done & outcome.simultaneous
            other |= newly_done & outcome.other
            finished |= newly_done
            return (
                state,
                policy_input,
                carry,
                finished,
                success,
                death,
                simultaneous,
                other,
                episode_return,
                episode_length,
            ), None

        final, _ = jax.lax.scan(
            body,
            initial,
            jax.random.split(rollout_key, max_steps),
        )
        (
            _state,
            _policy_input,
            _carry,
            completed,
            success,
            death,
            simultaneous,
            other,
            episode_return,
            episode_length,
        ) = final
        return EvaluationResult(
            completed=completed,
            success=success,
            death=death,
            simultaneous=simultaneous,
            other_terminal=other,
            timed_out=~completed,
            episode_return=episode_return,
            episode_length=episode_length,
        )

    runner = jax.jit(run) if compile else run
    result = runner(key)
    require_current(built.stamp, built.world_geometry_config)
    return result


def evaluate_controls(
    built: BuiltAgent,
    trained_params: RecurrentPolicyParams,
    config: PPOConfig,
    key: jax.Array,
    *,
    max_steps: int = EVALUATION_HORIZON_DEFAULT,
    trained_decode_mode: str = "factored_argmax",
    compile: bool = True,
    allow_short_horizon: bool = False,
) -> EvaluationSuite:
    """Run trained, uniform, seeded-untrained, and idle on identical resets.

    Warns twice if the horizon is wrong: once up front from the requested
    ``max_steps``, and again afterwards if the arms actually came back empty.
    The second warning is the load-bearing one -- it is measured rather than
    predicted, and it fires even when the horizon looked fine.
    """

    expected = (
        built.observation_size,
        built.logit_size,
        built.stamp.action_head_sizes,
        "factors",
    )
    actual = (
        config.observation_size,
        config.action_size,
        config.action_head_sizes,
        config.action_transport,
    )
    if actual != expected:
        raise ValueError("evaluation PPO config does not match BuiltAgent")
    untrained_key, evaluation_key = jax.random.split(key)
    untrained_params = initialize_policy(untrained_key, config)
    trained_policy, trained_carry = recurrent_policy(
        trained_params,
        built.stamp.action_head_sizes,
        decode_mode=trained_decode_mode,
    )
    untrained_policy, untrained_carry = recurrent_policy(
        untrained_params,
        built.stamp.action_head_sizes,
        decode_mode="factored_argmax",
    )

    def idle(carry, policy_input, _key):
        return carry, neutral_arsenal_policy_action_factors(
            int(policy_input.observation.shape[0])
        )

    kwargs = {
        "max_steps": max_steps,
        "compile": compile,
        "allow_short_horizon": True,  # warned once here, not four times
    }
    check_horizon(max_steps, allow_short_horizon=allow_short_horizon)
    suite = EvaluationSuite(
        trained=evaluate(
            built,
            trained_policy,
            evaluation_key,
            initial_carry=trained_carry,
            **kwargs,
        ),
        uniform_legal=evaluate(
            built,
            uniform_legal,
            evaluation_key,
            **kwargs,
        ),
        untrained_init=evaluate(
            built,
            untrained_policy,
            evaluation_key,
            initial_carry=untrained_carry,
            **kwargs,
        ),
        always_idle=evaluate(
            built,
            idle,
            evaluation_key,
            **kwargs,
        ),
        trained_decode_mode=trained_decode_mode,
        stamp_sha256=built.stamp.stamp_sha256(),
        scene=built.scene.metadata(),
        environment_state_selector_schema=(
            ENVIRONMENT_STATE_SELECTOR_SCHEMA
        ),
    )
    if suite.vacuous:
        warnings.warn(
            f"this evaluation measured nothing: {suite.diagnosis()}",
            RuntimeWarning,
            stacklevel=2,
        )
    return suite


def _factored_argmax(
    logits: jax.Array,
    head_sizes: tuple[int, ...],
) -> jax.Array:
    boundaries = []
    offset = 0
    for size in head_sizes[:-1]:
        offset += size
        boundaries.append(offset)
    return jnp.stack(
        tuple(
            jnp.argmax(head, axis=-1).astype(jnp.int32)
            for head in jnp.split(logits, tuple(boundaries), axis=-1)
        ),
        axis=-1,
    )


__all__ = [
    "EVALUATION_HORIZON_DEFAULT",
    "EVALUATION_HORIZON_FLOOR",
    "EvaluationResult",
    "EvaluationReplicate",
    "EvaluationSuite",
    "ArmResult",
    "PromotionGate",
    "assess_promotion",
    "check_horizon",
    "Policy",
    "evaluate",
    "evaluate_controls",
    "recurrent_policy",
]
