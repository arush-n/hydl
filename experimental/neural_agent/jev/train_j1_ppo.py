"""PPO for the J1 decision network over goal decisions, a semi-Markov decision process.

J1 (ADR-001 D2) makes one goal decision at a time: it picks a skill. The
environment then runs that frozen skill inside its own ``step`` until the
skill terminates or its commitment ends, and returns the reward discounted per
tick over that interval together with the number of ticks elapsed. One PPO
transition is therefore one decision, and the next decision's value is
discounted by ``gamma ** elapsed_ticks`` rather than by one tick's ``gamma``.

Formulas, for decision t that lasted k_t ticks and earned the in-interval
reward R_t = sum_{i < k_t} gamma ** i * r_i:

    delta_t = R_t + gamma ** k_t * (1 - done_t) * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + (gamma * lambda) ** k_t * (1 - done_t) * A_{t+1}

With every k_t = 1 this is ``calculate_gae`` from
``hytalegym.jax.training.ppo`` without time limits. The update is the
repository's clipped PPO: advantages standardized over the whole rollout,
ratio clip 0.2, clipped value loss weighted 0.5, entropy bonus 0.01,
global-norm clip 0.5, Adam with eps 1e-5, and 4 epochs of 8 minibatches, each
minibatch a set of whole lane sequences.

Public names:

* :class:`GoalDecisionEnv`: the protocol of a decision-level environment.
* :func:`collect_decisions`: one batched rollout, arrays ``[T, B]``.
* :func:`smdp_generalized_advantage`: GAE for variable-duration decisions,
  vendored from ``agents/dawn`` (local only, unpublished).
* :func:`decision_advantages`: that GAE with ``gamma ** elapsed_ticks``.
* :func:`train_j1_ppo`: the training loop.
* :class:`SyntheticGoalEnv`: a pure-JAX toy semi-MDP for smoke runs and tests.
* :class:`MlpPolicyParams`, :func:`init_mlp_policy`, :func:`apply_mlp_policy`:
  a local stand-in for the J1 network (see the TODO above them).

The CLI trains the stand-in on :class:`SyntheticGoalEnv` by default, on CPU in
seconds. ``--env sim`` trains J1 with a separate critic
(``jev/j1_actor_critic.py``) on ``jev/sim_goal_env.py`` over the
``combat_open_flat`` arena, and writes the J1Params to ``--out``::

    python -m experimental.neural_agent.jev.train_j1_ppo --updates 60
    python -m experimental.neural_agent.jev.train_j1_ppo --env sim --batch 8 \
        --decisions-per-update 6 --updates 3 --ticks 64 --out /tmp/j1_sim.npz
"""

from __future__ import annotations

import argparse
import math
import numbers
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple, Protocol, Sequence, runtime_checkable

import jax
import jax.numpy as jnp
import optax

from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_COUNT,
    BELIEF_LATENT_WIDTH,
    MAX_GOALS,
    PARAM_WIDTH,
    ComputeBudget,
    ExecutiveArrays,
    GoalPayload,
    SharedBeliefState,
)

# PPO settings equal to the PPOConfig defaults in
# HytaleRL/hytalegym/hytalegym/jax/training/types.py:27-37 (read 2026-10-16).
# They are copied rather than imported so that this module does not import the
# combat package. tests/experimental/neural_agent/test_na_jev_ppo.py checks
# that they still match.
GAMMA: float = 0.99
GAE_LAMBDA: float = 0.95
CLIP_EPSILON: float = 0.2
VALUE_COEFFICIENT: float = 0.5
ENTROPY_COEFFICIENT: float = 0.01
MAX_GRADIENT_NORM: float = 0.5
LEARNING_RATE: float = 3.0e-4
UPDATE_EPOCHS: int = 4
NUM_MINIBATCHES: int = 8

# Keys of every decision observation.
OBSERVATION_KEYS: frozenset[str] = frozenset(
    {"belief", "executive", "current_goal", "mask"}
)
# Logit given to a masked skill. It is finite, so the entropy term of a masked
# skill is 0 * -1e9 = 0 rather than NaN.
_MASKED_LOGIT = -1.0e9
# Columns of ExecutiveArrays.budget.
_BUDGET_WIDTH = 3

#: ``apply_fn(params, obs) -> (logits float32[B, K], value float32[B])``.
ApplyFn = Callable[[Any, dict[str, Any]], tuple[jax.Array, jax.Array]]


@runtime_checkable
class GoalDecisionEnv(Protocol):
    """A batched environment stepped once per J1 goal decision.

    ``reset`` and ``step`` must be pure JAX and traceable under ``jax.jit``.
    Every leaf of the state and of each observation has a leading lane axis B,
    because :func:`collect_decisions` resets finished lanes by selecting rows.

    An observation is a dict with exactly the keys in ``OBSERVATION_KEYS``:

    * ``"belief"``: :class:`SharedBeliefState` with ``features``
      float32[B, 10], ``feature_known`` bool[B, 10], ``latent`` float32[B, 64].
    * ``"executive"``: :class:`ExecutiveArrays` with ``skill_index``
      int32[B, 4], ``priority`` float32[B, 4], ``params`` float32[B, 4, 4],
      ``goal_valid`` bool[B, 4], ``budget`` float32[B, 3], ``version`` int32[B].
    * ``"current_goal"``: :class:`GoalPayload` of the goal in force.
    * ``"mask"``: bool[B, num_skills], True where a skill may be chosen. Every
      lane must allow at least one skill.

    Attributes:
        num_skills: Catalog size K. A decision is an int32 skill in [0, K).
        gamma: Per-tick discount that the environment applies inside one
            decision. The trainer uses the same value for ``gamma ** k``.
    """

    num_skills: int
    gamma: float

    def reset(self, keys: jax.Array) -> tuple[Any, dict[str, Any]]:
        """Start one episode per lane.

        Args:
            keys: B PRNG keys, one per lane (``jax.random.split(key, B)``).

        Returns:
            ``(state, obs)``: the state PyTree and the first observation.
        """

    def step(
        self, state: Any, skill_index: jax.Array, keys: jax.Array
    ) -> tuple[Any, dict[str, Any], jax.Array, jax.Array, jax.Array]:
        """Run the chosen skill, frozen, until it terminates or its commitment ends.

        Args:
            state: The state from ``reset`` or from the previous ``step``.
            skill_index: int32[B], the chosen skill, legal under ``obs["mask"]``.
            keys: B PRNG keys, one per lane.

        Returns:
            ``(state, obs, reward, elapsed_ticks, done)``. ``reward`` is
            float32[B], the decision's tick rewards r_i summed as
            ``sum_i gamma ** i * r_i``. ``elapsed_ticks`` is int32[B], at
            least 1. ``done`` is bool[B] and marks a terminal episode end,
            which is not bootstrapped. ``step`` does not reset finished lanes;
            :func:`collect_decisions` does.
        """


class DecisionRollout(NamedTuple):
    """T goal decisions over B lanes, time-major.

    Shapes, with T decisions, B lanes and K skills:
        obs: the observation dict of :class:`GoalDecisionEnv`, each leaf with
            leading axes [T, B]; ``obs["mask"]`` is bool[T, B, K].
        action: int32[T, B], the chosen skill.
        logp: float32[T, B], log-probability of ``action`` under the masked
            policy that chose it.
        value: float32[T, B], the value estimate at the decision.
        reward: float32[T, B], reward discounted per tick within the decision.
        elapsed_ticks: int32[T, B], ticks the decision lasted, at least 1.
        done: bool[T, B], True where the episode ended at this decision.
        bootstrap_value: float32[B], the value of the observation after the
            last decision. It is used only for lanes not done at T - 1.
    """

    obs: dict[str, Any]
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    elapsed_ticks: jax.Array
    done: jax.Array
    bootstrap_value: jax.Array


def _positive_int(value: Any, name: str) -> int:
    """Return ``value`` as an int of at least 1. Bools raise TypeError."""

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


def _check_observation(obs: Any, lanes: int, num_skills: int) -> None:
    """Raise unless ``obs`` has the protocol keys, record types and mask shape."""

    if not isinstance(obs, dict) or set(obs) != OBSERVATION_KEYS:
        raise ValueError(
            f"an observation must be a dict with keys {sorted(OBSERVATION_KEYS)}"
        )
    for name, kind in (
        ("belief", SharedBeliefState),
        ("executive", ExecutiveArrays),
        ("current_goal", GoalPayload),
    ):
        if not isinstance(obs[name], kind):
            raise TypeError(f'obs["{name}"] must be a {kind.__name__}')
    if obs["mask"].shape != (lanes, num_skills):
        raise ValueError(
            f'obs["mask"] must have shape {(lanes, num_skills)}, '
            f"got {obs['mask'].shape}"
        )


def _masked_policy(
    apply_fn: ApplyFn, params: Any, obs: dict[str, Any]
) -> tuple[jax.Array, jax.Array]:
    """Return ``(logits float32[N, K], value float32[N])`` with illegal skills masked."""

    logits, value = apply_fn(params, obs)
    mask = obs["mask"]
    if logits.shape != mask.shape or value.shape != mask.shape[:-1]:
        raise ValueError(
            f"apply_fn must return logits {mask.shape} and value {mask.shape[:-1]}, "
            f"got {logits.shape} and {value.shape}"
        )
    return jnp.where(mask, logits, _MASKED_LOGIT), value


def _action_log_probability(logits: jax.Array, action: jax.Array) -> jax.Array:
    """Return float32[N], log softmax(logits)[n, action[n]]."""

    log_probs = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(log_probs, action[:, None], axis=-1)[:, 0]


def _select_lanes(done: jax.Array, fresh: Any, live: Any) -> Any:
    """Return ``fresh`` rows where ``done`` is True and ``live`` rows elsewhere."""

    def pick(new: jax.Array, old: jax.Array) -> jax.Array:
        lane_mask = done.reshape(done.shape + (1,) * (old.ndim - 1))
        return jnp.where(lane_mask, new, old)

    return jax.tree.map(pick, fresh, live)


def collect_decisions(
    env: GoalDecisionEnv,
    params: Any,
    apply_fn: ApplyFn,
    key: jax.Array,
    num_decisions: int,
    batch: int,
) -> DecisionRollout:
    """Run the policy for ``num_decisions`` goal decisions in ``batch`` fresh lanes.

    Every lane starts from ``env.reset``. A lane whose decision ends its
    episode is reset before its next decision. Skills are sampled from
    softmax(logits) with illegal skills masked out. Traceable under ``jax.jit``.

    Args:
        env: A :class:`GoalDecisionEnv`.
        params: Policy parameters for ``apply_fn``.
        apply_fn: ``apply_fn(params, obs) -> (logits float32[B, K],
            value float32[B])`` for an observation dict with B lanes.
        key: A JAX PRNG key.
        num_decisions: T, a positive int.
        batch: B, a positive int.

    Returns:
        A :class:`DecisionRollout` with arrays [T, B].

    Raises:
        TypeError: If ``env`` is not a GoalDecisionEnv, ``num_decisions`` or
            ``batch`` is not an integer, or an observation record has the
            wrong type.
        ValueError: If ``num_decisions`` or ``batch`` is not positive, or the
            observation or the policy output has the wrong keys or shapes.
    """

    if not isinstance(env, GoalDecisionEnv):
        raise TypeError("env must implement GoalDecisionEnv (num_skills, gamma, reset, step)")
    steps = _positive_int(num_decisions, "num_decisions")
    lanes = _positive_int(batch, "batch")
    reset_key, scan_key = jax.random.split(key)
    first_state, first_obs = env.reset(jax.random.split(reset_key, lanes))
    _check_observation(first_obs, lanes, env.num_skills)

    def decide(carry, step_key):
        state, obs = carry
        action_key, env_key, refill_key = jax.random.split(step_key, 3)
        logits, value = _masked_policy(apply_fn, params, obs)
        action = jax.random.categorical(action_key, logits).astype(jnp.int32)
        logp = _action_log_probability(logits, action)
        next_state, next_obs, reward, elapsed, done = env.step(
            state, action, jax.random.split(env_key, lanes)
        )
        done = jnp.asarray(done, dtype=jnp.bool_)
        next_carry = jax.lax.cond(
            jnp.any(done),
            lambda live: _select_lanes(
                done, env.reset(jax.random.split(refill_key, lanes)), live
            ),
            lambda live: live,
            (next_state, next_obs),
        )
        record = (
            obs,
            action,
            logp,
            value,
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(elapsed, dtype=jnp.int32),
            done,
        )
        return next_carry, record

    (_, final_obs), records = jax.lax.scan(
        decide, (first_state, first_obs), jax.random.split(scan_key, steps)
    )
    _, bootstrap_value = _masked_policy(apply_fn, params, final_obs)
    return DecisionRollout(*records, bootstrap_value=bootstrap_value)


def smdp_generalized_advantage(
    reward: Any,
    value: Any,
    bootstrap_value: Any,
    valid: Any,
    terminated: Any,
    truncated: Any,
    bootstrap_discount: Any,
    trace_discount: Any,
) -> tuple[jax.Array, jax.Array]:
    """Compute GAE for variable-duration, decision-indexed transitions.

    ``reward`` is the discounted reward accumulated inside one physical
    decision interval. ``bootstrap_discount`` is ``gamma ** elapsed_ticks``
    and ``trace_discount`` is ``(gamma * lambda) ** elapsed_ticks``.  This
    keeps the optimizer indexed by deployable policy decisions without
    pretending that a 13-tick attack lifecycle lasted one tick.

    Truncations bootstrap from the final observation but stop the trace;
    authoritative terminals neither bootstrap nor continue the trace.  The
    caller must provide a padded suffix only: every last valid row is required
    to be terminal, truncated, or followed by another valid row.

    Vendored from ``agents/dawn/training/gpu/optimization.py:26`` (that file's
    sha256 was 17bbcb5c867c2400467deb0e149ea68cfaf71bb71c8d91020436d158081c4a73
    on 2026-10-16). ``agents/dawn`` is local-only and unpublished, so the
    function is copied here instead of imported. Only this paragraph differs
    from the source; keep the body in step with it.
    """

    rewards = jnp.asarray(reward, dtype=jnp.float32)
    values = jnp.asarray(value, dtype=jnp.float32)
    validity = jnp.asarray(valid, dtype=jnp.bool_)
    terminals = jnp.asarray(terminated, dtype=jnp.bool_)
    truncations = jnp.asarray(truncated, dtype=jnp.bool_)
    boot_discount = jnp.asarray(bootstrap_discount, dtype=jnp.float32)
    gae_discount = jnp.asarray(trace_discount, dtype=jnp.float32)
    expected = rewards.shape
    if rewards.ndim != 2:
        raise ValueError("SMDP transition arrays must have shape [decisions,batch]")
    for name, array in (
        ("value", values),
        ("valid", validity),
        ("terminated", terminals),
        ("truncated", truncations),
        ("bootstrap discount", boot_discount),
        ("trace discount", gae_discount),
    ):
        if array.shape != expected:
            raise ValueError(f"SMDP {name} must match reward shape")
    bootstrap = jnp.asarray(bootstrap_value, dtype=jnp.float32)
    if bootstrap.shape != expected[1:]:
        raise ValueError("SMDP bootstrap value must have shape [batch]")

    next_value = jnp.concatenate((values[1:], bootstrap[None, :]), axis=0)
    next_value = jnp.where(
        terminals,
        0.0,
        jnp.where(truncations, bootstrap[None, :], next_value),
    )

    def reverse(carry: jax.Array, item):
        (
            reward_t,
            value_t,
            next_t,
            valid_t,
            terminated_t,
            truncated_t,
            bootstrap_discount_t,
            trace_discount_t,
        ) = item
        delta = reward_t + bootstrap_discount_t * next_t - value_t
        trace_continues = ~(terminated_t | truncated_t)
        advantage = jnp.where(
            valid_t,
            delta + trace_discount_t * trace_continues.astype(jnp.float32) * carry,
            0.0,
        )
        return advantage, advantage

    _, advantages = jax.lax.scan(
        reverse,
        jnp.zeros_like(bootstrap),
        (
            rewards,
            values,
            next_value,
            validity,
            terminals,
            truncations,
            boot_discount,
            gae_discount,
        ),
        reverse=True,
    )
    return advantages, advantages + values


def decision_advantages(
    reward: jax.Array,
    value: jax.Array,
    done: jax.Array,
    elapsed_ticks: jax.Array,
    bootstrap_value: jax.Array,
    *,
    gamma: float,
    gae_lambda: float = GAE_LAMBDA,
) -> tuple[jax.Array, jax.Array]:
    """Return SMDP advantages and returns for decision-indexed transitions.

    A decision that lasted k ticks discounts the next value by ``gamma ** k``
    and the trace by ``(gamma * gae_lambda) ** k`` (module formulas). Every
    row is valid, ``done`` is an authoritative terminal, and nothing is
    truncated. With every k equal to 1 the result equals ``calculate_gae``.

    Args:
        reward: float32[T, B], reward discounted per tick within each decision.
        value: float32[T, B], value estimates at the decisions.
        done: bool[T, B], terminal decisions (no bootstrap; the trace stops).
        elapsed_ticks: int32[T, B], ticks each decision lasted, at least 1.
        bootstrap_value: float32[B], the value after the last decision.
        gamma: Per-tick discount.
        gae_lambda: Per-tick GAE lambda.

    Returns:
        ``(advantages, returns)``, both float32[T, B].
    """

    terminated = jnp.asarray(done, dtype=jnp.bool_)
    ticks = jnp.asarray(elapsed_ticks, dtype=jnp.float32)
    gamma_value = jnp.float32(gamma)
    return smdp_generalized_advantage(
        reward,
        value,
        bootstrap_value,
        jnp.ones_like(terminated),
        terminated,
        jnp.zeros_like(terminated),
        gamma_value**ticks,
        (gamma_value * jnp.float32(gae_lambda)) ** ticks,
    )


def _ppo_loss(
    params: Any, apply_fn: ApplyFn, minibatch: tuple[Any, ...]
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Clipped PPO loss for whole lane sequences, flattened from [T, b] to [T * b]."""

    obs, action, old_logp, old_value, advantage, target = jax.tree.map(
        lambda leaf: leaf.reshape((-1,) + leaf.shape[2:]), minibatch
    )
    logits, value = _masked_policy(apply_fn, params, obs)
    log_ratio = _action_log_probability(logits, action) - old_logp
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
    policy_loss = jnp.mean(
        jnp.maximum(-advantage * ratio, -advantage * clipped_ratio)
    )
    clipped_value = old_value + jnp.clip(value - old_value, -CLIP_EPSILON, CLIP_EPSILON)
    value_loss = 0.5 * jnp.mean(
        jnp.maximum(jnp.square(value - target), jnp.square(clipped_value - target))
    )
    log_probs = jax.nn.log_softmax(logits)
    entropy = jnp.mean(-jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1))
    total = policy_loss + VALUE_COEFFICIENT * value_loss - ENTROPY_COEFFICIENT * entropy
    return total, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approximate_kl": jnp.mean((ratio - 1.0) - log_ratio),
        "clip_fraction": jnp.mean(
            (jnp.abs(ratio - 1.0) > CLIP_EPSILON).astype(jnp.float32)
        ),
    }


def train_j1_ppo(
    env: GoalDecisionEnv,
    params: Any,
    apply_fn: ApplyFn,
    key: jax.Array,
    *,
    updates: int,
    decisions_per_update: int,
    batch: int,
    lr: float,
) -> tuple[Any, list[dict[str, float]]]:
    """Train a goal-decision policy with semi-MDP PPO.

    Each update collects ``decisions_per_update`` decisions in ``batch`` fresh
    lanes, computes :func:`decision_advantages` with ``gamma = env.gamma``,
    standardizes the advantages over the whole rollout, and runs
    ``UPDATE_EPOCHS`` epochs of ``NUM_MINIBATCHES`` minibatches of whole lane
    sequences through optax Adam (module formulas). One update is one compiled
    call; it is compiled once per call of this function.

    Args:
        env: A :class:`GoalDecisionEnv`.
        params: Initial policy parameters, a PyTree of float32 arrays.
        apply_fn: ``apply_fn(params, obs) -> (logits float32[B, K],
            value float32[B])``.
        key: A JAX PRNG key.
        updates: Number of PPO updates, a positive int.
        decisions_per_update: T, decisions per lane per update, a positive int.
        batch: B, lanes per update, a positive multiple of ``NUM_MINIBATCHES``.
        lr: Adam learning rate, finite and positive.

    Returns:
        ``(params, metrics)``. ``metrics`` holds one dict per update with
        ``update`` (1-based) and the floats ``mean_reward`` (per decision, over
        the rollout collected before that update's gradient steps),
        ``mean_elapsed_ticks``, ``episodes_completed``, ``policy_loss``,
        ``value_loss``, ``entropy``, ``approximate_kl`` and ``clip_fraction``.

    Raises:
        TypeError: If an integer or ``lr`` argument has the wrong type, or
            ``env`` is not a GoalDecisionEnv.
        ValueError: If a count is not positive, ``batch`` is not a multiple of
            ``NUM_MINIBATCHES``, or ``lr`` is not finite and positive.
    """

    count = _positive_int(updates, "updates")
    steps = _positive_int(decisions_per_update, "decisions_per_update")
    lanes = _positive_int(batch, "batch")
    if lanes % NUM_MINIBATCHES:
        raise ValueError(
            f"batch must be a multiple of NUM_MINIBATCHES={NUM_MINIBATCHES}, got {lanes}"
        )
    if isinstance(lr, bool) or not isinstance(lr, numbers.Real):
        raise TypeError(f"lr must be a real number, got {type(lr).__name__}")
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"lr must be finite and positive, got {lr}")
    if not isinstance(env, GoalDecisionEnv):
        raise TypeError("env must implement GoalDecisionEnv (num_skills, gamma, reset, step)")

    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRADIENT_NORM),
        optax.adam(float(lr), eps=1.0e-5),
    )

    def loss(current: Any, minibatch: tuple[Any, ...]):
        return _ppo_loss(current, apply_fn, minibatch)

    def update(current: Any, optimizer_state: Any, update_key: jax.Array):
        rollout_key, epoch_key = jax.random.split(update_key)
        rollout = collect_decisions(env, current, apply_fn, rollout_key, steps, lanes)
        advantages, returns = decision_advantages(
            rollout.reward,
            rollout.value,
            rollout.done,
            rollout.elapsed_ticks,
            rollout.bootstrap_value,
            gamma=env.gamma,
        )
        advantages = (advantages - jnp.mean(advantages)) / (
            jnp.std(advantages) + 1.0e-8
        )
        data = (
            rollout.obs,
            rollout.action,
            rollout.logp,
            rollout.value,
            advantages,
            returns,
        )

        def epoch(carry, permutation_key):
            groups = jax.random.permutation(permutation_key, lanes).reshape(
                NUM_MINIBATCHES, lanes // NUM_MINIBATCHES
            )

            def minibatch_step(inner, lane_index):
                minibatch_params, minibatch_state = inner
                minibatch = jax.tree.map(lambda leaf: leaf[:, lane_index], data)
                (_, stats), gradients = jax.value_and_grad(loss, has_aux=True)(
                    minibatch_params, minibatch
                )
                step_updates, minibatch_state = optimizer.update(
                    gradients, minibatch_state, minibatch_params
                )
                minibatch_params = optax.apply_updates(minibatch_params, step_updates)
                return (minibatch_params, minibatch_state), stats

            return jax.lax.scan(minibatch_step, carry, groups)

        (current, optimizer_state), stats = jax.lax.scan(
            epoch,
            (current, optimizer_state),
            jax.random.split(epoch_key, UPDATE_EPOCHS),
        )
        summary = {name: jnp.mean(values) for name, values in stats.items()}
        summary["mean_reward"] = jnp.mean(rollout.reward)
        summary["mean_elapsed_ticks"] = jnp.mean(rollout.elapsed_ticks.astype(jnp.float32))
        summary["episodes_completed"] = jnp.sum(rollout.done.astype(jnp.float32))
        return current, optimizer_state, summary

    compiled_update = jax.jit(update)
    optimizer_state = optimizer.init(params)
    metrics: list[dict[str, float]] = []
    for index in range(count):
        key, update_key = jax.random.split(key)
        params, optimizer_state, summary = compiled_update(
            params, optimizer_state, update_key
        )
        row: dict[str, float] = {"update": index + 1}
        row.update({name: float(value) for name, value in summary.items()})
        metrics.append(row)
    return params, metrics


class SyntheticState(NamedTuple):
    """State of :class:`SyntheticGoalEnv` for B lanes.

    Shapes:
        correct_skill: int32[B], the hidden skill that pays +1.
        priority: float32[B, MAX_GOALS], executive priorities; slot g names skill g.
        latent: float32[B, BELIEF_LATENT_WIDTH], distractor noise in the belief.
        current_skill: int32[B], the skill in force; 0 before the first decision.
        decisions: int32[B], decisions taken in this episode.
    """

    correct_skill: jax.Array
    priority: jax.Array
    latent: jax.Array
    current_skill: jax.Array
    decisions: jax.Array


class SyntheticGoalEnv:
    """Toy semi-MDP in which J1 must read E_t to pick the paying skill. Pure JAX.

    At reset each lane draws a hidden correct skill c uniformly from [0, 3).
    The executive names skill g in goal slot g for g < 3 (slot 3 is empty),
    with priority U(0.6, 1.0) for c and U(0.0, 0.4) for the others, so c is
    the highest-priority goal and is recoverable only from E_t. The belief
    carries noise only: every feature is unknown and the latent is N(0, 1),
    redrawn each decision. Every skill is legal, and skill 0 has no idle role.

    A decision lasts d ~ U{1, ..., max_ticks} ticks and pays +1 if the chosen
    skill is c, else -0.2, on its last tick, so ``step`` returns
    ``gamma ** (d - 1)`` times that. An episode is ``episode_decisions``
    decisions with the same c. With these constants a uniform policy earns
    about 0.19 per decision and the optimal policy about 0.97.

    Attributes:
        num_skills: 3.
        gamma: Per-tick discount, ``GAMMA``.
        max_ticks: Longest decision in ticks, 8.
        episode_decisions: Decisions per episode, 4.
    """

    num_skills: int = 3
    gamma: float = GAMMA
    max_ticks: int = 8
    episode_decisions: int = 4
    correct_payoff: float = 1.0
    wrong_payoff: float = -0.2

    def reset(self, keys: jax.Array) -> tuple[SyntheticState, dict[str, Any]]:
        """Start one episode per lane; see :meth:`GoalDecisionEnv.reset`.

        Args:
            keys: B PRNG keys, one per lane.

        Returns:
            ``(state, obs)`` with the observation dict of :class:`GoalDecisionEnv`.
        """

        state = jax.vmap(self._reset_lane)(keys)
        return state, self._observe(state)

    def step(
        self, state: SyntheticState, skill_index: jax.Array, keys: jax.Array
    ) -> tuple[SyntheticState, dict[str, Any], jax.Array, jax.Array, jax.Array]:
        """Run one decision of d ticks; see :meth:`GoalDecisionEnv.step`.

        Args:
            state: SyntheticState for B lanes.
            skill_index: int32[B], the chosen skill in [0, 3).
            keys: B PRNG keys, one per lane.

        Returns:
            ``(state, obs, reward float32[B], elapsed_ticks int32[B], done bool[B])``.
        """

        skill = jnp.asarray(skill_index, dtype=jnp.int32)
        elapsed, latent = jax.vmap(self._draw_decision)(keys)
        payoff = jnp.where(
            skill == state.correct_skill,
            jnp.float32(self.correct_payoff),
            jnp.float32(self.wrong_payoff),
        )
        reward = jnp.float32(self.gamma) ** (elapsed - 1).astype(jnp.float32) * payoff
        decisions = state.decisions + 1
        next_state = state._replace(
            latent=latent, current_skill=skill, decisions=decisions
        )
        done = decisions >= self.episode_decisions
        return next_state, self._observe(next_state), reward, elapsed, done

    def _reset_lane(self, key: jax.Array) -> SyntheticState:
        """Draw one lane's hidden skill, priorities and latent (unbatched)."""

        correct_key, priority_key, latent_key = jax.random.split(key, 3)
        correct = jax.random.randint(correct_key, (), 0, self.num_skills, dtype=jnp.int32)
        noise = jax.random.uniform(priority_key, (self.num_skills,), dtype=jnp.float32)
        ranked = jnp.where(
            jnp.arange(self.num_skills) == correct, 0.6 + 0.4 * noise, 0.4 * noise
        )
        priority = jnp.zeros((MAX_GOALS,), jnp.float32).at[: self.num_skills].set(ranked)
        return SyntheticState(
            correct_skill=correct,
            priority=priority,
            latent=jax.random.normal(latent_key, (BELIEF_LATENT_WIDTH,), jnp.float32),
            current_skill=jnp.zeros((), jnp.int32),
            decisions=jnp.zeros((), jnp.int32),
        )

    def _draw_decision(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Draw one lane's decision length int32[] and new latent float32[L]."""

        tick_key, latent_key = jax.random.split(key)
        ticks = jax.random.randint(
            tick_key, (), 1, self.max_ticks + 1, dtype=jnp.int32
        )
        latent = jax.random.normal(latent_key, (BELIEF_LATENT_WIDTH,), jnp.float32)
        return ticks, latent

    def _observe(self, state: SyntheticState) -> dict[str, Any]:
        """Build the observation dict for B lanes from ``state``."""

        lanes = state.correct_skill.shape[0]
        slots = jnp.arange(MAX_GOALS, dtype=jnp.int32)
        occupied = slots < self.num_skills
        budget = ComputeBudget()
        version = jnp.ones((lanes,), jnp.int32)
        executive = ExecutiveArrays(
            skill_index=jnp.broadcast_to(
                jnp.where(occupied, slots, -1), (lanes, MAX_GOALS)
            ),
            priority=state.priority,
            params=jnp.zeros((lanes, MAX_GOALS, PARAM_WIDTH), jnp.float32),
            goal_valid=jnp.broadcast_to(occupied, (lanes, MAX_GOALS)),
            budget=jnp.broadcast_to(
                jnp.asarray(
                    [
                        budget.jev_latency_ms,
                        budget.search_expansions,
                        budget.executive_calls_per_minute,
                    ],
                    jnp.float32,
                ),
                (lanes, _BUDGET_WIDTH),
            ),
            version=version,
        )
        # Before the first decision the goal in force is the idle goal of
        # contracts.idle_goal: skill 0, executive version 0, no slot.
        started = state.decisions > 0
        current_goal = GoalPayload(
            skill_index=state.current_skill,
            params=jnp.zeros((lanes, PARAM_WIDTH), jnp.float32),
            executive_version=jnp.where(started, version, 0),
            goal_slot=jnp.where(started, state.current_skill, -1),
        )
        belief = SharedBeliefState(
            features=jnp.zeros((lanes, BELIEF_FEATURE_COUNT), jnp.float32),
            feature_known=jnp.zeros((lanes, BELIEF_FEATURE_COUNT), jnp.bool_),
            latent=state.latent,
        )
        return {
            "belief": belief,
            "executive": executive,
            "current_goal": current_goal,
            "mask": jnp.ones((lanes, self.num_skills), jnp.bool_),
        }


# TODO(J1 network): switch to init_j1, apply_j1 and J1Params from
# experimental.neural_agent.jev.network. network.py did not exist when this
# trainer was started (2026-10-16), so a minimal stand-in lives here. Its input
# mirrors j1_inputs in network.py as read on 2026-10-16. apply_j1 as read then
# returns (skill_logits, replace_logit, params_head) and has no value head, so
# the switch also needs a critic: train_j1_ppo needs
# apply_fn(params, obs) -> (logits, value).
# 2026-10-17: jev/j1_actor_critic.py now provides that apply_fn, and --env sim
# uses it. The synthetic path keeps this stand-in so that its tests stay as
# they are.


def policy_input_width(num_skills: int) -> int:
    """Return D, the width of :func:`policy_input`: 2F + L + G(K + P + 2) + 4 + K.

    With F = 10, L = 64, G = P = 4 this is 2F + L + 5K + 28, the width of
    ``input_width`` in ``jev/network.py``; it is 127 for K = 3.

    Args:
        num_skills: Catalog size K, a positive int.

    Returns:
        The input width D.
    """

    skills = _positive_int(num_skills, "num_skills")
    slot_block = MAX_GOALS * (skills + 1 + PARAM_WIDTH + 1)
    return 2 * BELIEF_FEATURE_COUNT + BELIEF_LATENT_WIDTH + slot_block + _BUDGET_WIDTH + 1 + skills


def policy_input(obs: dict[str, Any]) -> jax.Array:
    """Build the J1 input matrix from a decision observation.

    The layout mirrors ``j1_inputs`` in ``jev/network.py``::

        [where(known, features, 0) | known | latent |
         one-hot skill per goal slot (an empty slot, -1, is zeros) | priority |
         params per slot | goal_valid | log1p(budget) | log1p(version) |
         one-hot current skill]

    The mask is not an input: the trainer applies it to the logits.

    Args:
        obs: The observation dict of :class:`GoalDecisionEnv` with N lanes and
            K = ``obs["mask"].shape[-1]`` skills.

    Returns:
        float32[N, D], with D = ``policy_input_width(K)``.
    """

    belief, executive = obs["belief"], obs["executive"]
    lanes, skills = obs["mask"].shape
    known = jnp.asarray(belief.feature_known, dtype=jnp.bool_)
    pieces = (
        jnp.where(known, belief.features, 0.0),
        known,
        belief.latent,
        jax.nn.one_hot(executive.skill_index, skills).reshape(lanes, MAX_GOALS * skills),
        executive.priority,
        executive.params.reshape(lanes, MAX_GOALS * PARAM_WIDTH),
        executive.goal_valid,
        jnp.log1p(jnp.asarray(executive.budget, dtype=jnp.float32)),
        jnp.log1p(jnp.asarray(executive.version, dtype=jnp.float32))[:, None],
        jax.nn.one_hot(obs["current_goal"].skill_index, skills),
    )
    return jnp.concatenate(
        [jnp.asarray(piece, dtype=jnp.float32) for piece in pieces], axis=-1
    )


class MlpPolicyParams(NamedTuple):
    """Local stand-in for the J1 network: two ReLU layers, skill and value heads.

    Shapes, with D = ``policy_input_width(K)``, H hidden units and K skills:
        w1: float32[D, H]. b1: float32[H].
        w2: float32[H, H]. b2: float32[H].
        w_skill: float32[H, K]. b_skill: float32[K].
        w_value: float32[H]. b_value: float32[] (a scalar).
    """

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w_skill: jax.Array
    b_skill: jax.Array
    w_value: jax.Array
    b_value: jax.Array


def init_mlp_policy(key: jax.Array, num_skills: int, hidden: int = 64) -> MlpPolicyParams:
    """Initialize :class:`MlpPolicyParams`.

    The hidden layers use He-normal weights. The skill head is scaled by 0.01
    so that the first policy is close to uniform. Biases start at zero.

    Args:
        key: A JAX PRNG key.
        num_skills: Catalog size K, a positive int.
        hidden: H, the width of both hidden layers, a positive int.

    Returns:
        MlpPolicyParams with the shapes listed on the class.
    """

    skills = _positive_int(num_skills, "num_skills")
    width_h = _positive_int(hidden, "hidden")
    width = policy_input_width(skills)
    key_w1, key_w2, key_skill, key_value = jax.random.split(key, 4)
    f32 = jnp.float32
    return MlpPolicyParams(
        w1=jax.random.normal(key_w1, (width, width_h), f32) * math.sqrt(2.0 / width),
        b1=jnp.zeros((width_h,), f32),
        w2=jax.random.normal(key_w2, (width_h, width_h), f32) * math.sqrt(2.0 / width_h),
        b2=jnp.zeros((width_h,), f32),
        w_skill=jax.random.normal(key_skill, (width_h, skills), f32) * 0.01,
        b_skill=jnp.zeros((skills,), f32),
        w_value=jax.random.normal(key_value, (width_h,), f32) * math.sqrt(1.0 / width_h),
        b_value=jnp.zeros((), f32),
    )


def apply_mlp_policy(
    params: MlpPolicyParams, obs: dict[str, Any]
) -> tuple[jax.Array, jax.Array]:
    """Return unmasked ``(logits float32[N, K], value float32[N])`` for N lanes.

    Args:
        params: MlpPolicyParams.
        obs: The observation dict of :class:`GoalDecisionEnv`.

    Returns:
        Skill logits and value estimates. :func:`collect_decisions` and the
        PPO loss apply ``obs["mask"]`` to the logits.
    """

    hidden_one = jax.nn.relu(policy_input(obs) @ params.w1 + params.b1)
    hidden_two = jax.nn.relu(hidden_one @ params.w2 + params.b2)
    logits = hidden_two @ params.w_skill + params.b_skill
    value = hidden_two @ params.w_value + params.b_value
    return logits, value


def main(argv: Sequence[str] | None = None) -> int:
    """Train with semi-MDP PPO and print metrics.

    ``--env synthetic`` (the default) trains the local stand-in on
    :class:`SyntheticGoalEnv`. ``--env sim`` trains J1 with a separate critic
    on ``SimGoalDecisionEnv`` over ``combat_open_flat``, prints the simulated
    lane-ticks per second (compilation included), and writes the J1Params to
    ``--out``.

    Args:
        argv: Command-line arguments without the program name, or None for
            ``sys.argv[1:]``.

    Returns:
        Process exit status 0.
    """

    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.jev.train_j1_ppo",
        description=(
            "Semi-MDP PPO over J1 goal decisions on the synthetic environment, or "
            "with --env sim on the combat_open_flat arena (CPU, no game server). "
            "On the synthetic environment a uniform policy earns about 0.19 reward "
            "per decision and the optimal one about 0.97."
        ),
    )
    parser.add_argument("--updates", type=int, default=60, help="PPO updates (default 60)")
    parser.add_argument(
        "--decisions-per-update",
        type=int,
        default=16,
        help="decisions per lane per update (default 16)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help=f"lanes, a multiple of {NUM_MINIBATCHES} (default 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE, help=f"Adam learning rate (default {LEARNING_RATE})"
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed (default 0)")
    parser.add_argument(
        "--env",
        choices=("synthetic", "sim"),
        default="synthetic",
        help="synthetic toy (default), or SimGoalDecisionEnv on combat_open_flat",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=256,
        help="--env sim: arena episode limit in ticks (default 256, the composition runner's)",
    )
    parser.add_argument(
        "--out", default=None, help="--env sim: J1Params .npz to write (required there)"
    )
    parser.add_argument(
        "--init-j1",
        default=None,
        help="--env sim: J1Params .npz to start from (for example the behaviour-cloned parameters) instead of init_j1",
    )
    args = parser.parse_args(argv)
    if args.env == "sim" and args.out is None:
        parser.error("--out is required with --env sim")
    if args.env == "synthetic" and args.out is not None:
        parser.error("--out applies to --env sim only")
    if args.out is not None and Path(args.out).suffix != ".npz":
        parser.error(f"--out must end in .npz, got {args.out}")
    if args.init_j1 is not None and args.env != "sim":
        parser.error("--init-j1 applies to --env sim only")

    init_key, train_key = jax.random.split(jax.random.key(args.seed))
    if args.env == "sim":
        from experimental.neural_agent.jev.j1_actor_critic import (
            apply_j1_actor_critic,
            init_j1_actor_critic,
            save_j1_params,
        )
        from experimental.neural_agent.jev.sim_goal_env import build_sim_goal_env

        env: GoalDecisionEnv = build_sim_goal_env(args.batch, args.ticks, seed=args.seed)
        params: Any = init_j1_actor_critic(init_key, env.num_skills)
        if args.init_j1 is not None:
            from experimental.neural_agent.jev.j1_actor_critic import with_actor_from_file

            params = with_actor_from_file(params, args.init_j1)
        apply_fn: ApplyFn = apply_j1_actor_critic
    else:
        env = SyntheticGoalEnv()
        params = init_mlp_policy(init_key, env.num_skills)
        apply_fn = apply_mlp_policy
    started = time.perf_counter()
    params, metrics = train_j1_ppo(
        env,
        params,
        apply_fn,
        train_key,
        updates=args.updates,
        decisions_per_update=args.decisions_per_update,
        batch=args.batch,
        lr=args.lr,
    )
    seconds = time.perf_counter() - started
    for row in metrics:
        line = (
            f"update {int(row['update']):4d}  mean_reward {row['mean_reward']:+.3f}  "
            f"entropy {row['entropy']:.3f}  approx_kl {row['approximate_kl']:.4f}  "
            f"clip_fraction {row['clip_fraction']:.3f}  value_loss {row['value_loss']:.4f}"
        )
        if args.env == "sim":
            line += (
                f"  mean_elapsed_ticks {row['mean_elapsed_ticks']:.2f}  "
                f"episodes_completed {int(row['episodes_completed'])}"
            )
        print(line)
    print(f"final mean_reward {metrics[-1]['mean_reward']:+.3f} after {len(metrics)} updates")
    if args.env == "sim":
        lane_ticks = sum(row["mean_elapsed_ticks"] for row in metrics) * (
            args.decisions_per_update * args.batch
        )
        print(
            f"sim: {lane_ticks:.0f} lane-ticks in {seconds:.1f} s including compilation, "
            f"{lane_ticks / seconds:.1f} lane-ticks/s"
        )
        print(f"wrote J1Params to {save_j1_params(args.out, params)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
