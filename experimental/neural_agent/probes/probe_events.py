"""Measure DecisionEvents, episode ends, reward and health through the ADK hierarchy.

Composes ``adk.architecture.hierarchy.HierarchicalAgentCore`` from a pass-through
belief, a constant manager and one of two actors (masked neutral, or the
``agent_skills/policies/pursuit_contact`` candidate), then collects with
``adk.runtime.hierarchical.compile_hierarchical_collector``.

Run from the repository root::

    PYTHONPATH=.:HytaleRL/hytalegym JAX_PLATFORMS=cpu \
        .venv/bin/python -m experimental.neural_agent.probes.probe_events

Writes ``probes/out/events.json``.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from adk.api import AgentKit
from adk.architecture.components import (
    ActorDecision,
    BeliefUpdate,
    GoalDecision,
    HierarchicalCarry,
)
from adk.architecture.decision_context import DecisionEvent
from adk.architecture.hierarchy import (
    GoalControl,
    HierarchicalAgentCore,
    HierarchicalCoreState,
    ManagedGoal,
    initialize_hierarchical_state,
)
from adk.core.spec import AgentSpec
from adk.runtime.hierarchical import (
    HierarchicalLoopTransition,
    compile_hierarchical_collector,
)
from hytalegym.jax.combat.observation import SELF_FLOAT_FEATURES
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    neutral_arsenal_policy_action_factors,
)

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "out"
PURSUIT_LOADER = REPO / "agent_skills" / "policies" / "pursuit_contact" / "policy.py"

HEALTH_INDEX = SELF_FLOAT_FEATURES.index("health_fraction")
HEAD_SIZES = tuple(int(size) for size in ARSENAL_POLICY_ACTION_HEAD_SIZES)
HEAD_OFFSETS = tuple(int(offset) for offset in np.cumsum((0,) + HEAD_SIZES[:-1]))
EVENT_COUNT = len(DecisionEvent)
MANAGER_CADENCE_SECONDS = 1.0

# The only name AgentKit's on-disk registry resolves (adk/agents/combat/spec.py),
# plus a probe-registered spec bound to the other built-in scene.
BASELINE = "combat/baseline"
OPEN_FLAT = "probe/open_flat_control"


def build_handle(name: str, batch: int, episode_limit: int | None):
    """``AgentKit().build(name, batch=...)``, optionally with an episode cap."""

    kit = AgentKit()
    if name == OPEN_FLAT:
        kit.register(
            AgentSpec(name=OPEN_FLAT, scene="combat/open_flat_control")
        )
    handle = kit.build(name, batch=batch)
    if episode_limit is not None:
        handle = handle.with_episode_limit(episode_limit)
    return handle


def goal_control(batch: int) -> GoalControl:
    return GoalControl(
        jnp.zeros((batch,), jnp.float32),
        jnp.full((batch,), jnp.inf, jnp.float32),
        jnp.zeros((batch, EVENT_COUNT), jnp.bool_),
    )


def passthrough_belief(state, actor_input, previous_action, previous_goal, delta, key):
    batch = actor_input.observation.shape[0]
    zeros = jnp.zeros((batch, 1), jnp.float32)
    return BeliefUpdate(
        state=state,
        belief=actor_input.observation,
        prior=zeros,
        posterior=zeros,
        uncertainty=zeros,
        diagnostics={},
    )


def constant_manager(batch: int) -> Callable[..., GoalDecision]:
    control = goal_control(batch)

    def manager(state, belief, current_goal, actor_input, key, *,
                decision_context=None, goal_runtime=None):
        return GoalDecision(
            state=state,
            goal=ManagedGoal(payload=current_goal, control=control),
            terminate=jnp.zeros((batch,), jnp.bool_),
            diagnostics={},
        )

    return manager


def neutral_within_mask(action_mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per head: the neutral bin if the mask allows it, else the first legal bin."""

    mask = jnp.asarray(action_mask, jnp.bool_)
    if mask.shape[-1] != sum(HEAD_SIZES):
        raise ValueError(f"mask width {mask.shape[-1]} != {sum(HEAD_SIZES)}")
    neutral = neutral_arsenal_policy_action_factors(mask.shape[0])
    chosen, fallback = [], []
    for head, (offset, size) in enumerate(zip(HEAD_OFFSETS, HEAD_SIZES)):
        legal = mask[:, offset:offset + size]
        wanted = neutral[:, head]
        allowed = jnp.take_along_axis(legal, wanted[:, None], axis=1)[:, 0]
        first_legal = jnp.argmax(legal, axis=1).astype(jnp.int32)
        chosen.append(jnp.where(allowed, wanted, first_legal))
        fallback.append(~allowed)
    return jnp.stack(chosen, axis=1), jnp.stack(fallback, axis=1)


def neutral_actor(state, belief, goal, actor_input, key):
    factors, fallback = neutral_within_mask(actor_input.action_mask)
    return ActorDecision(
        state=state,
        action_factors=factors,
        logits=None,
        behavior_log_probability=None,
        value=None,
        terminate_goal=None,
        diagnostics={"neutral_fallback": fallback},
    )


def load_pursuit(batch: int) -> tuple[Callable, jax.Array]:
    spec = importlib.util.spec_from_file_location("pursuit_contact_policy", PURSUIT_LOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load(batch)


def pursuit_actor(batch: int) -> tuple[Callable[..., ActorDecision], jax.Array]:
    policy, carry = load_pursuit(batch)

    def actor(state, belief, goal, actor_input, key):
        next_state, factors = policy(state, actor_input, key)
        return ActorDecision(
            state=next_state,
            action_factors=factors,
            logits=None,
            behavior_log_probability=None,
            value=None,
            terminate_goal=None,
            diagnostics={"neutral_fallback": jnp.zeros((batch, len(HEAD_SIZES)), jnp.bool_)},
        )

    return actor, carry


def compose(batch: int, actor_name: str) -> tuple[HierarchicalAgentCore, HierarchicalCoreState]:
    zeros = jnp.zeros((batch, 1), jnp.float32)
    if actor_name == "neutral":
        actor, actor_state = neutral_actor, zeros
    elif actor_name == "pursuit":
        actor, actor_state = pursuit_actor(batch)
    else:
        raise ValueError(actor_name)
    core = HierarchicalAgentCore(
        belief=passthrough_belief,
        manager=constant_manager(batch),
        actor=actor,
        manager_cadence_seconds=MANAGER_CADENCE_SECONDS,
    )
    carry = HierarchicalCarry(
        belief_state=zeros,
        manager_state=zeros,
        actor_state=actor_state,
        world_model_state=zeros,
        goal=ManagedGoal(payload=jnp.zeros((batch,), jnp.int32), control=goal_control(batch)),
        previous_action=neutral_arsenal_policy_action_factors(batch),
        goal_age=jnp.zeros((batch,), jnp.float32),
        action_age=jnp.zeros((batch,), jnp.float32),
    )
    return core, initialize_hierarchical_state(carry, manager_diagnostics={})


def record(transition: HierarchicalLoopTransition) -> dict[str, Any]:
    context = transition.next_decision_context
    before = transition.actor_input.legal_observation.base.self_f32
    after = transition.next_actor_input.legal_observation.base.self_f32
    return {
        "event_values": context.events.values,
        "event_known": context.events.known,
        "reward": transition.reward,
        "done": transition.done,
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "health_before": before[:, HEALTH_INDEX],
        "health_after": after[:, HEALTH_INDEX],
        "factors": transition.action_factors,
        "delta_seconds": context.timing.delta_seconds,
        "delta_known": context.timing.delta_seconds_known,
        "manager_due": transition.hierarchy_step.manager_due,
        "neutral_fallback": transition.hierarchy_step.actor_decision.diagnostics["neutral_fallback"],
    }


def timed_rollout(built, core, initial_state, steps: int, seed: int = 0):
    """Return (records, lower_s, compile_s, run_s) for one compiled rollout."""

    collect = compile_hierarchical_collector(
        built, core, record, steps, initial_state=initial_state
    )
    key = jax.random.PRNGKey(seed)
    start = time.perf_counter()
    lowered = collect.lower(key)
    lowered_at = time.perf_counter()
    compiled = lowered.compile()
    compiled_at = time.perf_counter()
    _, records = jax.block_until_ready(compiled(key))
    finished = time.perf_counter()
    return (
        jax.device_get(records),
        lowered_at - start,
        compiled_at - lowered_at,
        finished - compiled_at,
        compiled,
        key,
    )


def summarize(records: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(records["event_values"], bool)
    known = np.asarray(records["event_known"], bool)
    events = {
        event.name: {
            "known_steps": int(known[..., int(event)].sum()),
            "fired_steps": int((known[..., int(event)] & values[..., int(event)]).sum()),
        }
        for event in DecisionEvent
    }
    reward = np.asarray(records["reward"], np.float64)
    done = np.asarray(records["done"], bool)
    health_before = np.asarray(records["health_before"], np.float64)
    health_after = np.asarray(records["health_after"], np.float64)
    factors = np.asarray(records["factors"])
    delta = np.asarray(records["delta_seconds"], np.float64)
    delta_known = np.asarray(records["delta_known"], bool)
    split = records["terminated"] is not None
    damage = events["DAMAGE_RECEIVED"]
    return {
        "lane_steps": int(done.size),
        "events": events,
        "episodes_ended": int(done.sum()),
        "terminated": int(np.asarray(records["terminated"], bool).sum()) if split else None,
        "truncated": int(np.asarray(records["truncated"], bool).sum()) if split else None,
        "boundary_split_available": split,
        "first_done_step_per_lane": [
            int(np.argmax(done[:, lane])) if done[:, lane].any() else None
            for lane in range(done.shape[1])
        ],
        "mean_reward_per_lane_step": float(reward.mean()),
        "total_reward": float(reward.sum()),
        "nonzero_reward_steps": int((reward != 0).sum()),
        "damage_received_fires": damage["fired_steps"] > 0,
        "damage_received_steps": damage["fired_steps"],
        "health_fraction_decrease_steps": int((health_after < health_before).sum()),
        "health_fraction_min": float(health_after.min()),
        "health_fraction_initial_mean": float(health_before[0].mean()),
        "delta_seconds_known_steps": int(delta_known.sum()),
        "delta_seconds_values": sorted({round(float(x), 6) for x in delta[delta_known]}),
        "manager_due_steps": int(np.asarray(records["manager_due"], bool).sum()),
        "neutral_fallback_steps_per_head": {
            name: int(count)
            for name, count in zip(
                ARSENAL_POLICY_ACTION_HEAD_NAMES,
                np.asarray(records["neutral_fallback"], bool).sum(axis=(0, 1)),
            )
        },
        "factor_counts_per_head": {
            name: {
                str(int(value)): int(count)
                for value, count in zip(*np.unique(factors[..., head], return_counts=True))
            }
            for head, name in enumerate(ARSENAL_POLICY_ACTION_HEAD_NAMES)
        },
    }


def main(batch: int = 8, steps: int = 128, episode_limit: int = 128) -> dict[str, Any]:
    kit = AgentKit()
    result: dict[str, Any] = {
        "batch": batch,
        "steps": steps,
        "episode_limit_ticks": episode_limit,
        "manager_cadence_seconds": MANAGER_CADENCE_SECONDS,
        "jax_backend": jax.default_backend(),
        "registered_agent_names": kit.list_agents(),
        "registered_scene_names": kit.list_scenes(),
        "event_order": [event.name for event in DecisionEvent],
        "action_head_names": list(ARSENAL_POLICY_ACTION_HEAD_NAMES),
        "action_head_sizes": list(HEAD_SIZES),
        "health_fraction_source": (
            "actor_input.legal_observation.base.self_f32"
            f"[:, SELF_FLOAT_FEATURES.index('health_fraction')={HEALTH_INDEX}]"
        ),
        "scenes": {},
    }
    for name in (BASELINE, OPEN_FLAT):
        start = time.perf_counter()
        handle = build_handle(name, batch, episode_limit)
        scene_entry: dict[str, Any] = {
            "spec_scene": handle.spec.scene,
            "spec_loadout": handle.spec.loadout,
            "build_seconds": time.perf_counter() - start,
            "actors": {},
        }
        for actor_name in ("neutral", "pursuit"):
            core, initial_state = compose(batch, actor_name)
            records, lower_s, compile_s, run_s, _, _ = timed_rollout(
                handle.built, core, initial_state, steps
            )
            summary = summarize(records)
            summary["lower_seconds"] = lower_s
            summary["compile_seconds"] = compile_s
            summary["run_seconds"] = run_s
            scene_entry["actors"][actor_name] = summary
            print(
                f"{name} {actor_name}: ended={summary['episodes_ended']} "
                f"reward={summary['mean_reward_per_lane_step']:.4f} "
                f"damage={summary['damage_received_steps']} "
                f"compile={compile_s:.1f}s run={run_s:.2f}s",
                flush=True,
            )
        result["scenes"][name] = scene_entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "events.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
