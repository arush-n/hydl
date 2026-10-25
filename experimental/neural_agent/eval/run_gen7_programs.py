"""The decision layer's real programs on the gen-7 session: a two-step program (walk_to, then look_at) per lane.

The decision layer is ``jev/programs.py`` ``ProgramExecutor`` (unchanged), driven through
:class:`~experimental.neural_agent.jev.gen7_decider.Gen7ProgramDecider` by
:func:`~experimental.neural_agent.session_gen7.run_gen7_session`. The world, the trained bodies, the combiner and the state
tick are the architecture test's (``eval/run_architecture.py``). The program of every lane is: walk_to the other entity
until the belief's target distance is below the radius, then look_at it for a fixed hold. A fixed executive gives one
walk_to goal per lane at tick 0.

The report's end-state numbers come from the skill layer's own success kernels (``WalkToTasks.success``,
``LookAtTasks.success``) on the final state; the decision layer never reads them.

CLI (from the repo root; CPU)::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu nice -n 10 \\
        .venv/bin/python experimental/neural_agent/eval/run_gen7_programs.py --bodies DIR --out REPORT.json \\
        [--lanes 16] [--ticks 300] [--seed 77001] [--radius 1.0] [--budget-ms 20]
"""

# ruff: noqa: E402 (sys.path is set before the repo imports)
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:  # run as a script
    sys.path.insert(0, str(ROOT))

from experimental.neural_agent.belief.core import SharedBelief, init_belief_params
from experimental.neural_agent.contracts import (
    ComputeBudget,
    ExecutiveState,
    GoalSpec,
    feature_index,
)
from experimental.neural_agent.eval import run_architecture as arch
from experimental.neural_agent.executive.goal_language import Predicate
from experimental.neural_agent.jev.gen7_decider import (
    LOOK_EYE_HEIGHT,
    Gen7DecisionFacts,
    Gen7ProgramDecider,
)
from experimental.neural_agent.jev.programs import Program, ProgramStep
from experimental.neural_agent.requests import rejection_code
from experimental.neural_agent.session_gen7 import Gen7SessionConfig, run_gen7_session
from experimental.neural_agent.shared_state import SharedState
from experimental.neural_agent.shared_state.skill_writes import StateTickBinding
from experimental.neural_agent.skill_agent.neural_skills.look_at.task import (
    LookAtGoals,
    LookAtTasks,
)
from experimental.neural_agent.skill_agent.neural_skills.look_at.task import (
    AGENT_EYE_HEIGHT,
)
from experimental.neural_agent.skill_agent.neural_skills.spec import Pose
from experimental.neural_agent.skill_agent.neural_skills.walk_to.task import (
    WalkToGoals,
    WalkToTasks,
)
from experimental.neural_agent.skill_agent.runtime import state_tick
from experimental.neural_agent.skill_agent.runtime.combiner import AgentContext
from experimental.neural_agent.skill_agent.runtime.scratchpad import ANY_WORLD
from experimental.neural_agent.skill_agent.training import blocks, distill_v3
from experimental.neural_agent.skill_library.idle import IdleSkill
from hytalegym.jax.combat.types import AGENT_ENTITY

WALK_PACE: float = 1.0
"""The walk_to ``pace`` scalar, the gait code of a walk (the training's gaits are ints; the report names the choice)."""
LOOK_TOLERANCE_DEGREES: float = 3.0
"""look_at's tolerance, the training's value (``distill.sample_episodes`` samples 3.0)."""
LOOK_HOLD_SECONDS: float = 2.0
"""The look_at step's hold: an exit by timer, since the trained bodies never report done (``ModuleBody``)."""


def walk_then_look(radius: float) -> Program:
    """walk_to until the belief's target distance is below ``radius``, then look_at for ``LOOK_HOLD_SECONDS``."""

    return Program(
        steps=(
            ProgramStep(
                "walk_to",
                params=(radius, WALK_PACE),
                exit=Predicate("distance_below", radius),
            ),
            ProgramStep(
                "look_at",
                params=(LOOK_TOLERANCE_DEGREES,),
                exit="timer",
                hold_seconds=LOOK_HOLD_SECONDS,
            ),
        )
    )


class FixedExecutive:
    """One executive state at tick 0: a walk_to goal for every lane (addressee all)."""

    def __init__(self, radius: float) -> None:
        self.state = ExecutiveState(
            version=1,
            issued_tick=0,
            goals=(
                GoalSpec(
                    skill_id="walk_to",
                    params=(radius, WALK_PACE),
                    priority=1.0,
                    constraints=("walk to the other entity",),
                ),
            ),
            budget=ComputeBudget(),
            facts=("the other entity is the target",),
        )

    def step(self, tick: int, /) -> ExecutiveState | None:
        return self.state if tick == 0 else None


def run(
    bodies: Path, *, lanes: int, ticks: int, seed: int, radius: float, budget_ms: float
) -> dict[str, Any]:
    trunk, walk, look, manifest = arch.load_bodies(bodies)
    config = arch.reference_config(trunk, walk, look)[0]
    env = blocks.environment(lanes, config.maximum_turn_degrees)
    keys = jax.random.split(jax.random.key(seed), ticks + 3)
    spawn = jnp.broadcast_to(jnp.asarray(blocks.SPAWN_XZ, jnp.float32), (lanes, 2))
    other = jnp.broadcast_to(jnp.asarray(blocks.OTHER_XYZ, jnp.float32), (lanes, 3))
    neutral = np.broadcast_to(
        np.asarray(blocks.NEUTRAL, np.int32), (lanes, blocks.HEAD_COUNT)
    )
    carry = env.reset(
        jax.random.split(keys[0], lanes), spawn, jnp.zeros((lanes,), jnp.float32), other
    )
    carry, observation, _ = env.step(
        carry, jnp.asarray(neutral), jax.random.split(keys[1], lanes)
    )
    tick_seconds = distill_v3._world_tick(carry.runtime.combat)

    library = arch.build_library(trunk, walk, look, batched=True)
    combiner = library.combiner(layout="gen7")
    bodies_by_verb = library.executors()
    contracts = (
        IdleSkill.contract,
        library["walk_to"].contract.contract,
        library["look_at"].contract.contract,
    )

    belief = SharedBelief(params=init_belief_params(jax.random.key(seed)))
    programs = [walk_then_look(radius) for _ in range(lanes)]

    def facts_for_decider(facts: Any) -> Gen7DecisionFacts:
        return Gen7DecisionFacts(
            legal_observation=facts.observation,
            previous_action=np.asarray(facts.last, np.int32),
            done=np.zeros((lanes,), bool),
        )

    decider = Gen7ProgramDecider(
        programs,
        contracts=contracts,
        belief=belief,
        tick_seconds=tick_seconds,
        facts_for_decider=facts_for_decider,
        jit_belief=True,
    )
    document = SharedState()
    world = arch._StackWorld(env, carry, observation, keys, neutral, lanes)

    pose_type: list[type] = []

    def observe(transition: Any) -> tuple:
        obs, position, yaw, last12, reset = transition
        observation_, last_action, _, _ = state_tick.host_inputs(
            obs, jnp.asarray(last12), jnp.zeros((lanes,), bool)
        )
        fed = state_tick.own_pose_channel(observation_, position, yaw)
        if not pose_type:
            pose_type.append(type(fed))
        return pose_type[0](*fed), last_action, jnp.asarray(reset), None

    binding = StateTickBinding(
        trunk=trunk,
        state=distill_v3._start_state(trunk.params, lanes, trunk.config),
        observe=observe,
        verb_index=arch.VERB_INDEX,
        tick_seconds=tick_seconds,
        own_position=lambda transition: transition[1],
    )
    base = {lane: AgentContext(world_bounds=ANY_WORLD) for lane in range(lanes)}
    agents = list(range(lanes))
    illegal = 0
    first_inside = np.full(lanes, -1, np.int64)

    def transition_of(facts: Any) -> tuple:
        combat = facts.combat
        return (
            facts.observation,
            combat.position[:, AGENT_ENTITY],
            combat.yaw[:, AGENT_ENTITY],
            facts.last,
            np.full((lanes,), facts.first),
        )

    def contexts(tick: int, view: Any, facts: Any) -> dict:
        combat = facts.combat
        poses = Pose(
            combat.position[:, AGENT_ENTITY],
            combat.yaw[:, AGENT_ENTITY],
            combat.agent_head_yaw,
            combat.agent_head_pitch,
        )
        return state_tick.encode_contexts(
            view, base, agents, poses=poses, ticks=tick, action_masks=facts.mask
        )

    def record(index: int, facts: Any, steps: Any) -> None:
        nonlocal illegal
        illegal += int(
            np.sum(
                ~blocks.legal_rows(
                    np.asarray(facts.decided_mask), np.asarray(facts.executed)
                )
            )
        )
        planar = (
            np.asarray(facts.combat.position[:, AGENT_ENTITY])[:, [0, 2]]
            - np.asarray(blocks.OTHER_XYZ)[[0, 2]]
        )
        distance = np.hypot(planar[:, 0], planar[:, 1])
        inside = (first_inside < 0) & (distance <= radius)
        first_inside[inside] = index + 1

    session = run_gen7_session(
        world,
        decider,
        combiner,
        bodies_by_verb,
        binding,
        Gen7SessionConfig(ticks=ticks, decide_budget_ms=budget_ms, batch_bodies=True),
        contexts=contexts,
        transition_of=transition_of,
        head_count=blocks.HEAD_COUNT,
        document=document,
        executive=FixedExecutive(radius),
        record=record,
    )

    final = session.final_facts.combat
    other_xz = np.asarray(blocks.OTHER_XYZ)[[0, 2]]
    end_planar = np.asarray(final.position[:, AGENT_ENTITY])[:, [0, 2]] - other_xz
    end_distance = np.hypot(end_planar[:, 0], end_planar[:, 1])
    walk_goal = WalkToGoals(
        target_xz=jnp.broadcast_to(jnp.asarray(other_xz, jnp.float32), (lanes, 2)),
        radius=jnp.full((lanes,), radius, jnp.float32),
        gait=jnp.ones((lanes,), jnp.int32),
        horizon_ticks=jnp.full((lanes,), ticks, jnp.int32),
    )
    # Scored at the point the decision layer aims look_at at: the other entity's eye (gen7_decider.LOOK_EYE_HEIGHT).
    look_point = np.asarray(blocks.OTHER_XYZ, np.float32) + np.asarray([0.0, LOOK_EYE_HEIGHT, 0.0], np.float32)
    look_goal = LookAtGoals(
        target_xyz=jnp.broadcast_to(jnp.asarray(look_point), (lanes, 3)),
        tolerance_degrees=jnp.full((lanes,), LOOK_TOLERANCE_DEGREES, jnp.float32),
        horizon_ticks=jnp.full((lanes,), ticks, jnp.int32),
    )
    walk_ok = np.asarray(WalkToTasks.success(final, walk_goal))
    look_ok = np.asarray(LookAtTasks.success(final, look_goal))
    rejections: dict[str, int] = {}
    examples: dict[str, str] = {}
    for status in document.skills.statuses:
        if status.kind == "rejected":
            code = rejection_code(status.reason or "")
            rejections[code] = rejections.get(code, 0) + 1
            examples.setdefault(code, status.reason or "")
    steps = decider.executor.step_completions
    belief_now = decider._update.belief
    known = np.asarray(belief_now.feature_known)
    values = np.asarray(belief_now.features)
    visible, distance = (
        feature_index("target_visible"),
        feature_index("target_distance_blocks"),
    )
    metrics = session.metrics
    report = {
        "world": "training.blocks.environment (architecture test world)",
        "bodies": str(bodies),
        "manifest_run": manifest.get("run"),
        "lanes": lanes,
        "ticks": metrics.ticks,
        "seed": seed,
        "radius": radius,
        "tick_seconds": tick_seconds,
        "cadence_ticks": decider.cadence_ticks,
        "budget_ms": budget_ms,
        "decision_latency_ms": {
            "p50": metrics.host.latency_ms_p50,
            "p95": metrics.host.latency_ms_p95,
            "decisions": metrics.host.decisions,
            "misses": metrics.host.misses,
            "errors": metrics.host.errors,
        },
        "host_ms_per_tick": metrics.host_ms_per_tick,
        "statuses_by_kind": metrics.status_counts,
        "rejections_by_code": rejections,
        "rejection_examples": examples,
        "decision_layer": decider.counters(),
        "program_step_completions": {
            "walk_to": int(steps[:, 0].sum()),
            "look_at": int(steps[:, 1].sum()),
        },
        "programs_finished": int(decider.executor.programs_finished.sum()),
        "illegal_actions": illegal,
        "end": {
            "lanes_within_radius": int((end_distance <= radius).sum()),
            "walk_success_kernel": int(walk_ok.sum()),
            "look_success_kernel": int(look_ok.sum()),
            "end_planar_distance_median": float(np.median(end_distance)),
            "agent_y_median": float(
                np.median(np.asarray(final.position[:, AGENT_ENTITY])[:, 1])
            ),
            "other_y": float(blocks.OTHER_XYZ[1]),
            "look_pitch_to_target_eye_median_deg": float(
                np.median(
                    np.degrees(
                        np.arctan2(
                            float(look_point[1])
                            - (
                                np.asarray(final.position[:, AGENT_ENTITY])[:, 1]
                                + AGENT_EYE_HEIGHT
                            ),
                            np.maximum(end_distance, 1e-6),
                        )
                    )
                )
            ),
            "first_tick_inside_radius": [int(v) for v in first_inside],
        },
        "belief_at_end": {
            "target_visible_known_lanes": int(known[:, visible].sum()),
            "target_visible_lanes": int((values[:, visible] > 0.5).sum()),
            "target_distance_known_lanes": int(known[:, distance].sum()),
            "target_distance_median_blocks": float(
                np.median(values[known[:, distance], distance])
            )
            if known[:, distance].any()
            else None,
        },
        "wall_seconds": round(metrics.wall_seconds, 1),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bodies", type=Path, required=True, help="a v1_bodies/<run> folder"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=77001)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--budget-ms", type=float, default=20.0)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    report = run(
        args.bodies,
        lanes=args.lanes,
        ticks=args.ticks,
        seed=args.seed,
        radius=args.radius,
        budget_ms=args.budget_ms,
    )
    report["total_wall_seconds"] = round(time.perf_counter() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            report,
            indent=1,
            default=lambda v: v.item() if hasattr(v, "item") else str(v),
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "decision_latency_ms",
                    "statuses_by_kind",
                    "program_step_completions",
                    "programs_finished",
                    "illegal_actions",
                    "end",
                )
            },
            indent=1,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
