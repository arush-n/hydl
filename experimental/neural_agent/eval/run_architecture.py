"""The architecture test: the decision layer drives the skill layer's trained walk_to and look_at bodies through the
shared state, the combiner and the state tick (asked in the skill layer's ledger, dc9037a; owner, 2026-10-24: go).

One host tick for B lanes, in the default order (``--order fresh``):

1. state tick: ``shared_state.skill_writes.StateTickBinding.on_tick`` runs ``state_tick.step_lanes`` on this tick's
   observation (the state layer with the agent's own pose, then the memory trunk), and feeds the layer the requests that
   ended on the previous tick;
2. decision layer (:class:`GoalDecider`): judges each goal from its condition, withdraws the request of a goal that is
   done or out of time, and writes the live requests into S_t's ``requests`` section (writer ``jev``);
3. combiner on S_t (``skill_agent.runtime.scratchpad.combine_from_scratchpad``): grants each request its scope and runs
   its body, a ``ModuleBody`` per verb (``v1_bodies/<run>``: trunk.npz, walk_to.npz, look_at.npz) on the tick's
   ``AgentView``; the statuses go to ``skills``;
4. the simulator steps on the combined action (the gym's 12 heads).

``--order session`` runs step 1 after step 3, as ``session.run_session`` orders its hook (the state tick last), so the
bodies read the previous tick's view and the first tick is neutral.

Arms, on the same goals, spawns and keys (``distill_v3.evaluate``'s derivation from ``--seed``):

- ``reference``: the V1 student exactly as the skill layer evaluates it (``distill_v3.evaluate``: one generalist call per
  tick with both requests in their trained slots, masked argmax), from the same checkpoint;
- ``stack``: the loop above.

Both report ``distill._summarise``'s numbers (per mode success and held success, pace, radius, illegal decisions), so a
gap between them belongs to the architecture (separate body calls, the unmasked argmax, the request lowering, the tick
order), not to the model. ``stack`` adds the decision layer's view: goals done, each walk goal's TaskSpec score
(``HoldArea`` over ``blocks.HOLD_TICKS``, ``task_spec.hold_step``), the combiner's statuses, and host time per tick.

The world is the one the bodies were trained on: ``training.blocks.environment``, the committed engine with the motion
timing pinned to 1/30 s. Run under ``skill_agent/runtime/gen7.py`` to use a gen-7 render instead.

Each body lowers its request with the skill's own ``lowering`` (``neural_skills/<skill>/skill.py``: the trainer's request
row from the request's arguments and the agent's pose and decision tick, which the host reads from the simulator and
passes in each agent's ``AgentView`` with its legal action mask).

CLI (from the repo root; CPU)::

    PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu nice -n 10 \\
        .venv/bin/python experimental/neural_agent/eval/run_architecture.py --bodies DIR --out REPORT.json \\
        [--arms reference,stack] [--order fresh] [--lanes 96] [--seed 77001] [--held-out] [--ticks N] [--per-agent]

The stack arm batches each verb's bodies across agents (``state_tick.run_module_batches`` before the combiner, each
``ModuleBody`` reading its agent's stored step); ``--per-agent`` calls them one agent at a time. Both give the same
report (held out, 96 lanes, 2026-10-25: every non-timing field equal; combiner and bodies 68.6 -> 49.2 ms a tick).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:  # run as a script
    sys.path.insert(0, str(ROOT))

from experimental.neural_agent.requests import ActionRequest  # noqa: E402
from experimental.neural_agent.shared_state import RequestsSection, SharedState  # noqa: E402
from experimental.neural_agent.shared_state.skill_writes import StateTickBinding  # noqa: E402
from experimental.neural_agent.skill_agent.contract.surface import SCOPE_NAMES, scope_of  # noqa: E402
from experimental.neural_agent.skill_agent.model.generalist import ARG_GROUPS, GeneralistRequests  # noqa: E402
from experimental.neural_agent.skill_agent.model.skill_module import load_module, load_trunk  # noqa: E402
from experimental.neural_agent.skill_agent.neural_skills.look_at import skill as look_skill  # noqa: E402
from experimental.neural_agent.skill_agent.neural_skills.spec import VERBS, Pose  # noqa: E402
from experimental.neural_agent.skill_agent.neural_skills.walk_to import skill as walk_skill  # noqa: E402
from experimental.neural_agent.skill_agent.runtime import state_tick  # noqa: E402
from experimental.neural_agent.skill_agent.runtime.combiner import AgentContext, BodyStep  # noqa: E402
from experimental.neural_agent.skill_agent.runtime.library import SkillLibrary  # noqa: E402
from experimental.neural_agent.skill_agent.runtime.scratchpad import ANY_WORLD, combine_from_scratchpad  # noqa: E402
from experimental.neural_agent.skill_agent.training import blocks, distill, distill_v3  # noqa: E402
from experimental.neural_agent.tasks import task_spec as ts  # noqa: E402
from hytalegym.jax.combat.types import AGENT_ENTITY  # noqa: E402

VERB_INDEX = {verb: index for index, verb in enumerate(VERBS)}
"""The skill layer's verb vocabulary (``neural_skills/spec.py``), which the state layer's outcome columns read."""
GOAL_VERBS = ("walk_to", "look_at")
"""The verbs this decision layer issues (the vocabulary above grows as the skill layer adds skills)."""
REQUEST_SLACK_TICKS = 30
"""A request expires this many ticks after its goal's time is up; the decision layer withdraws it first."""


# ---- the checkpoint ----------------------------------------------------------------------------------------------


def load_bodies(directory: Path) -> tuple[Any, Any, Any, dict]:
    """``(trunk, walk_to module, look_at module, manifest)`` from a ``v1_bodies/<run>`` folder, each checked against its
    SHA-256 sidecar and against the trunk it was trained on."""

    trunk = load_trunk(directory / "trunk.npz")
    walk = load_module(directory / "walk_to.npz", trunk)
    look = load_module(directory / "look_at.npz", trunk)
    manifest = json.loads((directory / "manifest.json").read_text())
    return trunk, walk, look, manifest


def reference_config(trunk: Any, walk: Any, look: Any) -> tuple[distill_v3.DistillV3Config, dict]:
    """V1's configuration and parameters for the reference arm. V1 trained one generalist; its two exported modules
    carry the same parameters, which this checks."""

    a = jax.tree_util.tree_leaves(walk.params)
    b = jax.tree_util.tree_leaves(look.params)
    if len(a) != len(b) or not all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b)):
        raise ValueError("walk_to and look_at carry different parameters: the reference arm needs V1's one generalist")
    config = distill_v3.DistillV3Config(encoder=trunk.config, generalist=walk.config)
    return config, {"trunk": trunk.params, "generalist": walk.params}


# ---- the library -------------------------------------------------------------------------------------------------


LOWERINGS = {"walk_to": walk_skill.lowering, "look_at": look_skill.lowering}
SCOPES = {"walk_to": "locomotion", "look_at": "gaze"}


@dataclass(frozen=True)
class JointModuleBody:
    """Diagnostic body (``--joint``): ``state_tick.ModuleBody`` with every live request of a verb the module was trained
    beside in its request slots, as the generalist saw them in training (``distill._request_slots``), not only its own.
    Its tokens are ``(AgentView, the agent's live requests)``; its choice is ``ModuleBody``'s, the masked argmax on its
    scope's trained heads."""

    module: Any
    trunk: Any

    def __call__(self, request: Any, ticks_since_start: int, observation_tokens: Any, belief_tokens: Any) -> BodyStep:
        agent_view, live = observation_tokens
        view, row, pose, tick, action_mask = agent_view
        one = jax.tree_util.tree_map(lambda leaf: leaf[row:row + 1], view)
        slots = self.module.config.max_requests
        rows = [item for item in live if VERB_INDEX.get(item.verb, -1) in self.module.known][:slots]
        verb = np.full((1, slots), self.module.verb, np.int32)
        scope = np.full((1, slots), SCOPE_NAMES.index(self.module.scope), np.int32)
        args = np.zeros((1, slots, self.module.config.arg_width), np.float32)
        present = np.zeros((1, slots, len(ARG_GROUPS)), bool)
        valid = np.zeros((1, slots), bool)
        for slot, item in enumerate(rows):
            args[0, slot], present[0, slot] = LOWERINGS[item.verb](item, pose, tick)
            verb[0, slot], scope[0, slot] = VERB_INDEX[item.verb], SCOPE_NAMES.index(SCOPES[item.verb])
            valid[0, slot] = True
        none = jnp.full((1, slots), -1, jnp.int32)
        requests = GeneralistRequests(verb=jnp.asarray(verb), args=jnp.asarray(args), arg_present=jnp.asarray(present),
                                      entity_pointer=none, cell_pointer=none, scope=jnp.asarray(scope),
                                      valid=jnp.asarray(valid))
        out = state_tick._module_program(self.module.config, self.module.known, self.module.verb)(
            self.module.params, one.encoded, one.masks, requests)
        host = jax.device_get(out.head_logits)
        legal = np.asarray(action_mask, bool)
        choices = {}
        for index, name in enumerate(blocks.HEAD_NAMES):
            if scope_of(name) == self.module.scope:
                start, size = blocks.HEAD_STARTS[index], blocks.HEAD_SIZES[index]
                choices[name] = int(np.argmax(np.where(legal[start:start + size], np.asarray(host[name][0]), -np.inf)))
        return BodyStep(choices)


def build_library(trunk: Any, walk: Any, look: Any, *, joint: bool = False, batched: bool = False) -> SkillLibrary:
    """walk_to and look_at registered with their trained bodies and the skills' own lowerings (not from the catalogue,
    which this harness does not read). ``joint`` registers the diagnostic :class:`JointModuleBody` instead. ``batched``
    gives each ``ModuleBody`` an empty ``batch_results`` dict that :func:`run_stack` refills every tick."""

    def body(module: Any, lowering: Any) -> Any:
        if joint:
            return JointModuleBody(module, trunk)
        return state_tick.ModuleBody(module, trunk, lowering, batch_results={} if batched else None)

    library = SkillLibrary()
    library.create("walk_to", description="walk to a point at a pace and stop inside a radius", scopes=("locomotion",),
                   required=("position",), scalars=("radius", "pace"), done="arrived",
                   body=body(walk, walk_skill.lowering))
    library.create("look_at", description="turn the head to face a point", scopes=("gaze",), required=("position",),
                   scalars=("tolerance_degrees",), done="facing", body=body(look, look_skill.lowering))
    return library


# ---- the decision layer -------------------------------------------------------------------------------------------


class GoalDecider:
    """Each lane's goals become requests, and each request is withdrawn when its goal is done or out of time.

    A walk goal is a TaskSpec ``HoldArea``: inside the radius for ``HOLD_TICKS`` ticks in a row (``task_spec.hold_step``,
    scored with its reward). A look goal is the skill's own condition (``LookAtTasks.success``) held as long; TaskSpec has
    no look objective. A goal's time is its sampled horizon plus the hold, as the skill layer's evaluation allows.
    """

    def __init__(self, library: SkillLibrary, episode: distill.Episodes, ground_y: np.ndarray, tick_seconds: float):
        self.library = library
        self.tick_seconds = tick_seconds
        self.walk_target = np.asarray(episode.walk.target_xz, np.float64)
        self.walk_radius = np.asarray(episode.walk.radius, np.float64)
        self.walk_gait = np.asarray(episode.walk.gait)
        self.look_target = np.asarray(episode.look.target_xyz, np.float64)
        self.look_tolerance = np.asarray(episode.look.tolerance_degrees, np.float64)
        self.active = {"walk_to": np.asarray(episode.walk_active), "look_at": np.asarray(episode.look_active)}
        self.limit = {"walk_to": np.asarray(episode.walk.horizon_ticks) + blocks.HOLD_TICKS - 1,
                      "look_at": np.asarray(episode.look.horizon_ticks) + blocks.HOLD_TICKS - 1}
        self.ground_y = ground_y
        batch = self.walk_target.shape[0]
        self.streak = {verb: np.zeros(batch, np.int64) for verb in GOAL_VERBS}
        self.done_tick = {verb: np.full(batch, -1, np.int64) for verb in GOAL_VERBS}
        self.timed_out = {verb: np.zeros(batch, bool) for verb in GOAL_VERBS}
        self.walk_reward = np.zeros(batch, np.float64)
        self.live: dict[tuple[int, str], ActionRequest] = {}
        self._next_id = np.ones(batch, np.int64)
        self._specs = [self._walk_spec(lane) for lane in range(batch)]

    def _walk_spec(self, lane: int) -> ts.TaskSpec:
        x, z = self.walk_target[lane]
        objective = ts.HoldArea(center=(float(x), float(self.ground_y[lane]), float(z)),
                                radius=float(self.walk_radius[lane]), seconds=blocks.HOLD_TICKS / ts.TICK_RATE)
        limit_seconds = float(self.limit["walk_to"][lane] + 1) / ts.TICK_RATE
        return ts.TaskSpec(name="walk_to", world=ts.World(), bodies=(ts.Body(),), objectives=(objective,),
                           time_limit_seconds=max(limit_seconds, objective.seconds))

    def start(self, tick: int) -> None:
        """Issue every active goal's request."""

        for lane in range(self.walk_target.shape[0]):
            for verb in GOAL_VERBS:
                if self.active[verb][lane]:
                    self._issue(lane, verb, tick)

    def _issue(self, lane: int, verb: str, tick: int) -> None:
        horizon = (int(self.limit[verb][lane]) + 1 + REQUEST_SLACK_TICKS) * self.tick_seconds
        if verb == "walk_to":
            x, z = self.walk_target[lane]
            position = (float(x), float(self.ground_y[lane]), float(z))
            scalars = {"radius": float(self.walk_radius[lane]), "pace": float(self.walk_gait[lane])}
        else:
            position = tuple(float(v) for v in self.look_target[lane])
            scalars = {"tolerance_degrees": float(self.look_tolerance[lane])}
        request_id = int(self._next_id[lane])
        self._next_id[lane] += 1
        self.live[(lane, verb)] = self.library.request(verb, agent=lane, request_id=request_id, tick=tick,
                                                       horizon_seconds=horizon, position=position, scalars=scalars)

    def judge(self, tick: int, combat: Any, look_ok: np.ndarray) -> None:
        """One tick of each live goal's condition, on the state after the last step; ``tick`` counts the decisions taken."""

        position = np.asarray(combat.position[:, AGENT_ENTITY], np.float64)
        for (lane, verb) in list(self.live):
            if verb == "walk_to":
                spec = self._specs[lane]
                inside = ts.planar_distance(position[lane], spec.objectives[0].center) <= spec.objectives[0].radius
                reward, streak = ts.hold_step(self.streak[verb][lane], inside, blocks.HOLD_TICKS, spec)
                self.walk_reward[lane] += float(reward)
                done = bool(ts.hold_done(streak, tick, blocks.HOLD_TICKS, int(self.limit[verb][lane]) + 1))
            else:
                streak = np.where(look_ok[lane], self.streak[verb][lane] + 1, 0)
                done = bool(streak >= blocks.HOLD_TICKS) or tick > self.limit[verb][lane]
            self.streak[verb][lane] = int(streak)
            if done:
                if self.streak[verb][lane] >= blocks.HOLD_TICKS:
                    self.done_tick[verb][lane] = tick
                else:
                    self.timed_out[verb][lane] = True
                del self.live[(lane, verb)]  # withdrawn: a ModuleBody never reports done

    def write(self, document: SharedState, tick: int) -> None:
        live = tuple(sorted(self.live.values(), key=lambda request: (request.agent, request.request_id)))
        document.write("requests", RequestsSection(live=live), writer="jev", tick=tick)

    def report(self) -> dict:
        out = {}
        for verb in GOAL_VERBS:
            active = self.active[verb]
            count = int(active.sum())
            done = (self.done_tick[verb] >= 0) & active
            out[verb] = {"goals": count, "done": int(done.sum()), "timed_out": int((self.timed_out[verb] & active).sum()),
                         "still_live": int(sum(1 for lane, v in self.live if v == verb)),
                         "done_rate": float(done.sum() / count) if count else None,
                         "mean_done_tick": float(self.done_tick[verb][done].mean()) if done.any() else None}
        walk = self.active["walk_to"]
        out["walk_to"]["task_spec_hold_area_reward_mean"] = float(self.walk_reward[walk].mean()) if walk.any() else None
        return out


# ---- the stack arm ------------------------------------------------------------------------------------------------


def run_stack(directory: Path, *, seed: int, lanes: int, held_out: bool, order: str, max_ticks: int | None,
              joint: bool = False, batch: bool = True) -> dict:
    trunk, walk, look, _ = load_bodies(directory)
    config, _ = reference_config(trunk, walk, look)
    kernels = config.kernels()
    env = blocks.environment(lanes, config.maximum_turn_degrees)
    key_goals, key_rollout = jax.random.split(jax.random.key(seed))
    episode = distill.sample_episodes(key_goals, jnp.asarray(np.arange(lanes) % len(distill.MODES)), held_out=held_out)
    horizon = np.asarray(episode.horizon)
    ticks = int(horizon.max()) if max_ticks is None else min(int(horizon.max()), max_ticks)
    keys = jax.random.split(key_rollout, int(horizon.max()) + 2)

    spawn = jnp.broadcast_to(jnp.asarray(blocks.SPAWN_XZ, jnp.float32), (lanes, 2))
    other = jnp.broadcast_to(jnp.asarray(blocks.OTHER_XYZ, jnp.float32), (lanes, 3))
    neutral = np.broadcast_to(np.asarray(blocks.NEUTRAL, np.int32), (lanes, blocks.HEAD_COUNT))
    carry = env.reset(jax.random.split(keys[0], lanes), spawn, episode.spawn_yaw, other)
    carry, observation, _ = env.step(carry, jnp.asarray(neutral), jax.random.split(keys[1], lanes))
    tick_seconds = distill_v3._world_tick(carry.runtime.combat)

    batch = batch and not joint  # the diagnostic joint body has no batch path
    library = build_library(trunk, walk, look, joint=joint, batched=batch)
    combiner = library.combiner(layout="gen7")
    bodies = library.executors()
    document = SharedState()
    ground_y = np.asarray(carry.runtime.combat.position[:, AGENT_ENTITY, 1], np.float64)
    decider = GoalDecider(library, episode, ground_y, tick_seconds)

    pose_type: list[type] = []

    def observe(transition):
        obs, position, yaw, last12, reset = transition
        observation_, last_action, _, _ = state_tick.host_inputs(obs, jnp.asarray(last12), jnp.zeros((lanes,), bool))
        fed = state_tick.own_pose_channel(observation_, position, yaw)
        # own_pose_channel builds a new NamedTuple class on every call; one class for the run keeps the compiled state
        # tick from retracing every tick (6.5 s a tick in the first smoke).
        if not pose_type:
            pose_type.append(type(fed))
        return pose_type[0](*fed), last_action, jnp.asarray(reset), None

    binding = StateTickBinding(trunk=trunk, state=distill_v3._start_state(trunk.params, lanes, trunk.config),
                               observe=observe, verb_index=VERB_INDEX, tick_seconds=tick_seconds)
    base = {lane: AgentContext(world_bounds=ANY_WORLD) for lane in range(lanes)}
    agents = list(range(lanes))

    walk_active, look_active = np.asarray(episode.walk_active), np.asarray(episode.look_active)
    walk_horizon, look_horizon = np.asarray(episode.walk.horizon_ticks), np.asarray(episode.look.horizon_ticks)
    walk_hit = np.full(lanes, -1, np.int32)
    look_hit = np.full(lanes, -1, np.int32)
    walk_streak = np.zeros(lanes, np.int32)
    look_streak = np.zeros(lanes, np.int32)
    walk_held = np.full(lanes, -1, np.int32)
    look_held = np.full(lanes, -1, np.int32)
    positions = np.zeros((ticks + 1, lanes, 2), np.float32)
    yaws = np.zeros((ticks + 1, lanes), np.float32)
    seconds = np.zeros((ticks, lanes), np.float32)
    decisions = illegal = 0
    statuses: Counter = Counter()
    clock = Counter()

    def record_instant(index: int, combat: Any) -> None:
        positions[index] = np.asarray(combat.position[:, AGENT_ENTITY][:, [0, 2]])
        yaws[index] = np.asarray(combat.yaw[:, AGENT_ENTITY])

    def state_tick_now(index: int, tag: int) -> None:
        combat = carry.runtime.combat
        transition = (observation, combat.position[:, AGENT_ENTITY], combat.yaw[:, AGENT_ENTITY], last,
                      np.full((lanes,), index == 0))
        started = time.perf_counter()
        binding.on_tick(document, tag, False, transition)
        jax.block_until_ready(binding.view.encoded)
        clock["state_tick"] += time.perf_counter() - started

    record_instant(0, carry.runtime.combat)
    last = neutral
    look_ok = np.zeros(lanes, bool)
    for index in range(ticks):
        combat = carry.runtime.combat
        if order == "fresh":
            state_tick_now(index, index - 1)  # the statuses of the combiner's last tick reach the layer now

        started = time.perf_counter()
        if index == 0:
            decider.start(index)
        else:
            decider.judge(index, combat, look_ok)
        decider.write(document, index)
        clock["decision"] += time.perf_counter() - started

        started = time.perf_counter()
        if binding.view is None:  # the session order's first tick: no view yet
            executed = neutral
        else:
            poses = Pose(combat.position[:, AGENT_ENTITY], combat.yaw[:, AGENT_ENTITY], combat.agent_head_yaw,
                         combat.agent_head_pitch)
            contexts = state_tick.encode_contexts(binding.view, base, agents, poses=poses, ticks=index,
                                                  action_masks=carry.action_mask)
            if joint:
                contexts = {agent: replace(context, observation_tokens=(context.observation_tokens,
                                                                         document.requests.for_agent(agent)))
                            for agent, context in contexts.items()}
            if batch:  # one module pass per verb over its agents; each body then returns its agent's stored step
                live: dict[int, dict[str, Any]] = {}
                for agent in agents:
                    for request in document.requests.for_agent(agent):
                        if request.verb in live.setdefault(agent, {}):
                            raise RuntimeError(f"agent {agent} has two live {request.verb} requests; a batch holds one")
                        live[agent][request.verb] = request
                results = state_tick.run_module_batches(binding.view, contexts, bodies, live)
                for verb, body in bodies.items():
                    body.batch_results.clear()
                    body.batch_results.update(results.get(verb, {}))
            steps = combine_from_scratchpad(document, combiner, index, contexts)
            for step in steps.values():
                statuses.update(status.kind for status in step.statuses)
            executed = np.stack([np.asarray(steps[lane].factors[:blocks.HEAD_COUNT], np.int32) for lane in agents])
        clock["combiner_and_bodies"] += time.perf_counter() - started

        if order == "session":
            state_tick_now(index, index)

        keep = index < horizon
        mask = np.asarray(carry.action_mask)
        decisions += int(keep.sum())
        illegal += int(np.sum(keep & ~blocks.legal_rows(mask, executed)))

        started = time.perf_counter()
        carry, observation, _ = env.step(carry, jnp.asarray(executed), jax.random.split(keys[index + 2], lanes))
        last = executed
        combat = carry.runtime.combat
        clock["environment"] += time.perf_counter() - started
        if index % 25 == 0:
            print(f"stack {order} tick {index}/{ticks} ms/tick so far "
                  f"{ {name: round(1000.0 * value / (index + 1), 1) for name, value in clock.items()} }",
                  file=sys.stderr, flush=True)
        done_count = index + 1
        record_instant(done_count, combat)
        seconds[index] = np.asarray(combat.last_motion_delta_seconds)
        walk_ok, look_ok = (np.asarray(value) for value in kernels.success(combat, episode))
        walk_hit = np.where((walk_hit < 0) & walk_active & walk_ok & (done_count <= walk_horizon), done_count,
                            walk_hit).astype(np.int32)
        look_hit = np.where((look_hit < 0) & look_active & look_ok & (done_count <= look_horizon), done_count,
                            look_hit).astype(np.int32)
        walk_streak, walk_held = blocks.hold_update(walk_streak, walk_held, walk_ok, done_count, walk_horizon,
                                                    walk_active)
        look_streak, look_held = blocks.hold_update(look_streak, look_held, look_ok, done_count, look_horizon,
                                                    look_active)

    rollout = distill.Rollout(walk_hit=walk_hit, look_hit=look_hit, positions=positions, yaws=yaws, seconds=seconds,
                              decisions=decisions, illegal=illegal, teacher_illegal=0, samples=None,
                              walk_held=walk_held, look_held=look_held)
    report = {"arm": "stack_joint" if joint else "stack", "order": order, "held_out": held_out, "seed": seed, "lanes": lanes, "ticks": ticks,
              "modes": list(distill.MODES), **distill._summarise(episode, rollout)}
    report["tick_seconds"] = tick_seconds
    report["decision_layer"] = decider.report()
    report["combiner_statuses"] = dict(statuses)
    report["host_ms_per_tick"] = {name: round(1000.0 * value / ticks, 2) for name, value in clock.items()}
    report["host_ms_per_tick"]["total"] = round(1000.0 * sum(clock.values()) / ticks, 2)
    return report


def run_reference(directory: Path, *, seed: int, lanes: int, held_out: bool) -> dict:
    trunk, walk, look, _ = load_bodies(directory)
    config, params = reference_config(trunk, walk, look)
    report = distill_v3.evaluate(params, config, seed=seed, lanes=lanes, held_out=held_out)
    return {"arm": "reference", **report}


def _bars(report: dict) -> dict:
    """The headline numbers of a report, for the comparison table."""

    both = report["both"]
    return {"walk_held": report["walk_only"]["held"], "look_held": report["look_only"]["held"], "both_held": both["held"],
            "both_walk_held": both["walk_held"], "both_look_held": both["look_held"],
            "pace_within_5pct": report["walk_pace"]["within_5_percent"],
            "radius_kept": report["walk_radius"]["within_radius"], "illegal": report["illegal_decisions"],
            "decisions": report["decisions"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bodies", type=Path, required=True, help="a v1_bodies/<run> folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="reference,stack")
    parser.add_argument("--order", choices=("fresh", "session", "both"), default="fresh")
    parser.add_argument("--lanes", type=int, default=96)
    parser.add_argument("--seed", type=int, default=77001)
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--ticks", type=int, default=None, help="cap the stack arm's ticks (a smoke)")
    parser.add_argument("--joint", action="store_true",
                        help="the stack arm's bodies read every live request (diagnostic JointModuleBody)")
    parser.add_argument("--per-agent", action="store_true",
                        help="call each body per agent (no ModuleBatch); the batched default gives the same choices")
    args = parser.parse_args(argv)

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if not arms or any(arm not in ("reference", "stack") for arm in arms):
        parser.error("--arms takes reference and/or stack")
    orders = ("fresh", "session") if args.order == "both" else (args.order,)
    _, _, _, manifest = load_bodies(args.bodies)
    results: dict[str, Any] = {"bodies": str(args.bodies), "manifest_run": manifest.get("run"),
                               "manifest_arm": manifest.get("arm"), "argv": list(argv or sys.argv[1:]), "runs": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:  # after every arm, so a slow later arm never loses an earlier one
        results["bars"] = {name: _bars(report) for name, report in results["runs"].items()}
        args.out.write_text(json.dumps(results, indent=1, default=lambda value: value.item() if hasattr(value, "item")
                                       else str(value)) + "\n")

    if "reference" in arms:
        started = time.perf_counter()
        results["runs"]["reference"] = run_reference(args.bodies, seed=args.seed, lanes=args.lanes, held_out=args.held_out)
        results["runs"]["reference"]["wall_seconds"] = round(time.perf_counter() - started, 1)
        save()
    if "stack" in arms:
        for order in orders:
            started = time.perf_counter()
            report = run_stack(args.bodies, seed=args.seed, lanes=args.lanes, held_out=args.held_out, order=order,
                               max_ticks=args.ticks, joint=args.joint, batch=not args.per_agent)
            report["wall_seconds"] = round(time.perf_counter() - started, 1)
            report["bodies_batched"] = not args.per_agent and not args.joint
            results["runs"][f"stack_{order}" + ("_joint" if args.joint else "")] = report
            save()
    print(json.dumps(results["bars"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
