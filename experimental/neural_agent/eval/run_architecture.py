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

The loop is :func:`experimental.neural_agent.session_gen7.run_gen7_session`; this file supplies the world, the decider
and the bookkeeping. The session order (the state tick after the bodies, as ``session.run_session``) is no longer run:
``--order fresh`` is the only order (2026-10-25, the move to the runner).

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple, Sequence

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
from experimental.neural_agent.skill_agent.runtime.scratchpad import ANY_WORLD  # noqa: E402
from experimental.neural_agent.session_gen7 import Gen7SessionConfig, run_gen7_session  # noqa: E402
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


def build_library(trunk: Any, walk: Any, look: Any, *, joint: bool = False, batched: bool = False,
                  own_request: bool = False) -> SkillLibrary:
    """walk_to and look_at registered with their trained bodies and the skills' own lowerings (not from the catalogue,
    which this harness does not read). ``joint`` registers the diagnostic :class:`JointModuleBody` instead. ``batched``
    gives each ``ModuleBody`` an empty ``batch_results`` dict that :func:`run_stack` refills every tick."""

    def body(module: Any, lowering: Any) -> Any:
        if joint:
            return JointModuleBody(module, trunk)
        # The V1 bodies' slot order (walk_to, look_at): each module reads the agent's live request of the other verb, as it was
        # trained (the skill layer's 3e65786); ``own_request`` gives it only its own request, the 2026-10-24 arm.
        layout = None if own_request else state_tick.V1_LAYOUT
        return state_tick.ModuleBody(module, trunk, lowering, layout=layout, batch_results={} if batched else None)

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


class _StackFacts(NamedTuple):
    """The facts of one tick of the stack world (what the runner hands the caller's callables)."""

    observation: Any
    combat: Any
    last: Any  # the actions executed on the previous tick (neutral on tick 0)
    first: bool
    mask: Any  # the legal action mask this tick's decision reads
    decided_mask: Any  # after a step: the mask the step's actions were chosen under
    executed: Any  # after a step: the actions the step ran


class _StackWorld:
    """The environment for :func:`run_gen7_session`: the blocks environment, stepped on the runner's actions. Tick ``t``
    uses ``keys[t + 2]``; the caller has already run the reset and the neutral first step (``keys[0]``, ``keys[1]``)."""

    def __init__(self, env: Any, carry: Any, observation: Any, keys: Any, neutral: np.ndarray, lanes: int):
        self.env, self.carry, self.observation = env, carry, observation
        self.keys, self.neutral, self.lanes = keys, neutral, lanes
        self.index = 0

    def reset(self) -> _StackFacts:
        return _StackFacts(self.observation, self.carry.runtime.combat, self.neutral, True,
                           self.carry.action_mask, None, None)

    def step(self, actions: np.ndarray) -> _StackFacts:
        decided_mask = self.carry.action_mask
        self.carry, self.observation, _ = self.env.step(self.carry, jnp.asarray(actions),
                                                        jax.random.split(self.keys[self.index + 2], self.lanes))
        self.index += 1
        return _StackFacts(self.observation, self.carry.runtime.combat, actions, False,
                           self.carry.action_mask, decided_mask, actions)


class _StackDecider:
    """The decision layer as the runner's decider: the goals start on tick 0 and are judged on each later tick."""

    def __init__(self, goal: GoalDecider, lanes: int):
        self.goal = goal
        self.look_ok = np.zeros(lanes, bool)  # set by the run's record hook: each lane's look success on the last tick

    def decide(self, tick: int, document: SharedState, facts: _StackFacts, executive: Any) -> None:
        if tick == 0:
            self.goal.start(0)
        else:
            self.goal.judge(tick, facts.combat, self.look_ok)
        self.goal.write(document, tick)


def run_stack(directory: Path, *, seed: int, lanes: int, held_out: bool, order: str, max_ticks: int | None,
              joint: bool = False, batch: bool = True, own_request: bool = False) -> dict:
    if order != "fresh":
        raise ValueError("run_stack has one order, fresh: the session order is not in the gen-7 runner")
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
    library = build_library(trunk, walk, look, joint=joint, batched=batch, own_request=own_request)
    combiner = library.combiner(layout="gen7")
    bodies = library.executors()
    document = SharedState()
    ground_y = np.asarray(carry.runtime.combat.position[:, AGENT_ENTITY, 1], np.float64)
    goal = GoalDecider(library, episode, ground_y, tick_seconds)
    decider = _StackDecider(goal, lanes)
    world = _StackWorld(env, carry, observation, keys, neutral, lanes)

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

    def record_instant(index: int, combat: Any) -> None:
        positions[index] = np.asarray(combat.position[:, AGENT_ENTITY][:, [0, 2]])
        yaws[index] = np.asarray(combat.yaw[:, AGENT_ENTITY])

    def transition_of(facts: _StackFacts) -> tuple:
        combat = facts.combat
        return (facts.observation, combat.position[:, AGENT_ENTITY], combat.yaw[:, AGENT_ENTITY], facts.last,
                np.full((lanes,), facts.first))

    def contexts(tick: int, view: Any, facts: _StackFacts) -> dict:
        combat = facts.combat
        poses = Pose(combat.position[:, AGENT_ENTITY], combat.yaw[:, AGENT_ENTITY], combat.agent_head_yaw,
                     combat.agent_head_pitch)
        live = None if (joint or own_request) else {
            agent: {request.verb: request for request in document.requests.for_agent(agent)} for agent in agents}
        out = state_tick.encode_contexts(view, base, agents, poses=poses, ticks=tick, action_masks=facts.mask,
                                         live_requests=live)
        if joint:
            out = {agent: replace(context, observation_tokens=(context.observation_tokens,
                                                              document.requests.for_agent(agent)))
                   for agent, context in out.items()}
        return out

    def record(index: int, facts: _StackFacts, steps: Any) -> None:
        """After tick ``index``'s step: its legality, the post-step instants and the goal bookkeeping."""
        nonlocal decisions, illegal, walk_hit, look_hit, walk_streak, look_streak, walk_held, look_held
        keep = index < horizon
        decisions += int(keep.sum())
        illegal += int(np.sum(keep & ~blocks.legal_rows(np.asarray(facts.decided_mask), np.asarray(facts.executed))))
        done_count = index + 1
        combat = facts.combat
        record_instant(done_count, combat)
        seconds[index] = np.asarray(combat.last_motion_delta_seconds)
        walk_ok, look_ok = (np.asarray(value) for value in kernels.success(combat, episode))
        decider.look_ok = look_ok
        walk_hit = np.where((walk_hit < 0) & walk_active & walk_ok & (done_count <= walk_horizon), done_count,
                            walk_hit).astype(np.int32)
        look_hit = np.where((look_hit < 0) & look_active & look_ok & (done_count <= look_horizon), done_count,
                            look_hit).astype(np.int32)
        walk_streak, walk_held = blocks.hold_update(walk_streak, walk_held, walk_ok, done_count, walk_horizon,
                                                    walk_active)
        look_streak, look_held = blocks.hold_update(look_streak, look_held, look_ok, done_count, look_horizon,
                                                    look_active)

    record_instant(0, carry.runtime.combat)
    session = run_gen7_session(
        world, decider, combiner, bodies, binding, Gen7SessionConfig(ticks=ticks, batch_bodies=batch),
        contexts=contexts, transition_of=transition_of, head_count=blocks.HEAD_COUNT, document=document,
        record=record)

    rollout = distill.Rollout(walk_hit=walk_hit, look_hit=look_hit, positions=positions, yaws=yaws, seconds=seconds,
                              decisions=decisions, illegal=illegal, teacher_illegal=0, samples=None,
                              walk_held=walk_held, look_held=look_held)
    report = {"arm": "stack_joint" if joint else ("stack_own_request" if own_request else "stack"), "order": order, "held_out": held_out, "seed": seed, "lanes": lanes, "ticks": ticks,
              "modes": list(distill.MODES), **distill._summarise(episode, rollout)}
    report["tick_seconds"] = tick_seconds
    report["decision_layer"] = goal.report()
    report["combiner_statuses"] = dict(session.metrics.status_counts)
    ms = session.metrics.host_ms_per_tick
    report["host_ms_per_tick"] = {"state_tick": ms["state_tick"], "decision": ms["decision"],
                                  "combiner_and_bodies": ms["bodies"], "environment": ms["world"], "total": ms["total"]}
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
    parser.add_argument("--order", choices=("fresh",), default="fresh",
                        help="the only order: the session order is not in the gen-7 runner (session_gen7.py)")
    parser.add_argument("--lanes", type=int, default=96)
    parser.add_argument("--seed", type=int, default=77001)
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--ticks", type=int, default=None, help="cap the stack arm's ticks (a smoke)")
    parser.add_argument("--own-request", action="store_true",
                        help="each body reads only its own request (the 2026-10-24 arm); default: its trained layout's live requests")
    parser.add_argument("--joint", action="store_true",
                        help="the stack arm's bodies read every live request (diagnostic JointModuleBody)")
    parser.add_argument("--per-agent", action="store_true",
                        help="call each body per agent (no ModuleBatch); the batched default gives the same choices")
    args = parser.parse_args(argv)

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if not arms or any(arm not in ("reference", "stack") for arm in arms):
        parser.error("--arms takes reference and/or stack")
    orders = (args.order,)
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
                               max_ticks=args.ticks, joint=args.joint, batch=not args.per_agent,
                               own_request=args.own_request)
            report["wall_seconds"] = round(time.perf_counter() - started, 1)
            report["bodies_batched"] = not args.per_agent and not args.joint
            results["runs"][f"stack_{order}" + ("_joint" if args.joint else "") + ("_own_request" if args.own_request else "")] = report
            save()
    print(json.dumps(results["bars"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
