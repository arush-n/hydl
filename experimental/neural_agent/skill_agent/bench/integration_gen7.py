"""I1 integration check: the neural layer on the gen-7 simulator with perception_v2, game_clock and hearing on.

Run under the launcher, from the repo root (DIR is rendered once with ``--render``)::

    python -m experimental.neural_agent.skill_agent.runtime.gen7 --engine DIR --render \\
        --construct perception_v2 --construct game_clock --construct hearing -- \\
        experimental/neural_agent/skill_agent/bench/integration_gen7.py --ticks 300 --out REPORT.json

Each world runs the whole layer tick by tick: the state layer and the trunk through ``state_tick.step_lanes``; the
decision layer's writes at tick 10 (a ``remember`` tag and a relationship on the opponent, entity 1); the generalist
head, whose walk_to request names a fixed world point and whose executed action is the masked argmax of the heads of
the request's scope (neutral elsewhere). The state is saved at tick 150, loaded into a fresh state, and both copies step
tick 151 on the same input. The report is JSON: one entry per check with a ``status`` of ``pass``, ``fail``,
``not built`` (the part does not exist in this tree), ``not on this world`` (the world's observation does not carry
the construct) or ``not run``.

The sound and the hunger parts are detected, never faked: a missing slice is reported as ``not built``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.bench import agent_models as am
from experimental.neural_agent.skill_agent.bench import encoding_audit as audit
from experimental.neural_agent.skill_agent.contract import tokens_v3 as v3
from experimental.neural_agent.skill_agent.contract.surface import SCOPE_NAMES, scope_of
from experimental.neural_agent.skill_agent.model.encoder import init_encoder
from experimental.neural_agent.skill_agent.model.generalist import (
    ARG_GROUPS, ARG_FEATURES, GeneralistConfig, GeneralistRequests, generalist_apply_jit, init_generalist)
from experimental.neural_agent.skill_agent.model.skill_module import make_trunk
from experimental.neural_agent.skill_agent.neural_skills.spec import VERBS
from experimental.neural_agent.skill_agent.runtime import state_tick
from experimental.neural_agent.skill_agent.state import drives, layer, views
from experimental.neural_agent.skill_agent.state import registry
from experimental.neural_agent.skill_agent.state.frames import POSITION_SCALE_BLOCKS, world_to_body
from experimental.neural_agent.skill_agent.state.senses import Light
from experimental.neural_agent.skill_agent.state.written import Write
from experimental.neural_agent.skill_agent.training.blocks import HEAD_NAMES, HEAD_SIZES, HEAD_STARTS, NEUTRAL
from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors

ROOT = Path(__file__).resolve().parents[4]
BATCH = 2
"""Two agents, each in its own world row: the batch every check runs at."""
WALK_SCOPE = "locomotion"
"""The scope of the walk_to request; its heads are the locomotion heads."""
WRITE_TICK = 10
SAVE_TICK = 150
OPPONENT = 1
"""The opponent's entity id: entity 1 of the two-body world (entity 0 is the agent, the listener)."""
POINT_OFFSET = (4.0, 0.0, 0.0)
"""The walk_to point, in blocks from the agent's start (world axes)."""
REMEMBER = 1.0
RELATIONSHIP = (("closeness", 0.5), ("hostility", 0.6))
"""The relationship the decision layer writes about the opponent (state/written.py ``ENTITY_ENTRIES``, writer decision)."""
STATUS_PASS, STATUS_FAIL = "pass", "fail"
NOT_BUILT, NOT_ON_WORLD, NOT_RUN = "not built", "not on this world", "not run"
EXPECTED_TOKEN_WIDTHS = dict(am.CONFIG.token_types)


def _finite(tree: Any) -> bool:
    return all(bool(np.isfinite(np.asarray(leaf)).all()) for leaf in jax.tree_util.tree_leaves(tree)
               if np.asarray(leaf).dtype.kind == "f")


def tree_hash(tree: Any) -> str:
    """SHA-256 over every leaf's dtype, shape and bytes, in tree order."""

    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        digest.update(f"{array.dtype}{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def bit_mismatches(left: Any, right: Any) -> list[str]:
    """The leaves that are not bit-identical (dtype, shape and bytes), by key path."""

    out = []
    paths, _ = jax.tree_util.tree_flatten_with_path(left)
    for (path, a), b in zip(paths, jax.tree_util.tree_leaves(right)):
        x, y = np.asarray(a), np.asarray(b)
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            out.append(jax.tree_util.keystr(path))
    return out


# -- The two worlds ---------------------------------------------------------------------------------------------------


class Duel:
    """The two-body gen-7 duel as ``encoding_audit.duel_steps`` builds and steps it: the agent (entity 0) and the
    scripted opponent (entity 1, ``audit.chaser``), sword against sword, 3 to 12 blocks apart, facing each other."""

    name = "duel"

    def __init__(self, batch: int, seed: int = 0) -> None:
        from experimental.neural_agent.skill_agent.tasks.duel import BODY_Y
        from experimental.neural_agent.skill_agent.teaching.lookahead import VIEW_SECTOR_RADIANS
        from experimental.neural_agent.tasks.duel_arena import make_duel_arena

        self.batch = batch
        arena = make_duel_arena(batch, ("iron_sword", "iron_sword"), view_sector_full_angle_radians=VIEW_SECTOR_RADIANS,
                                legal_observation_sensor_range=True, max_health=(200.0, 200.0), arrows=(0, 0))
        rng = np.random.default_rng(seed)
        distance, angle = rng.uniform(3.0, 12.0, batch), rng.uniform(-np.pi, np.pi, batch)
        position = np.zeros((batch, 2, 3), np.float32)
        position[:, :, 1] = BODY_Y
        position[:, 1, 0], position[:, 1, 2] = distance * np.sin(angle), distance * np.cos(angle)
        facing = np.degrees(np.arctan2(-np.sin(angle), -np.cos(angle)))  # the gym's forward is (-sin, -cos) of yaw
        yaw = (np.stack([facing, facing + 180.0], axis=1) + 180.0) % 360.0 - 180.0
        self.arena = arena
        self.state, self._obs, self._opponent, self._mask, self._opponent_mask = arena.reset(
            jax.random.split(jax.random.key(seed), batch), entity_position=jnp.asarray(position),
            entity_health=jnp.full((batch, 2), 200.0, jnp.float32), entity_yaw=jnp.asarray(yaw, jnp.float32))
        self.step = jax.jit(arena.step)
        self.chase = jax.jit(audit.chaser)
        self.tables, self.effects = audit.registry_tables(), audit.effect_tables()

    def observation(self) -> Any:
        return self._obs

    def mask(self) -> jax.Array:
        return self._mask

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        combat = self.state.arsenal.combat
        return np.asarray(combat.position[:, 0], np.float64), np.asarray(combat.yaw[:, 0], np.float64)

    def kw(self) -> dict[str, Any]:
        return {"tables": self.tables, "effects": self.effects}

    def advance(self, factors: jax.Array, tick: int) -> None:
        keys = jax.random.split(jax.random.key(tick), self.batch)
        (self.state, self._obs, self._opponent, self._mask, self._opponent_mask, _, _) = self.step(
            self.state, factors, self.chase(self._opponent, self._opponent_mask), keys)

    def tick_seconds(self) -> float:
        return float(np.asarray(self.state.arsenal.combat.last_motion_delta_seconds).reshape(-1)[0])


class Region:
    """A single-actor Region world with its built-in target active (``encoding_audit.region_steps``): the agent
    (entity 0) and the target (entity 1) on real terrain."""

    name = "region"

    def __init__(self, batch: int) -> None:
        from worlds.region import region_scene

        self.batch = batch
        self.scene = region_scene(weapons=("iron_sword",) * batch, target_active=True)
        self.env, self.policy, _ = jax.jit(self.scene.environment.reset)(
            jax.random.split(jax.random.key(7), self.scene.batch))
        self.step = jax.jit(self.scene.environment.step)
        self.tables, self.effects = audit.registry_tables(), audit.effect_tables()

    def observation(self) -> Any:
        return self.env.learner_observation

    def mask(self) -> jax.Array:
        return self.env.action_mask

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        combat = self.env.runtime.combat
        return np.asarray(combat.position[:, 0], np.float64), np.asarray(combat.yaw[:, 0], np.float64)

    def kw(self) -> dict[str, Any]:
        light = self.env.light_tokens
        return {"tables": self.tables, "effects": self.effects,
                "light": Light(light.token_f32, light.token_mask & light.available[:, None]),
                "action_surface": self.env.action_surface,
                "block_reach_blocks": float(self.scene.maximum_distance)}

    def advance(self, factors: jax.Array, tick: int) -> None:
        self.env, self.policy = self.step(self.env, self.policy, factors,
                                          jax.random.split(jax.random.key(tick), self.batch))[:2]

    def tick_seconds(self) -> float:
        return float(np.asarray(self.env.runtime.combat.last_motion_delta_seconds).reshape(-1)[0])


WORLDS: dict[str, Callable[[int], Any]] = {"duel": Duel, "region": Region}
WORLD_NAMES = ("duel", "region")
"""The worlds every run covers, in order."""


# -- The agent, its captures and the head ---------------------------------------------------------------------------------


class Agent:
    """The fresh v3 trunk (``agent_models.CONFIG``) and the generalist head (``GeneralistConfig()``), from fixed seeds."""

    def __init__(self, batch: int) -> None:
        self.batch = batch
        self.params = init_encoder(jax.random.key(0), am.CONFIG)
        self.trunk = make_trunk(self.params, am.CONFIG)
        self.gconfig = GeneralistConfig()
        self.gparams = init_generalist(jax.random.key(2), self.gconfig, am.CONFIG)
        self.apply = generalist_apply_jit(self.gconfig)

    def start(self) -> layer.AgentState:
        return am.start_state([self.params] * self.batch, self.batch)


class Capture:
    """Records the token sets ``layer.step`` returns. ``state_tick`` calls ``layer.step`` through the module, so the
    wrapper sees exactly the tokens the trunk reads, and it changes nothing else. Inside, jit is off: ``step_lanes``
    compiles its tick (2026-10-24), which would trace the wrapper once and hide the arrays, so the check runs the tick
    eagerly, as it was built. The compiled tick equals the eager one (``test_sa_step_lanes_jit.py``)."""

    def __init__(self) -> None:
        self.tokens: dict | None = None
        self._original = layer.step
        self._eager = jax.disable_jit()

    def __enter__(self) -> "Capture":
        self._eager.__enter__()
        layer.step = self._wrapped
        return self

    def __exit__(self, *exc: Any) -> None:
        layer.step = self._original
        self._eager.__exit__(*exc)

    def _wrapped(self, *args: Any, **kwargs: Any) -> Any:
        tokens, state = self._original(*args, **kwargs)
        self.tokens = tokens
        return tokens, state


def decision_writes() -> list[Write]:
    """The decision layer's writes about the opponent at WRITE_TICK: a ``remember`` tag and a relationship."""

    return [Write("decision", "remember", REMEMBER, entity=OPPONENT)] + [
        Write("decision", name, value, entity=OPPONENT) for name, value in RELATIONSHIP]


def walk_requests(position: np.ndarray, yaw_degrees: np.ndarray, target: jax.Array, slots: int) -> GeneralistRequests:
    """One walk_to request in slot 0: the fixed world point, in the agent's body frame (forward, right, up), in units of
    POSITION_SCALE_BLOCKS, with the position group present. Every other slot is empty."""

    batch = position.shape[0]
    offset = target - jnp.asarray(position, jnp.float32)
    body = world_to_body(offset, jnp.deg2rad(jnp.asarray(yaw_degrees, jnp.float32))) / POSITION_SCALE_BLOCKS
    args = jnp.zeros((batch, slots, ARG_FEATURES), jnp.float32).at[:, 0, 0:3].set(body)
    present = jnp.zeros((batch, slots, len(ARG_GROUPS)), jnp.bool_).at[:, 0, 0].set(True)
    valid = jnp.zeros((batch, slots), jnp.bool_).at[:, 0].set(True)
    return GeneralistRequests(
        verb=jnp.full((batch, slots), VERBS.index("walk_to"), jnp.int32), args=args, arg_present=present,
        entity_pointer=jnp.full((batch, slots), -1, jnp.int32), cell_pointer=jnp.full((batch, slots), -1, jnp.int32),
        scope=jnp.full((batch, slots), SCOPE_NAMES.index(WALK_SCOPE), jnp.int32), valid=valid)


def head_factors(logits: dict[str, jax.Array], mask: jax.Array, scope: str) -> jax.Array:
    """int32[B, 12]: each gym head of the request's scope takes the masked argmax of its logits over its legal choices;
    every other head is neutral (the policy's own neutral choice)."""

    columns = []
    for head, name in enumerate(HEAD_NAMES):
        if scope_of(name) == scope:
            legal = mask[:, HEAD_STARTS[head]:HEAD_STARTS[head] + HEAD_SIZES[head]]
            columns.append(jnp.argmax(jnp.where(legal, logits[name], -jnp.inf), axis=-1).astype(jnp.int32))
        else:
            columns.append(jnp.full((mask.shape[0],), int(NEUTRAL[head]), jnp.int32))
    return jnp.stack(columns, axis=1)


# -- One run of one world ----------------------------------------------------------------------------------------------


class Record:
    """What one run measured, tick by tick, for the checks."""

    def __init__(self) -> None:
        self.width_ok = True
        self.finite_ok = True
        self.trunk_finite = True
        self.head_finite = True
        self.valid_min: dict[str, int] = {}
        self.valid_max: dict[str, int] = {}
        self.memory_changes: list[float] = []
        self.clock: list[np.ndarray] = []
        self.sleeping_ticks = 0
        self.heard: list[np.ndarray] = []
        self.times_ms: list[float] = []
        self.first_tick_ms: float | None = None
        self.applied: np.ndarray | None = None
        self.refusals: list | None = None
        self.write_tokens: dict | None = None
        self.write_entity_ids: np.ndarray | None = None
        self.write_facts: dict | None = None
        self.write_known: list | None = None
        self.last_pose: tuple[np.ndarray, np.ndarray] | None = None
        self.save_load: dict | None = None
        self.load_avg_start = os.getloadavg()[0]
        self.load_avg_end = self.load_avg_start

    def tokens(self, tokens: dict, tick: int, entity_ids: np.ndarray) -> None:
        for name, (features, valid) in tokens.items():
            features, valid = np.asarray(features), np.asarray(valid)
            self.width_ok &= features.shape[-1] == EXPECTED_TOKEN_WIDTHS[name]
            self.finite_ok &= bool(np.isfinite(features).all())
            counts = valid.sum(axis=-1)
            self.valid_min[name] = min(self.valid_min.get(name, 1 << 30), int(counts.min()))
            self.valid_max[name] = max(self.valid_max.get(name, -1), int(counts.max()))


def run_world(name: str, ticks: int, batch: int, agent: Agent, *, record: bool) -> dict[str, Any]:
    """One run: ``ticks`` ticks of the world, the whole layer each tick. ``record`` keeps the measurements (the first
    run); the determinism run only needs its final state."""

    world = WORLDS[name](batch)
    dt = world.tick_seconds()
    start_position, _ = world.pose()
    target = jnp.asarray(start_position + np.asarray(POINT_OFFSET), jnp.float32)
    state = agent.start()
    factors = neutral_arsenal_policy_action_factors(batch)
    rec = Record()
    rec.load_avg_start = os.getloadavg()[0]
    copy = None
    observed = {"constructs": {}}
    for tick in range(ticks):
        obs = world.observation()
        position, yaw = world.pose()
        observed["constructs"] = {"perception_v2": getattr(obs, "perception_v2", None) is not None,
                                  "game_clock": getattr(obs, "game_clock", None) is not None,
                                  "heard_sounds": getattr(obs, "heard_sounds", None) is not None}
        observation, last_action, _, _ = state_tick.host_inputs(
            obs, factors, jnp.zeros((batch,), jnp.bool_), own_position=jnp.asarray(position, jnp.float32),
            own_yaw_degrees=jnp.asarray(yaw, jnp.float32))
        reset = jnp.full((batch,), tick == 0)
        writes = [decision_writes() for _ in range(batch)] if tick == WRITE_TICK else None
        previous = state.latent
        t0 = time.perf_counter()
        with Capture() as capture:
            view, state, refusals = state_tick.step_lanes(agent.trunk, state, observation, last_action=last_action,
                                                          reset=reset, write_lists=writes, tick_seconds=dt,
                                                          **world.kw())
        out = agent.apply(agent.gparams, view.encoded, view.masks,
                          walk_requests(position, yaw, target, agent.gconfig.max_requests))
        factors = head_factors(out.head_logits, world.mask(), WALK_SCOPE)
        jax.block_until_ready((state.latent, factors))
        elapsed = 1000 * (time.perf_counter() - t0)
        if tick == 0 and record:
            rec.first_tick_ms = elapsed
        elif record:
            rec.times_ms.append(elapsed)
        if record:
            entity_ids = np.asarray(view.entity_ids)
            rec.tokens(capture.tokens, tick, entity_ids)
            rec.trunk_finite &= _finite(view.encoded) and _finite(state.drives)
            rec.head_finite &= _finite(out)
            rec.memory_changes.append(float(np.abs(np.asarray(state.latent) - np.asarray(previous)).max()))
            rec.last_pose = (position, yaw)
            clock_reading = getattr(observation, "game_clock", None)
            if clock_reading is not None:
                rec.clock.append(np.asarray(clock_reading.game_seconds_elapsed, np.float64).copy())
            heard = getattr(observation, "heard_sounds", None)
            if heard is not None:
                rec.heard.append(np.asarray(heard.valid).sum(axis=-1))
            self_feature, self_valid = capture.tokens["self"]
            asleep = (np.asarray(self_feature)[..., v3.SelfTokenV3.FEATURES.index("moving_sleeping")] > 0.5)
            rec.sleeping_ticks += int(np.sum(asleep & np.asarray(self_valid)))
            if tick == WRITE_TICK:
                rec.applied = np.asarray(state.written.applied)
                rec.refusals = refusals
                rec.write_tokens = capture.tokens
                rec.write_entity_ids = entity_ids
                rec.write_facts = views.facts(state, 0, own_position=position, body_yaw_degrees=yaw, dt_seconds=dt)
                rec.write_known = state_tick.known_entities(state, jnp.asarray(position, jnp.float32), dt_seconds=dt)
        if tick == SAVE_TICK:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state.npz"
                layer.save(path, state)
                copy = layer.load(path, like=agent.start())
                saved_bytes = path.stat().st_size if path.exists() else -1
            rec.save_load = {"tick": tick, "bytes": saved_bytes}
        elif copy is not None and tick == SAVE_TICK + 1:
            with Capture() as copy_capture:
                view_b, copy, _ = state_tick.step_lanes(agent.trunk, copy, observation, last_action=last_action,
                                                        reset=reset, write_lists=None, tick_seconds=dt, **world.kw())
            rec.save_load.update({
                "token_mismatches": bit_mismatches(capture.tokens, copy_capture.tokens),
                "view_mismatches": bit_mismatches(view, view_b),
                "state_mismatches": bit_mismatches(state, copy)})
            copy = None
        world.advance(factors, tick)
        if record and tick % 50 == 0:
            print(f"{name}: tick {tick} of {ticks}, {elapsed:.0f} ms, load {os.getloadavg()[0]:.0f}", flush=True)
    rec.load_avg_end = os.getloadavg()[0]
    return {"world": world, "state": state, "dt": dt, "start_position": start_position, "record": rec,
            "observed": observed, "final_hash": tree_hash(state)}


# -- The checks ---------------------------------------------------------------------------------------------------------


def _verdict(passed: bool, **detail: Any) -> dict[str, Any]:
    return {"status": STATUS_PASS if passed else STATUS_FAIL, **detail}


def _absent(reason: str) -> dict[str, Any]:
    return {"status": NOT_ON_WORLD, "reason": reason}


def _not_built(reason: str) -> dict[str, Any]:
    return {"status": NOT_BUILT, "reason": reason}


def provenance() -> dict[str, Any]:
    """Where each imported module came from: the worktree for the skill layer; the render for ``hytalegym.jax``."""

    names = ("experimental.neural_agent.skill_agent.state.layer", "experimental.neural_agent.skill_agent.runtime.state_tick",
             "experimental.neural_agent.skill_agent.model.generalist", "experimental.neural_agent.skill_agent.state.views",
             "experimental.neural_agent.skill_agent.bench.encoding_audit", "hytalegym.jax.combat")
    files = {name: str(Path(sys.modules[name].__file__).resolve()) for name in names}
    return {"files": files, "all_skill_layer_under_worktree": all(
        path.startswith(str(ROOT)) for name, path in files.items() if name.startswith("experimental")),
        "hytalegym_jax_is_render": not files["hytalegym.jax.combat"].startswith(str(ROOT)),
        "worktree": str(ROOT)}


def summarize(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """The checks of one world, from its recorded run and the determinism run's final state."""

    rec, state, dt = first["record"], first["state"], first["dt"]
    constructs = first["observed"]["constructs"]
    clocked = bool(constructs["game_clock"])
    checks: dict[str, Any] = {}

    checks["token_health"] = _verdict(rec.width_ok and rec.finite_ok, widths_match_encoder_config=rec.width_ok,
                                      all_values_finite=rec.finite_ok,
                                      valid_counts={n: {"min": rec.valid_min[n], "max": rec.valid_max[n]}
                                                    for n in rec.valid_min})
    checks["trunk_and_head_finite"] = _verdict(rec.trunk_finite and rec.head_finite,
                                               trunk_finite=rec.trunk_finite, head_finite=rec.head_finite)
    changes = rec.memory_changes
    checks["memory_changes_tick_to_tick"] = _verdict(min(changes) > 0, min_change=min(changes), max_change=max(changes))

    if clocked and rec.clock:
        readings = np.stack(rec.clock)
        steps = np.diff(readings, axis=0)
        checks["clock_advances_one_a_tick"] = _verdict(bool(np.allclose(steps, 1.0, rtol=1e-6)),
                                                       first_reading=readings[0].tolist(),
                                                       last_reading=readings[-1].tolist(),
                                                       step_min=float(steps.min()), step_max=float(steps.max()))
        seconds = readings[-1] - readings[0]
        expected = 1.0 - np.exp(-(seconds / 3600.0) / drives.SLEEP_RISE_HOURS)
        actual = np.asarray(state.drives.sleep_pressure)
        checks["sleep_pressure_formula"] = _verdict(bool(np.allclose(actual, expected, rtol=1e-4)) and rec.sleeping_ticks == 0,
                                                    actual=actual.tolist(), expected=expected.tolist(),
                                                    asleep_ticks=rec.sleeping_ticks)
    else:
        checks["clock_advances_one_a_tick"] = _absent("no game_clock on this world's observation")
        checks["sleep_pressure_formula"] = _absent("no game_clock on this world's observation")

    # Hunger is Minecraft's food model and needs no clock (../docs/VIRTUAL_HUMAN.md 20.9).
    food = np.asarray(state.drives.food, np.float64)
    saturation = np.asarray(state.drives.saturation, np.float64)
    exhaustion = np.asarray(state.drives.exhaustion, np.float64)
    checks["food_and_saturation"] = _verdict(
        bool(np.all((food >= 0.0) & (food <= drives.FOOD_MAX) & (saturation >= 0.0) & (saturation <= food))),
        food=food.tolist(), saturation=saturation.tolist())
    checks["exhaustion_stored"] = _verdict(
        bool(np.all((exhaustion >= 0.0) & (exhaustion < drives.EXHAUSTION_PER_POINT))), exhaustion=exhaustion.tolist())
    sprinted = np.asarray(registry.get(state.variables, "distance_sprinted"), np.float64)
    swum = np.asarray(registry.get(state.variables, "distance_swum"), np.float64)
    eaten = np.asarray(registry.get(state.variables, "food_eaten"), np.float64)
    spent = (drives.EXHAUSTION_PER_POINT * ((drives.FOOD_START - food) + (drives.SATURATION_START - saturation))
             + exhaustion)
    counted = drives.SPRINT_EXHAUSTION * sprinted + drives.SWIM_EXHAUSTION * swum
    bound_holds = bool(np.all(spent >= counted - 1e-3)) if not eaten.any() else None
    checks["hunger_bound"] = (_verdict(bound_holds, exhaustion_spent=spent.tolist(), counted=counted.tolist(),
                                       sprinted_blocks=sprinted.tolist(), swum_blocks=swum.tolist(),
                                       note="jumps and hits are not counters, so this is a bound, not equality")
                              if bound_holds is not None else
                              {"status": NOT_RUN, "reason": "food was eaten; meals are not counted as counters"})

    hearing_slice = importlib.util.find_spec("experimental.neural_agent.skill_agent.state.hearing") is not None
    sound_token = "sound" in v3.TOKEN_TYPES_V3
    if constructs["heard_sounds"]:
        published = np.stack(rec.heard) if rec.heard else np.zeros((0, BATCH), int)
        checks["hearing_published_by_simulator"] = _verdict(True, valid_rows_per_tick_min=int(published.min()),
                                                            valid_rows_per_tick_max=int(published.max()),
                                                            total=int(published.sum()))
    else:
        checks["hearing_published_by_simulator"] = _absent("no heard_sounds field on this world's observation")
    checks["hearing_kept_by_state_layer"] = (_not_built("state/hearing.py does not exist in this tree")
                                             if not hearing_slice else {"status": NOT_RUN})
    checks["sound_token"] = (_not_built("no 'sound' token in contract/tokens_v3.py TOKEN_TYPES_V3")
                             if not sound_token else {"status": NOT_RUN})

    applied = rec.applied[:, :3] if rec.applied is not None else None
    refusal = rec.refusals if rec.refusals is not None else None
    people = rec.write_facts["people"]
    person = next((p for p in people if p["id"] == OPPONENT), None)
    tagged = [next((e["tagged"] for e in lane if e["id"] == OPPONENT), None) for lane in rec.write_known]
    rows = np.asarray(rec.write_entity_ids)
    token_rows = []
    for lane in range(BATCH):
        where = np.flatnonzero(rows[lane] == OPPONENT)
        if where.size:
            feature = np.asarray(rec.write_tokens["entity"][0])[lane, where[0]]
            token_rows.append({"rel_known": float(feature[v3.EntityTokenV3.FEATURES.index("rel_known")]),
                               "rel_closeness": float(feature[v3.EntityTokenV3.FEATURES.index("rel_closeness")])})
        else:
            token_rows.append(None)
    relationship_ok = person is not None and person["relationship"] is not None and all(
        abs(person["relationship"][name] - value) < 1e-6 for name, value in RELATIONSHIP)
    checks["decision_writes"] = _verdict(
        bool(applied is not None and applied.all() and refusal is not None and all(r is None for lane in refusal for r in lane)
             and relationship_ok and all(t is not None and t > 0 for t in tagged)),
        applied_first_three_slots=None if applied is None else applied.tolist(), refusals=refusal,
        person_relationship=None if person is None else person["relationship"], tagged=tagged,
        entity_token_rows=token_rows, opponent_id=OPPONENT,
        note="the entity token row is absent when the opponent is not perceived or remembered at the write tick")

    position, yaw = rec.last_pose
    facts = views.facts(state, 0, own_position=position, body_yaw_degrees=yaw, dt_seconds=dt)
    try:
        json.dumps(facts, allow_nan=False)
        serial_ok = True
    except (TypeError, ValueError):
        serial_ok = False
    text = views.words(facts)
    checks["views"] = _verdict(
        serial_ok and bool(text.strip()) and (("It is " in text) == clocked)
        and (f"Entity {OPPONENT} (" in text or f"\n{OPPONENT}:" in text or text.startswith(f"{OPPONENT}:")),
        facts_json_serialisable=serial_ok, words_chars=len(text), clock_sentence=("It is " in text), clock_on_world=clocked,
        opponent_mentioned=bool(f"Entity {OPPONENT} (" in text or f"\n{OPPONENT}:" in text),
        words_head=text[:400])

    if rec.save_load is not None and "state_mismatches" in rec.save_load:
        sl = rec.save_load
        checks["save_and_load"] = _verdict(not (sl["token_mismatches"] or sl["view_mismatches"] or sl["state_mismatches"]),
                                           saved_at_tick=sl["tick"], bytes=sl["bytes"],
                                           token_mismatches=sl["token_mismatches"], view_mismatches=sl["view_mismatches"],
                                           state_leaf_mismatches=sl["state_mismatches"][:20],
                                           state_leaf_mismatch_count=len(sl["state_mismatches"]))
    else:
        checks["save_and_load"] = {"status": NOT_RUN, "reason": "the run ended before tick SAVE_TICK + 1"}

    checks["determinism"] = _verdict(first["final_hash"] == second["final_hash"], first_hash=first["final_hash"],
                                     second_hash=second["final_hash"])

    times = np.asarray(rec.times_ms, np.float64)
    checks["timing_ms_per_tick"] = {
        "status": "measured", "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
        "first_tick_ms_with_compile": rec.first_tick_ms, "ticks_timed": int(times.size), "batch": BATCH,
        "eager": True, "load_average_1m_start": rec.load_avg_start, "load_average_1m_end": rec.load_avg_end,
        "note": "step_lanes run eagerly, not jitted; the generalist is jitted"}
    return {"name": first["world"].name, "constructs_on_observation": constructs, "tick_seconds": dt,
            "start_position": first["start_position"].tolist(), "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="I1: the neural layer on the gen-7 simulator, every construct on.")
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args(argv)
    if options.ticks <= SAVE_TICK + 1:
        raise SystemExit(f"--ticks must exceed {SAVE_TICK + 1}")
    # The hearing construct's footstep tables are an lru_cache of jnp arrays. Filled inside the first jit trace (the
    # reset), they leak a tracer into the step's trace (UnexpectedTracerError on the Region world with hearing on,
    # 2026-10-24: hytalegym/sim/gen7/mechanics/heard_sounds.py _tables). Filling them here, before any trace, avoids it.
    # The module is not changed.
    from hytalegym.sim.gen7.mechanics import heard_sounds
    heard_sounds.footstep_tables_from_env()
    modules = provenance()
    print(json.dumps({"provenance": modules}, indent=1), flush=True)
    agent = Agent(BATCH)
    worlds: dict[str, Any] = {}
    for name in WORLD_NAMES:
        first = run_world(name, options.ticks, BATCH, agent, record=True)
        second = run_world(name, options.ticks, BATCH, agent, record=False)
        worlds[name] = summarize(first, second)
        print(json.dumps({name: {k: v["status"] if isinstance(v, dict) and "status" in v else v
                                 for k, v in worlds[name]["checks"].items()}}), flush=True)
    report = {"ticks": options.ticks, "batch": BATCH, "write_tick": WRITE_TICK, "save_tick": SAVE_TICK,
              "provenance": modules, "worlds": worlds}
    options.out.write_text(json.dumps(report, indent=1, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
