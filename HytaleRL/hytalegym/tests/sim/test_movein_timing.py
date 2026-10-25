"""Gen-7 entry-timed footsteps (switch ``Constructs.hearing_entry``, default False, acting only with ``hearing``).

``sim/gen7/mechanics/movein_timing.py`` implements the native TickMovementAudio rule (FINDINGS in the lane folder
``lane_movein_rule``, sections 2 and 5): MoveIn on a change of the feet cell, a repeat timer drawn in [0.5, 1.5] s, and a
countdown only while the body moves. One optional CombatState field carries the state (``footstep``, None without the switch, a
``FootstepState`` of cell, timer and emit with it); the draw is a hash of (row, entity, tick), so no key leaf is added.

Unit tests drive ``footstep_update`` on a real reset state of a render with the field (the arithmetic, no duel). Integration
tests run the duel through the launcher (the duel imports the gen-5 tree, so it runs only in a launcher process): the switch-off
identity against a render with the new rows removed (HEAD's overlay), and the switch-on duel's footstep timing.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen7 import overlay as gen7_overlay
from hytalegym.sim.gen7.mechanics import movein_timing

from .support import ARCHIVES, HYTALEGYM, REPOSITORY, mount_fresh, render_for_test, unmount

ASSETS = os.environ.get("HYTALE_ASSETS_ZIP", str(ARCHIVES["gen7"]))
BUSH = "Plant_Bush"
GRASS = "Soil_Grass"
BUSH_MOVE_IN = "SFX_Bush_MoveIn"
DT = 1.0 / 30.0
ENTITIES = 2


@pytest.fixture(autouse=True)
def _assets(monkeypatch):
    monkeypatch.setenv("HYTALE_ASSETS_ZIP", ASSETS)


# ---------------------------------------------------------------------------------------------------------------------
# Unit: footstep_update on a real reset state.
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    out = tmp_path_factory.mktemp("movein_unit") / "gen7"
    out = render_for_test(gen7_overlay.OVERLAY, out, source=HeadSource())
    package = mount_fresh(out)
    combat = importlib.import_module(package.__name__ + ".combat")
    arsenal = importlib.import_module(package.__name__ + ".combat.arsenal")
    params = combat.default_combat_params(microticks=1, target_active=False)
    loadout = arsenal.hytale_0_5_7_loadouts("iron_sword", target_profiles="iron_sword")
    config = arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=params.sensor_range)
    yield SimpleNamespace(package=package.__name__, arsenal=arsenal, params=params, config=config)
    unmount(package.__name__)


def _reset(engine):
    position = jnp.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]]], jnp.float32)
    health = jnp.asarray([[engine.params.agent_max_health, engine.params.target_max_health]], jnp.float32)
    yaw = jnp.zeros((1, 2), jnp.float32)
    state, _ = engine.arsenal.reset_arsenal_batch(
        jax.random.split(jax.random.key(570057), 1), engine.params, engine.config,
        entity_position=position, entity_health=health, entity_yaw=yaw,
    )
    # The reset hook sets the field only in a render with the switch on; the unit render is the default one.
    footstep = movein_timing.initial_footstep(1, ENTITIES)
    return state._replace(combat=state.combat._replace(footstep=footstep))


def _run(engine, cells, moving, *, start_tick: int = 0):
    """Drive footstep_update one tick at a time. ``cells[t]`` is the feet block id of each entity at tick t (an int)."""

    state = _reset(engine)
    record = []
    for index, cell in enumerate(cells):
        tick = start_tick + index
        speed = 1.0 if moving[index] else 0.0
        velocity = jnp.zeros((1, ENTITIES, 3), jnp.float32).at[:, :, 2].set(speed)
        combat = state.combat._replace(velocity=velocity, tick_count=jnp.full((1,), tick, jnp.int32))
        state = state._replace(combat=combat)
        world = SimpleNamespace(block_under_feet=jnp.full((1, ENTITIES), semantic_id(cell), jnp.int32))
        state = movein_timing.footstep_update(state, world, engine.params)
        record.append((np.asarray(state.combat.footstep.emit)[0].tolist(),
                       np.asarray(state.combat.footstep.timer)[0].tolist()))
    return record


def _bush_id() -> int:
    return semantic_id(BUSH_MOVE_IN)


def _emit_ticks(record, entity: int = 0) -> list[int]:
    return [index for index, (emit, _) in enumerate(record) if emit[entity] != 0]


def test_entering_a_bush_emits_one_move_in_on_the_entry_tick(engine) -> None:
    cells = [GRASS] * 5 + [BUSH] * 3
    record = _run(engine, cells, moving=[True] * 8)
    assert _emit_ticks(record) == [5]  # the entry tick only: the first repeat is 14 to 44 ticks later
    assert record[5][0][0] == _bush_id() and record[5][0][1] == _bush_id()
    assert all(record[i][0] == [0, 0] for i in range(5))


def test_the_first_repeat_comes_within_the_native_window_while_moving(engine) -> None:
    cells = [GRASS] + [BUSH] * 80
    record = _run(engine, cells, moving=[True] * 81)
    repeats = [t for t in _emit_ticks(record) if t > 1]
    assert repeats, "no repeat while moving on a bush"
    first = repeats[0] - 1  # ticks after the entry tick
    # The draw is in [0.5, 1.5) s, so the first repeat is 14 to 44 ticks after entry (the arming tick also counts down).
    assert 14 <= first <= 44, first
    # Every later repeat is again drawn inside the window.
    gaps = np.diff([1] + repeats)
    assert all(14 <= gap <= 45 for gap in gaps), gaps.tolist()


def test_standing_still_after_entry_gives_no_repeat(engine) -> None:
    cells = [GRASS] + [BUSH] * 80
    record = _run(engine, cells, moving=[False] * 81)
    assert _emit_ticks(record) == [1]  # the entry only: the countdown runs while moving
    assert all(timer[0] == -1.0 or timer[0] > 0.0 for _, timer in record)


def test_leaving_the_bush_onto_grass_gives_no_move_in_and_disarms(engine) -> None:
    cells = [GRASS] + [BUSH] * 3 + [GRASS] * 60
    record = _run(engine, cells, moving=[True] * 64)
    assert _emit_ticks(record) == [1]  # the entry; the exit onto grass plays nothing (Bush has no MoveOut)
    assert all(timer[0] == -1.0 for _, timer in record[4:])


def test_the_draw_is_deterministic_and_inside_the_native_range() -> None:
    ticks = jnp.arange(0, 3000, dtype=jnp.int32)
    values = np.asarray(movein_timing.draw_seconds(3000, ENTITIES, ticks))
    assert values.shape == (3000, ENTITIES)
    assert float(values.min()) >= 0.5 and float(values.max()) < 1.5
    # Spread over the range (a constant draw would fail): the mean of 6000 draws is near the midpoint 1.0.
    assert abs(float(values.mean()) - 1.0) < 0.05
    first = np.asarray(movein_timing.draw_seconds(1, ENTITIES, jnp.asarray([7], jnp.int32)))
    again = np.asarray(movein_timing.draw_seconds(1, ENTITIES, jnp.asarray([7], jnp.int32)))
    assert np.array_equal(first, again)


# ---------------------------------------------------------------------------------------------------------------------
# Integration: the duel through the launcher.
# ---------------------------------------------------------------------------------------------------------------------

PROBE = r'''
"""Duel probe run by the launcher: ``probe.py FLOOR STEPS WALK``. Prints one ``PROBE {json}`` line."""

import hashlib
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.tasks import duel_arena as duel
from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors

floor = sys.argv[1]
steps = int(sys.argv[2])
walk_on = sys.argv[3] == "1"
arena = duel.make_duel_arena(1, ("iron_sword", "iron_sword"), view_sector_full_angle_radians=None,
                             legal_observation_sensor_range=True, floor_block=None if floor == "none" else floor)
position = jnp.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]]], jnp.float32)
health = jnp.full((1, 2), 100.0, jnp.float32)
yaw = jnp.full((1, 2), 180.0, jnp.float32)
state, obs0, obs1, _, _ = arena.reset(jax.random.split(jax.random.key(0), 1), entity_position=position,
                                       entity_health=health, entity_yaw=yaw)
neutral = neutral_arsenal_policy_action_factors(1).astype(jnp.int32)
walk = neutral.at[:, 4].set(1) if walk_on else neutral


def digest(tree):
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        h.update(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
    return h.hexdigest()


report = {"obs": [], "combat": [], "footstep": []}


def snapshot(state, obs0, obs1):
    combat = state.arsenal.combat
    report["obs"].append([digest(obs0), digest(obs1)])
    report["combat"].append(digest({name: getattr(combat, name) for name in combat._fields if name != "footstep"}))
    footstep = getattr(combat, "footstep", None)  # absent in HEAD's render
    if footstep is not None:
        report["footstep"].append([np.asarray(footstep.emit)[0].tolist(),
                                   np.asarray(footstep.timer)[0].round(4).tolist()])


snapshot(state, obs0, obs1)
for tick in range(1, steps + 1):
    state, obs0, obs1, _, _, _, _ = arena.step(state, walk, walk, jax.random.split(jax.random.key(tick), 1))
    snapshot(state, obs0, obs1)
print("PROBE " + json.dumps(report))
'''


def _probe(work: Path, engine_dir: Path, constructs: tuple[str, ...], floor: str, steps: int, walk: bool) -> dict:
    probe = work / f"probe_{floor}_{len(constructs)}_{steps}_{int(walk)}.py"
    probe.write_text(PROBE, encoding="utf-8")
    command = [sys.executable, "-m", "experimental.neural_agent.skill_agent.runtime.gen7", "--engine", str(engine_dir)]
    for name in constructs:
        command += ["--construct", name]
    command += ["--", str(probe), floor, str(steps), "1" if walk else "0"]
    env = dict(
        os.environ,
        PYTHONPATH=f"{REPOSITORY}:{HYTALEGYM}",
        PYTHONDONTWRITEBYTECODE="1",
        JAX_PLATFORMS="cpu",
        HYTALERL_SERVER_VERSION="0.7.2",
        HYTALE_ASSETS_ZIP=ASSETS,
    )
    done = subprocess.run(command, cwd=REPOSITORY, env=env, capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stderr[-4000:]
    lines = [line for line in done.stdout.splitlines() if line.startswith("PROBE ")]
    assert lines, done.stdout[-2000:]
    return json.loads(lines[-1][len("PROBE "):])


@pytest.fixture(scope="module")
def renders(tmp_path_factory):
    """The new render, and HEAD's overlay (the same overlay with the movein rows removed) for the identity baseline."""

    root = tmp_path_factory.mktemp("movein_renders")
    new = render_for_test(gen7_overlay.OVERLAY, root / "new" / "gen7", source=HeadSource())
    head_overlay = replace(
        gen7_overlay.OVERLAY,
        patches=tuple(p for p in gen7_overlay.PATCHES if not p.id.startswith("movein_")),
        hooks=tuple(h for h in gen7_overlay.HOOKS if not h.id.startswith("movein_")),
    )
    head = render_for_test(head_overlay, root / "head" / "gen7", source=HeadSource())
    return SimpleNamespace(new=new, head=head, new_manifest=R.read_manifest(new), head_manifest=R.read_manifest(head))


def test_the_render_applies_the_three_leaves_and_two_hooks_only_in_the_new_render(renders) -> None:
    ids = {entry["id"] for entry in renders.new_manifest["applied"]}
    assert {"movein_field", "movein_reset", "movein_step"} <= ids
    assert not any(str(entry["id"]).startswith("movein_") for entry in renders.head_manifest["applied"])


@pytest.fixture(scope="module")
def identity(renders, tmp_path_factory):
    work = tmp_path_factory.mktemp("movein_identity")
    return SimpleNamespace(
        hearing_on_head=_probe(work, renders.head, ("hearing",), BUSH, 50, True),
        hearing_on_new_off=_probe(work, renders.new, ("hearing",), BUSH, 50, True),
        default_head=_probe(work, renders.head, (), "none", 50, True),
        default_new_off=_probe(work, renders.new, (), "none", 50, True),
    )


def test_switch_off_the_duel_observation_and_combat_state_match_head_over_50_steps_with_hearing(identity) -> None:
    assert identity.hearing_on_new_off["obs"] == identity.hearing_on_head["obs"]
    assert identity.hearing_on_new_off["combat"] == identity.hearing_on_head["combat"]
    assert len(identity.hearing_on_head["obs"]) == 51


def test_switch_off_the_default_duel_matches_head_over_50_steps(identity) -> None:
    assert identity.default_new_off["obs"] == identity.default_head["obs"]
    assert identity.default_new_off["combat"] == identity.default_head["combat"]


@pytest.fixture(scope="module")
def switched_on(renders, tmp_path_factory):
    work = tmp_path_factory.mktemp("movein_on")
    return _probe(work, renders.new, ("hearing", "hearing_entry"), BUSH, 120, True)


def test_switch_on_the_duel_plays_the_entry_then_repeats_inside_the_window(switched_on) -> None:
    for entity in (0, 1):
        ticks = [index for index, (emit, _) in enumerate(switched_on["footstep"]) if emit[entity] != 0]
        assert ticks[0] == 1, ticks[:3]  # the first step after reset: the reset cell is air, the floor is a bush
        gaps = np.diff(ticks)
        assert gaps.size and all(14 <= gap <= 45 for gap in gaps), gaps.tolist()
