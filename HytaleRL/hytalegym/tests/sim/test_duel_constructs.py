"""The gen-7 constructs on the two-body duel (``experimental/neural_agent/tasks/duel_arena.py``).

The duel's observation comes from ``observe_policy_actors_flat``, which never calls the encoder. The arena therefore
encodes each policy slot's actor-first view with the encoder the engine exposes; when a construct wraps that encoder
(``__wrapped__``, detected once at ``make_duel_arena``), the fields the wrapper adds are carried on each seat. The render
is unchanged: it has no ``training/`` module. The duel imports ``hytalegym.jax`` from the gen-5 tree, so it runs on a
mounted render only through the launcher (``experimental/neural_agent/skill_agent/runtime/gen7.py``, ``--construct``
switches constructs on in its process). The duel runs happen in launcher subprocesses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen7 import overlay as gen7_overlay
from hytalegym.sim.gen7.mechanics import game_clock, heard_sounds, perception_v2
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import ARCHIVES, HYTALEGYM, REPOSITORY, render_for_test

ASSETS = os.environ.get("HYTALE_ASSETS_ZIP", str(ARCHIVES["gen7"]))
FLOOR = "Plant_Bush"
CONSTRUCTS = ("game_clock", "hearing", "perception_v2")

PROBE = r'''
"""Duel probe run by the launcher: ``probe.py [FLOOR_BLOCK | none]``. Prints one ``PROBE {json}`` line."""

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.tasks import duel_arena as duel
from hytalegym.jax.combat import arsenal_runtime_config, default_combat_params, hytale_0_5_7_loadouts
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.observation import empty_injected_world_features
from hytalegym.jax.combat.observation.v3.encoding.encoder import encode_learner_observation_v3
from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors
from hytalegym.jax.training.multi_actor.views import actor_first_arsenal_view

floor = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "none" else None
floor_id = None if floor is None else semantic_id(floor)
arena = duel.make_duel_arena(1, ("iron_sword", "iron_sword"), view_sector_full_angle_radians=None,
                             legal_observation_sensor_range=True, floor_block=floor)
position = jnp.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]]], jnp.float32)
health = jnp.full((1, 2), 100.0, jnp.float32)
yaw = jnp.full((1, 2), 180.0, jnp.float32)  # both face +z: a walk forward keeps them 4 blocks apart
state, obs0, obs1, _, _ = arena.reset(jax.random.split(jax.random.key(0), 1), entity_position=position,
                                       entity_health=health, entity_yaw=yaw)
walk = neutral_arsenal_policy_action_factors(1).astype(jnp.int32).at[:, 4].set(1)  # locomotion head: gait 1, direction 1
bush = semantic_id("SFX_Bush_MoveIn")

report = {"fields": [list(obs0._fields)], "heard": {"0": [], "1": []}, "speed": []}
for tick in range(1, 61):
    state, obs0, obs1, _, _, _, _ = arena.step(state, walk, walk, jax.random.split(jax.random.key(tick), 1))
    velocity = np.asarray(state.arsenal.combat.velocity)[0]  # [entity, xyz]
    report["speed"].append([float(np.hypot(velocity[entity, 0], velocity[entity, 2])) for entity in range(2)])
    for seat, obs in (("0", obs0), ("1", obs1)):
        heard = getattr(obs, "heard_sounds", None)
        if heard is None:
            continue
        rows = [[bool(heard.valid[0, k]), int(heard.sound_id[0, k]), int(heard.source_entity_id[0, k]),
                 float(heard.gain[0, k])] for k in range(heard.valid.shape[1]) if bool(heard.valid[0, k])]
        report["heard"][seat].append(rows)

# The single-learner encoder on the same state: entity 0 as the learner, entity 1 on its actor-first view.
params = default_combat_params(microticks=1)._replace(target_active=jnp.asarray(False, dtype=jnp.bool_))
config = arsenal_runtime_config(hytale_0_5_7_loadouts(["iron_sword"], target_profiles=["iron_sword"]),
                                opponent_controller_mask=jnp.zeros((1, 2), dtype=jnp.bool_), legal_observation=True)
caps = duel._capabilities(state.arsenal, state.locomotion, params, config, sector=None, sensor_range=True,
                          floor_id=floor_id)
features = empty_injected_world_features(1)
view_state, view_config, view_caps = actor_first_arsenal_view(
    state.arsenal, config, caps, state.locomotion, jnp.ones((1,), jnp.int32))
encoded0 = encode_learner_observation_v3(state.arsenal, params, features, caps, config)
encoded1 = encode_learner_observation_v3(view_state, params, features, view_caps, view_config)


def same(left, right):
    if left is None or right is None:
        return left is None and right is None
    pairs = zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in pairs)


report["fields"].append(list(encoded0._fields))
for name in ("game_clock", "perception_v2"):
    report[name] = {
        "seat0": same(getattr(obs0, name, None), getattr(encoded0, name, None)),
        "seat1": same(getattr(obs1, name, None), getattr(encoded1, name, None)),
    }
print("PROBE " + json.dumps(report))
'''


@pytest.fixture(autouse=True)
def _assets(monkeypatch):
    monkeypatch.setenv("HYTALE_ASSETS_ZIP", ASSETS)


@pytest.fixture(scope="module")
def render_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("duel_render") / "gen7"
    out = render_for_test(gen7_overlay.OVERLAY, out, source=HeadSource())
    return out, R.read_manifest(out)


def _run_probe(tmp_path: Path, engine: Path, constructs: tuple[str, ...], floor: str) -> dict:
    """One duel probe in a launcher process with ``constructs`` switched on; returns its report."""

    probe = tmp_path / f"probe_{floor}_{len(constructs)}.py"
    probe.write_text(PROBE, encoding="utf-8")
    command = [sys.executable, "-m", "experimental.neural_agent.skill_agent.runtime.gen7", "--engine", str(engine)]
    for name in constructs:
        command += ["--construct", name]
    command += ["--", str(probe), floor]
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
def probes(render_dir, tmp_path_factory):
    engine, _ = render_dir
    work = tmp_path_factory.mktemp("duel_probes")
    return SimpleNamespace(
        off=_run_probe(work, engine, (), "none"),
        on=_run_probe(work, engine, CONSTRUCTS, "none"),
        floor=_run_probe(work, engine, CONSTRUCTS, FLOOR),
        perception=_run_probe(work, engine, ("perception_v2",), "none"),
    )


# ---------------------------------------------------------------------------------------------------------------------
# The switch and the render.
# ---------------------------------------------------------------------------------------------------------------------


def test_the_switches_are_false_in_the_defaults() -> None:
    assert Constructs().game_clock is False and Constructs().hearing is False and Constructs().perception_v2 is False
    assert GEN7.constructs.game_clock is False and GEN7.constructs.hearing is False
    assert GEN7.constructs.perception_v2 is False


def test_the_gen7_render_has_no_training_modules_and_the_baseline_module_count(render_dir) -> None:
    out, manifest = render_dir
    assert not (out / "training").exists()
    assert manifest["totals"]["modules"] == 398  # the module count before the duel's construct path (2026-10-24)
    assert not any(entry["path"].startswith("training/") for entry in manifest["applied"])


def test_the_encoder_wrappers_set_wrapped_and_the_bare_encoder_does_not() -> None:
    from hytalegym.jax.combat.observation.v3.encoding.encoder import encode_learner_observation_v3

    def bare(*args, **kwargs):
        return None

    assert not hasattr(encode_learner_observation_v3, "__wrapped__")
    for module in (game_clock, heard_sounds, perception_v2):
        construct = module.construct
        profile = SimpleNamespace(
            constructs=SimpleNamespace(game_clock=True, hearing=True, perception_v2=True),
            step_duration=SimpleNamespace(nominal_seconds=1 / 30),
        )
        assert hasattr(construct(bare, profile), "__wrapped__"), module.__name__


# ---------------------------------------------------------------------------------------------------------------------
# The duel, switch off and on.
# ---------------------------------------------------------------------------------------------------------------------


def test_switch_off_the_duel_observation_carries_no_construct_field(probes) -> None:
    fields = probes.off["fields"][0]
    assert "game_clock" not in fields and "heard_sounds" not in fields and "perception_v2" not in fields


def test_switch_on_each_seat_game_clock_equals_the_single_learner_encoder(probes) -> None:
    # The seats carry the same three added fields as the learner, in the learner's order.
    assert sorted(probes.on["fields"][0][-3:]) == sorted(["perception_v2", "game_clock", "heard_sounds"])
    assert probes.on["fields"][0][-3:] == probes.on["fields"][1][-3:]
    assert probes.on["game_clock"] == {"seat0": True, "seat1": True}


def test_switch_on_each_seat_perception_v2_equals_the_single_learner_encoder(probes) -> None:
    # Seat 1 is compared with the encoder on entity 1's actor-first view.
    assert probes.on["perception_v2"] == {"seat0": True, "seat1": True}


def test_perception_v2_alone_is_added_to_each_seat_and_equals_the_encoder(probes) -> None:
    # Only the perception wrapper is on: the arena finds it through __wrapped__ and the seats still carry the channel.
    assert probes.perception["fields"][0][-1:] == ["perception_v2"]
    assert probes.perception["perception_v2"] == {"seat0": True, "seat1": True}


def test_hearing_with_no_floor_block_hears_nothing_on_either_seat(probes) -> None:
    assert all(not row for seat in ("0", "1") for row in probes.on["heard"][seat])


def test_hearing_on_a_bush_floor_hears_each_other_entitys_footsteps(probes) -> None:
    # Both bodies move at every tick (speed above the 1e-3 threshold at the repeat ticks 30 and 60).
    assert all(speed > 1e-3 for pair in probes.floor["speed"][29:30] + probes.floor["speed"][59:60] for speed in pair)
    bush = _bush_id()
    for seat in ("0", "1"):
        heard = [entry for tick in probes.floor["heard"][seat] for entry in tick]
        # The other entity is view slot 1 on either seat (each seat is entity 0 of its own view).
        assert any(sound == bush and source == 1 and gain > 0.0 for _, sound, source, gain in heard), seat
        # Each seat's own footsteps are view slot 0 at distance zero.
        assert any(sound == bush and source == 0 for _, sound, source, _ in heard), seat


def test_switch_on_with_a_bush_floor_keeps_the_clock_and_channel_per_seat(probes) -> None:
    assert probes.floor["game_clock"] == {"seat0": True, "seat1": True}
    assert probes.floor["perception_v2"] == {"seat0": True, "seat1": True}


def _bush_id() -> int:
    from hytalegym.jax.combat.contracts.semantic import semantic_id

    return int(semantic_id("SFX_Bush_MoveIn"))
