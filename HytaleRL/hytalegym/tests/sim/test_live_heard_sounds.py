"""Live heard sounds (S12, lane C): the launcher's probe flag, the audio v1/v2 parse, the live heard list against the sim,
and the env's append of ``heard_sounds`` after ``game_clock``.

The live list is checked against the gen-7 sim's own ``heard_sounds_fields`` on the same footstep (one entity at a known
offset). The footstep content is read from Assets.zip (HYTALE_ASSETS_ZIP, else the archive the sim tests use).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.audio import AUDIO_SCHEMA, audio_space, empty_audio, parse_audio
from hytalegym.envs.live_heard_sounds import live_heard_sounds
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.world.audio.hearing import footstep_tables, load_footstep_content
from hytalegym.sim.gen7.mechanics import heard_sounds

from .support import ARCHIVES

ASSETS = os.environ.get("HYTALE_ASSETS_ZIP", str(ARCHIVES["gen7"]))
CONTENT = load_footstep_content(ASSETS)
TABLES = footstep_tables(CONTENT)
MOVE_IN = next(record for record in CONTENT.sounds if record.name == "SFX_Bush_MoveIn")
PLANT = "Plant_Bush"
STEP = 1.0 / 30.0
LAUNCHER = Path(__file__).resolve().parents[3] / "hytale-plugin-0.7.2" / "tools" / "launch_bridge_072.py"


@pytest.fixture(autouse=True)
def _assets(monkeypatch):
    monkeypatch.setenv("HYTALE_ASSETS_ZIP", ASSETS)


# ---------------------------------------------------------------------------------------------------------------------
# Helpers: wire frames and the sim's own heard list.
# ---------------------------------------------------------------------------------------------------------------------


def _event(kind=1, index=1, category=0, spatial=True, rel=(5.0, 0.0, 0.0), volume=1.0, pitch=1.0, age=0.0, slot=None):
    fields = [kind, index, category, spatial, *rel, volume, pitch, age]
    return fields if slot is None else [*fields, slot]


def _frame(events, version=2):
    return {
        "schema": AUDIO_SCHEMA,
        "version": version,
        "available": True,
        "capture_mask": [True, True, True],
        "events": events,
    }


def _parse(events, version=2):
    return parse_audio(_frame(events, version), step_duration_seconds=STEP)


class _Caps(NamedTuple):
    target_candidate_perceptible: Any
    target_candidate_perception_valid: Any
    block_under_feet: Any


def _sim_heard(offset=(5.0, 0.0, 0.0), perceptible=True):
    """The sim's heard list: entity 1 stands still-free at ``offset`` from entity 0, both on Plant_Bush; only entity 1
    moves, so the only footstep is entity 1's."""

    position = np.zeros((1, 2, 3), dtype=np.float32)
    position[0, 1] = offset
    velocity = np.zeros((1, 2, 3), dtype=np.float32)
    velocity[0, 1] = (1.0, 0.0, 0.0)
    state = SimpleNamespace(
        combat=SimpleNamespace(
            position=jnp.asarray(position),
            velocity=jnp.asarray(velocity),
            tick_count=jnp.zeros((1,), dtype=jnp.int32),
        )
    )
    flag = jnp.full((1, 2, 2), perceptible, dtype=jnp.bool_)
    caps = _Caps(flag, jnp.ones_like(flag), jnp.full((1, 2), semantic_id(PLANT), dtype=jnp.int32))
    return heard_sounds.heard_sounds_fields(state, caps, heard_sounds.footstep_tables_from_env())


def _curve(distance: float) -> float:
    start, maximum = MOVE_IN.start_distance, MOVE_IN.max_distance
    if distance <= start:
        return 1.0
    return max(0.0, min(1.0, (maximum - distance) / (maximum - start)))


MOVE_IN_TABLE = {1: MOVE_IN.name}


# ---------------------------------------------------------------------------------------------------------------------
# The launcher: the command is unchanged without the probe, and gains two arguments with it.
# ---------------------------------------------------------------------------------------------------------------------


def _launcher():
    spec = importlib.util.spec_from_file_location("launch_bridge_072", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_command(module):
    return module.build_server_command(
        "java",
        Path("/i/Server/HytaleServer.jar"),
        Path("/i/Assets.zip"),
        Path("/run/universe"),
        5556,
        25572,
    )


def test_the_launcher_command_is_unchanged_without_the_probe() -> None:
    assert _base_command(_launcher()) == [
        "java",
        "-Xshare:off",
        "-Dhytalerl.bridge.port=5556",
        "-jar",
        "/i/Server/HytaleServer.jar",
        "--assets",
        "/i/Assets.zip",
        "--disable-sentry",
        "--auth-mode",
        "offline",
        "--bind",
        "127.0.0.1:25572",
        "--universe",
        "/run/universe",
    ]


def test_the_probe_adds_the_two_early_plugin_arguments_last() -> None:
    module = _launcher()
    with_probe = module.build_server_command(
        "java",
        Path("/i/Server/HytaleServer.jar"),
        Path("/i/Assets.zip"),
        Path("/run/universe"),
        5556,
        25572,
        Path("/run/earlyplugins"),
    )
    assert with_probe[:-2] == _base_command(module)
    assert with_probe[-2:] == ["--accept-early-plugins", "--early-plugins=/run/earlyplugins"]


# ---------------------------------------------------------------------------------------------------------------------
# The audio parse: v1 (10 columns) and v2 (11 columns, source_slot), the existing keys and dtypes kept.
# ---------------------------------------------------------------------------------------------------------------------


def test_v1_frames_parse_with_source_slot_minus_one() -> None:
    out = _parse([_event(index=1)], version=1)
    assert out["valid"][15] == 1 and out["valid"][14] == 0
    assert out["source_slot"][15] == -1
    assert out["source_slot"].dtype == np.int8
    assert out["relative_position"][15].tolist() == [5.0, 0.0, 0.0]


def test_v2_frames_carry_the_source_slot_of_each_event() -> None:
    out = _parse([_event(index=1, slot=0), _event(index=1, slot=1), _event(index=1, slot=-1)])
    assert out["source_slot"][13:].tolist() == [0, 1, -1]
    assert out["source_slot"][:13].tolist() == [-1] * 13  # padding


def test_the_parsed_keys_and_dtypes_are_unchanged_plus_source_slot() -> None:
    out = _parse([_event(index=1, slot=1)])
    assert set(out) == set(empty_audio())
    assert set(audio_space().spaces) == set(out)
    for key, value in empty_audio().items():
        assert np.asarray(out[key]).dtype == np.asarray(value).dtype, key


def test_unknown_versions_and_widths_are_rejected() -> None:
    with pytest.raises(ValueError, match="version"):
        _parse([_event(index=1)], version=3)
    with pytest.raises(ValueError, match="version"):
        _parse([_event(index=1)], version=True)
    with pytest.raises(ValueError, match="field"):
        _parse([_event(index=1)], version=2)  # a v2 frame needs 11 fields
    with pytest.raises(ValueError, match="field"):
        _parse([_event(index=1, slot=1)], version=1)  # a v1 frame has 10
    with pytest.raises(ValueError, match="slot"):
        _parse([_event(index=1, slot=2)])


# ---------------------------------------------------------------------------------------------------------------------
# The live heard list against the sim's heard list for the same footstep.
# ---------------------------------------------------------------------------------------------------------------------


def test_a_live_footstep_matches_the_sim_fields_and_gain() -> None:
    offset = 5.0
    live = live_heard_sounds(_parse([_event(index=1, rel=(offset, 0.0, 0.0), slot=1)]), MOVE_IN_TABLE, TABLES, True)
    sim = _sim_heard((offset, 0.0, 0.0), perceptible=True)
    slot = next(k for k in range(16) if bool(sim.valid[0, k]) and float(sim.distance[0, k]) > 0.0)

    assert bool(live.valid[0, 0]) and int(np.sum(live.valid)) == 1
    assert int(live.sound_id[0, 0]) == int(sim.sound_id[0, slot]) == MOVE_IN.sound_id
    np.testing.assert_allclose(live.distance[0, 0], sim.distance[0, slot], rtol=1e-6)
    np.testing.assert_allclose(live.relative_position[0, 0], sim.relative_position[0, slot], rtol=1e-6)
    np.testing.assert_allclose(live.gain[0, 0], sim.gain[0, slot], rtol=1e-5)
    np.testing.assert_allclose(live.gain[0, 0], MOVE_IN.volume_linear * _curve(offset), rtol=1e-5)
    assert int(live.source_entity_id[0, 0]) == int(sim.source_entity_id[0, slot]) == 1
    assert float(live.age_seconds[0, 0]) == 0.0


def test_non_footstep_non_3d_and_out_of_table_events_are_dropped() -> None:
    audio = _parse(
        [
            _event(kind=0, spatial=False, rel=(0.0, 0.0, 0.0), index=1, slot=-1),  # 2-D footstep
            _event(index=2, slot=1),  # a sound that is not a footstep in the tables
            _event(index=3, slot=1),  # an index the sound table does not know
            _event(index=1, rel=(20.0, 0.0, 0.0), slot=1),  # past the footstep's MaxDistance
            _event(index=1, rel=(4.0, 0.0, 0.0), slot=1),  # the one heard footstep
        ],
        version=2,
    )
    heard = live_heard_sounds(audio, {1: MOVE_IN.name, 2: "SFX_Not_A_Footstep"}, TABLES, True)
    assert int(np.sum(heard.valid)) == 1
    assert float(heard.distance[0, 0]) == pytest.approx(4.0)


def test_source_slots_map_with_perceptible_on_and_off() -> None:
    events = [_event(index=1, rel=(3.0, 0.0, 0.0), slot=0), _event(index=1, rel=(4.0, 0.0, 0.0), slot=1),
              _event(index=1, rel=(2.0, 0.0, 0.0), slot=-1)]
    audio = _parse(events)
    on = live_heard_sounds(audio, MOVE_IN_TABLE, TABLES, True)
    off = live_heard_sounds(audio, MOVE_IN_TABLE, TABLES, False)
    assert on.source_entity_id[0, :3].tolist() == [0, 1, -1]
    assert off.source_entity_id[0, :3].tolist() == [0, -1, -1]


def test_the_list_pads_to_16_slots_and_is_all_invalid_without_events_or_table() -> None:
    audio = _parse([_event(index=1, rel=(float(d), 0.0, 0.0), slot=1) for d in (2, 3, 4)])
    heard = live_heard_sounds(audio, MOVE_IN_TABLE, TABLES, True)
    assert heard.valid.shape == (1, 16) and heard.relative_position.shape == (1, 16, 3)
    assert heard.valid[0].tolist() == [True] * 3 + [False] * 13
    assert heard.sound_id[0, 3:].tolist() == [0] * 13
    assert heard.source_entity_id[0, 3:].tolist() == [-1] * 13
    assert heard.gain[0, 3:].tolist() == [0.0] * 13
    # Before the sound table arrives every slot is invalid.
    empty = live_heard_sounds(audio, {}, TABLES, True)
    assert not np.any(empty.valid) and empty.valid.shape == (1, 16)


# ---------------------------------------------------------------------------------------------------------------------
# The env: the switch, and the append of heard_sounds last, after game_clock.
# ---------------------------------------------------------------------------------------------------------------------


def test_the_live_env_switch_needs_a_bool_and_the_learner_observation() -> None:
    from hytalegym.envs.hytale_env import HytaleEnv

    with pytest.raises(TypeError):
        HytaleEnv(learner_heard_sounds="yes")
    with pytest.raises(ValueError):
        HytaleEnv(learner_heard_sounds=True)  # no learner_v3_profile: nothing to append to


def test_the_live_env_appends_heard_sounds_last_after_game_clock() -> None:
    from hytalegym.envs.hytale_env import HytaleEnv
    from hytalegym.envs.live_game_clock import LiveGameClock
    from hytalegym.jax.combat.observation.v3.native.evidence import NativeActorEvidenceAssembly

    class Observation(NamedTuple):
        valid: int

    assembly = NativeActorEvidenceAssembly(
        evidence=None,
        observation=Observation(1),
        policy_observation=np.zeros(3, dtype=np.float32),
        policy_action_mask=np.ones(2, dtype=np.bool_),
        world_capabilities=None,
        light_tokens=None,
        actor_yaw_degrees=0.0,
        raw_wire={"actors": [{"present": True, "perceptible": True}, {"present": True, "perceptible": True}]},
    )
    env = SimpleNamespace(_live_game_clock=LiveGameClock(), _sound_event_loader=None, _footstep_tables=TABLES)
    clocked = HytaleEnv._with_live_game_clock(env, assembly, {"time_of_day": 5500})
    audio = _parse([_event(index=1, rel=(5.0, 0.0, 0.0), slot=1)])

    # The table has not arrived: heard_sounds is present, last, and all invalid.
    before = HytaleEnv._with_live_heard_sounds(env, clocked, audio)
    assert before.observation._fields == ("valid", "game_clock", "heard_sounds")
    assert not np.any(before.observation.heard_sounds.valid)

    # The table has arrived: the footstep is heard from the target, and the clock is kept.
    env._sound_event_loader = SimpleNamespace(asset_ids=MOVE_IN_TABLE)
    after = HytaleEnv._with_live_heard_sounds(env, clocked, audio)
    assert after.observation._fields == ("valid", "game_clock", "heard_sounds")
    assert after.observation.game_clock is clocked.observation.game_clock
    heard = after.observation.heard_sounds
    assert bool(heard.valid[0, 0]) and int(heard.sound_id[0, 0]) == MOVE_IN.sound_id
    assert int(heard.source_entity_id[0, 0]) == 1
    np.testing.assert_array_equal(after.policy_observation, assembly.policy_observation)
