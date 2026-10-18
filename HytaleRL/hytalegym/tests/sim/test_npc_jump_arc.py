"""NPC jump arc construct (gen 7, row 1c): gen 5 keeps g over the current tick; gen 7 integrates a jump's vertical
speed the way the server does (previous tick's duration, PhysicsMath cubic drag on the speed gained since takeoff)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.geometry.contract import FLAG_OPAQUE, FLAG_SOLID, cell_index, empty_geometry
from hytalegym.jax.combat import (
    ACTION_JUMP,
    ACTION_SIZE,
    AGENT_ENTITY,
    default_combat_params,
    reset_batch_geometry_at,
    step_batch_geometry,
)
from hytalegym.jax.combat import env as combat_env
from hytalegym.jax.combat.runtime import agent as runtime_agent
from hytalegym.jax.world import geometry_state_from_numpy
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import npc_jump_arc
from hytalegym.sim.gen7.profile import PROFILE as GEN7

# A live 0.7.2 jump in place (native_motion_differential --jump-steps 1, Kweebec_Razorleaf, bridge 686BC4BC,
# 2026-10-17): the tick durations the server stepped, then the NPC's height and vertical speed after each.
DTS = (0.034555916, 0.034722418, 0.034607332, 0.034217793, 0.034585375, 0.034543622, 0.034715168, 0.033333331,
       0.033333331, 0.033333335, 0.033333331)
NATIVE_Y = (65.1762013, 65.3148559, 65.4145982, 65.4753239, 65.4988325, 65.4840835, 65.4308895, 65.3427873,
            65.2191355, 65.0599369, 65.0)
NATIVE_VY = (5.09902, 3.99323, 2.882113, 1.774683, 0.679728, -0.426969, -1.532298, -2.643067, -3.709555, -4.775958,
             -1.798106)


def test_gen5_returns_both_originals() -> None:
    assert npc_jump_arc.capture_previous_delta(combat_env._microtick, GEN5) is combat_env._microtick
    original = runtime_agent._tick_agent_geometry_motion
    assert npc_jump_arc.fall_speed_integration(original, GEN5) is original


def test_terminal_speed_of_the_learner_model() -> None:
    # Kweebec_Sapling: 0.6 x 0.6 hitbox, no PhysicsValues (ModelAsset default mass 68, drag 0.5).
    assert npc_jump_arc.terminal_speed(0.6, 0.6) == pytest.approx(140.488, abs=1e-3)


def _floor() -> dict:
    frame = empty_geometry()
    frame["available"] = 1
    frame["exact_collision_shapes"] = 1
    frame["origin"][:] = [0, 65, 0]
    frame["agent_bounds"][:] = [-0.3, 0.0, -0.3, 0.3, 1.4, 0.3]
    frame["target_bounds"][:] = [-0.5, 0.0, -0.5, 0.5, 2.1, 0.5]
    frame["agent_los_offset"][:] = [0.0, 0.9, 0.0]
    frame["target_los_offset"][:] = [0.0, 0.9, 0.0]
    floor = cell_index(0, -1, 0)
    frame["cell_mask"][floor] = 1
    frame["flags"][floor] = FLAG_SOLID | FLAG_OPAQUE
    frame["collision_boxes"][floor, 0] = [0, 0, 0, 1, 1, 1]
    frame["collision_box_mask"][floor, 0] = 1
    return frame


def _jump_in_place() -> tuple[np.ndarray, np.ndarray]:
    geometry = geometry_state_from_numpy(_floor())
    params = default_combat_params(microticks=1, target_active=False, motion_model="npc_walk")
    position = np.asarray([[0.5, 65.0, 0.5]], dtype=np.float32)
    state, _ = reset_batch_geometry_at(
        jax.random.split(jax.random.PRNGKey(0), 1),
        params,
        geometry,
        position,
        position + np.asarray(params.target_offset, dtype=np.float32),
    )
    actions = jnp.zeros((len(DTS), 1, ACTION_SIZE), dtype=jnp.float32).at[0, 0, ACTION_JUMP].set(1.0)
    keys = jax.random.split(jax.random.PRNGKey(1), len(DTS)).reshape(len(DTS), 1, 2)

    def step(current, inputs):
        action, key, dt = inputs
        following, *_ = step_batch_geometry(current, action, key, params._replace(nominal_dt=dt, loaded_dt=dt), geometry)
        return following, (following.position[0, AGENT_ENTITY, 1], following.velocity[0, AGENT_ENTITY, 1])

    _, (y, vy) = jax.jit(lambda s, a, k, d: jax.lax.scan(step, s, (a, k, d)))(
        state, actions, keys, jnp.asarray(DTS, dtype=jnp.float32)
    )
    return np.asarray(y), np.asarray(vy)


def test_gen7_jump_in_place_matches_the_live_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(combat_env, "_microtick", npc_jump_arc.capture_previous_delta(combat_env._microtick, GEN7))
    monkeypatch.setattr(
        runtime_agent,
        "_tick_agent_geometry_motion",
        npc_jump_arc.fall_speed_integration(runtime_agent._tick_agent_geometry_motion, GEN7),
    )
    y, vy = _jump_in_place()
    # The native motion differential's bars: 5e-4 m in position, 1e-3 m/s in speed.
    assert np.max(np.abs(y - NATIVE_Y)) <= 5.0e-4
    assert np.max(np.abs(vy - NATIVE_VY)) <= 1.0e-3


def test_gen5_jump_in_place_misses_the_capture_by_the_f1c_residual() -> None:
    y, _ = _jump_in_place()
    assert np.max(np.abs(y - NATIVE_Y)) > 1.0e-3
