"""Capture and compare a bounded native NPC motion trajectory.

The command talks to an already-running Hytale bridge.  It captures the
native ledge fixture for the complete requested horizon, then initializes the
JAX NPC-walk state from that reset observation and replays the same forward
and optional jump actions with the measured one-tick durations.  A failed
comparison is kept as evidence; no trajectory suffix is discarded to make a
result pass.

Example::

    python -m adk.tools.native_motion_differential \
        --port 5583 --steps 30 --output adk/runs/native-motion.json

To compare a grounded jump with forward steering, pass ``--jump-steps``.  The
jump is requested for the first N captured ticks; zero preserves the default
forward-only diagnostic.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "hytalerl_native_motion_differential_v1"
POSITION_TOLERANCE = 5.0e-4
VELOCITY_TOLERANCE = 1.0e-3
GEOMETRY_SCHEMA = "hytale_geometry_v5"
GEOMETRY_VERSION = 5

_GEOMETRY_FIELDS = (
    "available",
    "exact_collision_shapes",
    "origin",
    "cell_mask",
    "runtime_block_id",
    "runtime_fluid_id",
    "shape_id",
    "rotation",
    "flags",
    "fluid_level",
    "fluid_fill_height",
    "support",
    "block_damage",
    "fluid_damage",
    "movement",
    "fluid_movement",
    "collision_boxes",
    "collision_box_mask",
    "agent_bounds",
    "target_bounds",
    "agent_los_offset",
    "target_los_offset",
    "contacts",
    "contact_mask",
    "grounded",
    "ceiling_contact",
    "target_los",
    "target_los_valid",
)
_GEOMETRY_FLOAT_FIELDS = frozenset(
    {
        "fluid_fill_height",
        "movement",
        "fluid_movement",
        "collision_boxes",
        "agent_bounds",
        "target_bounds",
        "agent_los_offset",
        "target_los_offset",
        "contacts",
    }
)


def validate_trajectory_continuity(
    trajectory: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    name: str = "trajectory",
) -> dict[str, np.ndarray]:
    """Validate one complete motion series without silently truncating it.

    Geometry fields are optional here because the native observation contract
    and the JAX state expose different diagnostics.  The comparator requires
    the native availability/exactness fields and the JAX exhaustion field
    explicitly before it can make a pass/fail decision.
    """

    optional_fields = (
        "geometry_available",
        "geometry_exact_collision_shapes",
        "geometry_exhausted",
    )

    if isinstance(trajectory, Mapping):
        try:
            positions = np.asarray(trajectory["position"], dtype=np.float64)
            velocities = np.asarray(trajectory["velocity"], dtype=np.float64)
            grounded_raw = np.asarray(trajectory["grounded"])
        except (KeyError, TypeError, ValueError) as failure:
            raise ValueError(f"{name} is missing motion data") from failure
        optional_raw = {
            field: np.asarray(trajectory[field])
            for field in optional_fields
            if field in trajectory
        }
    else:
        if not isinstance(trajectory, Sequence) or isinstance(
            trajectory,
            (str, bytes),
        ):
            raise ValueError(f"{name} must be a mapping or a state sequence")
        positions_list: list[Any] = []
        velocities_list: list[Any] = []
        grounded_list: list[Any] = []
        optional_lists: dict[str, list[Any]] = {
            field: [] for field in optional_fields
        }
        optional_presence: dict[str, list[bool]] = {
            field: [] for field in optional_fields
        }
        for index, row in enumerate(trajectory):
            if not isinstance(row, Mapping):
                raise ValueError(f"{name} state {index} is not an object")
            if row.get("step") != index:
                raise ValueError(
                    f"{name} state {index} has non-consecutive step label"
                )
            try:
                positions_list.append(row["position"])
                velocities_list.append(row["velocity"])
                grounded_list.append(row["grounded"])
            except KeyError as failure:
                raise ValueError(
                    f"{name} state {index} is missing {failure.args[0]!r}"
                ) from failure
            for field in optional_fields:
                present = field in row
                optional_presence[field].append(present)
                if present:
                    optional_lists[field].append(row[field])
        try:
            positions = np.asarray(positions_list, dtype=np.float64)
            velocities = np.asarray(velocities_list, dtype=np.float64)
            grounded_raw = np.asarray(grounded_list)
        except (TypeError, ValueError) as failure:
            raise ValueError(f"{name} contains malformed motion data") from failure
        optional_raw = {}
        for field in optional_fields:
            presence = optional_presence[field]
            if any(presence) and not all(presence):
                raise ValueError(
                    f"{name}.{field} is missing from some trajectory states"
                )
            if presence and all(presence):
                try:
                    optional_raw[field] = np.asarray(optional_lists[field])
                except (TypeError, ValueError) as failure:
                    raise ValueError(
                        f"{name}.{field} contains malformed boolean data"
                    ) from failure

    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError(f"{name}.position must have shape [states, 3]")
    if positions.shape[0] < 1:
        raise ValueError(f"{name} must contain an initial state")
    if velocities.shape != positions.shape:
        raise ValueError(f"{name}.velocity must match position shape")
    if grounded_raw.shape != (positions.shape[0],):
        raise ValueError(f"{name}.grounded must have one value per state")
    state_shape = (positions.shape[0],)
    for field, values in optional_raw.items():
        if values.shape != state_shape:
            raise ValueError(f"{name}.{field} must have one value per state")
    if not np.all(np.isfinite(positions)) or not np.all(
        np.isfinite(velocities)
    ):
        raise ValueError(f"{name} contains nonfinite position or velocity")

    grounded_numeric = _binary_series(grounded_raw, f"{name}.grounded")
    result = {
        "position": positions,
        "velocity": velocities,
        "grounded": grounded_numeric,
    }
    for field, values in optional_raw.items():
        result[field] = _binary_series(values, f"{name}.{field}")
    return result


def compare_motion_trajectories(
    native: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    jax: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    position_tolerance: float = POSITION_TOLERANCE,
    velocity_tolerance: float = VELOCITY_TOLERANCE,
) -> dict[str, Any]:
    """Compare every state in two complete trajectories.

    A length mismatch is an error rather than an invitation to compare only a
    common prefix.  Grounded is deliberately an exact boolean comparison.
    Native geometry availability/exactness and JAX's sticky exhaustion
    diagnostic are separate contracts; missing required fields fail closed.
    """

    _nonnegative_finite(position_tolerance, "position_tolerance")
    _nonnegative_finite(velocity_tolerance, "velocity_tolerance")
    native_arrays = validate_trajectory_continuity(native, name="native")
    jax_arrays = validate_trajectory_continuity(jax, name="jax")
    if native_arrays["position"].shape[0] != jax_arrays["position"].shape[0]:
        raise ValueError(
            "native and JAX trajectories have different lengths; refusing to crop"
        )
    for field in ("geometry_available", "geometry_exact_collision_shapes"):
        if field not in native_arrays:
            raise ValueError(f"native trajectory is missing {field}")
    if "geometry_exhausted" not in jax_arrays:
        raise ValueError("JAX trajectory is missing geometry_exhausted")

    position_error = np.abs(
        native_arrays["position"] - jax_arrays["position"]
    )
    velocity_error = np.abs(
        native_arrays["velocity"] - jax_arrays["velocity"]
    )
    grounded_mismatch = native_arrays["grounded"] != jax_arrays["grounded"]
    native_geometry_unavailable = ~native_arrays["geometry_available"]
    native_geometry_nonexact = ~native_arrays["geometry_exact_collision_shapes"]
    jax_geometry_exhausted = jax_arrays["geometry_exhausted"]
    geometry_failure = (
        native_geometry_unavailable
        | native_geometry_nonexact
        | jax_geometry_exhausted
    )
    failure = (
        np.any(position_error > float(position_tolerance), axis=1)
        | np.any(velocity_error > float(velocity_tolerance), axis=1)
        | grounded_mismatch
        | geometry_failure
    )
    failure_indices = np.flatnonzero(failure)
    first_failure = None
    if failure_indices.size:
        index = int(failure_indices[0])
        first_failure = {
            "state": index,
            "position_error": position_error[index].tolist(),
            "velocity_error": velocity_error[index].tolist(),
            "grounded_mismatch": bool(grounded_mismatch[index]),
            "native_geometry_unavailable": bool(
                native_geometry_unavailable[index]
            ),
            "native_geometry_nonexact": bool(native_geometry_nonexact[index]),
            "jax_geometry_exhausted": bool(jax_geometry_exhausted[index]),
        }

    return {
        "passed": not bool(failure_indices.size),
        "state_count": int(native_arrays["position"].shape[0]),
        "transition_count": int(native_arrays["position"].shape[0] - 1),
        "position_tolerance": float(position_tolerance),
        "velocity_tolerance": float(velocity_tolerance),
        "grounded_exact": True,
        "max_position_error": float(np.max(position_error)),
        "max_velocity_error": float(np.max(velocity_error)),
        "grounded_mismatch_count": int(np.count_nonzero(grounded_mismatch)),
        "native_geometry_unavailable_count": int(
            np.count_nonzero(native_geometry_unavailable)
        ),
        "native_geometry_nonexact_count": int(
            np.count_nonzero(native_geometry_nonexact)
        ),
        "jax_geometry_exhausted_count": int(
            np.count_nonzero(jax_geometry_exhausted)
        ),
        "position_abs": position_error.tolist(),
        "velocity_abs": velocity_error.tolist(),
        "grounded_mismatch": grounded_mismatch.astype(int).tolist(),
        "native_geometry_unavailable": native_geometry_unavailable.astype(
            int
        ).tolist(),
        "native_geometry_nonexact": native_geometry_nonexact.astype(int).tolist(),
        "jax_geometry_exhausted": jax_geometry_exhausted.astype(int).tolist(),
        "geometry_failure": geometry_failure.astype(int).tolist(),
        "failure_states": failure_indices.astype(int).tolist(),
        "first_failure": first_failure,
    }


def validate_timing_continuity(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_world_tick: int,
    initial_episode_engine_ticks: int,
) -> None:
    """Require one measured tick and continuous episode timing per row.

    ``native_world_tick`` is an absolute server scheduler counter.  The live
    bridge can advance it by more than one between paused requests, so the
    evidence requires strict progress rather than inventing a one-to-one
    mapping.  ``episode_engine_ticks`` is the request-local counter and must
    advance exactly by the reported executed ticks.
    """

    previous_world_tick = _exact_int(initial_world_tick, "initial_world_tick")
    previous_episode_ticks = _exact_int(
        initial_episode_engine_ticks,
        "initial_episode_engine_ticks",
    )
    for expected_step, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"timing row {expected_step} is not an object")
        step = _exact_int(record.get("step"), f"timing[{expected_step}].step")
        if step != expected_step:
            raise ValueError(f"timing row {expected_step} is not consecutive")
        executed = _exact_int(
            record.get("engine_ticks_executed"),
            f"timing[{expected_step}].engine_ticks_executed",
        )
        if executed != 1:
            raise ValueError(
                f"timing row {expected_step} executed {executed} ticks; expected 1"
            )
        world_tick = _exact_int(
            record.get("native_world_tick"),
            f"timing[{expected_step}].native_world_tick",
        )
        episode_ticks = _exact_int(
            record.get("episode_engine_ticks"),
            f"timing[{expected_step}].episode_engine_ticks",
        )
        if world_tick <= previous_world_tick:
            raise ValueError(
                f"native world tick did not advance at timing row {expected_step}"
            )
        if episode_ticks != previous_episode_ticks + executed:
            raise ValueError(f"episode tick gap at timing row {expected_step}")
        dt = _finite_number(
            record.get("dt_seconds"),
            f"timing[{expected_step}].dt_seconds",
        )
        if dt <= 0.0:
            raise ValueError(f"timing row {expected_step} has non-positive dt")
        previous_world_tick = world_tick
        previous_episode_ticks = episode_ticks


def _forward_jump_schedule(
    *,
    steps: int,
    jump_steps: int,
    forward_steps: int | None = None,
) -> list[dict[str, int]]:
    """Build a bounded forward/jump input schedule for the capture horizon."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if jump_steps < 0 or jump_steps > steps:
        raise ValueError("jump_steps must be between 0 and steps")
    if forward_steps is None:
        forward_steps = steps
    if forward_steps < 0 or forward_steps > steps:
        raise ValueError("forward_steps must be between 0 and steps")
    return [
        {
            "forward": int(step < forward_steps),
            "jump": int(step < jump_steps),
        }
        for step in range(steps)
    ]


def _capture_native_motion(
    *,
    host: str,
    port: int,
    seed: int,
    steps: int,
    jump_steps: int,
    forward_steps: int | None,
    expected_server_version: str,
    expected_bridge_sha256: str | None,
) -> dict[str, Any]:
    """Capture native motion and perform one JAX replay."""

    if steps < 1:
        raise ValueError("steps must be positive")
    action_schedule = _forward_jump_schedule(
        steps=steps,
        jump_steps=jump_steps,
        forward_steps=forward_steps,
    )
    effective_forward_steps = steps if forward_steps is None else forward_steps
    if not expected_server_version.strip():
        raise ValueError("expected_server_version must be non-empty")
    expected_bridge_sha256 = _normalize_sha256(
        expected_bridge_sha256,
        "expected_bridge_sha256",
    )

    # Keep the live dependencies out of offline validation/import paths.
    from hytalegym.envs.hytale_env import HytaleEnv
    from hytalegym.jax.combat import ACTION_FORWARD, ACTION_JUMP
    from hytalegym.geometry import GEOMETRY_SCHEMA as NATIVE_GEOMETRY_SCHEMA
    from hytalegym.jax.world import geometry_state_from_numpy
    from hytalegym.worldgen.region.stability import (
        current_native_evidence_jar_sha256,
    )
    from hytalegym.worldgen.region.stability import (
        require_current_native_runtime_bridge,
    )

    expected_bridge = expected_bridge_sha256 or current_native_evidence_jar_sha256()
    env = HytaleEnv(
        task="native_fidelity",
        host=host,
        port=port,
        backend="native",
        world="flat",
        npc_role="Kweebec_Razorleaf",
        combat_target_active=False,
        fidelity_fixture="ledge",
        ticks_per_step=1,
        # Leave one boundary beyond the captured horizon so Gym does not
        # manufacture a truncation on the final requested sample.
        max_episode_steps=steps + 1,
    )
    try:
        observation, reset_info = env.reset(seed=seed)
        runtime = _validate_native_runtime(
            reset_info,
            expected_server_version=expected_server_version,
            expected_bridge_sha256=expected_bridge,
            expected_role="Kweebec_Razorleaf",
            expected_geometry_schema=NATIVE_GEOMETRY_SCHEMA,
            require_current_native_runtime_bridge=require_current_native_runtime_bridge,
        )
        if not env.observation_space.contains(observation):
            raise ValueError("native reset observation is outside HytaleEnv space")
        geometry = observation.get("geometry")
        _validate_geometry_frame(geometry)
        # This is the same exact-shape conversion used by the geometry gate.
        geometry_state_from_numpy(geometry)

        native_reset = _native_reset_state(observation, reset_info)
        initial_world_tick = _exact_int(
            reset_info.get("native_world_tick"),
            "reset.native_world_tick",
        )
        initial_episode_ticks = _exact_int(
            reset_info.get("episode_engine_ticks"),
            "reset.episode_engine_ticks",
        )
        if _exact_int(
            reset_info.get("engine_ticks_executed"),
            "reset.engine_ticks_executed",
        ) != 0:
            raise ValueError("native reset executed engine ticks")
        if _finite_number(
            reset_info.get("engine_mean_delta_seconds"),
            "reset.engine_mean_delta_seconds",
        ) != 0.0:
            raise ValueError("native reset reported a nonzero tick duration")
        if reset_info.get("world_paused_between_steps") is not True:
            raise ValueError("native runtime did not report paused step boundaries")

        native_trajectory = [
            {
                "step": 0,
                "position": native_reset["position"],
                "velocity": native_reset["velocity"],
                "grounded": native_reset["grounded"],
                "geometry_available": native_reset["geometry_available"],
                "geometry_exact_collision_shapes": native_reset[
                    "geometry_exact_collision_shapes"
                ],
                "native_world_tick": initial_world_tick,
                "episode_engine_ticks": initial_episode_ticks,
            }
        ]
        actions: list[dict[str, Any]] = []
        timing: list[dict[str, Any]] = []
        for step, action in enumerate(action_schedule, start=1):
            next_observation, _, terminated, truncated, info = env.step(action)
            step_runtime = _validate_native_runtime(
                info,
                expected_server_version=expected_server_version,
                expected_bridge_sha256=expected_bridge,
                expected_role="Kweebec_Razorleaf",
                expected_geometry_schema=NATIVE_GEOMETRY_SCHEMA,
                require_current_native_runtime_bridge=require_current_native_runtime_bridge,
            )
            if step_runtime != runtime:
                raise ValueError(f"native runtime identity changed at step {step}")
            _validate_geometry_frame(
                next_observation.get("geometry"),
                require_exact=False,
            )
            row_timing = _timing_row(info, step, terminated, truncated)
            timing.append(row_timing)
            state = _native_motion_state(next_observation, step)
            state["native_world_tick"] = row_timing["native_world_tick"]
            state["episode_engine_ticks"] = row_timing["episode_engine_ticks"]
            native_trajectory.append(state)
            actions.append(
                {
                    "step": step,
                    "native": action,
                    "jax": {
                        "forward_index": ACTION_FORWARD,
                        "jump_index": ACTION_JUMP,
                    },
                    "dt_seconds": row_timing["dt_seconds"],
                }
            )
            if terminated or truncated:
                raise ValueError(
                    f"native trajectory ended at step {step}; refusing to crop"
                )
    finally:
        env.close()

    validate_timing_continuity(
        timing,
        initial_world_tick=initial_world_tick,
        initial_episode_engine_ticks=initial_episode_ticks,
    )
    jax_trajectory, jax_runtime = _replay_jax_once(
        native_reset,
        geometry,
        actions,
        seed=seed,
    )
    comparison = compare_motion_trajectories(native_trajectory, jax_trajectory)
    geometry_evidence = _geometry_evidence(geometry)
    return {
        "schema": SCHEMA,
        "claim": "bounded_npc_motion",
        "passed": bool(comparison["passed"]),
        "experiment": {
            "seed": seed,
            "requested_steps": steps,
            "jump_steps": jump_steps,
            "forward_steps": effective_forward_steps,
            "completed_steps": len(actions),
            "task": "native_fidelity",
            "world": "flat",
            "requested_fidelity_fixture": "ledge",
            "motion_model": "npc_walk",
            "action_schedule": {
                "forward": (
                    "every step"
                    if effective_forward_steps == steps
                    else "disabled"
                    if effective_forward_steps == 0
                    else "first N steps"
                ),
                "jump": "first N steps" if jump_steps else "disabled",
            },
            "trajectory_includes_reset_state": True,
            "posthoc_cropping": False,
        },
        "tolerances": {
            "position_abs": POSITION_TOLERANCE,
            "velocity_abs": VELOCITY_TOLERANCE,
            "grounded": "exact",
        },
        "geometry_semantics": {
            "native_available": "observation.geometry.available per state",
            "native_exact_collision_shapes": (
                "observation.geometry.exact_collision_shapes per state"
            ),
            "jax_geometry_exhausted": (
                "CombatState.geometry_exhausted after reset and every replay state"
            ),
            "failure_policy": (
                "native unavailable/nonexact or any JAX exhaustion fails closed"
            ),
        },
        "runtime": {
            "native": runtime,
            "jax": jax_runtime,
        },
        "reset": {
            "native": native_reset,
            "jax": jax_trajectory[0],
        },
        "geometry": geometry_evidence,
        "actions": actions,
        "timing": timing,
        "native_trajectory": native_trajectory,
        "jax_trajectory": jax_trajectory,
        "errors": comparison,
    }


def _replay_jax_once(
    native_reset: Mapping[str, Any],
    geometry_frame: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Initialize from native reset data and execute exactly one JAX scan."""

    import jax
    import jax.numpy as jnp

    from hytalegym.jax.combat import (
        ACTION_FORWARD,
        ACTION_JUMP,
        ACTION_SIZE,
        AGENT_ENTITY,
        default_combat_params,
        reset_batch_geometry_at,
        step_batch_geometry,
    )
    from hytalegym.jax.world import geometry_state_from_numpy

    geometry = geometry_state_from_numpy(geometry_frame)
    params = default_combat_params(
        microticks=1,
        target_active=False,
        motion_model="npc_walk",
    )
    native_position = np.asarray(native_reset["position"], dtype=np.float32)
    native_velocity = np.asarray(native_reset["velocity"], dtype=np.float32)
    target_position = native_position + np.asarray(
        params.target_offset,
        dtype=np.float32,
    )
    reset_keys = jax.random.split(jax.random.PRNGKey(seed), 1)
    state, _ = reset_batch_geometry_at(
        reset_keys,
        params,
        geometry,
        native_position[None, :],
        target_position[None, :],
    )
    if bool(np.asarray(state.agent_grounded)[0]) != bool(
        native_reset["grounded"]
    ):
        raise ValueError("JAX geometry reset disagrees with native grounded state")

    yaw = np.float32(native_reset["yaw"])
    pitch = np.float32(native_reset["pitch"])
    state = state._replace(
        velocity=state.velocity.at[:, AGENT_ENTITY].set(native_velocity),
        yaw=state.yaw.at[:, AGENT_ENTITY].set(yaw),
        agent_head_yaw=jnp.asarray([yaw], dtype=jnp.float32),
        desired_yaw=jnp.asarray([yaw], dtype=jnp.float32),
        desired_body_yaw=jnp.asarray([yaw], dtype=jnp.float32),
        pitch=jnp.asarray([pitch], dtype=jnp.float32),
        agent_head_pitch=jnp.asarray([pitch], dtype=jnp.float32),
        desired_pitch=jnp.asarray([pitch], dtype=jnp.float32),
        agent_grounded=jnp.asarray(
            [bool(native_reset["grounded"])],
            dtype=jnp.bool_,
        ),
        agent_fall_start_y=jnp.asarray(
            [native_position[1]],
            dtype=jnp.float32,
        ),
    )

    action_rows = len(actions)
    action_tensor = jnp.zeros(
        (action_rows, 1, ACTION_SIZE),
        dtype=jnp.float32,
    )
    action_tensor = action_tensor.at[:, :, ACTION_FORWARD].set(
        jnp.asarray(
            [float(action["native"]["forward"]) for action in actions],
            dtype=jnp.float32,
        )[:, None]
    )
    action_tensor = action_tensor.at[:, :, ACTION_JUMP].set(
        jnp.asarray(
            [float(action["native"].get("jump", 0)) for action in actions],
            dtype=jnp.float32,
        )[:, None]
    )
    tick_deltas = jnp.asarray(
        [float(action["dt_seconds"]) for action in actions],
        dtype=jnp.float32,
    )
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), action_rows).reshape(
        action_rows,
        1,
        2,
    )

    def rollout_step(current, inputs):
        action, key, tick_delta = inputs
        timed_params = params._replace(
            nominal_dt=tick_delta,
            loaded_dt=tick_delta,
        )
        next_state, _, _, _, _ = step_batch_geometry(
            current,
            action,
            key,
            timed_params,
            geometry,
        )
        return next_state, (
            next_state.position[:, AGENT_ENTITY],
            next_state.velocity[:, AGENT_ENTITY],
            next_state.agent_grounded,
            next_state.geometry_exhausted,
        )

    # The only transition call: one scan over the complete captured horizon.
    _, trace = jax.jit(
        lambda initial, action_values, key_values, delta_values: jax.lax.scan(
            rollout_step,
            initial,
            (action_values, key_values, delta_values),
        )
    )(state, action_tensor, keys, tick_deltas)
    positions = np.asarray(trace[0][:, 0])
    velocities = np.asarray(trace[1][:, 0])
    grounded = np.asarray(trace[2][:, 0], dtype=bool)
    geometry_exhausted = np.asarray(trace[3][:, 0], dtype=bool)
    jax_rows = [
        {
            "step": 0,
            "position": native_position.astype(np.float64).tolist(),
            "velocity": native_velocity.astype(np.float64).tolist(),
            "grounded": bool(native_reset["grounded"]),
            "geometry_exhausted": bool(np.asarray(state.geometry_exhausted)[0]),
        }
    ]
    for index in range(action_rows):
        jax_rows.append(
            {
                "step": index + 1,
                "position": positions[index].astype(np.float64).tolist(),
                "velocity": velocities[index].astype(np.float64).tolist(),
                "grounded": bool(grounded[index]),
                "geometry_exhausted": bool(geometry_exhausted[index]),
            }
        )
    return jax_rows, {
        "jax_version": str(jax.__version__),
        "backend": str(jax.default_backend()),
        "motion_model": "npc_walk",
        "geometry_schema": GEOMETRY_SCHEMA,
        "geometry_version": GEOMETRY_VERSION,
        "reset_prng_seed": seed,
        "action_prng_seed": seed + 1,
        "replay": "one_jitted_lax_scan",
    }


def _validate_native_runtime(
    info: Mapping[str, Any],
    *,
    expected_server_version: str,
    expected_bridge_sha256: str,
    expected_role: str,
    expected_geometry_schema: str,
    require_current_native_runtime_bridge: Any,
) -> dict[str, Any]:
    if not isinstance(info, Mapping):
        raise ValueError("native response omitted runtime info")
    if info.get("backend") != "native":
        raise RuntimeError("motion evidence requires the native backend")
    server_version = _required_text(info.get("native_server_version"), "server")
    if server_version != expected_server_version:
        raise RuntimeError(
            f"wrong native server version {server_version!r}; "
            f"expected {expected_server_version!r}"
        )
    bridge_sha256 = require_current_native_runtime_bridge(
        info,
        expected=expected_bridge_sha256,
    )
    if info.get("geometry_schema") != expected_geometry_schema:
        raise RuntimeError("native geometry schema is not the expected contract")
    role = _required_text(info.get("npc_role"), "npc_role")
    if role != expected_role:
        raise RuntimeError(f"wrong native NPC role {role!r}; expected {expected_role!r}")
    fixture = _required_text(info.get("fidelity_fixture"), "fidelity_fixture")
    if "three_block_trench" not in fixture:
        raise RuntimeError(
            f"native fixture {fixture!r} is not the ledge trench fixture"
        )
    if info.get("world_template") != "flat":
        raise RuntimeError("motion evidence requires the flat native world")
    if info.get("task") != "native_fidelity":
        raise RuntimeError("native task identity is not native_fidelity")
    native_tick_rate = _exact_int(info.get("native_tick_rate"), "native_tick_rate")
    if native_tick_rate != 30:
        raise RuntimeError(f"native tick rate {native_tick_rate} is not 30")
    native_time_dilation = _finite_number(
        info.get("native_time_dilation"),
        "native_time_dilation",
    )
    if native_time_dilation != 1.0:
        raise RuntimeError("native time dilation is not 1.0")

    return {
        "backend": "native",
        "native_server_version": server_version,
        "bridge_sha256": bridge_sha256,
        "world_template": "flat",
        "world": _required_text(info.get("world"), "world"),
        "worldgen_provider": _required_text(
            info.get("worldgen_provider"),
            "worldgen_provider",
        ),
        "worldgen_version": _required_text(
            info.get("worldgen_version"),
            "worldgen_version",
        ),
        "npc_role": role,
        "fidelity_fixture": fixture,
        "task": "native_fidelity",
        "native_tick_rate": native_tick_rate,
        "native_time_dilation": native_time_dilation,
        "geometry_schema": expected_geometry_schema,
        "expected_bridge_sha256": expected_bridge_sha256,
    }


def _timing_row(
    info: Mapping[str, Any],
    step: int,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    executed = _exact_int(info.get("engine_ticks_executed"), "engine_ticks_executed")
    dt = _finite_number(info.get("engine_mean_delta_seconds"), "engine_mean_delta_seconds")
    simulated = _finite_number(
        info.get("engine_simulated_seconds"),
        "engine_simulated_seconds",
    )
    minimum = _finite_number(
        info.get("engine_min_delta_seconds"),
        "engine_min_delta_seconds",
    )
    maximum = _finite_number(
        info.get("engine_max_delta_seconds"),
        "engine_max_delta_seconds",
    )
    if executed != 1:
        raise ValueError(f"step {step} did not execute exactly one native tick")
    if dt <= 0.0 or minimum <= 0.0 or maximum <= 0.0:
        raise ValueError(f"step {step} has non-positive native timing")
    if minimum > dt or dt > maximum:
        raise ValueError(f"step {step} mean dt is outside native min/max")
    if not math.isclose(simulated, dt * executed, rel_tol=1.0e-6, abs_tol=1.0e-7):
        raise ValueError(f"step {step} native timing fields disagree")
    if info.get("world_paused_between_steps") is not True:
        raise ValueError(f"step {step} did not report a paused boundary")
    return {
        "step": step,
        "engine_ticks_executed": executed,
        "dt_seconds": dt,
        "engine_simulated_seconds": simulated,
        "engine_min_delta_seconds": minimum,
        "engine_max_delta_seconds": maximum,
        "native_world_tick": _exact_int(
            info.get("native_world_tick"),
            f"step[{step}].native_world_tick",
        ),
        "episode_engine_ticks": _exact_int(
            info.get("episode_engine_ticks"),
            f"step[{step}].episode_engine_ticks",
        ),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _native_reset_state(
    observation: Mapping[str, Any],
    info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **_native_motion_state(observation, 0),
        "yaw": _scalar_finite(observation.get("yaw"), "reset.yaw"),
        "pitch": _scalar_finite(observation.get("pitch"), "reset.pitch"),
        "native_world_tick": _exact_int(
            info.get("native_world_tick"),
            "reset.native_world_tick",
        ),
        "episode_engine_ticks": _exact_int(
            info.get("episode_engine_ticks"),
            "reset.episode_engine_ticks",
        ),
    }


def _native_motion_state(
    observation: Mapping[str, Any],
    step: int,
) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        raise ValueError(f"native observation at step {step} is not an object")
    position = _vector_finite(observation.get("position"), f"step[{step}].position")
    velocity = _vector_finite(observation.get("velocity"), f"step[{step}].velocity")
    geometry = observation.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError(f"native geometry missing at step {step}")
    grounded = _binary_bool(geometry.get("grounded"), f"step[{step}].grounded")
    geometry_available = _binary_bool(
        geometry.get("available"),
        f"step[{step}].geometry_available",
    )
    geometry_exact = _binary_bool(
        geometry.get("exact_collision_shapes"),
        f"step[{step}].geometry_exact_collision_shapes",
    )
    return {
        "step": step,
        "position": position,
        "velocity": velocity,
        "grounded": grounded,
        "geometry_available": geometry_available,
        "geometry_exact_collision_shapes": geometry_exact,
    }


def _validate_geometry_frame(frame: Any, *, require_exact: bool = True) -> None:
    if not isinstance(frame, Mapping):
        raise ValueError("native geometry is not an object")
    missing = [field for field in _GEOMETRY_FIELDS if field not in frame]
    if missing:
        raise ValueError(f"native geometry is missing fields: {', '.join(missing)}")
    available = _binary_int(frame["available"], "geometry.available")
    exact = _binary_int(frame["exact_collision_shapes"], "geometry.exact")
    if require_exact and available != 1:
        raise ValueError("native geometry is unavailable")
    if require_exact and exact != 1:
        raise ValueError("native geometry does not contain exact collision shapes")
    for field in _GEOMETRY_FIELDS:
        try:
            values = np.asarray(frame[field])
        except (TypeError, ValueError) as failure:
            raise ValueError(f"geometry field {field!r} is not an array") from failure
        if values.dtype.kind not in "biuf":
            raise ValueError(f"geometry field {field!r} is not numeric")
        if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
            raise ValueError(f"geometry field {field!r} contains nonfinite data")
        if field in _GEOMETRY_FLOAT_FIELDS and not np.all(np.isfinite(values)):
            raise ValueError(f"geometry field {field!r} contains nonfinite data")


def _geometry_evidence(frame: Mapping[str, Any]) -> dict[str, Any]:
    _validate_geometry_frame(frame)
    return {
        "schema": GEOMETRY_SCHEMA,
        "version": GEOMETRY_VERSION,
        "content_sha256": _geometry_content_sha256(frame),
        "data": {field: _jsonable(frame[field]) for field in _GEOMETRY_FIELDS},
    }


def _geometry_content_sha256(frame: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{GEOMETRY_SCHEMA}:{GEOMETRY_VERSION}\n".encode("ascii"))
    for field in _GEOMETRY_FIELDS:
        values = np.asarray(frame[field])
        canonical = np.ascontiguousarray(values)
        if canonical.dtype.kind in "biuf":
            canonical = canonical.astype(canonical.dtype.newbyteorder("<"), copy=False)
        digest.update(field.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(canonical.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(canonical.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest().upper()


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _exact_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _binary_int(value: Any, name: str) -> int:
    parsed = _exact_int(value, name)
    if parsed not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return parsed


def _binary_bool(value: Any, name: str) -> bool:
    return bool(_binary_int(value, name))


def _binary_series(value: Any, name: str) -> np.ndarray:
    try:
        numeric = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as failure:
        raise ValueError(f"{name} is not boolean data") from failure
    if not np.all(np.isfinite(numeric)) or np.any(
        (numeric != 0.0) & (numeric != 1.0)
    ):
        raise ValueError(f"{name} must contain only 0/1 values")
    return numeric.astype(bool)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _nonnegative_finite(value: Any, name: str) -> float:
    parsed = _finite_number(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _scalar_finite(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != () or not np.isfinite(array).item():
        raise ValueError(f"{name} must be one finite scalar")
    return float(array)


def _vector_finite(value: Any, name: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    return array.tolist()


def _normalize_sha256(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256")
    return value.upper()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _failure_evidence(arguments: argparse.Namespace, failure: Exception) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "claim": "bounded_npc_motion",
        "passed": False,
        "error": f"{type(failure).__name__}: {failure}",
        "experiment": {
            "seed": arguments.seed,
            "requested_steps": arguments.steps,
            "jump_steps": arguments.jump_steps,
            "forward_steps": (
                arguments.steps
                if arguments.forward_steps is None
                else arguments.forward_steps
            ),
            "task": "native_fidelity",
            "world": "flat",
            "requested_fidelity_fixture": "ledge",
            "motion_model": "npc_walk",
            "posthoc_cropping": False,
        },
        "prerequisites": {
            "server": f"already-running Hytale bridge at {arguments.host}:{arguments.port}",
            "expected_server_version": arguments.expected_server_version,
            "expected_bridge_sha256": arguments.expected_bridge_sha256,
        },
    }


def _write_evidence(path: Path, evidence: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(
            _jsonable(evidence),
            target,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        target.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5583)
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Predeclared one-tick capture horizon.",
    )
    parser.add_argument(
        "--jump-steps",
        type=int,
        default=0,
        help="Request jump for the first N ticks (0 disables jumping).",
    )
    parser.add_argument(
        "--forward-steps",
        type=int,
        default=None,
        help="Request forward for the first N ticks (default: every tick).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-server-version", default="0.5.9")
    parser.add_argument("--expected-bridge-sha256", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.port < 1 or arguments.port > 65535:
            raise ValueError("port must be in [1, 65535]")
        if arguments.seed <= 0 or arguments.seed >= 2**31:
            raise ValueError("seed must be in (0, 2^31)")
        if arguments.steps < 1:
            raise ValueError("steps must be positive")
        _forward_jump_schedule(
            steps=arguments.steps,
            jump_steps=arguments.jump_steps,
            forward_steps=arguments.forward_steps,
        )
        from hytalegym.worldgen.native_lease import acquire_native_evidence_lease

        with acquire_native_evidence_lease(
            host=arguments.host,
            port=arguments.port,
            purpose="native_motion_differential",
        ):
            evidence = _capture_native_motion(
                host=arguments.host,
                port=arguments.port,
                seed=arguments.seed,
                steps=arguments.steps,
                jump_steps=arguments.jump_steps,
                forward_steps=arguments.forward_steps,
                expected_server_version=arguments.expected_server_version,
                expected_bridge_sha256=arguments.expected_bridge_sha256,
            )
    except Exception as failure:
        evidence = _failure_evidence(arguments, failure)
        _write_evidence(arguments.output, evidence)
        print(json.dumps({"schema": SCHEMA, "passed": False, "error": str(failure)}))
        return 1

    _write_evidence(arguments.output, evidence)
    summary = {
        "schema": SCHEMA,
        "passed": bool(evidence["passed"]),
        "output": str(arguments.output.resolve()),
        "max_position_error": evidence["errors"]["max_position_error"],
        "max_velocity_error": evidence["errors"]["max_velocity_error"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
