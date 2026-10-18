"""Export the static half of the prospective Dodge-corridor contract.

The live JVM owns actor pose, grounding, bounding box, and block geometry.  The
checkpoint owns the motion scalars used by the JAX prospective sweep.  Keeping
those halves separate prevents the standalone mod from baking one weapon,
role, or material's numbers into Java.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np


SCHEMA = "hytalerl_dodge_corridor_motion_v1"
VERSION = 1


def write_dodge_motion_parameters(
    directory: Path,
    runtime,
    params,
    *,
    lane: int,
    motion_timing_profile: int,
    entity: int = 0,
) -> Path:
    """Write one checkpoint-pinned, role-agnostic motion parameter row."""

    from hytalegym.jax.combat.arsenal.environment import (
        DODGE_CLEARANCE_TICK_CAPACITY,
    )
    from hytalegym.jax.combat.mechanics.schema.contract import (
        APPLIED_RESISTANCE_STYLE_FORCE_VELOCITY,
        DODGE_VELOCITY_REMOVAL_SQUARED,
    )

    rules = runtime.mechanics_rules

    def scalar(value, *, entity_axis: bool = False) -> float:
        array = np.asarray(value)
        if entity_axis:
            return float(array[lane, entity])
        return float(array.reshape(-1)[0])

    values: tuple[tuple[str, object], ...] = (
        ("schema", SCHEMA),
        ("version", VERSION),
        (
            "execution_profile",
            int(np.asarray(rules.dodge_execution_profile)[lane, entity]),
        ),
        ("authored_force", scalar(rules.dodge_force, entity_axis=True)),
        ("knockback_scale", scalar(params.agent_knockback_scale)),
        ("nominal_delta_seconds", scalar(params.nominal_dt)),
        ("loaded_delta_seconds", scalar(params.loaded_dt)),
        (
            "server_ticks_per_second",
            scalar(rules.server_ticks_per_second),
        ),
        ("motion_timing_profile", int(motion_timing_profile)),
        (
            "configured_air_resistance",
            scalar(rules.dodge_air_resistance, entity_axis=True),
        ),
        (
            "configured_air_resistance_max",
            scalar(rules.dodge_air_resistance_max, entity_axis=True),
        ),
        (
            "configured_ground_resistance",
            scalar(rules.dodge_ground_resistance, entity_axis=True),
        ),
        (
            "configured_ground_resistance_max",
            scalar(rules.dodge_ground_resistance_max, entity_axis=True),
        ),
        (
            "configured_resistance_threshold",
            scalar(rules.dodge_resistance_threshold, entity_axis=True),
        ),
        # JAX uses VelocityThresholdStyle.Exp for the authored client config.
        ("configured_resistance_style", 1),
        (
            "native_horizontal_factor",
            scalar(params.legacy_motion_controller_horizontal_factor),
        ),
        (
            "native_movement_velocity_resistance",
            scalar(params.agent_movement_velocity_resistance),
        ),
        (
            "native_ground_drag_base",
            scalar(params.agent_force_ground_drag_base),
        ),
        ("native_air_drag_min", scalar(params.agent_force_air_drag_min)),
        ("native_air_drag_max", scalar(params.agent_force_air_drag_max)),
        (
            "native_air_drag_min_speed",
            scalar(params.agent_force_air_drag_min_speed),
        ),
        (
            "native_air_drag_max_speed",
            scalar(params.agent_force_air_drag_max_speed),
        ),
        (
            "native_reference_ticks_per_second",
            scalar(params.agent_force_reference_ticks_per_second),
        ),
        (
            "native_per_axis_deadzone",
            scalar(params.agent_force_per_axis_deadzone),
        ),
        (
            "native_resistance_style",
            int(APPLIED_RESISTANCE_STYLE_FORCE_VELOCITY),
        ),
        ("velocity_removal_squared", DODGE_VELOCITY_REMOVAL_SQUARED),
        ("clearance_tick_capacity", DODGE_CLEARANCE_TICK_CAPACITY),
    )
    path = directory / "dodge_motion.tsv"
    path.write_text(
        "\n".join(f"{name}\t{value}" for name, value in values) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "SCHEMA",
    "VERSION",
    "write_dodge_motion_parameters",
    "write_randomized_oracle",
]


def write_randomized_oracle(
    path: Path,
    *,
    count: int = 384,
    seed: int = 0xD0D6E6,
) -> Path:
    """Write seeded JAX motion rows consumed by the Java recurrence gate."""

    import jax
    import jax.numpy as jnp

    from hytalegym.jax.combat import default_combat_params
    from hytalegym.jax.combat.mechanics.runtime.motion import (
        damp_applied_velocity,
        resolve_dodge_launch_velocity,
    )

    if jax.default_backend() != "gpu":
        raise SystemExit(
            "Dodge motion oracle publication requires WSL/CUDA; "
            f"observed {jax.default_backend()!r}"
        )
    if count < 256:
        raise ValueError("randomized Dodge oracle requires at least 256 rows")

    rng = np.random.default_rng(seed)
    profile = rng.integers(0, 2, count, dtype=np.int32)
    timing_profile = rng.integers(0, 3, count, dtype=np.int32)
    starting_tick = rng.integers(0, 1_000_000, count, dtype=np.int64)
    yaw = rng.uniform(-180.0, 180.0, count).astype(np.float32)
    grounded = rng.integers(0, 2, count, dtype=np.int8).astype(bool)
    force = rng.uniform(0.5, 25.0, count).astype(np.float32)
    knockback = rng.uniform(0.1, 1.8, count).astype(np.float32)
    configured_air = rng.uniform(0.70, 0.99, count).astype(np.float32)
    configured_air_max = rng.uniform(0.50, 0.98, count).astype(np.float32)
    configured_ground = rng.uniform(0.65, 0.98, count).astype(np.float32)
    configured_ground_max = rng.uniform(0.45, 0.95, count).astype(np.float32)
    threshold = rng.uniform(0.5, 15.0, count).astype(np.float32)
    style = rng.integers(0, 2, count, dtype=np.int32)
    native_horizontal = rng.uniform(0.05, 0.5, count).astype(np.float32)
    native_movement = rng.uniform(0.05, 0.8, count).astype(np.float32)
    native_ground = rng.uniform(0.55, 0.95, count).astype(np.float32)
    native_air_min = rng.uniform(0.7, 0.98, count).astype(np.float32)
    native_air_max = rng.uniform(0.8, 0.999, count).astype(np.float32)
    native_speed_min = rng.uniform(0.25, 8.0, count).astype(np.float32)
    native_speed_max = (
        native_speed_min + rng.uniform(0.25, 8.0, count)
    ).astype(np.float32)
    native_reference = rng.uniform(30.0, 90.0, count).astype(np.float32)
    native_deadzone = rng.uniform(0.01, 0.25, count).astype(np.float32)
    # The compiled runtime primitive owns this constant; unlike the remaining
    # motion fields it is not a per-case parameter.  Keeping it fixed here
    # makes the oracle exercise the same stop test both inside the damping
    # primitive and around the prospective loop.
    removal = np.full(count, 1.0e-3, dtype=np.float32)
    tps = np.full(count, 30.0, dtype=np.float32)
    dt = np.full(count, 1.0 / 30.0, dtype=np.float32)
    loaded_dt = rng.uniform(0.034, 0.065, count).astype(np.float32)
    capacity = np.full(count, 128, dtype=np.int32)

    f32 = jnp.float32
    radians = jnp.deg2rad(jnp.asarray(yaw))
    authored = jnp.stack(
        (
            -jnp.sin(radians),
            jnp.zeros_like(radians),
            -jnp.cos(radians),
        ),
        axis=1,
    ) * jnp.asarray(force)[:, None]
    params = default_combat_params()._replace(
        agent_knockback_scale=jnp.asarray(knockback),
        legacy_motion_controller_horizontal_factor=jnp.asarray(
            native_horizontal),
        agent_movement_velocity_resistance=jnp.asarray(native_movement),
        agent_force_ground_drag_base=jnp.asarray(native_ground),
        agent_force_air_drag_min=jnp.asarray(native_air_min),
        agent_force_air_drag_max=jnp.asarray(native_air_max),
        agent_force_air_drag_min_speed=jnp.asarray(native_speed_min),
        agent_force_air_drag_max_speed=jnp.asarray(native_speed_max),
        agent_force_reference_ticks_per_second=jnp.asarray(native_reference),
        # This parameter is compared directly with the final XYZ axis in the
        # JAX primitive, so its generated leading batch needs a singleton axis.
        agent_force_per_axis_deadzone=jnp.asarray(native_deadzone)[:, None],
    )
    current = resolve_dodge_launch_velocity(
        authored,
        jnp.asarray(profile),
        params,
    )
    displacement = jnp.zeros_like(current)
    for index in range(128):
        native = jnp.asarray(profile) == 0
        moving = jnp.where(
            native,
            jnp.any(current != f32(0.0), axis=1),
            jnp.sum(current * current, axis=1) >= jnp.asarray(removal),
        )
        prospective_tick = starting_tick + index + 1
        odd = (prospective_tick % 2) == 1
        loaded_tick = np.where(timing_profile == 1, odd, ~odd)
        step_dt = np.where(
            (timing_profile != 0) & loaded_tick,
            loaded_dt,
            dt,
        ).astype(np.float32)
        displacement = displacement + jnp.where(
            moving[:, None],
            current * jnp.asarray(step_dt)[:, None],
            f32(0.0),
        )
        current = damp_applied_velocity(
            current,
            jnp.asarray(grounded),
            jnp.asarray(configured_air),
            jnp.asarray(configured_air_max),
            jnp.asarray(configured_ground),
            jnp.asarray(configured_ground_max),
            jnp.asarray(threshold),
            jnp.where(native, jnp.int32(2), jnp.asarray(style)),
            jnp.zeros(count, dtype=jnp.bool_),
            jnp.asarray(tps),
            params,
        )
    final = np.asarray(current)
    forward = np.asarray(displacement)
    native = profile == 0
    complete = np.where(
        native,
        ~np.any(final != np.float32(0.0), axis=1),
        np.sum(final * final, axis=1) < removal,
    )
    backward = forward * np.asarray((-1.0, 1.0, -1.0), dtype=np.float32)
    left = np.stack(
        (forward[:, 2], forward[:, 1], -forward[:, 0]), axis=1)
    right = np.stack(
        (-forward[:, 2], forward[:, 1], forward[:, 0]), axis=1)
    expected = np.stack((forward, backward, left, right), axis=1)

    header = (
        "# execution_profile timing_profile starting_tick yaw grounded "
        "force knockback nominal_dt loaded_dt air air_max ground ground_max "
        "threshold style native_horizontal native_movement native_ground "
        "native_air_min native_air_max native_speed_min native_speed_max "
        "native_reference native_deadzone removal complete displacement[12]"
    )
    lines = [header]
    for index in range(count):
        scalars = (
            profile[index], timing_profile[index], starting_tick[index],
            yaw[index], int(grounded[index]), force[index], knockback[index],
            dt[index], loaded_dt[index], configured_air[index],
            configured_air_max[index], configured_ground[index],
            configured_ground_max[index], threshold[index], style[index],
            native_horizontal[index], native_movement[index],
            native_ground[index], native_air_min[index], native_air_max[index],
            native_speed_min[index], native_speed_max[index],
            native_reference[index], native_deadzone[index], removal[index],
            int(complete[index]), *expected[index].reshape(-1),
        )
        lines.append("\t".join(str(value) for value in scalars))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--count", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0xD0D6E6)
    arguments = parser.parse_args()
    if arguments.oracle is None:
        parser.error("--oracle is required for direct invocation")
    write_randomized_oracle(
        arguments.oracle,
        count=arguments.count,
        seed=arguments.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
