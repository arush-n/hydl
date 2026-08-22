"""Export raw state inputs and the derived fields JAX computes from them.

`AssembleTest` certifies the step *after* this one: 37 pre-computed structured
fields -> 8,271 columns. This exports the step before it -- the raw simulation
quantities plus what the shared projector derives from them -- so the derivation
itself can be certified rather than assumed.

Round 1 covers `self_f32` (`observation/encoder.py:_self_features`), the 16
columns describing the agent's own body. Every input is something a Java server
can read directly from its components, which is what makes the port possible at
all: velocity, yaw, pitch, health, grounded, attack state, and a handful of
scalar `CombatParams`.

States come from a real rollout rather than being synthesised, so the columns
carry realistic magnitudes and the clip at +/-1 is actually exercised.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

JVM_AGENT_ROOT = Path(__file__).resolve().parents[2]
OUT = JVM_AGENT_ROOT / "case" / "projection"
PROFILE, SEED, BATCH, STEPS = "iron_sword", 570057, 4, 24


def _movement_fixture() -> list[str]:
    """Exercise every native movement-state bit and both availability states."""

    from hytalegym.jax.world import (
        MOVEMENT_STATE_COUNT,
        MOVEMENT_STATE_ORDER,
        native_movement_state_evidence,
    )

    all_bits = (1 << MOVEMENT_STATE_COUNT) - 1
    cases: list[tuple[str, int, bool]] = [("missing", 0, False)]
    cases.extend(
        (f"bit_{index:02d}_{name}", 1 << index, True)
        for index, name in enumerate(MOVEMENT_STATE_ORDER)
    )
    cases.extend(
        (
            ("all", all_bits, True),
            # Missing evidence with asserted state is non-canonical.
            ("invalid_missing_nonzero", 1, False),
            ("invalid_negative", -1, True),
            ("invalid_high_bit", 1 << MOVEMENT_STATE_COUNT, True),
        )
    )

    rows: list[tuple[str, int, bool, np.ndarray, np.ndarray, bool]] = []
    for label, bits, available in cases:
        evidence = native_movement_state_evidence(
            jnp.asarray([bits], dtype=jnp.int32),
            jnp.asarray([available], dtype=jnp.bool_),
        )
        rows.append(
            (
                label,
                bits,
                available,
                np.asarray(evidence.values[0], dtype=np.float32),
                # The low-level World helper deliberately carries row-level
                # availability as [B, 1]; the native observation codec fans it
                # out to all 23 learner fields. Export that policy-facing shape.
                np.broadcast_to(
                    np.asarray(evidence.available[0], dtype=np.bool_),
                    (MOVEMENT_STATE_COUNT,),
                ).copy(),
                bool(np.asarray(evidence.invalid[0])),
            )
        )

    values = np.stack([row[3] for row in rows])
    masks = np.stack([row[4] for row in rows])
    value_spread = values.max(axis=0) - values.min(axis=0)
    mask_spread = masks.astype(np.int8).max(axis=0) - masks.astype(np.int8).min(axis=0)
    if values.shape[1] != len(MOVEMENT_STATE_ORDER):
        raise SystemExit(
            "movement fixture width differs from MOVEMENT_STATE_ORDER"
        )
    if not np.all(value_spread > 0):
        constant = np.flatnonzero(value_spread <= 0).tolist()
        raise SystemExit(
            f"movement fixture leaves value columns constant: {constant}"
        )
    if not np.all(mask_spread > 0):
        constant = np.flatnonzero(mask_spread <= 0).tolist()
        raise SystemExit(
            f"movement fixture leaves mask columns constant: {constant}"
        )
    if {row[5] for row in rows} != {False, True}:
        raise SystemExit("movement fixture never exercises invalid evidence")

    lines = [
        "# label\tbits\trow_available\texpected_values[23]"
        "\texpected_mask[23]\texpected_invalid"
    ]
    for label, bits, available, expected, mask, invalid in rows:
        lines.append(
            "\t".join(
                (
                    label,
                    str(bits),
                    str(int(available)),
                    ",".join(repr(float(value)) for value in expected),
                    ",".join(str(int(value)) for value in mask),
                    str(int(invalid)),
                )
            )
        )
    return lines


def _actor_world_fixture(state, params, config) -> list[str]:
    """Drive every actor-world value and availability channel through JAX."""

    from hytalegym.jax.combat import empty_injected_world_features
    from hytalegym.jax.combat.observation.v3.encoding.encoder import (
        project_learner_observation_v3_actor_evidence,
    )
    from hytalegym.jax.training import open_flat_arsenal_world_capabilities

    batch = state.combat.health.shape[0]
    rows = []
    heights = (-2.0, 0.0, 0.5, 4.0, 12.0)
    maximums = (0.5, 4.0, 8.0)
    for index in range(32):
        controller_available = index % 2 == 0
        controller_in_fluid = (index // 2) % 2 == 0
        submersion_available = (index // 3) % 2 == 0
        feet_submerged = (index // 4) % 2 == 0
        eyes_submerged = (index // 5) % 2 == 0
        drop_available = (index // 6) % 2 == 0
        support_found = (index // 7) % 2 == 0
        drop_height = heights[index % len(heights)]
        maximum_drop_height = maximums[index % len(maximums)]
        case_params = params._replace(
            agent_walk_max_drop_height=jnp.float32(maximum_drop_height)
        )
        capabilities = open_flat_arsenal_world_capabilities(
            state, case_params, config=config
        )._replace(
            actor_controller_medium_available=jnp.full(
                (batch,), controller_available, dtype=jnp.bool_
            ),
            actor_controller_in_fluid=jnp.full(
                (batch,), controller_in_fluid, dtype=jnp.bool_
            ),
            actor_submersion_available=jnp.full(
                (batch,), submersion_available, dtype=jnp.bool_
            ),
            actor_feet_submerged=jnp.full(
                (batch,), feet_submerged, dtype=jnp.bool_
            ),
            actor_eyes_submerged=jnp.full(
                (batch,), eyes_submerged, dtype=jnp.bool_
            ),
            actor_drop_available=jnp.full(
                (batch,), drop_available, dtype=jnp.bool_
            ),
            actor_drop_support_found=jnp.full(
                (batch,), support_found, dtype=jnp.bool_
            ),
            actor_drop_height=jnp.full(
                (batch,), drop_height, dtype=jnp.float32
            ),
        )
        evidence = project_learner_observation_v3_actor_evidence(
            state,
            case_params,
            empty_injected_world_features(batch),
            capabilities,
            config,
        )
        rows.append(
            (
                index,
                controller_available,
                controller_in_fluid,
                submersion_available,
                feet_submerged,
                eyes_submerged,
                drop_available,
                support_found,
                drop_height,
                maximum_drop_height,
                np.asarray(evidence.actor_world_f32[0], dtype=np.float32),
                np.asarray(evidence.actor_world_mask[0], dtype=np.bool_),
            )
        )

    values = np.stack([row[10] for row in rows])
    masks = np.stack([row[11] for row in rows])
    value_spread = values.max(axis=0) - values.min(axis=0)
    mask_spread = masks.astype(np.int8).max(axis=0) - masks.astype(np.int8).min(axis=0)
    if values.shape[1] != 5 or masks.shape[1] != 3:
        raise SystemExit("actor-world fixture contract width drift")
    if not np.all(value_spread > 0):
        raise SystemExit(
            "actor-world fixture leaves value columns constant: "
            f"{np.flatnonzero(value_spread <= 0).tolist()}"
        )
    if not np.all(mask_spread > 0):
        raise SystemExit(
            "actor-world fixture leaves mask columns constant: "
            f"{np.flatnonzero(mask_spread <= 0).tolist()}"
        )

    lines = [
        "# case\tcontroller_available\tcontroller_in_fluid"
        "\tsubmersion_available\tfeet_submerged\teyes_submerged"
        "\tdrop_available\tdrop_support_found\tdrop_height"
        "\tmaximum_drop_height\texpected_values[5]\texpected_mask[3]"
    ]
    for row in rows:
        lines.append(
            "\t".join(
                (
                    str(row[0]),
                    *(str(int(value)) for value in row[1:8]),
                    repr(row[8]),
                    repr(row[9]),
                    ",".join(repr(float(value)) for value in row[10]),
                    ",".join(str(int(value)) for value in row[11]),
                )
            )
        )
    return lines


def _resource_fixture(state, params, config) -> list[str]:
    """Drive every entity/resource span, value and availability through JAX."""

    from hytalegym.jax.combat import empty_injected_world_features
    from hytalegym.jax.combat.observation.v3.encoding.encoder import (
        project_learner_observation_v3_actor_evidence,
    )
    from hytalegym.jax.training import open_flat_arsenal_world_capabilities

    shape = tuple(state.mechanics.resources.shape)
    if shape[1:] != (2, 7):
        raise SystemExit(f"resource fixture expected [B,2,7], got {shape}")
    batch = shape[0]
    flat_index = np.arange(14, dtype=np.float32).reshape(2, 7)
    rows = []
    for case in range(32):
        minimum_row = -((flat_index % 3.0) + 1.0)
        span_row = np.where(
            ((case + flat_index.astype(np.int32)) % 4) == 0,
            np.float32(0.0),
            ((flat_index % 5.0) + 1.0) * np.float32(1 + case % 3),
        ).astype(np.float32)
        maximum_row = minimum_row + span_row
        fraction = (
            ((case * 3 + flat_index.astype(np.int32)) % 13).astype(np.float32)
            - np.float32(2.0)
        ) / np.float32(8.0)
        current_row = minimum_row + span_row * fraction
        current = np.broadcast_to(current_row, shape).copy()
        minimum = np.broadcast_to(minimum_row, shape).copy()
        maximum = np.broadcast_to(maximum_row, shape).copy()
        case_state = state._replace(
            mechanics=state.mechanics._replace(
                resources=jnp.asarray(current, dtype=jnp.float32)
            )
        )
        case_rules = config.mechanics_rules._replace(
            resource_minimum=jnp.asarray(minimum, dtype=jnp.float32),
            resource_maximum=jnp.asarray(maximum, dtype=jnp.float32),
        )
        case_config = config._replace(mechanics_rules=case_rules)
        capabilities = open_flat_arsenal_world_capabilities(
            case_state, params, config=case_config
        )
        evidence = project_learner_observation_v3_actor_evidence(
            case_state,
            params,
            empty_injected_world_features(batch),
            capabilities,
            case_config,
        )
        rows.append(
            (
                case,
                current_row.reshape(-1),
                minimum_row.reshape(-1),
                maximum_row.reshape(-1),
                np.asarray(evidence.resource_f32[0], dtype=np.float32).reshape(-1),
                np.asarray(evidence.resource_mask[0], dtype=np.bool_).reshape(-1),
            )
        )

    values = np.stack([row[4] for row in rows])
    masks = np.stack([row[5] for row in rows])
    value_spread = values.max(axis=0) - values.min(axis=0)
    mask_spread = masks.astype(np.int8).max(axis=0) - masks.astype(np.int8).min(axis=0)
    if values.shape[1] != 14 or masks.shape[1] != 14:
        raise SystemExit("resource fixture contract width drift")
    if not np.all(value_spread > 0):
        raise SystemExit(
            "resource fixture leaves value columns constant: "
            f"{np.flatnonzero(value_spread <= 0).tolist()}"
        )
    if not np.all(mask_spread > 0):
        raise SystemExit(
            "resource fixture leaves mask columns constant: "
            f"{np.flatnonzero(mask_spread <= 0).tolist()}"
        )

    lines = [
        "# case\tcurrent[14]\tminimum[14]\tmaximum[14]"
        "\texpected_values[14]\texpected_mask[14]"
    ]
    for row in rows:
        lines.append(
            "\t".join(
                (
                    str(row[0]),
                    *(
                        ",".join(repr(float(value)) for value in array)
                        for array in row[1:5]
                    ),
                    ",".join(str(int(value)) for value in row[5]),
                )
            )
        )
    return lines


def _defense_fixture(state, params, config) -> list[str]:
    """Exercise all seven defensive columns for both entity rows."""

    from hytalegym.jax.combat import empty_injected_world_features
    from hytalegym.jax.combat.observation.v3.encoding.encoder import (
        project_learner_observation_v3_actor_evidence,
    )
    from hytalegym.jax.training import open_flat_arsenal_world_capabilities

    batch = state.combat.health.shape[0]
    rows = []
    for case in range(32):
        entity = np.arange(2, dtype=np.float32)
        guard = ((case + entity.astype(np.int32)) % 2) == 0
        broken = ((case // 2 + entity.astype(np.int32)) % 2) == 0
        remaining = ((case + 2 * entity) % 9).astype(np.float32) / 4.0
        duration = (0.25 + ((case + entity) % 5) * 0.5).astype(np.float32)
        applied = np.stack(
            (
                ((case % 7) - 3) * (entity + 1.0),
                ((case % 5) - 2) * (entity + 0.5),
                ((case % 11) - 5) * (2.0 - entity),
            ),
            axis=1,
        ).astype(np.float32)
        force = (0.5 + ((case + entity) % 6)).astype(np.float32)
        regen = (-4.0 + ((case + 3 * entity) % 14) * 0.5).astype(np.float32)
        immunity = (-20.0 + ((case * 9 + entity * 17) % 150)).astype(np.float32)
        health = np.where(
            ((case + entity.astype(np.int32)) % 5) == 0,
            np.float32(0.0),
            np.float32(10.0 + case) + entity,
        ).astype(np.float32)

        def batch_rows(row):
            return jnp.asarray(
                np.broadcast_to(row, (batch,) + row.shape), dtype=row.dtype
            )

        case_state = state._replace(
            combat=state.combat._replace(health=batch_rows(health)),
            mechanics=state.mechanics._replace(
                guard_active=batch_rows(guard),
                stamina_broken=batch_rows(broken),
                dodge_invulnerability_remaining_seconds=batch_rows(remaining),
                applied_velocity=batch_rows(applied),
                stamina_regen_delay_seconds=batch_rows(regen),
                control_immunity=batch_rows(immunity),
            ),
        )
        case_rules = config.mechanics_rules._replace(
            dodge_invulnerability_seconds=batch_rows(duration),
            dodge_force=batch_rows(force),
        )
        case_config = config._replace(mechanics_rules=case_rules)
        capabilities = open_flat_arsenal_world_capabilities(
            case_state, params, config=case_config
        )
        evidence = project_learner_observation_v3_actor_evidence(
            case_state,
            params,
            empty_injected_world_features(batch),
            capabilities,
            case_config,
        )
        rows.append(
            (
                case,
                guard,
                broken,
                remaining,
                duration,
                applied.reshape(-1),
                force,
                regen,
                immunity,
                health,
                np.asarray(evidence.defense_f32[0], dtype=np.float32).reshape(-1),
            )
        )

    expected = np.stack([row[10] for row in rows])
    spread = expected.max(axis=0) - expected.min(axis=0)
    if expected.shape[1] != 14:
        raise SystemExit("defense fixture contract width drift")
    if not np.all(spread > 0):
        raise SystemExit(
            "defense fixture leaves columns constant: "
            f"{np.flatnonzero(spread <= 0).tolist()}"
        )
    lines = [
        "# case\tguard[2]\tstamina_broken[2]\tdodge_remaining[2]"
        "\tdodge_duration[2]\tapplied_velocity[6]\tdodge_force[2]"
        "\tstamina_regen_delay[2]\tcontrol_immunity[2]\thealth[2]"
        "\texpected[14]"
    ]
    for row in rows:
        arrays = []
        for array in row[1:]:
            values = np.asarray(array).reshape(-1)
            if values.dtype == np.bool_:
                arrays.append(",".join(str(int(value)) for value in values))
            else:
                arrays.append(",".join(repr(float(value)) for value in values))
        lines.append("\t".join((str(row[0]), *arrays)))
    return lines


def _status_fixture(state, params, config) -> list[str]:
    """Exercise all status slots, masks and six normalized feature columns."""

    from hytalegym.jax.combat import empty_injected_world_features
    from hytalegym.jax.combat.observation.v3.encoding.encoder import (
        project_learner_observation_v3_actor_evidence,
    )
    from hytalegym.jax.training import open_flat_arsenal_world_capabilities

    batch = state.combat.health.shape[0]
    slot_index = np.arange(16, dtype=np.float32).reshape(2, 8)
    rows = []
    for case in range(40):
        active = ((case + slot_index.astype(np.int32)) % 3) != 0
        remaining = ((case * 7 + slot_index * 3) % 61).astype(np.float32)
        cycle_cooldown = np.where(
            ((case + slot_index.astype(np.int32)) % 5) == 0,
            np.float32(0.0),
            np.float32(0.25) + ((case + slot_index) % 7) * np.float32(0.5),
        ).astype(np.float32)
        cycle_elapsed = (
            ((case * 2 + slot_index) % 11) * np.float32(0.3)
        ).astype(np.float32)
        damage = (((case + slot_index * 2) % 17) - 3.0).astype(np.float32)
        healing = (((case * 3 + slot_index) % 13) - 2.0).astype(np.float32)
        resource_id = (
            ((case + slot_index.astype(np.int32)) % 10) - 2
        ).astype(np.int32)
        resource_delta = (
            ((case * 5 + slot_index * 3) % 23) - 11.0
        ).astype(np.float32)
        speed = (
            ((case * 2 + slot_index) % 9) * np.float32(0.4)
        ).astype(np.float32)
        maximum_health = np.asarray(
            (50.0 + case % 9, 31.0 + (case * 2) % 11), dtype=np.float32
        )
        resource_span = (
            np.float32(0.5)
            + ((case + np.arange(14).reshape(2, 7)) % 8).astype(np.float32)
        )

        def batch_rows(row, dtype=None):
            return jnp.asarray(
                np.broadcast_to(row, (batch,) + row.shape),
                dtype=dtype if dtype is not None else row.dtype,
            )

        statuses = state.mechanics.statuses._replace(
            remaining_seconds=batch_rows(remaining, jnp.float32),
            cycle_elapsed_seconds=batch_rows(cycle_elapsed, jnp.float32),
            cycle_cooldown_seconds=batch_rows(cycle_cooldown, jnp.float32),
            damage_per_cycle=batch_rows(damage, jnp.float32),
            healing_per_cycle=batch_rows(healing, jnp.float32),
            resource_id=batch_rows(resource_id, jnp.int32),
            resource_delta_per_cycle=batch_rows(resource_delta, jnp.float32),
            speed_multiplier=batch_rows(speed, jnp.float32),
            active=batch_rows(active, jnp.bool_),
        )
        case_state = state._replace(
            mechanics=state.mechanics._replace(statuses=statuses)
        )
        minimum = np.zeros((batch, 2, 7), dtype=np.float32)
        maximum = np.broadcast_to(resource_span, (batch, 2, 7)).copy()
        case_rules = config.mechanics_rules._replace(
            resource_minimum=jnp.asarray(minimum),
            resource_maximum=jnp.asarray(maximum),
        )
        case_config = config._replace(mechanics_rules=case_rules)
        case_params = params._replace(
            agent_max_health=jnp.float32(maximum_health[0]),
            target_max_health=jnp.float32(maximum_health[1]),
        )
        capabilities = open_flat_arsenal_world_capabilities(
            case_state, case_params, config=case_config
        )
        evidence = project_learner_observation_v3_actor_evidence(
            case_state,
            case_params,
            empty_injected_world_features(batch),
            capabilities,
            case_config,
        )
        rows.append(
            (
                case,
                maximum_health,
                resource_span.reshape(-1),
                remaining.reshape(-1),
                cycle_elapsed.reshape(-1),
                cycle_cooldown.reshape(-1),
                damage.reshape(-1),
                healing.reshape(-1),
                resource_id.reshape(-1),
                resource_delta.reshape(-1),
                speed.reshape(-1),
                active.reshape(-1),
                np.asarray(evidence.status_f32[0], dtype=np.float32).reshape(-1),
                np.asarray(evidence.status_mask[0], dtype=np.bool_).reshape(-1),
            )
        )

    expected = np.stack([row[12] for row in rows])
    masks = np.stack([row[13] for row in rows])
    spread = expected.max(axis=0) - expected.min(axis=0)
    mask_spread = masks.astype(np.int8).max(axis=0) - masks.astype(np.int8).min(axis=0)
    if expected.shape[1] != 96 or masks.shape[1] != 16:
        raise SystemExit("status fixture contract width drift")
    if not np.all(spread > 0):
        raise SystemExit(
            "status fixture leaves columns constant: "
            f"{np.flatnonzero(spread <= 0).tolist()}"
        )
    if not np.all(mask_spread > 0):
        raise SystemExit(
            "status fixture leaves masks constant: "
            f"{np.flatnonzero(mask_spread <= 0).tolist()}"
        )

    lines = [
        "# case\tmaximum_health[2]\tresource_span[14]\tremaining[16]"
        "\tcycle_elapsed[16]\tcycle_cooldown[16]\tdamage[16]"
        "\thealing[16]\tresource_id[16]\tresource_delta[16]"
        "\tspeed_multiplier[16]\tactive[16]\texpected[96]"
        "\texpected_mask[16]"
    ]
    for row in rows:
        arrays = []
        for array in row[1:]:
            values = np.asarray(array).reshape(-1)
            if values.dtype == np.bool_ or np.issubdtype(values.dtype, np.integer):
                arrays.append(",".join(str(int(value)) for value in values))
            else:
                arrays.append(",".join(repr(float(value)) for value in values))
        lines.append("\t".join((str(row[0]), *arrays)))
    return lines


def _dodge_fixture(state, params, config) -> list[str]:
    """Exercise native-authored directions and every actor-readiness gate."""

    from hytalegym.jax.combat import empty_injected_world_features
    from hytalegym.jax.combat.mechanics import (
        RESOURCE_STAMINA,
        STATUS_FLAG_DISABLE_MOVEMENT,
    )
    from hytalegym.jax.combat.observation.v3.encoding.encoder import (
        project_learner_observation_v3_actor_evidence,
    )
    from hytalegym.jax.training import open_flat_arsenal_world_capabilities

    batch = state.combat.health.shape[0]
    cases: list[tuple[str, np.ndarray, bool, bool, float, float]] = []
    for bits in range(16):
        cases.append(
            (
                f"ready_corridor_{bits:04b}",
                np.asarray([(bits & (1 << i)) != 0 for i in range(4)]),
                True,
                True,
                2.0,
                2.0,
            )
        )
    cases.extend(
        (
            ("movement_disabled", np.ones(4, dtype=np.bool_), False, True, 3.0, 2.0),
            ("dead", np.ones(4, dtype=np.bool_), True, False, 3.0, 2.0),
            ("stamina_below", np.ones(4, dtype=np.bool_), True, True, 1.999, 2.0),
            ("stamina_equal", np.ones(4, dtype=np.bool_), True, True, 2.0, 2.0),
            ("zero_cost", np.ones(4, dtype=np.bool_), True, True, 0.0, 0.0),
        )
    )

    rows = []
    for label, corridor, movement_enabled, alive, stamina, cost in cases:
        resources = np.asarray(state.mechanics.resources).copy()
        resources[:, 0, RESOURCE_STAMINA] = np.float32(stamina)
        health = np.asarray(state.combat.health).copy()
        health[:, 0] = np.float32(10.0 if alive else 0.0)
        flags = np.zeros_like(np.asarray(state.mechanics.statuses.flags))
        active = np.zeros_like(np.asarray(state.mechanics.statuses.active))
        if not movement_enabled:
            flags[:, 0, 0] = np.uint32(STATUS_FLAG_DISABLE_MOVEMENT)
            active[:, 0, 0] = True
        statuses = state.mechanics.statuses._replace(
            flags=jnp.asarray(flags), active=jnp.asarray(active)
        )
        case_state = state._replace(
            combat=state.combat._replace(health=jnp.asarray(health)),
            mechanics=state.mechanics._replace(
                resources=jnp.asarray(resources), statuses=statuses
            ),
        )
        dodge_cost = np.asarray(config.mechanics_rules.dodge_cost).copy()
        dodge_cost[:, 0] = np.float32(cost)
        case_rules = config.mechanics_rules._replace(
            dodge_cost=jnp.asarray(dodge_cost)
        )
        case_config = config._replace(mechanics_rules=case_rules)
        capabilities = open_flat_arsenal_world_capabilities(
            case_state, params, config=case_config
        )
        corridors = np.asarray(capabilities.dodge_corridor_clear).copy()
        corridors[:, 0, :] = corridor
        capabilities = capabilities._replace(
            dodge_corridor_clear=jnp.asarray(corridors)
        )
        evidence = project_learner_observation_v3_actor_evidence(
            case_state,
            params,
            empty_injected_world_features(batch),
            capabilities,
            case_config,
        )
        rows.append(
            (
                label,
                corridor,
                movement_enabled,
                alive,
                stamina,
                cost,
                np.asarray(evidence.dodge_action_mask[0], dtype=np.bool_),
            )
        )

    corridors = np.stack([row[1] for row in rows]).astype(np.int8)
    expected = np.stack([row[6] for row in rows]).astype(np.int8)
    if expected.shape[1] != 4:
        raise SystemExit("dodge fixture contract width drift")
    if not np.all(corridors.max(axis=0) > corridors.min(axis=0)):
        raise SystemExit("dodge fixture leaves a corridor input constant")
    # Native NPCs author only left/right. Forward/back must stay closed, while
    # both authored columns must vary or the fixture is vacuous.
    if np.any(expected[:, :2]) or not np.all(
        expected[:, 2:].max(axis=0) > expected[:, 2:].min(axis=0)
    ):
        raise SystemExit("dodge fixture does not exercise the authored mask")

    lines = [
        "# case\tcorridor_clear[4]\tmovement_enabled\talive\tstamina"
        "\tdodge_cost\texpected_mask[4]"
    ]
    for label, corridor, movement_enabled, alive, stamina, cost, mask in rows:
        lines.append(
            "\t".join(
                (
                    label,
                    ",".join(str(int(value)) for value in corridor),
                    str(int(movement_enabled)),
                    str(int(alive)),
                    repr(float(stamina)),
                    repr(float(cost)),
                    ",".join(str(int(value)) for value in mask),
                )
            )
        )
    return lines


def _ability_fixture(state, params, config) -> list[str]:
    """Exercise all 32 slots and all 11 dynamic ability feature columns."""

    from hytalegym.jax.combat import empty_injected_world_features
    from hytalegym.jax.combat.observation.v3.encoding.encoder import (
        project_learner_observation_v3_actor_evidence,
    )
    from hytalegym.jax.training import open_flat_arsenal_world_capabilities

    batch = state.combat.health.shape[0]
    slot = np.arange(32, dtype=np.int32).reshape(2, 16)
    ability_resource = np.arange(224, dtype=np.int32).reshape(2, 16, 7)
    rows = []
    for case in range(40):
        # Case 0 makes every slot structurally present and legal. Case 1 masks
        # all of them. Together they force every mask/legal bit to vary; the
        # remaining cases exercise the dynamic numeric paths.
        authored_mask = np.ones((2, 16), dtype=np.bool_)
        if case == 1:
            authored_mask[:] = False
        elif case > 1:
            authored_mask = ((slot + case) % 4) != 0
        equipped = np.ones((2,), dtype=np.bool_)
        overflow = np.zeros((2,), dtype=np.bool_)
        if case == 2:
            equipped[0] = False
        if case == 3:
            overflow[1] = True

        duration = (
            np.float32(0.05)
            + ((slot * 3 + case * 5) % 37).astype(np.float32)
            * np.float32(0.11)
        )
        authored_cooldown = (
            np.float32(0.25)
            + ((slot + case * 2) % 13).astype(np.float32)
            * np.float32(0.35)
        )
        remaining_cooldown = (
            ((slot * 2 + case * 3) % 19).astype(np.float32)
            * np.float32(0.4)
        )
        if case == 0:
            remaining_cooldown[:] = 0.0

        active_slot = np.full((2,), -1, dtype=np.int32)
        if 8 <= case < 24:
            active_slot[0] = case - 8
            active_slot[1] = (case - 1) % 16
            authored_mask[0, active_slot[0]] = True
            authored_mask[1, active_slot[1]] = True
        elapsed = np.asarray(
            (0.2 + case * 0.17, 0.4 + case * 0.13), dtype=np.float32
        )

        resource_span = (
            np.float32(0.5)
            + ((np.arange(14).reshape(2, 7) + case) % 9).astype(np.float32)
        )
        if case % 7 == 0 and case != 0:
            resource_span[:, case % 7] = 0.0
        resources = (
            np.float32(0.25)
            + ((np.arange(14).reshape(2, 7) * 2 + case) % 11).astype(np.float32)
        )
        minimum = (
            ((ability_resource + case * 3) % 12).astype(np.float32)
            * np.float32(0.6)
        )
        if case == 0:
            resources[:] = 100.0
            minimum[:] = 0.0
        authored_cost = (
            np.float32(0.1)
            + ((ability_resource * 3 + case) % 17).astype(np.float32)
            * np.float32(0.2)
        )
        cost_kind = ((ability_resource + case) % 4).astype(np.int32)

        def batch_rows(row, dtype=None):
            return jnp.asarray(
                np.broadcast_to(row, (batch,) + row.shape),
                dtype=dtype if dtype is not None else row.dtype,
            )

        loadout = config.loadout._replace(
            equipped=batch_rows(equipped, jnp.bool_),
            ability_id=batch_rows(
                np.zeros((2, 16), dtype=np.int32), jnp.int32
            ),
            ability_mask=batch_rows(authored_mask, jnp.bool_),
            ability_evidence=batch_rows(
                np.zeros((2, 16), dtype=np.int32), jnp.int32
            ),
            ability_duration_seconds=batch_rows(duration, jnp.float32),
            ability_cooldown_seconds=batch_rows(
                authored_cooldown, jnp.float32
            ),
            ability_scheduler_prelude_ticks=batch_rows(
                np.zeros((2, 16), dtype=np.int32), jnp.int32
            ),
            ability_resource_cost=batch_rows(authored_cost, jnp.float32),
            ability_resource_cost_kind=batch_rows(cost_kind, jnp.int32),
            ability_resource_minimum=batch_rows(minimum, jnp.float32),
            ability_requirements=batch_rows(
                np.zeros((2, 16), dtype=np.uint32), jnp.uint32
            ),
            overflow=batch_rows(overflow, jnp.bool_),
        )
        minimum_resources = np.zeros((2, 7), dtype=np.float32)
        rules = config.mechanics_rules._replace(
            resource_minimum=batch_rows(minimum_resources, jnp.float32),
            resource_maximum=batch_rows(resource_span, jnp.float32),
        )
        case_config = config._replace(loadout=loadout, mechanics_rules=rules)

        health = np.full_like(np.asarray(state.combat.health), 100.0)
        statuses = state.mechanics.statuses._replace(
            flags=jnp.zeros_like(state.mechanics.statuses.flags),
            active=jnp.zeros_like(state.mechanics.statuses.active),
        )
        mechanics = state.mechanics._replace(
            resources=batch_rows(resources, jnp.float32),
            guard_held=jnp.zeros_like(state.mechanics.guard_held),
            statuses=statuses,
            failure_bits=jnp.zeros_like(state.mechanics.failure_bits),
        )
        arsenal = state.arsenal._replace(
            active_ability_slot=batch_rows(active_slot, jnp.int32),
            ability_elapsed_seconds=batch_rows(elapsed, jnp.float32),
            ability_scheduler_tick=jnp.full_like(
                state.arsenal.ability_scheduler_tick, 100
            ),
            ability_cooldown_seconds=batch_rows(
                remaining_cooldown, jnp.float32
            ),
            ability_charge_count=jnp.ones_like(
                batch_rows(np.ones((2, 16), dtype=np.int32), jnp.int32)
            ),
            ability_charge_timer_seconds=batch_rows(
                np.zeros((2, 16), dtype=np.float32), jnp.float32
            ),
            failure_bits=jnp.zeros_like(state.arsenal.failure_bits),
        )
        case_state = state._replace(
            combat=state.combat._replace(health=jnp.asarray(health)),
            mechanics=mechanics,
            arsenal=arsenal,
        )
        capabilities = open_flat_arsenal_world_capabilities(
            case_state, params, config=case_config
        )
        evidence = project_learner_observation_v3_actor_evidence(
            case_state,
            params,
            empty_injected_world_features(batch),
            capabilities,
            case_config,
        )
        expected = np.asarray(evidence.ability_f32[0], dtype=np.float32).reshape(-1)
        expected_mask = np.asarray(
            evidence.ability_mask[0], dtype=np.bool_
        ).reshape(-1)
        authoritative_legal = np.asarray(
            evidence.ability_legal[0], dtype=np.bool_
        ).reshape(-1)
        rows.append(
            (
                case,
                resources.reshape(-1),
                resource_span.reshape(-1),
                duration.reshape(-1),
                authored_cooldown.reshape(-1),
                remaining_cooldown.reshape(-1),
                active_slot,
                elapsed,
                authored_cost.reshape(-1),
                cost_kind.reshape(-1),
                minimum.reshape(-1),
                authored_mask.reshape(-1),
                equipped,
                overflow,
                authoritative_legal,
                expected,
                expected_mask,
            )
        )

    expected = np.stack([row[15] for row in rows])
    masks = np.stack([row[16] for row in rows]).astype(np.int8)
    legal = np.stack([row[14] for row in rows]).astype(np.int8)
    if expected.shape[1] != 352 or masks.shape[1] != 32 or legal.shape[1] != 32:
        raise SystemExit("ability fixture contract width drift")
    spread = expected.max(axis=0) - expected.min(axis=0)
    if not np.all(spread > 0):
        raise SystemExit(
            "ability fixture leaves value columns constant: "
            f"{np.flatnonzero(spread <= 0).tolist()}"
        )
    if not np.all(masks.max(axis=0) > masks.min(axis=0)):
        raise SystemExit("ability fixture leaves structural masks constant")
    if not np.all(legal.max(axis=0) > legal.min(axis=0)):
        raise SystemExit("ability fixture leaves legality bits constant")

    lines = [
        "# case\tresources[14]\tresource_span[14]\tduration[32]"
        "\tauthored_cooldown[32]\tremaining_cooldown[32]"
        "\tactive_slot[2]\telapsed[2]\tauthored_cost[224]"
        "\tcost_kind[224]\tresource_minimum[224]\tauthored_mask[32]"
        "\tequipped[2]\toverflow[2]\tauthoritative_legal[32]"
        "\texpected[352]\texpected_mask[32]"
    ]
    for row in rows:
        fields = [str(row[0])]
        for array in row[1:]:
            values = np.asarray(array).reshape(-1)
            if values.dtype == np.bool_ or np.issubdtype(values.dtype, np.integer):
                fields.append(",".join(str(int(value)) for value in values))
            else:
                fields.append(",".join(repr(float(value)) for value in values))
        lines.append("\t".join(fields))
    return lines


def _inventory_fixture() -> list[str]:
    """Exercise every semantic inventory token and container validity gate."""

    from hytalegym.jax.combat.observation.v3.tokens.inventory import (
        encode_inventory_policy_tokens,
    )
    from hytalegym.jax.world import ActorInventoryTokens

    token_index = np.arange(76, dtype=np.int32)
    defaults = np.asarray((36, 4, 9, 4, 23, 0), dtype=np.int32)
    rows = []
    for case in range(82):
        available = case != 1
        container_available = (
            np.ones((6,), dtype=np.bool_)
            if case < 3
            else ((np.arange(6) + case) % 4 != 0)
        )
        container_capacity = defaults.copy()
        for container in range(5):
            container_capacity[container] = max(
                0, int(defaults[container]) - ((case + container) % 3)
            )
        container_capacity[5] = case + 1
        if case == 2:
            container_capacity[0] = defaults[0] + 1

        token_mask = (
            np.ones((76,), dtype=np.bool_)
            if case == 0
            else ((token_index + case) % 3 != 0)
        )
        container_id = ((token_index + case + 1) % 6).astype(np.int32)
        selected_capacity = container_capacity[
            np.clip(container_id, 0, 5)
        ]
        container_slot = np.mod(
            token_index + case,
            np.maximum(selected_capacity, 1),
        ).astype(np.int32)
        item_id = (
            (token_index + 1) * np.int32(65537)
            + np.int32(case * 131071)
        ).astype(np.int32)
        quantity = (1 + (token_index * 3 + case) % 4096).astype(np.int32)
        durability = (
            np.float32(0.05)
            + ((token_index * 5 + case) % 19).astype(np.float32)
            * np.float32(0.05)
        )
        active = (
            np.ones((76,), dtype=np.bool_)
            if case == 0
            else ((token_index + case) % 5 == 0)
        )

        def actor_row(value, dtype=None):
            array = np.asarray(value)
            return jnp.asarray(
                array.reshape((1, 1) + array.shape),
                dtype=dtype if dtype is not None else array.dtype,
            )

        source = ActorInventoryTokens(
            available=jnp.asarray([[available]], dtype=jnp.bool_),
            capacity_exceeded=jnp.zeros((1, 1), dtype=jnp.bool_),
            diagnostics=jnp.zeros((1, 1), dtype=jnp.uint32),
            container_available=actor_row(container_available, jnp.bool_),
            container_capacity=actor_row(container_capacity, jnp.int32),
            token_mask=actor_row(token_mask, jnp.bool_),
            source_slot=actor_row(token_index, jnp.int32),
            container_id=actor_row(container_id, jnp.int32),
            container_slot=actor_row(container_slot, jnp.int32),
            item_id=actor_row(item_id, jnp.int32),
            quantity=actor_row(quantity, jnp.int32),
            durability=actor_row(durability, jnp.float32),
            max_durability=actor_row(
                np.ones((76,), dtype=np.float32), jnp.float32
            ),
            durability_fraction=actor_row(durability, jnp.float32),
            metadata_hash=jnp.zeros((1, 1, 76, 2), dtype=jnp.uint32),
            metadata_present=actor_row(
                np.zeros((76,), dtype=np.bool_), jnp.bool_
            ),
            metadata_hash_valid=actor_row(
                np.zeros((76,), dtype=np.bool_), jnp.bool_
            ),
            active=actor_row(active, jnp.bool_),
            provenance=actor_row(
                np.ones((76,), dtype=np.uint8), jnp.uint8
            ),
        )
        expected = encode_inventory_policy_tokens(source)
        rows.append(
            (
                f"case_{case:02d}" if case != 2 else "fixed_over_capacity",
                available,
                container_available,
                container_capacity,
                token_mask,
                container_id,
                container_slot,
                item_id,
                quantity,
                durability,
                active,
                bool(np.asarray(expected.available)[0]),
                np.asarray(expected.container_f32[0], dtype=np.float32),
                np.asarray(expected.container_mask[0], dtype=np.bool_),
                np.asarray(expected.token_f32[0], dtype=np.float32).reshape(-1),
                np.asarray(expected.token_mask[0], dtype=np.bool_),
            )
        )

    available = np.asarray([row[11] for row in rows], dtype=np.int8)
    container_values = np.stack([row[12] for row in rows])
    container_masks = np.stack([row[13] for row in rows]).astype(np.int8)
    token_values = np.stack([row[14] for row in rows])
    token_masks = np.stack([row[15] for row in rows]).astype(np.int8)
    if (
        container_values.shape[1] != 6
        or container_masks.shape[1] != 6
        or token_values.shape[1] != 532
        or token_masks.shape[1] != 76
    ):
        raise SystemExit("inventory fixture contract width drift")
    if not (available.min() == 0 and available.max() == 1):
        raise SystemExit("inventory fixture never varies overall availability")
    if not np.all(container_values.max(axis=0) > container_values.min(axis=0)):
        raise SystemExit("inventory fixture leaves container values constant")
    if not np.all(container_masks.max(axis=0) > container_masks.min(axis=0)):
        raise SystemExit("inventory fixture leaves container masks constant")
    spread = token_values.max(axis=0) - token_values.min(axis=0)
    if not np.all(spread > 0):
        raise SystemExit(
            "inventory fixture leaves token values constant: "
            f"{np.flatnonzero(spread <= 0).tolist()}"
        )
    if not np.all(token_masks.max(axis=0) > token_masks.min(axis=0)):
        raise SystemExit("inventory fixture leaves token masks constant")
    if rows[2][11]:
        raise SystemExit("fixed-container overflow did not fail availability")

    lines = [
        "# case\tavailable\tcontainer_available[6]\tcontainer_capacity[6]"
        "\ttoken_mask[76]\tcontainer_id[76]\tcontainer_slot[76]"
        "\titem_id[76]\tquantity[76]\tdurability_fraction[76]"
        "\tactive[76]\texpected_available\texpected_container_f32[6]"
        "\texpected_container_mask[6]\texpected_token_f32[532]"
        "\texpected_token_mask[76]"
    ]
    for row in rows:
        fields = [str(row[0])]
        for value in row[1:]:
            array = np.asarray(value).reshape(-1)
            if array.dtype == np.bool_ or np.issubdtype(array.dtype, np.integer):
                fields.append(",".join(str(int(item)) for item in array))
            else:
                fields.append(",".join(repr(float(item)) for item in array))
        lines.append("\t".join(fields))
    return lines


def _light_fixture() -> list[str]:
    """Exercise all light columns, validity gates and row-level failures."""

    from hytalegym.jax.combat.observation.v3.tokens.light import (
        align_actor_light_policy_tokens,
        encode_actor_light_policy_tokens,
    )
    from hytalegym.jax.world import ActorBlockLightTokens

    capacity = 44
    rows = []

    def evaluate(
        label: str,
        *,
        source_available: bool,
        source_mask: np.ndarray,
        light_valid: np.ndarray,
        sky: np.ndarray,
        block: np.ndarray,
        tint: np.ndarray,
        geometry_mask: np.ndarray,
        geometry_available: bool,
        actor_valid: bool,
    ) -> None:
        def actor_row(value, dtype):
            array = np.asarray(value)
            return jnp.asarray(
                array.reshape((1, 1) + array.shape), dtype=dtype
            )

        source = ActorBlockLightTokens(
            available=jnp.asarray([[source_available]], dtype=jnp.bool_),
            diagnostics=jnp.zeros((1, 1), dtype=jnp.uint32),
            token_mask=actor_row(source_mask, jnp.bool_),
            light_valid=actor_row(light_valid, jnp.bool_),
            sky_light=actor_row(sky, jnp.uint8),
            block_light_rgb=actor_row(block, jnp.uint8),
            tint_rgb=actor_row(tint, jnp.uint8),
            environment_valid=actor_row(
                np.zeros((capacity,), dtype=np.bool_), jnp.bool_
            ),
            environment_code=actor_row(
                np.zeros((capacity,), dtype=np.uint8), jnp.uint8
            ),
            provenance=actor_row(
                np.ones((capacity,), dtype=np.uint8), jnp.uint8
            ),
        )
        encoded = encode_actor_light_policy_tokens(source)
        expected = align_actor_light_policy_tokens(
            encoded,
            jnp.asarray(geometry_mask[None, :], dtype=jnp.bool_),
            jnp.asarray([geometry_available], dtype=jnp.bool_),
            jnp.asarray([actor_valid], dtype=jnp.bool_),
        )
        rows.append(
            (
                label,
                source_available,
                source_mask,
                light_valid,
                sky,
                block,
                tint,
                geometry_mask,
                geometry_available,
                actor_valid,
                bool(np.asarray(expected.available)[0]),
                np.asarray(expected.token_f32[0], dtype=np.float32).reshape(-1),
                np.asarray(expected.token_mask[0], dtype=np.bool_),
            )
        )

    for token in range(capacity):
        source_mask = np.zeros((capacity,), dtype=np.bool_)
        source_mask[token] = True
        sky = np.zeros((capacity,), dtype=np.uint8)
        block = np.zeros((capacity, 3), dtype=np.uint8)
        tint = np.zeros((capacity, 3), dtype=np.uint8)
        sky[token] = np.uint8(token % 15 + 1)
        block[token] = np.asarray(
            tuple((token + offset) % 15 + 1 for offset in (1, 5, 9)),
            dtype=np.uint8,
        )
        tint[token] = np.asarray(
            tuple((token * 17 + offset) % 255 + 1 for offset in (3, 79, 157)),
            dtype=np.uint8,
        )
        for invalid_channel in (-1, 0, 1, 2):
            light_valid = np.zeros((capacity, 3), dtype=np.bool_)
            light_valid[token] = True
            label = f"token_{token:02d}_all_valid"
            if invalid_channel >= 0:
                light_valid[token, invalid_channel] = False
                label = f"token_{token:02d}_channel_{invalid_channel}_absent"
            evaluate(
                label,
                source_available=True,
                source_mask=source_mask,
                light_valid=light_valid,
                sky=sky,
                block=block,
                tint=tint,
                geometry_mask=source_mask.copy(),
                geometry_available=True,
                actor_valid=True,
            )

    zeros_mask = np.zeros((capacity,), dtype=np.bool_)
    zeros_valid = np.zeros((capacity, 3), dtype=np.bool_)
    zeros_sky = np.zeros((capacity,), dtype=np.uint8)
    zeros_rgb = np.zeros((capacity, 3), dtype=np.uint8)
    one_mask = zeros_mask.copy()
    one_mask[0] = True
    one_valid = zeros_valid.copy()
    one_valid[0] = True

    for label, source_available, geometry_available, actor_valid in (
        ("source_unavailable", False, True, True),
        ("geometry_unavailable", True, False, True),
        ("actor_invalid", True, True, False),
    ):
        evaluate(
            label,
            source_available=source_available,
            source_mask=one_mask,
            light_valid=one_valid,
            sky=zeros_sky,
            block=zeros_rgb,
            tint=zeros_rgb,
            geometry_mask=one_mask.copy(),
            geometry_available=geometry_available,
            actor_valid=actor_valid,
        )

    mismatched_geometry = one_mask.copy()
    mismatched_geometry[0] = False
    mismatched_geometry[1] = True
    evaluate(
        "geometry_mask_mismatch",
        source_available=True,
        source_mask=one_mask,
        light_valid=one_valid,
        sky=zeros_sky,
        block=zeros_rgb,
        tint=zeros_rgb,
        geometry_mask=mismatched_geometry,
        geometry_available=True,
        actor_valid=True,
    )

    malformed_sky = zeros_sky.copy()
    malformed_sky[0] = 16
    evaluate(
        "malformed_sky",
        source_available=True,
        source_mask=one_mask,
        light_valid=one_valid,
        sky=malformed_sky,
        block=zeros_rgb,
        tint=zeros_rgb,
        geometry_mask=one_mask.copy(),
        geometry_available=True,
        actor_valid=True,
    )
    malformed_block = zeros_rgb.copy()
    malformed_block[0, 1] = 16
    evaluate(
        "malformed_block",
        source_available=True,
        source_mask=one_mask,
        light_valid=one_valid,
        sky=zeros_sky,
        block=malformed_block,
        tint=zeros_rgb,
        geometry_mask=one_mask.copy(),
        geometry_available=True,
        actor_valid=True,
    )

    available = np.asarray([row[10] for row in rows], dtype=np.int8)
    values = np.stack([row[11] for row in rows])
    masks = np.stack([row[12] for row in rows]).astype(np.int8)
    if values.shape[1] != capacity * 10 or masks.shape[1] != capacity:
        raise SystemExit("light fixture contract width drift")
    if not (available.min() == 0 and available.max() == 1):
        raise SystemExit("light fixture never varies row availability")
    spread = values.max(axis=0) - values.min(axis=0)
    if not np.all(spread > 0):
        raise SystemExit(
            "light fixture leaves value columns constant: "
            f"{np.flatnonzero(spread <= 0).tolist()}"
        )
    if not np.all(masks.max(axis=0) > masks.min(axis=0)):
        raise SystemExit("light fixture leaves token masks constant")
    failed = {row[0]: row[10] for row in rows}
    for label in (
        "source_unavailable", "geometry_unavailable", "actor_invalid",
        "geometry_mask_mismatch", "malformed_sky", "malformed_block",
    ):
        if failed[label]:
            raise SystemExit(f"light row-level gate stayed open for {label}")

    lines = [
        "# case\tsource_available\tsource_mask[44]\tlight_valid[132]"
        "\tsky_light[44]\tblock_light_rgb[132]\ttint_rgb[132]"
        "\tgeometry_mask[44]\tgeometry_available\tactor_valid"
        "\texpected_available\texpected_token_f32[440]"
        "\texpected_token_mask[44]"
    ]
    for row in rows:
        fields = [str(row[0])]
        for value in row[1:]:
            array = np.asarray(value).reshape(-1)
            if array.dtype == np.bool_ or np.issubdtype(array.dtype, np.integer):
                fields.append(",".join(str(int(item)) for item in array))
            else:
                fields.append(",".join(repr(float(item)) for item in array))
        lines.append("\t".join(fields))
    return lines


def _geometry_fixture() -> list[str]:
    """Exercise all geometry columns, padding, and malformed-row gates."""

    from hytalegym.jax.combat.observation.v3.tokens.world import (
        encode_world_geometry_policy_tokens,
        mask_world_geometry_policy_tokens,
    )
    from hytalegym.jax.world import WorldGeometryTokenObservation

    token_capacity = 44
    rows = []

    def blank(edge_capacity: int = 12, box_capacity: int = 9) -> dict:
        return {
            "source_available": True,
            "token_mask": np.zeros((token_capacity,), dtype=np.bool_),
            "token_kind": np.zeros((token_capacity,), dtype=np.uint8),
            "token_provenance": np.zeros((token_capacity,), dtype=np.uint8),
            "relative_position": np.zeros((token_capacity, 3), dtype=np.float32),
            "clearance": np.zeros((token_capacity,), dtype=np.float32),
            "semantic_flags": np.zeros((token_capacity,), dtype=np.uint16),
            "dynamic_blocked": np.zeros((token_capacity,), dtype=np.bool_),
            "edge_mask": np.zeros(
                (token_capacity, edge_capacity), dtype=np.bool_
            ),
            "edge_destination": np.full(
                (token_capacity, edge_capacity), -1, dtype=np.int32
            ),
            "edge_cost": np.zeros(
                (token_capacity, edge_capacity), dtype=np.float32
            ),
            "edge_kind": np.zeros(
                (token_capacity, edge_capacity), dtype=np.uint8
            ),
            "edge_flags": np.zeros(
                (token_capacity, edge_capacity), dtype=np.uint8
            ),
            "collision_box_mask": np.zeros(
                (token_capacity, box_capacity), dtype=np.bool_
            ),
            "collision_boxes_relative": np.zeros(
                (token_capacity, box_capacity, 6), dtype=np.float32
            ),
            "actor_valid": True,
        }

    def activate(data: dict, token: int = 0) -> None:
        data["token_mask"][token] = True
        data["token_kind"][token] = np.uint8(1 + token % 2)
        data["token_provenance"][token] = np.uint8(1 + token % 3)
        data["relative_position"][token] = np.asarray(
            (
                -(1.0 + token % 7),
                0.25 + token % 9,
                1.5 + token % 11,
            ),
            dtype=np.float32,
        )
        data["clearance"][token] = np.float32(0.5 + token % 8)
        data["semantic_flags"][token] = np.uint16((token + 1) * 1000)
        data["dynamic_blocked"][token] = True
        edge_capacity = data["edge_mask"].shape[1]
        data["edge_mask"][token] = True
        data["edge_destination"][token] = token
        data["edge_cost"][token] = np.asarray(
            tuple(0.25 + edge * 0.75 for edge in range(edge_capacity)),
            dtype=np.float32,
        )
        data["edge_kind"][token] = np.asarray(
            tuple(1 + edge * 7 for edge in range(edge_capacity)),
            dtype=np.uint8,
        )
        data["edge_flags"][token] = np.asarray(
            tuple(2 + edge * 11 for edge in range(edge_capacity)),
            dtype=np.uint8,
        )
        box_capacity = data["collision_box_mask"].shape[1]
        data["collision_box_mask"][token] = True
        for box in range(box_capacity):
            data["collision_boxes_relative"][token, box] = np.asarray(
                (
                    -5.0 - box * 0.1,
                    -3.0 - box * 0.2,
                    -1.0 - box * 0.3,
                    1.0 + box * 0.4,
                    3.0 + box * 0.5,
                    5.0 + box * 0.6,
                ),
                dtype=np.float32,
            )

    def evaluate(label: str, data: dict) -> None:
        def actor_row(value, dtype):
            array = np.asarray(value)
            return jnp.asarray(
                array.reshape((1, 1) + array.shape), dtype=dtype
            )

        source = WorldGeometryTokenObservation(
            available=jnp.asarray(
                [[data["source_available"]]], dtype=jnp.bool_
            ),
            capacity_exceeded=jnp.zeros((1, 1), dtype=jnp.bool_),
            diagnostics=jnp.zeros((1, 1), dtype=jnp.uint32),
            source_available=jnp.ones((1, 1, 2), dtype=jnp.bool_),
            source_overflow=jnp.zeros((1, 1, 2), dtype=jnp.bool_),
            source_tile_index=jnp.zeros((1, 1, 2), dtype=jnp.int32),
            token_mask=actor_row(data["token_mask"], jnp.bool_),
            token_kind=actor_row(data["token_kind"], jnp.uint8),
            token_provenance=actor_row(
                data["token_provenance"], jnp.uint8
            ),
            source_index=actor_row(
                np.arange(token_capacity, dtype=np.int32), jnp.int32
            ),
            relative_position=actor_row(
                data["relative_position"], jnp.float32
            ),
            clearance=actor_row(data["clearance"], jnp.float32),
            semantic_flags=actor_row(data["semantic_flags"], jnp.uint16),
            dynamic_blocked=actor_row(data["dynamic_blocked"], jnp.bool_),
            edge_mask=actor_row(data["edge_mask"], jnp.bool_),
            edge_destination=actor_row(
                data["edge_destination"], jnp.int32
            ),
            edge_cost=actor_row(data["edge_cost"], jnp.float32),
            edge_kind=actor_row(data["edge_kind"], jnp.uint8),
            edge_flags=actor_row(data["edge_flags"], jnp.uint8),
            collision_box_mask=actor_row(
                data["collision_box_mask"], jnp.bool_
            ),
            collision_boxes_relative=actor_row(
                data["collision_boxes_relative"], jnp.float32
            ),
        )
        encoded = encode_world_geometry_policy_tokens(source)
        expected = mask_world_geometry_policy_tokens(
            encoded,
            jnp.asarray([data["actor_valid"]], dtype=jnp.bool_),
        )
        rows.append(
            (
                label,
                data["source_available"],
                data["token_mask"],
                data["token_kind"],
                data["token_provenance"],
                data["relative_position"],
                data["clearance"],
                data["semantic_flags"],
                data["dynamic_blocked"],
                data["edge_mask"].shape[1],
                data["edge_mask"],
                data["edge_destination"],
                data["edge_cost"],
                data["edge_kind"],
                data["edge_flags"],
                data["collision_box_mask"].shape[1],
                data["collision_box_mask"],
                data["collision_boxes_relative"],
                data["actor_valid"],
                bool(np.asarray(expected.available)[0]),
                np.asarray(expected.token_f32[0], dtype=np.float32).reshape(-1),
                np.asarray(expected.token_mask[0], dtype=np.bool_),
            )
        )

    for token in range(token_capacity):
        data = blank()
        activate(data, token)
        evaluate(f"token_{token:02d}_full", data)

    padded = blank(edge_capacity=3, box_capacity=2)
    activate(padded)
    evaluate("short_axes_padded", padded)

    failures = {}
    for label in (
        "source_unavailable", "actor_invalid", "invalid_token_kind",
        "invalid_provenance", "nonfinite_position", "nonfinite_clearance",
        "edge_destination_inactive", "edge_destination_negative",
        "edge_destination_high", "nonfinite_edge_cost", "nonfinite_box",
    ):
        data = blank()
        activate(data)
        if label == "source_unavailable":
            data["source_available"] = False
        elif label == "actor_invalid":
            data["actor_valid"] = False
        elif label == "invalid_token_kind":
            data["token_kind"][0] = 3
        elif label == "invalid_provenance":
            data["token_provenance"][0] = 4
        elif label == "nonfinite_position":
            data["relative_position"][0, 1] = np.nan
        elif label == "nonfinite_clearance":
            data["clearance"][0] = np.inf
        elif label == "edge_destination_inactive":
            data["edge_destination"][0, 0] = 1
        elif label == "edge_destination_negative":
            data["edge_destination"][0, 0] = -1
        elif label == "edge_destination_high":
            data["edge_destination"][0, 0] = token_capacity
        elif label == "nonfinite_edge_cost":
            data["edge_cost"][0, 0] = np.nan
        elif label == "nonfinite_box":
            data["collision_boxes_relative"][0, 0, 4] = np.nan
        evaluate(label, data)
        failures[label] = rows[-1][19]

    available = np.asarray([row[19] for row in rows], dtype=np.int8)
    values = np.stack([row[20] for row in rows])
    masks = np.stack([row[21] for row in rows]).astype(np.int8)
    if values.shape[1] != token_capacity * 131 or masks.shape[1] != token_capacity:
        raise SystemExit("geometry fixture contract width drift")
    if not (available.min() == 0 and available.max() == 1):
        raise SystemExit("geometry fixture never varies row availability")
    spread = values.max(axis=0) - values.min(axis=0)
    if not np.all(spread > 0):
        raise SystemExit(
            "geometry fixture leaves value columns constant: "
            f"{np.flatnonzero(spread <= 0).tolist()}"
        )
    if not np.all(masks.max(axis=0) > masks.min(axis=0)):
        raise SystemExit("geometry fixture leaves token masks constant")
    for label, result in failures.items():
        if result:
            raise SystemExit(f"geometry row-level gate stayed open for {label}")

    lines = [
        "# case\tsource_available\ttoken_mask[44]\ttoken_kind[44]"
        "\ttoken_provenance[44]\trelative_position[132]\tclearance[44]"
        "\tsemantic_flags[44]\tdynamic_blocked[44]\tedge_capacity"
        "\tedge_mask\tedge_destination\tedge_cost\tedge_kind\tedge_flags"
        "\tbox_capacity\tcollision_box_mask\tcollision_boxes_relative"
        "\tactor_valid\texpected_available\texpected_token_f32[5764]"
        "\texpected_token_mask[44]"
    ]

    def encoded(value) -> str:
        array = np.asarray(value).reshape(-1)
        if array.dtype == np.bool_ or np.issubdtype(array.dtype, np.integer):
            return ",".join(str(int(item)) for item in array)
        parts = []
        for item in array:
            number = float(item)
            if np.isnan(number):
                parts.append("NaN")
            elif np.isposinf(number):
                parts.append("Infinity")
            elif np.isneginf(number):
                parts.append("-Infinity")
            else:
                parts.append(repr(number))
        return ",".join(parts)

    for row in rows:
        lines.append("\t".join([str(row[0])] + [encoded(v) for v in row[1:]]))
    return lines


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "movement-only":
        movement_lines = _movement_fixture()
        OUT.mkdir(parents=True, exist_ok=True)
        # The helper runs every coverage assertion before returning. Only a
        # fully exercised fixture can replace the previous file.
        (OUT / "movement_features.txt").write_text(
            "\n".join(movement_lines) + "\n", encoding="utf-8"
        )
        print(f"movement rows: {len(movement_lines) - 1}")
        print(f"wrote {OUT / 'movement_features.txt'}")
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "inventory-only":
        inventory_lines = _inventory_fixture()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "inventory_features.txt").write_text(
            "\n".join(inventory_lines) + "\n", encoding="utf-8"
        )
        print(f"inventory rows: {len(inventory_lines) - 1}")
        print(f"wrote {OUT / 'inventory_features.txt'}")
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "light-only":
        light_lines = _light_fixture()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "light_features.txt").write_text(
            "\n".join(light_lines) + "\n", encoding="utf-8"
        )
        print(f"light rows: {len(light_lines) - 1}")
        print(f"wrote {OUT / 'light_features.txt'}")
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "geometry-only":
        geometry_lines = _geometry_fixture()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "geometry_features.txt").write_text(
            "\n".join(geometry_lines) + "\n", encoding="utf-8"
        )
        print(f"geometry rows: {len(geometry_lines) - 1}")
        print(f"wrote {OUT / 'geometry_features.txt'}")
        return 0

    from hytalegym.jax.combat import (
        arsenal_runtime_config,
        default_combat_params,
        hytale_0_5_7_loadouts,
        neutral_arsenal_policy_action_factors,
    )
    from hytalegym.jax.combat.arsenal.schema.contract import (
        OBSERVATION_CAPACITY,
    )
    from hytalegym.jax.combat.observation import COMBAT_FLOAT_FEATURES
    from hytalegym.jax.combat.types import AGENT_ENTITY
    try:
        from hytalegym.jax.combat.observation.v1.runtime.encoder import (
            _AGENT_ATTACK_EXECUTING,
            _self_features,
        )
    except ModuleNotFoundError:
        from hytalegym.jax.combat.observation.encoder import (
            _AGENT_ATTACK_EXECUTING,
            _self_features,
        )
    from hytalegym.jax.training import (
        make_arsenal_ppo_environment,
        open_flat_arsenal_world_capabilities,
    )
    from adk.validation import reset_keys, stream

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in {
        "actor-world-only", "resource-only", "defense-only", "status-only",
        "dodge-only", "ability-only",
    }:
        actor_params = default_combat_params(microticks=1, target_active=True)
        actor_config = arsenal_runtime_config(hytale_0_5_7_loadouts((PROFILE,)))
        actor_env = make_arsenal_ppo_environment(
            actor_params,
            actor_config,
            world_capability_provider=open_flat_arsenal_world_capabilities,
        )
        actor_state, _, _ = actor_env.reset(reset_keys(SEED, 1))
        OUT.mkdir(parents=True, exist_ok=True)
        if mode == "actor-world-only":
            lines = _actor_world_fixture(
                actor_state.runtime, actor_params, actor_config
            )
            target = OUT / "actor_world_features.txt"
        elif mode == "resource-only":
            lines = _resource_fixture(
                actor_state.runtime, actor_params, actor_config
            )
            target = OUT / "resource_features.txt"
        elif mode == "defense-only":
            lines = _defense_fixture(
                actor_state.runtime, actor_params, actor_config
            )
            target = OUT / "defense_features.txt"
        elif mode == "status-only":
            lines = _status_fixture(
                actor_state.runtime, actor_params, actor_config
            )
            target = OUT / "status_features.txt"
        elif mode == "dodge-only":
            lines = _dodge_fixture(
                actor_state.runtime, actor_params, actor_config
            )
            target = OUT / "dodge_features.txt"
        else:
            lines = _ability_fixture(
                actor_state.runtime, actor_params, actor_config
            )
            target = OUT / "ability_features.txt"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{mode} rows: {len(lines) - 1}")
        print(f"wrote {target}")
        return 0

    params = default_combat_params(microticks=1, target_active=True)
    config = arsenal_runtime_config(hytale_0_5_7_loadouts((PROFILE,) * BATCH))
    env = make_arsenal_ppo_environment(
        params,
        config,
        world_capability_provider=open_flat_arsenal_world_capabilities,
    )
    factors = neutral_arsenal_policy_action_factors(BATCH)
    keys = stream(SEED, STEPS, BATCH)
    state, observation, _ = env.reset(reset_keys(SEED, BATCH))

    rows: list[dict] = []
    for step in range(STEPS):
        combat = state.runtime.combat
        # The projector's own input, so the exported expectation is what JAX
        # actually produces rather than a re-derivation of it here.
        combat_observation = state.learner_observation.base.combat_f32
        expected, expected_i32 = _self_features(combat, params, combat_observation)

        for lane in range(BATCH):
            rows.append(
                {
                    "step": step,
                    "lane": lane,
                    # --- raw inputs, all readable from server components ---
                    "velocity": [
                        float(v) for v in np.asarray(combat.velocity)[lane, AGENT_ENTITY]
                    ],
                    "yaw_degrees": float(np.asarray(combat.yaw)[lane, AGENT_ENTITY]),
                    "pitch_degrees": float(np.asarray(combat.pitch)[lane]),
                    "health": float(np.asarray(combat.health)[lane, AGENT_ENTITY]),
                    "grounded": bool(np.asarray(combat.agent_grounded)[lane]),
                    "attack_executing": float(
                        np.asarray(combat_observation)[lane, _AGENT_ATTACK_EXECUTING]
                    ),
                    "attack_cooldown_seconds": float(
                        np.asarray(combat.agent_attack_cooldown_seconds)[lane]
                    ),
                    "knockback_control_lock": bool(
                        np.asarray(combat.knockback_control_lock)[lane]
                    ),
                    "ticks_since_damage": float(
                        np.asarray(combat.ticks_since_agent_damage)[lane]
                    ),
                    "applied_vertical_velocity": float(
                        np.asarray(combat.agent_applied_vertical_velocity)[lane]
                    ),
                    "fall_speed": float(np.asarray(combat.agent_fall_speed)[lane]),
                    "motion_delta_seconds": float(
                        np.asarray(combat.last_motion_delta_seconds)[lane]
                    ),
                    # --- what JAX derived ---
                    "expected": [float(v) for v in np.asarray(expected)[lane]],
                }
            )

        result = env.step_detailed(state, observation, factors, keys[step])
        state, observation = result[0], result[1]

    # A neutral rollout leaves the agent standing still, so 15 of 16 columns
    # come out constant and the comparison proves almost nothing. Sweep
    # synthetic values through the *real* JAX function -- taking a valid state
    # and replacing only the fields the derivation reads, so the rest of the
    # plumbing stays authentic -- and cover negatives plus magnitudes past the
    # +/-1 clip.
    rng = np.random.default_rng(SEED)
    combat = state.runtime.combat
    observation_row = np.asarray(state.learner_observation.base.combat_f32)

    def like(reference, values):
        array = np.asarray(reference).copy()
        return jnp.asarray(values, dtype=array.dtype)

    for sweep in range(24):
        velocity = np.asarray(combat.velocity).copy()
        velocity[:, AGENT_ENTITY, :] = rng.uniform(-12.0, 12.0, (BATCH, 3))
        yaw = np.asarray(combat.yaw).copy()
        yaw[:, AGENT_ENTITY] = rng.uniform(-540.0, 540.0, BATCH)
        health = np.asarray(combat.health).copy()
        health[:, AGENT_ENTITY] = rng.uniform(0.0, 105.0, BATCH)
        # `alive` is health > 0, so uniform sampling never exercises it --
        # it would read constant-1 and the column would be untested. Force a
        # dead lane on alternating sweeps.
        health[sweep % BATCH, AGENT_ENTITY] = 0.0 if sweep % 2 == 0 else 105.0
        observation = observation_row.copy()
        observation[:, _AGENT_ATTACK_EXECUTING] = rng.uniform(0.0, 1.0, BATCH)

        swept = combat._replace(
            velocity=like(combat.velocity, velocity),
            yaw=like(combat.yaw, yaw),
            health=like(combat.health, health),
            pitch=like(combat.pitch, rng.uniform(-120.0, 120.0, BATCH)),
            agent_grounded=like(
                combat.agent_grounded, rng.integers(0, 2, BATCH)),
            agent_attack_cooldown_seconds=like(
                combat.agent_attack_cooldown_seconds, rng.uniform(0.0, 3.0, BATCH)),
            knockback_control_lock=like(
                combat.knockback_control_lock, rng.integers(0, 2, BATCH)),
            ticks_since_agent_damage=like(
                combat.ticks_since_agent_damage, rng.integers(0, 900, BATCH)),
            agent_applied_vertical_velocity=like(
                combat.agent_applied_vertical_velocity,
                rng.uniform(-15.0, 15.0, BATCH)),
            agent_fall_speed=like(
                combat.agent_fall_speed, rng.uniform(-20.0, 20.0, BATCH)),
            last_motion_delta_seconds=like(
                combat.last_motion_delta_seconds, rng.uniform(0.0, 0.09, BATCH)),
        )
        expected, _ = _self_features(swept, params, jnp.asarray(observation))

        for lane in range(BATCH):
            rows.append(
                {
                    "step": 1000 + sweep,
                    "lane": lane,
                    "velocity": [float(v) for v in velocity[lane, AGENT_ENTITY]],
                    "yaw_degrees": float(yaw[lane, AGENT_ENTITY]),
                    "pitch_degrees": float(np.asarray(swept.pitch)[lane]),
                    "health": float(health[lane, AGENT_ENTITY]),
                    "grounded": bool(np.asarray(swept.agent_grounded)[lane]),
                    "attack_executing": float(
                        observation[lane, _AGENT_ATTACK_EXECUTING]),
                    "attack_cooldown_seconds": float(
                        np.asarray(swept.agent_attack_cooldown_seconds)[lane]),
                    "knockback_control_lock": bool(
                        np.asarray(swept.knockback_control_lock)[lane]),
                    "ticks_since_damage": float(
                        np.asarray(swept.ticks_since_agent_damage)[lane]),
                    "applied_vertical_velocity": float(
                        np.asarray(swept.agent_applied_vertical_velocity)[lane]),
                    "fall_speed": float(np.asarray(swept.agent_fall_speed)[lane]),
                    "motion_delta_seconds": float(
                        np.asarray(swept.last_motion_delta_seconds)[lane]),
                    "expected": [float(v) for v in np.asarray(expected)[lane]],
                }
            )

    # ---- target_f32 -------------------------------------------------------
    # Same shape of test, one level harder: this derivation reads two entities,
    # quantises the offset through the wire fixed-point scale (deliberately, so
    # JAX matches what Java publishes), and masks everything by visibility.
    # Four of its columns are copied straight from the combat row, which is a
    # separate derivation, so those are exported as inputs rather than
    # recomputed here.
    try:
        from hytalegym.jax.combat.observation.v1.runtime.encoder import (
            _TARGET_ATTACK_PROGRESS,
            _TARGET_FACING_ERROR,
            _TARGET_HEAD_FACING_ERROR,
            _TARGET_HEAD_PITCH,
            _TARGET_VISIBLE,
            _target_features,
            _target_ids,
            _target_observation_evidence,
        )
    except ModuleNotFoundError:
        from hytalegym.jax.combat.observation.encoder import (
            _TARGET_ATTACK_PROGRESS,
            _TARGET_FACING_ERROR,
            _TARGET_HEAD_FACING_ERROR,
            _TARGET_HEAD_PITCH,
            _TARGET_VISIBLE,
            _target_features,
            _target_ids,
            _target_observation_evidence,
        )
    from hytalegym.jax.combat.types import TARGET_ENTITY

    target_rows: list[dict] = []
    for sweep in range(32):
        positions = np.asarray(combat.position).copy()
        positions[:, AGENT_ENTITY, :] = rng.uniform(-30.0, 30.0, (BATCH, 3))
        positions[:, TARGET_ENTITY, :] = rng.uniform(-30.0, 30.0, (BATCH, 3))
        velocities = np.asarray(combat.velocity).copy()
        velocities[:, AGENT_ENTITY, :] = rng.uniform(-9.0, 9.0, (BATCH, 3))
        velocities[:, TARGET_ENTITY, :] = rng.uniform(-9.0, 9.0, (BATCH, 3))
        yaws = np.asarray(combat.yaw).copy()
        yaws[:, AGENT_ENTITY] = rng.uniform(-540.0, 540.0, BATCH)
        healths = np.asarray(combat.health).copy()
        healths[:, TARGET_ENTITY] = rng.uniform(0.0, 105.0, BATCH)

        observation = observation_row.copy()
        # Half the lanes blind, so the masked branch is exercised rather than
        # assumed -- a derivation that ignores the mask passes otherwise.
        observation[:, _TARGET_VISIBLE] = (
            1.0 if sweep % 2 == 0 else 0.0
        )
        observation[: BATCH // 2, _TARGET_VISIBLE] = 1.0
        observation[BATCH // 2 :, _TARGET_VISIBLE] = 0.0
        for column in (
            _TARGET_FACING_ERROR, _TARGET_HEAD_FACING_ERROR,
            _TARGET_HEAD_PITCH, _TARGET_ATTACK_PROGRESS,
        ):
            observation[:, column] = rng.uniform(-1.0, 1.0, BATCH)

        swept = combat._replace(
            position=like(combat.position, positions),
            velocity=like(combat.velocity, velocities),
            yaw=like(combat.yaw, yaws),
            health=like(combat.health, healths),
        )
        observation_j = jnp.asarray(observation)
        evidence = _target_observation_evidence(
            swept, params, observation_j, _target_ids(swept, None)
        )
        expected, _, mask = _target_features(
            evidence,
            swept.yaw[:, AGENT_ENTITY],
            params,
            observation_j,
        )

        for lane in range(BATCH):
            target_rows.append(
                {
                    "sweep": sweep,
                    "lane": lane,
                    "agent_position": [
                        float(v) for v in positions[lane, AGENT_ENTITY]],
                    "agent_velocity": [
                        float(v) for v in velocities[lane, AGENT_ENTITY]],
                    "agent_yaw_degrees": float(yaws[lane, AGENT_ENTITY]),
                    "target_position": [
                        float(v) for v in positions[lane, TARGET_ENTITY]],
                    "target_velocity": [
                        float(v) for v in velocities[lane, TARGET_ENTITY]],
                    "target_health": float(healths[lane, TARGET_ENTITY]),
                    "visible": float(observation[lane, _TARGET_VISIBLE]),
                    "facing_error": float(observation[lane, _TARGET_FACING_ERROR]),
                    "head_facing_error": float(
                        observation[lane, _TARGET_HEAD_FACING_ERROR]),
                    "head_pitch": float(observation[lane, _TARGET_HEAD_PITCH]),
                    "attack_progress": float(
                        observation[lane, _TARGET_ATTACK_PROGRESS]),
                    "expected": [float(v) for v in np.asarray(expected)[lane]],
                    "expected_mask": bool(np.asarray(mask)[lane]),
                }
            )

    target_lines = [
        "# agentPos(3) agentVel(3) agentYaw targetPos(3) targetVel(3) "
        "targetHealth visible facingError headFacingError headPitch "
        "attackProgress | expected[16] mask"
    ]
    for row in target_rows:
        target_lines.append("\t".join(str(v) for v in (
            row["sweep"], row["lane"],
            *(repr(v) for v in row["agent_position"]),
            *(repr(v) for v in row["agent_velocity"]),
            repr(row["agent_yaw_degrees"]),
            *(repr(v) for v in row["target_position"]),
            *(repr(v) for v in row["target_velocity"]),
            repr(row["target_health"]), repr(row["visible"]),
            repr(row["facing_error"]), repr(row["head_facing_error"]),
            repr(row["head_pitch"]), repr(row["attack_progress"]),
            ",".join(repr(v) for v in row["expected"]),
            int(row["expected_mask"]),
        )))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "target_features.txt").write_text(
        "\n".join(target_lines) + "\n", encoding="utf-8")

    target_flat = np.asarray([r["expected"] for r in target_rows], dtype=np.float32)
    print(f"target rows   : {len(target_rows)}")
    print(f"target varying: "
          f"{int((target_flat.max(axis=0) - target_flat.min(axis=0) > 1e-9).sum())}/16")
    print(f"target visible: {sum(1 for r in target_rows if r['expected_mask'])}"
          f"/{len(target_rows)}")

    # ---- combat_f32 -------------------------------------------------------
    # The 24-value actor observation, exported at *both* levels so a failure
    # localises: `target_evidence` (masking, wire quantisation, bearing) and
    # `_encode_actor_observation` (projection and normalisation) are separate
    # ports and separate columns in the fixture.
    try:
        from hytalegym.jax.combat.runtime.reset import (
            ActorEvidence,
            _encode_actor_observation,
            target_evidence,
        )
        from hytalegym.jax.combat.opponents.legacy.controller import _target_phase
    except ModuleNotFoundError:
        from hytalegym.jax.combat._reset import (
            ActorEvidence,
            _encode_actor_observation,
            target_evidence,
        )
        from hytalegym.jax.combat._target import _target_phase

    combat_rows: list[dict] = []
    for sweep in range(32):
        positions = np.asarray(combat.position).copy()
        positions[:, AGENT_ENTITY, :] = rng.uniform(-30.0, 30.0, (BATCH, 3))
        # Spread the target across and past the sensor range so the detection
        # branch flips on distance rather than being forced every lane.
        positions[:, TARGET_ENTITY, :] = positions[:, AGENT_ENTITY, :] + rng.uniform(
            -20.0, 20.0, (BATCH, 3)
        )
        velocities = np.asarray(combat.velocity).copy()
        velocities[:, AGENT_ENTITY, :] = rng.uniform(-9.0, 9.0, (BATCH, 3))
        velocities[:, TARGET_ENTITY, :] = rng.uniform(-9.0, 9.0, (BATCH, 3))
        yaws = np.asarray(combat.yaw).copy()
        yaws[:, AGENT_ENTITY] = rng.uniform(-540.0, 540.0, BATCH)
        # The target's own yaw drives facing_error, and the +/-180 wrap is the
        # part most likely to be ported wrong, so sample past a full turn.
        yaws[:, TARGET_ENTITY] = rng.uniform(-540.0, 540.0, BATCH)
        healths = np.asarray(combat.health).copy()
        healths[:, AGENT_ENTITY] = rng.uniform(0.0, 105.0, BATCH)
        healths[:, TARGET_ENTITY] = rng.uniform(0.0, 61.0, BATCH)

        # Phase is driven entirely by the target's attack timers. Left at rest
        # every row reports IDLE, four of the five one-hot columns read constant
        # zero, and a wrong one-hot passes. Drive the fields `_target_phase`
        # actually reads and cover all five outcomes; the assertion below makes
        # a regression in coverage loud instead of silent.
        windup_ticks = np.asarray(params.target_windup_ticks)
        sweep_ticks = np.asarray(params.target_sweep_ticks)
        recovery_ticks = np.asarray(params.target_recovery_ticks)
        attack_index = np.full(BATCH, -1, dtype=np.int32)
        elapsed = np.zeros(BATCH, dtype=np.int32)
        queued = np.zeros(BATCH, dtype=bool)
        cooldown_seconds = np.zeros(BATCH, dtype=np.float32)
        last_index = np.full(BATCH, -1, dtype=np.int32)
        for lane in range(BATCH):
            index = (sweep + lane) % len(windup_ticks)
            windup = int(windup_ticks[index])
            sweep_length = int(sweep_ticks[index])
            recovery = int(recovery_ticks[index])
            case = (sweep * BATCH + lane) % 6
            if case == 0:
                pass  # no root, not queued, not cooling -> IDLE, index -1
            elif case == 1:
                attack_index[lane] = index  # root but elapsed 0 -> IDLE
            elif case == 2:
                attack_index[lane] = index
                elapsed[lane] = max(1, windup)  # -> WINDUP
            elif case == 3:
                attack_index[lane] = index
                elapsed[lane] = windup + max(1, sweep_length)  # -> SWEEP
            elif case == 4:
                attack_index[lane] = index
                elapsed[lane] = windup + sweep_length + max(1, recovery)
            else:
                # No root, but cooling -- the branch that reports
                # `last_target_attack_index` rather than the live one.
                cooldown_seconds[lane] = 0.5
                last_index[lane] = index

        swept = combat._replace(
            position=like(combat.position, positions),
            velocity=like(combat.velocity, velocities),
            yaw=like(combat.yaw, yaws),
            health=like(combat.health, healths),
            target_head_yaw=like(
                combat.target_head_yaw, rng.uniform(-540.0, 540.0, BATCH)),
            target_head_pitch=like(
                combat.target_head_pitch, rng.uniform(-90.0, 90.0, BATCH)),
            # `agent_attack_executing` is one of the 18 columns that actually
            # reaches the policy, and it reads constant-0 unless the agent's own
            # attack timers are driven. Alternate both of its sources, since it
            # is the OR of a hit delay and a cooldown.
            agent_hit_delay=like(
                combat.agent_hit_delay,
                np.where(np.arange(BATCH) % 4 == 1, 3, 0)),
            agent_attack_cooldown_seconds=like(
                combat.agent_attack_cooldown_seconds,
                np.where(np.arange(BATCH) % 4 == 2, 0.75, 0.0)),
            target_attack_index=like(combat.target_attack_index, attack_index),
            target_attack_elapsed_ticks=like(
                combat.target_attack_elapsed_ticks, elapsed),
            target_attack_queued=like(combat.target_attack_queued, queued),
            target_attack_cooldown_seconds=like(
                combat.target_attack_cooldown_seconds, cooldown_seconds),
            last_target_attack_index=like(
                combat.last_target_attack_index, last_index),
        )
        # Blind half the lanes outright; distance handles the rest.
        line_of_sight = np.ones(BATCH, dtype=bool)
        line_of_sight[BATCH // 2 :] = sweep % 2 == 0
        # Every third sweep aims at a non-primary entity. That gate suppresses
        # the attack phase, progress and reported index regardless of what the
        # shared attack state machine holds, and it is invisible to a fixture
        # that only ever aims at TARGET_ENTITY.
        primary = sweep % 3 != 2
        target_ids = np.full(
            BATCH, TARGET_ENTITY if primary else AGENT_ENTITY, dtype=np.int32)
        evidence = target_evidence(
            swept,
            params,
            perception_line_of_sight=jnp.asarray(line_of_sight),
            target_entity_id=jnp.asarray(target_ids),
        )
        phase, progress, reported_index, _ = _target_phase(swept, params)
        agent_health_fraction = (
            swept.health[:, AGENT_ENTITY] / params.agent_max_health
        )
        attack_executing = (swept.agent_hit_delay > 0) | (
            swept.agent_attack_cooldown_seconds > 0.0
        )
        expected = _encode_actor_observation(
            ActorEvidence(
                velocity=swept.velocity[:, AGENT_ENTITY],
                yaw_radians=jnp.deg2rad(swept.yaw[:, AGENT_ENTITY]),
                health_fraction=agent_health_fraction,
                attack_executing=attack_executing,
                target=evidence,
            ),
            params,
        )

        # The learner does not consume the legacy target-attack index emitted
        # by _encode_actor_observation.  Arsenal replaces combat_f32[21] with
        # the authored ability slot divided by its fixed learner capacity.
        # Exercise that final policy-visible contract here; leaving the legacy
        # / (target_damage_count - 1) value produced a green but vacuous gate
        # whenever the opponent happened to use slot zero.
        attack_column = COMBAT_FLOAT_FEATURES.index("target_attack_index")
        visible_reported_slot = jnp.where(
            primary,
            reported_index,
            jnp.full_like(reported_index, -1),
        )
        normalized_slot = jnp.where(
            visible_reported_slot >= 0,
            visible_reported_slot.astype(jnp.float32)
            / jnp.float32(max(OBSERVATION_CAPACITY - 1, 1)),
            jnp.float32(-1.0),
        )
        expected = expected.at[:, attack_column].set(
            jnp.where(evidence.perceptible, normalized_slot, jnp.float32(0.0))
        )

        # `target_evidence` gathers by entity id, so on a non-primary sweep the
        # entity it actually reads is the agent. The fixture must record the
        # gathered row, not TARGET_ENTITY's, or Java is handed different inputs
        # than JAX saw.
        gathered = TARGET_ENTITY if primary else AGENT_ENTITY
        for lane in range(BATCH):
            combat_rows.append(
                {
                    "sweep": sweep,
                    "lane": lane,
                    "primary_target": primary,
                    # --- inputs to target_evidence ---
                    "agent_position": [
                        float(v) for v in positions[lane, AGENT_ENTITY]],
                    "target_position": [
                        float(v) for v in positions[lane, gathered]],
                    "target_velocity": [
                        float(v) for v in velocities[lane, gathered]],
                    "target_health": float(healths[lane, gathered]),
                    "target_yaw": float(yaws[lane, gathered]),
                    "target_head_yaw": float(
                        np.asarray(swept.target_head_yaw)[lane]),
                    "target_head_pitch": float(
                        np.asarray(swept.target_head_pitch)[lane]),
                    "perceptible": bool(np.asarray(evidence.perceptible)[lane]),
                    "attack_phase": int(np.asarray(phase)[lane]),
                    "attack_progress": float(np.asarray(progress)[lane]),
                    "reported_attack_index": int(
                        np.asarray(reported_index)[lane]),
                    # --- inputs to the encoder ---
                    "agent_velocity": [
                        float(v) for v in velocities[lane, AGENT_ENTITY]],
                    "agent_yaw": float(yaws[lane, AGENT_ENTITY]),
                    "agent_health_fraction": float(
                        np.asarray(agent_health_fraction)[lane]),
                    "attack_executing": bool(
                        np.asarray(attack_executing)[lane]),
                    # --- what JAX derived ---
                    "expected": [float(v) for v in np.asarray(expected)[lane]],
                }
            )

    combat_lines = [
        "# primaryTarget agentPos(3) targetPos(3) targetVel(3) targetHealth "
        "targetYaw targetHeadYaw targetHeadPitch perceptible phase progress "
        "reportedIndex agentVel(3) agentYaw agentHealthFraction "
        "attackExecuting | expected[24]"
    ]
    for row in combat_rows:
        combat_lines.append("\t".join(str(v) for v in (
            row["sweep"], row["lane"], int(row["primary_target"]),
            *(repr(v) for v in row["agent_position"]),
            *(repr(v) for v in row["target_position"]),
            *(repr(v) for v in row["target_velocity"]),
            repr(row["target_health"]), repr(row["target_yaw"]),
            repr(row["target_head_yaw"]), repr(row["target_head_pitch"]),
            int(row["perceptible"]), row["attack_phase"],
            repr(row["attack_progress"]), row["reported_attack_index"],
            *(repr(v) for v in row["agent_velocity"]),
            repr(row["agent_yaw"]), repr(row["agent_health_fraction"]),
            int(row["attack_executing"]),
            ",".join(repr(v) for v in row["expected"]),
        )))
    combat_flat = np.asarray(
        [r["expected"] for r in combat_rows], dtype=np.float32)
    combat_visible = sum(1 for r in combat_rows if r["perceptible"])
    print(f"combat rows   : {len(combat_rows)}")
    print(f"combat varying: "
          f"{int((combat_flat.max(axis=0) - combat_flat.min(axis=0) > 1e-9).sum())}/24")
    print(f"combat visible: {combat_visible}/{len(combat_rows)}")
    reached = sorted({r["attack_phase"] for r in combat_rows})
    print(f"combat phases : {reached}")
    if reached != [0, 1, 2, 3, 4]:
        raise SystemExit(
            f"combat fixture reaches phases {reached}, needs all of [0..4]; "
            "the one-hot block would be certified vacuously"
        )
    # The three states of the attack index are distinct and easy to collapse:
    # -1 "visible, not attacking", 0 "cannot see it", and a real index.
    indices = {r["reported_attack_index"] for r in combat_rows}
    if -1 not in indices or not any(index > 0 for index in indices):
        raise SystemExit(f"combat fixture attack indices {sorted(indices)} "
                         "do not cover absent and non-zero ability slots")
    # Unlike the phase block, this column is not excluded from the arsenal
    # policy -- an untested constant here is a real hole, not a cosmetic one.
    if len({r["attack_executing"] for r in combat_rows}) != 2:
        raise SystemExit("combat fixture never varies agent_attack_executing")
    if len({r["primary_target"] for r in combat_rows}) != 2:
        raise SystemExit(
            "combat fixture never aims at a non-primary entity -- the gate that "
            "suppresses phase, progress and attack index would be untested"
        )
    constant = [
        Projection_name
        for Projection_name, spread in zip(
            [
                "agent_forward_velocity", "agent_right_velocity",
                "agent_vertical_velocity", "agent_health_fraction",
                "agent_yaw_sin", "agent_yaw_cos",
                "visible_target_forward", "visible_target_right",
                "visible_target_planar_distance",
                "visible_target_health_fraction",
                "target_visible", "agent_attack_executing",
                "target_phase_idle", "target_phase_windup",
                "target_phase_sweep", "target_phase_recovery",
                "target_phase_cooldown", "target_attack_progress",
                "target_facing_error", "target_forward_velocity",
                "target_right_velocity", "target_attack_index",
                "target_head_facing_error", "target_head_pitch",
            ],
            combat_flat.max(axis=0) - combat_flat.min(axis=0),
        )
        if spread <= 1e-9
    ]
    if constant:
        raise SystemExit(
            f"combat fixture leaves {len(constant)} column(s) constant: "
            f"{constant} -- they would be certified vacuously"
        )

    # Written only after every coverage check passes. Writing first would leave
    # a fixture on disk that a later test run consumes happily while the export
    # that produced it had already been rejected.
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "combat_features.txt").write_text(
        "\n".join(combat_lines) + "\n", encoding="utf-8")

    movement_lines = _movement_fixture()
    actor_world_lines = _actor_world_fixture(state.runtime, params, config)
    resource_lines = _resource_fixture(state.runtime, params, config)
    defense_lines = _defense_fixture(state.runtime, params, config)
    status_lines = _status_fixture(state.runtime, params, config)
    dodge_lines = _dodge_fixture(state.runtime, params, config)
    ability_lines = _ability_fixture(state.runtime, params, config)
    inventory_lines = _inventory_fixture()
    (OUT / "movement_features.txt").write_text(
        "\n".join(movement_lines) + "\n", encoding="utf-8")
    (OUT / "actor_world_features.txt").write_text(
        "\n".join(actor_world_lines) + "\n", encoding="utf-8")
    (OUT / "resource_features.txt").write_text(
        "\n".join(resource_lines) + "\n", encoding="utf-8")
    (OUT / "defense_features.txt").write_text(
        "\n".join(defense_lines) + "\n", encoding="utf-8")
    (OUT / "status_features.txt").write_text(
        "\n".join(status_lines) + "\n", encoding="utf-8")
    (OUT / "dodge_features.txt").write_text(
        "\n".join(dodge_lines) + "\n", encoding="utf-8")
    (OUT / "ability_features.txt").write_text(
        "\n".join(ability_lines) + "\n", encoding="utf-8")
    (OUT / "inventory_features.txt").write_text(
        "\n".join(inventory_lines) + "\n", encoding="utf-8")

    scalars = {
        "agent_max_speed": float(params.agent_max_speed),
        "vertical_speed_scale": float(params.vertical_speed_scale),
        "agent_max_health": float(params.agent_max_health),
        "agent_attack_pause_max_seconds": float(params.agent_attack_pause_max_seconds),
        "regen_delay_ticks": float(params.regen_delay_ticks),
        "agent_walk_max_fall_speed": float(params.agent_walk_max_fall_speed),
        "loaded_dt": float(params.loaded_dt),
        "wire_fixed_point_scale": float(params.wire_fixed_point_scale),
        "target_chase_speed": float(params.target_chase_speed),
        "target_max_health": float(params.target_max_health),
        # The combat row divides target offsets by the *sensor range*, not by
        # NEARBY_ENTITY_RADIUS_BLOCKS as the target row does. Different numbers.
        "sensor_range": float(params.sensor_range),
        "facing_error_degrees_scale": float(params.facing_error_degrees_scale),
        "head_pitch_degrees_scale": float(params.head_pitch_degrees_scale),
        "target_attack_index_divisor": float(
            max(OBSERVATION_CAPACITY - 1, 1)
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "self_features.json").write_text(
        json.dumps({"params": scalars, "rows": rows}, indent=1), encoding="utf-8"
    )

    # Flat mirror so the Java side needs no JSON parser, matching actions.txt.
    lines = [
        "# " + "\t".join(
            ("step", "lane", "vx", "vy", "vz", "yaw", "pitch", "health", "grounded",
             "attackExecuting", "attackCooldown", "knockbackLock", "ticksSinceDamage",
             "appliedVertical", "fallSpeed", "motionDelta", "expected[16]")
        )
    ]
    for row in rows:
        lines.append("\t".join(str(v) for v in (
            row["step"], row["lane"],
            repr(row["velocity"][0]), repr(row["velocity"][1]), repr(row["velocity"][2]),
            repr(row["yaw_degrees"]), repr(row["pitch_degrees"]), repr(row["health"]),
            int(row["grounded"]), repr(row["attack_executing"]),
            repr(row["attack_cooldown_seconds"]), int(row["knockback_control_lock"]),
            repr(row["ticks_since_damage"]), repr(row["applied_vertical_velocity"]),
            repr(row["fall_speed"]), repr(row["motion_delta_seconds"]),
            ",".join(repr(v) for v in row["expected"]),
        )))
    (OUT / "self_features.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUT / "params.txt").write_text(
        "\n".join(f"{k}\t{v!r}" for k, v in scalars.items()) + "\n", encoding="utf-8"
    )

    flat = np.asarray([r["expected"] for r in rows], dtype=np.float32)
    print(f"rows          : {len(rows)} ({STEPS} steps x {BATCH} lanes)")
    print(f"columns       : {flat.shape[1]}")
    print(f"non-zero cols : {int((np.abs(flat) > 0).any(axis=0).sum())}/16")
    print(f"value range   : [{flat.min():+.4f}, {flat.max():+.4f}]")
    print("params        :")
    for key, value in scalars.items():
        print(f"  {key:32} {value}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
