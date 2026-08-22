"""Generation-safe binding for newly spawned Arsenal projectile/area slots."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import (
    AREA_CAPACITY,
    CROSSBOW_FAMILY_IDS,
    PROJECTILE_ARROW,
    PROJECTILE_BIG_ARROW,
    PROJECTILE_CAPACITY,
    SHORTBOW_FAMILY_IDS,
    ArsenalState,
)
from hytalegym.jax.combat.controllers import (
    RangedControllerInfo,
    RangedControllerRules,
)
from hytalegym.jax.combat.entities import (
    ENTITY_CAPACITY,
    EntityRoster,
    validate_roster_layout,
)
from hytalegym.jax.combat.entities.impacts.schema.types import (
    EntityImpactBindingInfo,
    EntityImpactBindings,
)
from hytalegym.jax.combat.entities.interactions import (
    CROSSBOW_IMPACT_BIG_ARROW,
    CROSSBOW_IMPACT_NONE,
    CROSSBOW_IMPACT_STANDARD,
)


def bind_arsenal_launches(
    bindings: EntityImpactBindings,
    before: ArsenalState,
    after: ArsenalState,
    roster: EntityRoster,
    controller_rules: RangedControllerRules,
    controller_info: RangedControllerInfo,
    source_friendly_fire: jax.Array,
) -> tuple[EntityImpactBindings, EntityImpactBindingInfo]:
    """Bind new slots and verify ranged launch counts atomically per row."""

    validate_roster_layout(roster)
    batch = roster.active.shape[0]
    _validate_layout(
        bindings,
        before,
        after,
        controller_rules,
        controller_info,
        source_friendly_fire,
        batch,
    )
    projectile = after.projectiles
    area = after.areas
    new_projectile = projectile.active & ~before.projectiles.active
    new_area = area.active & ~before.areas.active
    projectile_owner = jnp.clip(projectile.owner_entity_id, 0, ENTITY_CAPACITY - 1)
    area_owner = jnp.clip(area.owner_entity_id, 0, ENTITY_CAPACITY - 1)
    projectile_source_valid = (
        new_projectile
        & (projectile.owner_entity_id >= 0)
        & (projectile.owner_entity_id < ENTITY_CAPACITY)
        & _gather(roster.active, projectile_owner)
    )
    area_source_valid = (
        new_area
        & (area.owner_entity_id >= 0)
        & (area.owner_entity_id < ENTITY_CAPACITY)
        & _gather(roster.active, area_owner)
    )
    family = _gather(controller_rules.weapon_family, projectile_owner)
    crossbow = new_projectile & _family_matches(family, CROSSBOW_FAMILY_IDS)
    shortbow = new_projectile & _family_matches(family, SHORTBOW_FAMILY_IDS)
    verified_source = controller_info.ability_requested
    verified = new_projectile & _gather(verified_source, projectile_owner)
    owner_one_hot = jax.nn.one_hot(
        projectile_owner,
        ENTITY_CAPACITY,
        dtype=jnp.int32,
    )
    observed = jnp.sum(
        owner_one_hot * verified[..., None].astype(jnp.int32),
        axis=1,
    )
    expected = jnp.where(
        controller_info.valid[:, None],
        controller_info.projectile_count,
        jnp.int32(0),
    )
    kind_invalid = (
        (
            crossbow
            & ~(
                (projectile.kind == PROJECTILE_ARROW)
                | (projectile.kind == PROJECTILE_BIG_ARROW)
            )
        )
        | (shortbow & (projectile.kind != PROJECTILE_ARROW))
        | (new_projectile & (projectile.kind == PROJECTILE_BIG_ARROW) & ~crossbow)
    )
    invalid = (
        jnp.any(new_projectile & ~projectile_source_valid, axis=1)
        | jnp.any(new_area & ~area_source_valid, axis=1)
        | jnp.any(observed != expected, axis=1)
        | jnp.any(kind_invalid, axis=1)
        | (
            jnp.any(verified_source, axis=1)
            & (~controller_info.valid | (controller_info.failure_bits != jnp.uint32(0)))
        )
    )
    overflow = bindings.overflow | invalid
    valid = ~overflow
    projectile_generation = _gather(roster.generation, projectile_owner)
    area_generation = _gather(roster.generation, area_owner)
    multiplier = jnp.where(
        verified,
        _gather(
            controller_info.projectile_damage_multiplier,
            projectile_owner,
        ),
        jnp.float32(1.0),
    )
    interaction = jnp.where(
        crossbow & (projectile.kind == PROJECTILE_BIG_ARROW),
        CROSSBOW_IMPACT_BIG_ARROW,
        jnp.where(
            crossbow,
            CROSSBOW_IMPACT_STANDARD,
            CROSSBOW_IMPACT_NONE,
        ),
    )
    projectile_friendly_fire = _gather(source_friendly_fire, projectile_owner)
    area_friendly_fire = _gather(source_friendly_fire, area_owner)
    candidate = bindings._replace(
        projectile_source_generation=_replace_slots(
            bindings.projectile_source_generation,
            projectile_generation,
            new_projectile,
            after.projectiles.active,
            jnp.uint32(0),
        ),
        projectile_interaction_kind=_replace_slots(
            bindings.projectile_interaction_kind,
            interaction,
            new_projectile,
            after.projectiles.active,
            jnp.int32(CROSSBOW_IMPACT_NONE),
        ),
        projectile_damage_multiplier=_replace_slots(
            bindings.projectile_damage_multiplier,
            multiplier,
            new_projectile,
            after.projectiles.active,
            jnp.float32(1.0),
        ),
        projectile_knockback_yaw_degrees=_replace_slots(
            bindings.projectile_knockback_yaw_degrees,
            jnp.where(crossbow, jnp.nan, jnp.float32(0.0)),
            new_projectile,
            after.projectiles.active,
            jnp.float32(0.0),
        ),
        projectile_friendly_fire=_replace_slots(
            bindings.projectile_friendly_fire,
            projectile_friendly_fire,
            new_projectile,
            after.projectiles.active,
            True,
        ),
        area_source_generation=_replace_slots(
            bindings.area_source_generation,
            area_generation,
            new_area,
            after.areas.active,
            jnp.uint32(0),
        ),
        area_friendly_fire=_replace_slots(
            bindings.area_friendly_fire,
            area_friendly_fire,
            new_area,
            after.areas.active,
            True,
        ),
    )
    result = _select_tree(valid, candidate, bindings)._replace(overflow=overflow)
    return result, EntityImpactBindingInfo(
        projectile_bound=new_projectile & valid[:, None],
        area_bound=new_area & valid[:, None],
        crossbow_yaw_required=crossbow & valid[:, None],
        expected_ranged_projectiles=expected,
        observed_ranged_projectiles=observed,
        overflow=overflow,
        valid=valid,
    )


def set_projectile_impact_yaw(
    bindings: EntityImpactBindings,
    yaw_degrees: jax.Array,
    valid: jax.Array,
) -> EntityImpactBindings:
    """Inject the authoritative impact-context yaw for selected slots."""

    shape = bindings.projectile_knockback_yaw_degrees.shape
    _field(yaw_degrees, shape, jnp.float32, "yaw_degrees")
    _field(valid, shape, jnp.bool_, "valid")
    return bindings._replace(
        projectile_knockback_yaw_degrees=jnp.where(
            valid,
            yaw_degrees,
            bindings.projectile_knockback_yaw_degrees,
        )
    )


def _replace_slots(current, value, new, active, default):
    cleared = jnp.where(active, current, default)
    return jnp.where(new, value, cleared)


def _family_matches(family, candidates):
    result = jnp.zeros_like(family, dtype=jnp.bool_)
    for candidate in candidates:
        result |= family == jnp.int32(candidate)
    return result


def _gather(array, slot):
    batch = jnp.arange(array.shape[0]).reshape(
        (array.shape[0],) + (1,) * (slot.ndim - 1)
    )
    return array[batch, slot]


def _validate_layout(
    bindings,
    before,
    after,
    rules,
    info,
    friendly_fire,
    batch,
):
    projectile = (batch, PROJECTILE_CAPACITY)
    area = (batch, AREA_CAPACITY)
    entity = (batch, ENTITY_CAPACITY)
    for label, arsenal in (("before", before), ("after", after)):
        _field(
            arsenal.projectiles.active,
            projectile,
            jnp.bool_,
            f"{label}.projectiles.active",
        )
        _field(
            arsenal.projectiles.owner_entity_id,
            projectile,
            jnp.int32,
            f"{label}.projectiles.owner_entity_id",
        )
        _field(
            arsenal.projectiles.kind,
            projectile,
            jnp.int32,
            f"{label}.projectiles.kind",
        )
        _field(
            arsenal.areas.active,
            area,
            jnp.bool_,
            f"{label}.areas.active",
        )
        _field(
            arsenal.areas.owner_entity_id,
            area,
            jnp.int32,
            f"{label}.areas.owner_entity_id",
        )
    for name, shape, dtype in (
        ("projectile_source_generation", projectile, jnp.uint32),
        ("projectile_interaction_kind", projectile, jnp.int32),
        ("projectile_damage_multiplier", projectile, jnp.float32),
        (
            "projectile_knockback_yaw_degrees",
            projectile,
            jnp.float32,
        ),
        ("projectile_friendly_fire", projectile, jnp.bool_),
        ("area_source_generation", area, jnp.uint32),
        ("area_friendly_fire", area, jnp.bool_),
        ("overflow", (batch,), jnp.bool_),
    ):
        _field(getattr(bindings, name), shape, dtype, f"bindings.{name}")
    _field(
        rules.weapon_family,
        entity,
        jnp.int32,
        "controller_rules.weapon_family",
    )
    _field(
        info.ability_requested,
        entity,
        jnp.bool_,
        "controller_info.ability_requested",
    )
    _field(
        info.projectile_count,
        entity,
        jnp.int32,
        "controller_info.projectile_count",
    )
    _field(
        info.projectile_damage_multiplier,
        entity,
        jnp.float32,
        "controller_info.projectile_damage_multiplier",
    )
    _field(info.failure_bits, (batch,), jnp.uint32, "info.failure_bits")
    _field(info.valid, (batch,), jnp.bool_, "info.valid")
    _field(
        friendly_fire,
        entity,
        jnp.bool_,
        "source_friendly_fire",
    )


def _field(value, shape, dtype, name):
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")


def _select_tree(mask, selected, fallback):
    return jax.tree_util.tree_map(
        lambda yes, no: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (yes.ndim - 1)),
            yes,
            no,
        ),
        selected,
        fallback,
    )
