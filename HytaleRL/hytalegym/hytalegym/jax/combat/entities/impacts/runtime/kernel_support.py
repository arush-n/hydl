"""Small array/PRNG utilities shared by the impact kernel helpers."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import (
    AREA_GENERIC,
    AREA_NONE,
    FORCE_ADD,
    FORCE_DIRECTION_LOCAL,
    FORCE_DIRECTION_POINT,
    FORCE_SET,
    PROJECTILE_ARROW,
    PROJECTILE_BIG_ARROW,
    PROJECTILE_NONE,
    PROJECTILE_SPEAR,
    ArsenalState,
)
from hytalegym.jax.combat.arsenal.projectiles.runtime import (
    _standard_velocity,
)
from hytalegym.jax.combat.entities import (
    ENTITY_CAPACITY,
    SELECTOR_CAPACITY,
    TEAM_NONE,
    EntityStatusPayloads,
    EntityTargetSelection,
)
from hytalegym.jax.combat.entities.effects import (
    EntityEffectState,
    apply_selected_entity_effects,
    invalid_effect_rows,
    stale_effect_source_rows,
)
from hytalegym.jax.combat.entities.impacts.schema.contract import (
    IMPACT_CAPABILITIES,
    IMPACT_FAILURE_AMBIGUOUS_TRIGGER,
    IMPACT_FAILURE_CROSSBOW,
    IMPACT_FAILURE_DAMAGE_OVERFLOW,
    IMPACT_FAILURE_INVALID_BINDING,
    IMPACT_FAILURE_INVALID_DT,
    IMPACT_FAILURE_INVALID_STATE,
    IMPACT_FAILURE_MECHANICS,
    IMPACT_FAILURE_MIXED_INTERACTION_ORDER,
    IMPACT_FAILURE_QUERY,
    IMPACT_FAILURE_STALE_SOURCE,
    IMPACT_FAILURE_STALE_TARGET,
    IMPACT_FAILURE_UPSTREAM,
)
from hytalegym.jax.combat.entities.impacts.runtime.damage import (
    DenseImpactDamage,
    apply_dense_impact_damage,
)
from hytalegym.jax.combat.entities.impacts.schema.types import (
    EntityImpactBindings,
    EntityImpactInfo,
    EntityImpactQueries,
    EntityImpactState,
)
from hytalegym.jax.combat.entities.impacts.schema.validation import (
    validate_impact_layout,
)
from hytalegym.jax.combat.entities.interactions import (
    CROSSBOW_IMPACT_BIG_ARROW,
    CROSSBOW_IMPACT_KIND_COUNT,
    CROSSBOW_IMPACT_NONE,
    CROSSBOW_IMPACT_STANDARD,
    CrossbowImpactCommands,
    apply_crossbow_impacts,
    entity_interaction_state,
)
from hytalegym.jax.combat.mechanics import (
    CONTROL_IMMUNITY_MAXIMUM,
    DAMAGE_COUNT,
    RESOURCE_COUNT,
    STATUS_FLAG_DEBUFF,
    STATUS_FLAG_DISABLE_ABILITIES,
    STATUS_FLAG_DISABLE_MOVEMENT,
    STATUS_FLAG_DISABLE_SPRINT,
    STATUS_FLAG_CONTROL_IMMUNITY_GATED,
    STATUS_FLAG_IGNORE_KNOCKBACK,
    STATUS_FLAG_INVULNERABLE,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
    CombatMechanicsRules,
)


_STATUS_FLAG_MASK = (
    STATUS_FLAG_INVULNERABLE
    | STATUS_FLAG_DISABLE_MOVEMENT
    | STATUS_FLAG_DISABLE_ABILITIES
    | STATUS_FLAG_IGNORE_KNOCKBACK
    | STATUS_FLAG_DEBUFF
    | STATUS_FLAG_DISABLE_SPRINT
    | STATUS_FLAG_CONTROL_IMMUNITY_GATED
)


def _reduce_query_bits(value, relevant):
    return jnp.bitwise_or.reduce(
        jnp.where(relevant, value, jnp.uint32(0)),
        axis=1,
    )


def _pad_queries(value, fill):
    missing = SELECTOR_CAPACITY - value.shape[1]
    if missing < 0:
        raise ValueError(f"query axis exceeds capacity {SELECTOR_CAPACITY}")
    if missing == 0:
        return value
    padding = [(0, 0)] * value.ndim
    padding[1] = (0, missing)
    return jnp.pad(value, padding, constant_values=fill)


def _gather(array, slot):
    batch = jnp.arange(array.shape[0]).reshape(
        (array.shape[0],) + (1,) * (slot.ndim - 1)
    )
    return array[batch, slot]


def _batch_dt(value, batch):
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"dt_seconds must be scalar or [{batch}]")
    return result


def _fold_random_keys(keys, namespace):
    if keys is None:
        return None
    return jax.vmap(lambda key: jax.random.fold_in(key, jnp.uint32(namespace)))(keys)


def _set_failure(bits, mask, code):
    return jnp.where(mask, bits | jnp.uint32(code), bits)


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
