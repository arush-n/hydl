"""Crowd projectile, explosion, and persistent-area integration."""

# ruff: noqa: F401,F403

from hytalegym.jax.combat.entities.impacts.schema.contract import *
from hytalegym.jax.combat.entities.impacts.producers.binding import (
    bind_arsenal_launches,
    set_projectile_impact_yaw,
)
from hytalegym.jax.combat.entities.impacts.producers.areas import (
    launch_entity_areas,
)
from hytalegym.jax.combat.entities.impacts.factory import (
    empty_entity_area_launch_commands,
    empty_entity_area_launch_world,
    empty_entity_impact_bindings,
    empty_entity_impact_queries,
    empty_entity_impact_state,
    empty_entity_projectile_launch_commands,
    empty_entity_projectile_launch_world,
    empty_entity_ranged_launch_world,
    entity_impact_state,
    hytale_0_5_7_entity_area_programs,
    hytale_0_5_7_entity_projectile_programs,
    hytale_0_5_7_ranged_projectile_programs,
)
from hytalegym.jax.combat.entities.impacts.runtime.kernel import (
    tick_entity_impacts,
)
from hytalegym.jax.combat.entities.impacts.producers.launch import (
    launch_entity_projectiles,
    launch_ranged_controller_projectiles,
)
from hytalegym.jax.combat.entities.impacts.schema.spec import (
    entity_impacts_contract_json,
    entity_impacts_contract_manifest,
    entity_impacts_contract_sha256,
)
from hytalegym.jax.combat.entities.impacts.schema.types import *
from hytalegym.jax.combat.entities.impacts.schema.validation import (
    validate_impact_layout,
)
from hytalegym.jax.combat.entities.impacts.producers.world_binding import (
    bind_entity_only_explosion_candidates,
)

__all__ = [name for name in globals() if not name.startswith("_")]
