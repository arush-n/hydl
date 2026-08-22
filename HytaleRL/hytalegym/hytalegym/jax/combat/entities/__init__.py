"""Fixed-capacity logical entities for multi-target JAX combat."""

from hytalegym.jax.combat.entities.schema.contract import *
from hytalegym.jax.combat.entities.runtime.damage import (
    apply_selected_entity_damage,
    pack_entity_damage,
)
from hytalegym.jax.combat.entities.factory import (
    empty_entity_combat_state,
    empty_entity_damage_payloads,
    empty_entity_lifecycle_commands,
    empty_entity_roster,
    empty_entity_selector_queries,
    empty_entity_status_payloads,
)
from hytalegym.jax.combat.entities.runtime.lifecycle import (
    apply_entity_combat_lifecycle,
    apply_entity_lifecycle,
)
from hytalegym.jax.combat.entities.runtime.selectors import select_entity_targets
from hytalegym.jax.combat.entities.schema.spec import (
    HYTALE_0_5_7_SERVER_JAR_SHA256,
    combat_entities_contract_json,
    combat_entities_contract_manifest,
    combat_entities_contract_sha256,
)
from hytalegym.jax.combat.entities.runtime.status import (
    apply_selected_entity_statuses,
    pack_entity_statuses,
)
from hytalegym.jax.combat.entities.schema.types import *
from hytalegym.jax.combat.entities.schema.validation import (
    invalid_roster_rows,
    validate_damage_payload_layout,
    validate_lifecycle_layout,
    validate_roster_layout,
    validate_selector_layout,
    validate_selection_layout,
    validate_status_payload_layout,
)

__all__ = [name for name in globals() if not name.startswith("_")]
