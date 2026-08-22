"""Source-backed ordering contract for Hytale 0.5.7 explosions."""

from __future__ import annotations

import hashlib
import json

from hytalegym.worldgen.native_explosion import (
    native_explosion_candidate_contract_sha256,
)
from hytalegym.worldgen.native_explosion_mutation import (
    native_explosion_mutation_host_contract_sha256,
)


NATIVE_EXPLOSION_ORDERING_SCHEMA = "hytalerl_native_explosion_ordering_v1"
NATIVE_EXPLOSION_ORDERING_VERSION = 1
EXPLOSION_UTILS_CLASS_SHA256 = (
    "03FE187757565EA414053D795136044F3362246B26136B5B1A6387DDA2F6D4B5"
)


def native_explosion_ordering_contract() -> dict[str, object]:
    """Describe execution phases without inventing hash-set iteration order."""

    return {
        "schema": NATIVE_EXPLOSION_ORDERING_SCHEMA,
        "version": NATIVE_EXPLOSION_ORDERING_VERSION,
        "native_source": {
            "class": "com.hypixel.hytale.server.core.entity.ExplosionUtils",
            "class_sha256": EXPLOSION_UTILS_CLASS_SHA256,
            "entry_point": "performExplosion",
        },
        "execution_phases": [
            "processTargetBlocks_ray_admission_and_block_mutation",
            "processTargetEntities_damage_then_knockback_per_entity",
            "performExplosionEffects_particles_then_sound",
        ],
        "phase_order": "strict_blocks_then_entities_then_visual_effects",
        "within_phase_order": {
            "blocks": "ObjectOpenHashSet_iteration_unspecified",
            "entities": "ReferenceOpenHashSet_iteration_unspecified",
            "not_a_fidelity_key": True,
        },
        "wire_order": {
            "entity_candidate_probe": "UUID_text_ascending_serialization_only",
            "explosion_mutation_probe": "canonical_XYZ_serialization_only",
            "native_execution_order_claimed": False,
        },
        "safe_consumer": (
            "identity_indexed_permutation_equivariant_aggregation;fail_closed_"
            "when_same_explosion_entity_effects_are_order_dependent"
        ),
        "upstream": {
            "candidate_contract_sha256": (native_explosion_candidate_contract_sha256()),
            "mutation_contract_sha256": (
                native_explosion_mutation_host_contract_sha256()
            ),
        },
    }


def native_explosion_ordering_contract_sha256() -> str:
    payload = json.dumps(
        native_explosion_ordering_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "EXPLOSION_UTILS_CLASS_SHA256",
    "NATIVE_EXPLOSION_ORDERING_SCHEMA",
    "NATIVE_EXPLOSION_ORDERING_VERSION",
    "native_explosion_ordering_contract",
    "native_explosion_ordering_contract_sha256",
]
