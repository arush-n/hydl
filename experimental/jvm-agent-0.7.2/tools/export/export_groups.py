"""Export the raw structured fields the observation encoder consumes.

Deliberately exports *unmasked, un-reshaped, full-entity-axis* arrays. The Java
side must do the selection (agent row), the mask multiply, the reshape and the
concatenation itself -- that is the logic being ported, and exporting
pre-flattened groups would test nothing.

Mirrors `observation/v3/policy/layout.py:405-455`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

JVM_AGENT_ROOT = Path(__file__).resolve().parents[2]
CASE = JVM_AGENT_ROOT / "case" / "groups"
PROFILE, SEED = "iron_sword", 570057


def main() -> int:
    import jax
    import jax.numpy as jnp
    from hytalegym.jax.combat import (
        arsenal_runtime_config, default_combat_params, hytale_0_5_7_loadouts,
    )
    from hytalegym.jax.combat.observation.v3.policy import (
        arsenal_policy_observation,
    )
    from hytalegym.jax.combat.observation.v3.policy.layout import (
        _ARSENAL_POLICY_COMBAT_FLOAT_INDICES, AGENT_ENTITY,
        align_actor_light_policy_tokens, mask_inventory_policy_tokens,
    )
    from hytalegym.jax.combat.observation.v3.tokens.inventory import (
        inventory_policy_tokens_from_state,
    )
    from hytalegym.jax.training import (
        make_arsenal_ppo_environment, open_flat_arsenal_world_capabilities,
    )
    from adk.validation import reset_keys

    region = len(sys.argv) > 1 and sys.argv[1] == "region"
    global CASE
    if region:
        CASE = CASE.parent / "groups-region"
    CASE.mkdir(parents=True, exist_ok=True)

    if region:
        # Real terrain, so the 5,764 geometry columns carry signal instead of
        # zeros. A flat combat scene leaves ~99% of the vector at zero, where
        # an assembler bug in those columns would pass unnoticed.
        from worlds.region import region_scene

        scene = region_scene(weapons=(PROFILE,), held_item=None,
                             native_evidence=False)
        runtime = scene.runtime
        state, observation, _ = scene.environment.reset(
            jnp.stack([jax.random.key(SEED)])
        )
    else:
        runtime = arsenal_runtime_config(hytale_0_5_7_loadouts((PROFILE,)))
        env = make_arsenal_ppo_environment(
            default_combat_params(microticks=1, target_active=True),
            runtime,
            world_capability_provider=open_flat_arsenal_world_capabilities,
        )
        state, observation, _ = env.reset(reset_keys(SEED, 1))
    reference = np.asarray(observation)[0]
    o = state.learner_observation

    inventory = mask_inventory_policy_tokens(
        inventory_policy_tokens_from_state(
            state.runtime.inventory, runtime.inventory_layout,
            actor_valid=o.valid,
        ),
        o.valid,
    )
    light = align_actor_light_policy_tokens(
        state.light_tokens,
        o.world_geometry.token_mask,
        o.world_geometry.available,
        o.valid,
    )
    block = state.action_surface.block_candidates
    recipe = state.action_surface.recipe_encoding

    # Raw fields, batch row 0. Entity-axis arrays keep the entity axis so the
    # Java side has to select AGENT_ENTITY itself.
    fields = {
        "base_self_f32": o.base.self_f32,
        "base_target_f32": o.base.target_f32,
        "base_target_mask": o.base.target_mask,
        "base_combat_f32": o.base.combat_f32,
        "resource_f32": o.resource_f32,
        "resource_mask": o.resource_mask,
        "defense_f32": o.defense_f32,
        "status_f32": o.status_f32,
        "status_mask": o.status_mask,
        "ability_f32": o.ability_f32,
        "ability_mask": o.ability_mask,
        "ability_legal": o.ability_legal,
        "actor_world_f32": o.actor_world_f32,
        "actor_world_mask": o.actor_world_mask,
        "movement_state_f32": o.movement_state_f32,
        "movement_state_mask": o.movement_state_mask,
        "geometry_token_f32": o.world_geometry.token_f32,
        "geometry_token_mask": o.world_geometry.token_mask,
        "geometry_available": o.world_geometry.available,
        "light_token_f32": light.token_f32,
        "light_available": light.available,
        "inventory_container_f32": inventory.container_f32,
        "inventory_container_mask": inventory.container_mask,
        "inventory_token_f32": inventory.token_f32,
        "inventory_token_mask": inventory.token_mask,
        "inventory_available": inventory.available,
        "skill_action_mask": o.skill_action_mask,
        "jump_action_mask": o.jump_action_mask,
        "guard_action_mask": o.guard_action_mask,
        "dodge_action_mask": o.dodge_action_mask,
        "valid": o.valid,
        "block_candidate_f32": block.candidate_f32,
        "block_candidate_mask": block.candidate_mask,
        "block_available": block.available,
        "recipe_embedding": recipe.candidate_embedding,
        "recipe_mask": recipe.candidate_mask,
        "recipe_available": recipe.available,
    }

    manifest = {
        "agent_entity": int(AGENT_ENTITY),
        "combat_float_indices": [
            int(i) for i in _ARSENAL_POLICY_COMBAT_FLOAT_INDICES
        ],
        "observation_size": int(reference.size),
        "fields": {},
    }
    for name, value in fields.items():
        array = np.asarray(value)[0]  # drop the batch axis
        boolean = array.dtype == np.bool_
        flat = np.ascontiguousarray(
            array.astype(np.uint8 if boolean else np.float32)
        )
        (CASE / f"{name}.bin").write_bytes(flat.tobytes())
        manifest["fields"][name] = {
            "shape": [int(d) for d in array.shape],
            "kind": "bool" if boolean else "f32",
        }
        print(f"  {name:26s} {str(array.shape):18s} {'bool' if boolean else 'f32'}")

    np.ascontiguousarray(reference, dtype=np.float32).tofile(
        CASE / "expected_observation.bin"
    )
    (CASE / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    print(f"\nAGENT_ENTITY = {manifest['agent_entity']}")
    print(f"combat float indices ({len(manifest['combat_float_indices'])}): "
          f"{manifest['combat_float_indices']}")
    print(f"wrote {len(fields)} fields + expected_observation to {CASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
