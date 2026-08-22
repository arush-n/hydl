"""Export whole states -- raw inputs *and* all 37 groups -- for the end-to-end test.

The three existing tests each certify one link and none of them certify the
chain:

  * `ProjectionTest` -- raw simulation state -> the derived fields
  * `AssembleTest`   -- 37 structured fields -> the 8,271-column vector
  * `Main`           -- an 8,271-column vector -> logits and value

A stage can be correct in isolation and still be wired up wrong: a derived group
written to the wrong slot, the agent row taken from the wrong entity, a field
read from a stale copy. That failure survives all three tests and appears only
when one state is pushed through the whole pipeline.

**Several states, not one.** A reset state has the agent at rest -- velocity
zero, yaw round, target idle -- which is the regime *least* likely to expose a
float32 difference or a masking mistake. Parity there is real but weak evidence.
So this steps the environment and captures a few states along the way, and
refuses to write anything unless they actually differ from rest.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

JVM_AGENT_ROOT = Path(__file__).resolve().parents[2]
OUT = JVM_AGENT_ROOT / "case" / "parity"
SEED = 570057
MOVEMENT_STATE_COUNT = 23
DEFAULT_AGENT_WALK_MAX_DROP_HEIGHT = 3.0

#: One weapon per logical batch lane. The exporter evaluates these in bounded
#: physical chunks: compiling all twelve together peaked above 41 GB private
#: memory on Windows, close enough to the commit limit to kill the host before
#: the fixture gate could finish.
#:
#: Chosen to span *kinds*, not tiers: two weapons of the same kind differ only
#: in damage numbers and exercise the same columns. Melee reach and swing,
#: shield and guard, bow and crossbow projectiles, staff and spellbook casts,
#: thrown bombs and consumed potions all drive different ability loadouts,
#: different legality masks and different action surfaces.
PROFILES = (
    "iron_sword",       # the baseline the other fixtures use
    "iron_mace",
    "iron_daggers",
    "iron_spear",
    "iron_battleaxe",
    "iron_shield",      # defensive: guard and block columns
    "iron_shortbow",    # projectiles
    "iron_crossbow",
    "iron_staff",
    "fire_spellbook",   # casts and area effects
    "bombs",            # thrown
    "potions",          # consumed, no weapon at all
)

#: Opponent per lane, rotated one place so no matchup is a mirror. A mirror
#: matchup makes the agent and target rows suspiciously similar, which is
#: exactly the condition under which a row-selection bug hides.
OPPONENTS = PROFILES[1:] + PROFILES[:1]

#: Fewer lanes for the region scene, which carries real terrain per lane and is
#: far heavier. Breadth there comes from the geometry, not the weapon count.
REGION_PROFILES = ("iron_sword", "iron_shield", "iron_shortbow", "iron_staff")

#: Ticks to capture at. 0 is the reset state; the later ones let the target
#: close, turn and attack, which is where the target row and the combat row
#: stop being trivial.
#:
#: Kept short deliberately. The arsenal state PyTree is large, and an
#: unchunked 12-lane run exceeded 41 GB private memory. The diversity that
#: matters comes from the weapons and the early fight, not from tick count.
CAPTURE_AT = (0, 8, 24, 48)
DEFAULT_FLAT_PROFILE_CHUNK_SIZE = 3


def _pack_bool_bits(values: np.ndarray) -> int:
    packed = 0
    for index, enabled in enumerate(np.asarray(values, dtype=np.bool_)):
        if bool(enabled):
            packed |= 1 << index
    return packed


def _policy_equivalent_raw_ability(
    values: np.ndarray,
    masks: np.ndarray,
    legal: np.ndarray,
) -> np.ndarray:
    """Invert a finished ability row for integration-only fixture migration.

    ProjectionTest certifies the real JAX derivation. Existing parity captures
    predate raw ability export, so this constructs one valid raw state whose
    projection is exactly the recorded row without claiming it was the
    historical server state.
    """

    ability = np.asarray(values, dtype=np.float32).reshape(2, 16, 11)
    mask = np.asarray(masks, dtype=np.uint8).reshape(2, 16)
    legal = np.asarray(legal, dtype=np.uint8).reshape(2, 16)
    resources = np.zeros((2, 7), dtype=np.float32)
    spans = np.ones((2, 7), dtype=np.float32)
    durations = ability[..., 0].copy()
    authored_cooldowns = np.ones((2, 16), dtype=np.float32)
    remaining_cooldowns = ability[..., 1].copy()
    active_slots = np.full((2,), -1, dtype=np.float32)
    elapsed = np.zeros((2,), dtype=np.float32)
    for entity in range(2):
        active = np.flatnonzero(np.abs(ability[entity, :, 2]) > 1.0e-8)
        if active.size > 1:
            raise SystemExit(
                "ability fixture has more than one active-progress slot for "
                f"entity {entity}: {active.tolist()}"
            )
        if active.size == 1:
            selected = int(active[0])
            active_slots[entity] = float(selected)
            elapsed[entity] = (
                ability[entity, selected, 2]
                * max(float(durations[entity, selected]), 1.0e-6)
            )
    authored_cost = ability[..., 4:].copy()
    cost_kind = np.ones((2, 16, 7), dtype=np.float32)
    minima = np.zeros((2, 16, 7), dtype=np.float32)
    unaffordable = (ability[..., 3] < 0.5) & (mask != 0)
    minima[..., 0] = unaffordable.astype(np.float32)
    raw = np.concatenate(
        (
            resources.reshape(-1),
            spans.reshape(-1),
            durations.reshape(-1),
            authored_cooldowns.reshape(-1),
            remaining_cooldowns.reshape(-1),
            active_slots,
            elapsed,
            authored_cost.reshape(-1),
            cost_kind.reshape(-1),
            minima.reshape(-1),
            mask.astype(np.float32).reshape(-1),
            np.ones((2,), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
            legal.astype(np.float32).reshape(-1),
        )
    ).astype(np.float32)
    if raw.shape != (868,):
        raise SystemExit(f"policy-equivalent raw ability width drift: {raw.shape}")
    return raw


def _policy_equivalent_raw_inventory(
    available: np.ndarray,
    container_values: np.ndarray,
    container_masks: np.ndarray,
    token_values: np.ndarray,
    token_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert one finished inventory policy row for parity integration."""

    row_available = bool(np.asarray(available).reshape(-1)[0])
    containers = np.asarray(container_values, dtype=np.float32).reshape(6)
    container_mask = np.asarray(container_masks, dtype=np.uint8).reshape(6)
    tokens = np.asarray(token_values, dtype=np.float32).reshape(76, 7)
    token_mask = np.asarray(token_masks, dtype=np.uint8).reshape(76)
    defaults = np.asarray((36, 4, 9, 4, 23, 0), dtype=np.int64)
    capacities = np.zeros((6,), dtype=np.int64)
    capacities[:5] = np.rint(containers[:5] * defaults[:5]).astype(np.int64)
    capacities[5] = int(round(float(containers[5]) * np.iinfo(np.int32).max))
    capacities = np.clip(capacities, 0, np.iinfo(np.int32).max).astype(np.int32)

    container_id = np.rint(tokens[:, 0] * 5.0).astype(np.int32)
    safe_container = np.clip(container_id, 0, 5)
    selected_capacity = capacities[safe_container]
    container_slot = np.rint(
        tokens[:, 1] * (selected_capacity.astype(np.float64) + 1.0) - 1.0
    ).astype(np.int32)
    low = np.rint(tokens[:, 2] * 0xFFFF).astype(np.int64)
    high = np.rint(tokens[:, 3] * 0x7FFF).astype(np.int64)
    item_id = (low | (high << 16)).astype(np.int32)
    quantity = np.rint(
        np.exp2(tokens[:, 4].astype(np.float64) * 31.0) - 1.0
    )
    quantity = np.clip(quantity, 0, np.iinfo(np.int32).max).astype(np.int32)
    durability = tokens[:, 5].astype(np.float32)
    active = (tokens[:, 6] > 0.5).astype(np.float32)
    flags = np.concatenate(
        (
            np.asarray((row_available,), dtype=np.uint8),
            container_mask.astype(np.uint8),
            token_mask.astype(np.uint8),
            active.astype(np.uint8),
        )
    ).astype(np.uint8)
    integers = np.concatenate(
        (capacities, container_id, container_slot, item_id, quantity)
    ).astype(np.int32)
    floats = durability.astype(np.float32)
    if flags.shape != (159,) or integers.shape != (310,) or floats.shape != (76,):
        raise SystemExit(
            "policy-equivalent raw inventory width drift: "
            f"{flags.shape}/{integers.shape}/{floats.shape}"
        )
    return flags, integers, floats


def _augment_existing_structured_inputs() -> int:
    """Add raw inputs for certified structured groups without rerunning rollout.

    The finished movement arrays are a lossless representation of the raw JAX
    Walk value/availability bitsets. Other groups are inverted into a
    policy-equivalent raw representation. ProjectionTest separately uses the
    real JAX projector and exercises every derivation gate; this migration only
    proves those derived values travel through the production assembler.
    """

    updates: list[tuple[Path, str, dict[str, bytes]]] = []
    for directory in sorted(path for path in OUT.iterdir() if path.is_dir()):
        inputs = directory / "inputs.txt"
        value_path = directory / "movement_state_f32.bin"
        mask_path = directory / "movement_state_mask.bin"
        if not inputs.exists():
            continue
        values = np.fromfile(value_path, dtype=np.float32)
        masks = np.fromfile(mask_path, dtype=np.uint8)
        actor_values = np.fromfile(
            directory / "actor_world_f32.bin", dtype=np.float32
        )
        actor_masks = np.fromfile(
            directory / "actor_world_mask.bin", dtype=np.uint8
        )
        resource_values = np.fromfile(
            directory / "resource_f32.bin", dtype=np.float32
        )
        resource_masks = np.fromfile(
            directory / "resource_mask.bin", dtype=np.uint8
        )
        defense_values = np.fromfile(
            directory / "defense_f32.bin", dtype=np.float32
        )
        status_values = np.fromfile(
            directory / "status_f32.bin", dtype=np.float32
        )
        status_masks = np.fromfile(
            directory / "status_mask.bin", dtype=np.uint8
        )
        dodge_masks = np.fromfile(
            directory / "dodge_action_mask.bin", dtype=np.uint8
        )
        ability_values = np.fromfile(
            directory / "ability_f32.bin", dtype=np.float32
        )
        ability_masks = np.fromfile(
            directory / "ability_mask.bin", dtype=np.uint8
        )
        ability_legal = np.fromfile(
            directory / "ability_legal.bin", dtype=np.uint8
        )
        inventory_container_values = np.fromfile(
            directory / "inventory_container_f32.bin", dtype=np.float32
        )
        inventory_container_masks = np.fromfile(
            directory / "inventory_container_mask.bin", dtype=np.uint8
        )
        inventory_token_values = np.fromfile(
            directory / "inventory_token_f32.bin", dtype=np.float32
        )
        inventory_token_masks = np.fromfile(
            directory / "inventory_token_mask.bin", dtype=np.uint8
        )
        inventory_available = np.fromfile(
            directory / "inventory_available.bin", dtype=np.uint8
        )
        if values.shape != (MOVEMENT_STATE_COUNT,) or masks.shape != (
            MOVEMENT_STATE_COUNT,
        ):
            raise SystemExit(
                f"{directory.name}: movement fixture width drift "
                f"{values.shape}/{masks.shape}"
            )
        if not np.all(np.isin(values, (0.0, 1.0))):
            raise SystemExit(
                f"{directory.name}: movement values are not a boolean bitset"
            )
        if not np.all(np.isin(masks, (0, 1))):
            raise SystemExit(
                f"{directory.name}: movement masks are not boolean"
            )
        if np.any((values != 0.0) & (masks == 0)):
            raise SystemExit(
                f"{directory.name}: unavailable movement value is nonzero"
            )
        if actor_values.shape != (5,) or actor_masks.shape != (3,):
            raise SystemExit(
                f"{directory.name}: actor-world fixture width drift "
                f"{actor_values.shape}/{actor_masks.shape}"
            )
        if not np.all(np.isin(actor_masks, (0, 1))):
            raise SystemExit(
                f"{directory.name}: actor-world masks are not boolean"
            )
        if resource_values.shape != (14,) or resource_masks.shape != (14,):
            raise SystemExit(
                f"{directory.name}: resource fixture width drift "
                f"{resource_values.shape}/{resource_masks.shape}"
            )
        if not np.all(np.isin(resource_masks, (0, 1))):
            raise SystemExit(f"{directory.name}: resource masks are not boolean")
        if defense_values.shape != (14,):
            raise SystemExit(
                f"{directory.name}: defense fixture width drift "
                f"{defense_values.shape}"
            )
        if status_values.shape != (96,) or status_masks.shape != (16,):
            raise SystemExit(
                f"{directory.name}: status fixture width drift "
                f"{status_values.shape}/{status_masks.shape}"
            )
        if not np.all(np.isin(status_masks, (0, 1))):
            raise SystemExit(f"{directory.name}: status masks are not boolean")
        if dodge_masks.shape != (4,) or not np.all(np.isin(dodge_masks, (0, 1))):
            raise SystemExit(
                f"{directory.name}: dodge fixture width/value drift "
                f"{dodge_masks.shape}"
            )
        if (
            ability_values.shape != (352,)
            or ability_masks.shape != (32,)
            or ability_legal.shape != (32,)
        ):
            raise SystemExit(
                f"{directory.name}: ability fixture width drift "
                f"{ability_values.shape}/{ability_masks.shape}/"
                f"{ability_legal.shape}"
            )
        if not np.all(np.isin(ability_masks, (0, 1))) or not np.all(
            np.isin(ability_legal, (0, 1))
        ):
            raise SystemExit(f"{directory.name}: ability masks are not boolean")
        if (
            inventory_container_values.shape != (6,)
            or inventory_container_masks.shape != (6,)
            or inventory_token_values.shape != (532,)
            or inventory_token_masks.shape != (76,)
            or inventory_available.shape != (1,)
        ):
            raise SystemExit(
                f"{directory.name}: inventory fixture width drift"
            )
        if not np.all(np.isin(inventory_container_masks, (0, 1))) or not np.all(
            np.isin(inventory_token_masks, (0, 1))
        ) or not np.all(np.isin(inventory_available, (0, 1))):
            raise SystemExit(
                f"{directory.name}: inventory masks are not boolean"
            )
        value_bits = _pack_bool_bits(values != 0.0)
        available_bits = _pack_bool_bits(masks != 0)
        actor_keys = {
            "actor_controller_medium_available",
            "actor_controller_in_fluid",
            "actor_submersion_available",
            "actor_feet_submerged",
            "actor_eyes_submerged",
            "actor_drop_available",
            "actor_drop_support_found",
            "actor_drop_height",
            "agent_walk_max_drop_height",
        }
        lines = [
            line
            for line in inputs.read_text(encoding="utf-8").splitlines()
            if not line.startswith("movement_value_bits\t")
            and not line.startswith("movement_available_bits\t")
            and line.split("\t", 1)[0] not in actor_keys
        ]
        lines.extend(
            (
                f"movement_value_bits\t{float(value_bits)!r}",
                f"movement_available_bits\t{float(available_bits)!r}",
                f"actor_controller_medium_available\t{float(actor_masks[0])!r}",
                f"actor_controller_in_fluid\t{float(actor_values[0] > 0.5)!r}",
                f"actor_submersion_available\t{float(actor_masks[1])!r}",
                f"actor_feet_submerged\t{float(actor_values[1] > 0.5)!r}",
                f"actor_eyes_submerged\t{float(actor_values[2] > 0.5)!r}",
                f"actor_drop_available\t{float(actor_masks[2])!r}",
                f"actor_drop_support_found\t{float(actor_values[3] > 0.5)!r}",
                "actor_drop_height\t"
                f"{float(actor_values[4] * DEFAULT_AGENT_WALK_MAX_DROP_HEIGHT)!r}",
                "agent_walk_max_drop_height\t"
                f"{DEFAULT_AGENT_WALK_MAX_DROP_HEIGHT!r}",
            )
        )
        resource_minimum = np.zeros((14,), dtype=np.float32)
        resource_maximum = resource_masks.astype(np.float32)
        defense = defense_values.reshape(2, 7)
        defense_velocity = np.zeros((2, 3), dtype=np.float32)
        defense_velocity[:, 0] = defense[:, 3]
        raw_defense = np.concatenate(
            (
                defense[:, 0],
                defense[:, 1],
                defense[:, 2],
                np.ones((2,), dtype=np.float32),
                defense_velocity.reshape(-1),
                np.ones((2,), dtype=np.float32),
                defense[:, 4] * np.float32(2.0),
                defense[:, 5] * np.float32(100.0),
                defense[:, 6],
            )
        ).astype(np.float32)
        status = status_values.reshape(2, 8, 6)
        active = status_masks.astype(np.float32)
        raw_status = np.concatenate(
            (
                np.ones((2,), dtype=np.float32),
                np.ones((14,), dtype=np.float32),
                (status[..., 0] * np.float32(30.0)).reshape(-1),
                status[..., 1].reshape(-1),
                np.ones((16,), dtype=np.float32),
                status[..., 2].reshape(-1),
                status[..., 3].reshape(-1),
                np.zeros((16,), dtype=np.float32),
                status[..., 4].reshape(-1),
                (status[..., 5] * np.float32(2.0)).reshape(-1),
                active,
            )
        ).astype(np.float32)
        if raw_status.shape != (160,):
            raise SystemExit(
                f"{directory.name}: raw status width drift {raw_status.shape}"
            )
        raw_dodge = np.asarray(
            (
                1.0,
                1.0,
                float(dodge_masks[2]),
                float(dodge_masks[3]),
                1.0,  # movement enabled
                1.0,  # alive
                2.0,  # stamina
                2.0,  # authored dodge cost
            ),
            dtype=np.float32,
        )
        raw_ability = _policy_equivalent_raw_ability(
            ability_values, ability_masks, ability_legal
        )
        raw_inventory_flags, raw_inventory_i32, raw_inventory_f32 = (
            _policy_equivalent_raw_inventory(
            inventory_available,
            inventory_container_values,
            inventory_container_masks,
            inventory_token_values,
            inventory_token_masks,
            )
        )
        raw = {
            "raw_resource_current.bin": np.ascontiguousarray(
                resource_values, dtype=np.float32
            ).tobytes(),
            "raw_resource_minimum.bin": resource_minimum.tobytes(),
            "raw_resource_maximum.bin": resource_maximum.tobytes(),
            "raw_defense.bin": raw_defense.tobytes(),
            "raw_status.bin": raw_status.tobytes(),
            "raw_dodge.bin": raw_dodge.tobytes(),
            "raw_ability.bin": raw_ability.tobytes(),
            "raw_inventory_flags.bin": raw_inventory_flags.tobytes(),
            "raw_inventory_i32.bin": raw_inventory_i32.tobytes(),
            "raw_inventory_f32.bin": raw_inventory_f32.tobytes(),
        }
        updates.append((inputs, "\n".join(lines) + "\n", raw))
    if not updates:
        raise SystemExit("no existing parity states were found to augment")
    # Validate every state before replacing the first file.
    for path, content, raw in updates:
        path.write_text(content, encoding="utf-8")
        for name, payload in raw.items():
            (path.parent / name).write_bytes(payload)
    print(f"augmented structured raw inputs in {len(updates)} parity states")
    return 0


def _migrate_arsenal_attack_slot_divisor() -> int:
    """Move existing captures from legacy attack count to learner slot width.

    The captured combat column is already the Arsenal compatibility value.
    Only the raw scalar used by Java to reconstruct its authored slot was
    stale.  Validate every state before replacing any file so a partial
    migration cannot leave one fixture generation with two meanings.
    """

    from hytalegym.jax.combat.arsenal.schema.contract import (
        OBSERVATION_CAPACITY,
    )

    divisor = float(max(OBSERVATION_CAPACITY - 1, 1))
    updates: list[tuple[Path, str]] = []
    for directory in sorted(path for path in OUT.iterdir() if path.is_dir()):
        inputs = directory / "inputs.txt"
        if not inputs.exists():
            continue
        lines = inputs.read_text(encoding="utf-8").splitlines()
        matches = [
            index
            for index, line in enumerate(lines)
            if line.startswith("target_attack_index_divisor\t")
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"{inputs} has {len(matches)} attack-slot divisor rows"
            )
        index = matches[0]
        current = float(lines[index].split("\t", 1)[1])
        if current not in {4.0, divisor}:
            raise SystemExit(
                f"{inputs} has unexpected attack-slot divisor {current}"
            )
        lines[index] = f"target_attack_index_divisor\t{divisor}"
        updates.append((inputs, "\n".join(lines) + "\n"))

    if not updates:
        raise SystemExit("no parity states found for attack-slot migration")
    for path, content in updates:
        path.write_text(content, encoding="utf-8")
    print(
        f"migrated {len(updates)} parity states to authored ability-slot "
        f"divisor {divisor:g}"
    )
    return 0


def _mode_state_directories(path: Path, *, region: bool) -> list[Path]:
    """Return only the flat or Region state directories in ``path``."""

    if not path.exists():
        return []
    return sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and child.name.startswith("region_") == region
    )


def _commit_mode_fixture(
    staging_out: Path,
    target_out: Path,
    *,
    region: bool,
) -> None:
    """Install one completely-exported mode with rollback on ordinary errors.

    Exporting the flat corpus can take more than an hour.  Mutating the live
    fixture before that work finishes turns an OOM, Ctrl-C, or wrapper timeout
    into a half-written certified corpus.  Keep the long-running work beside
    the target on the same volume, validate it, and make the destructive
    interval the final rename-only commit.

    The old manifest is moved first.  A hard process kill during the short
    commit window therefore fails the parity gate visibly instead of blessing
    mixed old/new directories.  The hash-named rollback directory is retained
    by the filesystem in that case; normal Python exceptions restore it.
    """

    mode = "region" if region else "flat"
    staged_manifest = staging_out / f"manifest_{mode}.json"
    if not staged_manifest.is_file():
        raise RuntimeError(f"staged {mode} export has no manifest")
    manifest = json.loads(staged_manifest.read_text(encoding="utf-8"))
    declared = {
        str(state["directory"])
        for state in manifest.get("states", ())
    }
    staged_directories = _mode_state_directories(staging_out, region=region)
    present = {directory.name for directory in staged_directories}
    if not declared or declared != present:
        raise RuntimeError(
            f"staged {mode} manifest/directory mismatch: "
            f"declared={sorted(declared)} present={sorted(present)}"
        )

    target_out.mkdir(parents=True, exist_ok=True)
    target_manifest = target_out / staged_manifest.name
    rollback = Path(
        tempfile.mkdtemp(
            prefix=f".parity-{mode}-rollback-",
            dir=target_out.parent,
        )
    )
    installed: list[Path] = []
    try:
        if target_manifest.exists():
            target_manifest.replace(rollback / target_manifest.name)
        for directory in _mode_state_directories(target_out, region=region):
            directory.replace(rollback / directory.name)
        for directory in staged_directories:
            destination = target_out / directory.name
            directory.replace(destination)
            installed.append(destination)
        staged_manifest.replace(target_manifest)
    except BaseException:
        for directory in reversed(installed):
            if directory.exists():
                directory.replace(staging_out / directory.name)
        for previous in sorted(rollback.iterdir()):
            previous.replace(target_out / previous.name)
        raise
    else:
        shutil.rmtree(rollback)
        shutil.rmtree(staging_out)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "migrate-attack-slot-divisor":
        return _migrate_arsenal_attack_slot_divisor()
    if len(sys.argv) > 1 and sys.argv[1] in {
        "augment-structured", "augment-movement"
    }:
        return _augment_existing_structured_inputs()

    import jax
    import jax.numpy as jnp
    from hytalegym.jax.combat import (
        arsenal_runtime_config,
        default_combat_params,
        hytale_0_5_7_loadouts,
        neutral_arsenal_policy_action_factors,
    )
    # The live Gym restructured `jax/combat` (see its PACKAGE-MAP.md): `_reset`
    # moved to `runtime/reset` and `_target` to `opponents/legacy/controller`.
    # The fallback keeps historical flat-case inspection readable, but current
    # flat and Region fixture publication must run against the live Gym. The
    # frozen snapshot predates `world_actions` and cannot publish this contract.
    try:
        from hytalegym.jax.combat.runtime.reset import target_evidence
        from hytalegym.jax.combat.opponents.legacy.controller import _target_phase
    except ModuleNotFoundError:
        from hytalegym.jax.combat._reset import target_evidence
        from hytalegym.jax.combat._target import _target_phase
    from hytalegym.jax.combat.observation import COMBAT_FLOAT_FEATURES
    from hytalegym.jax.combat.observation.v3.policy.layout import (
        AGENT_ENTITY,
        align_actor_light_policy_tokens,
        arsenal_policy_action_mask,
        mask_inventory_policy_tokens,
    )
    from hytalegym.jax.combat.observation.v3.tokens.inventory import (
        inventory_policy_tokens_from_state,
    )
    from hytalegym.jax.combat.arsenal.programs.ability import (
        ability_lifecycle_legality_view,
    )
    from hytalegym.jax.combat.types import TARGET_ENTITY
    from hytalegym.jax.training import (
        make_arsenal_ppo_environment,
        open_flat_arsenal_world_capabilities,
    )
    from adk.validation import reset_keys, stream

    # `region` swaps the open flat world for real terrain. That matters more
    # than it sounds: the 5,764 geometry columns are all zero in a flat scene,
    # so ~70% of the observation is certified against nothing without it. A
    # pickaxe makes a block interaction legal *when* the native camera ray has
    # selected a reachable block.  It does not guarantee that the reset pose
    # points at one; an empty row is therefore state evidence, not proof that
    # the provider is absent.
    region = len(sys.argv) > 1 and sys.argv[1] == "region"
    all_profiles = REGION_PROFILES if region else PROFILES
    all_opponents = all_profiles[1:] + all_profiles[:1]
    requested_chunk_size = os.environ.get(
        "HYTALERL_PARITY_PROFILE_BATCH_SIZE"
    )
    if requested_chunk_size is None:
        # Region captures only reset rows and its four lanes stay well below
        # the flat stepped-run peak. Keep one Region batch so its established
        # fixture remains byte-for-byte reproducible.
        chunk_size = (
            len(all_profiles) if region else DEFAULT_FLAT_PROFILE_CHUNK_SIZE
        )
    else:
        try:
            chunk_size = int(requested_chunk_size)
        except ValueError as error:
            raise SystemExit(
                "HYTALERL_PARITY_PROFILE_BATCH_SIZE must be an integer"
            ) from error
        if chunk_size < 1:
            raise SystemExit(
                "HYTALERL_PARITY_PROFILE_BATCH_SIZE must be positive"
            )
        chunk_size = min(chunk_size, len(all_profiles))
    # Region geometry is recomputed while stepping, but availability is
    # intermittent: one measured trajectory returned 4 unavailable rows in 12
    # moving steps even though the agent remained inside the capture footprint.
    # Reset is the cheap guaranteed-present capture used by this fixture. Any
    # future stepped Region capture must inspect `geometry_available` and reject
    # an absent row instead of recording zero terrain as coverage.
    capture_at = (0,) if region else CAPTURE_AT
    params = default_combat_params(microticks=1, target_active=True)

    if region:
        from worlds.region import region_scene

    # Per-lane behaviour rather than one shared action. Idling leaves the agent
    # at rest and the body-frame projection certified on zeros; making every
    # lane do the same thing leaves the capture set narrower than its size
    # suggests. Head 0 is base_action -- 2 approach, 3 retreat, 4/5 strafe,
    # 6 attack, 7 approach+attack.
    all_factors = np.asarray(
        neutral_arsenal_policy_action_factors(len(all_profiles))
    ).copy()
    all_factors[:, 0] = (
        [2, 7, 4, 2, 6, 3, 2, 7, 5, 2, 6, 4] * 2
    )[: len(all_profiles)]

    # The long-running export is transactional.  Build beside ``case/parity``
    # so the final directory renames stay on one volume; do not touch the
    # certified corpus until every row and the new manifest are complete.
    mode = "region" if region else "flat"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    staging_out = Path(
        tempfile.mkdtemp(
            prefix=f".parity-{mode}-staging-",
            dir=OUT.parent,
        )
    )
    manifest_path = staging_out / f"manifest_{mode}.json"
    print(f"staging {mode} fixture under {staging_out}")

    total = max(capture_at) + 1
    all_step_keys = stream(SEED, total, len(all_profiles))
    all_reset_keys = reset_keys(SEED, len(all_profiles))
    captured: list[dict] = []
    for chunk_start in range(0, len(all_profiles), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(all_profiles))
        profiles = all_profiles[chunk_start:chunk_end]
        opponents = all_opponents[chunk_start:chunk_end]
        batch = len(profiles)
        print(
            f"profile batch {chunk_start // chunk_size + 1}: "
            f"{profiles}"
        )
        if region:
            scene = region_scene(
                weapons=profiles,
                held_item="Tool_Pickaxe_Iron",
                native_evidence=False,
            )
            runtime = scene.runtime
            env = scene.environment
            reset_argument = jnp.stack([jax.random.key(SEED)] * batch)
        else:
            runtime = arsenal_runtime_config(
                hytale_0_5_7_loadouts(
                    profiles,
                    target_profiles=opponents,
                )
            )
            env = make_arsenal_ppo_environment(
                params,
                runtime,
                world_capability_provider=(
                    open_flat_arsenal_world_capabilities
                ),
            )
            # Preserve the original unchunked lane keys. A chunk boundary is
            # an execution detail and must not change stochastic fixtures.
            reset_argument = all_reset_keys[chunk_start:chunk_end]
        factors = jnp.asarray(all_factors[chunk_start:chunk_end])
        keys = all_step_keys[:, chunk_start:chunk_end]
        state, observation, _ = env.reset(reset_argument)

        for tick in range(total):
            if tick in capture_at:
                for lane, profile in enumerate(profiles):
                    captured.append(
                        capture(
                            staging_out / f"{'region_' if region else ''}"
                                          f"{profile}_t{tick:03d}",
                            tick,
                            lane,
                            profile,
                            state,
                            observation,
                            params,
                            runtime,
                            AGENT_ENTITY,
                            TARGET_ENTITY,
                            COMBAT_FLOAT_FEATURES,
                            target_evidence,
                            _target_phase,
                            align_actor_light_policy_tokens,
                            mask_inventory_policy_tokens,
                            inventory_policy_tokens_from_state,
                            arsenal_policy_action_mask,
                            ability_lifecycle_legality_view,
                        )
                    )
            if tick >= max(capture_at):
                break
            result = env.step_detailed(
                state,
                observation,
                factors,
                keys[tick],
            )
            previous = state
            state, observation = result[0], result[1]
            del result, previous
            if tick % 8 == 0:
                gc.collect()

        del state, observation, env, runtime, factors, keys, reset_argument
        if region:
            del scene
        # XLA retains executables and host literals after their state leaves
        # scope. Clearing at a bounded chunk boundary trades recompilation for
        # a hard memory ceiling on the documented 64-GB host.
        jax.clear_caches()
        gc.collect()

    # A capture set that never leaves rest would make the whole end-to-end test
    # a statement about zeros. Refuse to write one.
    moving = [c for c in captured if abs(c["agent_speed"]) > 1e-6]
    engaged = [c for c in captured if c["target_attack_phase"] != 0]
    hurt = [c for c in captured if c["agent_health_fraction"] < 1.0]
    phases = sorted({c["target_attack_phase"] for c in captured})
    observations = {c["observation_digest"] for c in captured}
    print()
    print(f"captured       : {len(captured)} states "
          f"({len(all_profiles)} weapons x {len(capture_at)} ticks"
          f"{', region' if region else ''})")
    print(f"agent moving   : {len(moving)}/{len(captured)}")
    print(f"target engaged : {len(engaged)}/{len(captured)} (phases {phases})")
    print(f"agent damaged  : {len(hurt)}/{len(captured)}")
    print(f"visible        : "
          f"{sum(1 for c in captured if c['target_visible'] > 0.5)}/{len(captured)}")
    print(f"distinct obs   : {len(observations)}/{len(captured)} unique observations")

    # Breadth is the point of this fixture, so failing to achieve it must be an
    # error rather than a quiet weakening of every claim built on top of it.
    # Motion and combat diversity are the flat fixture's job -- it steps through
    # a fight. The Region fixture deliberately uses the reset row because it is
    # guaranteed to carry geometry; stepped Region rows are valid only after an
    # explicit availability check. Its own requirement is geometry signal,
    # asserted below.
    if not region:
        if not moving:
            raise SystemExit(
                "every captured state has the agent at rest -- the body-frame "
                "projection would be certified on zeros"
            )
        if not engaged:
            raise SystemExit(
                "the target never attacked in any captured state -- the combat "
                "row's phase and progress columns would be certified on zeros"
            )
    if len(observations) < len(all_profiles):
        raise SystemExit(
            f"only {len(observations)} distinct observations across "
            f"{len(captured)} states -- the weapons are not actually producing "
            "different worlds, so the breadth is nominal"
        )

    # Which groups actually carry signal. A group that is zero in every state is
    # certified vacuously: the assembler copies zeros into the right slots and
    # passes without the group's logic mattering.
    live = {
        name
        for c in captured
        for name, count in c["signal"].items()
        if count > 0
    }
    dead = sorted(set(captured[0]["signal"]) - live)
    print(f"groups w/ signal: {len(live)}/{len(captured[0]['signal'])}")
    if dead:
        print(f"all-zero groups : {dead}")

    if region:
        # The whole point of the region scene. Terrain must reach the geometry
        # tokens, and the held pickaxe must reach the block candidates -- the
        # existing `case/groups-region` fixture has geometry but *no* blocks,
        # because it is built with no held item, leaving 978 block and recipe
        # columns certified against zeros.
        required = {
            "geometry_token_f32": "real terrain never reached the tokens",
            "geometry_token_mask": "no geometry token was ever marked present",
            "geometry_available": "the geometry surface never became available",
        }
        missing = {n: why for n, why in required.items() if n not in live}
        if missing:
            raise SystemExit(
                "region capture is missing signal in "
                f"{len(missing)} required group(s):\n  "
                + "\n  ".join(f"{n}: {why}" for n, why in missing.items())
            )
        # Blocks are reported, not required. The camera provider is bound, but
        # this reset pose can legitimately point at no reachable non-air cell;
        # a held pickaxe controls legality after selection, not camera aim.
        # The executor half is separately and knowably unavailable:
        # `evidence_status()` reports every one of its 17 generations stranded
        # against the deployed bridge, which closes `use_off_on`,
        # `block_primary_secondary_trigger` and the recipe slots. Failing the
        # export on that would be failing it for someone else's stale artifact.
        blocks = [
            n for n in ("block_candidate_f32", "block_candidate_mask",
                        "block_available", "recipe_embedding", "recipe_mask",
                        "recipe_available")
            if n not in live
        ]
        if blocks:
            print(f"NOTE: {len(blocks)} block/recipe group(s) still all-zero "
                  f"with terrain and a held pickaxe: {blocks}")
            print("      the reset camera selected no block; recipe/executor "
                  "evidence is also stranded against the deployed bridge.")

    # One manifest per mode. A single shared manifest.json meant whichever
    # export ran last described only its own states and silently orphaned the
    # other mode's -- and ParityTest only checked that *a* manifest existed, so
    # those orphans were still scored. Mode-scoped manifests let the test
    # require that every directory it scores was blessed by some export.
    manifest_path.write_text(
        json.dumps(
            {
                "profiles": list(all_profiles),
                "opponents": list(all_opponents),
                "seed": SEED,
                "captured_at": list(capture_at),
                "states": captured,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    _commit_mode_fixture(staging_out, OUT, region=region)
    print(f"wrote {len(captured)} states to {OUT}")
    return 0


def capture(
    directory,
    tick,
    lane,
    profile,
    state,
    observation,
    params,
    runtime,
    AGENT_ENTITY,
    TARGET_ENTITY,
    COMBAT_FLOAT_FEATURES,
    target_evidence,
    _target_phase,
    align_actor_light_policy_tokens,
    mask_inventory_policy_tokens,
    inventory_policy_tokens_from_state,
    arsenal_policy_action_mask,
    ability_lifecycle_legality_view,
) -> dict:
    """Write one state's raw inputs, all 37 groups, and the observation."""

    # The arsenal compatibility projection publishes the authored ability
    # slot, not the legacy target-attack index.  Keep this normaliser tied to
    # the learner-facing axis so a loadout with fewer attacks cannot silently
    # change the meaning of combat_f32[21].
    from hytalegym.jax.combat.arsenal.schema.contract import (
        OBSERVATION_CAPACITY,
    )

    directory.mkdir(parents=True, exist_ok=True)
    combat = state.runtime.combat
    combat_row = np.asarray(state.learner_observation.base.combat_f32)[lane]
    o = state.learner_observation

    # Arsenal opponents publish their authored-ability lifecycle through the
    # compatibility projection.  The legacy target state machine may remain
    # idle while an authored ability is live, so it is not a valid breadth
    # signal for this fixture.  Keep the legacy values below in ``inputs.txt``
    # (Java still needs to reproduce the raw projection), but use the actual
    # learner-facing phase for the coverage gate and diagnostic output.
    learner_target_phase = int(
        np.argmax(
            [
                combat_row[COMBAT_FLOAT_FEATURES.index(name)]
                for name in (
                    "target_phase_idle",
                    "target_phase_windup",
                    "target_phase_sweep",
                    "target_phase_recovery",
                    "target_phase_cooldown",
                )
            ]
        )
    )

    def scalar(value) -> float:
        return float(np.asarray(value)[lane].reshape(-1)[0])

    def entity(array, index) -> np.ndarray:
        return np.asarray(array)[lane, index]

    def combat_column(name: str) -> float:
        return float(combat_row[COMBAT_FLOAT_FEATURES.index(name)])

    evidence = target_evidence(combat, params)
    phase, progress, reported_index, _ = _target_phase(combat, params)

    agent_velocity = entity(combat.velocity, AGENT_ENTITY)
    agent_position = entity(combat.position, AGENT_ENTITY)
    target_position = entity(combat.position, TARGET_ENTITY)
    target_velocity = entity(combat.velocity, TARGET_ENTITY)
    movement_values = np.asarray(combat.agent_walk_movement_state.values)[lane]
    movement_available = np.asarray(
        combat.agent_walk_movement_state.available)[lane]

    def pack_bits(bits) -> int:
        packed = 0
        for index, enabled in enumerate(np.asarray(bits, dtype=np.bool_)):
            if bool(enabled):
                packed |= 1 << index
        return packed

    values: dict[str, float] = {
        "agent_velocity_x": float(agent_velocity[0]),
        "agent_velocity_y": float(agent_velocity[1]),
        "agent_velocity_z": float(agent_velocity[2]),
        "agent_yaw_degrees": scalar(combat.yaw[:, AGENT_ENTITY]),
        "agent_pitch_degrees": scalar(combat.pitch),
        "agent_health": scalar(combat.health[:, AGENT_ENTITY]),
        "agent_grounded": float(bool(scalar(combat.agent_grounded))),
        "agent_attack_cooldown_seconds": scalar(
            combat.agent_attack_cooldown_seconds),
        "agent_knockback_control_lock": float(
            bool(scalar(combat.knockback_control_lock))),
        "agent_ticks_since_damage": scalar(combat.ticks_since_agent_damage),
        "agent_applied_vertical_velocity": scalar(
            combat.agent_applied_vertical_velocity),
        "agent_fall_speed": scalar(combat.agent_fall_speed),
        "agent_motion_delta_seconds": scalar(combat.last_motion_delta_seconds),
        "agent_attack_executing": combat_column("agent_attack_executing"),
        "agent_position_x": float(agent_position[0]),
        "agent_position_y": float(agent_position[1]),
        "agent_position_z": float(agent_position[2]),
        "target_position_x": float(target_position[0]),
        "target_position_y": float(target_position[1]),
        "target_position_z": float(target_position[2]),
        "target_velocity_x": float(target_velocity[0]),
        "target_velocity_y": float(target_velocity[1]),
        "target_velocity_z": float(target_velocity[2]),
        "target_health": scalar(combat.health[:, TARGET_ENTITY]),
        "target_visible": combat_column("target_visible"),
        "target_facing_error": combat_column("target_facing_error"),
        "target_head_facing_error": combat_column("target_head_facing_error"),
        "target_head_pitch": combat_column("target_head_pitch"),
        "target_attack_progress": combat_column("target_attack_progress"),
        "target_yaw_degrees": scalar(combat.yaw[:, TARGET_ENTITY]),
        "target_head_yaw_degrees": scalar(combat.target_head_yaw),
        "target_head_pitch_degrees": scalar(combat.target_head_pitch),
        "target_perceptible": float(bool(scalar(evidence.perceptible))),
        "target_attack_phase": scalar(phase),
        "target_attack_progress_raw": scalar(progress),
        "target_reported_attack_index": scalar(reported_index),
        "agent_health_fraction": scalar(
            combat.health[:, AGENT_ENTITY] / params.agent_max_health),
        "agent_attack_executing_bool": float(
            bool(scalar((combat.agent_hit_delay > 0)
                        | (combat.agent_attack_cooldown_seconds > 0.0)))),
        "agent_max_speed": float(params.agent_max_speed),
        "vertical_speed_scale": float(params.vertical_speed_scale),
        "agent_max_health": float(params.agent_max_health),
        "agent_attack_pause_max_seconds": float(
            params.agent_attack_pause_max_seconds),
        "agent_force_per_axis_deadzone": float(
            params.agent_force_per_axis_deadzone),
        "regen_delay_ticks": float(params.regen_delay_ticks),
        "agent_walk_max_fall_speed": float(params.agent_walk_max_fall_speed),
        "loaded_dt": float(params.loaded_dt),
        "wire_fixed_point_scale": float(params.wire_fixed_point_scale),
        "target_chase_speed": float(params.target_chase_speed),
        "target_max_health": float(params.target_max_health),
        "sensor_range": float(params.sensor_range),
        "facing_error_degrees_scale": float(params.facing_error_degrees_scale),
        "head_pitch_degrees_scale": float(params.head_pitch_degrees_scale),
        "target_attack_index_divisor": float(
            max(OBSERVATION_CAPACITY - 1, 1)
        ),
        "movement_value_bits": float(pack_bits(movement_values)),
        "movement_available_bits": float(pack_bits(movement_available)),
        # The environment does not retain the capability object after encoding;
        # these are the lossless policy-facing raw facts. Fresh derivation
        # fixtures separately drive values outside the clip range.
        "actor_controller_medium_available": float(
            bool(np.asarray(o.actor_world_mask)[lane, 0])
        ),
        "actor_controller_in_fluid": float(
            np.asarray(o.actor_world_f32)[lane, 0] > 0.5
        ),
        "actor_submersion_available": float(
            bool(np.asarray(o.actor_world_mask)[lane, 1])
        ),
        "actor_feet_submerged": float(
            np.asarray(o.actor_world_f32)[lane, 1] > 0.5
        ),
        "actor_eyes_submerged": float(
            np.asarray(o.actor_world_f32)[lane, 2] > 0.5
        ),
        "actor_drop_available": float(
            bool(np.asarray(o.actor_world_mask)[lane, 2])
        ),
        "actor_drop_support_found": float(
            np.asarray(o.actor_world_f32)[lane, 3] > 0.5
        ),
        "actor_drop_height": float(
            np.asarray(o.actor_world_f32)[lane, 4]
            * params.agent_walk_max_drop_height
        ),
        "agent_walk_max_drop_height": float(params.agent_walk_max_drop_height),
    }
    (directory / "inputs.txt").write_text(
        "\n".join(f"{key}\t{value!r}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )

    # The other 34 groups, in the same raw form export_groups.py writes: full
    # entity axis, unmasked, so the Java side still does the selection and the
    # mask multiply itself.
    inventory = mask_inventory_policy_tokens(
        inventory_policy_tokens_from_state(
            state.runtime.inventory, runtime.inventory_layout, actor_valid=o.valid,
        ),
        o.valid,
    )
    light = align_actor_light_policy_tokens(
        state.light_tokens,
        o.world_geometry.token_mask,
        o.world_geometry.available,
        o.valid,
    )
    surface = state.action_surface
    block = surface.block_candidates
    recipe = surface.recipe_encoding
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
    signal: dict[str, int] = {}
    for name, value in fields.items():
        array = np.asarray(value)[lane]
        boolean = array.dtype == np.bool_
        flat = np.ascontiguousarray(
            array.astype(np.uint8 if boolean else np.float32))
        (directory / f"{name}.bin").write_bytes(flat.tobytes())
        signal[name] = int(np.count_nonzero(flat))

    raw_fields = {
        "raw_resource_current": state.runtime.mechanics.resources,
        "raw_resource_minimum": runtime.mechanics_rules.resource_minimum,
        "raw_resource_maximum": runtime.mechanics_rules.resource_maximum,
    }
    for name, value in raw_fields.items():
        array = np.ascontiguousarray(
            np.asarray(value)[lane], dtype=np.float32
        )
        (directory / f"{name}.bin").write_bytes(array.tobytes())

    # Preserve the structured policy row, not merely its final embedding. The
    # JVM projection consumes these actor-visible fields and the exact exported
    # encoder weights; native recipe identity and execution diagnostics remain
    # intentionally absent.
    from tools.export.recipe_candidates import write_recipe_policy_view

    write_recipe_policy_view(directory, surface.recipe_candidates, lane=lane)

    mechanics = state.runtime.mechanics
    rules = runtime.mechanics_rules
    raw_defense = np.concatenate(
        (
            np.asarray(mechanics.guard_active)[lane].astype(np.float32),
            np.asarray(mechanics.stamina_broken)[lane].astype(np.float32),
            np.asarray(mechanics.dodge_invulnerability_remaining_seconds)[lane],
            np.asarray(rules.dodge_invulnerability_seconds)[lane],
            np.asarray(mechanics.applied_velocity)[lane].reshape(-1),
            np.asarray(rules.dodge_force)[lane],
            np.asarray(mechanics.stamina_regen_delay_seconds)[lane],
            np.asarray(mechanics.control_immunity)[lane],
            np.asarray(state.runtime.combat.health)[lane],
        )
    ).astype(np.float32)
    if raw_defense.shape != (22,):
        raise SystemExit(f"raw defense width drift: {raw_defense.shape}")
    (directory / "raw_defense.bin").write_bytes(raw_defense.tobytes())

    statuses = mechanics.statuses
    raw_status = np.concatenate(
        (
            np.asarray(
                (params.agent_max_health, params.target_max_health),
                dtype=np.float32,
            ),
            (
                np.asarray(rules.resource_maximum)[lane]
                - np.asarray(rules.resource_minimum)[lane]
            ).reshape(-1),
            np.asarray(statuses.remaining_seconds)[lane].reshape(-1),
            np.asarray(statuses.cycle_elapsed_seconds)[lane].reshape(-1),
            np.asarray(statuses.cycle_cooldown_seconds)[lane].reshape(-1),
            np.asarray(statuses.damage_per_cycle)[lane].reshape(-1),
            np.asarray(statuses.healing_per_cycle)[lane].reshape(-1),
            np.asarray(statuses.resource_id)[lane].astype(np.float32).reshape(-1),
            np.asarray(statuses.resource_delta_per_cycle)[lane].reshape(-1),
            np.asarray(statuses.speed_multiplier)[lane].reshape(-1),
            np.asarray(statuses.active)[lane].astype(np.float32).reshape(-1),
        )
    ).astype(np.float32)
    if raw_status.shape != (160,):
        raise SystemExit(f"raw status width drift: {raw_status.shape}")
    (directory / "raw_status.bin").write_bytes(raw_status.tobytes())

    # The PPO wrapper retains the projected dodge mask, not the world
    # capability row that produced it. Use a policy-equivalent corridor here;
    # ProjectionTest independently drives the real JAX projector through
    # corridor, movement-disable, death, and stamina-threshold cases.
    dodge_mask = np.asarray(o.dodge_action_mask)[lane].astype(np.float32)
    raw_dodge = np.asarray(
        (
            1.0,
            1.0,
            float(dodge_mask[2]),
            float(dodge_mask[3]),
            1.0,
            1.0,
            2.0,
            2.0,
        ),
        dtype=np.float32,
    )
    (directory / "raw_dodge.bin").write_bytes(raw_dodge.tobytes())

    observable_arsenal = ability_lifecycle_legality_view(
        state.runtime.arsenal, runtime.loadout
    )
    loadout = runtime.loadout
    # Authored loadouts are compact (currently six slots for this matchup),
    # while learner-v3 has a checkpoint-pinned 16-slot observation capacity.
    # The encoder pads at that boundary. Raw export must do the same before
    # flattening; concatenating the compact arrays produced 368 values and made
    # a fresh production profile impossible to export even though the finished
    # learner row was correctly 16-wide.
    observation_capacity = int(np.asarray(o.ability_mask).shape[-1])

    def padded_ability_slots(value, fill=0):
        array = np.asarray(value)[lane]
        if array.ndim < 2:
            raise SystemExit(
                f"ability metadata lost entity/slot axes: {array.shape}"
            )
        authored_capacity = int(array.shape[1])
        if authored_capacity > observation_capacity:
            raise SystemExit(
                "authored ability capacity exceeds learner observation: "
                f"{authored_capacity} > {observation_capacity}"
            )
        padding = [(0, 0), (0, observation_capacity - authored_capacity)]
        padding.extend((0, 0) for _ in range(array.ndim - 2))
        return np.pad(array, padding, constant_values=fill)

    raw_ability = np.concatenate(
        (
            np.asarray(mechanics.resources)[lane].reshape(-1),
            (
                np.asarray(rules.resource_maximum)[lane]
                - np.asarray(rules.resource_minimum)[lane]
            ).reshape(-1),
            padded_ability_slots(
                loadout.ability_duration_seconds
            ).reshape(-1),
            padded_ability_slots(
                loadout.ability_cooldown_seconds
            ).reshape(-1),
            padded_ability_slots(
                state.runtime.arsenal.ability_cooldown_seconds
            ).reshape(-1),
            np.asarray(observable_arsenal.active_ability_slot)[lane]
                .astype(np.float32)
                .reshape(-1),
            np.asarray(state.runtime.arsenal.ability_elapsed_seconds)[lane]
                .reshape(-1),
            padded_ability_slots(loadout.ability_resource_cost).reshape(-1),
            padded_ability_slots(loadout.ability_resource_cost_kind)
                .astype(np.float32).reshape(-1),
            padded_ability_slots(
                loadout.ability_resource_minimum
            ).reshape(-1),
            padded_ability_slots(loadout.ability_mask)
                .astype(np.float32)
                .reshape(-1),
            np.asarray(loadout.equipped)[lane].astype(np.float32).reshape(-1),
            np.asarray(loadout.overflow)[lane].astype(np.float32).reshape(-1),
            np.asarray(o.ability_legal)[lane].astype(np.float32).reshape(-1),
        )
    ).astype(np.float32)
    if raw_ability.shape != (868,):
        raise SystemExit(f"raw ability width drift: {raw_ability.shape}")
    (directory / "raw_ability.bin").write_bytes(raw_ability.tobytes())

    inventory_flags, inventory_i32, inventory_f32 = (
        _policy_equivalent_raw_inventory(
            np.asarray(inventory.available)[lane : lane + 1],
            np.asarray(inventory.container_f32)[lane],
            np.asarray(inventory.container_mask)[lane],
            np.asarray(inventory.token_f32)[lane],
            np.asarray(inventory.token_mask)[lane],
        )
    )
    (directory / "raw_inventory_flags.bin").write_bytes(
        inventory_flags.tobytes()
    )
    (directory / "raw_inventory_i32.bin").write_bytes(
        inventory_i32.tobytes()
    )
    (directory / "raw_inventory_f32.bin").write_bytes(
        inventory_f32.tobytes()
    )

    reference = np.asarray(observation)[lane]
    np.ascontiguousarray(reference, dtype=np.float32).tofile(
        directory / "expected_observation.bin")

    # This state's own legality, not the one baked into `case/`. Using a
    # fixed mask would still compare Java against JAX fairly -- both sides see
    # it -- but the decisions would be the ones a differently-equipped actor
    # would take, and weapons whose abilities are illegal here would look
    # interchangeable.
    # `recipe_candidates` is the view; `recipe_encoding` next to it is the
    # checkpoint-stable embedding and is rejected here. `use_available` and
    # `block_trigger_available` default to *closed* when omitted, which would
    # quietly forbid the use and block-trigger heads in every state.
    mask = np.asarray(
        arsenal_policy_action_mask(
            state.learner_observation,
            surface.block_candidates,
            surface.recipe_candidates,
            use_available=surface.use_available,
            block_trigger_available=surface.block_trigger_available,
        )
    )[lane]
    # Derived, not a literal: this was pinned at 99 and went stale when the
    # action surface grew to 124, which made the exporter reject a correct
    # current capture in the voice of a Java-side bug.
    from hytalegym.jax.combat.observation.v3 import (
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
    )

    expected_actions = int(sum(ARSENAL_POLICY_ACTION_HEAD_SIZES))
    if mask.shape != (expected_actions,):
        raise SystemExit(
            f"action mask is {mask.shape}, expected ({expected_actions},) -- "
            "the Java side would read a differently-sized legality and "
            "mis-gate every head"
        )
    np.ascontiguousarray(mask.astype(np.uint8)).tofile(
        directory / "action_mask.bin")

    speed = float(np.hypot(agent_velocity[0], agent_velocity[2]))
    print(f"  t{tick:<4d} {profile:16s} speed {speed:6.3f} "
          f"yaw {values['agent_yaw_degrees']:+8.2f} "
          f"phase {learner_target_phase} "
          f"vis {values['target_visible']:.0f} "
          f"hp {values['agent_health_fraction']:.2f} "
          f"dist {float(np.linalg.norm(target_position - agent_position)):6.2f}")
    return {
        "tick": tick,
        "profile": profile,
        "directory": directory.name,
        "agent_speed": speed,
        "agent_yaw_degrees": values["agent_yaw_degrees"],
        "agent_health_fraction": values["agent_health_fraction"],
        "target_attack_phase": learner_target_phase,
        "target_visible": values["target_visible"],
        # A content hash, not a non-zero count: two different observations can
        # easily share a non-zero count, so counting would report breadth that
        # is not there.
        "observation_digest": hashlib.sha256(
            np.ascontiguousarray(reference, dtype=np.float32).tobytes()
        ).hexdigest()[:16],
        "signal": signal,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Transactionally regenerate the flat and Region JVM parity "
            "fixtures from the live Gym."
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=(
            "flat",
            "region",
            "augment-structured",
            "augment-movement",
            "migrate-attack-slot-divisor",
        ),
        default="flat",
        help="fixture mode or one explicit migration operation (default: flat)",
    )
    parser.parse_args()
    sys.exit(main())
