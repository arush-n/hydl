"""Keyed exact-Region combat resets with an explicit navigability filter."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.geometry.contract import FLAG_OPAQUE
from hytalegym.jax.combat.env import (
    reset_batch_geometry_at,
    reset_batch_region_at,
    step_batch_geometry,
    step_batch_region,
)
from hytalegym.jax.combat.types import ACTION_SIZE, CombatParams
from hytalegym.jax.training.arsenal import ArsenalResetBatch
from hytalegym.jax.world import (
    REGION_TRAVERSAL_POLICY_STAGING_CAPACITY,
    WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY,
    GeometryProvider,
    GeometryState,
    RegionGeometryState,
    region_aabb_core_available,
    world_geometry_token_contract_sha256,
)
from hytalegym.rulesets import combat_ruleset_sha256, load_combat_ruleset


REGION_RESET_ACTOR_FORWARD = (0.0, 0.0, -1.0)
REGION_COMBAT_RESET_CORE_ADMISSION_SCHEMA = (
    "hytalerl_region_combat_reset_core_admission_v1"
)


class RegionCombatResetPool(NamedTuple):
    """Padded per-Region sources and directed edges; masks are load-bearing."""

    agent_position: jax.Array
    target_position: jax.Array
    source_mask: jax.Array
    destination_mask: jax.Array
    source_node: jax.Array
    destination_node: jax.Array


class RegionCombatResetSelection(NamedTuple):
    """One keyed reset selection per compiled environment lane."""

    reset: ArsenalResetBatch
    source_slot: jax.Array
    destination_slot: jax.Array
    source_node: jax.Array
    destination_node: jax.Array


class RegionResetCoreAvailability(NamedTuple):
    """Per-lane proof that both reset actors fit inside a Region core."""

    agent: jax.Array
    target: jax.Array
    pair: jax.Array


class RegionCombatResetAdmission(NamedTuple):
    """Lane-local pool masks admitted by exact Region core geometry."""

    source_mask: jax.Array
    destination_mask: jax.Array


RegionResetProvider = Callable[[jax.Array], ArsenalResetBatch]
_POOL_ARRAY_NAMES = (
    "agent_position",
    "target_position",
    "source_mask",
    "destination_mask",
    "source_node",
    "destination_node",
)


def _combat_params_sha256(params: CombatParams) -> str:
    """Hash the exact array-only parameter tree used by the filter."""

    digest = hashlib.sha256()
    for name, leaf in zip(params._fields, params, strict=True):
        value = np.ascontiguousarray(np.asarray(jax.device_get(leaf)))
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest().upper()


def region_reset_filter_combat_contract(
    params: CombatParams,
) -> dict[str, object]:
    """Identify the exact Combat dynamics used to admit Region resets."""

    ruleset = load_combat_ruleset()
    microticks = np.asarray(jax.device_get(params.microticks))
    target_active = np.asarray(jax.device_get(params.target_active))
    if microticks.shape or target_active.shape:
        raise ValueError("reset-filter runtime parameters must be scalar")
    return {
        "hytale_version": ruleset["hytale_server_version"],
        "combat_model_version": ruleset["version"],
        "combat_ruleset_sha256": combat_ruleset_sha256(),
        "combat_params_sha256": _combat_params_sha256(params),
        "microticks": int(microticks),
        "target_active": bool(target_active),
        "filter_action": "all_zero_combat_action_v1",
        "action_size": ACTION_SIZE,
    }


def region_reset_actor_evidence_contract(
    *,
    token_capacity: int,
    maximum_distance: float,
) -> dict[str, object]:
    """Identify the legal actor-evidence surface used before navigation."""

    return {
        "source_contract_sha256": world_geometry_token_contract_sha256(),
        "producer": "produce_actor_region_world_geometry_tokens",
        "acceptance": "available_and_any_token_mask",
        "token_capacity": int(token_capacity),
        "maximum_distance": float(maximum_distance),
        "traversal_staging_capacity": (
            REGION_TRAVERSAL_POLICY_STAGING_CAPACITY
        ),
        "geometry_provenance": (
            WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY
        ),
        "role_opaque_flag": FLAG_OPAQUE,
        "forward": list(REGION_RESET_ACTOR_FORWARD),
    }


def region_combat_reset_core_admission_contract() -> dict[str, object]:
    """Describe the actor-legal exact-core admission used by Region training."""

    return {
        "schema": REGION_COMBAT_RESET_CORE_ADMISSION_SCHEMA,
        "evaluation_stage": "fixture_composition_once",
        "availability_query": "region_aabb_core_available",
        "required_entities": ["agent", "target"],
        "geometry_surface": "immutable_captured_region_core",
        "mutable_overlay": "excluded_from_core_membership",
        "source_admission": (
            "pool_source_and_agent_core_and_any_admitted_destination"
        ),
        "destination_admission": (
            "pool_destination_and_agent_core_and_target_core"
        ),
        "position_policy": "retain_exact_no_clip_no_move",
        "empty_lane_policy": "reject_fixture_composition",
        "reset_runtime": "sample_precomputed_masks_without_geometry_query",
        "geometry_exhaustion_runtime": "unchanged_fail_closed",
    }


def region_combat_reset_core_admission_contract_sha256() -> str:
    """Hash the task-distribution semantics of exact-core reset admission."""

    payload = json.dumps(
        region_combat_reset_core_admission_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def validate_region_combat_reset_pool(pool: RegionCombatResetPool) -> None:
    """Reject malformed or empty pools before tracing a trainer."""

    if pool.source_mask.ndim != 2:
        raise ValueError("source_mask must have shape (world, source)")
    if pool.destination_mask.ndim != 3:
        raise ValueError(
            "destination_mask must have shape (world, source, edge)"
        )
    world_count, source_capacity = pool.source_mask.shape
    if world_count < 1 or source_capacity < 1:
        raise ValueError("reset pool must contain at least one world and source")
    if pool.destination_mask.shape[:2] != (world_count, source_capacity):
        raise ValueError("source and destination masks do not align")
    edge_capacity = pool.destination_mask.shape[2]
    if edge_capacity < 1:
        raise ValueError("reset pool must contain at least one edge slot")
    expected_agent_positions = (world_count, source_capacity, 3)
    expected_target_positions = (
        world_count,
        source_capacity,
        edge_capacity,
        3,
    )
    expected_sources = (world_count, source_capacity)
    expected_destinations = (world_count, source_capacity, edge_capacity)
    if pool.agent_position.shape != expected_agent_positions:
        raise ValueError(
            f"agent_position must have shape {expected_agent_positions}"
        )
    if pool.target_position.shape != expected_target_positions:
        raise ValueError(
            f"target_position must have shape {expected_target_positions}"
        )
    if pool.source_node.shape != expected_sources:
        raise ValueError(f"source_node must have shape {expected_sources}")
    if pool.destination_node.shape != expected_destinations:
        raise ValueError(
            f"destination_node must have shape {expected_destinations}"
        )

    source_mask = np.asarray(
        jax.device_get(pool.source_mask),
        dtype=np.bool_,
    )
    destination_mask = np.asarray(
        jax.device_get(pool.destination_mask),
        dtype=np.bool_,
    )
    if np.any(np.sum(source_mask, axis=1) == 0):
        raise ValueError("every selected Region needs a navigable reset")
    if np.any(source_mask & ~np.any(destination_mask, axis=2)):
        raise ValueError("every active source needs a navigable destination")
    if np.any(destination_mask & ~source_mask[:, :, None]):
        raise ValueError("masked sources cannot expose destination edges")
    agent = np.asarray(jax.device_get(pool.agent_position))
    target = np.asarray(jax.device_get(pool.target_position))
    source = np.asarray(jax.device_get(pool.source_node))
    destination = np.asarray(jax.device_get(pool.destination_node))
    if not np.all(np.isfinite(agent[source_mask])) or not np.all(
        np.isfinite(target[destination_mask])
    ):
        raise ValueError("active reset positions must be finite")
    if np.any(source[source_mask] < 0) or np.any(
        destination[destination_mask] < 0
    ):
        raise ValueError("active reset node IDs must be non-negative")


def region_combat_reset_pool_sha256(
    pool: RegionCombatResetPool,
    semantic_metadata: dict[str, object],
) -> str:
    """Hash task-defining metadata and exact pool arrays."""

    digest = hashlib.sha256(
        json.dumps(
            semantic_metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for leaf in pool:
        value = np.ascontiguousarray(np.asarray(jax.device_get(leaf)))
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest().upper()


def load_region_combat_reset_pool(path: Path) -> RegionCombatResetPool:
    """Load and validate one non-pickled NPZ reset-pool artifact."""

    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        if names != set(_POOL_ARRAY_NAMES):
            raise ValueError(
                "reset pool arrays do not match the exact schema: "
                f"{sorted(names)}"
            )
        pool = RegionCombatResetPool(
            *(jnp.asarray(archive[name]) for name in _POOL_ARRAY_NAMES)
        )
    validate_region_combat_reset_pool(pool)
    return pool


def sample_region_combat_resets(
    keys: jax.Array,
    environment_world_id: jax.Array,
    pool: RegionCombatResetPool,
    *,
    admission: RegionCombatResetAdmission | None = None,
) -> RegionCombatResetSelection:
    """Sample uniformly from active slots for each lane's selected Region."""

    batch = keys.shape[0]
    if environment_world_id.shape != (batch,):
        raise ValueError("environment_world_id must have shape (batch,)")

    world = environment_world_id.astype(jnp.int32)
    if admission is None:
        source_mask = pool.source_mask[world]
        destination_masks = pool.destination_mask[world]
    else:
        expected_source = (batch, pool.source_mask.shape[1])
        expected_destination = (batch,) + pool.destination_mask.shape[1:]
        if admission.source_mask.shape != expected_source:
            raise ValueError(
                f"admission source_mask must have shape {expected_source}"
            )
        if admission.destination_mask.shape != expected_destination:
            raise ValueError(
                "admission destination_mask must have shape "
                f"{expected_destination}"
            )
        source_mask = admission.source_mask
        destination_masks = admission.destination_mask
    source_count = jnp.sum(source_mask, axis=1, dtype=jnp.int32)
    source_key, destination_key = jax.vmap(
        lambda key: jax.random.split(key, 2)
    )(keys).swapaxes(0, 1)
    source_rank = jax.vmap(
        lambda key, count: jax.random.randint(
            key,
            (),
            jnp.int32(0),
            count,
            dtype=jnp.int32,
        )
    )(source_key, source_count)
    source_slot = jnp.argmax(
        jnp.cumsum(source_mask.astype(jnp.int32), axis=1)
        > source_rank[:, None],
        axis=1,
    )
    destination_mask = destination_masks[
        jnp.arange(batch, dtype=jnp.int32),
        source_slot,
    ]
    destination_count = jnp.sum(
        destination_mask,
        axis=1,
        dtype=jnp.int32,
    )
    destination_rank = jax.vmap(
        lambda key, count: jax.random.randint(
            key,
            (),
            jnp.int32(0),
            count,
            dtype=jnp.int32,
        )
    )(destination_key, destination_count)
    destination_slot = jnp.argmax(
        jnp.cumsum(destination_mask.astype(jnp.int32), axis=1)
        > destination_rank[:, None],
        axis=1,
    )
    return RegionCombatResetSelection(
        reset=ArsenalResetBatch(
            agent_position=pool.agent_position[world, source_slot],
            target_position=pool.target_position[
                world,
                source_slot,
                destination_slot,
            ],
        ),
        source_slot=source_slot,
        destination_slot=destination_slot,
        source_node=pool.source_node[world, source_slot],
        destination_node=pool.destination_node[
            world,
            source_slot,
            destination_slot,
        ],
    )


def make_region_combat_reset_provider(
    pool: RegionCombatResetPool,
    environment_world_id: jax.Array,
    *,
    geometry: RegionGeometryState | None = None,
) -> RegionResetProvider:
    """Bind a pool and optional exact-core admission to fixed Region lanes.

    Core admission is computed once when the fixture is composed. Reset calls
    only sample the retained masks, so exact geometry adds no per-reset query
    and the runtime's fail-closed ``geometry_exhausted`` semantics stay intact.
    """

    validate_region_combat_reset_pool(pool)
    world_ids = np.asarray(
        jax.device_get(environment_world_id),
        dtype=np.int64,
    )
    if world_ids.ndim != 1:
        raise ValueError("environment_world_id must have shape (batch,)")
    if np.any(world_ids < 0) or np.any(world_ids >= pool.source_mask.shape[0]):
        raise ValueError("environment_world_id indexes outside the reset pool")
    admission = (
        None
        if geometry is None
        else region_combat_reset_core_admission(
            pool,
            environment_world_id,
            geometry,
        )
    )
    if admission is not None:
        admitted = np.asarray(jax.device_get(admission.source_mask), dtype=np.bool_)
        if np.any(~np.any(admitted, axis=1)):
            failed = np.flatnonzero(~np.any(admitted, axis=1)).tolist()
            raise ValueError(
                "Region reset pool has no exact-core candidate for lanes "
                f"{failed}"
            )

    def provider(keys: jax.Array) -> ArsenalResetBatch:
        return sample_region_combat_resets(
            keys,
            environment_world_id,
            pool,
            admission=admission,
        ).reset

    return provider


def region_combat_reset_core_admission(
    pool: RegionCombatResetPool,
    environment_world_id: jax.Array,
    geometry: RegionGeometryState,
) -> RegionCombatResetAdmission:
    """Filter one reset pool to actor AABBs covered by each Region core.

    The returned masks are lane-local because multiple environment lanes may
    select different Region artifacts even when they share the same pool.
    Positions are never clipped or moved; an unavailable pair is simply not a
    sampling candidate.
    """

    validate_region_combat_reset_pool(pool)
    world = jnp.asarray(environment_world_id, dtype=jnp.int32)
    if world.ndim != 1:
        raise ValueError("environment_world_id must have shape (batch,)")
    batch = world.shape[0]
    if batch < 1:
        raise ValueError("environment_world_id must not be empty")
    if pool.source_mask.shape[0] < 1:
        raise ValueError("reset pool must contain at least one world")
    world_host = np.asarray(jax.device_get(world), dtype=np.int64)
    if np.any(world_host < 0) or np.any(world_host >= pool.source_mask.shape[0]):
        raise ValueError("environment_world_id indexes outside the reset pool")
    if geometry.environment_world_id.shape != (batch,):
        raise ValueError("Region geometry world IDs must match the environment batch")

    # Compute each assigned pool world once, even when many rollout lanes share
    # it.  Core coverage is immutable and actor bounds are reset-pinned, so the
    # result can be indexed back to lanes without repeating the expensive cell
    # lookup for every environment.
    unique_world, representative_lane, lane_to_unique = np.unique(
        world_host,
        return_index=True,
        return_inverse=True,
    )
    representative_lane = representative_lane.astype(np.int32, copy=False)
    geometry_world_host = np.asarray(
        jax.device_get(geometry.environment_world_id),
        dtype=np.int64,
    )
    for pool_world in unique_world:
        lanes = np.flatnonzero(world_host == pool_world)
        if np.unique(geometry_world_host[lanes]).size != 1:
            raise ValueError(
                "lanes sharing one reset-pool world select different Region geometry"
            )
        for name in ("agent_bounds", "target_bounds"):
            value = np.asarray(jax.device_get(getattr(geometry, name)))
            if value.shape[0] == batch and not np.all(value[lanes] == value[lanes[0]]):
                raise ValueError(
                    f"lanes sharing one reset-pool world have different {name}"
                )

    def representative_entity_leaf(value: jax.Array) -> jax.Array:
        if value.shape[0] == 1:
            return value
        if value.shape[0] != batch:
            raise ValueError("Region geometry entity rows must match the batch")
        return value[jnp.asarray(representative_lane, dtype=jnp.int32)]

    representative_geometry = geometry._replace(
        environment_world_id=geometry.environment_world_id[
            jnp.asarray(representative_lane, dtype=jnp.int32)
        ],
        agent_bounds=representative_entity_leaf(geometry.agent_bounds),
        target_bounds=representative_entity_leaf(geometry.target_bounds),
        agent_los_offset=representative_entity_leaf(geometry.agent_los_offset),
        target_los_offset=representative_entity_leaf(geometry.target_los_offset),
        # Mutable overlays do not change whether a coordinate belongs to the
        # immutable captured core. Excluding them keeps this one-time admission
        # independent of episode state.
        mutable_blocks=None,
    )
    unique_world_index = jnp.asarray(unique_world, dtype=jnp.int32)
    agent_position = pool.agent_position[unique_world_index]
    target_position = pool.target_position[unique_world_index]
    unique_count = unique_world_index.shape[0]
    source_capacity = agent_position.shape[1]
    edge_capacity = target_position.shape[2]

    agent_core = jax.vmap(
        lambda position: region_aabb_core_available(
            representative_geometry,
            position,
            representative_geometry.agent_bounds,
        ),
        in_axes=1,
        out_axes=1,
    )(agent_position)
    flattened_target = target_position.reshape(
        (unique_count, source_capacity * edge_capacity, 3)
    )
    target_core = jax.vmap(
        lambda position: region_aabb_core_available(
            representative_geometry,
            position,
            representative_geometry.target_bounds,
        ),
        in_axes=1,
        out_axes=1,
    )(flattened_target).reshape(
        (unique_count, source_capacity, edge_capacity)
    )

    destination_by_world = (
        pool.destination_mask[unique_world_index]
        & agent_core[..., None]
        & target_core
    )
    source_by_world = (
        pool.source_mask[unique_world_index]
        & agent_core
        & jnp.any(destination_by_world, axis=2)
    )
    lane_index = jnp.asarray(lane_to_unique, dtype=jnp.int32)
    return RegionCombatResetAdmission(
        source_mask=source_by_world[lane_index],
        destination_mask=destination_by_world[lane_index],
    )


def region_reset_core_availability(
    reset: ArsenalResetBatch,
    geometry: RegionGeometryState,
) -> RegionResetCoreAvailability:
    """Prove that an injected reset stays on Combat's certified Region core.

    A frozen Region includes a one-chunk evidence halo around its three-by-three
    core.  The halo proves collision-source reach; it is not an actor-legal
    training area.  External reset producers should require ``pair`` before
    invoking the environment.  This function intentionally reports invalid
    lanes instead of clipping or resampling them, preserving the runtime's
    fail-closed ``geometry_exhausted`` semantics.
    """

    if reset.agent_position.ndim != 2 or reset.agent_position.shape[1] != 3:
        raise ValueError("reset agent_position must have shape (batch, 3)")
    if reset.target_position.shape != reset.agent_position.shape:
        raise ValueError(
            "reset target_position must match agent_position shape"
        )
    agent = region_aabb_core_available(
        geometry,
        reset.agent_position,
        geometry.agent_bounds,
    )
    target = region_aabb_core_available(
        geometry,
        reset.target_position,
        geometry.target_bounds,
    )
    return RegionResetCoreAvailability(
        agent=agent,
        target=target,
        pair=agent & target,
    )


def idle_navigation_filter_keys(
    lane_keys: jax.Array,
    *,
    decision_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """Derive reset/step keys from one stable key per control and candidate."""

    if lane_keys.ndim < 2 or lane_keys.shape[1] < 1:
        raise ValueError("lane_keys must have shape (control, candidate, ...)")
    if decision_steps < 1:
        raise ValueError("decision_steps must be positive")

    def split(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = jax.random.split(key, decision_steps + 1)
        return keys[0], keys[1:]

    reset_keys, step_keys = jax.vmap(jax.vmap(split))(lane_keys)
    return reset_keys, jnp.swapaxes(step_keys, 1, 2)


def idle_target_navigation_survival_mask(
    reset_keys: jax.Array,
    step_keys: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider,
    agent_position: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    """Require every idle control to avoid the fail-closed navigation seam."""

    control_count, batch = reset_keys.shape[:2]
    if step_keys.shape[:1] != (control_count,) or step_keys.shape[2] != batch:
        raise ValueError("reset and step key batches do not match")
    if agent_position.shape != (batch, 3):
        raise ValueError("agent_position must have shape (batch, 3)")
    if target_position.shape != (batch, 3):
        raise ValueError("target_position must have shape (batch, 3)")

    actions = jnp.zeros((batch, ACTION_SIZE), dtype=jnp.float32)

    if isinstance(geometry, GeometryState):
        reset_at = reset_batch_geometry_at
        step = step_batch_geometry
    elif isinstance(geometry, RegionGeometryState):
        reset_at = reset_batch_region_at
        step = step_batch_region
    else:
        raise TypeError("unsupported exact geometry provider")

    def run_control(keys: tuple[jax.Array, jax.Array]) -> jax.Array:
        one_reset_keys, one_step_keys = keys
        state, _ = reset_at(
            one_reset_keys,
            params,
            geometry,
            agent_position,
            target_position,
        )
        initial_valid = state.agent_grounded & ~state.geometry_exhausted

        def advance(current, keys_for_step):
            next_state, _, _, _, _ = step(
                current,
                actions,
                keys_for_step,
                params,
                geometry,
            )
            valid = ~(
                next_state.target_navigation_unsupported
                | next_state.geometry_exhausted
            )
            return next_state, valid

        _, valid_trace = jax.lax.scan(advance, state, one_step_keys)
        return initial_valid & jnp.all(valid_trace, axis=0)

    controls = jax.lax.map(run_control, (reset_keys, step_keys))
    return jnp.all(controls, axis=0)


__all__ = [
    "REGION_COMBAT_RESET_CORE_ADMISSION_SCHEMA",
    "REGION_RESET_ACTOR_FORWARD",
    "RegionCombatResetPool",
    "RegionCombatResetAdmission",
    "RegionCombatResetSelection",
    "RegionResetCoreAvailability",
    "RegionResetProvider",
    "idle_navigation_filter_keys",
    "idle_target_navigation_survival_mask",
    "load_region_combat_reset_pool",
    "make_region_combat_reset_provider",
    "region_reset_actor_evidence_contract",
    "region_combat_reset_pool_sha256",
    "region_combat_reset_core_admission",
    "region_combat_reset_core_admission_contract",
    "region_combat_reset_core_admission_contract_sha256",
    "region_reset_filter_combat_contract",
    "region_reset_core_availability",
    "sample_region_combat_resets",
    "validate_region_combat_reset_pool",
]
