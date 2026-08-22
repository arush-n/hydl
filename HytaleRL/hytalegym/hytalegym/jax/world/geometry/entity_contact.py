"""Native-shaped swept-AABB contact against neutral entity scenes."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world.entities.privileged import PrivilegedEntityScene


Array = jax.Array
ENTITY_FIRST_CONTACT_SCHEMA = "hytalerl_entity_first_contact_v1"
ENTITY_FIRST_CONTACT_VERSION = 1

_DIRECTION_EPSILON = jnp.float32(1.0e-5)
_TIE_EPSILON = jnp.float32(1.0e-6)


class EntityFirstContactResult(NamedTuple):
    """Nearest physical entity contact with explicit fail-closed state."""

    available: Array
    hit: Array
    hit_fraction: Array
    contact_point: Array
    entity_slot: Array
    entity_identity_words: Array
    admitted_entity_mask: Array
    ambiguous_tie: Array
    invalid: Array


def entity_swept_aabb_first_contact_result(
    scene: PrivilegedEntityScene,
    position: Array,
    displacement: Array,
    projectile_bounds: Array,
    query_mask: Array,
    candidate_mask: Array,
) -> EntityFirstContactResult:
    """Return the nearest native-eligible entity along each projectile sweep.

    ``candidate_mask`` owns scenario exclusions such as self, creator, team,
    projectile entities, and dead entities. World intersects only candidates
    that are active, physical, and collidable. Exact-distance ties fail closed
    because Hytale's two spatial structures do not publish a stable tie order.
    """

    if not isinstance(scene, PrivilegedEntityScene):
        raise TypeError("scene must be a PrivilegedEntityScene")
    points = jnp.asarray(position, dtype=jnp.float32)
    delta = jnp.asarray(displacement, dtype=jnp.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("position must have shape [batch, queries, 3]")
    if delta.shape != points.shape:
        raise ValueError("displacement must match position")
    batch, queries, _ = points.shape
    entities = scene.entity_mask.shape[1]
    if scene.entity_mask.shape[0] != batch:
        raise ValueError("scene batch must match query batch")

    bounds = _broadcast_bounds(projectile_bounds, batch, queries)
    requested = _array(query_mask, (batch, queries), jnp.bool_, "query_mask")
    requested_candidates = _array(
        candidate_mask,
        (batch, queries, entities),
        jnp.bool_,
        "candidate_mask",
    )
    scene_shape = (batch, entities)
    for name, value, tail in (
        ("active", scene.active, ()),
        ("collidable", scene.collidable, ()),
        ("physical_mask", scene.physical_mask, ()),
        ("position", scene.position, (3,)),
        ("local_bounds", scene.local_bounds, (6,)),
        ("identity_words", scene.identity_words, (2,)),
    ):
        if value.shape != scene_shape + tail:
            raise ValueError(f"scene {name} has an invalid shape")

    finite_query = jnp.all(
        jnp.isfinite(points) & jnp.isfinite(delta), axis=2
    ) & jnp.all(jnp.isfinite(bounds), axis=2)
    positive_projectile_bounds = jnp.all(
        bounds[..., :3] < bounds[..., 3:],
        axis=2,
    )
    moving = jnp.any(jnp.abs(delta) >= _DIRECTION_EPSILON, axis=2)
    scene_ready = scene.available & scene.physical_available & ~scene.capacity_exceeded
    invalid = requested & (
        ~finite_query | ~positive_projectile_bounds | ~moving | ~scene_ready[:, None]
    )
    query_valid = requested & ~invalid

    target_bounds = scene.local_bounds
    target_finite = jnp.all(jnp.isfinite(scene.position), axis=2) & jnp.all(
        jnp.isfinite(target_bounds), axis=2
    )
    target_positive = jnp.all(
        target_bounds[..., :3] < target_bounds[..., 3:],
        axis=2,
    )
    physical = (
        scene.entity_mask
        & scene.physical_mask
        & scene.active
        & scene.collidable
        & target_finite
        & target_positive
    )
    admitted = requested_candidates & physical[:, None, :]

    # CollisionMath.intersectSweptAABBs expands the stationary target by the
    # moving box (Minkowski sum), then intersects the projectile reference
    # position against that expanded AABB.
    expanded_minimum = (
        scene.position[:, None, :, :]
        + target_bounds[:, None, :, :3]
        - bounds[:, :, None, 3:]
    )
    expanded_maximum = (
        scene.position[:, None, :, :]
        + target_bounds[:, None, :, 3:]
        - bounds[:, :, None, :3]
    )
    ray_position = points[:, :, None, :]
    ray_delta = delta[:, :, None, :]
    nonzero = jnp.abs(ray_delta) >= _DIRECTION_EPSILON
    safe_delta = jnp.where(nonzero, ray_delta, 1.0)
    first = (expanded_minimum - ray_position) / safe_delta
    second = (expanded_maximum - ray_position) / safe_delta
    axis_entry = jnp.where(
        nonzero,
        jnp.minimum(first, second),
        jnp.where(
            (ray_position >= expanded_minimum) & (ray_position <= expanded_maximum),
            -jnp.inf,
            jnp.inf,
        ),
    )
    axis_exit = jnp.where(
        nonzero,
        jnp.maximum(first, second),
        jnp.where(
            (ray_position >= expanded_minimum) & (ray_position <= expanded_maximum),
            jnp.inf,
            -jnp.inf,
        ),
    )
    entry = jnp.maximum(jnp.max(axis_entry, axis=3), 0.0)
    exit = jnp.min(axis_exit, axis=3)
    intersects = admitted & query_valid[:, :, None] & (entry <= exit) & (entry <= 1.0)
    fractions = jnp.where(intersects, entry, jnp.inf)
    nearest = jnp.min(fractions, axis=2)
    has_candidate = jnp.isfinite(nearest)
    tied = intersects & (jnp.abs(fractions - nearest[:, :, None]) <= _TIE_EPSILON)
    ambiguous = query_valid & (jnp.sum(tied, axis=2) > 1)
    selected = jnp.argmin(fractions, axis=2).astype(jnp.int32)
    hit = query_valid & has_candidate & ~ambiguous
    fraction = jnp.where(hit, nearest, jnp.float32(1.0))

    selected_identity = jnp.take_along_axis(
        scene.identity_words[:, None, :, :],
        selected[:, :, None, None],
        axis=2,
    )[:, :, 0, :]
    contact = points + delta * fraction[..., None]
    return EntityFirstContactResult(
        available=query_valid & ~ambiguous,
        hit=hit,
        hit_fraction=fraction,
        contact_point=jnp.where(hit[..., None], contact, 0.0),
        entity_slot=jnp.where(hit, selected, jnp.int32(-1)),
        entity_identity_words=jnp.where(
            hit[..., None],
            selected_identity,
            jnp.uint32(0),
        ),
        admitted_entity_mask=admitted & query_valid[:, :, None],
        ambiguous_tie=ambiguous,
        invalid=invalid,
    )


def entity_first_contact_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable physical entity-contact contract."""

    return {
        "schema": ENTITY_FIRST_CONTACT_SCHEMA,
        "version": ENTITY_FIRST_CONTACT_VERSION,
        "producer": "entity_swept_aabb_first_contact_result",
        "source": [
            "EntityCollisionProvider.computeNearest",
            "EntityCollisionProvider.defaultEntityFilter",
            "CollisionMath.intersectSweptAABBs",
        ],
        "shape": {
            "position": ["batch", "queries", 3],
            "displacement": ["batch", "queries", 3],
            "projectile_bounds": ["batch_or_one", "queries_or_one", 6],
            "query_mask": ["batch", "queries"],
            "candidate_mask": ["batch", "queries", "entities"],
        },
        "eligibility": {
            "world": "active_and_physical_and_collidable",
            "consumer": [
                "exclude_self",
                "exclude_creator",
                "exclude_projectile_entities",
                "exclude_death_component",
                "scenario_relationship_policy",
            ],
            "missing_physical_profile": "fail_complete_environment_closed",
        },
        "intersection": {
            "method": "target_AABB_minkowski_sum_projectile_AABB_then_ray",
            "stationary_axis_epsilon": 1.0e-5,
            "fraction": "projectile_reference_position_in_[0,1]",
            "overlap_at_start": 0.0,
        },
        "ordering": {
            "nearest": "strictly_smallest_fraction",
            "equal_fraction": (
                "unpublished_native_spatial_iteration_order_fail_closed"
            ),
        },
        "provenance": "native_shaped_exact_AABB_math",
        "certification": "randomized_eager_and_jit_permutation_controls",
        "not_modelled": [
            "block_contact",
            "fluid_contact",
            "bounce_or_slide_response",
            "impact_chain_dispatch",
        ],
    }


def entity_first_contact_contract_sha256() -> str:
    payload = json.dumps(
        entity_first_contact_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _broadcast_bounds(value: Array, batch: int, queries: int) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.shape == (6,):
        result = result[None, None, :]
    elif result.ndim == 2 and result.shape[1] == 6:
        result = result[:, None, :]
    if result.ndim != 3 or result.shape[2] != 6:
        raise ValueError(
            "projectile_bounds must have shape [6], [batch|1, 6], or "
            "[batch|1, queries|1, 6]"
        )
    if result.shape[0] not in (1, batch) or result.shape[1] not in (1, queries):
        raise ValueError("projectile_bounds cannot broadcast to query shape")
    return jnp.broadcast_to(result, (batch, queries, 6))


def _array(value: Array, shape: tuple[int, ...], dtype, label: str) -> Array:
    result = jnp.asarray(value)
    if result.dtype != dtype:
        raise TypeError(f"{label} must use dtype {dtype}")
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    return result


__all__ = [
    "ENTITY_FIRST_CONTACT_SCHEMA",
    "ENTITY_FIRST_CONTACT_VERSION",
    "EntityFirstContactResult",
    "entity_first_contact_contract",
    "entity_first_contact_contract_sha256",
    "entity_swept_aabb_first_contact_result",
]
