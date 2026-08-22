"""JAX lookup over exact native spawn-marker metadata."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.world.surrogate.types import (
    SurrogateSpawnMarkerCatalog,
    SurrogateSpawnMarkerSelection,
)
from hytalegym.worldgen.surrogate.spawn_markers import (
    CompiledSpawnMarkerCatalog,
    DEFAULT_SPAWN_MARKER_CATALOG_CAPACITY,
    DEFAULT_SPAWN_MARKER_CHOICE_CAPACITY,
)

_NATIVE_WEIGHT_MAX_SAMPLE = 0.99999
_NATIVE_WEIGHT_EPSILON = 9.99999999995449e-6


def spawn_marker_catalog_to_jax(
    catalog: CompiledSpawnMarkerCatalog,
) -> SurrogateSpawnMarkerCatalog:
    """Transfer validated host arrays without publishing provenance strings."""

    if not isinstance(catalog, CompiledSpawnMarkerCatalog):
        raise TypeError("catalog must be a CompiledSpawnMarkerCatalog")
    return SurrogateSpawnMarkerCatalog(
        marker_mask=jnp.asarray(catalog.marker_mask),
        marker_identity_words=jnp.asarray(catalog.marker_identity_words),
        asset_sha256_words=jnp.asarray(catalog.asset_sha256_words),
        realtime_respawn=jnp.asarray(catalog.realtime_respawn),
        manual_trigger=jnp.asarray(catalog.manual_trigger),
        exclusion_radius=jnp.asarray(catalog.exclusion_radius),
        maximum_drop_height=jnp.asarray(catalog.maximum_drop_height),
        deactivation_distance=jnp.asarray(catalog.deactivation_distance),
        deactivation_seconds=jnp.asarray(catalog.deactivation_seconds),
        choice_mask=jnp.asarray(catalog.choice_mask),
        role_present=jnp.asarray(catalog.role_present),
        role_identity_words=jnp.asarray(catalog.role_identity_words),
        choice_weight=jnp.asarray(catalog.choice_weight),
        respawn_seconds=jnp.asarray(catalog.respawn_seconds),
        flock_required=jnp.asarray(catalog.flock_required),
    )


def select_spawn_marker_configuration(
    catalog: SurrogateSpawnMarkerCatalog,
    marker_identity_words: jax.Array,
    unit_sample: jax.Array,
) -> SurrogateSpawnMarkerSelection:
    """Select weighted metadata; this does not create or tick an NPC."""

    _catalog_shape(catalog)
    identities = jnp.asarray(marker_identity_words)
    samples = jnp.asarray(unit_sample)
    if identities.ndim < 1 or identities.shape[-1] != 2:
        raise ValueError("marker_identity_words must end in dimension 2")
    if samples.shape != identities.shape[:-1]:
        raise ValueError("unit_sample must match marker identity leading dimensions")

    result_shape = samples.shape
    flat_identities = identities.reshape((-1, 2))
    flat_samples = samples.reshape((-1,))
    matches = (
        jnp.all(
            flat_identities[:, None, :]
            == catalog.marker_identity_words[None, :, :],
            axis=2,
        )
        & catalog.marker_mask[None, :]
    )
    match_count = jnp.sum(matches, axis=1)
    marker_index = jnp.argmax(matches, axis=1)
    row_mask = catalog.choice_mask[marker_index]
    row_weights = jnp.where(
        row_mask,
        catalog.choice_weight[marker_index],
        0.0,
    )
    total_weight = jnp.sum(row_weights, axis=1)
    sample_valid = (
        jnp.isfinite(flat_samples)
        & (flat_samples >= 0.0)
        & (flat_samples < 1.0)
    )
    threshold = (
        jnp.minimum(flat_samples, _NATIVE_WEIGHT_MAX_SAMPLE) * total_weight
    )
    cumulative = jnp.cumsum(row_weights, axis=1)
    eligible = (
        row_mask
        & (threshold[:, None] - cumulative <= _NATIVE_WEIGHT_EPSILON)
    )
    choice_index = jnp.argmax(eligible, axis=1)
    chosen = jnp.arange(flat_samples.size)
    selected = eligible[chosen, choice_index]
    available = (
        (match_count == 1)
        & sample_valid
        & (total_weight > 0.0)
        & selected
    )
    selected_role = catalog.role_present[marker_index, choice_index]
    selected_identity = catalog.role_identity_words[
        marker_index,
        choice_index,
    ]
    selected_respawn = catalog.respawn_seconds[marker_index, choice_index]
    selected_flock = catalog.flock_required[marker_index, choice_index]

    return SurrogateSpawnMarkerSelection(
        available=available.reshape(result_shape),
        choice_index=jnp.where(available, choice_index, -1).reshape(result_shape),
        role_present=(available & selected_role).reshape(result_shape),
        role_identity_words=jnp.where(
            available[:, None] & selected_role[:, None],
            selected_identity,
            jnp.uint32(0),
        ).reshape(result_shape + (2,)),
        respawn_seconds=jnp.where(
            available,
            selected_respawn,
            0.0,
        ).reshape(result_shape),
        realtime_respawn=jnp.where(
            available,
            catalog.realtime_respawn[marker_index],
            False,
        ).reshape(result_shape),
        manual_trigger=jnp.where(
            available,
            catalog.manual_trigger[marker_index],
            False,
        ).reshape(result_shape),
        flock_required=(available & selected_flock).reshape(result_shape),
        behavior_supported=jnp.zeros(result_shape, dtype=jnp.bool_),
    )


def spawn_marker_selection_contract() -> dict[str, object]:
    """Return the narrow certification boundary for artifact manifests."""

    return {
        "version": 1,
        "asset_metadata": "native_0_5_7_hash_locked",
        "identity": "hytalerl_entity_type_v1",
        "capacity": "compile_time_shape",
        "default_marker_capacity": DEFAULT_SPAWN_MARKER_CATALOG_CAPACITY,
        "default_choice_capacity": DEFAULT_SPAWN_MARKER_CHOICE_CAPACITY,
        "weighted_selection": "native_algorithm_float32",
        "selection_sample": "caller_supplied_unit_interval",
        "noop_choices_supported": True,
        "native_rng_equivalent": False,
        "artifact_bound": True,
        "spawn_placement_supported": False,
        "flock_expansion_supported": False,
        "respawn_lifecycle_supported": False,
        "behavior_supported": False,
    }


def _catalog_shape(catalog: SurrogateSpawnMarkerCatalog) -> tuple[int, int]:
    if not isinstance(catalog, SurrogateSpawnMarkerCatalog):
        raise TypeError("catalog must be a SurrogateSpawnMarkerCatalog")
    if catalog.marker_mask.ndim != 1:
        raise ValueError("spawn marker marker_mask must have shape [M]")
    marker_capacity = catalog.marker_mask.shape[0]
    if catalog.choice_mask.ndim != 2:
        raise ValueError("spawn marker choice_mask must have shape [M, R]")
    choice_capacity = catalog.choice_mask.shape[1]
    marker_shapes = (
        catalog.realtime_respawn,
        catalog.manual_trigger,
        catalog.exclusion_radius,
        catalog.maximum_drop_height,
        catalog.deactivation_distance,
        catalog.deactivation_seconds,
    )
    choice_shapes = (
        catalog.role_present,
        catalog.choice_weight,
        catalog.respawn_seconds,
        catalog.flock_required,
    )
    if (
        marker_capacity <= 0
        or choice_capacity <= 0
        or catalog.marker_identity_words.shape != (marker_capacity, 2)
        or catalog.asset_sha256_words.shape != (marker_capacity, 8)
        or any(value.shape != (marker_capacity,) for value in marker_shapes)
        or any(
            value.shape != (marker_capacity, choice_capacity)
            for value in choice_shapes
        )
        or catalog.role_identity_words.shape
        != (marker_capacity, choice_capacity, 2)
    ):
        raise ValueError("spawn marker catalog arrays disagree")
    return marker_capacity, choice_capacity


__all__ = [
    "select_spawn_marker_configuration",
    "spawn_marker_catalog_to_jax",
    "spawn_marker_selection_contract",
]
