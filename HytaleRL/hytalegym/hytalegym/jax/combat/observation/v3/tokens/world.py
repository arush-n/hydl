"""Actor-safe policy encoding for the public World geometry-token contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import MAX_DETAIL_BOXES
from hytalegym.jax.world import (
    WORLD_TOKEN_KIND_COLLISION_EXCEPTION,
    WORLD_TOKEN_PROVENANCE_SURROGATE_TRAVERSAL,
    WorldGeometryTokenObservation,
    world_geometry_token_contract,
    world_geometry_token_contract_sha256,
)


Array = jax.Array

WORLD_GEOMETRY_TOKEN_CAPACITY = 44
WORLD_TOKEN_DEFAULT_EDGE_CAPACITY = 12
WORLD_TOKEN_DEFAULT_MAXIMUM_DISTANCE = 4.0
WORLD_TOKEN_BASE_FLOAT_SIZE = 8
WORLD_TOKEN_EDGE_FLOAT_SIZE = 5
WORLD_TOKEN_COLLISION_BOX_FLOAT_SIZE = 7
WORLD_GEOMETRY_SHARED_VISUAL_SCHEMA = "hytalerl_world_geometry_shared_visual_v1"
WORLD_GEOMETRY_SHARED_VISUAL_FEATURE_SIZE = WORLD_TOKEN_BASE_FLOAT_SIZE

WORLD_GEOMETRY_SOURCE_POLICY_FIELDS = (
    "available",
    "token_mask",
    "token_kind",
    "token_provenance",
    "relative_position",
    "clearance",
    "semantic_flags",
    "dynamic_blocked",
    "edge_mask",
    "edge_destination",
    "edge_cost",
    "edge_kind",
    "edge_flags",
    "collision_box_mask",
    "collision_boxes_relative",
)

if tuple(world_geometry_token_contract()["policy_fields"]) != (
    WORLD_GEOMETRY_SOURCE_POLICY_FIELDS
):
    raise RuntimeError("World geometry-token policy field contract drift")


@dataclass(frozen=True)
class WorldGeometryPolicyConfig:
    """Static learner-side capacities for one compiled policy."""

    token_capacity: int = WORLD_GEOMETRY_TOKEN_CAPACITY
    edge_capacity: int = WORLD_TOKEN_DEFAULT_EDGE_CAPACITY
    maximum_distance: float = WORLD_TOKEN_DEFAULT_MAXIMUM_DISTANCE

    def __post_init__(self) -> None:
        if (
            isinstance(self.token_capacity, bool)
            or not isinstance(self.token_capacity, int)
            or self.token_capacity < 1
        ):
            raise ValueError("token_capacity must be a positive integer")
        if (
            isinstance(self.edge_capacity, bool)
            or not isinstance(self.edge_capacity, int)
            or self.edge_capacity < 1
        ):
            raise ValueError("edge_capacity must be a positive integer")
        if (
            isinstance(self.maximum_distance, bool)
            or not isinstance(self.maximum_distance, (int, float))
            or not 0.0 < float(self.maximum_distance)
        ):
            raise ValueError("maximum_distance must be positive")

    @property
    def feature_size(self) -> int:
        return world_geometry_policy_feature_size(self.edge_capacity)


DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG = WorldGeometryPolicyConfig()


class WorldGeometryPolicyTokens(NamedTuple):
    """Policy-only World token rows; producer diagnostics cannot enter this type."""

    available: Array
    token_f32: Array
    token_mask: Array


def normalize_world_geometry_policy_config(
    value: WorldGeometryPolicyConfig | Mapping[str, object] | None,
) -> WorldGeometryPolicyConfig:
    """Return a validated static token configuration."""

    if value is None:
        return DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG
    if isinstance(value, WorldGeometryPolicyConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("world geometry policy config must be a mapping or config")
    allowed = {
        "token_capacity",
        "edge_capacity",
        "maximum_distance",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            "unknown world geometry policy config fields: " + ", ".join(sorted(unknown))
        )
    return WorldGeometryPolicyConfig(
        token_capacity=value.get(
            "token_capacity",
            WORLD_GEOMETRY_TOKEN_CAPACITY,
        ),
        edge_capacity=value.get(
            "edge_capacity",
            WORLD_TOKEN_DEFAULT_EDGE_CAPACITY,
        ),
        maximum_distance=value.get(
            "maximum_distance",
            WORLD_TOKEN_DEFAULT_MAXIMUM_DISTANCE,
        ),
    )


def world_geometry_policy_config_manifest(
    config: WorldGeometryPolicyConfig | Mapping[str, object] | None = None,
) -> dict[str, int | float | str]:
    """Return checkpoint-safe static token encoder identity."""

    value = normalize_world_geometry_policy_config(config)
    return {
        "token_capacity": value.token_capacity,
        "edge_capacity": value.edge_capacity,
        "detail_box_capacity": MAX_DETAIL_BOXES,
        "feature_size": value.feature_size,
        "maximum_distance": float(value.maximum_distance),
        "capacity_evidence": (
            "artifacts/worldgen/world-token-capacity-v3.json:first_zero_overflow"
        ),
        "traversal_provenance": "surrogate",
        "source_contract_sha256": world_geometry_token_contract_sha256(),
    }


def world_geometry_policy_feature_size(edge_capacity: int) -> int:
    """Return encoded floats per token for a static traversal edge capacity."""

    if (
        isinstance(edge_capacity, bool)
        or not isinstance(edge_capacity, int)
        or edge_capacity < 1
    ):
        raise ValueError("edge_capacity must be a positive integer")
    return (
        WORLD_TOKEN_BASE_FLOAT_SIZE
        + edge_capacity * WORLD_TOKEN_EDGE_FLOAT_SIZE
        + MAX_DETAIL_BOXES * WORLD_TOKEN_COLLISION_BOX_FLOAT_SIZE
    )


def world_geometry_policy_flat_size(
    config: WorldGeometryPolicyConfig | Mapping[str, object] | None = None,
) -> int:
    """Return floats contributed to the dense Arsenal policy surface."""

    value = normalize_world_geometry_policy_config(config)
    return 1 + value.token_capacity + value.token_capacity * value.feature_size


def world_geometry_shared_visual_size(
    config: WorldGeometryPolicyConfig | Mapping[str, object] | None = None,
) -> int:
    """Return the compact Java/JAX-shared visual width used by native IL."""

    value = normalize_world_geometry_policy_config(config)
    return 1 + value.token_capacity * (
        WORLD_GEOMETRY_SHARED_VISUAL_FEATURE_SIZE + 1
    )


def world_geometry_shared_visual_dense_offsets(
    config: WorldGeometryPolicyConfig | Mapping[str, object] | None = None,
) -> tuple[int, ...]:
    """Locate compact visual fields inside the flattened World-token group."""

    value = normalize_world_geometry_policy_config(config)
    token = tuple(
        row * value.feature_size + feature
        for row in range(value.token_capacity)
        for feature in range(WORLD_GEOMETRY_SHARED_VISUAL_FEATURE_SIZE)
    )
    mask_start = value.token_capacity * value.feature_size
    return (*token, *range(mask_start, mask_start + value.token_capacity + 1))


def world_geometry_shared_visual(
    tokens: WorldGeometryPolicyTokens,
) -> Array:
    """Project compact geometry semantics shared exactly by Java and JAX.

    Edge graphs and detailed boxes remain JAX-side supplemental inputs.  The
    shared view keeps token kind/provenance, relative position, clearance,
    semantic flags, dynamic occupancy, token validity, and row availability.
    """

    if not isinstance(tokens, WorldGeometryPolicyTokens):
        raise TypeError("tokens must be WorldGeometryPolicyTokens")
    if tokens.token_f32.ndim != 3 or tokens.token_f32.shape[:2] != (
        *tokens.token_mask.shape,
    ):
        raise ValueError("World token arrays have inconsistent shapes")
    if tokens.available.shape != tokens.token_mask.shape[:1]:
        raise ValueError("World token availability has an inconsistent shape")
    if tokens.token_f32.shape[2] < WORLD_GEOMETRY_SHARED_VISUAL_FEATURE_SIZE:
        raise ValueError("World tokens omit the shared visual base fields")
    return jnp.concatenate(
        (
            tokens.token_f32[..., :WORLD_GEOMETRY_SHARED_VISUAL_FEATURE_SIZE].reshape(
                (tokens.token_f32.shape[0], -1)
            ),
            tokens.token_mask.astype(jnp.float32),
            tokens.available[:, None].astype(jnp.float32),
        ),
        axis=1,
    )


def world_geometry_shared_visual_manifest(
    config: WorldGeometryPolicyConfig | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return stable identity for the compact cross-runtime visual branch."""

    value = normalize_world_geometry_policy_config(config)
    return {
        "schema": WORLD_GEOMETRY_SHARED_VISUAL_SCHEMA,
        "source_contract_sha256": world_geometry_token_contract_sha256(),
        "token_capacity": value.token_capacity,
        "token_base_feature_count": WORLD_GEOMETRY_SHARED_VISUAL_FEATURE_SIZE,
        "fields": [
            "token_kind",
            "token_provenance",
            "relative_position_x",
            "relative_position_y",
            "relative_position_z",
            "clearance",
            "semantic_flags",
            "dynamic_blocked",
            "token_mask",
            "available",
        ],
        "jax_only_supplemental": [
            "traversal_edges",
            "collision_detail_boxes",
            "actor_light",
            "inventory",
            "action_candidates",
        ],
        "dense_offsets": list(world_geometry_shared_visual_dense_offsets(value)),
    }


def world_geometry_shared_visual_sha256(
    config: WorldGeometryPolicyConfig | Mapping[str, object] | None = None,
) -> str:
    payload = json.dumps(
        world_geometry_shared_visual_manifest(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def empty_world_geometry_policy_tokens(
    batch: int,
    config: WorldGeometryPolicyConfig | Mapping[str, object] | None = None,
) -> WorldGeometryPolicyTokens:
    """Return a fixed-shape unavailable token row."""

    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    value = normalize_world_geometry_policy_config(config)
    return WorldGeometryPolicyTokens(
        available=jnp.zeros((batch,), dtype=jnp.bool_),
        token_f32=jnp.zeros(
            (batch, value.token_capacity, value.feature_size),
            dtype=jnp.float32,
        ),
        token_mask=jnp.zeros(
            (batch, value.token_capacity),
            dtype=jnp.bool_,
        ),
    )


def encode_world_geometry_policy_tokens(
    source: WorldGeometryTokenObservation,
    *,
    actor_index: int = 0,
    config: WorldGeometryPolicyConfig | Mapping[str, object] | None = None,
) -> WorldGeometryPolicyTokens:
    """Copy only declared policy fields and fail malformed active rows closed."""

    if not isinstance(source, WorldGeometryTokenObservation):
        raise TypeError("source must be WorldGeometryTokenObservation")
    value = normalize_world_geometry_policy_config(config)
    _validate_source_shapes(source, actor_index, value)

    available = source.available[:, actor_index]
    token_mask = source.token_mask[:, actor_index] & available[:, None]
    token_kind = source.token_kind[:, actor_index]
    token_provenance = source.token_provenance[:, actor_index]
    relative_position = source.relative_position[:, actor_index]
    clearance = source.clearance[:, actor_index]
    semantic_flags = source.semantic_flags[:, actor_index]
    dynamic_blocked = source.dynamic_blocked[:, actor_index]
    edge_mask = source.edge_mask[:, actor_index] & token_mask[..., None]
    edge_destination = source.edge_destination[:, actor_index]
    edge_cost = source.edge_cost[:, actor_index]
    edge_kind = source.edge_kind[:, actor_index]
    edge_flags = source.edge_flags[:, actor_index]
    collision_box_mask = (
        source.collision_box_mask[:, actor_index] & token_mask[..., None]
    )
    collision_boxes = source.collision_boxes_relative[:, actor_index]

    edge_mask, edge_destination, edge_cost, edge_kind, edge_flags = _pad_edges(
        edge_mask,
        edge_destination,
        edge_cost,
        edge_kind,
        edge_flags,
        value.edge_capacity,
    )
    collision_box_mask, collision_boxes = _pad_collision_boxes(
        collision_box_mask,
        collision_boxes,
    )

    finite_tokens = jnp.all(jnp.isfinite(relative_position), axis=2) & jnp.isfinite(
        clearance
    )
    finite_edges = jnp.isfinite(edge_cost)
    finite_boxes = jnp.all(jnp.isfinite(collision_boxes), axis=3)
    destination = jnp.clip(
        edge_destination,
        0,
        value.token_capacity - 1,
    )
    destination_active = jnp.take_along_axis(
        token_mask[..., None],
        destination,
        axis=1,
    )
    edge_well_formed = (
        (edge_destination >= 0)
        & (edge_destination < value.token_capacity)
        & destination_active
        & finite_edges
    )
    token_kind_valid = token_kind <= WORLD_TOKEN_KIND_COLLISION_EXCEPTION
    provenance_valid = token_provenance <= WORLD_TOKEN_PROVENANCE_SURROGATE_TRAVERSAL
    row_valid = (
        jnp.all(
            ~token_mask | (finite_tokens & token_kind_valid & provenance_valid), axis=1
        )
        & jnp.all(~edge_mask | edge_well_formed, axis=(1, 2))
        & jnp.all(~collision_box_mask | finite_boxes, axis=(1, 2))
    )
    available &= row_valid
    token_mask &= available[:, None]
    edge_mask &= token_mask[..., None]
    collision_box_mask &= token_mask[..., None]

    distance_scale = jnp.float32(value.maximum_distance)
    base = jnp.stack(
        (
            token_kind.astype(jnp.float32)
            / jnp.float32(WORLD_TOKEN_KIND_COLLISION_EXCEPTION),
            token_provenance.astype(jnp.float32)
            / jnp.float32(WORLD_TOKEN_PROVENANCE_SURROGATE_TRAVERSAL),
            relative_position[..., 0] / distance_scale,
            relative_position[..., 1] / distance_scale,
            relative_position[..., 2] / distance_scale,
            clearance / distance_scale,
            semantic_flags.astype(jnp.float32) / jnp.float32(65535.0),
            dynamic_blocked.astype(jnp.float32),
        ),
        axis=2,
    )
    base = jnp.clip(base, -1.0, 1.0)

    edge_features = jnp.stack(
        (
            edge_mask.astype(jnp.float32),
            jnp.where(
                edge_mask,
                (edge_destination.astype(jnp.float32) + 1.0)
                / jnp.float32(value.token_capacity + 1),
                0.0,
            ),
            jnp.clip(edge_cost / distance_scale, 0.0, 1.0),
            edge_kind.astype(jnp.float32) / jnp.float32(255.0),
            edge_flags.astype(jnp.float32) / jnp.float32(255.0),
        ),
        axis=3,
    ).reshape(
        (
            token_mask.shape[0],
            value.token_capacity,
            value.edge_capacity * WORLD_TOKEN_EDGE_FLOAT_SIZE,
        )
    )
    box_features = jnp.concatenate(
        (
            collision_box_mask[..., None].astype(jnp.float32),
            jnp.clip(collision_boxes / distance_scale, -1.0, 1.0),
        ),
        axis=3,
    ).reshape(
        (
            token_mask.shape[0],
            value.token_capacity,
            MAX_DETAIL_BOXES * WORLD_TOKEN_COLLISION_BOX_FLOAT_SIZE,
        )
    )
    token_gate = token_mask[..., None]
    encoded = jnp.where(
        token_gate,
        jnp.concatenate((base, edge_features, box_features), axis=2),
        jnp.float32(0.0),
    )
    return WorldGeometryPolicyTokens(
        available=available,
        token_f32=encoded,
        token_mask=token_mask,
    )


def mask_world_geometry_policy_tokens(
    tokens: WorldGeometryPolicyTokens,
    valid: Array,
) -> WorldGeometryPolicyTokens:
    """Apply the outer learner-observation validity gate."""

    available = tokens.available & valid
    mask = tokens.token_mask & available[:, None]
    return WorldGeometryPolicyTokens(
        available=available,
        token_f32=jnp.where(mask[..., None], tokens.token_f32, 0.0),
        token_mask=mask,
    )


def _validate_source_shapes(
    source: WorldGeometryTokenObservation,
    actor_index: int,
    config: WorldGeometryPolicyConfig,
) -> None:
    if isinstance(actor_index, bool) or not isinstance(actor_index, int):
        raise TypeError("actor_index must be an integer")
    if source.available.ndim != 2:
        raise ValueError("source.available must have shape [B, A]")
    batch, actors = source.available.shape
    if not 0 <= actor_index < actors:
        raise ValueError("actor_index is outside the source actor axis")
    token_shape = (batch, actors, config.token_capacity)
    scalar_fields = (
        "token_mask",
        "token_kind",
        "token_provenance",
        "clearance",
        "semantic_flags",
        "dynamic_blocked",
    )
    for name in scalar_fields:
        if getattr(source, name).shape != token_shape:
            raise ValueError(f"source.{name} must have shape {token_shape}")
    if source.relative_position.shape != token_shape + (3,):
        raise ValueError(
            f"source.relative_position must have shape {token_shape + (3,)}"
        )
    edge_shape = source.edge_mask.shape
    if (
        len(edge_shape) != 4
        or edge_shape[:3] != token_shape
        or edge_shape[3] > config.edge_capacity
    ):
        raise ValueError(
            "source edge axis must match [B, A, T] and fit configured capacity"
        )
    for name in ("edge_destination", "edge_cost", "edge_kind", "edge_flags"):
        if getattr(source, name).shape != edge_shape:
            raise ValueError(f"source.{name} must match source.edge_mask")
    box_shape = source.collision_box_mask.shape
    if (
        len(box_shape) != 4
        or box_shape[:3] != token_shape
        or box_shape[3] > MAX_DETAIL_BOXES
    ):
        raise ValueError(
            "source collision-box axis must match [B, A, T] and public maximum"
        )
    if source.collision_boxes_relative.shape != box_shape + (6,):
        raise ValueError(
            "source.collision_boxes_relative must match collision_box_mask"
        )


def _pad_edges(
    mask: Array,
    destination: Array,
    cost: Array,
    kind: Array,
    flags: Array,
    capacity: int,
) -> tuple[Array, Array, Array, Array, Array]:
    padding = capacity - mask.shape[2]
    if padding == 0:
        return mask, destination, cost, kind, flags
    widths = ((0, 0), (0, 0), (0, padding))
    return (
        jnp.pad(mask, widths),
        jnp.pad(destination, widths, constant_values=-1),
        jnp.pad(cost, widths),
        jnp.pad(kind, widths),
        jnp.pad(flags, widths),
    )


def _pad_collision_boxes(mask: Array, boxes: Array) -> tuple[Array, Array]:
    padding = MAX_DETAIL_BOXES - mask.shape[2]
    if padding == 0:
        return mask, boxes
    return (
        jnp.pad(mask, ((0, 0), (0, 0), (0, padding))),
        jnp.pad(boxes, ((0, 0), (0, 0), (0, padding), (0, 0))),
    )


__all__ = [
    "WORLD_GEOMETRY_SHARED_VISUAL_SCHEMA",
    "DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG",
    "WORLD_GEOMETRY_TOKEN_CAPACITY",
    "WORLD_GEOMETRY_SOURCE_POLICY_FIELDS",
    "WORLD_TOKEN_DEFAULT_EDGE_CAPACITY",
    "WORLD_TOKEN_DEFAULT_MAXIMUM_DISTANCE",
    "WorldGeometryPolicyConfig",
    "WorldGeometryPolicyTokens",
    "empty_world_geometry_policy_tokens",
    "encode_world_geometry_policy_tokens",
    "mask_world_geometry_policy_tokens",
    "normalize_world_geometry_policy_config",
    "world_geometry_policy_config_manifest",
    "world_geometry_policy_feature_size",
    "world_geometry_policy_flat_size",
    "world_geometry_shared_visual",
    "world_geometry_shared_visual_dense_offsets",
    "world_geometry_shared_visual_manifest",
    "world_geometry_shared_visual_sha256",
    "world_geometry_shared_visual_size",
]
