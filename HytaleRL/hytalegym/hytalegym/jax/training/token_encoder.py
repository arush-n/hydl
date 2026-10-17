"""Shared token encoder over the dense Arsenal policy row (architecture brief 2.4).

The dense row is sliced with the live layout accessors, in the exact
concatenation order of ``arsenal_policy_observation``; no offset or width is
copied. Each token set -- world geometry, the geometry-aligned actor light
rows, and actor inventory -- is encoded by one MLP shared across its token axis
(Dense -> LayerNorm -> GELU -> Dense) and pooled under its mask as masked mean
concatenated with masked max, so slot order cannot matter and masked-out slots
cannot contribute. The remaining per-actor scalars (the base block) and the
block/recipe candidate surface each get their own Dense -> LayerNorm -> GELU.
Candidates stay dense on purpose: the ``block_none_plus_candidates`` head
indexes candidate slots, so pooling would erase the identity it selects by.

Parameters do not depend on ``token_capacity``; a capacity change is a config
change, not a reshape. Not wired into training: ``policy.py`` and ``ppo.py``
still use the dense ``encoder_input``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import OBSERVATION_CAPACITY
from hytalegym.jax.combat.inventory import CONTAINER_COUNT
from hytalegym.jax.combat.mechanics import RESOURCE_COUNT, STATUS_CAPACITY
from hytalegym.jax.combat.observation.v1.schema.contract import (
    SELF_FLOAT_FEATURES,
    TARGET_FLOAT_FEATURES,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.blocks import (
    BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE,
)
from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_COMBAT_FLOAT_FEATURES,
    ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
    ARSENAL_POLICY_RECIPE_CANDIDATE_EMBEDDING_SIZE,
    arsenal_policy_observation_size,
)
from hytalegym.jax.combat.observation.v3.schema.contract import (
    ABILITY_FLOAT_SIZE,
    ACTOR_WORLD_FLOAT_SIZE,
    ACTOR_WORLD_MASK_SIZE,
    DEFENSE_FLOAT_SIZE,
    DODGE_ACTION_COUNT,
    MOVEMENT_STATE_SIZE,
    STATUS_FLOAT_SIZE,
)
from hytalegym.jax.combat.observation.v3.tokens.inventory import (
    INVENTORY_POLICY_FLOAT_SIZE,
    INVENTORY_POLICY_TOKEN_CAPACITY,
)
from hytalegym.jax.combat.observation.v3.tokens.light import (
    ACTOR_LIGHT_POLICY_FEATURE_SIZE,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
    WorldGeometryPolicyConfig,
    normalize_world_geometry_policy_config,
)
from hytalegym.jax.combat.skills import SKILL_COUNT
from hytalegym.jax.policy import DenseParams
from hytalegym.jax.world import ACTOR_RECIPE_CANDIDATE_CAPACITY


Array = jax.Array

_GEOMETRY_FEATURES = "world_geometry.token_f32_masked"
_GEOMETRY_MASK = "world_geometry.token_mask"
_LIGHT_FEATURES = "actor_light.token_f32_masked"
_LIGHT_AVAILABLE = "actor_light.available"
_INVENTORY_FEATURES = "actor_inventory.token_f32_masked"
_INVENTORY_MASK = "actor_inventory.token_mask"
_CANDIDATE_SEGMENTS = (
    "block_candidates.candidate_f32_masked",
    "block_candidates.candidate_mask",
    "block_candidates.available",
    "recipe_candidates.candidate_embedding_masked",
    "recipe_candidates.candidate_mask",
    "recipe_candidates.available",
)
# Slot-indexed token data never enters a dense encoder; masks are used to pool.
_TOKEN_SEGMENTS = frozenset(
    (
        _GEOMETRY_FEATURES,
        _GEOMETRY_MASK,
        _LIGHT_FEATURES,
        _INVENTORY_FEATURES,
        _INVENTORY_MASK,
    )
)
_LAYER_NORM_EPSILON = 1e-5


@dataclass(frozen=True)
class TokenEncoderConfig:
    """Encoder widths; part of the checkpoint contract once wired in."""

    token_hidden: int = 128
    base_hidden: int = 128
    fused: int = 256
    light_hidden: int = 32
    inventory_hidden: int = 64
    candidate_hidden: int = 128
    world_geometry: WorldGeometryPolicyConfig = DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "world_geometry":
                if not isinstance(value, WorldGeometryPolicyConfig):
                    raise TypeError("world_geometry must be a WorldGeometryPolicyConfig")
            elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{item.name} must be a positive integer")


class LayerNormParams(NamedTuple):
    scale: Array
    offset: Array


class TokenSetEncoderParams(NamedTuple):
    """Dense -> LayerNorm -> GELU -> Dense, shared by every token of one set."""

    input: DenseParams
    norm: LayerNormParams
    output: DenseParams


class DenseBlockParams(NamedTuple):
    """Dense -> LayerNorm -> GELU."""

    dense: DenseParams
    norm: LayerNormParams


class TokenEncoderParams(NamedTuple):
    geometry: TokenSetEncoderParams
    light: TokenSetEncoderParams
    inventory: TokenSetEncoderParams
    base: DenseBlockParams
    candidates: DenseBlockParams
    fusion: DenseBlockParams


def dense_row_segments(
    world_geometry_config: WorldGeometryPolicyConfig | None = None,
) -> tuple[tuple[str, int], ...]:
    """Return ``(manifest group, width)`` in ``arsenal_policy_observation`` order."""

    geometry = normalize_world_geometry_policy_config(world_geometry_config)
    tokens = geometry.token_capacity
    return (
        ("base.self_f32", len(SELF_FLOAT_FEATURES)),
        ("base.target_f32", len(TARGET_FLOAT_FEATURES)),
        ("base.target_mask", 1),
        (
            "base.combat_f32[arsenal_policy_features]",
            len(ARSENAL_POLICY_COMBAT_FLOAT_FEATURES),
        ),
        ("agent.resource_f32_masked", RESOURCE_COUNT),
        ("agent.resource_mask", RESOURCE_COUNT),
        ("agent.defense_f32", DEFENSE_FLOAT_SIZE),
        ("agent.status_f32_masked", STATUS_CAPACITY * STATUS_FLOAT_SIZE),
        ("agent.status_mask", STATUS_CAPACITY),
        ("agent.ability_f32_masked", OBSERVATION_CAPACITY * ABILITY_FLOAT_SIZE),
        ("agent.ability_mask", OBSERVATION_CAPACITY),
        ("agent.ability_legal", OBSERVATION_CAPACITY),
        ("actor_world_f32_masked", ACTOR_WORLD_FLOAT_SIZE),
        ("actor_world_mask", ACTOR_WORLD_MASK_SIZE),
        ("movement_state_f32_masked", MOVEMENT_STATE_SIZE),
        ("movement_state_mask", MOVEMENT_STATE_SIZE),
        (_GEOMETRY_FEATURES, tokens * geometry.feature_size),
        (_GEOMETRY_MASK, tokens),
        ("world_geometry.available", 1),
        (_LIGHT_FEATURES, tokens * ACTOR_LIGHT_POLICY_FEATURE_SIZE),
        (_LIGHT_AVAILABLE, 1),
        ("actor_inventory.container_f32_masked", CONTAINER_COUNT),
        ("actor_inventory.container_mask", CONTAINER_COUNT),
        (
            _INVENTORY_FEATURES,
            INVENTORY_POLICY_TOKEN_CAPACITY * INVENTORY_POLICY_FLOAT_SIZE,
        ),
        (_INVENTORY_MASK, INVENTORY_POLICY_TOKEN_CAPACITY),
        ("actor_inventory.available", 1),
        ("skill_action_mask", SKILL_COUNT),
        ("jump_action_mask", 1),
        ("guard_action_mask", 1),
        ("dodge_action_mask", DODGE_ACTION_COUNT),
        ("valid", 1),
        (
            "block_candidates.candidate_f32_masked",
            ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY
            * BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE,
        ),
        (
            "block_candidates.candidate_mask",
            ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
        ),
        ("block_candidates.available", 1),
        (
            "recipe_candidates.candidate_embedding_masked",
            ACTOR_RECIPE_CANDIDATE_CAPACITY
            * ARSENAL_POLICY_RECIPE_CANDIDATE_EMBEDDING_SIZE,
        ),
        ("recipe_candidates.candidate_mask", ACTOR_RECIPE_CANDIDATE_CAPACITY),
        ("recipe_candidates.available", 1),
    )


def dense_row_layout(
    world_geometry_config: WorldGeometryPolicyConfig | None = None,
) -> dict[str, tuple[int, int]]:
    """Return ``{group: (start, stop)}``; fails closed on any width drift."""

    return dict(
        _dense_row_layout(normalize_world_geometry_policy_config(world_geometry_config))
    )


@lru_cache(maxsize=None)
def _dense_row_layout(
    geometry: WorldGeometryPolicyConfig,
) -> tuple[tuple[str, tuple[int, int]], ...]:
    spans = []
    start = 0
    for name, width in dense_row_segments(geometry):
        spans.append((name, (start, start + width)))
        start += width
    expected = arsenal_policy_observation_size(geometry)
    if start != expected:
        raise RuntimeError(f"token encoder layout drift: {start} != {expected}")
    return tuple(spans)


def _input_widths(config: TokenEncoderConfig) -> dict[str, int]:
    segments = dict(dense_row_segments(config.world_geometry))
    candidates = sum(segments[name] for name in _CANDIDATE_SEGMENTS)
    base = sum(
        width
        for name, width in segments.items()
        if name not in _TOKEN_SEGMENTS and name not in _CANDIDATE_SEGMENTS
    )
    return {"base": base, "candidates": candidates}


def init_token_encoder(
    key: Array,
    config: TokenEncoderConfig | None = None,
) -> TokenEncoderParams:
    """Initialize all-float32 parameters (Glorot uniform, as ``policy.py``)."""

    config = TokenEncoderConfig() if config is None else config
    widths = _input_widths(config)
    fusion_input = (
        2 * config.token_hidden
        + 2 * config.light_hidden
        + 2 * config.inventory_hidden
        + config.base_hidden
        + config.candidate_hidden
    )
    keys = jax.random.split(key, 6)
    return TokenEncoderParams(
        geometry=_init_token_set(
            keys[0], config.world_geometry.feature_size, config.token_hidden
        ),
        light=_init_token_set(
            keys[1], ACTOR_LIGHT_POLICY_FEATURE_SIZE, config.light_hidden
        ),
        inventory=_init_token_set(
            keys[2], INVENTORY_POLICY_FLOAT_SIZE, config.inventory_hidden
        ),
        base=_init_dense_block(keys[3], widths["base"], config.base_hidden),
        candidates=_init_dense_block(
            keys[4], widths["candidates"], config.candidate_hidden
        ),
        fusion=_init_dense_block(keys[5], fusion_input, config.fused),
    )


def apply_token_encoder(
    params: TokenEncoderParams,
    dense_observation: Array,
    config: TokenEncoderConfig | None = None,
) -> Array:
    """Map float32[..., arsenal_policy_observation_size()] to float32[..., fused]."""

    config = TokenEncoderConfig() if config is None else config
    geometry = config.world_geometry
    layout = dict(_dense_row_layout(geometry))
    row = jnp.asarray(dense_observation, dtype=jnp.float32)
    expected = arsenal_policy_observation_size(geometry)
    if row.ndim < 1 or row.shape[-1] != expected:
        raise ValueError(
            f"dense observation width {row.shape[-1:]} != live layout {expected}"
        )
    lead = row.shape[:-1]

    def take(name: str) -> Array:
        start, stop = layout[name]
        return row[..., start:stop]

    geometry_mask = take(_GEOMETRY_MASK) > 0.5
    # Light rows carry no mask of their own: the layout reuses geometry's mask
    # and clears the whole light row unless its availability bit is set.
    light_mask = geometry_mask & (take(_LIGHT_AVAILABLE) > 0.5)
    inventory_mask = take(_INVENTORY_MASK) > 0.5
    base = jnp.concatenate(
        [
            take(name)
            for name in layout
            if name not in _TOKEN_SEGMENTS and name not in _CANDIDATE_SEGMENTS
        ],
        axis=-1,
    )
    candidates = jnp.concatenate([take(name) for name in _CANDIDATE_SEGMENTS], axis=-1)
    tokens = geometry.token_capacity
    encoded = (
        _apply_token_set(
            params.geometry,
            take(_GEOMETRY_FEATURES).reshape(lead + (tokens, geometry.feature_size)),
            geometry_mask,
        ),
        _apply_token_set(
            params.light,
            take(_LIGHT_FEATURES).reshape(
                lead + (tokens, ACTOR_LIGHT_POLICY_FEATURE_SIZE)
            ),
            light_mask,
        ),
        _apply_token_set(
            params.inventory,
            take(_INVENTORY_FEATURES).reshape(
                lead + (INVENTORY_POLICY_TOKEN_CAPACITY, INVENTORY_POLICY_FLOAT_SIZE)
            ),
            inventory_mask,
        ),
        _apply_dense_block(params.base, base),
        _apply_dense_block(params.candidates, candidates),
    )
    return _apply_dense_block(params.fusion, jnp.concatenate(encoded, axis=-1))


def parameter_count(params) -> int:
    """Count scalars in any parameter PyTree (arrays or ``ShapeDtypeStruct``)."""

    return sum(math.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))


def parameter_breakdown(params: TokenEncoderParams) -> dict[str, int]:
    """Return per-encoder parameter counts; the values sum to the total."""

    return {name: parameter_count(getattr(params, name)) for name in params._fields}


def compare_to_dense_baseline(
    encoder_size: int = 64,
    config: TokenEncoderConfig | None = None,
) -> dict[str, object]:
    """Compare the dense ``encoder_input`` against this encoder, from live widths."""

    config = TokenEncoderConfig() if config is None else config
    observation_size = arsenal_policy_observation_size(config.world_geometry)
    shapes = jax.eval_shape(
        partial(init_token_encoder, config=config), jax.random.PRNGKey(0)
    )
    breakdown = parameter_breakdown(shapes)
    geometry_rows = dict(dense_row_segments(config.world_geometry))[_GEOMETRY_FEATURES]
    # Mirrors policy._dense: kernel [observation_size, encoder_size] plus bias.
    dense_encoder_input = observation_size * encoder_size + encoder_size
    dense_geometry = geometry_rows * encoder_size
    return {
        "observation_size": observation_size,
        "encoder_size": encoder_size,
        "dense_encoder_input_parameters": dense_encoder_input,
        "dense_geometry_token_parameters": dense_geometry,
        "token_path_parameters": breakdown["geometry"],
        "token_path_reduction": dense_geometry / breakdown["geometry"],
        "token_encoder_parameters": sum(breakdown.values()),
        "token_encoder_breakdown": breakdown,
        "dense_output_width": encoder_size,
        "token_encoder_output_width": config.fused,
    }


def _init_dense(key: Array, input_size: int, output_size: int) -> DenseParams:
    limit = math.sqrt(6.0 / (input_size + output_size))
    return DenseParams(
        kernel=jax.random.uniform(
            key,
            (input_size, output_size),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        ),
        bias=jnp.zeros((output_size,), dtype=jnp.float32),
    )


def _init_layer_norm(size: int) -> LayerNormParams:
    return LayerNormParams(
        scale=jnp.ones((size,), dtype=jnp.float32),
        offset=jnp.zeros((size,), dtype=jnp.float32),
    )


def _init_token_set(key: Array, features: int, hidden: int) -> TokenSetEncoderParams:
    first, second = jax.random.split(key)
    return TokenSetEncoderParams(
        input=_init_dense(first, features, hidden),
        norm=_init_layer_norm(hidden),
        output=_init_dense(second, hidden, hidden),
    )


def _init_dense_block(key: Array, inputs: int, outputs: int) -> DenseBlockParams:
    return DenseBlockParams(
        dense=_init_dense(key, inputs, outputs),
        norm=_init_layer_norm(outputs),
    )


def _linear(params: DenseParams, inputs: Array) -> Array:
    return inputs @ params.kernel + params.bias


def _layer_norm(params: LayerNormParams, inputs: Array) -> Array:
    mean = jnp.mean(inputs, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(inputs - mean), axis=-1, keepdims=True)
    normalized = (inputs - mean) * jax.lax.rsqrt(variance + _LAYER_NORM_EPSILON)
    return normalized * params.scale + params.offset


def _apply_dense_block(params: DenseBlockParams, inputs: Array) -> Array:
    return jax.nn.gelu(_layer_norm(params.norm, _linear(params.dense, inputs)))


def _apply_token_set(
    params: TokenSetEncoderParams,
    tokens: Array,
    mask: Array,
) -> Array:
    """Shared per-token MLP, then masked mean concatenated with masked max."""

    keep = mask[..., None]
    # Zero masked inputs too, so a non-finite masked slot cannot poison gradients.
    tokens = jnp.where(keep, tokens, 0.0)
    hidden = jax.nn.gelu(_layer_norm(params.norm, _linear(params.input, tokens)))
    hidden = _linear(params.output, hidden)
    count = jnp.sum(mask, axis=-1, keepdims=True).astype(jnp.float32)
    # `where`, not multiplication: a non-finite masked slot cannot leak through.
    mean = jnp.sum(jnp.where(keep, hidden, 0.0), axis=-2) / jnp.maximum(count, 1.0)
    floor = jnp.finfo(jnp.float32).min
    maximum = jnp.max(jnp.where(keep, hidden, floor), axis=-2)
    maximum = jnp.where(count > 0.0, maximum, 0.0)
    return jnp.concatenate((mean, maximum), axis=-1)


__all__ = [
    "DenseBlockParams",
    "LayerNormParams",
    "TokenEncoderConfig",
    "TokenEncoderParams",
    "TokenSetEncoderParams",
    "apply_token_encoder",
    "compare_to_dense_baseline",
    "dense_row_layout",
    "dense_row_segments",
    "init_token_encoder",
    "parameter_breakdown",
    "parameter_count",
]
