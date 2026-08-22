"""Read the structured observation, and turn it into whatever your net wants.

The dense ``policy_input.observation`` row is one *projection* of the learner
observation, not the observation itself.  A transformer, GNN, or world model
usually wants the structure back: sets of entities, projectiles, terrain
patches, and world-geometry tokens, each with the mask that says which slots
are real.

This module publishes that structure and a small encoder registry so an
architecture can pick its own view without every author re-deriving the tree by
introspection.  Nothing here re-implements Gym encoding; it selects and
reshapes what the environment already produced.

Measured on a live ``combat/fail_closed`` build at batch 2 **on 2026-07-31**:
the structured tree held 8263 elements per environment against a dense row of
8259, so the flat projection was nearly lossless -- what it discards is
structure, not magnitude.

That pair is stale on both sides.  ``arsenal_policy_observation_size()``
returned **8271** when measured on 2026-08-03, so the tree figure needs
re-measuring alongside it; do not assume the 8263/8259 margin still holds, and
do not update one number without the other -- swapping only the dense width
inverts the comparison and makes the projection look lossy.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v3.world_tokens import (
    WORLD_TOKEN_BASE_FLOAT_SIZE,
)


class TokenSet(NamedTuple):
    """A padded set of slots plus the mask marking the occupied ones.

    ``values`` is ``(..., slots, features)`` and ``mask`` is ``(..., slots)``.
    Attention layers want exactly this pair; never read ``values`` without
    applying ``mask``, because padded slots hold arbitrary values.
    """

    values: jax.Array
    mask: jax.Array

    @property
    def slots(self) -> int:
        return int(self.values.shape[-2])

    @property
    def features(self) -> int:
        return int(self.values.shape[-1])

    def masked(self, fill: float = 0.0) -> jax.Array:
        """Return ``values`` with padded slots replaced by ``fill``."""

        return jnp.where(self.mask[..., None], self.values, fill)

    def count(self) -> jax.Array:
        """Number of occupied slots per row."""

        return jnp.sum(self.mask.astype(jnp.int32), axis=-1)


#: Set-structured groups, as ``name -> (container, values_field, mask_field)``.
#: ``container`` is ``"base"`` for the shared combat observation, ``"arsenal"``
#: for groups that carry a per-entity axis, or ``"world_geometry"``.
SET_GROUPS: dict[str, tuple[str, str, str]] = {
    "entity": ("base", "entity_f32", "entity_mask"),
    "projectile": ("base", "projectile_f32", "projectile_mask"),
    "hazard": ("base", "hazard_f32", "hazard_mask"),
    "terrain": ("base", "terrain_f32", "terrain_mask"),
    "traversal": ("base", "traversal_f32", "traversal_mask"),
    "interaction": ("base", "interaction_f32", "interaction_mask"),
    "status": ("arsenal", "status_f32", "status_mask"),
    "ability": ("arsenal", "ability_f32", "ability_mask"),
    "world_token": ("world_geometry", "token_f32", "token_mask"),
}

#: Fixed-width vector groups, as ``name -> (container, field)``.
VECTOR_GROUPS: dict[str, tuple[str, str]] = {
    "self": ("base", "self_f32"),
    "target": ("base", "target_f32"),
    "combat": ("base", "combat_f32"),
    "actor_world": ("arsenal", "actor_world_f32"),
    "movement_state": ("arsenal", "movement_state_f32"),
    "resource": ("arsenal", "resource_f32"),
    "defense": ("arsenal", "defense_f32"),
}


#: Groups that :data:`GROUP_FEATURES` names but neither reader above pulls, as
#: ``name -> (container, field)``.  Keyed by the ``GROUP_FEATURES`` spelling
#: rather than a short name, because these have no float counterpart to shorten
#: against and the schema is what a caller wants to index by.
#:
#: These are the integer columns and the availability masks.  The masks are the
#: reason this exists: ``resource_mask`` is the only thing that separates "this
#: stat is empty" from "this family has no such stat", so a probe that reads
#: ``resource_f32`` alone cannot tell a real zero from an absent one.
AUXILIARY_GROUPS: dict[str, tuple[str, str]] = {
    "self_i32": ("base", "self_i32"),
    "target_i32": ("base", "target_i32"),
    "combat_i32": ("base", "combat_i32"),
    "entity_i32": ("base", "entity_i32"),
    "projectile_i32": ("base", "projectile_i32"),
    "hazard_i32": ("base", "hazard_i32"),
    "status_i32": ("arsenal", "status_i32"),
    "ability_i32": ("arsenal", "ability_i32"),
    "weapon_i32": ("arsenal", "weapon_i32"),
    "actor_world_mask": ("arsenal", "actor_world_mask"),
    "resource_mask": ("arsenal", "resource_mask"),
}


def _container(legal_observation: Any, kind: str) -> Any:
    if kind == "base":
        return legal_observation.base
    if kind == "world_geometry":
        return legal_observation.world_geometry
    return legal_observation


def observation_sets(legal_observation: Any) -> dict[str, TokenSet]:
    """Return every set-structured group keyed by name.

    ``status`` and ``ability`` keep their per-entity axis, so their shapes are
    ``(batch, entities, slots, features)`` while the rest are
    ``(batch, slots, features)``.  An environment holds two entities: the agent
    and its target.
    """

    sets: dict[str, TokenSet] = {}
    for name, (kind, values_field, mask_field) in SET_GROUPS.items():
        container = _container(legal_observation, kind)
        sets[name] = TokenSet(
            values=getattr(container, values_field),
            mask=getattr(container, mask_field),
        )
    return sets


def observation_vectors(legal_observation: Any) -> dict[str, jax.Array]:
    """Return the fixed-width vector groups keyed by name."""

    return {
        name: getattr(_container(legal_observation, kind), field)
        for name, (kind, field) in VECTOR_GROUPS.items()
    }


def observation_auxiliaries(legal_observation: Any) -> dict[str, jax.Array]:
    """Integer and availability-mask groups, keyed as in ``GROUP_FEATURES``.

    ``observation_vectors`` and ``observation_sets`` between them cover the
    float half of the schema.  These eleven groups are on the observation tree
    and named in ``GROUP_FEATURES``, but no other reader pulls them, so a probe
    that never asks for them by name never sees them at all.

    Leading axes differ per group -- some carry a per-entity axis, some a slot
    axis -- but the feature axis is always last, which is what
    ``column_activity`` needs.
    """

    return {
        name: getattr(_container(legal_observation, kind), field)
        for name, (kind, field) in AUXILIARY_GROUPS.items()
    }


class LeafSpec(NamedTuple):
    """One structured leaf, for printing a build's observation surface."""

    path: str
    shape: tuple[int, ...]
    dtype: str


def describe_observation(legal_observation: Any) -> tuple[LeafSpec, ...]:
    """Enumerate every leaf with its shape and dtype.

    Useful when sizing an encoder: the geometry configuration changes token
    capacity, which changes both this tree and the dense row width.
    """

    leaves = jax.tree_util.tree_flatten_with_path(legal_observation)[0]
    described = []
    for path, leaf in leaves:
        if not hasattr(leaf, "shape"):
            continue
        described.append(
            LeafSpec(
                path=jax.tree_util.keystr(path),
                shape=tuple(int(dim) for dim in leaf.shape),
                dtype=str(leaf.dtype),
            )
        )
    return tuple(described)


# -- encoder registry ---------------------------------------------------------


class Encoder(NamedTuple):
    """A named view of an actor input, plus one line on what it produces."""

    encode: Callable[[Any], Any]
    description: str


_ENCODERS: dict[str, Encoder] = {}


def register_encoder(name: str, encoder: Encoder) -> Encoder:
    """Register an observation view under a stable name."""

    if not isinstance(name, str) or not name:
        raise ValueError("encoder name must be a non-empty string")
    if not isinstance(encoder, Encoder):
        raise TypeError("encoder must be an Encoder")
    existing = _ENCODERS.get(name)
    if existing is not None and existing != encoder:
        raise ValueError(f"encoder already registered differently: {name!r}")
    _ENCODERS[name] = encoder
    return encoder


def get_encoder(name: str) -> Encoder:
    try:
        return _ENCODERS[name]
    except KeyError as error:
        raise KeyError(
            f"unknown encoder {name!r}; registered: {list_encoders()}"
        ) from error


def list_encoders() -> list[str]:
    return sorted(_ENCODERS)


register_encoder(
    "flat",
    Encoder(
        encode=lambda actor_input: actor_input.observation,
        description="the dense (batch, observation_size) float32 row",
    ),
)
register_encoder(
    "sets",
    Encoder(
        encode=lambda actor_input: observation_sets(
            actor_input.legal_observation
        ),
        description="every set-structured group as name -> TokenSet",
    ),
)
register_encoder(
    "vectors",
    Encoder(
        encode=lambda actor_input: observation_vectors(
            actor_input.legal_observation
        ),
        description="fixed-width vector groups as name -> array",
    ),
)
register_encoder(
    "world_tokens",
    Encoder(
        encode=lambda actor_input: observation_sets(
            actor_input.legal_observation
        )["world_token"],
        description="world geometry as a single TokenSet",
    ),
)


# -- opt-in rasterizer --------------------------------------------------------

#: Columns 2, 3, 4 of a world token are agent-relative XYZ divided by the
#: scene's ``maximum_distance`` and clipped to [-1, 1].
#:
#: Derived by reading the Gym encoder at
#: ``hytalegym/jax/combat/observation/v3/world_tokens.py:282-297``, which stacks
#: ``(kind, provenance, rel_x, rel_y, rel_z, clearance, semantic, blocked)``.
#: The Gym publishes the *width* of that block but not the column names, so the
#: guard below catches a resize, not a reordering.  If upstream ever permutes
#: those eight columns this rasterizer misplaces voxels silently -- re-derive
#: from that function before trusting a grid across a Gym upgrade.
WORLD_TOKEN_POSITION_COLUMNS: tuple[int, int, int] = (2, 3, 4)

if WORLD_TOKEN_BASE_FLOAT_SIZE != 8:
    raise RuntimeError(
        "world token base feature block changed width "
        f"({WORLD_TOKEN_BASE_FLOAT_SIZE} != 8); re-derive "
        "WORLD_TOKEN_POSITION_COLUMNS from the Gym encoder before rasterizing"
    )


def rasterize(
    tokens: TokenSet,
    *,
    resolution: int = 16,
    feature_columns: tuple[int, ...] | None = None,
    position_columns: tuple[int, int, int] = WORLD_TOKEN_POSITION_COLUMNS,
) -> jax.Array:
    """Bin a token set into a dense voxel grid a CNN can consume.

    Returns ``(batch, channels, resolution, resolution, resolution)`` where
    channel 0 is occupancy (how many tokens landed in the voxel) and the
    remaining channels are the sums of ``feature_columns`` over those tokens.

    This is a faithful re-binning of positions the environment already
    published; it is **not** a Gym-sanctioned spatial view.  The observation is
    natively a token set, and set/attention encoders lose nothing.  Reach for a
    grid only when the architecture specifically needs convolution, and expect
    collisions: several tokens can share a voxel, and any token outside the
    scene's ``maximum_distance`` is clipped to the boundary rather than dropped.
    """

    if not isinstance(tokens, TokenSet):
        raise TypeError("rasterize expects a TokenSet")
    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise TypeError("resolution must be an integer")
    if resolution < 1:
        raise ValueError("resolution must be positive")
    values = jnp.asarray(tokens.values)
    if values.ndim != 3:
        raise ValueError(
            "rasterize expects (batch, slots, features); groups with an "
            f"entity axis must be selected first, got {values.shape}"
        )
    width = values.shape[-1]
    if max(position_columns) >= width:
        raise ValueError(
            f"position columns {position_columns} exceed feature width {width}"
        )
    if feature_columns is None:
        feature_columns = ()
    for column in feature_columns:
        if column >= width:
            raise ValueError(
                f"feature column {column} exceeds feature width {width}"
            )

    mask = jnp.asarray(tokens.mask)
    positions = values[..., list(position_columns)]
    # [-1, 1] -> [0, resolution - 1]
    indices = jnp.floor((positions + 1.0) * 0.5 * resolution).astype(jnp.int32)
    indices = jnp.clip(indices, 0, resolution - 1)

    occupancy = mask.astype(jnp.float32)
    channels = [occupancy]
    for column in feature_columns:
        channels.append(jnp.where(mask, values[..., column], 0.0))
    stacked = jnp.stack(channels, axis=-1)  # (batch, slots, channels)

    def scatter_one(slot_indices, slot_channels):
        grid = jnp.zeros(
            (resolution, resolution, resolution, stacked.shape[-1]),
            dtype=jnp.float32,
        )
        return grid.at[
            slot_indices[:, 0], slot_indices[:, 1], slot_indices[:, 2]
        ].add(slot_channels)

    grids = jax.vmap(scatter_one)(indices, stacked)
    # (batch, D, H, W, C) -> (batch, C, D, H, W), the layout conv kernels expect
    return jnp.transpose(grids, (0, 4, 1, 2, 3))


register_encoder(
    "world_grid",
    Encoder(
        encode=lambda actor_input: rasterize(
            observation_sets(actor_input.legal_observation)["world_token"]
        ),
        description=(
            "world geometry rasterized to (batch, 1, 16, 16, 16) occupancy; "
            "call rasterize() directly to choose resolution and channels"
        ),
    ),
)


__all__ = [
    "AUXILIARY_GROUPS",
    "SET_GROUPS",
    "VECTOR_GROUPS",
    "WORLD_TOKEN_POSITION_COLUMNS",
    "Encoder",
    "LeafSpec",
    "TokenSet",
    "describe_observation",
    "get_encoder",
    "list_encoders",
    "observation_auxiliaries",
    "observation_sets",
    "observation_vectors",
    "rasterize",
    "register_encoder",
]
