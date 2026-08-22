"""The information beyond the combat observation: crafting, blocks, light, items.

``policy_input.observation`` and ``legal_observation`` describe the fight.  An
agent that builds, gathers, crafts, or navigates by light needs more, and the
environment already produces it -- just not anywhere the ADK previously
pointed:

* ``ActorPolicyInput.action_surface`` is a second structured surface holding
  block-action candidates, full crafting recipes, and a recipe embedding.  It
  was exposed but undocumented and had no accessors.
* Light rides on the **environment state**, not the observation:
  ``state.environment.light_tokens``.  Nothing in ``legal_observation``
  mentions it, which is why it looked absent.
* Inventory is primarily a **native-lane** surface, assembled from a live
  server frame.  The JAX lane can encode it from an inventory state, but no
  built-in scene produces one.

Every column layout below was read out of the Gym encoder rather than guessed,
and each is cited.  Where the Gym publishes a width but not column names, the
guard catches a resize and *not* a reordering -- stated at each site.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.inventory import CONTAINER_COUNT
from hytalegym.jax.combat.observation.v3.block_affordances import (
    BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE,
    BLOCK_AFFORDANCE_TAG_BIT_COUNT,
    BLOCK_GATHER_TYPE_BIT_COUNT,
)
from hytalegym.jax.combat.observation.v3.inventory_tokens import (
    INVENTORY_POLICY_FLOAT_FEATURES,
    INVENTORY_POLICY_FLOAT_SIZE,
)
from hytalegym.jax.combat.observation.v3.light_policy_tokens import (
    ACTOR_LIGHT_POLICY_FEATURES,
    ACTOR_LIGHT_POLICY_FEATURE_SIZE,
)

from adk.policy.fields import field_span
from adk.policy.observations import TokenSet


# -- what is actually live ----------------------------------------------------


class SurfaceStatus(NamedTuple):
    """Whether one optional surface is present, and how much of it is filled."""

    name: str
    available: bool
    occupied: int
    detail: str = ""

    def __bool__(self) -> bool:
        return self.available and self.occupied > 0


def _occupied(mask: Any) -> int:
    return int(jnp.sum(jnp.asarray(mask))) if mask is not None else 0


def _flag(value: Any) -> bool:
    return bool(jnp.any(jnp.asarray(value))) if value is not None else False


def surface_flags(actor_input: Any, state: Any = None) -> dict[str, jax.Array]:
    """Per-surface availability as **arrays**, for use inside a traced loop.

    This is the fast path: it never calls ``int()`` or ``bool()``, so it does
    not synchronize with the device and is safe under ``jit``/``scan``.  Use it
    to gate an encoder branch with ``jnp.where``; use
    :func:`describe_availability` only on the host, where a stall is fine.
    """

    legal = actor_input.legal_observation
    surface = getattr(actor_input, "action_surface", None)
    flags: dict[str, jax.Array] = {
        "world_geometry": legal.world_geometry.available,
    }
    if surface is not None:
        flags["block_candidates"] = surface.block_candidates.available
        flags["recipe_candidates"] = surface.recipe_candidates.available
        flags["recipe_encoding"] = surface.recipe_encoding.available
    if state is not None:
        raw = _raw_light_tokens(state)
        if raw is not None:
            flags["light"] = raw.available
    return flags


def describe_availability(
    actor_input: Any,
    state: Any = None,
) -> dict[str, SurfaceStatus]:
    """Report which optional surfaces carry data for this build.

    Start here.  Most surfaces are empty unless the scene binds the matching
    provider, and an encoder written against an absent surface trains on zeros
    without ever failing.  Pass ``state`` as well to include light, which lives
    on the environment state rather than the observation.

    **Host-side only.**  This reduces arrays to Python ``bool``/``int``, which
    forces a device synchronization.  Call it at startup or when debugging, not
    inside a rollout -- :func:`surface_flags` is the traced equivalent.
    """

    legal = actor_input.legal_observation
    surface = getattr(actor_input, "action_surface", None)
    report: dict[str, SurfaceStatus] = {}

    geometry = legal.world_geometry
    report["world_geometry"] = SurfaceStatus(
        "world_geometry",
        _flag(geometry.available),
        _occupied(geometry.token_mask),
        "bind world_token_provider or load a Region scene",
    )

    if surface is None:
        report["action_surface"] = SurfaceStatus(
            "action_surface", False, 0, "no action surface on this input"
        )
    else:
        blocks = surface.block_candidates
        report["block_candidates"] = SurfaceStatus(
            "block_candidates",
            _flag(blocks.available),
            _occupied(blocks.candidate_mask),
            "bind action_surface_provider",
        )
        recipes = surface.recipe_candidates
        report["recipe_candidates"] = SurfaceStatus(
            "recipe_candidates",
            _flag(recipes.available),
            _occupied(recipes.candidate_mask),
            "bind action_surface_provider",
        )
        encoding = surface.recipe_encoding
        report["recipe_encoding"] = SurfaceStatus(
            "recipe_encoding",
            _flag(encoding.available),
            _occupied(encoding.candidate_mask),
            "set recipe_candidate_encoder_params on the scene",
        )

    if state is not None:
        raw = _raw_light_tokens(state)
        report["light"] = SurfaceStatus(
            "light",
            raw is not None and _flag(raw.available),
            0 if raw is None else _occupied(raw.token_mask),
            "bind world_light_token_provider",
        )

    return report


def available_surfaces(
    actor_input: Any,
    state: Any = None,
) -> tuple[str, ...]:
    """Names of the surfaces that are both available and non-empty."""

    return tuple(
        name
        for name, status in describe_availability(actor_input, state).items()
        if status
    )


# -- block action candidates --------------------------------------------------

#: Candidate columns, from ``encode_block_action_candidate_policy_view`` at
#: ``block_affordances.py:206-212``: three normalized relative-position floats
#: followed by ``_affordance_features`` (``block_affordances.py:510-533``),
#: which is tag bits, then gather bits, then required tool quality.
#:
#: The Gym publishes these widths as constants but not the column order, so the
#: guard below catches a resize, not a permutation.
BLOCK_POSITION_COLUMNS: tuple[int, int, int] = (0, 1, 2)
BLOCK_TAG_COLUMNS = (3, 3 + BLOCK_AFFORDANCE_TAG_BIT_COUNT)
BLOCK_GATHER_COLUMNS = (
    BLOCK_TAG_COLUMNS[1],
    BLOCK_TAG_COLUMNS[1] + BLOCK_GATHER_TYPE_BIT_COUNT,
)
BLOCK_TOOL_QUALITY_COLUMN = BLOCK_GATHER_COLUMNS[1]

if BLOCK_TOOL_QUALITY_COLUMN + 1 != BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE:
    raise RuntimeError(
        "block candidate feature layout changed "
        f"({BLOCK_TOOL_QUALITY_COLUMN + 1} != "
        f"{BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE}); re-derive the column "
        "ranges from block_affordances.encode_block_action_candidate_policy_view"
    )


def _require_action_surface(actor_input: Any) -> Any:
    """Fail with the reason, not an AttributeError on ``None``.

    ``BridgePolicyInput.as_policy_input`` builds an ``ActorPolicyInput`` with
    ``action_surface=None``: the native lane carries the combat observation but
    not the action surface.  Code written against the JAX lane hits this, and
    the cause should be legible.
    """

    surface = getattr(actor_input, "action_surface", None)
    if surface is None:
        raise ValueError(
            "this input carries no action surface, so block and recipe "
            "candidates are unavailable; the native bridge lane publishes "
            "legal_observation only. Use describe_availability() to check "
            "before reading, or surface_flags() inside a traced loop"
        )
    return surface


def block_candidates(actor_input: Any) -> TokenSet:
    """Block-action candidate slots as a masked set."""

    view = _require_action_surface(actor_input).block_candidates
    return TokenSet(values=view.candidate_f32, mask=view.candidate_mask)


def block_positions(candidates: TokenSet) -> jax.Array:
    """Agent-relative XYZ per candidate, normalized by ``maximum_distance``.

    Same convention and scale as world-geometry tokens, so the two can be
    rasterized into one grid.
    """

    return candidates.values[..., list(BLOCK_POSITION_COLUMNS)]


def block_affordance_tags(candidates: TokenSet) -> jax.Array:
    """The 16 affordance tag bits per candidate, as 0/1 floats (LSB first)."""

    start, stop = BLOCK_TAG_COLUMNS
    return candidates.values[..., start:stop]


def block_gather_types(candidates: TokenSet) -> jax.Array:
    """The 8 gather-type bits per candidate, as 0/1 floats (LSB first)."""

    start, stop = BLOCK_GATHER_COLUMNS
    return candidates.values[..., start:stop]


def block_tool_quality(candidates: TokenSet) -> jax.Array:
    """Required tool quality per candidate."""

    return candidates.values[..., BLOCK_TOOL_QUALITY_COLUMN]


# -- crafting -----------------------------------------------------------------


class RecipeView(NamedTuple):
    """Crafting candidates with their inputs, outputs, and gating.

    Unlike the combat groups these are mostly integer ids and counts, not
    normalized floats: they are meant to be embedded or looked up, not fed
    straight into a dense layer.
    """

    available: jax.Array
    mask: jax.Array
    input_item_id: jax.Array
    input_quantity: jax.Array
    input_mask: jax.Array
    output_item_id: jax.Array
    output_quantity: jax.Array
    output_mask: jax.Array
    bench_type: jax.Array
    tier_level: jax.Array
    requirement_mask: jax.Array
    knowledge_required: jax.Array
    required_memories_level: jax.Array
    time_seconds: jax.Array

    def count(self) -> jax.Array:
        return jnp.sum(self.mask.astype(jnp.int32), axis=-1)


def recipe_candidates(actor_input: Any) -> RecipeView:
    """Named view over the crafting candidates the action surface publishes."""

    view = _require_action_surface(actor_input).recipe_candidates
    policy = view.policy
    return RecipeView(
        available=view.available,
        mask=view.candidate_mask,
        input_item_id=policy.input_item_id,
        input_quantity=policy.input_quantity,
        input_mask=policy.input_mask,
        output_item_id=policy.output_item_id,
        output_quantity=policy.output_quantity,
        output_mask=policy.output_mask,
        bench_type=policy.requirement_bench_type,
        tier_level=policy.requirement_tier_level,
        requirement_mask=policy.requirement_mask,
        knowledge_required=policy.knowledge_required,
        required_memories_level=policy.required_memories_level,
        time_seconds=policy.time_seconds,
    )


def recipe_embedding(actor_input: Any) -> TokenSet:
    """The precomputed recipe embedding, if the scene configured an encoder.

    Requires ``recipe_candidate_encoder_params`` on the scene; without it the
    rows are present but unavailable.
    """

    encoding = _require_action_surface(actor_input).recipe_encoding
    return TokenSet(values=encoding.candidate_embedding, mask=encoding.candidate_mask)


# -- light --------------------------------------------------------------------

#: Column names published by the Gym, so these are exact rather than derived.
LIGHT_FEATURES: tuple[str, ...] = tuple(ACTOR_LIGHT_POLICY_FEATURES)
_LIGHT_INDEX = {name: index for index, name in enumerate(LIGHT_FEATURES)}

if len(LIGHT_FEATURES) != ACTOR_LIGHT_POLICY_FEATURE_SIZE:
    raise RuntimeError("actor light policy feature list disagrees with its size")


def _raw_light_tokens(state: Any) -> Any:
    """The upstream ``ActorLightPolicyTokens``, which also carries ``available``."""

    environment = getattr(state, "environment", state)
    return getattr(environment, "light_tokens", None)


def light_tokens(state: Any) -> TokenSet | None:
    """Per-geometry-token lighting, or ``None`` when the scene has no light.

    Light is carried on the environment state
    (``ArsenalPPOEnvironmentState.light_tokens``), not on the observation --
    searching ``legal_observation`` for it finds nothing.

    The returned pair drops the upstream per-row ``available`` flag; use
    :func:`describe_availability` when you need to know whether the scene
    produced light at all, rather than inferring it from an all-false mask.
    """

    tokens = _raw_light_tokens(state)
    if tokens is None:
        return None
    return TokenSet(values=tokens.token_f32, mask=tokens.token_mask)


def _light_column(tokens: TokenSet, name: str) -> jax.Array:
    return tokens.values[..., _LIGHT_INDEX[name]]


#: Both RGB triples are contiguous in the published layout, so they are views
#: rather than stacks.  Resolved once at import; a reorder upstream turns these
#: into a loud error instead of a silent channel swap.
_BLOCK_RGB_SPAN = field_span(
    "light_token_f32", ("block_light_red", "block_light_green", "block_light_blue")
)
_TINT_RGB_SPAN = field_span(
    "light_token_f32", ("tint_red", "tint_green", "tint_blue")
)


def light_rgb(tokens: TokenSet) -> jax.Array:
    """Block-light RGB per token, ``(..., slots, 3)``, normalized to [0, 1].

    The Gym divides raw 0-15 light levels by 15.0, so these are already scaled.
    O(1): a contiguous slice, not a stack.
    """

    return tokens.values[..., _BLOCK_RGB_SPAN]


def light_tint(tokens: TokenSet) -> jax.Array:
    """Tint RGB per token, ``(..., slots, 3)``, normalized to [0, 1] (raw /255).

    O(1): a contiguous slice, not a stack.
    """

    return tokens.values[..., _TINT_RGB_SPAN]


def sky_light(tokens: TokenSet) -> jax.Array:
    """Sky light per token, normalized to [0, 1]."""

    return _light_column(tokens, "sky_light")


def light_validity(tokens: TokenSet) -> dict[str, jax.Array]:
    """The three per-token validity flags, which gate the values above."""

    return {
        name: _light_column(tokens, name)
        for name in ("sky_valid", "block_light_rgb_valid", "tint_rgb_valid")
    }


# -- inventory ----------------------------------------------------------------

#: Column names published by the Gym.
INVENTORY_FEATURES: tuple[str, ...] = tuple(INVENTORY_POLICY_FLOAT_FEATURES)
_INVENTORY_INDEX = {name: index for index, name in enumerate(INVENTORY_FEATURES)}

if len(INVENTORY_FEATURES) != INVENTORY_POLICY_FLOAT_SIZE:
    raise RuntimeError("inventory policy feature list disagrees with its size")

#: Divisors applied by ``encode_inventory_policy_tokens``
#: (``inventory_tokens.py:295-313``); the decoders below invert them.
_ITEM_ID_LOW_SCALE = 0xFFFF
_ITEM_ID_HIGH_SCALE = 0x7FFF
_QUANTITY_LOG_SCALE = 31.0


def inventory_tokens(source: Any) -> TokenSet | None:
    """Inventory slots from anything carrying ``InventoryPolicyTokens``.

    Accepts the tokens themselves, or a native assembly/evidence object that
    holds them.  Returns ``None`` when the source has no inventory, which is
    the normal case in the JAX lane: no built-in scene produces one, and the
    surface is assembled from a live server frame.
    """

    tokens = source
    for attribute in ("inventory_tokens", "evidence"):
        candidate = getattr(tokens, attribute, None)
        if candidate is not None:
            tokens = candidate
    values = getattr(tokens, "token_f32", None)
    mask = getattr(tokens, "token_mask", None)
    if values is None or mask is None:
        return None
    return TokenSet(values=values, mask=mask)


def _inventory_column(tokens: TokenSet, name: str) -> jax.Array:
    return tokens.values[..., _INVENTORY_INDEX[name]]


def decode_item_ids(tokens: TokenSet) -> jax.Array:
    """Reassemble int32 item ids from their split, normalized halves.

    The encoder stores ``item_id & 0xFFFF`` and ``(item_id >> 16) & 0x7FFF`` as
    separate normalized floats, so neither column is usable on its own.
    """

    low = jnp.round(
        _inventory_column(tokens, "item_id_low_16") * _ITEM_ID_LOW_SCALE
    ).astype(jnp.int32)
    high = jnp.round(
        _inventory_column(tokens, "item_id_high_15") * _ITEM_ID_HIGH_SCALE
    ).astype(jnp.int32)
    return jnp.where(tokens.mask, low | (high << 16), -1)


def decode_quantities(tokens: TokenSet) -> jax.Array:
    """Invert the ``log2(quantity + 1) / 31`` encoding back to counts."""

    encoded = _inventory_column(tokens, "quantity_log2_fraction")
    counts = jnp.round(jnp.exp2(encoded * _QUANTITY_LOG_SCALE) - 1.0)
    return jnp.where(tokens.mask, counts.astype(jnp.int32), 0)


def decode_container_ids(tokens: TokenSet) -> jax.Array:
    """Invert ``container_id / (CONTAINER_COUNT - 1)``."""

    encoded = _inventory_column(tokens, "container_id")
    ids = jnp.round(encoded * (CONTAINER_COUNT - 1)).astype(jnp.int32)
    return jnp.where(tokens.mask, ids, -1)


def item_durability(tokens: TokenSet) -> jax.Array:
    """Durability fraction in [0, 1] per slot."""

    return _inventory_column(tokens, "durability_fraction")


__all__ = [
    "BLOCK_GATHER_COLUMNS",
    "BLOCK_POSITION_COLUMNS",
    "BLOCK_TAG_COLUMNS",
    "BLOCK_TOOL_QUALITY_COLUMN",
    "INVENTORY_FEATURES",
    "LIGHT_FEATURES",
    "RecipeView",
    "SurfaceStatus",
    "available_surfaces",
    "surface_flags",
    "block_affordance_tags",
    "block_candidates",
    "block_gather_types",
    "block_positions",
    "block_tool_quality",
    "decode_container_ids",
    "decode_item_ids",
    "decode_quantities",
    "describe_availability",
    "inventory_tokens",
    "item_durability",
    "light_rgb",
    "light_tint",
    "light_tokens",
    "light_validity",
    "recipe_candidates",
    "recipe_embedding",
    "sky_light",
]
