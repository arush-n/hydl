"""The v2 adapter: the legal observation as ``skill_tokens_v2`` token sets, and the registry lookups the content
features come from (``docs/PERCEPTION.md`` section 5, build step 3).

Like the v1 adapter (``model/observation_adapter.py``), ``legal_tokens_v2`` reads the learner observation and the
caller's body yaw, never the simulator state. Its output is ``{type: (features float32[B, N_t, F_t],
mask bool[B, N_t])}`` for ``encoder.trunk_types(TOKEN_SCHEMA_V2)``, the encoder's layout.

Filled today
------------
* ``self``: the v1 self token's features (the v1 adapter's), then the maximum health, ``holds`` and the held item's row
  at 0: the observation publishes neither the agent's maximum health nor its held item's id (``PERCEPTION_ASKS.md``,
  ask 7).
* ``entity``: every published nearby-entity row, up to the capacity (the observation carries 16; slots above that stay
  invalid until ask 3): the position in blocks and the velocity in the body frame, visible, last-seen age 0 (the
  observation has no memory), health, alive, its facing relative to the body (its world yaw, published as sine and
  cosine, minus the body yaw), its attack phase and progress. Its role is not published (ask 1), so ``known``, its
  attitude, ``player``, the health ratio, its kind's row and description stay 0, and what it holds is ``unknown``
  (ask 5).
* ``block_candidate``, ``inventory_slot``, ``geometry``, ``kit``: invalid. The observation carries no block candidates
  (ask 4) and no inventory (ask 2); geometry and kit stay as the v1 adapter leaves them.

Ready for the asks
------------------
``registry_tables`` puts the registry on the device keyed by semantic id (the gym's ``semantic_id``: 31-bit FNV-1a of
the asset id, the scheme the observation's item ids use); ``lookup`` maps ids to table rows inside ``jit``, row 0 being
the unknown row of zeros; ``tool_relations`` compares the agent's tools with block candidates. When an ask lands, the
adapter reads the new field and gathers its rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract.surface import (
    ATTACK_PHASES,
    BLOCK_PROPERTIES,
    CREATURE_PROPERTIES,
    DESCRIPTION_WIDTH,
    ITEM_PROPERTIES,
    TOKEN_SCHEMA_V2,
    EntityTokenV2,
    SelfTokenV2,
)
from experimental.neural_agent.skill_agent.model import observation_adapter
from experimental.neural_agent.skill_agent.model.encoder import TRUNK_TYPES, trunk_types
from experimental.neural_agent.skill_agent.perception.registry import Registry
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.observation.v1.schema.contract import (
    ENTITY_FLOAT_FEATURES,
    ENTITY_INTEGER_FEATURES,
    NEARBY_ENTITY_RADIUS_BLOCKS,
)
from hytalegym.jax.combat.observation.v3.schema.types import LearnerCombatObservationV3

_FLOAT = {name: index for index, name in enumerate(ENTITY_FLOAT_FEATURES)}
_INTEGER = {name: index for index, name in enumerate(ENTITY_INTEGER_FEATURES)}
_ENTITY = {name: index for index, name in enumerate(EntityTokenV2.FEATURES)}
_V1_SELF_ONLY = {name: 1 if name in ("entity", "self") else 0 for name, _ in TRUNK_TYPES}


class RegistryTables(NamedTuple):
    """The content registry on the device, keyed by semantic id. Row 0 of every table is the unknown row (zeros, no
    gather type, attitude -1); ``*_keys`` are the sorted semantic ids and ``*_rows`` the table row of each."""

    item_keys: jax.Array  # int32[I]
    item_rows: jax.Array  # int32[I]
    item_features: jax.Array  # float32[1 + I, len(ITEM_PROPERTIES)]
    item_descriptions: jax.Array  # float32[1 + I, DESCRIPTION_WIDTH]
    tool_power: jax.Array  # float32[1 + I, G]
    tool_quality: jax.Array  # float32[1 + I, G]
    block_keys: jax.Array  # int32[K]
    block_rows: jax.Array  # int32[K]
    block_features: jax.Array  # float32[1 + K, len(BLOCK_PROPERTIES)]
    block_descriptions: jax.Array  # float32[1 + K, DESCRIPTION_WIDTH]: the block's item description
    block_gather: jax.Array  # int32[1 + K]: a column of tool_power, or -1
    block_quality: jax.Array  # float32[1 + K]
    creature_keys: jax.Array  # int32[C]
    creature_rows: jax.Array  # int32[C]
    creature_features: jax.Array  # float32[1 + C, len(CREATURE_PROPERTIES)]
    creature_descriptions: jax.Array  # float32[1 + C, DESCRIPTION_WIDTH]
    attitude_to_player: jax.Array  # int32[1 + C]: an index into ATTITUDES, or -1


def _keyed(ids: tuple[str, ...], kind: str) -> tuple[np.ndarray, np.ndarray]:
    hashes = np.asarray([semantic_id(asset) for asset in ids], dtype=np.int64)
    if len(np.unique(hashes)) != len(hashes):
        first: dict[int, str] = {}
        clashes = []
        for value, asset in zip(hashes.tolist(), ids):
            if value in first:
                clashes.append((first[value], asset))
            first.setdefault(value, asset)
        raise ValueError(f"{kind} ids collide under semantic_id: {clashes[:5]}")
    order = np.argsort(hashes, kind="stable")
    return hashes[order].astype(np.int32), (order + 1).astype(np.int32)


def _padded(array: np.ndarray, fill: float = 0.0) -> jax.Array:
    pad = np.full((1, *array.shape[1:]), fill, dtype=array.dtype)
    return jnp.asarray(np.concatenate([pad, array]))


def registry_tables(registry: Registry) -> RegistryTables:
    """The tables ``lookup`` and ``tool_relations`` read. Refuses a registry whose property rows or description width
    differ from those ``skill_tokens_v2`` pins, or whose ids collide under ``semantic_id``."""

    manifest = registry.manifest
    pinned = {"item_features": ITEM_PROPERTIES, "block_features": BLOCK_PROPERTIES,
              "creature_features": CREATURE_PROPERTIES}
    for key, names in pinned.items():
        if tuple(manifest[key]) != names:
            raise ValueError(f"the registry's {key} differ from {TOKEN_SCHEMA_V2}'s")
    if manifest["description_width"] != DESCRIPTION_WIDTH:
        raise ValueError(f"the registry's description width {manifest['description_width']} is not "
                         f"{TOKEN_SCHEMA_V2}'s {DESCRIPTION_WIDTH}")
    item_keys, item_rows = _keyed(registry.item_ids, "item")
    block_keys, block_rows = _keyed(registry.block_ids, "block")
    creature_keys, creature_rows = _keyed(registry.creature_ids, "creature")
    item_index = {item: row for row, item in enumerate(registry.item_ids)}
    block_items = np.asarray([item_index[block] for block in registry.block_ids], dtype=np.int64)
    return RegistryTables(
        item_keys=jnp.asarray(item_keys), item_rows=jnp.asarray(item_rows),
        item_features=_padded(registry.item_features), item_descriptions=_padded(registry.item_descriptions),
        tool_power=_padded(registry.tool_power), tool_quality=_padded(registry.tool_quality),
        block_keys=jnp.asarray(block_keys), block_rows=jnp.asarray(block_rows),
        block_features=_padded(registry.block_features),
        block_descriptions=_padded(registry.item_descriptions[block_items]),
        block_gather=_padded(registry.block_gather.astype(np.int32), -1),
        block_quality=_padded(registry.block_quality),
        creature_keys=jnp.asarray(creature_keys), creature_rows=jnp.asarray(creature_rows),
        creature_features=_padded(registry.creature_features),
        creature_descriptions=_padded(registry.creature_descriptions),
        attitude_to_player=_padded(registry.attitude_to_player.astype(np.int32), -1),
    )


def lookup(keys: jax.Array, rows: jax.Array, ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The table row of each semantic id of ``ids`` (any shape) and whether the registry holds it; row 0, the unknown
    row, for id 0 or an id the registry does not hold."""

    ids = jnp.asarray(ids, dtype=jnp.int32)
    if keys.shape[0] == 0:
        return jnp.zeros(ids.shape, jnp.int32), jnp.zeros(ids.shape, jnp.bool_)
    position = jnp.clip(jnp.searchsorted(keys, ids), 0, keys.shape[0] - 1)
    found = (keys[position] == ids) & (ids != 0)
    return jnp.where(found, rows[position], 0), found


def tool_relations(tables: RegistryTables, tool_rows: jax.Array, tool_present: jax.Array, held_row: jax.Array,
                   block_rows: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """How the agent's tools fare against each block candidate: ``BlockCandidateTokenV2``'s ``best_power``,
    ``best_quality_margin``, ``held_power`` and ``held_quality_margin``, each float32[B, K].

    Args:
        tables: From ``registry_tables``.
        tool_rows: int32[B, S], the item rows of the agent's carried items; ``tool_present`` bool[B, S] marks real ones.
        held_row: int32[B], the held item's row (0 for none).
        block_rows: int32[B, K], the candidates' block rows.

    A tool's power against a block is its power for the block's gather type; the best carried tool is the one with
    the highest. A margin is that tool's quality for the gather type minus the quality the block asks, over 5 and
    clipped to [-1, 1]; it is -1 when the tool has no power against the block.
    """

    gather = tables.block_gather[block_rows]
    column = jnp.maximum(gather, 0)
    required = tables.block_quality[block_rows]
    applies = gather >= 0
    power = tables.tool_power[tool_rows[:, :, None], column[:, None, :]]
    quality = tables.tool_quality[tool_rows[:, :, None], column[:, None, :]]
    power = jnp.where(tool_present[:, :, None] & applies[:, None, :], power, 0.0)
    best = jnp.argmax(power, axis=1)
    best_power = jnp.max(power, axis=1)
    best_quality = jnp.take_along_axis(quality, best[:, None, :], axis=1)[:, 0, :]
    held_power = jnp.where(applies, tables.tool_power[held_row[:, None], column], 0.0)
    held_quality = tables.tool_quality[held_row[:, None], column]

    def margin(tool_quality: jax.Array, tool_power: jax.Array) -> jax.Array:
        return jnp.where(tool_power > 0, jnp.clip((tool_quality - required) / 5.0, -1.0, 1.0), -1.0)

    return best_power, margin(best_quality, best_power), held_power, margin(held_quality, held_power)


def legal_tokens_v2(
    observation: LearnerCombatObservationV3,
    *,
    capacities: Mapping[str, int],
    body_yaw_degrees: jax.Array,
) -> dict[str, tuple[jax.Array, jax.Array]]:
    """Build the ``skill_tokens_v2`` token sets of B agents from their legal observation.

    Args:
        observation: The learner observation of ``ArsenalEnvironment.step``. It is never the simulator state.
        capacities: Slots per token type; must name every type of ``trunk_types(TOKEN_SCHEMA_V2)``. ``entity`` and
            ``self`` need at least one slot.
        body_yaw_degrees: float32[B], each agent's body yaw in degrees (the observation does not publish it).

    Returns:
        ``{type: (features float32[B, N_t, F_t], mask bool[B, N_t])}`` in the encoder's layout.

    Raises:
        ValueError: If ``entity`` or ``self`` has no slot, or ``body_yaw_degrees`` is not shaped (B,).
    """

    for name in ("entity", "self"):
        if capacities[name] < 1:
            raise ValueError(f"capacities[{name!r}] must be at least 1, got {capacities[name]}")
    v1 = observation_adapter.legal_tokens(observation, capacities=_V1_SELF_ONLY, body_yaw_degrees=body_yaw_degrees)
    v1_self = v1["self"][0][:, 0, :]
    batch = v1_self.shape[0]
    tokens: dict[str, tuple[jax.Array, jax.Array]] = {}
    for name, cls in trunk_types(TOKEN_SCHEMA_V2):
        capacity = capacities[name]
        features = jnp.zeros((batch, capacity, len(cls.FEATURES)), dtype=jnp.float32)
        mask = jnp.zeros((batch, capacity), dtype=jnp.bool_)
        if name == "self":
            # v1's features but its last (marked), then the maximum health, holds and the held row, all unpublished.
            features = features.at[:, 0, :v1_self.shape[1] - 1].set(v1_self[:, :-1])
            features = features.at[:, 0, SelfTokenV2.FEATURES.index("marked")].set(v1_self[:, -1])
            mask = mask.at[:, 0].set(True)
        elif name == "entity":
            count = min(capacity, observation.base.entity_f32.shape[1])
            rows, present = _entity_rows(observation.base, jnp.asarray(body_yaw_degrees, jnp.float32), count)
            features = features.at[:, :count, :].set(rows)
            mask = mask.at[:, :count].set(present)
        tokens[name] = (features, mask)
    return tokens


def _entity_rows(base, body_yaw_degrees: jax.Array, count: int) -> tuple[jax.Array, jax.Array]:
    """The first ``count`` nearby-entity rows as ``EntityTokenV2`` features [B, count, F] and their presence."""

    row = base.entity_f32[:, :count, :]
    present = base.entity_mask[:, :count].astype(jnp.bool_)
    world_yaw = jnp.arctan2(row[..., _FLOAT["yaw_sin"]], row[..., _FLOAT["yaw_cos"]])
    facing = world_yaw - jnp.deg2rad(body_yaw_degrees)[:, None]
    phase = jax.nn.one_hot(base.entity_i32[:, :count, _INTEGER["attack_phase"]], len(ATTACK_PHASES))
    columns = {
        "forward": row[..., _FLOAT["relative_forward"]] * NEARBY_ENTITY_RADIUS_BLOCKS,
        "right": row[..., _FLOAT["relative_right"]] * NEARBY_ENTITY_RADIUS_BLOCKS,
        "up": row[..., _FLOAT["relative_up"]] * NEARBY_ENTITY_RADIUS_BLOCKS,
        "v_forward": row[..., _FLOAT["relative_velocity_forward"]],
        "v_right": row[..., _FLOAT["relative_velocity_right"]],
        "v_up": row[..., _FLOAT["relative_velocity_up"]],
        "visible": row[..., _FLOAT["visible"]],
        "health_fraction": jnp.clip(row[..., _FLOAT["health_fraction"]], 0.0, 1.0),
        "alive": row[..., _FLOAT["alive"]],
        "yaw_sin": jnp.sin(facing),
        "yaw_cos": jnp.cos(facing),
        "attack_progress": jnp.clip(row[..., _FLOAT["attack_progress"]], 0.0, 1.0),
        "held_unknown": jnp.ones(present.shape, jnp.float32),  # not published (ask 5)
        **{f"phase_{name}": phase[..., index] for index, name in enumerate(ATTACK_PHASES)},
    }
    features = jnp.zeros((*present.shape, len(EntityTokenV2.FEATURES)), dtype=jnp.float32)
    for name, value in columns.items():
        features = features.at[..., _ENTITY[name]].set(value)
    return jnp.where(present[..., None], features, 0.0), present


__all__ = [
    "RegistryTables",
    "legal_tokens_v2",
    "lookup",
    "registry_tables",
    "tool_relations",
]
