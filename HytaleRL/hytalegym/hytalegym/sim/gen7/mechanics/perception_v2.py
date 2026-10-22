"""Gen-7 perception channel ``perception_v2`` (``sim/gen7/PERCEPTION_V2.md``). Switch ``Constructs.perception_v2``, default False.

When the switch is on, the render's hook on ``encode_learner_observation_v3`` returns the v3 observation with one more
field, ``perception_v2``: the target's kind (NPC or player), its role id (the NPC role asset's semantic id, from the config's
``entity_role_semantic_ids``), its held weapon (a semantic id), the agent's inventory tokens, its body yaw and its own
world-frame position. Every entity field is blanked where the observation's entity row is blanked: the row is the same gate
the learner already applies (target perception, line of sight with the detail-hitbox rule, and the opt-in view sector and
sensor range, which reach ``line_of_sight``), so the channel adds no perception of its own. The v3 fields are passed through
unchanged. The role id is 0 for a player-backed target (a player has no NPC role asset) and for an empty role name.
"""

from __future__ import annotations

import importlib
import inspect
from functools import lru_cache, wraps
from typing import Any, NamedTuple

import jax.numpy as jnp

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.observation.v1.schema.contract import ENTITY_KIND_NPC
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY

SWITCH = "perception_v2"
IMPLEMENTED = True

#: Row capacity of the entity arrays; the render uses it. The channel builder takes any capacity >= 1.
ENTITY_CAPACITY = 16
#: Kinds: 0 is no entity (padding or blanked row), NPC as in the v1 contract, PLAYER beside it (channel-local).
ENTITY_KIND_NONE = 0
ENTITY_KIND_PLAYER = 2


class PerceptionV2(NamedTuple):
    """The channel's arrays (``PERCEPTION_V2.md`` section 2). Entity arrays are [B, C]; row 0 is the target."""

    entity_kind: Any  # int32 [B, C]
    entity_role_id: Any  # int32 [B, C]: the target's NPC role asset (semantic id); 0 for a player or no role
    entity_held_item_id: Any  # int32 [B, C]
    inventory_i32: Any  # int32 [B, S, 3]: container_id, container_slot, item_id; 0 where the token is absent
    inventory_f32: Any  # float32 [B, S, 7]: the v3 inventory token features, unchanged
    inventory_mask: Any  # bool [B, S]
    body_yaw_sin: Any  # float32 [B]
    body_yaw_cos: Any  # float32 [B]
    self_position: Any  # float32 [B, 3]: the agent's own world-frame position (ruling 2026-10-19, PERCEPTION_V2.md section 2)


def _rendered_root(original) -> str:
    """The rendered package (``hytalegym.sim.genN.engine``) that ``original`` was rendered into."""

    module = inspect.unwrap(original).__module__
    root, marker, _ = module.partition(".combat.")
    if not marker:
        raise ValueError(f"{module} is not a rendered combat module")
    return root


def _actor_inventory_source(inventory_module, inventory, layout, valid):
    """The agent's inventory source, built the way ``inventory_policy_tokens_from_state`` builds it
    (``combat/observation/v3/tokens/inventory.py``, the body of that function): the same arguments, so the floats
    the channel publishes are that function's floats (checked by ``tests/sim/test_perception_v2.py``)."""

    m = inventory_module
    m._validate_source_layout(layout)
    batch = inventory.item_id.shape[0]
    item_id = inventory.item_id[:, :1]
    quantity = inventory.quantity[:, :1]
    durability = inventory.durability[:, :1]
    maximum = inventory.max_durability[:, :1]
    metadata = inventory.metadata_hash[:, :1]
    present = (item_id >= 0) & (quantity > 0)
    metadata_present = jnp.any(metadata != 0, axis=3) & present
    container_available = jnp.ones((batch, 1, m.CONTAINER_COUNT), dtype=jnp.bool_)
    container_capacity = jnp.broadcast_to(layout.capacities[None, None, :], (batch, 1, m.CONTAINER_COUNT))
    active_slots = jnp.full((batch, 1, m.CONTAINER_COUNT), -1, dtype=jnp.int32)
    active_slots = active_slots.at[:, 0, 2].set(inventory.active_hotbar_slot[:, 0])
    active_slots = active_slots.at[:, 0, 3].set(inventory.active_utility_slot[:, 0])
    active_slots = active_slots.at[:, 0, 4].set(inventory.active_tools_slot[:, 0])
    return m.produce_actor_inventory_tokens(
        item_id,
        quantity,
        durability,
        maximum,
        metadata,
        metadata_present,
        present,
        slot_container_id=layout.container_id,
        slot_container_index=layout.container_slot,
        source_available=(inventory.failure_bits == 0)[:, None],
        container_available=container_available,
        container_capacity=container_capacity,
        active_container_slot=active_slots,
        actor_legal=valid[:, None],
        token_capacity=m.INVENTORY_POLICY_TOKEN_CAPACITY,
        provenance=m.INVENTORY_TOKEN_PROVENANCE_SIMULATED,
        required_container_mask=m._NO_REQUIRED_CONTAINERS,
    )


def channel(root: str, state, config, observation, *, entity_capacity: int = ENTITY_CAPACITY) -> PerceptionV2:
    """The channel of one batch, from the render's inputs (the same ones the v3 encoder takes, plus its output).

    Args:
        root: The rendered package, e.g. ``hytalegym.sim.gen7.engine``.
        state: ``ArsenalEnvironmentState`` (privileged; read only for the agent's inventory and body yaw).
        config: ``ArsenalRuntimeConfig``: the loadout's weapon ids, the player-backed mask, the inventory layout.
        observation: The v3 observation the encoder returned for the same inputs; its ``base.entity_mask`` is the gate.
        entity_capacity: Rows of the entity arrays; row 0 is the target, the rest are padding (id 0).
    """

    if isinstance(entity_capacity, bool) or not isinstance(entity_capacity, int) or entity_capacity < 1:
        raise ValueError(f"entity_capacity must be an int >= 1, got {entity_capacity!r}")
    inventory_module = importlib.import_module(f"{root}.combat.observation.v3.tokens.inventory")
    source = _actor_inventory_source(inventory_module, state.inventory, config.inventory_layout, observation.valid)
    tokens = inventory_module.encode_inventory_policy_tokens(source)
    ids = jnp.stack(
        (source.container_id[:, 0], source.container_slot[:, 0], source.item_id[:, 0]),
        axis=-1,
    ).astype(jnp.int32)
    inventory_i32 = jnp.where(tokens.token_mask[..., None], ids, 0)

    present = observation.base.entity_mask[:, 0].astype(jnp.bool_)
    player = config.player_backed_actor_mask[:, TARGET_ENTITY].astype(jnp.bool_)
    kind = jnp.where(present, jnp.where(player, ENTITY_KIND_PLAYER, ENTITY_KIND_NPC), ENTITY_KIND_NONE)
    # The loadout's no-weapon sentinel is -1; the channel publishes semantic ids or 0 (PERCEPTION_V2.md section 4),
    # so a non-positive weapon id reads as nothing held (reported by the skill layer on the open-flat arena).
    weapon = config.loadout.weapon_id[:, TARGET_ENTITY]
    held = jnp.where(present & (weapon > 0), weapon, 0)
    batch = present.shape[0]

    def rows(values):
        return jnp.zeros((batch, entity_capacity), jnp.int32).at[:, 0].set(values.astype(jnp.int32))

    # The config's role ids (construct_roles) give each entity's role asset; the channel publishes the target's, and an NPC
    # role only: a player-backed target has no NPC role asset, so the field is 0 for it (as it is for a hidden row).
    role_ids = config.entity_role_semantic_ids
    target_role = (
        jnp.zeros((batch,), jnp.int32)
        if role_ids is None
        else jnp.asarray(role_ids, dtype=jnp.int32)[:, TARGET_ENTITY]
    )
    role = jnp.where(present & ~player, target_role, 0)
    yaw = jnp.deg2rad(state.combat.yaw[:, AGENT_ENTITY].astype(jnp.float32))
    return PerceptionV2(
        entity_kind=rows(kind),
        entity_role_id=rows(role),
        entity_held_item_id=rows(held),
        inventory_i32=inventory_i32.astype(jnp.int32),
        inventory_f32=tokens.token_f32,
        inventory_mask=tokens.token_mask,
        body_yaw_sin=jnp.sin(yaw),
        body_yaw_cos=jnp.cos(yaw),
        self_position=state.combat.position[:, AGENT_ENTITY].astype(jnp.float32),
    )


@lru_cache(maxsize=None)
def observation_type(base: type) -> type:
    """``base`` with one more field, ``perception_v2``, last. One class per v3 observation class (the fields of the rendered
    class are read, so the layout follows the render)."""

    fields = [(name, Any) for name in base._fields] + [("perception_v2", Any)]
    return NamedTuple(f"{base.__name__}WithPerceptionV2", fields)


def construct_roles(original, profile):
    """Hook for ``arsenal_runtime_config``: the config's semantic role ids (``entity_role_semantic_ids``, int32 [B, E]).

    The names are the caller's ``entity_role_ids`` (the duel's ``roles=``), or the matchup's pair when none is passed
    (``default_combat_role_ids``, the names the status builder already reads). A name is ``semantic_id(name)``; an empty
    name (no role, or an entity past the roles) is 0. The hook is applied after the other config hooks, so it reads the names
    before the hit-interrupt hook replaces its measured roles with "". Switch off: ``original`` is returned.
    """

    value = profile.constructs.perception_v2
    if value is False:
        return original
    if value is not True:
        raise ValueError(f"perception_v2 must be a bool, got {value!r}")

    @wraps(original)
    def arsenal_runtime_config(loadout: Any, *args: Any, **kwargs: Any) -> Any:
        config = original(loadout, *args, **kwargs)
        batch, entities = config.loadout.weapon_id.shape
        names = kwargs.get("entity_role_ids")
        if names is None:
            roles = importlib.import_module(_rendered_root(original) + ".combat.mechanics.sources.role_statuses")
            names = roles.default_combat_role_ids(entities)
        ids = jnp.asarray([semantic_id(name) if name else 0 for name in names], dtype=jnp.int32)
        return config._replace(entity_role_semantic_ids=jnp.broadcast_to(ids[None, :], (batch, entities)))

    return arsenal_runtime_config


def construct(original, profile):
    """Return ``original`` when the switch is off; the encoder with the channel when it is on (the render's hook)."""

    value = profile.constructs.perception_v2
    if value is False:
        return original
    if value is not True:
        raise ValueError(f"perception_v2 must be a bool, got {value!r}")

    def encode_with_channel(state, params, world_features, world_capabilities, config, *, world_geometry=None):
        observation = original(
            state, params, world_features, world_capabilities, config, world_geometry=world_geometry
        )
        value_ = channel(_rendered_root(original), state, config, observation)
        return observation_type(type(observation))(*observation, value_)

    return encode_with_channel
