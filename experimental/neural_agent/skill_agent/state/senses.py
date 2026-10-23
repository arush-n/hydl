"""The senses: the agent's legal observation in, this tick's ``skill_tokens_v3`` rows out, stateless (the state
layer's input, ``docs/VIRTUAL_HUMAN.md`` section 3; ``docs/PERCEPTION.md``
section 10; owner, 2026-10-20 night: the agent perceives the world, knows everything plausible about itself, and
reads and remembers space).

What it reads, and nothing else:

- the learner observation (``LearnerCombatObservationV3``), with the gen-7 ``perception_v2`` channel when present;
- the agent's legal per-actor views the actor input carries beside it: the action surface (block and recipe
  candidates) and the light rows aligned with the geometry tokens;
- the registry tables (``perception/registry.py``, a constant) and the running engine's compiled loadouts (a constant:
  what the agent knows of its own weapon);
- the neural layer's own carries: the action history (``state/action_history.py``) and the spatial memory
  (``state/memory.py``), which add the ``act`` tokens and the remembered rows.

It never reads the simulator's state. Every spatial row goes through ``state/frames.py``: rows the gym publishes
in world axes (projectiles, hazards, geometry, block candidates) are rotated into the body frame first, and every
spatial token gets the same ``where`` block.

Legality. A row the observation publishes without a perception gate is gated here where it can be: projectiles and
hazards keep only what lies in the view sector (``view_half_angle_degrees`` either side of the head's yaw). The
target's ability and weapon rows, which the gym does not gate by perceptibility, are not read at all. Both stop once
the simulator gates them (the coordinator's S7).

Seams. Fields the simulator does not publish yet read zero, with ``known`` 0: creature role, kind and attitude
(S2), other entities beyond row 0 (S3, they fill as soon as the rows are published), held items (S4), block ids (S5),
and the agent's maximum health. A status effect's id is read through the effects registry (``effects=``,
``perception/effects.py``): which effect it is, with ``effect_known`` 0 for an id the registry does not hold; without
the registry its effect columns read zero.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract import tokens_v3 as v3
from experimental.neural_agent.skill_agent.contract.surface import ATTACK_PHASES, ATTITUDES
from experimental.neural_agent.skill_agent.perception import adapter as v2
from experimental.neural_agent.skill_agent.state.action_history import ActionHistory, act_features
from experimental.neural_agent.skill_agent.state.frames import POSITION_SCALE_BLOCKS, where, world_to_body
from experimental.neural_agent.skill_agent.tasks.simple_tasks import AGENT_EYE_HEIGHT
from hytalegym.jax.combat import hytale_0_5_7_loadouts
from hytalegym.jax.combat.arsenal.profiles.hytale_0_5_7 import CORE_PROFILE_NAMES
from hytalegym.jax.combat.arsenal.schema import contract as arsenal_contract
from hytalegym.jax.combat.observation.v1.schema.contract import (
    ENTITY_FLOAT_FEATURES,
    ENTITY_INTEGER_FEATURES,
    HAZARD_FLOAT_FEATURES,
    HAZARD_RADIUS_BLOCKS,
    NEARBY_ENTITY_RADIUS_BLOCKS,
    PROJECTILE_FLOAT_FEATURES,
    PROJECTILE_RADIUS_BLOCKS,
    SELF_FLOAT_FEATURES,
)
from hytalegym.jax.combat.observation.v3.schema.contract import (
    ABILITY_FLOAT_FEATURES,
    ABILITY_INTEGER_FEATURES,
    STATUS_INTEGER_FEATURES,
    WEAPON_INTEGER_FEATURES,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    WORLD_TOKEN_BASE_FLOAT_SIZE,
    WORLD_TOKEN_DEFAULT_MAXIMUM_DISTANCE,
    WORLD_TOKEN_EDGE_FLOAT_SIZE,
)
from hytalegym.jax.combat.types import AGENT_ENTITY

if TYPE_CHECKING:  # the annotation only: the registry imports the perception adapter, which imports the encoder
    from experimental.neural_agent.skill_agent.perception.effects import EffectTables

VIEW_HALF_ANGLE_DEGREES: float = 60.0
"""Half the 120-degree view sector the neural agent's arenas use, centred on the head's yaw."""
ENTITY_CENTRE_HEIGHT: float = 0.9
"""An entity's reference point above its feet, for its elevation from the eye (a Trork is about 1.8 tall)."""
ASSUMED_PROJECTILE_TERMINAL: float = 50.0
"""Blocks a second. The observation divides a projectile's velocity by its own terminal velocity, which it does not
publish; the closest approach's distance does not depend on it, its time does, and is computed with this (the iron
shortbow's terminal velocity on both lines)."""
APPROACH_HORIZON_SECONDS: float = 2.0
APPROACH_DISTANCE_BLOCKS: float = 4.0

_SELF = {name: index for index, name in enumerate(SELF_FLOAT_FEATURES)}
_ENTITY = {name: index for index, name in enumerate(ENTITY_FLOAT_FEATURES)}
_ENTITY_INT = {name: index for index, name in enumerate(ENTITY_INTEGER_FEATURES)}
_PROJECTILE = {name: index for index, name in enumerate(PROJECTILE_FLOAT_FEATURES)}
_HAZARD = {name: index for index, name in enumerate(HAZARD_FLOAT_FEATURES)}
_ABILITY = {name: index for index, name in enumerate(ABILITY_FLOAT_FEATURES)}
_ABILITY_ID = ABILITY_INTEGER_FEATURES.index("ability_id")
_WEAPON_ID = WEAPON_INTEGER_FEATURES.index("weapon_id")
_COLUMNS = {name: {feature: index for index, feature in enumerate(cls.FEATURES)}
            for name, cls in v3.TOKEN_TYPES_V3.items()}
_WHERE = len(v3.WHERE_FEATURES)


class Light(NamedTuple):
    """The light rows aligned with the observation's geometry tokens: float32[B, G, 10] and their mask bool[B, G]
    (the gym's ``ActorLightPolicyTokens``)."""

    token_f32: jax.Array
    token_mask: jax.Array


class Remembered(NamedTuple):
    """The spatial memory's rows for this tick (``state/memory.py``), already in the v3 layout and the current
    body frame: per remembered type, features float32[B, M_t, F_t] and mask bool[B, M_t]."""

    entity: tuple[jax.Array, jax.Array]
    geometry: tuple[jax.Array, jax.Array]
    block_candidate: tuple[jax.Array, jax.Array]


class Pose(NamedTuple):
    """The agent's own pose this tick, which every spatial row is read against: the body's yaw, the head's yaw less
    the body's and the head's pitch (radians), the cosine of the view sector's half angle, and its world position
    when the perception channel publishes it (None otherwise; the stored memory needs it)."""

    body_yaw: jax.Array  # [B]
    head_left: jax.Array  # [B]
    head_pitch: jax.Array  # [B], positive up
    view_cos: jax.Array  # []
    position: jax.Array | None = None  # [B, 3], world


class Perceived(NamedTuple):
    """One tick's perceived rows, unpacked: per type (features [B, N, F], mask [B, N]) or None, in the v3 layout;
    the pose; and the entity rows' ids int32[B, E] (the stored memory keys entities by them)."""

    rows: dict
    pose: Pose
    entity_ids: jax.Array


def perceive_rows(observation: Any, *, body_yaw_degrees: jax.Array | None = None,
                  tables: v2.RegistryTables | None = None, action_surface: Any = None,
                  block_reach_blocks: float | None = None, light: Light | None = None,
                  inventory_capacity: int = v3.CAPACITIES_V3["inventory_slot"],
                  view_half_angle_degrees: float = VIEW_HALF_ANGLE_DEGREES,
                  effects: EffectTables | None = None) -> Perceived:
    """This tick's perceived rows of every type but ``act`` (stateless; the arguments as :func:`legal_tokens_v3`'s)."""

    channel = getattr(observation, "perception_v2", None)
    if body_yaw_degrees is None:
        if channel is None:
            raise ValueError("body_yaw_degrees is required: the observation carries no perception_v2 channel")
        body_yaw = jnp.arctan2(channel.body_yaw_sin, channel.body_yaw_cos)
    else:
        body_yaw = jnp.deg2rad(jnp.asarray(body_yaw_degrees, jnp.float32))
    if action_surface is not None and block_reach_blocks is None:
        raise ValueError("block_reach_blocks is required with action_surface")
    base = observation.base
    head_left = _wrap(jnp.arctan2(base.self_f32[:, _SELF["yaw_sin"]], base.self_f32[:, _SELF["yaw_cos"]]) - body_yaw)
    head_pitch = base.self_f32[:, _SELF["pitch"]] * (jnp.pi / 2)
    position = None if channel is None else jnp.asarray(channel.self_position, jnp.float32)
    pose = Pose(body_yaw, head_left, head_pitch, jnp.cos(jnp.deg2rad(view_half_angle_degrees)), position)
    rows = {
        "self": _self_rows(observation, channel, pose, tables),
        "kit": _kit_rows(observation),
        "status": _status_rows(observation, effects),
        "entity": _entity_rows(observation, channel, pose, tables),
        "projectile": _projectile_rows(base, pose),
        "hazard": _hazard_rows(base, pose),
        "geometry": _geometry_rows(observation, light, pose),
        "block_candidate": _block_rows(action_surface, block_reach_blocks, pose),
        "inventory_slot": _inventory_rows(channel, tables, inventory_capacity),
        "recipe": _recipe_rows(action_surface),
        "act": None,
    }
    return Perceived(rows, pose, base.entity_i32[..., _ENTITY_INT["entity_id"]])


def pack(rows: Mapping[str, Any], batch: int, capacities: Mapping[str, int] = v3.CAPACITIES_V3
         ) -> dict[str, tuple[jax.Array, jax.Array]]:
    """Every type of ``TRUNK_TYPE_NAMES_V3`` packed into its capacity (valid rows first, in order; a missing type
    empty)."""

    return {name: _fit(rows.get(name), batch, capacities[name], len(v3.TOKEN_TYPES_V3[name].FEATURES))
            for name in v3.TRUNK_TYPE_NAMES_V3}


def legal_tokens_v3(observation: Any, *, body_yaw_degrees: jax.Array | None = None,
                    capacities: Mapping[str, int] = v3.CAPACITIES_V3, tables: v2.RegistryTables | None = None,
                    action_surface: Any = None, block_reach_blocks: float | None = None, light: Light | None = None,
                    actions: ActionHistory | None = None, remembered: Remembered | None = None,
                    view_half_angle_degrees: float = VIEW_HALF_ANGLE_DEGREES,
                    effects: EffectTables | None = None) -> dict[str, tuple[jax.Array, jax.Array]]:
    """Build the ``skill_tokens_v3`` token sets of B agents (one tick; ``state/layer.py`` runs the stateful
    tick with the memory).

    Args:
        observation: The learner observation (never the simulator state); its ``perception_v2`` channel, when
            present, fills the inventory, the body yaw, the agent's position and the entity content columns.
        body_yaw_degrees: float32[B], each body's yaw in degrees (the agent's own proprioception); None takes it from
            the perception channel.
        capacities: Slots per type (``tokens_v3.CAPACITIES_V3`` by default); perceived rows fill first and remembered
            rows the slots after them.
        tables: The registry rows by id (``adapter.registry_tables``); None leaves every registry column unknown.
        action_surface: The actor's ``ArsenalActionSurfaceEvidence`` (block and recipe candidates), or None.
        block_reach_blocks: The environment's block-candidate distance scale; required with ``action_surface``.
        light: The light rows aligned with the geometry tokens, or None.
        actions: The agent's action history; None leaves the ``act`` tokens invalid.
        remembered: The stored memory's rows for this tick, or None.
        view_half_angle_degrees: The view sector's half angle for the rows the gym does not gate.
        effects: The status-effect registry's tables (``perception/effects.effect_tables``), or None: the status token's
            effect columns then read zero.

    Returns:
        ``{type: (features float32[B, N_t, F_t], mask bool[B, N_t])}`` for every type of ``TRUNK_TYPE_NAMES_V3``.
    """

    perceived = perceive_rows(observation, body_yaw_degrees=body_yaw_degrees, tables=tables,
                              action_surface=action_surface, block_reach_blocks=block_reach_blocks, light=light,
                              inventory_capacity=capacities["inventory_slot"],
                              view_half_angle_degrees=view_half_angle_degrees, effects=effects)
    rows = dict(perceived.rows)
    if actions is not None:
        rows["act"] = (act_features(actions), jnp.ones(actions.last_choice.shape, jnp.bool_))
    if remembered is not None:
        for name in ("entity", "geometry", "block_candidate"):
            rows[name] = append_rows(rows[name], getattr(remembered, name))
    return pack(rows, observation.base.self_f32.shape[0], capacities)


def _wrap(angle: jax.Array) -> jax.Array:
    return jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi


def where_of(position: jax.Array, pose: Pose, reference_height: float) -> jax.Array:
    """The ``where`` block of body-frame positions float32[B, N, 3] (from the agent's feet), for a reference point
    ``reference_height`` above them (an entity's centre, a block's middle)."""

    return where(position, pose.head_left[:, None], pose.head_pitch[:, None],
                 jnp.float32(reference_height - AGENT_EYE_HEIGHT))


def _in_view(position: jax.Array, pose: Pose) -> jax.Array:
    """bool[B, N]: in the view sector around the head's yaw (the cosine of the head-frame bearing, horizontal)."""

    bearing = jnp.arctan2(position[..., 1], position[..., 0]) + pose.head_left[:, None]
    return jnp.cos(bearing) >= pose.view_cos


def _put(rows: jax.Array, columns: Mapping[str, jax.Array], type_name: str) -> jax.Array:
    """``rows`` float32[B, N, F] with ``columns`` (name -> [B, N] or [B, N, k] for a run of names starting there)."""

    index = _COLUMNS[type_name]
    for name, value in columns.items():
        value = jnp.asarray(value, jnp.float32)
        if value.ndim == rows.ndim - 1:
            rows = rows.at[..., index[name]].set(value)
        else:
            rows = rows.at[..., index[name]:index[name] + value.shape[-1]].set(value)
    return rows


def _zeros(batch: int, count: int, type_name: str) -> jax.Array:
    return jnp.zeros((batch, count, len(v3.TOKEN_TYPES_V3[type_name].FEATURES)), jnp.float32)


def _self_rows(observation, channel, pose: Pose, tables) -> tuple[jax.Array, jax.Array]:
    base = observation.base
    s = base.self_f32
    batch = s.shape[0]
    rows = _zeros(batch, 1, "self")
    # The self row's velocity is in the head's frame; turn it into the body's (the head is turned head_left).
    forward, right = s[:, _SELF["forward_velocity"]], s[:, _SELF["right_velocity"]]
    sin, cos = jnp.sin(-pose.head_left), jnp.cos(-pose.head_left)
    v_forward, v_right = forward * cos + right * sin, right * cos - forward * sin
    world = observation.actor_world_f32
    known = observation.actor_world_mask
    movement_mask = jnp.asarray(observation.movement_state_mask, jnp.bool_)
    if movement_mask.ndim == 1:  # one availability per agent
        movement_mask = jnp.broadcast_to(movement_mask[:, None], observation.movement_state_f32.shape)
    movement = observation.movement_state_f32 * movement_mask
    resources = observation.resource_f32[:, AGENT_ENTITY]
    resource_known = jnp.asarray(observation.resource_mask[:, AGENT_ENTITY], jnp.float32)
    columns = {
        "v_forward": v_forward, "v_right": v_right, "v_up": s[:, _SELF["vertical_velocity"]],
        "speed": jnp.hypot(forward, right), "grounded": s[:, _SELF["grounded"]],
        "fall_speed": s[:, _SELF["fall_speed"]], "applied_vertical_velocity": s[:, _SELF["applied_vertical_velocity"]],
        "motion_delta": s[:, _SELF["motion_delta"]],
        "in_fluid": world[:, 0] * known[:, 0], "feet_submerged": world[:, 1] * known[:, 1],
        "eyes_submerged": world[:, 2] * known[:, 1], "drop_support": world[:, 3] * known[:, 2],
        "drop_height": world[:, 4] * known[:, 2], "medium_known": known[:, 0], "submersion_known": known[:, 1],
        "drop_known": known[:, 2],
        "moving_idle": movement, "movement_known": movement_mask.any(axis=-1),
        "knockback_control_lock": s[:, _SELF["knockback_control_lock"]],
        "damage_recovery": s[:, _SELF["damage_recovery"]], "alive": s[:, _SELF["alive"]],
        "head_yaw_left_sin": jnp.sin(pose.head_left), "head_yaw_left_cos": jnp.cos(pose.head_left),
        "head_pitch_sin": jnp.sin(pose.head_pitch), "head_pitch_cos": jnp.cos(pose.head_pitch),
        "heading_sin": jnp.sin(pose.body_yaw), "heading_cos": jnp.cos(pose.body_yaw),
        "heading_known": jnp.ones(batch), "position_known": jnp.full(batch, channel is not None, jnp.float32),
        "health_fraction": s[:, _SELF["health_fraction"]],
        "resource_stamina": resources * resource_known, "resource_stamina_known": resource_known,
        "defense_guard_active": observation.defense_f32[:, AGENT_ENTITY],
        "attack_executing": s[:, _SELF["attack_executing"]], "attack_cooldown": s[:, _SELF["attack_cooldown"]],
    }
    if tables is not None:
        row, found = v2.lookup(tables.item_keys, tables.item_rows, observation.weapon_i32[:, AGENT_ENTITY, _WEAPON_ID])
        columns |= {"holds": found, "held_item_block": tables.item_features[row], "held_known": found,
                    "description_0": tables.item_descriptions[row]}
    rows = _put(rows[:, 0], columns, "self")[:, None]
    return rows, jnp.ones((batch, 1), jnp.bool_)


class _KitTable(NamedTuple):
    ids: np.ndarray  # int32[K]
    values: np.ndarray  # float32[K, 9]: hold_count, threshold 0, 1, indefinite, charged, projectile, speed, gravity, reach


_KIT_COLUMNS = ("hold_count", "hold_threshold_0", "hold_threshold_1", "hold_indefinite", "charged", "projectile",
                "projectile_speed", "projectile_gravity", "reach")


@cache
def _kit_table() -> _KitTable:
    """What the running engine's compiled loadouts say of each ability id, for every core profile: read once (so a
    gen-7 render's redirected thresholds and speeds are the ones read). The reach is the measured strike reach of a
    weapon's first ability (``runtime/scripted_bodies.MELEE_REACH_BLOCKS``, imported here to keep perception free of
    the runtime at import time)."""

    from experimental.neural_agent.skill_agent.runtime.scripted_bodies import MELEE_REACH_BLOCKS

    names = tuple(CORE_PROFILE_NAMES)
    with jax.ensure_compile_time_eval():
        loadout = hytale_0_5_7_loadouts(names)
    ids = np.asarray(loadout.ability_id)[:, 0]
    count = np.asarray(loadout.ability_hold_count)[:, 0]
    thresholds = np.asarray(loadout.ability_hold_threshold_seconds)[:, 0]
    children = np.asarray(loadout.ability_hold_child_slot)[:, 0]
    indefinite = np.asarray(loadout.ability_hold_allow_indefinite)[:, 0]
    events = np.asarray(loadout.event_f32)[:, 0]
    present = np.asarray(loadout.event_mask)[:, 0]
    table: dict[int, list[float]] = {}
    for weapon, name in enumerate(names):
        for slot in range(ids.shape[1]):
            ability = int(ids[weapon, slot])
            if ability <= 0 or ability in table:
                continue
            holds = int(count[weapon, slot])
            last = int(children[weapon, slot, holds - 1]) if holds else slot
            launches = [row for row, live in zip(events[weapon, slot], present[weapon, slot])
                        if live and row[arsenal_contract.EF_PROJECTILE_SPEED] > 0.0]
            speed = float(launches[0][arsenal_contract.EF_PROJECTILE_SPEED]) if launches else 0.0
            gravity = float(launches[0][arsenal_contract.EF_PROJECTILE_GRAVITY]) if launches else 0.0
            reach = MELEE_REACH_BLOCKS.get(name, 0.0) if slot == 0 else 0.0
            first = [float(value) for value in thresholds[weapon, slot, :holds][:v3.HOLD_THRESHOLDS]]
            first += [0.0] * (v3.HOLD_THRESHOLDS - len(first))
            table[ability] = [holds / 4.0, *first, float(indefinite[weapon, slot]), float(last != slot),
                              float(bool(launches)), speed / 100.0, gravity / 50.0,
                              reach / POSITION_SCALE_BLOCKS]
    keys = np.asarray(sorted(table), np.int32)
    return _KitTable(keys, np.asarray([table[int(key)] for key in keys], np.float32))


def _kit_rows(observation) -> tuple[jax.Array, jax.Array]:
    ability_f = observation.ability_f32[:, AGENT_ENTITY]  # [B, 16, 11]
    ids = observation.ability_i32[:, AGENT_ENTITY, :, _ABILITY_ID]
    mask = jnp.asarray(observation.ability_mask[:, AGENT_ENTITY], jnp.bool_)
    batch, slots = ids.shape
    table = _kit_table()
    match = ids[..., None] == jnp.asarray(table.ids)
    known = match.any(axis=-1) & mask
    values = jnp.where(known[..., None], jnp.asarray(table.values)[jnp.argmax(match, axis=-1)], 0.0)
    rows = _zeros(batch, slots, "kit")
    rows = _put(rows, {
        "slot_0": jnp.broadcast_to(jnp.eye(slots, dtype=jnp.float32), (batch, slots, slots)),
        "duration_fraction": ability_f[..., _ABILITY["duration_fraction"]],
        "cooldown_remaining_fraction": ability_f[..., _ABILITY["cooldown_remaining_fraction"]],
        "active_progress": ability_f[..., _ABILITY["active_progress"]],
        "affordable": ability_f[..., _ABILITY["affordable"]],
        "legal": jnp.asarray(observation.ability_legal[:, AGENT_ENTITY], jnp.float32),
        "cost_stamina": ability_f[..., _ABILITY["resource_cost_0"]:_ABILITY["resource_cost_0"] + len(v3.RESOURCES)],
        "hold_count": values, "known": known,
    }, "kit")
    return jnp.where(mask[..., None], rows, 0.0), mask


def _status_rows(observation, effects: EffectTables | None = None) -> tuple[jax.Array, jax.Array]:
    """The agent's status effects: timing, flags, and (with ``effects``) which effect each is, by its published id."""

    status = observation.status_f32[:, AGENT_ENTITY]  # [B, 8, 6]
    flags = jnp.asarray(observation.status_flags[:, AGENT_ENTITY], jnp.uint32)  # [B, 8]: the flag word
    bits = (flags[..., None] >> jnp.arange(len(v3.STATUS_FLAGS), dtype=jnp.uint32)) & jnp.uint32(1)
    mask = jnp.asarray(observation.status_mask[:, AGENT_ENTITY], jnp.bool_)
    columns = {"remaining_fraction": status, "flag_invulnerable": bits}
    if effects is not None:
        ids = observation.status_i32[:, AGENT_ENTITY, :, STATUS_INTEGER_FEATURES.index("effect_id")]  # [B, 8]
        row, found = v2.lookup(effects.keys, effects.rows, ids)
        columns |= {f"effect_{v3.EFFECT_FEATURES[0]}": effects.features[row],
                    "effect_description_0": effects.descriptions[row], "effect_known": found}
    rows = _put(_zeros(*mask.shape, "status"), columns, "status")
    return jnp.where(mask[..., None], rows, 0.0), mask


def _entity_rows(observation, channel, pose: Pose, tables) -> tuple[jax.Array, jax.Array]:
    base = observation.base
    f = base.entity_f32  # [B, E, 16], the body frame
    i = base.entity_i32
    mask = jnp.asarray(base.entity_mask, jnp.bool_)
    batch, count = mask.shape
    scale = jnp.float32(NEARBY_ENTITY_RADIUS_BLOCKS)
    position = jnp.stack((f[..., _ENTITY["relative_forward"]], f[..., _ENTITY["relative_right"]],
                          f[..., _ENTITY["relative_up"]]), axis=-1) * scale
    yaw = jnp.arctan2(f[..., _ENTITY["yaw_sin"]], f[..., _ENTITY["yaw_cos"]]) - pose.body_yaw[:, None]
    phase = i[..., _ENTITY_INT["attack_phase"]]
    columns = {
        "forward": where_of(position, pose, ENTITY_CENTRE_HEIGHT),
        "v_forward": f[..., _ENTITY["relative_velocity_forward"]], "v_right": f[..., _ENTITY["relative_velocity_right"]],
        "v_up": f[..., _ENTITY["relative_velocity_up"]], "yaw_sin": jnp.sin(yaw), "yaw_cos": jnp.cos(yaw),
        "facing_error": f[..., _ENTITY["facing_error"]], "head_facing_error": f[..., _ENTITY["head_facing_error"]],
        "visible": f[..., _ENTITY["visible"]], "health_fraction": f[..., _ENTITY["health_fraction"]],
        "alive": f[..., _ENTITY["alive"]], "hostile": f[..., _ENTITY["hostile"]],
        "phase_idle": jax.nn.one_hot(phase, len(ATTACK_PHASES)),
        "attack_progress": f[..., _ENTITY["attack_progress"]],
        "attack_index": jnp.clip(i[..., _ENTITY_INT["attack_index"]], 0, 8) / 8.0,
        "held_kind_unknown": jnp.ones((batch, count)),
    }
    if channel is not None:
        columns |= _entity_content(channel, tables, count)
    rows = _put(_zeros(batch, count, "entity"), columns, "entity")
    return jnp.where(mask[..., None], rows, 0.0), mask


def _entity_content(channel, tables, count: int) -> dict[str, jax.Array]:
    """The perception channel's entity columns: player, what is held, and with the registry the role's kind,
    attitude and description (zero, ``known`` 0, while the role id is the channel's placeholder 0: S2)."""

    kind = channel.entity_kind[:, :count]
    held = channel.entity_held_item_id[:, :count]
    columns = {"player": kind == v2.PLAYER_KIND, "held_kind_unknown": held > 0, "held_kind_nothing": held <= 0}
    if tables is None:
        return columns
    item, item_found = v2.lookup(tables.item_keys, tables.item_rows, held)
    flags = tables.item_features[item][..., [v3.ITEM_PROPERTIES_V3.index(name) for name in v2.HELD_CATEGORIES]] > 0
    first, item_known = jnp.argmax(flags, axis=-1), item_found & (held > 0)
    columns["held_kind_unknown"] = (held > 0) & ~item_found
    for index, name in enumerate(v2.HELD_CATEGORIES):
        columns[f"held_kind_{name}"] = item_known & flags.any(axis=-1) & (first == index)
    columns["held_kind_other"] = item_known & ~flags.any(axis=-1)
    columns["held_item_block"] = jnp.where(item_known[..., None], tables.item_features[item], 0.0)
    creature, role_known = v2.lookup(tables.creature_keys, tables.creature_rows, channel.entity_role_id[:, :count])
    attitude = jnp.where(role_known, tables.attitude_to_player[creature], -1)
    columns |= {"known": role_known, "attitude_ignore": jax.nn.one_hot(attitude, len(ATTITUDES)),
                "kind_non_gameplay": jnp.where(role_known[..., None], tables.creature_features[creature], 0.0),
                "description_0": jnp.where(role_known[..., None], tables.creature_descriptions[creature], 0.0)}
    return columns


def _projectile_rows(base, pose: Pose) -> tuple[jax.Array, jax.Array]:
    f = base.projectile_f32
    relative = f[..., _PROJECTILE["relative_x"]:_PROJECTILE["relative_x"] + 3] * jnp.float32(PROJECTILE_RADIUS_BLOCKS)
    velocity = f[..., _PROJECTILE["velocity_x"]:_PROJECTILE["velocity_x"] + 3]
    position = world_to_body(relative, pose.body_yaw[:, None])
    motion = world_to_body(velocity, pose.body_yaw[:, None])
    # Closest approach to the agent along the projectile's straight line (blocks a second: ASSUMED_PROJECTILE_TERMINAL).
    real = motion * jnp.float32(ASSUMED_PROJECTILE_TERMINAL)
    speed_squared = jnp.maximum(jnp.sum(real * real, axis=-1), 1e-6)
    time = jnp.clip(-jnp.sum(position * real, axis=-1) / speed_squared, 0.0, APPROACH_HORIZON_SECONDS)
    miss = jnp.linalg.norm(position + real * time[..., None], axis=-1)
    hostile = f[..., _PROJECTILE["hostile"]]
    mask = jnp.asarray(base.projectile_mask, jnp.bool_) & _in_view(position, pose)
    rows = _put(_zeros(*mask.shape, "projectile"), {
        "forward": where_of(position, pose, 0.0), "v_forward": motion, "speed": jnp.linalg.norm(motion, axis=-1),
        "approach_time": time / APPROACH_HORIZON_SECONDS,
        "approach_distance": jnp.minimum(miss / APPROACH_DISTANCE_BLOCKS, 1.0),
        "hostile": hostile, "mine": 1.0 - hostile, "damage_fraction": f[..., _PROJECTILE["damage_fraction"]],
        "radius_fraction": f[..., _PROJECTILE["radius_fraction"]], "age_fraction": f[..., _PROJECTILE["age_fraction"]],
        "lifetime_remaining": f[..., _PROJECTILE["lifetime_remaining"]],
    }, "projectile")
    return jnp.where(mask[..., None], rows, 0.0), mask


def _hazard_rows(base, pose: Pose) -> tuple[jax.Array, jax.Array]:
    f = base.hazard_f32
    relative = f[..., _HAZARD["relative_x"]:_HAZARD["relative_x"] + 3] * jnp.float32(HAZARD_RADIUS_BLOCKS)
    position = world_to_body(relative, pose.body_yaw[:, None])
    hostile = f[..., _HAZARD["hostile"]]
    mask = jnp.asarray(base.hazard_mask, jnp.bool_) & _in_view(position, pose)
    rows = _put(_zeros(*mask.shape, "hazard"), {
        "forward": where_of(position, pose, 0.0),
        "half_x": f[..., _HAZARD["half_extent_x"]:_HAZARD["half_extent_x"] + 3],
        "intensity": f[..., _HAZARD["intensity"]], "duration_remaining": f[..., _HAZARD["duration_remaining"]],
        "damage_rate": f[..., _HAZARD["damage_rate"]], "activation": f[..., _HAZARD["activation"]],
        "hostile": hostile, "mine": 1.0 - hostile,
    }, "hazard")
    return jnp.where(mask[..., None], rows, 0.0), mask


def _geometry_rows(observation, light: Light | None, pose: Pose) -> tuple[jax.Array, jax.Array]:
    geometry = observation.world_geometry
    f = geometry.token_f32  # [B, G, 131]: kind, provenance, x, y, z (/4, world), clearance, flags, dynamic; edges; boxes
    scale = jnp.float32(WORLD_TOKEN_DEFAULT_MAXIMUM_DISTANCE)
    position = world_to_body(f[..., 2:5] * scale, pose.body_yaw[:, None])
    flags = jnp.round(f[..., 6] * 65535.0).astype(jnp.int32)
    bits = (flags[..., None] >> jnp.arange(len(v3.GEOMETRY_FLAGS))) & 1
    mask = jnp.asarray(geometry.token_mask, jnp.bool_) & jnp.asarray(geometry.available, jnp.bool_)[:, None]
    columns = {"forward": where_of(position, pose, 0.0), "kind": f[..., 0], "provenance": f[..., 1],
               "clearance": f[..., 5], "flag_solid": bits, "dynamic_blocked": f[..., 7],
               "edge_0_mask": _edges(f, position),
               "box_0_mask": f[..., WORLD_TOKEN_BASE_FLOAT_SIZE + v3.GEOMETRY_EDGES * WORLD_TOKEN_EDGE_FLOAT_SIZE:]}
    if light is not None:
        columns["light_sky_valid"] = jnp.where(jnp.asarray(light.token_mask, jnp.bool_)[..., None], light.token_f32,
                                               0.0)
    rows = _put(_zeros(*mask.shape, "geometry"), columns, "geometry")
    return jnp.where(mask[..., None], rows, 0.0), mask


def _edges(f: jax.Array, position: jax.Array) -> jax.Array:
    """The gym's traversal edges (mask, destination, cost, kind, flags per edge) as the schema's (mask, to_forward,
    to_right, to_up, cost, kind, flags): the destination, which the gym gives as ``(index + 1) / (N + 1)`` into its own
    token list (``observation/v3/tokens/world.py``), becomes its displacement from this cell in the body frame, in
    blocks over 16. ``position``: float32[B, N, 3], each token's place in the body frame, in blocks."""

    batch, count = f.shape[:2]
    start = WORLD_TOKEN_BASE_FLOAT_SIZE
    gym = f[..., start:start + v3.GEOMETRY_EDGES * WORLD_TOKEN_EDGE_FLOAT_SIZE].reshape(batch, count, v3.GEOMETRY_EDGES,
                                                                            WORLD_TOKEN_EDGE_FLOAT_SIZE)
    index = jnp.clip(jnp.round(gym[..., 1] * (count + 1)).astype(jnp.int32) - 1, 0, count - 1)
    destination = jax.vmap(lambda places, indices: places[indices])(position, index)  # [B, N, E, 3]
    present = gym[..., :1] > 0.5
    to = jnp.where(present, (destination - position[:, :, None, :]) / jnp.float32(POSITION_SCALE_BLOCKS), 0.0)
    return jnp.concatenate((gym[..., :1], to, gym[..., 2:]), axis=-1).reshape(batch, count, -1)


def _block_rows(action_surface, reach: float | None, pose: Pose) -> tuple[jax.Array, jax.Array] | None:
    if action_surface is None:
        return None
    view = action_surface.block_candidates
    f = view.candidate_f32  # [B, K, 28]: position (/reach, world), 16 affordance bits, 8 gather bits, quality
    position = world_to_body(f[..., :3] * jnp.float32(reach), pose.body_yaw[:, None])
    mask = jnp.asarray(view.candidate_mask, jnp.bool_) & jnp.asarray(view.available, jnp.bool_)[:, None]
    rows = _put(_zeros(*mask.shape, "block_candidate"), {
        "forward": where_of(position, pose, 0.5), "affordance_breakable": f[..., 3:3 + len(v3.AFFORDANCES)],
        "gather_bit_0": f[..., 19:19 + v3.GATHER_BITS], "required_quality": f[..., 27],
    }, "block_candidate")
    return jnp.where(mask[..., None], rows, 0.0), mask


def _inventory_rows(channel, tables, capacity: int) -> tuple[jax.Array, jax.Array] | None:
    """The agent's own inventory from the perception channel, as v2's rows (the same features)."""

    if channel is None or capacity < 1:
        return None
    return v2._inventory_rows(channel, tables, capacity)


def _recipe_rows(action_surface) -> tuple[jax.Array, jax.Array] | None:
    if action_surface is None:
        return None
    encoding = action_surface.recipe_encoding
    mask = jnp.asarray(encoding.candidate_mask, jnp.bool_) & jnp.asarray(encoding.available, jnp.bool_)[:, None]
    embedding = jnp.where(mask[..., None], encoding.candidate_embedding, 0.0)
    rows = _put(_zeros(*mask.shape, "recipe"), {"recipe_0": embedding[..., :v3.RECIPE_WIDTH]}, "recipe")
    return rows, mask


def append_rows(perceived, remembered):
    """Perceived rows followed by remembered rows (either may be None)."""

    if perceived is None:
        return remembered
    return (jnp.concatenate((perceived[0], remembered[0]), axis=1),
            jnp.concatenate((perceived[1], remembered[1]), axis=1))


def fit_ids(ids: jax.Array, mask: jax.Array, capacity: int) -> jax.Array:
    """int32[B, capacity]: the ids of rows packed as :func:`pack` packs them (valid rows first, in their order, the
    first ``capacity`` kept), -1 for a padded slot."""

    order = jnp.argsort(~mask, axis=1, stable=True)
    ids, mask = jnp.take_along_axis(ids, order, axis=1), jnp.take_along_axis(mask, order, axis=1)
    if ids.shape[1] < capacity:
        ids = jnp.pad(ids, ((0, 0), (0, capacity - ids.shape[1])), constant_values=-1)
        mask = jnp.pad(mask, ((0, 0), (0, capacity - mask.shape[1])))
    return jnp.where(mask[:, :capacity], ids[:, :capacity], -1)


def _fit(built, batch: int, capacity: int, width: int) -> tuple[jax.Array, jax.Array]:
    """Rows packed into ``capacity`` slots, valid rows first in their order; the rest padded invalid. More valid rows
    than slots keeps the first ``capacity`` (perceived before remembered)."""

    if built is None:
        return jnp.zeros((batch, capacity, width), jnp.float32), jnp.zeros((batch, capacity), jnp.bool_)
    features, mask = built
    order = jnp.argsort(~mask, axis=1, stable=True)
    features = jnp.take_along_axis(features, order[..., None], axis=1)
    mask = jnp.take_along_axis(mask, order, axis=1)
    count = features.shape[1]
    if count < capacity:
        features = jnp.pad(features, ((0, 0), (0, capacity - count), (0, 0)))
        mask = jnp.pad(mask, ((0, 0), (0, capacity - count)))
    return features[:, :capacity], mask[:, :capacity]


__all__ = ["APPROACH_DISTANCE_BLOCKS", "APPROACH_HORIZON_SECONDS", "ASSUMED_PROJECTILE_TERMINAL",
           "ENTITY_CENTRE_HEIGHT", "VIEW_HALF_ANGLE_DEGREES", "Light", "Perceived", "Pose", "Remembered",
           "append_rows", "fit_ids", "legal_tokens_v3", "pack", "perceive_rows", "where_of"]
