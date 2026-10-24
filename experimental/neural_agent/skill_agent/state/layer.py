"""One tick of the state layer (``docs/VIRTUAL_HUMAN.md`` section 3; the token contract is ``docs/PERCEPTION.md``
section 10): the agent's state brought up to date from its senses, and read by the body as tokens.

The stateless parts are ``state/senses.py`` (the legal observation to this tick's rows) and
``state/frames.py`` (space). The state is :class:`AgentState`: the agent's action history
(``state/action_history.py``), its stored memory (``state/memory.py``: the spatial map and the salient store), its
variables (``state/registry.py``: the persona, ``state/persona.py``, and the emotional state the affect rules publish,
``state/affect.py``; ``docs/VIRTUAL_HUMAN.md``), the affect rules' own working state, and the trunk's latent memory (``model/encoder.py``, ``memory_slots``; the trunk call fills it, this module only carries and
stores it). One tick, :func:`step`:

1. the action history takes the action the agent sent last tick, and a respawn restarts it (16.2; the alarm and startle
   restart in item 3, 20.2);
2. the senses read this tick's perceived rows;
3. the emotional state appraises them, with the persona's gains, and is published to the variables;
4. the event log registers this tick's changes (``state/events.py``, section 21), and its counters are published to the
   variables;
5. the top layers' writes are applied (``state/written.py``), and the live ``remember`` entries become the stores' tags;
6. the stores take them, and the long-term places take this tick's events, marks and fear (``state/places.py``,
   section 22);
7. the people the agent perceives get their interaction facts: met, meetings, fights and the last interaction
   (``state/social.py``, section 26);
8. the stores give back what is remembered but not perceived;
9. every entity token, perceived or remembered, gets the agent's feelings toward it (``feel_*``, provisional by rule,
   26.5), and the relationship the decision layer wrote about it, with its personal-space distance (``rel_*``,
   ``role_*``, ``space_target``; section 26);
10. everything is packed into the ``skill_tokens_v3`` token sets the trunk reads, with the persona and affect
    tokens, whose columns are the body-read variables (``tokens_v3.PERSONA_FEATURES``, ``AFFECT_FEATURES``), and the
    place and event tokens, read from the places and the event log the steps above updated (sections 22, 21 and 24).

:func:`save` and :func:`load` write and read a state as one ``.npz`` (memory "stored as well", owner 2026-10-20 night),
with the variables' names and the token layout, so a save migrates by name to a later registry or schema
(``docs/VIRTUAL_HUMAN.md`` section 14, rules 2 and 4).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract import tokens_v3 as v3
from experimental.neural_agent.skill_agent.contract import variables
from experimental.neural_agent.skill_agent.contract.surface import ITEM_PROPERTIES
from experimental.neural_agent.skill_agent.perception.adapter import lookup
from experimental.neural_agent.skill_agent.state import clock
from experimental.neural_agent.skill_agent.state import registry
from experimental.neural_agent.skill_agent.state import affect as feeling
from experimental.neural_agent.skill_agent.state import events
from experimental.neural_agent.skill_agent.state import memory as stores
from experimental.neural_agent.skill_agent.state import places
from experimental.neural_agent.skill_agent.state import social
from experimental.neural_agent.skill_agent.state import written
from experimental.neural_agent.skill_agent.state.action_history import (
    ActionHistory,
    act_features,
    initial_history,
    update_history,
)
from experimental.neural_agent.skill_agent.state.frames import POSITION_SCALE_BLOCKS, world_to_body
from experimental.neural_agent.skill_agent.state.senses import (
    Perceived,
    Pose,
    _put,
    _zeros,
    append_rows,
    fit_ids,
    pack,
    perceive_rows,
    where_of,
)
from experimental.neural_agent.skill_agent.state.persona import TRAITS, neutral_persona
from hytalegym.jax.combat.observation.v3.tokens.inventory import INVENTORY_POLICY_FLOAT_FEATURES

_SELF = {name: index for index, name in enumerate(v3.SelfTokenV3.FEATURES)}
_ENTITY = {name: index for index, name in enumerate(v3.EntityTokenV3.FEATURES)}
assert tuple(name for name in variables.NAMES if name in events.COUNTERS) == events.COUNTERS, (
    "the event log's counters are registry variables, in events.COUNTERS' order (contract/variables.py)")


class AgentState(NamedTuple):
    """The neural layer's perception state for B agents."""

    history: ActionHistory
    memory: stores.SpatialMemory
    latent: jax.Array | None = None  # float32[B, S, W]: the trunk's last ``EncoderOutput.memory``
    variables: jax.Array | None = None  # float32[B, V]: ``contract/variables.NAMES``, read through ``registry``
    affect: feeling.AffectState | None = None  # the affect rules' working state
    entity_ids: jax.Array | None = None  # int32[B, E]: the id of each entity token row last tick, -1 for padding
    object_slots: dict | None = None  # per important type: int32[B, N], each token row's store slot (-1 for none)
    background: jax.Array | None = None  # float32[B, K, W]: the trunk's last ``EncoderOutput.background``
    written: written.Written | None = None  # the top layers' written entries (state/written.py)
    events: events.EventState | None = None  # the event log and the counters (state/events.py)
    places: places.PlaceState | None = None  # the long-term spatial memory (state/places.py, section 22)
    social: social.SocialState | None = None  # the people's interaction facts (state/social.py, section 26)


def initial_state(batch: int, latent: jax.Array | None = None, persona: jax.Array | None = None,
                  capacities: Mapping[str, int] = v3.CAPACITIES_V3, object_width: int = 0,
                  background: jax.Array | None = None) -> AgentState:
    """The state at an episode's start (``latent`` and ``background``: the trunk's initial slots,
    ``encoder.initial_memory`` and ``initial_background``, which the trunk call fills and this layer carries;
    ``persona``: float32[B, len(persona.TRAITS)], the population's mean when None; ``capacities``: those
    :func:`step` packs to; ``object_width``: the trunk's width when it keeps object slots, ``EncoderConfig.object_types``,
    else 0)."""

    persona = neutral_persona(batch) if persona is None else jnp.asarray(persona, jnp.float32)
    affect = feeling.initial_affect(persona)
    values = registry.put(registry.defaults(batch), TRAITS, persona)
    values = registry.put(values, v3.AFFECT_FEATURES, feeling.affect_features(affect))
    slots = {name: jnp.full((batch, capacities[name]), -1, jnp.int32) for name in ("entity", "block_candidate")}
    return AgentState(initial_history(batch), stores.initial_memory(batch, latent_width=object_width), latent, values,
                      affect, slots["entity"], slots, background, written=written.initial_written(batch),
                      events=events.initial_events(batch, entities=capacities["entity"],
                                                   slots=capacities["inventory_slot"], outcomes=events.OUTCOME_SLOTS),
                      places=places.initial_places(batch), social=social.initial_social(batch))


def _restarts(reset: jax.Array | None, respawn: jax.Array | None) -> jax.Array | None:
    """The lanes whose action history starts over: a new episode (``reset``) or a respawn (``respawn``); None if neither
    is given."""

    if respawn is None:
        return reset
    return respawn if reset is None else reset | respawn


_PLACE_REMEMBER: int = [field.name for field in variables.PLACE_ENTRIES].index("remember")
"""The column of a place entry's ``remember`` priority (``state/written.py``)."""


def _logged(log: events.EventLog, now: jax.Array, kind: str) -> jax.Array:
    """bool[B]: the log registered an event of ``kind`` this tick (the entries stamped ``now``)."""

    return jnp.any((log.kind == events.EVENT_KINDS.index(kind)) & (log.tick == now[:, None]), axis=1)


PLACE_PULL_BLOCKS: float = 16.0
"""A place's pull on the body halves at this distance, in blocks [E]: the places the body reads rank by
``strength / (1 + distance / PLACE_PULL_BLOCKS)`` after the marked ones (section 24)."""
PLACE_VISITS_HORIZON: float = 1000.0
"""A place's visits are log-scaled to [0, 1] over this many visits [E]."""
PLACE_SINCE_HORIZON_SECONDS: float = 45000.0
"""The time since a place's last visit is log-scaled to [0, 1] over this many seconds [E] (section 22): 12.5 h."""
EVENT_AGE_HORIZON_SECONDS: float = 81.0
"""An event's age is log-scaled to [0, 1] over this many seconds [E]: 81 s (1800 ticks at 0.045 s a tick)."""


def _take(values: jax.Array, order: jax.Array) -> jax.Array:
    """``values`` float32[B, P, ...] (or bool, int) at the slots ``order`` int32[B, C]: [B, C, ...]."""

    return jnp.take_along_axis(values, order.reshape(order.shape + (1,) * (values.ndim - 2)), axis=1)


def _place_rows(placed: places.PlaceState, pose: Pose, capacity: int, tick_seconds: float
                ) -> tuple[jax.Array, jax.Array]:
    """The places the body reads (section 24): the marked ones first, then by ``strength / (1 + distance /
    PLACE_PULL_BLOCKS)``, the first ``capacity`` kept. Each has its ``where`` block from me, its kinds, its strength,
    its visits and time since its last visit (log-scaled), and whether I am inside it. Needs ``pose.position``."""

    displacement, distance, valid = places.relative(placed, pose.position, pose.body_yaw)
    store = placed.places
    pull = store.strength / (1.0 + distance / PLACE_PULL_BLOCKS)
    rank = jnp.where(valid, -jnp.where(store.marked, jnp.inf, pull), jnp.inf)  # the lowest ranks first
    order = jnp.argsort(rank, axis=1, stable=True)[:, :capacity]
    since = jnp.maximum(placed.now[:, None] - _take(store.last, order), 0)
    rows = _put(_zeros(placed.now.shape[0], capacity, "place"), {
        "forward": where_of(_take(displacement, order), pose, 0.0),
        "place_stay": _take(store.kinds, order),
        "strength": _take(store.strength, order),
        "visits_log": jnp.minimum(jnp.log1p(_take(store.visits, order)) / math.log1p(PLACE_VISITS_HORIZON), 1.0),
        "since_visit_log": jnp.minimum(jnp.log1p(since) / math.log1p(float(clock.about(PLACE_SINCE_HORIZON_SECONDS,
                                                                                       tick_seconds))), 1.0),
        "inside": order == placed.inside[:, None],
    }, "place")
    chosen = _take(valid, order)
    return jnp.where(chosen[..., None], rows, 0.0), chosen


def _event_rows(logged: events.EventState, pose: Pose, entity_ids: jax.Array, capacity: int, tick_seconds: float
                ) -> tuple[jax.Array, jax.Array]:
    """The last ``capacity`` events, oldest first (section 24): kind, age, amount, inferred or observed, whether the
    other party is a known entity now (one of this tick's ``entity_ids``), and where it happened: its ``where`` block
    from me, when I have a position and the event has one (``where_known``; an event at the world origin reads as
    having none)."""

    recent = events.recent(logged, capacity)
    batch = recent.kind.shape[0]
    age = jnp.maximum(logged.now[:, None] - recent.tick, 0)
    other_known = (recent.other >= 0) & jnp.any(recent.other[..., None] == entity_ids[:, None, :], axis=-1)
    where_known = jnp.zeros((batch, capacity), jnp.bool_)
    block = jnp.zeros((batch, capacity, len(v3.WHERE_FEATURES)), jnp.float32)
    if pose.position is not None:
        where_known = recent.valid & jnp.any(recent.world != 0.0, axis=-1)
        offset = recent.world - pose.position[:, None, :]
        block = jnp.where(where_known[..., None], where_of(world_to_body(offset, pose.body_yaw[:, None]), pose, 0.0),
                          0.0)
    rows = _put(_zeros(batch, capacity, "event"), {
        "forward": block,
        "event_damage_taken": jax.nn.one_hot(recent.kind, len(events.EVENT_KINDS)),
        "age_log": jnp.minimum(jnp.log1p(age) / math.log1p(float(clock.about(EVENT_AGE_HORIZON_SECONDS, tick_seconds))),
                               1.0),
        "amount": jnp.clip(recent.amount, -1.0, 1.0),
        "inferred": recent.inferred,
        "other_known": other_known,
        "where_known": where_known,
    }, "event")
    return jnp.where(recent.valid[..., None], rows, 0.0), recent.valid


_RELATION_FIELDS: Mapping[str, str] = {
    "rel_closeness": "closeness", "rel_ability": "ability", "rel_benevolence": "benevolence",
    "rel_integrity": "integrity", "rel_hostility": "hostility", "rel_fear": "fear", "rel_anger": "anger",
    "role_friend": "friend", "role_family": "family", "role_leader": "leader", "role_follower": "follower",
    "role_rival": "rival", "role_trader": "trader", "rel_ingroup": "ingroup"}
"""Each relationship column of the entity token (``tokens_v3.RELATION_FEATURES``) and the entry's field it reads."""
_ENTRY_FIELDS: Mapping[str, int] = {field.name: index for index, field in enumerate(variables.ENTITY_ENTRIES)}


def _relations(record: written.Written, ids: jax.Array, valid: jax.Array, feelings: jax.Array) -> jax.Array:
    """float32[B, E, len(RELATION_FEATURES)]: each entity row's relationship (section 26.2). ``rel_known`` is 1 when the
    decision layer holds an entry about its id (the defaults otherwise), and ``space_target`` is the personal-space
    distance over ``social.SPACE_SCALE_BLOCKS``, from the written closeness and the larger of the written and the rule's
    hostility and fear (``feelings`` float32[B, E, 4], ``tokens_v3.FEELINGS``). Zeros on padding rows."""

    fields, held = written.entity_entry(record, jnp.where(valid, ids, -1))

    def field(name: str) -> jax.Array:
        return fields[..., _ENTRY_FIELDS[name]]

    hostility = jnp.maximum(field("hostility"), feelings[..., v3.FEELINGS.index("hostility")])
    fear = jnp.maximum(field("fear"), feelings[..., v3.FEELINGS.index("fear")])
    space = social.space_target(field("closeness"), hostility, fear) / social.SPACE_SCALE_BLOCKS
    columns = [held.astype(jnp.float32) if name == "rel_known" else space if name == "space_target"
               else field(_RELATION_FIELDS[name]) for name in v3.RELATION_FEATURES]
    return jnp.where(valid[..., None], jnp.stack(columns, axis=-1), 0.0)


def _chances(rows: jax.Array, present: jax.Array) -> jax.Array:
    """bool[B, N]: the perceived entity rows with a chance to strike me that I see (``state/social.py``, ``spared``):
    alive, in reach (horizontally, ``SPARE_REACH_BLOCKS``) and facing me within ``SPARE_FACING_DEGREES``. The first two
    columns are its place ahead and to the right, in units of ``POSITION_SCALE_BLOCKS``, and ``yaw_sin``, ``yaw_cos``
    are its yaw minus mine, so its facing in my body frame is ``(cos rel, -sin rel)``."""

    place = rows[..., :2] * jnp.float32(POSITION_SCALE_BLOCKS)
    rel = jnp.arctan2(rows[..., _ENTITY["yaw_sin"]], rows[..., _ENTITY["yaw_cos"]])
    facing = jnp.stack((jnp.cos(rel), -jnp.sin(rel)), axis=-1)
    distance = jnp.linalg.norm(place, axis=-1)
    toward_me = jnp.sum(facing * -place, axis=-1)  # the facing's dot product with the direction from it to me
    faces = toward_me >= distance * math.cos(math.radians(social.SPARE_FACING_DEGREES))
    return present & (rows[..., _ENTITY["alive"]] > 0.5) & (distance <= social.SPARE_REACH_BLOCKS) & faces


def step(observation: Any, state: AgentState, *, last_action: jax.Array,
             reset: jax.Array | None = None, respawn: jax.Array | None = None, writes: written.Writes | None = None,
             capacities: Mapping[str, int] = v3.CAPACITIES_V3, outcomes: jax.Array | None = None,
             outcome_verbs: jax.Array | None = None, tick_seconds: float = clock.DEFAULT_TICK_SECONDS,
             **adapter: Any
             ) -> tuple[dict[str, tuple[jax.Array, jax.Array]], AgentState]:
    """One tick (module docstring): the token sets and the next state.

    Args:
        observation: The learner observation.
        state: Last tick's state.
        last_action: int32[B, 21], the gen-7 action the agent sent last tick.
        reset: bool[B], lanes that start a new episode this tick (their history, stores and emotional state start
            over, the persona stays; the trunk's latent memory is reset by its caller).
        respawn: bool[B], lanes that respawned this tick (16.2, normal death): their alarm, startle and action history
            start over, and nothing else. The stores, the written entries, the event log, the persona, the feelings and
            the mood are kept (20.2), since the agent is the same person after its death.
        writes: This tick's writes from the top layers (``written.lower``), or None.
        capacities: Slots per type.
        outcomes: int32[B, OUTCOME_SLOTS], this tick's request outcomes from the runtime (0 none, 1 done, 2 failed,
            3 aborted), or zeros when None.
        outcome_verbs: int32[B, OUTCOME_SLOTS], each request's verb id in ``neural_skills/spec.py`` ``VERBS`` (the
            modules' append-only vocabulary; -1 for a verb outside it), or zeros when None.
        tick_seconds: The world's tick length in seconds (section 27, the state layer's one clock): gen 7's 1/30 s by
            default; a gen-5 world passes ``clock.GEN5_TICK_SECONDS``, and live its measured mean. Every rule's window
            and fade is in seconds and is turned into ticks from it.
        **adapter: Passed to ``senses.perceive_rows``: ``body_yaw_degrees``, ``tables`` (the content registry),
            ``action_surface``, ``block_reach_blocks``, ``light``, ``view_half_angle_degrees``, and ``effects`` (the
            status-effect registry's tables, ``perception/effects.effect_tables``; None reads the effect columns as
            zero, with ``effect_known`` 0).
    """

    history = update_history(state.history, last_action, _restarts(reset, respawn))
    perceived = perceive_rows(observation, inventory_capacity=capacities["inventory_slot"], **adapter)
    affect = feeling.update(state.affect, perceived, registry.take(state.variables, TRAITS), reset, respawn=respawn,
                            tick_seconds=tick_seconds)
    values = registry.put(state.variables, v3.AFFECT_FEATURES, feeling.affect_features(affect))
    if outcomes is None:
        outcomes = jnp.zeros((observation.base.self_f32.shape[0], events.OUTCOME_SLOTS), jnp.int32)
    if outcome_verbs is None:
        outcome_verbs = jnp.zeros_like(outcomes)
    observed = _observed(observation, perceived, affect, adapter.get("tables"), capacities, outcomes, outcome_verbs,
                         respawn)
    logged = events.update(state.events, observed, reset, tick_seconds=tick_seconds)
    values = registry.put(values, events.COUNTERS, logged.counters)
    record = written.apply(state.written, writes, reset)
    memory = stores.update(state.memory, perceived, written.tags(record), reset, tick_seconds=tick_seconds)
    placed = places.update(
        state.places, position=observed.position, has_position=observed.has_position,
        danger=_logged(logged.log, logged.now, "damage_taken"), alarm=affect.fear,
        find=_logged(logged.log, logged.now, "item_gained"), respawned=observed.respawned,
        marks_world=record.place.world,
        marks_live=record.place.valid & (record.place.values[..., _PLACE_REMEMBER] > 0), reset=reset,
        tick_seconds=tick_seconds)
    entity_rows, present = perceived.rows["entity"]  # this tick's perceived entities, before the remembered ones
    seen_ids = jnp.where(present, perceived.entity_ids[:, :entity_rows.shape[1]], -1)
    seen_world = None if perceived.pose.position is None else stores._world_of(entity_rows, perceived.pose)
    written_ids = jnp.where(record.entity.valid, record.entity.key_id, -1)
    seen_chance = _chances(entity_rows, present)
    seen_alive = present & (entity_rows[..., _ENTITY["alive"]] > 0.5)
    socialed = social.update(state.social, seen_ids, seen_world, seen_chance, logged, written_ids, reset,
                             tick_seconds=tick_seconds, seen_alive=seen_alive)
    remembered = stores.read(memory, perceived.pose, tick_seconds=tick_seconds)
    rows = dict(perceived.rows)
    rows["act"] = (act_features(history, tick_seconds), jnp.ones(history.last_choice.shape, jnp.bool_))
    entity_ids = perceived.entity_ids[:, :rows["entity"][0].shape[1]]
    if remembered is not None:
        for name in ("entity", "geometry", "block_candidate"):
            rows[name] = append_rows(rows[name], getattr(remembered, name))
        entity_ids = jnp.concatenate([entity_ids, jnp.where(remembered.entity[1], memory.entity.key_id, -1)], axis=1)
    features, valid = rows["entity"]
    feel = v3.EntityTokenV3.FEATURES.index(f"feel_{v3.FEELINGS[0]}")
    feelings = jnp.where(valid[..., None], feeling.feelings_for(affect, jnp.where(valid, entity_ids, -1)), 0.0)
    relation = v3.EntityTokenV3.FEATURES.index(v3.RELATION_FEATURES[0])
    relations = _relations(record, entity_ids, valid, feelings)
    rows["entity"] = (features.at[..., feel:feel + len(v3.FEELINGS)].set(feelings)
                      .at[..., relation:relation + len(v3.RELATION_FEATURES)].set(relations), valid)
    batch = observation.base.self_f32.shape[0]
    one = jnp.ones((batch, 1), jnp.bool_)
    if perceived.pose.position is not None:  # the long-term places need a position (22.1)
        rows["place"] = _place_rows(placed, perceived.pose, capacities["place"], tick_seconds)
    rows["event"] = _event_rows(logged, perceived.pose, entity_ids, capacities["event"], tick_seconds)
    for name in ("persona", "affect"):  # the body-read variables, the token's ``marked`` 0
        columns = registry.take(values, v3.TOKEN_TYPES_V3[name].FEATURES[:-1])
        rows[name] = (jnp.pad(columns, ((0, 0), (0, 1)))[:, None, :], one)
    tokens = pack(rows, batch, capacities)
    ids = fit_ids(entity_ids, valid, capacities["entity"])  # aligned with tokens["entity"], perceived and remembered
    slots = stores.object_slots(memory, tokens, ids, perceived.pose)
    return tokens, AgentState(history, memory, state.latent, values, affect, ids, slots, state.background,
                              written=record, events=logged, places=placed, social=socialed)


def _present_first(present: jax.Array, count: int, *columns: tuple[jax.Array, Any]) -> list[jax.Array]:
    """Each ``(values, fill)`` in ``columns`` as [B, count]: the present rows first (a stable sort, so the others keep
    their order), the first ``count`` kept, and the rest padded with ``fill``, so the width is fixed whatever the number
    present."""

    order = jnp.argsort(~jnp.asarray(present, jnp.bool_), axis=1, stable=True)[:, :count]
    return [jnp.pad(jnp.take_along_axis(values, order, axis=1), ((0, 0), (0, count - order.shape[1])),
                    constant_values=fill) for values, fill in columns]


def _observed(observation: Any, perceived: Perceived, affect: feeling.AffectState, tables: Any,
              capacities: Mapping[str, int], outcomes: jax.Array, outcome_verbs: jax.Array,
              respawn: jax.Array | None) -> events.Observed:
    """What the event log reads this tick (``state/events.py``, section 21.1), all of it legal: my health, position,
    movement and attack; the damage the affect rules credited to an entity; the entities I perceive; my inventory,
    with each item's consumable flag when the registry is given (``tables``); the request outcomes the runtime
    reports; and whether I respawned (``respawn``, 16.2; False when None)."""

    batch = outcomes.shape[0]
    respawned = jnp.zeros((batch,), jnp.bool_) if respawn is None else jnp.asarray(respawn, jnp.bool_)
    own = perceived.rows["self"][0][:, 0]  # the self token's features, float32[B, F]
    moving = jnp.stack([own[:, _SELF[f"moving_{name}"]] for name in events.MOVES], axis=1)
    if perceived.pose.position is None:  # no perception channel, so no position and no distance
        position, has_position = jnp.zeros((batch, 3), jnp.float32), jnp.zeros((batch,), jnp.bool_)
    else:
        position, has_position = perceived.pose.position, jnp.ones((batch,), jnp.bool_)
    entities, present = perceived.rows["entity"]
    ids = jnp.where(present, perceived.entity_ids[:, :entities.shape[1]], -1)
    distance = jnp.linalg.norm(entities[..., :3], axis=-1) * POSITION_SCALE_BLOCKS
    entity_ids, entity_health, entity_distance = _present_first(
        present, capacities["entity"], (ids, -1), (entities[..., _ENTITY["health_fraction"]], 0.0), (distance, 0.0))

    channel = getattr(observation, "perception_v2", None)
    slots = capacities["inventory_slot"]
    held = jnp.zeros((batch, slots), jnp.bool_)
    item_ids = jnp.zeros((batch, slots), jnp.int32)
    counts = jnp.zeros((batch, slots), jnp.float32)
    if channel is not None:
        held = jnp.asarray(channel.inventory_mask, jnp.bool_)
        item_ids = jnp.where(held, channel.inventory_i32[..., 2], 0)
        fraction = channel.inventory_f32[..., INVENTORY_POLICY_FLOAT_FEATURES.index("quantity_log2_fraction")]
        # The gym encodes log2(q + 1) / 31 (inventory.py), so 2 ** (31 f) - 1 recovers the quantity exactly.
        counts = jnp.where(held, jnp.round(jnp.exp2(31.0 * fraction) - 1.0), 0.0)
    consumable = jnp.zeros((batch, slots), jnp.bool_)
    if tables is not None:
        row, _ = lookup(tables.item_keys, tables.item_rows, item_ids)
        consumable = held & (tables.item_features[row][..., ITEM_PROPERTIES.index("consumable")] > 0.5)
    item_ids, item_counts, item_consumable = _present_first(held, slots, (item_ids, 0), (counts, 0.0),
                                                            (consumable, False))
    return events.Observed(
        health=own[:, _SELF["health_fraction"]], position=position, has_position=has_position, moving=moving,
        attack_executing=own[:, _SELF["attack_executing"]] > 0.5, attributed=affect.attributed,
        entity_ids=entity_ids, entity_health=entity_health, entity_distance=entity_distance,
        item_ids=item_ids, item_counts=item_counts, item_consumable=item_consumable,
        outcomes=outcomes, outcome_verbs=outcome_verbs, respawned=respawned)


_STORE_TYPES: Mapping[str, str] = {"geometry": "geometry", "entity": "entity", "block": "block_candidate"}
"""Each store of ``SpatialMemory`` and the token type of its rows."""
_ENTRY_VALUES: Mapping[str, tuple[str, tuple[variables.Variable, ...]]] = {
    ".written.entity.values": ("written.entity", variables.ENTITY_ENTRIES),
    ".written.place.values": ("written.place", variables.PLACE_ENTRIES),
}
"""Each written entry table's stored fields (``written.py``): its layout name and its declared fields."""


def _layout() -> dict[str, list[str]]:
    layout = {name: list(cls.FEATURES) for name, cls in v3.TOKEN_TYPES_V3.items()}
    layout["written.entity"] = [field.name for field in variables.ENTITY_ENTRIES]
    layout["written.place"] = [field.name for field in variables.PLACE_ENTRIES]
    return layout


def save(path: str | Path, state: AgentState) -> None:
    """Write ``state`` to ``path`` (``.npz``), each array under its place in the state, with the variables' names
    (``__variables__``) and the token layout (``__layout__``) and their fingerprints (``__fingerprint__``)."""

    leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    layout = _layout()
    np.savez(path, **{jax.tree_util.keystr(key): np.asarray(value) for key, value in leaves},
             __variables__=np.asarray(variables.NAMES), __layout__=np.asarray(json.dumps(layout)),
             __fingerprint__=np.asarray([variables.fingerprint(list(variables.NAMES)), variables.fingerprint(layout)]))


def _first_slots(saved: np.ndarray, template: jax.Array) -> np.ndarray:
    """``template`` with its first slots (axis 1) replaced by ``saved``: a written table saved with fewer slots fills
    the first of them, and the rest keep the state's own values."""

    filled = np.array(template)
    filled[:, :saved.shape[1]] = saved
    return filled


def load(path: str | Path, like: AgentState, *, maps: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
         dropped: Mapping[str, Sequence[str]] | None = None) -> AgentState:
    """Read a state written by :func:`save`, shaped as ``like`` (a state of the same batch and capacities).

    A save made under another variable list or token layout is migrated by name (``docs/VIRTUAL_HUMAN.md`` section 14):
    a variable it lacks takes its default, and a stored row's new column reads 0. A variable or column that is gone must
    be merged into a new one by ``maps`` or listed in ``dropped``, both keyed by ``"variables"`` or a stored token type
    (``variables.transfer``). The written entries' fields migrate by name the same way, keyed by ``"written.entity"`` or
    ``"written.place"``. A written table saved with fewer slots than ``like``'s fills its first slots and keeps
    ``like``'s values in the rest (step 9: 16 entity entries, now 150). A part of the state the save lacks (added since,
    as the object latents, the background summary on 2026-10-22, the written entries on step 2 of section 19, and the
    event log on step 4 of section 21, which a save from before it loads as an empty log, and the places on step 5 of
    section 22, which load with none) takes ``like``'s value.
    """

    maps, dropped = maps or {}, dropped or {}
    with np.load(path) as stored:
        if "__variables__" not in stored:
            raise ValueError(f"{path} was saved without the variables' names (before the registry), so it cannot be "
                             "migrated by name")
        names = tuple(str(name) for name in stored["__variables__"])
        layout = json.loads(str(stored["__layout__"]))
        paths, treedef = jax.tree_util.tree_flatten_with_path(like)
        values = []
        for key, template in paths:
            name = jax.tree_util.keystr(key)
            if name not in stored:  # a part added to the state since this save
                values.append(jnp.asarray(template))
                continue
            value = stored[name]
            if name == ".variables":
                value = registry.migrate(value, names, maps.get("variables"), dropped.get("variables", ()))
            if name in _ENTRY_VALUES:  # a written entry's fields, migrated by name as a stored row's columns are
                table, fields = _ENTRY_VALUES[name]
                today = tuple(field.name for field in fields)
                value = variables.remap(value, layout.get(table, today), today, maps.get(table), dropped.get(table, ()),
                                        fill=[field.default for field in fields])
            if name.startswith((".written.entity.", ".written.place.")) and value.shape[1] < np.shape(template)[1]:
                value = _first_slots(value, template)  # a table saved with fewer slots (16 entity entries, step 9: 150)
            parts = name.split(".")
            if len(parts) == 4 and parts[1] == "memory" and parts[3] == "rows":
                kind = _STORE_TYPES[parts[2]]
                value = variables.remap(value, layout[kind], v3.TOKEN_TYPES_V3[kind].FEATURES, maps.get(kind),
                                        dropped.get(kind, ()))
            if value.shape != np.shape(template):
                raise ValueError(f"{name} is {value.shape}, expected {np.shape(template)}")
            values.append(jnp.asarray(value))
    return jax.tree_util.tree_unflatten(treedef, values)


__all__ = ["AgentState", "initial_state", "load", "save", "step"]
