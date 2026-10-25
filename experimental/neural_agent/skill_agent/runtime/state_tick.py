"""The body's path on token schema v3 (step 1b of ``docs/VIRTUAL_HUMAN.md`` section 13, approved 2026-10-22): each
tick the state layer (``state/layer.py``) builds the agent's tokens, the trunk runs once for every agent with its latent
memory, and each skill module reads the result; a request's entity handle points at the entity token row that holds it,
perceived or remembered. The v1 path is unchanged (``model/freezing.py`` ``encode_contexts``, ``VerbBody``).

- :func:`tick`: the state layer, then the trunk once: the latent memory and the background summary carried (reset where
  a lane starts over), and each important thing's object latent read from its store slot and written back
  (``docs/VIRTUAL_HUMAN.md`` section 18).
- :func:`trunk_step`: :func:`tick`'s trunk half, on tokens the layer has already stepped (the training unroll repeats it).
- :func:`entity_rows` and :func:`point`: the v3 resolver, an entity handle to the token row that holds it, inside the
  jitted tick (``contract/references.py`` resolves v1 token records, and stays free of JAX).
- :func:`module_factors`: a skill module's action for every agent: the masked argmax on its scope's heads, neutral
  elsewhere, as the skill trainers' student rule.
- :func:`encode_contexts` and :class:`ModuleBody`: the same for the combiner. Each agent's context carries its row of
  the tick and every entity it knows, perceived or remembered, so a request may name a remembered entity.
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract.surface import SCOPE_NAMES, scope_of
from experimental.neural_agent.skill_agent.model.encoder import (
    EncoderOutput,
    apply_encoder,
    initial_background,
    initial_memory,
)
from experimental.neural_agent.skill_agent.model.generalist import ARG_FEATURES, ARG_GROUPS, GeneralistRequests
from experimental.neural_agent.skill_agent.neural_skills.spec import VERBS, Pose
from experimental.neural_agent.skill_agent.model.skill_module import SkillModule, Trunk, check_trunk, module_outputs
from experimental.neural_agent.skill_agent.runtime.combiner import AgentContext, BodyStep
from experimental.neural_agent.skill_agent.contract.tokens_v3 import CAPACITIES_V3, EntityTokenV3
from experimental.neural_agent.skill_agent.state import clock, layer
from experimental.neural_agent.skill_agent.state.action_history import HEAD_NEUTRAL
from experimental.neural_agent.skill_agent.state.written import Write, lower
from experimental.neural_agent.skill_agent.training.blocks import HEAD_NAMES, HEAD_SIZES, HEAD_STARTS, NEUTRAL
from hytalegym.jax.combat.observation.v3.tokens.inventory import INVENTORY_POLICY_FLOAT_FEATURES


class StateView(NamedTuple):
    """One tick of B agents as every skill module reads it."""

    encoded: EncoderOutput  # the trunk's output, once for all agents
    masks: dict[str, jax.Array]  # token type -> bool[B, N]: which token slots hold a row
    entity_ids: jax.Array  # int32[B, E]: the id of each entity token row (perceived or remembered), -1 for padding


def tick(trunk: Trunk, state: layer.AgentState, observation: Any, *, last_action: jax.Array,
         reset: jax.Array | None = None, tick_seconds: float = clock.DEFAULT_TICK_SECONDS, **step: Any
         ) -> tuple[StateView, layer.AgentState]:
    """One tick for B agents: ``layer.step`` (``step`` takes its keyword arguments: ``writes``, ``capacities`` and the
    senses' inputs), then the trunk on its tokens. A lane that ``reset`` starts its latent memory and its background
    summary from the trunk's initial slots (its stores, object latents included, start empty in ``layer.step``).
    ``tick_seconds`` is the world's tick (section 27): gen 7's by default, ``clock.GEN5_TICK_SECONDS`` on gen 5."""

    tokens, state = layer.step(observation, state, last_action=last_action, reset=reset,
                               tick_seconds=tick_seconds, **step)
    return trunk_step(trunk, tokens, state, reset)


def trunk_step(trunk: Trunk, tokens: Mapping[str, tuple[Any, Any]], state: layer.AgentState,
               reset: jax.Array | None) -> tuple[StateView, layer.AgentState]:
    """The trunk's half of :func:`tick`, on the ``tokens`` of a state that ``layer.step`` has just returned: the same
    reads and writes, which ``training/sequences.py`` repeats through time. The stores were emptied by the layer, so
    only the latent memory and the background are restarted here, where ``reset`` is True."""
    batch = state.entity_ids.shape[0]
    latent = _carried(state.latent, initial_memory(trunk.params, trunk.config, batch), reset)
    background = _carried(state.background, initial_background(trunk.params, trunk.config, batch), reset)
    stores = {"entity": state.memory.entity, "block_candidate": state.memory.block}
    objects = {name: _gather(stores[name].latent, state.object_slots[name]) for name in trunk.config.object_types}
    encoded = apply_encoder(trunk.params, tokens, trunk.config, memory=latent, background=background,
                            objects=objects or None)
    memory = state.memory
    if trunk.config.object_types:  # each row's next latent goes back to its thing's slot
        written = {name: stores[name]._replace(latent=_scatter(stores[name].latent, state.object_slots[name],
                                                               encoded.objects[name]))
                   for name in trunk.config.object_types}
        memory = memory._replace(entity=written.get("entity", memory.entity),
                                 block=written.get("block_candidate", memory.block))
    view = StateView(encoded, {name: valid for name, (_, valid) in tokens.items()}, state.entity_ids)
    return view, state._replace(latent=encoded.memory, background=encoded.background, memory=memory)


_ENTITY = {name: index for index, name in enumerate(EntityTokenV3.FEATURES)}
_ATTITUDES = ("ignore", "hostile", "neutral", "friendly", "revered")


def known_entities(state: layer.AgentState, own_position: Any = None, dt_seconds: float = clock.DEFAULT_TICK_SECONDS
                   ) -> list[list[dict[str, Any]]]:
    """Each lane's known entities, for the shared scratchpad (host side, at a decision; the owner's ruling of
    2026-10-22: the decision layer names entities by the ids this layer perceives). Read from the entity store
    (``state/memory.py``), which holds every entity perceived lately, by id, with where it was and its columns as last
    perceived. Per lane, visible ones first, then the most recently seen:

    ``{"id", "visible", "age_s", "position" [x, y, z] (world, blocks; as last seen), "distance" (blocks, when
    ``own_position`` float[B, 3] is given, else None), "hostile", "attitude" (or None while unknown), "health",
    "alive", "player", "tagged"}``. The ids are the ones requests (``point``) and writes (``state/written.py``) take.
    Empty lists without the agent's position (the stores do not anchor then)."""

    store, now = state.memory.entity, np.asarray(state.memory.now)
    valid, ids, ticks = np.asarray(store.valid), np.asarray(store.key_id), np.asarray(store.tick)
    world, rows, tags = np.asarray(store.world), np.asarray(store.rows), np.asarray(store.tag)
    here = None if own_position is None else np.asarray(own_position, np.float64)
    lanes = []
    for lane in range(valid.shape[0]):
        known = []
        for slot in np.flatnonzero(valid[lane] & (ids[lane] >= 0)):
            row = rows[lane, slot]
            attitudes = row[[_ENTITY[f"attitude_{name}"] for name in _ATTITUDES]]
            known.append({
                "id": int(ids[lane, slot]), "visible": bool(ticks[lane, slot] == now[lane]),
                "age_s": float((now[lane] - ticks[lane, slot]) * dt_seconds),
                "position": [float(x) for x in world[lane, slot]],
                "distance": None if here is None else float(np.linalg.norm(world[lane, slot] - here[lane])),
                "hostile": float(max(row[_ENTITY["attitude_hostile"]], row[_ENTITY["hostile"]])),
                "attitude": _ATTITUDES[int(np.argmax(attitudes))] if row[_ENTITY["known"]] > 0.5 else None,
                "health": float(row[_ENTITY["health_fraction"]]), "alive": bool(row[_ENTITY["alive"]] > 0.5),
                "player": bool(row[_ENTITY["player"]] > 0.5), "tagged": float(tags[lane, slot])})
        known.sort(key=lambda entity: (not entity["visible"], entity["age_s"]))
        lanes.append(known)
    return lanes


class OwnPose(NamedTuple):
    """A perception channel carrying only the agent's own pose (its position and body yaw: proprioception, which the
    agent knows) and nothing about others: entity kinds, held items, roles and the inventory read empty. It stands in
    for ``perception_v2`` where the gym publishes none, as on the multi-actor duel path (``docs/PERCEPTION.md`` 10.5),
    so the stores can anchor and the event log can measure distance."""

    body_yaw_sin: jax.Array  # float32[B]
    body_yaw_cos: jax.Array  # float32[B]
    self_position: jax.Array  # float32[B, 3]: world, blocks
    entity_kind: jax.Array  # int32[B, N]
    entity_held_item_id: jax.Array  # int32[B, N]
    entity_role_id: jax.Array  # int32[B, N]
    inventory_i32: jax.Array  # int32[B, S, 3]
    inventory_f32: jax.Array  # float32[B, S, len(INVENTORY_POLICY_FLOAT_FEATURES)]
    inventory_mask: jax.Array  # bool[B, S]


def own_pose_channel(observation: Any, position: jax.Array, body_yaw_degrees: jax.Array) -> Any:
    """``observation`` with an :class:`OwnPose` channel as its ``perception_v2``: ``position`` float32[B, 3] and
    ``body_yaw_degrees`` float32[B] are the agent's own. An observation that already carries a channel is returned as
    it is."""

    if getattr(observation, "perception_v2", None) is not None:
        return observation
    batch, entities = observation.base.entity_mask.shape[:2]
    slots = CAPACITIES_V3["inventory_slot"]
    radians = jnp.deg2rad(jnp.asarray(body_yaw_degrees, jnp.float32))
    zeros = jnp.zeros((batch, entities), jnp.int32)
    channel = OwnPose(jnp.sin(radians), jnp.cos(radians), jnp.asarray(position, jnp.float32), zeros, zeros, zeros,
                      jnp.zeros((batch, slots, 3), jnp.int32),
                      jnp.zeros((batch, slots, len(INVENTORY_POLICY_FLOAT_FEATURES)), jnp.float32),
                      jnp.zeros((batch, slots), jnp.bool_))
    with_channel = _with_pose_channel(tuple(observation._fields))
    return with_channel(*observation, channel)


@functools.lru_cache(maxsize=None)
def _with_pose_channel(fields: tuple[str, ...]) -> type:
    """The NamedTuple class of an observation with ``perception_v2`` appended, one per tuple of field names. The host
    calls :func:`own_pose_channel` every tick: a class built per call is a new pytree type, so the compiled tick
    retraces each time (6.5 s a tick in the architecture test, 2026-10-24)."""

    return NamedTuple("WithOwnPose", [(name, Any) for name in fields] + [("perception_v2", Any)])


def host_inputs(observation: Any, action_factors: jax.Array, done: jax.Array, *, own_position: jax.Array | None = None,
                own_yaw_degrees: jax.Array | None = None) -> tuple[Any, jax.Array, jax.Array, None]:
    """What :func:`step_lanes` needs from one host transition, in plain arrays (the session's transition type stays the
    host's): ``observation``, the one the agent sees after the step; ``action_factors`` int32[B, 12], the gym's heads
    it sent; ``done`` bool[B], lanes whose episode ended, so the next observation starts a new one (an
    auto-reset); and, where the observation carries no perception channel, the agent's own position and body yaw
    for :func:`own_pose_channel`.

    Returns ``(observation, last_action int32[B, 21], reset bool[B], respawn)``: the gym's 12 heads then the planned
    heads at neutral; ``respawn`` is None (the simulator never respawns within an episode, ``docs/VIRTUAL_HUMAN.md``
    20.1)."""

    if own_position is not None and own_yaw_degrees is not None:
        observation = own_pose_channel(observation, own_position, own_yaw_degrees)
    factors = jnp.asarray(action_factors, jnp.int32)
    width = len(HEAD_NEUTRAL) - factors.shape[1]
    planned = jnp.broadcast_to(jnp.asarray(HEAD_NEUTRAL[factors.shape[1]:]), (factors.shape[0], width))
    return observation, jnp.concatenate([factors, planned], axis=1), jnp.asarray(done, jnp.bool_), None


def step_lanes(trunk: Trunk, state: layer.AgentState, observation: Any, *, last_action: jax.Array,
               reset: jax.Array | None = None, write_lists: Sequence[Sequence[Write]] | None = None,
               **step: Any) -> tuple[StateView, layer.AgentState, list[list[str | None]] | None]:
    """One host tick, as the host loop calls it (``docs/VIRTUAL_HUMAN.md`` sections 19 to 21; the loop and the shared
    scratchpad are the coordinator's, the device side is this one): the top layers' writes for this tick, given as each
    lane's ``Write`` list in lane order (``shared_state.skill_writes.writes_for_tick`` at a decision; None between
    decisions, since entries persist), lowered and checked by ``state/written.py`` ``lower``, then :func:`tick`.
    ``step`` carries ``layer.step``'s other keywords: ``respawn``, ``outcomes``, ``outcome_verbs`` and the senses'.

    Returns the view, the state, and each write's refusal reason (None where it passed; None instead of the lists when
    ``write_lists`` is None), aligned with ``write_lists`` for ``record_write_results``. A write that passed can still
    be refused as full on the device: ``state.written.applied`` bool[B, W] says which took effect, the passing writes
    in slot order.

    Raises:
        ValueError: If ``write_lists`` does not have one list per lane.
    """

    if write_lists is None:
        writes, reasons = None, None
    else:
        if len(write_lists) != state.entity_ids.shape[0]:
            raise ValueError(f"write_lists needs one list per lane ({state.entity_ids.shape[0]}), "
                             f"got {len(write_lists)}")
        writes, reasons = lower(write_lists)
    static = {name: _frozen(value) for name, value in step.items() if _is_static(value)}
    arrays = {name: value for name, value in step.items() if not _is_static(value)}
    program = _tick_program(trunk.config, tuple(sorted(static.items())))
    view, state = program(trunk.params, state, observation, last_action, reset, writes, arrays)
    return view, state, reasons


def _is_static(value: Any) -> bool:
    """A keyword that fixes the program rather than feeding it: a plain number, string, None, or a mapping of them
    (``tick_seconds``, ``capacities``)."""

    if isinstance(value, Mapping):
        return all(_is_static(item) for item in value.values())
    return value is None or isinstance(value, (bool, int, float, str))


class _Frozen(NamedTuple):
    """A mapping keyword, hashable for the program cache."""

    items: tuple


def _frozen(value: Any) -> Any:
    return _Frozen(tuple(sorted(value.items()))) if isinstance(value, Mapping) else value


@functools.lru_cache(maxsize=None)
def _tick_program(config: Any, static: tuple) -> Callable[..., Any]:
    """:func:`tick` compiled once per trunk config and set of fixed keywords, so a host loop does not run it eagerly
    (about 1 s a tick at batch 2 eagerly: the gen-7 integration check, 2026-10-24). A frozen mapping comes back as a
    dict."""

    fixed = {name: dict(value.items) if isinstance(value, _Frozen) else value for name, value in static}

    def run(params, state, observation, last_action, reset, writes, arrays):
        return tick(Trunk(config, params, ""), state, observation, last_action=last_action, reset=reset,
                    writes=writes, **arrays, **fixed)

    return jax.jit(run)


def _carried(held: jax.Array | None, initial: jax.Array | None, reset: jax.Array | None) -> jax.Array | None:
    if initial is None:
        return None
    if held is None:
        return initial
    return held if reset is None else jnp.where(reset[:, None, None], initial, held)


def _gather(latents: jax.Array, slots: jax.Array) -> jax.Array:
    """float32[B, N, W]: each row's object latent from its store slot int32[B, N] (zeros for -1)."""

    rows = jnp.take_along_axis(latents, jnp.maximum(slots, 0)[..., None], axis=1)
    return jnp.where((slots >= 0)[..., None], rows, 0.0)


def _scatter(latents: jax.Array, slots: jax.Array, rows: jax.Array) -> jax.Array:
    """``latents`` float32[B, C, W] with each row's value written to its slot (a slot holds at most one row)."""

    to_slot = slots[..., None] == jnp.arange(latents.shape[1])[None, None, :]  # [B, N, C]
    hit = to_slot.any(axis=1)
    source = jnp.argmax(to_slot, axis=1)  # [B, C]
    return jnp.where(hit[..., None], jnp.take_along_axis(rows, source[..., None], axis=1), latents)


def entity_rows(handles: jax.Array, row_ids: jax.Array) -> jax.Array:
    """int32[B, R]: for each request's entity handle int32[B, R] (-1 for none), the index of the entity token row
    holding it in ``row_ids`` int32[B, E] (``AgentState.entity_ids``), perceived or remembered; -1 when none does."""

    match = (row_ids[:, None, :] == handles[:, :, None]) & (handles[:, :, None] >= 0)
    return jnp.where(match.any(axis=-1), jnp.argmax(match, axis=-1), -1).astype(jnp.int32)


def point(view: StateView, requests: GeneralistRequests, handles: jax.Array) -> GeneralistRequests:
    """``requests`` with each slot's entity pointer the row holding its entity handle int32[B, R] (-1 for none)."""

    return requests._replace(entity_pointer=entity_rows(handles, view.entity_ids))


@functools.lru_cache(maxsize=None)
def _module_program(config: Any, known: tuple[int, ...], verb: int) -> Callable[..., Any]:
    return jax.jit(lambda params, encoded, masks, requests: module_outputs(params, config, known, verb, encoded, masks,
                                                                           requests))


def module_factors(module: SkillModule, trunk: Trunk, view: StateView, requests: GeneralistRequests,
                   action_mask: jax.Array | None = None) -> jax.Array:
    """int32[B, 12]: the module's choice on each gym head of its scope (the masked argmax where ``action_mask``, the
    flat legal mask, is given), neutral on every other head."""

    check_trunk(module, trunk)
    out = _module_program(module.config, module.known, module.verb)(module.params, view.encoded, view.masks, requests)
    columns = []
    for head, name in enumerate(HEAD_NAMES):
        logits = out.head_logits[name]
        if action_mask is not None:
            logits = jnp.where(action_mask[:, HEAD_STARTS[head]:HEAD_STARTS[head] + HEAD_SIZES[head]], logits, -jnp.inf)
        choice = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        mine = scope_of(name) == module.scope
        columns.append(choice if mine else jnp.full(choice.shape, int(NEUTRAL[head]), jnp.int32))
    return jnp.stack(columns, axis=1)


class AgentView(NamedTuple):
    """One agent's row of a tick, as a combiner body receives it in ``AgentContext.observation_tokens``: the tick's token
    sets and the row; the agent's pose (a ``neural_skills.spec.Pose`` with leading dimension 1: world blocks and degrees,
    as ``distill.build_requests`` reads them from ``combat``), its decision tick, and its legal action mask (bool[124],
    the env's), which the body's masked argmax reads."""

    view: StateView
    row: int
    pose: Pose
    tick: int
    action_mask: Any


class JointView(NamedTuple):
    """An agent's :class:`AgentView` with its live requests this tick, which :func:`encode_contexts` hands a body when it
    is given ``live_requests``. ``live`` maps each verb name to the agent's live request (the middle layer's
    ``ActionRequest``), for any verbs; a body with a ``layout`` reads its trained neighbours' requests from it."""

    agent: AgentView
    live: Mapping[str, Any]


def encode_contexts(view: StateView, contexts: Mapping[int, AgentContext], agents: Sequence[int], *, poses: Pose,
                    ticks: int | Sequence[int], action_masks: Any,
                    live_requests: Mapping[int, Mapping[str, Any]] | None = None) -> dict[int, AgentContext]:
    """Agent -> its context with its row of ``view`` (``agents[i]`` is row i), its pose, decision tick and legal mask
    (row i of ``poses`` with leading dimension B, of ``ticks`` (one tick, or one per row) and of ``action_masks``
    bool[B, 124]: the caller's, as ``distill.build_requests``' inputs are), and every entity it knows: those it knew,
    and those its entity rows hold, perceived or remembered. Its ``observation_tokens`` is the :class:`AgentView`; with
    ``live_requests`` (agent -> verb name -> its live request this tick), it is a :class:`JointView` carrying that
    agent's live requests (none for an agent it lacks)."""

    ids = np.asarray(view.entity_ids)
    tick_rows = np.broadcast_to(np.asarray(ticks), (len(agents),))
    masks = np.asarray(action_masks)
    out = dict(contexts)
    for row, agent in enumerate(agents):
        held = frozenset(int(value) for value in ids[row] if value >= 0)
        pose = Pose(*(leaf[row:row + 1] for leaf in poses))
        tokens = AgentView(view, row, pose, int(tick_rows[row]), masks[row])
        if live_requests is not None:
            tokens = JointView(tokens, dict(live_requests.get(agent, {})))
        out[agent] = replace(contexts[agent], observation_tokens=tokens,
                             known_entities=contexts[agent].known_entities | held)
    return out


Lowering = Callable[[Any, Pose, int], tuple[np.ndarray, np.ndarray]]
"""A skill's lowering of a request: ``(request, pose, tick) -> (args float32[15], present bool[6])`` (``ARG_GROUPS``),
the request's arguments as the skill's trainer builds them (``neural_skills/<skill>/skill.py``'s ``lowering``), from the
agent's pose (leading dimension 1) and its decision tick."""


V1_LAYOUT: tuple[str, ...] = ("walk_to", "look_at")
"""The slot order V1's generalist was trained with (``training/distill.py``, ``_request_slots``): walk_to in slot 0 and
look_at in slot 1, on every body (the bodies of ``scratchpad/v1_bodies/*``)."""


def v2_layout(skill: str, beside: Sequence[str]) -> tuple[str, ...]:
    """The slot order a skill trained by ``training/skill_dagger.py`` was trained with: the skill in slot 0, then the
    skills beside it in their strategy's order (``beside``, verb names)."""

    return (skill, *beside)


@functools.lru_cache(maxsize=None)
def _skill(verb: str) -> tuple[int, str, Lowering]:
    """A verb's id, scope and lowering, from its skill folder (``neural_skills/<verb>/skill.py``)."""

    skill = importlib.import_module(f"experimental.neural_agent.skill_agent.neural_skills.{verb}.skill")
    spec = skill.spec()
    return spec.verb, spec.scope, skill.lowering


def _unjoin(observation_tokens: Any) -> tuple[AgentView, Mapping[str, Any]]:
    """A body's tokens as its agent view and its live requests: a :class:`JointView`'s, or an :class:`AgentView`'s with
    none."""

    if isinstance(observation_tokens, JointView):
        return observation_tokens.agent, observation_tokens.live
    if isinstance(observation_tokens, AgentView):
        return observation_tokens, {}
    raise TypeError("a v3 module body reads an AgentView or a JointView (runtime/state_tick.py encode_contexts)")


@dataclass(frozen=True)
class ModuleBody:
    """A skill module as a combiner body on the v3 path, in the request slots it was trained with, and the masked argmax
    of each head of its scope among the heads it was trained on (``training/blocks.HEAD_NAMES``); the other gen-7 heads
    are left out, so the combiner keeps them neutral. The mask is the agent's legal action mask
    (``AgentView.action_mask``), as ``module_factors`` and ``distill_v3._student`` take it.

    ``layout`` (None: its own request alone, in slot 0, as before): the verb names of the slots the module was trained
    with (:data:`V1_LAYOUT`, :func:`v2_layout`). Slot s holds the request of verb ``layout[s]``: the body's own request
    for its own verb, and for another verb the agent's live request of that verb (``JointView.live``, lowered by that
    skill's own ``lowering``). A verb with no live request leaves its slot empty, with its verb and scope set and valid
    False, as ``training/distill.py`` and ``training/skill_dagger.py`` build it. Each entity handle is resolved to its
    token row.

    ``report_done`` (False: never done, and the middle layer withdraws the request, as before): done when the module's
    own done logit is above 0, that is when its sigmoid, the probability of done, is above 0.5 (``model/skill_module.py``).
    """

    module: SkillModule
    trunk: Trunk
    lower: Lowering
    layout: tuple[str, ...] | None = None
    report_done: bool = False

    def __post_init__(self) -> None:
        if self.layout is None:
            return
        if self.module.skill not in self.layout:
            raise ValueError(f"layout {self.layout} does not hold the module's own skill {self.module.skill!r}")
        if len(set(self.layout)) != len(self.layout) or len(self.layout) > self.module.config.max_requests:
            raise ValueError(f"layout {self.layout} repeats a verb or has more slots than the module's "
                             f"{self.module.config.max_requests}")
        for name in self.layout:
            if name not in VERBS or VERBS.index(name) not in self.module.known:
                raise ValueError(f"layout names {name!r}, a verb the module does not know (known {self.module.known})")

    def requests(self, request: Any, observation_tokens: Any) -> GeneralistRequests:
        """The module's request slots for this tick, as its layout names them (see the class docstring)."""

        agent, live = _unjoin(observation_tokens)
        own = self.module
        slots = own.config.max_requests
        names = self.layout or (own.skill,)
        # with no layout, every slot names the module's own verb and scope, as before; with one, an empty slot names verb
        # 0 and scope 0, as the trainers' build_requests do (an invalid slot's verb and scope never reach the outputs)
        rest_verb, rest_scope = (own.verb, SCOPE_NAMES.index(own.scope)) if self.layout is None else (0, 0)
        verb = np.full(slots, rest_verb, np.int32)
        scope = np.full(slots, rest_scope, np.int32)
        valid = np.zeros(slots, bool)
        args = np.zeros((slots, ARG_FEATURES), np.float32)
        present = np.zeros((slots, len(ARG_GROUPS)), bool)
        handles = np.full(slots, -1, np.int32)
        for slot, name in enumerate(names):
            if name == own.skill:
                verb_id, scope_name, lower, item = own.verb, own.scope, self.lower, request
            else:
                verb_id, scope_name, lower = _skill(name)
                item = live.get(name)
            verb[slot], scope[slot] = verb_id, SCOPE_NAMES.index(scope_name)
            if item is None:
                continue
            args[slot], present[slot] = lower(item, agent.pose, agent.tick)
            valid[slot] = True
            handles[slot] = -1 if item.args.entity is None else int(item.args.entity)
        return GeneralistRequests(
            verb=jnp.asarray(verb)[None], args=jnp.asarray(args)[None], arg_present=jnp.asarray(present)[None],
            entity_pointer=entity_rows(jnp.asarray(handles)[None], agent.view.entity_ids[agent.row:agent.row + 1]),
            cell_pointer=jnp.full((1, slots), -1, jnp.int32), scope=jnp.asarray(scope)[None],
            valid=jnp.asarray(valid)[None])

    def __call__(self, request: Any, ticks_since_start: int, observation_tokens: Any, belief_tokens: Any) -> BodyStep:
        agent, _ = _unjoin(observation_tokens)
        one = jax.tree_util.tree_map(lambda leaf: leaf[agent.row:agent.row + 1], agent.view)
        requests = self.requests(request, observation_tokens)
        check_trunk(self.module, self.trunk)
        out = _module_program(self.module.config, self.module.known, self.module.verb)(
            self.module.params, one.encoded, one.masks, requests)
        host = jax.device_get(out.head_logits)
        legal = np.asarray(agent.action_mask, bool)
        choices: dict[str, int] = {}
        for index, name in enumerate(HEAD_NAMES):
            if scope_of(name) != self.module.scope:
                continue
            start, size = HEAD_STARTS[index], HEAD_SIZES[index]
            logits = np.where(legal[start:start + size], np.asarray(host[name][0]), -np.inf)
            choices[name] = int(np.argmax(logits))
        # the done logit: its sigmoid is above 0.5 exactly when the logit is above 0
        done = self.report_done and bool(np.asarray(jax.device_get(out.done))[0] > 0.0)
        return BodyStep(choices, done=done)


__all__ = ["AgentView", "JointView", "Lowering", "ModuleBody", "OwnPose", "StateView", "V1_LAYOUT", "encode_contexts",
           "entity_rows", "host_inputs", "known_entities", "module_factors", "own_pose_channel", "point", "step_lanes",
           "tick", "trunk_step", "v2_layout"]
