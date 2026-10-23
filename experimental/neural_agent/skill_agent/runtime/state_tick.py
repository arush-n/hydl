"""The body's path on token schema v3 (step 1b of ``docs/VIRTUAL_HUMAN.md`` section 13, approved 2026-10-22): each
tick the state layer (``state/layer.py``) builds the agent's tokens, the trunk runs once for every agent with its latent
memory, and each skill module reads the result; a request's entity handle points at the entity token row that holds it,
perceived or remembered. The v1 path is unchanged (``model/freezing.py`` ``encode_contexts``, ``VerbBody``).

- :func:`tick`: the state layer, then the trunk once: the latent memory and the background summary carried (reset where
  a lane starts over), and each important thing's object latent read from its store slot and written back
  (``docs/VIRTUAL_HUMAN.md`` section 18).
- :func:`entity_rows` and :func:`point`: the v3 resolver, an entity handle to the token row that holds it, inside the
  jitted tick (``contract/references.py`` resolves v1 token records, and stays free of JAX).
- :func:`module_factors`: a skill module's action for every agent: the masked argmax on its scope's heads, neutral
  elsewhere, as the skill trainers' student rule.
- :func:`encode_contexts` and :class:`ModuleBody`: the same for the combiner. Each agent's context carries its row of
  the tick and every entity it knows, perceived or remembered, so a request may name a remembered entity.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract.surface import SCOPE_NAMES, head_layout, scope_of
from experimental.neural_agent.skill_agent.model.encoder import (
    EncoderOutput,
    apply_encoder,
    initial_background,
    initial_memory,
)
from experimental.neural_agent.skill_agent.model.generalist import ARG_GROUPS, GeneralistRequests
from experimental.neural_agent.skill_agent.model.skill_module import SkillModule, Trunk, check_trunk, module_outputs
from experimental.neural_agent.skill_agent.runtime.combiner import AgentContext, BodyStep
from experimental.neural_agent.skill_agent.contract.tokens_v3 import CAPACITIES_V3
from experimental.neural_agent.skill_agent.state import layer
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
         reset: jax.Array | None = None, **step: Any) -> tuple[StateView, layer.AgentState]:
    """One tick for B agents: ``layer.step`` (``step`` takes its keyword arguments: ``writes``, ``capacities`` and the
    senses' inputs), then the trunk on its tokens. A lane that ``reset`` starts its latent memory and its background
    summary from the trunk's initial slots (its stores, object latents included, start empty in ``layer.step``)."""

    tokens, state = layer.step(observation, state, last_action=last_action, reset=reset, **step)
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
    with_channel = NamedTuple("WithOwnPose", [(name, Any) for name in observation._fields] + [("perception_v2", Any)])
    return with_channel(*observation, channel)


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
    view, state = tick(trunk, state, observation, last_action=last_action, reset=reset, writes=writes, **step)
    return view, state, reasons


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
    """One agent's row of a tick, as a combiner body receives it in ``AgentContext.observation_tokens``."""

    view: StateView
    row: int


def encode_contexts(view: StateView, contexts: Mapping[int, AgentContext], agents: Sequence[int]
                    ) -> dict[int, AgentContext]:
    """Agent -> its context with its row of ``view`` (``agents[i]`` is row i) and every entity it knows: those it
    knew, and those its entity rows hold, perceived or remembered."""

    ids = np.asarray(view.entity_ids)
    out = dict(contexts)
    for row, agent in enumerate(agents):
        held = frozenset(int(value) for value in ids[row] if value >= 0)
        out[agent] = replace(contexts[agent], observation_tokens=AgentView(view, row),
                             known_entities=contexts[agent].known_entities | held)
    return out


Lowering = Callable[[Any], tuple[np.ndarray, np.ndarray]]
"""A skill's own lowering of a request: ``request -> (args float32[15], present bool[6])`` (``ARG_GROUPS``)."""


@dataclass(frozen=True)
class ModuleBody:
    """A skill module as a combiner body on the v3 path: its request in slot 0 (the entity pointer resolved from the
    request's entity handle), the argmax of each head of its scope. It never reports done; the middle layer withdraws
    the request."""

    module: SkillModule
    trunk: Trunk
    lower: Lowering

    def __call__(self, request: Any, ticks_since_start: int, observation_tokens: Any, belief_tokens: Any) -> BodyStep:
        if not isinstance(observation_tokens, AgentView):
            raise TypeError("a v3 module body reads an AgentView (runtime/state_tick.py encode_contexts)")
        view, row = observation_tokens
        one = jax.tree_util.tree_map(lambda leaf: leaf[row:row + 1], view)
        args, present = self.lower(request)
        slots = self.module.config.max_requests
        handle = -1 if request.args.entity is None else int(request.args.entity)
        requests = GeneralistRequests(
            verb=jnp.full((1, slots), self.module.verb, jnp.int32),
            args=jnp.zeros((1, slots, len(args)), jnp.float32).at[0, 0].set(jnp.asarray(args, jnp.float32)),
            arg_present=jnp.zeros((1, slots, len(ARG_GROUPS)), bool).at[0, 0].set(jnp.asarray(present, bool)),
            entity_pointer=jnp.full((1, slots), -1, jnp.int32), cell_pointer=jnp.full((1, slots), -1, jnp.int32),
            scope=jnp.full((1, slots), SCOPE_NAMES.index(self.module.scope), jnp.int32),
            valid=jnp.zeros((1, slots), bool).at[0, 0].set(True))
        requests = point(one, requests, jnp.full((1, slots), -1, jnp.int32).at[0, 0].set(handle))
        check_trunk(self.module, self.trunk)
        out = _module_program(self.module.config, self.module.known, self.module.verb)(
            self.module.params, one.encoded, one.masks, requests)
        host = jax.device_get(out.head_logits)
        return BodyStep({head.name: int(np.argmax(host[head.name][0])) for head in head_layout("gen7")
                         if scope_of(head.name) == self.module.scope})


__all__ = ["AgentView", "Lowering", "ModuleBody", "OwnPose", "StateView", "encode_contexts", "entity_rows",
           "host_inputs", "module_factors", "own_pose_channel", "point", "step_lanes", "tick"]
