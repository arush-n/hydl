"""The body's path on token schema v3 (step 1b of ``docs/VIRTUAL_HUMAN.md`` section 13, approved 2026-10-22): each
tick the state layer (``state/layer.py``) builds the agent's tokens, the trunk runs once for every agent with its latent
memory, and each skill module reads the result; a request's entity handle points at the entity token row that holds it,
perceived or remembered. The v1 path is unchanged (``model/freezing.py`` ``encode_contexts``, ``VerbBody``).

- :func:`tick`: the state layer, then the trunk once (the latent memory carried, and reset where a lane starts over).
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
from experimental.neural_agent.skill_agent.model.encoder import EncoderOutput, apply_encoder, initial_memory
from experimental.neural_agent.skill_agent.model.generalist import ARG_GROUPS, GeneralistRequests
from experimental.neural_agent.skill_agent.model.skill_module import SkillModule, Trunk, check_trunk, module_outputs
from experimental.neural_agent.skill_agent.runtime.combiner import AgentContext, BodyStep
from experimental.neural_agent.skill_agent.state import layer
from experimental.neural_agent.skill_agent.training.blocks import HEAD_NAMES, HEAD_SIZES, HEAD_STARTS, NEUTRAL


class StateView(NamedTuple):
    """One tick of B agents as every skill module reads it."""

    encoded: EncoderOutput  # the trunk's output, once for all agents
    masks: dict[str, jax.Array]  # token type -> bool[B, N]: which token slots hold a row
    entity_ids: jax.Array  # int32[B, E]: the id of each entity token row (perceived or remembered), -1 for padding


def tick(trunk: Trunk, state: layer.AgentState, observation: Any, *, last_action: jax.Array,
         reset: jax.Array | None = None, **step: Any) -> tuple[StateView, layer.AgentState]:
    """One tick for B agents: ``layer.step`` (``step`` takes its keyword arguments: ``tags``, ``capacities`` and the
    senses' inputs), then the trunk on its tokens. A lane that ``reset`` starts its latent memory from the trunk's
    initial slots."""

    tokens, state = layer.step(observation, state, last_action=last_action, reset=reset, **step)
    latent = state.latent
    if trunk.config.memory_slots:
        initial = initial_memory(trunk.params, trunk.config, state.entity_ids.shape[0])
        if latent is None:
            latent = initial
        elif reset is not None:
            latent = jnp.where(reset[:, None, None], initial, latent)
    encoded = apply_encoder(trunk.params, tokens, trunk.config, memory=latent)
    view = StateView(encoded, {name: valid for name, (_, valid) in tokens.items()}, state.entity_ids)
    return view, state._replace(latent=encoded.memory)


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


__all__ = ["AgentView", "Lowering", "ModuleBody", "StateView", "encode_contexts", "entity_rows", "module_factors",
           "point", "tick"]
