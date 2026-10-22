"""One tick of the state layer (``docs/VIRTUAL_HUMAN.md`` section 3; the token contract is ``docs/PERCEPTION.md``
section 10): the agent's state brought up to date from its senses, and read by the body as tokens.

The stateless parts are ``state/senses.py`` (the legal observation to this tick's rows) and
``state/frames.py`` (space). The state is :class:`AgentState`: the agent's action history
(``state/action_history.py``), its stored memory (``state/memory.py``: the spatial map and the salient store), its
variables (``state/registry.py``: the persona, ``state/persona.py``, and the emotional state the affect rules publish,
``state/affect.py``; ``docs/VIRTUAL_HUMAN.md``), the affect rules' own working state, and the trunk's latent memory (``model/encoder.py``, ``memory_slots``; the trunk call fills it, this module only carries and
stores it). One tick, :func:`step`:

1. the action history takes the action the agent sent last tick;
2. the senses read this tick's perceived rows;
3. the emotional state appraises them, with the persona's gains, and is published to the variables;
4. the stores take them, and the middle layer's live tags;
5. the stores give back what is remembered but not perceived;
6. every entity token, perceived or remembered, gets the agent's feelings toward it (``feel_*``);
7. everything is packed into the ``skill_tokens_v3`` token sets the trunk reads, with the persona and affect
   tokens, whose columns are the body-read variables (``tokens_v3.PERSONA_FEATURES``, ``AFFECT_FEATURES``).

:func:`save` and :func:`load` write and read a state as one ``.npz`` (memory "stored as well", owner 2026-10-20 night),
with the variables' names and the token layout, so a save migrates by name to a later registry or schema
(``docs/VIRTUAL_HUMAN.md`` section 14, rules 2 and 4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract import tokens_v3 as v3
from experimental.neural_agent.skill_agent.contract import variables
from experimental.neural_agent.skill_agent.state import registry
from experimental.neural_agent.skill_agent.state import affect as feeling
from experimental.neural_agent.skill_agent.state import memory as stores
from experimental.neural_agent.skill_agent.state.action_history import (
    ActionHistory,
    act_features,
    initial_history,
    update_history,
)
from experimental.neural_agent.skill_agent.state.senses import append_rows, fit_ids, pack, perceive_rows
from experimental.neural_agent.skill_agent.state.persona import TRAITS, neutral_persona


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
                      affect, slots["entity"], slots, background)


def step(observation: Any, state: AgentState, *, last_action: jax.Array,
             reset: jax.Array | None = None, tags: stores.Tags | None = None,
             capacities: Mapping[str, int] = v3.CAPACITIES_V3, **adapter: Any
             ) -> tuple[dict[str, tuple[jax.Array, jax.Array]], AgentState]:
    """One tick (module docstring): the token sets and the next state.

    Args:
        observation: The learner observation.
        state: Last tick's state.
        last_action: int32[B, 21], the gen-7 action the agent sent last tick.
        reset: bool[B], lanes that start a new episode this tick (their history, stores and emotional state start
            over, the persona stays; the trunk's latent memory is reset by its caller).
        tags: The middle layer's live tags, or None.
        capacities: Slots per type.
        **adapter: Passed to ``adapter_v3.perceive_rows`` (``body_yaw_degrees``, ``tables``, ``action_surface``,
            ``block_reach_blocks``, ``light``, ``view_half_angle_degrees``).
    """

    history = update_history(state.history, last_action, reset)
    perceived = perceive_rows(observation, inventory_capacity=capacities["inventory_slot"], **adapter)
    affect = feeling.update(state.affect, perceived, registry.take(state.variables, TRAITS), reset)
    values = registry.put(state.variables, v3.AFFECT_FEATURES, feeling.affect_features(affect))
    memory = stores.update(state.memory, perceived, tags, reset)
    remembered = stores.read(memory, perceived.pose)
    rows = dict(perceived.rows)
    rows["act"] = (act_features(history), jnp.ones(history.last_choice.shape, jnp.bool_))
    entity_ids = perceived.entity_ids[:, :rows["entity"][0].shape[1]]
    if remembered is not None:
        for name in ("entity", "geometry", "block_candidate"):
            rows[name] = append_rows(rows[name], getattr(remembered, name))
        entity_ids = jnp.concatenate([entity_ids, jnp.where(remembered.entity[1], memory.entity.key_id, -1)], axis=1)
    features, valid = rows["entity"]
    feel = v3.EntityTokenV3.FEATURES.index(f"feel_{v3.FEELINGS[0]}")
    feelings = jnp.where(valid[..., None], feeling.feelings_for(affect, jnp.where(valid, entity_ids, -1)), 0.0)
    rows["entity"] = (features.at[..., feel:feel + len(v3.FEELINGS)].set(feelings), valid)
    batch = observation.base.self_f32.shape[0]
    one = jnp.ones((batch, 1), jnp.bool_)
    for name in ("persona", "affect"):  # the body-read variables, the token's ``marked`` 0
        columns = registry.take(values, v3.TOKEN_TYPES_V3[name].FEATURES[:-1])
        rows[name] = (jnp.pad(columns, ((0, 0), (0, 1)))[:, None, :], one)
    tokens = pack(rows, batch, capacities)
    ids = fit_ids(entity_ids, valid, capacities["entity"])  # aligned with tokens["entity"], perceived and remembered
    slots = stores.object_slots(memory, tokens, ids, perceived.pose)
    return tokens, AgentState(history, memory, state.latent, values, affect, ids, slots, state.background)


_STORE_TYPES: Mapping[str, str] = {"geometry": "geometry", "entity": "entity", "block": "block_candidate"}
"""Each store of ``SpatialMemory`` and the token type of its rows."""


def _layout() -> dict[str, list[str]]:
    return {name: list(cls.FEATURES) for name, cls in v3.TOKEN_TYPES_V3.items()}


def save(path: str | Path, state: AgentState) -> None:
    """Write ``state`` to ``path`` (``.npz``), each array under its place in the state, with the variables' names
    (``__variables__``) and the token layout (``__layout__``) and their fingerprints (``__fingerprint__``)."""

    leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    layout = _layout()
    np.savez(path, **{jax.tree_util.keystr(key): np.asarray(value) for key, value in leaves},
             __variables__=np.asarray(variables.NAMES), __layout__=np.asarray(json.dumps(layout)),
             __fingerprint__=np.asarray([variables.fingerprint(list(variables.NAMES)), variables.fingerprint(layout)]))


def load(path: str | Path, like: AgentState, *, maps: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
         dropped: Mapping[str, Sequence[str]] | None = None) -> AgentState:
    """Read a state written by :func:`save`, shaped as ``like`` (a state of the same batch and capacities).

    A save made under another variable list or token layout is migrated by name (``docs/VIRTUAL_HUMAN.md`` section 14):
    a variable it lacks takes its default, and a stored row's new column reads 0. A variable or column that is gone
    must be merged into a new one by ``maps`` or listed in ``dropped``, both keyed by ``"variables"`` or a stored token
    type (``variables.transfer``). A part of the state the save lacks (added since, as the object latents and the
    background summary on 2026-10-22) takes ``like``'s value.
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
