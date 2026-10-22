"""One tick of the state layer (``docs/VIRTUAL_HUMAN.md`` section 3; the token contract is ``docs/PERCEPTION.md``
section 10): the agent's state brought up to date from its senses, and read by the body as tokens.

The stateless parts are ``state/senses.py`` (the legal observation to this tick's rows) and
``state/frames.py`` (space). The state is :class:`AgentState`: the agent's action history
(``state/action_history.py``), its stored memory (``state/memory.py``: the spatial map and the salient store), its
persona and emotional state (``state/persona.py``, ``state/affect.py``; ``docs/VIRTUAL_HUMAN.md``) and the
trunk's latent memory (``model/encoder.py``, ``memory_slots``; the trunk call fills it, this module only carries and
stores it). One tick, :func:`step`:

1. the action history takes the action the agent sent last tick;
2. the senses read this tick's perceived rows;
3. the emotional state appraises them, with the persona's gains;
4. the stores take them, and the middle layer's live tags;
5. the stores give back what is remembered but not perceived;
6. every entity token, perceived or remembered, gets the agent's feelings toward it (``feel_*``);
7. everything is packed into the ``skill_tokens_v3`` token sets the trunk reads, the persona and affect tokens
   with them.

:func:`save` and :func:`load` write and read a state as one ``.npz`` (memory "stored as well", owner 2026-10-20 night).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.contract import tokens_v3 as v3
from experimental.neural_agent.skill_agent.state import affect as feeling
from experimental.neural_agent.skill_agent.state import memory as stores
from experimental.neural_agent.skill_agent.state.action_history import (
    ActionHistory,
    act_features,
    initial_history,
    update_history,
)
from experimental.neural_agent.skill_agent.state.senses import append_rows, pack, perceive_rows
from experimental.neural_agent.skill_agent.state.persona import neutral_persona


class AgentState(NamedTuple):
    """The neural layer's perception state for B agents."""

    history: ActionHistory
    memory: stores.SpatialMemory
    latent: jax.Array | None = None  # float32[B, S, W]: the trunk's last ``EncoderOutput.memory``
    persona: jax.Array | None = None  # float32[B, P]: the agent's traits (``tokens_v3.PERSONA_TRAITS``), fixed
    affect: feeling.AffectState | None = None


def initial_state(batch: int, latent: jax.Array | None = None, persona: jax.Array | None = None) -> AgentState:
    """The state at an episode's start (``latent``: the trunk's initial slots, ``encoder.initial_memory``;
    ``persona``: float32[B, P], the population's mean when None)."""

    persona = neutral_persona(batch) if persona is None else jnp.asarray(persona, jnp.float32)
    return AgentState(initial_history(batch), stores.initial_memory(batch), latent, persona,
                           feeling.initial_affect(persona))


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
    affect = feeling.update(state.affect, perceived, state.persona, reset)
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
    rows["persona"] = (jnp.pad(state.persona, ((0, 0), (0, 1)))[:, None, :], one)
    rows["affect"] = (jnp.pad(feeling.affect_features(affect), ((0, 0), (0, 1)))[:, None, :], one)
    tokens = pack(rows, batch, capacities)
    return tokens, AgentState(history, memory, state.latent, state.persona, affect)


def save(path: str | Path, state: AgentState) -> None:
    """Write ``state`` to ``path`` (``.npz``), each array under its place in the state."""

    leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    np.savez(path, **{jax.tree_util.keystr(key): np.asarray(value) for key, value in leaves})


def load(path: str | Path, like: AgentState) -> AgentState:
    """Read a state written by :func:`save`, shaped as ``like`` (a state of the same batch and capacities)."""

    with np.load(path) as stored:
        paths, treedef = jax.tree_util.tree_flatten_with_path(like)
        values = []
        for key, template in paths:
            value = stored[jax.tree_util.keystr(key)]
            if value.shape != np.shape(template):
                raise ValueError(f"{jax.tree_util.keystr(key)} is {value.shape}, expected {np.shape(template)}")
            values.append(jnp.asarray(value))
    return jax.tree_util.tree_unflatten(treedef, values)


__all__ = ["AgentState", "initial_state", "load", "save", "step"]
