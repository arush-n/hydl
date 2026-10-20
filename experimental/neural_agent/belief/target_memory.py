"""The belief's remembered target position (task #96, 2026-10-19).

While the target is seen, the belief records where it is in the world frame. The agent's own position is the gen-7
perception channel's ``self_position`` (``sim/gen7/PERCEPTION_V2.md``). The target's offset is read from the legal
observation's target row, which the encoder measures in the agent's body frame (``observation/v1/runtime/encoder.py``,
``_target_features`` with ``state.yaw[:, AGENT_ENTITY]``); the channel publishes that body yaw as ``body_yaw_sin`` and
``body_yaw_cos``. While the target is unseen the record is held, and ``ticks_since_target_seen`` gives its age.

Without the channel nothing is recorded. Gen 5's legal observation has no self position, and its ``self_f32``
``yaw_sin`` and ``yaw_cos`` are the HEAD yaw, while the target row is measured from the BODY yaw. Using the head yaw
would be wrong whenever the head turns away from the body, so the position stays unknown there and the requests keep
naming the entity, as before.

Formulas, with R = NEARBY_ENTITY_RADIUS_BLOCKS, (s, c) = (body_yaw_sin, body_yaw_cos), and f, r, u the target row's
relative_forward, relative_right and relative_up (fractions of R, clipped by the encoder to [-1, 1]):

    forward = (-s, 0, -c),  right = (c, 0, -s),  up = (0, 1, 0)
    target  = self_position + R * (f * forward + r * right + u * up)

A sighting is recorded only when the observation is valid, the target is visible, the channel is present and none of
f, r, u is clipped (each |x| < 1).

Requests (:func:`target_arguments`): for a verb whose schema takes the entity or the position (``any_of``), the remembered
position while the target is unseen and the memory knows it; a verb whose schema needs the entity keeps it. No shared
``requests.ARGUMENT_SCHEMAS`` entry takes that ``any_of`` yet (task #96: proposed, held back because the skill layer
would lose its entity check), so the request rule is inert in the shipped schemas. The memory is not a drive input: the
belief's features, known flags and latent do not change.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from adk.policy import fields
from experimental.neural_agent.belief.features import VISIBLE_THRESHOLD
from experimental.neural_agent.contracts import feature_index
from hytalegym.jax.combat.observation.v1.schema.contract import (
    NEARBY_ENTITY_RADIUS_BLOCKS,
)

_REL_FORWARD = fields.field_index("target_f32", "relative_forward")
_REL_RIGHT = fields.field_index("target_f32", "relative_right")
_REL_UP = fields.field_index("target_f32", "relative_up")
_VISIBLE = feature_index("target_visible")
_ENTITY_OR_POSITION: tuple[str, str] = ("entity", "position")


class TargetMemory(NamedTuple):
    """The remembered target position, per lane.

    Shapes, with B lanes:
        position: float32[B, 3], the world-frame (x, y, z) of the last recorded sighting; zero where unknown.
        known: bool[B], False until a sighting is recorded (never, without the perception channel).
    """

    position: jax.Array
    known: jax.Array


def initial_target_memory(lanes: int) -> TargetMemory:
    """Return the episode-start memory: nothing known."""

    return TargetMemory(
        position=jnp.zeros((lanes, 3), dtype=jnp.float32),
        known=jnp.zeros((lanes,), dtype=jnp.bool_),
    )


def sighting(legal_observation: Any, valid: jax.Array, visible: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return this tick's sighting as ``(position float32[B, 3], recorded bool[B])``.

    Args:
        legal_observation: The legal observation. With the perception channel it also carries ``perception_v2``.
        valid: bool[B], ``legal_observation.valid``.
        visible: bool[B], the belief's target-visible predicate (``combat_f32.target_visible > 0.5``).
    """

    channel = getattr(legal_observation, "perception_v2", None)
    lanes = valid.shape[0]
    if channel is None:
        return jnp.zeros((lanes, 3), dtype=jnp.float32), jnp.zeros((lanes,), dtype=jnp.bool_)
    target = legal_observation.base.target_f32
    forward, right, up = target[:, _REL_FORWARD], target[:, _REL_RIGHT], target[:, _REL_UP]
    sin, cos = channel.body_yaw_sin, channel.body_yaw_cos
    radius = jnp.float32(NEARBY_ENTITY_RADIUS_BLOCKS)
    offset = radius * jnp.stack((forward * -sin + right * cos, up, -forward * cos - right * sin), axis=-1)
    position = channel.self_position.astype(jnp.float32) + offset.astype(jnp.float32)
    unclipped = (jnp.abs(forward) < 1.0) & (jnp.abs(right) < 1.0) & (jnp.abs(up) < 1.0)
    return position, valid & visible & unclipped


def update_target_memory(previous: TargetMemory, position: jax.Array, recorded: jax.Array) -> TargetMemory:
    """Take the sightings where ``recorded`` holds and keep the previous memory elsewhere."""

    return TargetMemory(
        position=jnp.where(recorded[:, None], position, previous.position),
        known=previous.known | recorded,
    )


def host_target_memory(belief_update: Any) -> TargetMemory | None:
    """The belief's memory as numpy arrays, for the decision layer's request builder.

    The memory is held in the belief's carry (``SharedBeliefCarry.target_memory``) and reaches the decision layer as
    ``BeliefUpdate.state``. ``SharedBeliefState`` (``contracts.py``) is not edited by this change: sub-module work does
    not edit ``contracts.py`` (``neural_agent/SKILL.md`` rule 1). Returns None when there is no memory to read.
    """

    memory = getattr(getattr(belief_update, "state", None), "target_memory", None)
    if memory is None:
        return None
    return TargetMemory(
        position=np.asarray(memory.position, dtype=np.float64),
        known=np.asarray(memory.known, dtype=np.bool_),
    )


def target_arguments(
    schema: Any, features_row: Any, memory: TargetMemory | None, lane: int, entity: int
) -> tuple[int | None, tuple[float, float, float] | None]:
    """Return ``(entity, position)`` for a request of ``schema`` on ``lane``: what the verb takes in this state.

    A verb that needs the entity keeps it. A verb that takes the entity or the position gives the remembered
    position while the target is unseen and the memory knows it, and the entity otherwise. Any other verb, and an
    unknown one, takes neither.
    """

    if schema is None:
        return None, None
    if schema.entity:
        return entity, None
    if _ENTITY_OR_POSITION not in schema.any_of:
        return None, None
    seen = float(features_row[_VISIBLE]) > VISIBLE_THRESHOLD
    if not seen and memory is not None and bool(memory.known[lane]):
        return None, tuple(float(value) for value in memory.position[lane])
    return entity, None
