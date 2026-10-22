"""The legal observation as the skill layer's token sets: the adapter from the simulator to ``model/encoder.py``.

Input: the learner observation that ``ArsenalEnvironment.step`` returns, a ``LearnerCombatObservationV3`` pytree
(``observation/v3/schema/types.py``). The adapter never reads the simulator state (``CombatState``): its signature
takes no state. Every value comes from ``observation``, plus one explicit scalar, the body yaw (see Frame).

Output: ``{type: (features float32[B, N_t, F_t], mask bool[B, N_t])}`` for the types in ``encoder.TRUNK_TYPES``, the
shape ``encoder.stack_batch`` returns and ``encoder.apply_encoder`` reads.

Tokens filled
-------------
* ``self``: slot 0, valid for every agent. Health, the stamina, mana, signature-energy and oxygen meters, grounded,
  yaw as sin and cos of the body yaw, the head's yaw offset from the body, pitch, and speed.
* ``entity``: slot 0 is the opponent, read from the observation's nearby-entity row (slot 0 of ``base.entity_f32``).
  It is valid exactly when the observation publishes that row. Slots 1 onward are invalid.
* ``block_candidate``, ``inventory_slot``, ``geometry``, ``kit``: all invalid, features zero. The flat world has no
  blocks, and inventory and kit come later. The observation's block, inventory and geometry tokens are not read.

Perceptibility
--------------
The nearby-entity row is published only when the opponent is perceptible: alive, on the actor's line of sight, at
planar distance <= ``params.sensor_range`` (16 blocks; ``reset.py:789-791``), and at 3-D distance <= 24 blocks
(``NEARBY_ENTITY_RADIUS_BLOCKS``; ``observation/v1/runtime/encoder.py:113``). The entity mask is that row's presence.
Field of view does not gate it: the two-entity path that the flat world takes has no view sector
(``arsenal/environment.py:2456-2459``, ``2699-2740``).

Frame
-----
The skill layer's frame is the BODY frame: forward = (-sin yaw, 0, -cos yaw), right = (cos yaw, 0, -sin yaw), up = y,
with yaw the body yaw in degrees.

* Entity positions and velocities need no rotation. The nearby-entity row is built from the body yaw,
  ``state.yaw[:, AGENT_ENTITY]`` (``observation/v1/runtime/encoder.py:108``, ``:178``). Measured at tick 1, body yaw 90
  and head yaw 45, for a world offset (-3, 0, -4): the row gives (3, 4, 0), the body-frame vector. The head-frame
  vector would be (4.95, 0.71, 0).
* Two observation fields are in the HEAD frame, and the adapter reads neither for positions: the self row's yaw
  sin/cos (``observation/v1/runtime/encoder.py:323``; measured sin 0.707 at head 45) and the 24-value combat prefix
  (``reset.py:485``, ``:513``). The self row's sin and cos are the head yaw, so the adapter reads them for one thing
  only: the self token's ``head_yaw_offset``, the head yaw minus the body yaw.
* The body yaw is not published as a value anywhere in the observation. So the self token's yaw comes from the
  caller's ``body_yaw_degrees``: the agent's own heading, which the agent knows. This is the one input beyond the
  observation. Without it the self token would have to carry the head yaw, which is the wrong frame.

Units
-----
* Entity position: blocks. The row is divided by ``NEARBY_ENTITY_RADIUS_BLOCKS`` (24); the adapter multiplies back.
* Entity velocity: the observation's unit, the relative velocity over ``params.target_chase_speed``. That scale is not
  published, so the values are not blocks per second.
* Meters: the observation's fractions, clipped to [0, 1]. Stamina is over the observation's own minimum and maximum.
* Speed: horizontal speed over ``params.agent_maximum_gait_speed``, the observation's unit. It is the magnitude of the
  head-frame projections, so it does not depend on the frame.
* Pitch: radians of the head pitch (the observation publishes pitch / 90).

Gaps: features the legal observation does not provide
-----------------------------------------------------
* Body yaw: not published; supplied by the caller (see Frame).
* Held-item durability: not published; 0.
* Mana, gen 5 only: the stock engine's flat world has no mana resource (the observation's mask is False), so 0. On gen
  7 every reset carries Mana 100 / 100 since 65a74bb (as the live 0.7.2 body spawns), so the meter reads 1.0 at spawn;
  anything trained on gen 7 before that commit saw 0 here.
* Role: the nearby-entity row publishes only ``hostile``, which the writer sets to 1 for every row
  (``observation/v1/runtime/encoder.py:132``). Every opponent is therefore ``enemy``. ``ally``, ``animal``,
  ``merchant`` and ``player`` are never produced.
* Last-seen age and last-seen position: the observation has no memory. A visible opponent has age 0, and a hidden
  one is invalid.
* Marked: 0. This adapter has no instructed target.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.contract.surface import ENTITY_ROLES
from experimental.neural_agent.skill_agent.model.encoder import TRUNK_TYPES
from hytalegym.jax.combat.mechanics import (
    RESOURCE_MANA,
    RESOURCE_OXYGEN,
    RESOURCE_SIGNATURE_ENERGY,
    RESOURCE_STAMINA,
)
from hytalegym.jax.combat.observation.v1.schema.contract import (
    ENTITY_FLOAT_FEATURES,
    NEARBY_ENTITY_RADIUS_BLOCKS,
    SELF_FLOAT_FEATURES,
)
from hytalegym.jax.combat.observation.v3.schema.types import LearnerCombatObservationV3
from hytalegym.jax.combat.types import AGENT_ENTITY

_ENTITY_INDEX: dict[str, int] = {name: index for index, name in enumerate(ENTITY_FLOAT_FEATURES)}
_SELF_INDEX: dict[str, int] = {name: index for index, name in enumerate(SELF_FLOAT_FEATURES)}
#: The nearby-entity row's slot that holds the opponent. The flat world has one opponent.
_OPPONENT_SLOT = 0


def legal_tokens(
    observation: LearnerCombatObservationV3,
    *,
    capacities: Mapping[str, int],
    body_yaw_degrees: jax.Array,
) -> dict[str, tuple[jax.Array, jax.Array]]:
    """Build the token sets of B agents from their legal observation.

    Args:
        observation: The learner observation of ``ArsenalEnvironment.step`` (``observation.base`` and the resource
            fields are read). It is never the simulator state.
        capacities: Slots per token type. Must name every type in ``encoder.TRUNK_TYPES``. ``entity`` and ``self``
            need at least one slot.
        body_yaw_degrees: float32[B], each agent's body yaw in degrees. The observation does not publish it.

    Returns:
        ``{type: (features float32[B, N_t, F_t], mask bool[B, N_t])}`` in the encoder's layout.

    Raises:
        ValueError: If ``body_yaw_degrees`` is not shaped (B,), or ``entity`` or ``self`` has no slot.
    """

    batch = observation.valid.shape[0]
    for name in ("entity", "self"):
        if capacities[name] < 1:
            raise ValueError(f"capacities[{name!r}] must be at least 1, got {capacities[name]}")
    body_yaw = jnp.asarray(body_yaw_degrees, dtype=jnp.float32)
    if body_yaw.shape != (batch,):
        raise ValueError(f"body_yaw_degrees must have shape ({batch},), got {body_yaw.shape}")

    opponent, opponent_present = _opponent_token(observation.base)
    agent = _self_token(observation, body_yaw)
    tokens: dict[str, tuple[jax.Array, jax.Array]] = {}
    for name, cls in TRUNK_TYPES:
        capacity = capacities[name]
        features = jnp.zeros((batch, capacity, len(cls.FEATURES)), dtype=jnp.float32)
        mask = jnp.zeros((batch, capacity), dtype=jnp.bool_)
        if name == "entity":
            features = features.at[:, 0, :].set(jnp.where(opponent_present[:, None], opponent, 0.0))
            mask = mask.at[:, 0].set(opponent_present)
        elif name == "self":
            features = features.at[:, 0, :].set(agent)
            mask = mask.at[:, 0].set(True)
        tokens[name] = (features, mask)
    return tokens


def _opponent_token(base) -> tuple[jax.Array, jax.Array]:
    """The opponent's features [B, 16] (``EntityToken.FEATURES`` order) and its presence [B]."""

    row = base.entity_f32[:, _OPPONENT_SLOT, :]
    present = base.entity_mask[:, _OPPONENT_SLOT].astype(jnp.bool_)
    hostile = row[:, _ENTITY_INDEX["hostile"]] > 0.5
    roles = jnp.zeros((row.shape[0], len(ENTITY_ROLES)), dtype=jnp.float32)
    roles = roles.at[:, ENTITY_ROLES.index("enemy")].set(hostile.astype(jnp.float32))
    roles = roles.at[:, ENTITY_ROLES.index("unknown")].set((~hostile).astype(jnp.float32))
    position = jnp.stack(
        [row[:, _ENTITY_INDEX[name]] for name in ("relative_forward", "relative_right", "relative_up")], axis=1
    ) * NEARBY_ENTITY_RADIUS_BLOCKS
    velocity = jnp.stack(
        [
            row[:, _ENTITY_INDEX[name]]
            for name in ("relative_velocity_forward", "relative_velocity_right", "relative_velocity_up")
        ],
        axis=1,
    )
    visible = row[:, _ENTITY_INDEX["visible"]][:, None]
    # No memory in the observation: an opponent that is visible this tick was seen this tick.
    last_seen_age = jnp.zeros_like(visible)
    health = row[:, _ENTITY_INDEX["health_fraction"]][:, None]
    marked = jnp.zeros_like(visible)
    features = jnp.concatenate([roles, position, velocity, visible, last_seen_age, health, marked], axis=1)
    return features, present


def _self_token(observation: LearnerCombatObservationV3, body_yaw: jax.Array) -> jax.Array:
    """The agent's features [B, 13] (``SelfToken.FEATURES`` order); yaw is the body yaw, supplied by the caller."""

    row = observation.base.self_f32
    meters = observation.resource_f32[:, AGENT_ENTITY, :]
    yaw = jnp.deg2rad(body_yaw)
    head = jnp.arctan2(row[:, _SELF_INDEX["yaw_sin"]], row[:, _SELF_INDEX["yaw_cos"]])  # the head yaw, radians
    offset = head - yaw
    head_yaw_offset = jnp.arctan2(jnp.sin(offset), jnp.cos(offset))  # wrapped to [-pi, pi]
    forward = row[:, _SELF_INDEX["forward_velocity"]]
    right = row[:, _SELF_INDEX["right_velocity"]]
    zeros = jnp.zeros_like(yaw)

    def meter(resource: int) -> jax.Array:
        return jnp.clip(meters[:, resource], 0.0, 1.0)

    features = jnp.stack(
        [
            jnp.clip(row[:, _SELF_INDEX["health_fraction"]], 0.0, 1.0),
            meter(RESOURCE_STAMINA),
            meter(RESOURCE_MANA),
            meter(RESOURCE_SIGNATURE_ENERGY),
            meter(RESOURCE_OXYGEN),
            zeros,  # held durability: not published
            row[:, _SELF_INDEX["grounded"]],
            jnp.sin(yaw),
            jnp.cos(yaw),
            row[:, _SELF_INDEX["pitch"]] * (math.pi / 2.0),
            jnp.sqrt(forward * forward + right * right),
            head_yaw_offset,
            zeros,  # marked
        ],
        axis=1,
    )
    return features.astype(jnp.float32)
