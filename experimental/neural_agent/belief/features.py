"""Instantaneous shared-belief features, read from the legal structured observation.

``observation_readings`` derives the eight features that need no history from
``ActorPolicyInput.legal_observation``, a ``LearnerCombatObservationV3``. The
two tick counters need history, so ``core.py`` adds them. Privileged simulator
state is never read. Column sources and formulas are recorded in SKILL.md.

Known-flag rules:

* ``valid`` is ``legal_observation.valid``. When it is False the upstream
  encoder has zeroed every base leaf, so no feature is known in that lane.
* The encoder publishes target fields only while ``combat_f32.target_visible``
  is set. A hidden target encodes as all-zero target fields (spec.py,
  ``target_evidence``). Target features are known only when the target is visible.
* The speed features need a scale that the observation does not publish: the
  gait and chase speeds in blocks per second. They are known only when the
  caller supplies the scale. Without it they read zero and are marked unknown.
* The encoder clips ``target_f32`` to [-1, 1]. The closing speed is computed from
  clipped inputs in that case, so it is marked unknown rather than published.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from adk.policy import fields
from experimental.neural_agent.contracts import BELIEF_FEATURE_NAMES
from hytalegym.jax.combat.clock import HYTALE_TICKS_PER_SECOND
from hytalegym.jax.combat.observation.v1.schema.contract import (
    NEARBY_ENTITY_RADIUS_BLOCKS,
)
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerCombatObservationV3,
)

# Columns, resolved once from the live contract names (fields.py fails at
# import if a name is unknown, so a renamed column cannot shift silently).
_SELF_HEALTH = fields.field_index("self_f32", "health_fraction")
_SELF_FORWARD_VELOCITY = fields.field_index("self_f32", "forward_velocity")
_SELF_RIGHT_VELOCITY = fields.field_index("self_f32", "right_velocity")
_TARGET_RELATIVE_FORWARD = fields.field_index("target_f32", "relative_forward")
_TARGET_RELATIVE_RIGHT = fields.field_index("target_f32", "relative_right")
_TARGET_RELATIVE_VELOCITY_FORWARD = fields.field_index(
    "target_f32", "relative_velocity_forward"
)
_TARGET_RELATIVE_VELOCITY_RIGHT = fields.field_index(
    "target_f32", "relative_velocity_right"
)
_TARGET_DISTANCE = fields.field_index("target_f32", "distance")
_TARGET_HEALTH = fields.field_index("target_f32", "health_fraction")
_TARGET_BEARING_SIN = fields.field_index("target_f32", "bearing_sin")
_TARGET_BEARING_COS = fields.field_index("target_f32", "bearing_cos")
_COMBAT_TARGET_VISIBLE = fields.field_index("combat_f32", "target_visible")

#: The encoder's visibility predicate, ``perceptible = target_visible > 0.5``.
VISIBLE_THRESHOLD: float = 0.5
#: Planar distance, in blocks, below which the closing direction is undefined.
#: It matches the encoder's own ``safe_planar`` floor.
PLANAR_EPSILON_BLOCKS: float = 1.0e-6

_INSTANTANEOUS_NAMES: tuple[str, ...] = tuple(
    name
    for name in BELIEF_FEATURE_NAMES
    if name not in ("ticks_since_damage", "ticks_since_target_seen")
)


class ObservationReadings(NamedTuple):
    """Features that need no history, and the predicates the counters use.

    Shapes, with B lanes:
        valid: bool[B], ``legal_observation.valid``.
        visible: bool[B], ``combat_f32.target_visible > 0.5``.
        self_health: float32[B], ``self_f32.health_fraction``, the counter input.
        values: eight names to float32[B], zero where the feature is unknown.
        known: the same eight names to bool[B].
    """

    valid: jax.Array
    visible: jax.Array
    self_health: jax.Array
    values: dict[str, jax.Array]
    known: dict[str, jax.Array]


def _require_columns(array: Any, group: str) -> None:
    """Raise ValueError unless ``array`` is float (B, width) for ``group``."""

    width = len(fields.field_names(group))
    if getattr(array, "ndim", None) != 2 or array.shape[1] != width:
        raise ValueError(f"{group} must have shape (B, {width}), got {array.shape}")


def positive_scale(name: str, scale: float | None) -> float | None:
    """Return ``scale`` as a float that is positive and finite, or None.

    Args:
        name: Field name used in error messages.
        scale: A real number in blocks per second, or None when unknown.

    Returns:
        The scale as a Python float, or None when ``scale`` is None.

    Raises:
        TypeError: If ``scale`` is a bool or not a real number.
        ValueError: If ``scale`` is not positive and finite.
    """

    if scale is None:
        return None
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError(f"{name} must be a real number or None, got {type(scale).__name__}")
    value = float(scale)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


def observation_readings(
    legal_observation: LearnerCombatObservationV3,
    *,
    maximum_gait_speed: float | None,
    chase_speed: float | None,
) -> ObservationReadings:
    """Derive the eight history-free belief features from one legal observation.

    Formulas, with ``r = NEARBY_ENTITY_RADIUS_BLOCKS = 24.0``, ``T =
    HYTALE_TICKS_PER_SECOND = 30`` and ``e = PLANAR_EPSILON_BLOCKS``:

    * ``target_visible = 1[target_visible > 0.5]``, known iff ``valid``.
    * ``target_distance_blocks = r * target_f32.distance``, known iff visible.
    * ``target_bearing_sin = target_f32.bearing_sin``, known iff visible.
    * ``target_bearing_cos = target_f32.bearing_cos``, known iff visible.
    * ``self_health_fraction = self_f32.health_fraction``, known iff ``valid``.
    * ``target_health_fraction = target_f32.health_fraction``, known iff visible.
    * ``self_speed_blocks_per_tick = hypot(fwd, right) * V_gait / T``, where
      ``fwd`` and ``right`` are ``self_f32.forward_velocity`` and
      ``self_f32.right_velocity``. Known iff ``valid`` and ``V_gait`` is given.
    * ``closing_speed_blocks_per_tick = -(a . v) / |a| / T``, where
      ``a = r * (rel_fwd, rel_right)`` in blocks, ``v = V_chase * (vel_fwd,
      vel_right)`` in blocks per second, and ``|a| = hypot(a)``. Known iff
      visible, ``valid``, ``|a| > e``, ``V_chase`` is given, and none of the
      four encoder inputs ``rel_fwd, rel_right, vel_fwd, vel_right`` is clipped
      (each ``|x| < 1`` in ``target_f32``). This is the
      planar (horizontal) radial rate, positive when the target approaches.

    Args:
        legal_observation: The ``LearnerCombatObservationV3`` of
            ``ActorPolicyInput.legal_observation``.
        maximum_gait_speed: ``agent_maximum_gait_speed`` in blocks per second,
            the denominator of ``self_f32.forward_velocity`` and
            ``right_velocity``, or None when unknown.
        chase_speed: ``target_chase_speed`` in blocks per second, the
            denominator of ``target_f32.relative_velocity_*``, or None.

    Returns:
        ObservationReadings with arrays of shape (B,) for each entry.

    Raises:
        TypeError: If ``legal_observation`` is not a ``LearnerCombatObservationV3``
            or a scale is not a real number.
        ValueError: If a base group has the wrong width, ``valid`` has the wrong
            shape, or a scale is not positive and finite.
    """

    if not isinstance(legal_observation, LearnerCombatObservationV3):
        raise TypeError("legal_observation must be a LearnerCombatObservationV3")
    gait = positive_scale("maximum_gait_speed", maximum_gait_speed)
    chase = positive_scale("chase_speed", chase_speed)
    base = legal_observation.base
    _require_columns(base.self_f32, "self_f32")
    _require_columns(base.target_f32, "target_f32")
    _require_columns(base.combat_f32, "combat_f32")
    lanes = base.self_f32.shape[0]
    valid = jnp.asarray(legal_observation.valid, dtype=jnp.bool_)
    if valid.shape != (lanes,):
        raise ValueError(f"legal_observation.valid must have shape ({lanes},)")

    self_f32 = base.self_f32
    target_f32 = base.target_f32
    visible = base.combat_f32[:, _COMBAT_TARGET_VISIBLE] > VISIBLE_THRESHOLD
    self_health = self_f32[:, _SELF_HEALTH]
    zeros = jnp.zeros((lanes,), dtype=jnp.float32)
    no_lanes = jnp.zeros((lanes,), dtype=jnp.bool_)

    rel_forward = NEARBY_ENTITY_RADIUS_BLOCKS * target_f32[:, _TARGET_RELATIVE_FORWARD]
    rel_right = NEARBY_ENTITY_RADIUS_BLOCKS * target_f32[:, _TARGET_RELATIVE_RIGHT]
    planar_blocks = jnp.hypot(rel_forward, rel_right)
    direction_defined = planar_blocks > PLANAR_EPSILON_BLOCKS
    safe_planar = jnp.where(direction_defined, planar_blocks, 1.0)

    if gait is None:
        self_speed, self_speed_known = zeros, no_lanes
    else:
        self_speed = (
            jnp.hypot(
                self_f32[:, _SELF_FORWARD_VELOCITY],
                self_f32[:, _SELF_RIGHT_VELOCITY],
            )
            * gait
            / HYTALE_TICKS_PER_SECOND
        )
        self_speed_known = valid

    if chase is None:
        closing, closing_known = zeros, no_lanes
    else:
        vel_forward = chase * target_f32[:, _TARGET_RELATIVE_VELOCITY_FORWARD]
        vel_right = chase * target_f32[:, _TARGET_RELATIVE_VELOCITY_RIGHT]
        radial_rate = (rel_forward * vel_forward + rel_right * vel_right) / safe_planar
        closing = -radial_rate / HYTALE_TICKS_PER_SECOND
        # The encoder clips target_f32 to [-1, 1]. A clipped offset (24 blocks or
        # more on an axis) or a clipped relative velocity (faster than the chase
        # speed on an axis) is not the true value, and the closing rate is not a
        # bound on it. Closing is therefore unknown when any of the four inputs is clipped.
        unclipped = (
            (jnp.abs(target_f32[:, _TARGET_RELATIVE_FORWARD]) < 1.0)
            & (jnp.abs(target_f32[:, _TARGET_RELATIVE_RIGHT]) < 1.0)
            & (jnp.abs(target_f32[:, _TARGET_RELATIVE_VELOCITY_FORWARD]) < 1.0)
            & (jnp.abs(target_f32[:, _TARGET_RELATIVE_VELOCITY_RIGHT]) < 1.0)
        )
        closing_known = valid & visible & direction_defined & unclipped

    seen_known = valid & visible
    values = {
        "target_visible": visible.astype(jnp.float32),
        "target_distance_blocks": NEARBY_ENTITY_RADIUS_BLOCKS
        * target_f32[:, _TARGET_DISTANCE],
        "target_bearing_sin": target_f32[:, _TARGET_BEARING_SIN],
        "target_bearing_cos": target_f32[:, _TARGET_BEARING_COS],
        "self_health_fraction": self_health,
        "target_health_fraction": target_f32[:, _TARGET_HEALTH],
        "self_speed_blocks_per_tick": self_speed,
        "closing_speed_blocks_per_tick": closing,
    }
    known = {
        "target_visible": valid,
        "target_distance_blocks": seen_known,
        "target_bearing_sin": seen_known,
        "target_bearing_cos": seen_known,
        "self_health_fraction": valid,
        "target_health_fraction": seen_known,
        "self_speed_blocks_per_tick": self_speed_known,
        "closing_speed_blocks_per_tick": closing_known,
    }
    return ObservationReadings(
        valid=valid,
        visible=visible,
        self_health=self_health,
        values={name: jnp.where(known[name], values[name], zeros) for name in _INSTANTANEOUS_NAMES},
        known=known,
    )


__all__ = [
    "PLANAR_EPSILON_BLOCKS",
    "VISIBLE_THRESHOLD",
    "ObservationReadings",
    "observation_readings",
    "positive_scale",
]
