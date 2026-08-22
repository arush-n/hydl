"""Omniscient opponent observation shared by every contract in this package.

These lessons train *combat*, not search. A policy that loses the opponent
behind terrain and then spends the episode re-acquiring is learning a different
problem, and its reward becomes dominated by visibility luck. So every contract
writes exact relative pose and velocity into the learner's target slot and
marks the target available, whatever occlusion the engine reports.

This is a training-time supplement only. It patches the observation the learner
consumes; it does not change the published live-policy contract, so a policy
trained here still runs against the ordinary Java observation. The cost is that
any occlusion-handling behaviour has to be taught by a later stage.
"""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.observation.v1.schema.contract import (
    NEARBY_ENTITY_RADIUS_BLOCKS,
    SELF_FLOAT_FEATURES,
    TARGET_FLOAT_FEATURES,
)


OMNISCIENT_CONTEXT_SCHEMA = "arena-contract-omniscient-context-v1"

#: Target features overwritten with exact geometry, in observation order.
CONTEXT_FEATURES = (
    "relative_forward",
    "relative_right",
    "relative_up",
    "relative_velocity_forward",
    "relative_velocity_right",
    "relative_velocity_up",
    "planar_distance",
    "distance",
    "bearing_sin",
    "bearing_cos",
)
_CONTEXT_INDICES = tuple(
    TARGET_FLOAT_FEATURES.index(name) for name in CONTEXT_FEATURES
)
_VISIBLE_INDEX = TARGET_FLOAT_FEATURES.index("visible")
_TARGET_START = len(SELF_FLOAT_FEATURES)
_TARGET_END = _TARGET_START + len(TARGET_FLOAT_FEATURES)


def omniscient_context_manifest() -> dict[str, object]:
    """Describe the supplement so a run records what the learner could see."""

    return {
        "schema": OMNISCIENT_CONTEXT_SCHEMA,
        "supplemented_features": CONTEXT_FEATURES,
        "visibility": "target_is_always_available_through_occlusion",
        "scope": "training_supplement_only",
        "java_transfer": "not_required_by_the_live_policy_contract",
    }


def make_omniscient_observation_transform(roles, params, *, speed_scale=None):
    """Patch exact relative target geometry into the learner's observation.

    ``speed_scale`` normalises relative velocity. It defaults to the target
    chase speed on ``params`` so the encoding matches the pursuit lesson; pass
    an explicit value for contracts whose opponent moves on a different scale.
    """

    lanes = jnp.arange(roles.learner_policy_slot.shape[0], dtype=jnp.int32)
    context_indices = jnp.asarray(_CONTEXT_INDICES, dtype=jnp.int32)

    def transform(arena, observation):
        combat = arena.arsenal.combat
        learner = roles.learner_actor_index
        target = roles.target_actor_index
        position = combat.position[lanes, learner]
        target_position = combat.position[lanes, target]
        relative_velocity = (
            combat.velocity[lanes, target] - combat.velocity[lanes, learner]
        )
        yaw = jnp.deg2rad(combat.yaw[lanes, learner])
        offset = target_position - position
        forward_x, forward_z = -jnp.sin(yaw), -jnp.cos(yaw)
        right_x, right_z = jnp.cos(yaw), -jnp.sin(yaw)
        forward = offset[:, 0] * forward_x + offset[:, 2] * forward_z
        right = offset[:, 0] * right_x + offset[:, 2] * right_z
        velocity_forward = (
            relative_velocity[:, 0] * forward_x + relative_velocity[:, 2] * forward_z
        )
        velocity_right = (
            relative_velocity[:, 0] * right_x + relative_velocity[:, 2] * right_z
        )
        planar = jnp.hypot(forward, right)
        safe_planar = jnp.maximum(planar, jnp.float32(1.0e-6))
        radius = jnp.float32(NEARBY_ENTITY_RADIUS_BLOCKS)
        reference = (
            params.target_chase_speed if speed_scale is None else speed_scale
        )
        speed = jnp.maximum(
            jnp.asarray(reference, dtype=jnp.float32), jnp.float32(1.0e-6)
        )
        vertical = jnp.maximum(
            jnp.asarray(params.vertical_speed_scale, dtype=jnp.float32),
            jnp.float32(1.0e-6),
        )
        context = jnp.stack(
            (
                forward / radius,
                right / radius,
                offset[:, 1] / radius,
                velocity_forward / speed,
                velocity_right / speed,
                relative_velocity[:, 1] / vertical,
                planar / radius,
                jnp.linalg.norm(offset, axis=1) / radius,
                right / safe_planar,
                forward / safe_planar,
            ),
            axis=1,
        )
        context = jnp.clip(context, -1.0, 1.0).astype(jnp.float32)

        slot = roles.learner_policy_slot
        target_f32 = observation.structured.base.target_f32
        patched = target_f32[lanes, slot].at[:, context_indices].set(context)
        patched = patched.at[:, _VISIBLE_INDEX].set(jnp.float32(1.0))
        target_f32 = target_f32.at[lanes, slot].set(patched)
        target_mask = observation.structured.base.target_mask.at[lanes, slot].set(True)
        structured = observation.structured._replace(
            base=observation.structured.base._replace(
                target_f32=target_f32,
                target_mask=target_mask,
            )
        )
        dense = observation.dense.at[lanes, slot, _TARGET_START:_TARGET_END].set(
            patched
        )
        dense = dense.at[lanes, slot, _TARGET_END].set(jnp.float32(1.0))
        return observation._replace(structured=structured, dense=dense)

    return transform


__all__ = [
    "CONTEXT_FEATURES",
    "OMNISCIENT_CONTEXT_SCHEMA",
    "make_omniscient_observation_transform",
    "omniscient_context_manifest",
]
