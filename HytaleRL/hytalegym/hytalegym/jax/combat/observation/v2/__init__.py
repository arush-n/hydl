"""Version-2 learner observation with episode-pinned melee equipment."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.loadout import (
    MeleeLoadoutBatch,
    MeleeLoadoutObservation,
    encode_melee_loadout,
)
from hytalegym.jax.combat.observation.v1.runtime.actions import decode_learner_actions
from hytalegym.jax.combat.observation.v1.runtime.encoder import encode_learner_observation
from hytalegym.jax.combat.observation.v1.schema.types import (
    CombatSceneFeatures,
    InjectedWorldFeatures,
    LearnerActionDecode,
    LearnerCombatObservation,
)
from hytalegym.jax.combat.skills import (
    SKILL_APPROACH_ATTACK,
    SKILL_ATTACK,
    SKILL_IDLE,
    SKILL_RETREAT_ATTACK,
)
from hytalegym.jax.combat.types import CombatParams, CombatState


LEARNER_OBSERVATION_V2_SCHEMA = "hytalerl_combat_observation_v2"
LEARNER_OBSERVATION_V2_VERSION = 2
V2_FAILURE_LOADOUT_OVERFLOW = 1 << 6


class LearnerCombatObservationV2(NamedTuple):
    """Frozen v1 observation plus explicit equipped-weapon profiles."""

    schema_version: jax.Array
    base: LearnerCombatObservation
    loadout: MeleeLoadoutObservation
    action_mask: jax.Array
    valid: jax.Array
    failure_bits: jax.Array


def encode_learner_observation_v2(
    state: CombatState,
    params: CombatParams,
    combat_observation: jax.Array,
    scene: CombatSceneFeatures,
    world: InjectedWorldFeatures,
    loadout: MeleeLoadoutBatch,
) -> LearnerCombatObservationV2:
    """Encode the v1 combat/world contract and the episode loadout."""

    base = encode_learner_observation(
        state,
        params,
        combat_observation,
        scene,
        world,
    )
    return extend_learner_observation_v2(base, params, loadout)


def extend_learner_observation_v2(
    base: LearnerCombatObservation,
    params: CombatParams,
    loadout: MeleeLoadoutBatch,
) -> LearnerCombatObservationV2:
    """Attach normalized loadout features without mutating v1."""

    loadout_observation = encode_melee_loadout(loadout, params)
    valid = base.valid & loadout_observation.valid
    has_melee_attack = loadout_observation.equipped_mask & jnp.any(
        loadout_observation.attack_mask, axis=1
    )
    action_mask = base.action_mask
    for action_id in (
        SKILL_ATTACK,
        SKILL_APPROACH_ATTACK,
        SKILL_RETREAT_ATTACK,
    ):
        action_mask = action_mask.at[:, action_id].set(
            action_mask[:, action_id] & has_melee_attack
        )
    safe_action_mask = jnp.zeros_like(action_mask)
    safe_action_mask = safe_action_mask.at[:, SKILL_IDLE].set(True)
    action_mask = jnp.where(
        valid[:, None],
        action_mask,
        safe_action_mask,
    )
    loadout_failure = jnp.where(
        loadout_observation.valid,
        jnp.uint32(0),
        jnp.uint32(V2_FAILURE_LOADOUT_OVERFLOW),
    )
    failure_bits = jnp.bitwise_or(base.overflow_bits, loadout_failure)
    return LearnerCombatObservationV2(
        schema_version=jnp.full(
            base.valid.shape,
            LEARNER_OBSERVATION_V2_VERSION,
            dtype=jnp.int32,
        ),
        base=base,
        loadout=loadout_observation,
        action_mask=action_mask,
        valid=valid,
        failure_bits=failure_bits,
    )


def decode_learner_actions_v2(
    observation: LearnerCombatObservationV2,
    action_ids: jax.Array,
    *,
    maximum_turn_degrees: float = 45.0,
) -> LearnerActionDecode:
    """Decode against the v2 legality mask while retaining v1 semantics."""

    base = observation.base._replace(
        action_mask=observation.action_mask,
        valid=observation.valid,
    )
    return decode_learner_actions(
        base,
        action_ids,
        maximum_turn_degrees=maximum_turn_degrees,
    )
