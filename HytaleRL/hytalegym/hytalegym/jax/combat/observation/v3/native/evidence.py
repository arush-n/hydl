"""Project versioned native server facts through the shared learner-v3 encoder.

Java publishes copied server facts and independent availability.  This module
validates that wire, materializes a target-gated projection state, binds
World's public exact-geometry producers, and then calls the same actor-evidence
projector and encoder as the JAX environment.  It contains no item-specific
branches and never treats unavailable evidence as an all-clear.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal.schema.contract import (
    ABILITY_CAPACITY,
    INTERACTION_QUEUE_DELAY_TICKS,
)
from hytalegym.jax.combat.arsenal.factory import empty_arsenal_commands
from hytalegym.jax.combat.arsenal.programs.outer_roots import (
    resolve_observed_execution_slot,
)
from hytalegym.jax.combat.arsenal.profiles import (
    hytale_0_5_7_loadouts,
    hytale_0_5_7_native_profile_bindings,
    native_ability_requested_charge_time_seconds,
)
from hytalegym.jax.combat.arsenal.runtime import (
    arsenal_runtime_config,
    reset_arsenal_batch,
)
from hytalegym.jax.combat.mechanics import (
    MECHANICS_FAILURE_INVALID_STATE,
    STATUS_CAPACITY,
)
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.observation.v1.runtime.factory import (
    empty_injected_world_features,
)
from hytalegym.jax.combat.observation.v3.encoding.encoder import (
    encode_learner_observation_v3_from_actor_evidence,
    project_learner_observation_v3_actor_evidence,
)
from hytalegym.jax.combat.observation.v3.tokens.inventory import (
    inventory_policy_tokens_from_native_frame,
)
from hytalegym.jax.combat.observation.v3.tokens.light import (
    ActorLightPolicyTokens,
    empty_actor_light_policy_tokens,
)
from hytalegym.jax.combat.observation.v3.native.channels.light import (
    NativePerceptionCapture,
    capture_native_actor_light_policy_tokens,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    arsenal_policy_action_mask,
    arsenal_policy_observation,
    decode_arsenal_policy_actions,
)
from hytalegym.jax.combat.observation.v3.schema.spec import (
    learner_observation_v3_actor_evidence_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerCombatObservationV3,
    LearnerObservationV3ActorEvidence,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    WorldGeometryPolicyConfig,
    empty_world_geometry_policy_tokens,
    encode_world_geometry_policy_tokens,
    normalize_world_geometry_policy_config,
)
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    ENTITY_COUNT,
    TARGET_ENTITY,
    CombatParams,
    default_combat_params,
)
from hytalegym.rulesets import (
    NATIVE_STATUS_PROJECTION_OMIT_DERIVED_MECHANIC,
    NATIVE_STATUS_PROJECTION_OMIT_VISUAL_ONLY,
    load_native_status_projections,
)
from .codec.decode import (  # noqa: F401  (re-exported: callers unchanged)
    NATIVE_ACTOR_EVIDENCE_GEOMETRY_CELL_COUNT,
    NATIVE_ACTOR_EVIDENCE_SCHEMA,
    NATIVE_ACTOR_EVIDENCE_VERSION,
    _StatusProgram,
    _ability,
    _absent_actor_has_evidence,
    _active_ability_slot,
    _actor,
    _array,
    _boolean,
    _combat_info,
    _exact_geometry,
    _integer,
    _loadout_mismatch,
    _mapping,
    _movement_row,
    _movement_state,
    _native_use_action_available,
    _number,
    _required_resources_available,
    _resources,
    _sequence,
    _status,
    _status_programs,
    _statuses,
    _string,
    _wire_frame,
)


NATIVE_ACTOR_EVIDENCE_STATUS_CAPACITY = 8
NATIVE_ACTOR_EVIDENCE_ABILITY_CAPACITY = 16
# Native-only status rows that cannot be reconstructed as a complete compiled
# StatusState are invalid mechanics state. Derive the public diagnostic alias
# from the mechanics contract so the fail-closed bit cannot drift.
NATIVE_ACTOR_EVIDENCE_UNKNOWN_STATUS_FAILURE = (
    MECHANICS_FAILURE_INVALID_STATE
)
_UNKNOWN_STATUS_FAILURE = np.uint32(
    NATIVE_ACTOR_EVIDENCE_UNKNOWN_STATUS_FAILURE
)


@dataclass(frozen=True)
class NativeActorEvidenceAssembly:
    """One native policy row plus the state needed to decode its next action."""

    evidence: LearnerObservationV3ActorEvidence
    observation: LearnerCombatObservationV3
    policy_observation: np.ndarray
    policy_action_mask: np.ndarray
    world_capabilities: Any
    light_tokens: ActorLightPolicyTokens
    actor_yaw_degrees: float
    raw_wire: Mapping[str, Any]
    action_surface_snapshot: Any | None = None


class NativeActorEvidenceAssembler:
    """Stateful one-lane adapter for one reset-pinned catalog matchup."""

    def __init__(
        self,
        agent_profile: str,
        *,
        target_profile: str = "",
        params: CombatParams | None = None,
        ticks_per_step: int = 4,
        world_geometry_config: WorldGeometryPolicyConfig | None = None,
        native_perception_capture: NativePerceptionCapture | None = None,
    ) -> None:
        if not isinstance(agent_profile, str) or not agent_profile:
            raise ValueError("agent_profile must name a catalog profile")
        if not isinstance(target_profile, str):
            raise TypeError("target_profile must be a string")
        if (
            isinstance(ticks_per_step, bool)
            or not isinstance(ticks_per_step, int)
            or ticks_per_step <= 0
        ):
            raise ValueError("ticks_per_step must be a positive integer")
        self.params = default_combat_params() if params is None else params
        self.agent_profile = agent_profile
        self.ticks_per_step = ticks_per_step
        self.world_geometry_config = normalize_world_geometry_policy_config(
            world_geometry_config
        )
        if native_perception_capture is not None and not callable(
            native_perception_capture
        ):
            raise TypeError("native_perception_capture must be callable or None")
        self.native_perception_capture = native_perception_capture
        loadout = hytale_0_5_7_loadouts(
            agent_profile,
            target_profiles=target_profile,
        )
        self.config = arsenal_runtime_config(
            loadout,
            specialize=False,
            sensor_range=self.params.sensor_range,
        )
        keys = jax.random.split(jax.random.key(570057), 1)
        self._template, _ = reset_arsenal_batch(
            keys,
            self.params,
            self.config,
        )
        self._status_programs = _status_programs(self.config.loadout)
        self._omitted_native_statuses = frozenset(
            effect_id
            for effect_id, projection in load_native_status_projections().items()
            if projection.projection
            in {
                NATIVE_STATUS_PROJECTION_OMIT_VISUAL_ONLY,
                NATIVE_STATUS_PROJECTION_OMIT_DERIVED_MECHANIC,
            }
        )
        self.native_profile = hytale_0_5_7_native_profile_bindings(
            agent_profile
        )
        native_to_authored = self.native_profile.authored_ability_slots
        if len(native_to_authored) != len(self.native_profile.abilities):
            raise ValueError(
                "native profile bindings and authored slots must have equal length"
            )
        if (
            len(set(native_to_authored)) != len(native_to_authored)
            or any(
                slot < 0 or slot >= ABILITY_CAPACITY
                for slot in native_to_authored
            )
        ):
            raise ValueError(
                "native profile authored ability slots must be unique and in range"
            )
        self._native_to_authored_ability_slot = tuple(native_to_authored)
        authored_to_native = np.full(
            (ABILITY_CAPACITY,),
            -1,
            dtype=np.int32,
        )
        authored_mapping = self.native_profile.authored_to_native_ability_slots
        if len(authored_mapping) > ABILITY_CAPACITY or any(
            native_slot < -1 or native_slot >= len(self.native_profile.abilities)
            for native_slot in authored_mapping
        ):
            raise ValueError(
                "authored-to-native ability slots must fit both fixed capacities"
            )
        authored_to_native[: len(authored_mapping)] = authored_mapping
        represented = {
            int(native_slot)
            for native_slot in authored_mapping
            if native_slot >= 0
        }
        if represented != set(range(len(self.native_profile.abilities))):
            raise ValueError("every native ability root must have an authored slot")
        for native_slot, authored_slot in enumerate(native_to_authored):
            if authored_to_native[authored_slot] != native_slot:
                raise ValueError(
                    "native representative slot disagrees with authored mapping"
                )
        self._authored_to_native_ability_slot = authored_to_native
        self.reset()

    @property
    def negotiation_options(self) -> dict[str, object]:
        return {
            "learner_v3_actor_evidence_contract_sha256": (
                learner_observation_v3_actor_evidence_contract_sha256()
            ),
            "learner_v3_actor_evidence_entity_count": ENTITY_COUNT,
            "learner_v3_actor_evidence_status_capacity": STATUS_CAPACITY,
            "learner_v3_actor_evidence_ability_capacity": ABILITY_CAPACITY,
            "learner_v3_actor_evidence_resource_stat_ids": list(
                self.native_profile.resource_stat_ids
            ),
        }

    @property
    def reset_options(self) -> dict[str, object]:
        """Return exact evidence negotiation and asset-derived slot bindings."""

        options = dict(self.negotiation_options)
        if self.native_profile.item_id:
            options["native_combat_item_id"] = self.native_profile.item_id
        if self.native_profile.abilities:
            options.update(
                {
                    "native_combat_ability_interaction_ids": [
                        binding.interaction_id
                        for binding in self.native_profile.abilities
                    ],
                    "native_combat_ability_interaction_types": [
                        binding.interaction_type
                        for binding in self.native_profile.abilities
                    ],
                }
            )
        if self.native_profile.guard is not None:
            options.update(
                {
                    "native_guard_interaction_id": (
                        self.native_profile.guard.interaction_id
                    ),
                    "native_guard_interaction_type": (
                        self.native_profile.guard.interaction_type
                    ),
                }
            )
        from hytalegym.jax.combat.arsenal.training_inventory import (
            native_training_hotbar_items_from_config,
        )

        hotbar_items = native_training_hotbar_items_from_config(self.config)
        if len(hotbar_items) > 1:
            # External root prerequisites require the bounded Player-backed
            # inventory fixture. Keep ordinary NPC-only combat profiles on
            # their existing actor path.
            from hytalegym.worldgen import native_world_verb_reset_options

            options.update(
                native_world_verb_reset_options(hotbar_items=hotbar_items)
            )
        return options

    def reset(self) -> None:
        self._first_world_tick: int | None = None
        self._last_world_tick: int | None = None
        self._episode_tick_count = 0
        self._last_health: float | None = None
        self._ticks_since_damage = 0
        self._agent_attack_sequence = 0
        self._target_attack_sequence = 0
        self._agent_attack_active = False
        self._target_attack_active = False
        self._tracked_ability_slot = -1
        self._tracked_ability_execution_slot = -1
        self._tracked_ability_elapsed_seconds = 0.0
        self._pending_authored_ability_slot = -1
        self._last_guard_held = False
        self._cooldowns = np.zeros(
            (ENTITY_COUNT, ABILITY_CAPACITY),
            dtype=np.float32,
        )
        self._charge_times = np.asarray(
            self.config.loadout.ability_charge_times_seconds[0],
            dtype=np.float32,
        )
        self._charge_capacity = np.asarray(
            self.config.loadout.ability_charge_capacity[0],
            dtype=np.int32,
        )
        self._charge_counts = self._charge_capacity.copy()
        self._charge_timers = np.zeros(
            (ENTITY_COUNT, ABILITY_CAPACITY),
            dtype=np.float32,
        )
        self._interrupt_recharge = np.asarray(
            self.config.loadout.ability_interrupt_recharge[0],
            dtype=np.bool_,
        )

    def assemble(
        self,
        legacy_observation: Mapping[str, Any],
        wire_value: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> NativeActorEvidenceAssembly:
        """Decode a bridge frame and return the complete dense policy row."""

        wire = _wire_frame(wire_value)
        if not wire["available"]:
            reason = wire["unavailable_reason"] or "unavailable"
            raise ValueError(f"native actor evidence is unavailable: {reason}")
        state, resource_available, bound_abilities = self._projection_state(
            wire,
            info,
        )
        geometry = _exact_geometry(legacy_observation, wire)
        world, world_tokens, raw_world_tokens = self._world_evidence(
            state,
            geometry,
            wire,
        )
        light_tokens = empty_actor_light_policy_tokens(
            1,
            self.world_geometry_config.token_capacity,
        )
        if raw_world_tokens is not None and self.native_perception_capture is not None:
            light_tokens = capture_native_actor_light_policy_tokens(
                raw_world_tokens,
                state.combat.position[:, AGENT_ENTITY : AGENT_ENTITY + 1, :],
                self.native_perception_capture,
            )
        projected = project_learner_observation_v3_actor_evidence(
            state,
            self.params,
            empty_injected_world_features(1),
            world,
            self.config,
            world_geometry=world_tokens,
        )

        resource_mask = projected.resource_mask & jnp.asarray(
            resource_available
        )
        native_movement_values, native_movement_mask = _movement_row(
            wire["actors"][AGENT_ENTITY]
        )
        # The native wire transports all MovementStates fields.  The learner
        # row may expose only fields the accelerated Walk producer can also
        # populate; otherwise native availability becomes an unearned
        # observed-false value during JAX training.  Derive that boundary from
        # the shared projection instead of copying a second field-name list.
        movement_state_mask = (
            jnp.asarray(native_movement_mask)
            & projected.movement_state_mask
        )
        evidence = projected._replace(
            resource_f32=jnp.where(
                resource_mask,
                projected.resource_f32,
                jnp.float32(0.0),
            ),
            resource_mask=resource_mask,
            ability_legal=(
                projected.ability_legal
                & jnp.asarray(bound_abilities)
                & jnp.asarray(
                    _required_resources_available(
                        self.config.loadout,
                        resource_available,
                    )
                )
            ),
            movement_state_f32=jnp.where(
                movement_state_mask,
                jnp.asarray(native_movement_values),
                jnp.float32(0.0),
            ),
            movement_state_mask=movement_state_mask,
            skill_action_mask=(
                projected.skill_action_mask
                & jnp.asarray(wire["skill_action_mask"])[None, :]
            ),
            jump_action_mask=(
                projected.jump_action_mask
                & jnp.asarray([wire["jump_action_available"]])
            ),
            guard_action_mask=(
                projected.guard_action_mask
                & jnp.asarray([wire["guard_action_available"]])
            ),
            dodge_action_mask=(
                projected.dodge_action_mask
                & jnp.asarray(wire["dodge_action_mask"])[None, :]
            ),
            door_action_mask=(
                projected.door_action_mask
                & jnp.asarray(wire["door_action_mask"])[None, ...]
            ),
            loadout_failure=(
                projected.loadout_failure
                | jnp.asarray([_loadout_mismatch(wire, self.config.loadout)])
                | jnp.asarray([bool(wire["loadout_failure"])])
            ),
            mechanics_failure_bits=(
                projected.mechanics_failure_bits
                | jnp.asarray(
                    [wire["mechanics_failure_bits"]],
                    dtype=jnp.uint32,
                )
            ),
            arsenal_failure_bits=(
                projected.arsenal_failure_bits
                | jnp.asarray(
                    [wire["arsenal_failure_bits"]],
                    dtype=jnp.uint32,
                )
            ),
        )
        observation = encode_learner_observation_v3_from_actor_evidence(evidence)
        dense = np.asarray(
            arsenal_policy_observation(
                observation,
                inventory_tokens=inventory_policy_tokens_from_native_frame(
                    info.get("native_inventory")
                ),
                light_tokens=light_tokens,
            ),
            dtype=np.float32,
        )[0]
        action_mask = np.asarray(
            arsenal_policy_action_mask(
                observation,
                use_available=jnp.asarray(
                    [_native_use_action_available(info)],
                    dtype=jnp.bool_,
                ),
            ),
            dtype=np.bool_,
        )[0]
        if not np.all(np.isfinite(dense)):
            raise ValueError("native learner-v3 observation contains non-finite data")
        return NativeActorEvidenceAssembly(
            evidence=evidence,
            observation=observation,
            policy_observation=dense,
            policy_action_mask=action_mask,
            world_capabilities=world,
            light_tokens=light_tokens,
            actor_yaw_degrees=float(
                wire["actors"][AGENT_ENTITY]["yaw_degrees"]
            ),
            raw_wire=wire,
        )

    def bind_policy_action_surface(
        self,
        assembly: NativeActorEvidenceAssembly,
        surface,
        info: Mapping[str, Any],
    ) -> NativeActorEvidenceAssembly:
        """Attach one actor-safe surface and its host-only exact snapshot.

        The surface provider runs after native evidence assembly.  This method
        proves that its visible masks and privileged bindings describe the
        same immediately preceding row, then rebuilds the dense observation
        and all twelve action masks through the ordinary policy functions.
        """

        from hytalegym.jax.combat.observation.v3.native.channels.world_actions import (
            NativePolicyActionSurface,
        )

        if not isinstance(assembly, NativeActorEvidenceAssembly):
            raise TypeError("assembly must be NativeActorEvidenceAssembly")
        if not isinstance(surface, NativePolicyActionSurface):
            raise TypeError("surface must be NativePolicyActionSurface")
        bridge = info.get("bridge_sha256")
        epoch = info.get("native_world_verb_epoch")
        if bridge != surface.snapshot.bridge_sha256:
            raise ValueError("native action surface came from another bridge")
        if epoch != surface.snapshot.world_epoch:
            raise ValueError("native action surface came from another World epoch")

        block_mask = np.asarray(surface.block_candidates.candidate_mask)
        recipe_mask = np.asarray(surface.recipe_candidates.candidate_mask)
        recipe_encoding_mask = np.asarray(surface.recipe_encoding.candidate_mask)
        if block_mask.shape != (1, len(surface.snapshot.block_candidate_mask)) or (
            tuple(bool(value) for value in block_mask[0])
            != surface.snapshot.block_candidate_mask
        ):
            raise ValueError(
                "native block policy mask and privileged snapshot differ"
            )
        if recipe_mask.shape != (1, len(surface.snapshot.recipe_candidate_mask)) or (
            tuple(bool(value) for value in recipe_mask[0])
            != surface.snapshot.recipe_candidate_mask
        ):
            raise ValueError(
                "native recipe policy mask and privileged snapshot differ"
            )
        if not np.array_equal(recipe_encoding_mask, recipe_mask):
            raise ValueError("native recipe encoding mask differs from policy mask")

        supported = {
            value.strip()
            for value in str(info.get("supported_actions", "")).split(",")
            if value.strip()
        }
        requested_support = set()
        if bool(np.asarray(surface.use_available)[0]):
            requested_support.add("use")
        triggers = np.asarray(surface.block_trigger_available)[0]
        if bool(triggers[0]):
            requested_support.add("break_block")
        if bool(triggers[1]):
            requested_support.add("place_block")
        if bool(np.any(recipe_mask)):
            requested_support.add("craft_recipe")
        missing = sorted(requested_support - supported)
        if missing:
            raise ValueError(
                "native action surface advertises unsupported verbs: "
                + ",".join(missing)
            )

        inventory_tokens = inventory_policy_tokens_from_native_frame(
            info.get("native_inventory")
        )
        dense = np.asarray(
            arsenal_policy_observation(
                assembly.observation,
                surface.block_candidates,
                surface.recipe_encoding,
                inventory_tokens=inventory_tokens,
                light_tokens=assembly.light_tokens,
            ),
            dtype=np.float32,
        )[0]
        action_mask = np.asarray(
            arsenal_policy_action_mask(
                assembly.observation,
                surface.block_candidates,
                surface.recipe_candidates,
                use_available=jnp.asarray(surface.use_available),
                block_trigger_available=jnp.asarray(
                    surface.block_trigger_available
                ),
            ),
            dtype=np.bool_,
        )[0]
        if not np.all(np.isfinite(dense)):
            raise ValueError("native action-surface observation is non-finite")
        return replace(
            assembly,
            policy_observation=dense,
            policy_action_mask=action_mask,
            action_surface_snapshot=surface.snapshot,
        )

    def decode_policy_action(
        self,
        factors: Any,
        assembly: NativeActorEvidenceAssembly,
    ):
        values = np.asarray(factors)
        expected = (len(ARSENAL_POLICY_ACTION_HEAD_SIZES),)
        if values.shape != expected:
            raise ValueError(f"native learner-v3 action must have shape {expected}")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("native learner-v3 action factors must be integers")
        factors_i32 = values.astype(np.int32, copy=False)
        for index, size in enumerate(ARSENAL_POLICY_ACTION_HEAD_SIZES):
            if not 0 <= int(factors_i32[index]) < size:
                raise ValueError(
                    f"native learner-v3 action factor {index} is out of range"
                )
        action = decode_arsenal_policy_actions(
            jnp.asarray(factors_i32[None, :])
        )
        self._last_guard_held = bool(np.asarray(action.guard_held)[0])
        from hytalegym.jax.combat.observation.v3.policy.actions import (
            decode_learner_arsenal_action,
        )

        decoded = decode_learner_arsenal_action(
            assembly.observation,
            action,
            assembly.world_capabilities,
            desired_body_yaw_degrees=jnp.asarray(
                [assembly.actor_yaw_degrees],
                dtype=jnp.float32,
            ),
        )
        return action, decoded

    def requested_charge_time_seconds(self, action) -> float:
        """Use the selected authored program clock; never branch on a weapon."""

        slot = int(np.asarray(action.ability_slot)[0])
        return native_ability_requested_charge_time_seconds(
            self.config.loadout,
            slot,
            source_entity_index=AGENT_ENTITY,
        )

    def native_ability_slot(self, action) -> int:
        """Map one authored policy slot onto the compact reset-bound list."""

        slot = int(np.asarray(action.ability_slot)[0])
        self._pending_authored_ability_slot = slot
        if slot < 0:
            return -1
        if slot >= ABILITY_CAPACITY:
            raise ValueError("authored ability slot is out of range")
        return int(self._authored_to_native_ability_slot[slot])

    def _projection_state(self, wire, info):
        actors = wire["actors"]
        abilities = wire["abilities"]
        world_tick = int(wire["world_tick"])
        resources, resource_available = _resources(
            actors,
            self.config.mechanics_rules,
        )
        self._record_accepted_ability(
            info,
            resources,
            resource_available,
        )
        delta_seconds, _ = self._advance_clocks(
            world_tick,
            actors[AGENT_ENTITY],
            info,
        )
        present = np.asarray(
            [actor["present"] and actor["perceptible"] for actor in actors],
            dtype=np.bool_,
        )

        def rows(name: str, width: int | None = None, dtype=np.float32):
            values = [
                actor[name] if present[index] else (
                    np.zeros(width, dtype=dtype) if width else 0
                )
                for index, actor in enumerate(actors)
            ]
            return np.asarray(values, dtype=dtype)[None, ...]

        position = rows("position", 3)
        velocity = rows("velocity", 3)
        expected_health_maximum = np.asarray(
            [
                _scalar(self.params.agent_max_health),
                _scalar(self.params.target_max_health),
            ],
            dtype=np.float32,
        )
        health = np.asarray(
            [
                [
                    (
                        np.clip(
                            actors[entity]["health"]
                            / actors[entity]["max_health"],
                            0.0,
                            1.0,
                        )
                        * expected_health_maximum[entity]
                    )
                    if present[entity]
                    else 0.0
                    for entity in range(ENTITY_COUNT)
                ]
            ],
            dtype=np.float32,
        )
        yaw = rows("yaw_degrees")
        raw = _combat_info(info)
        agent_active = bool(raw["agent_attack_executing"])
        target_active = bool(
            present[TARGET_ENTITY] and raw["target_attack_phase"] != 0
        )
        if agent_active and not self._agent_attack_active:
            self._agent_attack_sequence += 1
        if target_active and not self._target_attack_active:
            self._target_attack_sequence += 1
        self._agent_attack_active = agent_active
        self._target_attack_active = target_active

        native_active_slot = _active_ability_slot(abilities)
        if native_active_slot >= len(
            self._native_to_authored_ability_slot
        ):
            raise ValueError("active native ability slot is not reset-bound")
        active_slot = -1
        if native_active_slot >= 0:
            tracked = self._tracked_ability_slot
            if (
                tracked >= 0
                and self._authored_to_native_ability_slot[tracked]
                == native_active_slot
            ):
                active_slot = tracked
            else:
                active_slot = self._native_to_authored_ability_slot[
                    native_active_slot
                ]
        execution_slot = active_slot
        active_elapsed = np.zeros((1, ENTITY_COUNT), dtype=np.float32)
        if active_slot >= 0:
            controlled_timing = (
                "engine_ticks_executed" in info
                or "engine_mean_delta_seconds" in info
            )
            if controlled_timing and self._tracked_ability_slot == active_slot:
                active_elapsed[0, AGENT_ENTITY] = np.float32(
                    self._tracked_ability_elapsed_seconds
                )
            elif controlled_timing:
                raise ValueError(
                    "active native ability lacks controlled lifecycle start evidence"
                )
            else:
                start = abilities[native_active_slot]["start_world_tick"]
                if start >= 0:
                    elapsed = max(0, world_tick - start) / _scalar(
                        self.params.ticks_per_second
                    )
                    active_elapsed[0, AGENT_ENTITY] = elapsed
                    self._tracked_ability_slot = active_slot
                    self._tracked_ability_elapsed_seconds = elapsed
            if self._tracked_ability_slot != active_slot:
                self._tracked_ability_slot = active_slot
                self._tracked_ability_execution_slot = (
                    self._execution_slot_for_root(
                        active_slot,
                        resources,
                        resource_available,
                    )
                )
            elif self._tracked_ability_execution_slot < 0:
                self._tracked_ability_execution_slot = (
                    self._execution_slot_for_root(
                        active_slot,
                        resources,
                        resource_available,
                    )
                )
            execution_slot = self._tracked_ability_execution_slot
        else:
            self._tracked_ability_slot = -1
            self._tracked_ability_execution_slot = -1
            self._tracked_ability_elapsed_seconds = 0.0
        mechanics = self._template.mechanics._replace(
            resources=jnp.asarray(resources),
            stamina_regen_delay_seconds=jnp.asarray(
                [[actor["defense_values"][4] for actor in actors]],
                dtype=jnp.float32,
            ),
            stamina_broken=jnp.asarray(
                [[actor["defense_values"][1] > 0.5 for actor in actors]]
            ),
            guard_held=jnp.zeros(
                (1, ENTITY_COUNT),
                dtype=jnp.bool_,
            ).at[0, AGENT_ENTITY].set(self._last_guard_held),
            guard_active=jnp.asarray(
                [[actor["defense_values"][0] > 0.5 for actor in actors]]
            ),
            control_immunity=jnp.asarray(
                [[actor["defense_values"][5] for actor in actors]],
                dtype=jnp.float32,
            ),
            applied_velocity=jnp.asarray(
                [[
                    actor["motion_force"]["projected_velocity"]
                    for actor in actors
                ]],
                dtype=jnp.float32,
            ),
            dodge_invulnerability_remaining_seconds=jnp.asarray(
                [[actor["defense_values"][2] for actor in actors]],
                dtype=jnp.float32,
            ),
        )
        statuses, unknown_status = _statuses(
            actors,
            self._status_programs,
            self._omitted_native_statuses,
            mechanics.statuses,
        )
        mechanics_failure = np.uint32(wire["mechanics_failure_bits"])
        if unknown_status:
            mechanics_failure |= _UNKNOWN_STATUS_FAILURE
        mechanics = mechanics._replace(
            statuses=statuses,
            failure_bits=jnp.asarray([mechanics_failure], dtype=jnp.uint32),
        )
        episode_tick = self._episode_tick_count
        agent = actors[AGENT_ENTITY]
        force_speed = float(agent["defense_values"][3])
        pending_attack = active_slot if active_slot >= 0 else (
            0 if agent_active else -1
        )
        combat = self._template.combat._replace(
            position=jnp.asarray(position),
            velocity=jnp.asarray(velocity),
            health=jnp.asarray(health),
            yaw=jnp.asarray(yaw),
            target_head_yaw=jnp.asarray(
                [
                    raw["target_head_yaw_degrees"]
                    if present[TARGET_ENTITY]
                    else 0.0
                ],
                dtype=jnp.float32,
            ),
            target_head_pitch=jnp.asarray(
                [
                    raw["target_head_pitch_degrees"]
                    if present[TARGET_ENTITY]
                    else 0.0
                ],
                dtype=jnp.float32,
            ),
            # The bridge reports one orientation per actor and no separate
            # agent head, unlike `target_head_yaw_degrees` above. For a
            # bridge-driven actor that single pitch IS the look pitch -- the
            # body never pitches, because `body.setTranslation` only ever gets
            # a planar vector -- so it seeds the head, and the body stays
            # level. Head yaw takes the reported yaw for the same reason: both
            # steerings are commanded to it and only a slow separates them.
            # PROTOCOL GAP: a slowed native actor's head would lag its body and
            # the bridge cannot currently express that.
            pitch=jnp.zeros((1,), dtype=jnp.float32),
            agent_head_pitch=jnp.asarray(
                [agent["pitch_degrees"]],
                dtype=jnp.float32,
            ),
            agent_head_yaw=jnp.asarray(
                yaw[:, AGENT_ENTITY],
                dtype=jnp.float32,
            ),
            tick_count=jnp.asarray([episode_tick], dtype=jnp.int32),
            motion_timing_profile=jnp.zeros((1,), dtype=jnp.int32),
            last_motion_delta_seconds=jnp.asarray(
                [delta_seconds],
                dtype=jnp.float32,
            ),
            attack_sequence_index=jnp.asarray(
                [self._agent_attack_sequence],
                dtype=jnp.int32,
            ),
            agent_attack_cooldown_seconds=jnp.asarray(
                [1.0 if agent_active else 0.0],
                dtype=jnp.float32,
            ),
            pending_agent_attack_index=jnp.asarray(
                [pending_attack],
                dtype=jnp.int32,
            ),
            target_attack_sequence_index=jnp.asarray(
                [self._target_attack_sequence],
                dtype=jnp.int32,
            ),
            target_attack_index=jnp.asarray(
                [
                    raw["target_attack_index"]
                    if present[TARGET_ENTITY]
                    else -1
                ],
                dtype=jnp.int32,
            ),
            target_attack_elapsed_ticks=jnp.asarray(
                [
                    raw["target_attack_elapsed_ticks"]
                    if present[TARGET_ENTITY]
                    else 0
                ],
                dtype=jnp.int32,
            ),
            target_attack_queued=jnp.zeros((1,), dtype=jnp.bool_),
            ticks_since_agent_damage=jnp.asarray(
                [self._ticks_since_damage],
                dtype=jnp.int32,
            ),
            agent_applied_vertical_velocity=jnp.asarray(
                [
                    velocity[0, AGENT_ENTITY, 1]
                    if _movement_state(agent, 2)
                    else 0.0
                ],
                dtype=jnp.float32,
            ),
            agent_fall_speed=jnp.asarray(
                [min(0.0, float(velocity[0, AGENT_ENTITY, 1]))],
                dtype=jnp.float32,
            ),
            knockback_control_lock=jnp.asarray(
                [
                    force_speed
                    > _scalar(self.params.agent_force_per_axis_deadzone)
                ]
            ),
            agent_grounded=jnp.asarray([_movement_state(agent, 15)]),
        )
        arsenal = self._template.arsenal._replace(
            active_ability_slot=jnp.asarray(
                [[execution_slot, -1]],
                dtype=jnp.int32,
            ),
            active_ability_root_slot=jnp.asarray(
                [[active_slot, -1]],
                dtype=jnp.int32,
            ),
            ability_elapsed_seconds=jnp.asarray(active_elapsed),
            # A native ``active`` row is emitted only after
            # InteractionManager drains chainStartQueue. The shared encoder
            # hides JAX's earlier internal queued handle, so project that
            # native admission fact onto the same generic scheduler boundary
            # instead of leaving the template's zero tick (which would hide
            # every live native ability again).
            ability_scheduler_tick=jnp.asarray(
                [[
                    (
                        INTERACTION_QUEUE_DELAY_TICKS + 1
                        if active_slot >= 0
                        else 0
                    ),
                    0,
                ]],
                dtype=jnp.int32,
            ),
            ability_cooldown_seconds=jnp.asarray(
                self._cooldowns[None, ...],
                dtype=jnp.float32,
            ),
            ability_charge_count=jnp.asarray(
                self._charge_counts[None, ...],
                dtype=jnp.int32,
            ),
            ability_charge_timer_seconds=jnp.asarray(
                self._charge_timers[None, ...],
                dtype=jnp.float32,
            ),
            failure_bits=jnp.asarray(
                [wire["arsenal_failure_bits"]],
                dtype=jnp.uint32,
            ),
        )
        state = self._template._replace(
            combat=combat,
            mechanics=mechanics,
            arsenal=arsenal,
        )
        bound = np.zeros(
            (1, ENTITY_COUNT, ABILITY_CAPACITY),
            dtype=np.bool_,
        )
        for ability in abilities:
            native_slot = ability["slot"]
            if native_slot >= len(self._native_to_authored_ability_slot):
                continue
            binding = self.native_profile.abilities[native_slot]
            root_bound = bool(
                ability["authored"]
                and ability["host_legal"]
                and ability["interaction_id"] == binding.interaction_id
                and ability["interaction_type"] == binding.interaction_type
            )
            for slot, mapped_native_slot in enumerate(
                self._authored_to_native_ability_slot
            ):
                if mapped_native_slot != native_slot:
                    continue
                expected = int(
                    self.config.loadout.ability_id[0, AGENT_ENTITY, slot]
                )
                bound[0, AGENT_ENTITY, slot] = root_bound and expected > 0
        return state, resource_available, bound

    def _advance_clocks(
        self,
        world_tick: int,
        agent,
        info: Mapping[str, Any],
    ) -> tuple[float, int]:
        """Advance projected clocks from controlled-step timing evidence.

        ``world_tick`` is the live World's absolute clock. It continues while
        a request waits in the host/bridge queue and therefore is not the
        number of simulated ticks executed for this environment transition.
        Native responses publish the controlled count and measured mean delta
        explicitly. Hytale deliberately has two clocks here:

        * ``InteractionManager.currentTime`` advances by ``World.tickStepNanos``;
        * ``CooldownHandler`` (including charge replenishment) consumes the
          engine-supplied float ``dt``.

        Keep the authored interaction lifecycle on the fixed 30 Hz clock while
        retaining the measured delta for cooldowns and motion observations.

        The absolute-clock fallback is retained only for old/offline evidence
        fixtures which predate the timing fields.
        """

        if self._first_world_tick is None:
            self._first_world_tick = world_tick
        if self._last_world_tick is not None and world_tick < self._last_world_tick:
            raise ValueError("native actor-evidence world tick moved backwards")

        timing_fields_present = (
            "engine_ticks_executed" in info
            or "engine_mean_delta_seconds" in info
        )
        if timing_fields_present:
            if not {
                "engine_ticks_executed",
                "engine_mean_delta_seconds",
            } <= set(info):
                raise ValueError(
                    "native controlled-step timing evidence is incomplete"
                )
            elapsed_ticks = _integer(
                info.get("engine_ticks_executed"),
                "engine_ticks_executed",
            )
            mean_delta_seconds = _number(
                info.get("engine_mean_delta_seconds"),
                "engine_mean_delta_seconds",
            )
            if elapsed_ticks < 0:
                raise ValueError("engine_ticks_executed must be nonnegative")
            if elapsed_ticks == 0:
                if abs(mean_delta_seconds) > 1.0e-9:
                    raise ValueError(
                        "zero executed ticks require zero mean delta seconds"
                    )
                cooldown_delta_seconds = 0.0
                observation_delta_seconds = _scalar(self.params.nominal_dt)
            else:
                if not np.isfinite(mean_delta_seconds) or mean_delta_seconds <= 0.0:
                    raise ValueError(
                        "executed native ticks require a positive finite mean delta"
                    )
                cooldown_delta_seconds = mean_delta_seconds * elapsed_ticks
                observation_delta_seconds = mean_delta_seconds
            interaction_delta_seconds = (
                elapsed_ticks * _scalar(self.params.nominal_dt)
            )
        else:
            if self._last_world_tick is None:
                elapsed_ticks = self.ticks_per_step
            else:
                elapsed_ticks = world_tick - self._last_world_tick
            interaction_delta_seconds = elapsed_ticks / _scalar(
                self.params.ticks_per_second
            )
            cooldown_delta_seconds = interaction_delta_seconds
            observation_delta_seconds = interaction_delta_seconds

        self._last_world_tick = world_tick
        self._episode_tick_count += elapsed_ticks
        if self._tracked_ability_slot >= 0:
            self._tracked_ability_elapsed_seconds += interaction_delta_seconds
        self._cooldowns = np.maximum(
            self._cooldowns - np.float32(cooldown_delta_seconds),
            np.float32(0.0),
        )
        recharging = self._charge_counts < self._charge_capacity
        charge_index = np.clip(
            self._charge_counts,
            0,
            self._charge_times.shape[2] - 1,
        )
        charge_time = np.take_along_axis(
            self._charge_times,
            charge_index[..., None],
            axis=2,
        )[..., 0]
        self._charge_timers += np.where(
            recharging,
            np.float32(cooldown_delta_seconds),
            np.float32(0.0),
        )
        replenished = recharging & (self._charge_timers >= charge_time)
        self._charge_counts = np.where(
            replenished,
            self._charge_counts + np.int32(1),
            self._charge_counts,
        )
        self._charge_timers = np.where(
            replenished,
            np.float32(0.0),
            self._charge_timers,
        )
        health = float(agent["health"])
        if self._last_health is not None and health < self._last_health - 1.0e-6:
            self._ticks_since_damage = 0
        elif health > 0.0 and health < float(agent["max_health"]):
            self._ticks_since_damage += elapsed_ticks
        self._last_health = health
        return observation_delta_seconds, elapsed_ticks

    def _record_accepted_ability(
        self,
        info: Mapping[str, Any],
        resources: np.ndarray,
        resource_available: np.ndarray,
    ) -> None:
        """Mirror native CooldownHandler admission from bridge lifecycle evidence."""

        if "native_ability_accepted" not in info:
            return
        if not _boolean(
            info.get("native_ability_accepted"),
            "native_ability_accepted",
        ):
            self._pending_authored_ability_slot = -1
            return
        native_slot = _integer(
            info.get(
                "native_ability_accepted_slot",
                info.get("native_ability_slot"),
            ),
            "native_ability_accepted_slot",
        )
        if not 0 <= native_slot < len(
            self._native_to_authored_ability_slot
        ):
            raise ValueError("accepted native ability slot is out of range")
        slot = self._native_to_authored_ability_slot[native_slot]
        pending = self._pending_authored_ability_slot
        if (
            0 <= pending < ABILITY_CAPACITY
            and self._authored_to_native_ability_slot[pending] == native_slot
        ):
            slot = pending
        self._pending_authored_ability_slot = -1
        if not bool(self.config.loadout.ability_mask[0, AGENT_ENTITY, slot]):
            raise ValueError("accepted native ability is absent from the loadout")
        if self._charge_counts[AGENT_ENTITY, slot] <= 0:
            raise ValueError("native ability accepted with no remaining charge")

        self._tracked_ability_slot = slot
        self._tracked_ability_execution_slot = self._execution_slot_for_root(
            slot,
            resources,
            resource_available,
        )
        self._tracked_ability_elapsed_seconds = 0.0
        self._charge_counts[AGENT_ENTITY, slot] -= np.int32(1)
        if self._interrupt_recharge[AGENT_ENTITY, slot]:
            self._charge_timers[AGENT_ENTITY, slot] = np.float32(0.0)
        self._cooldowns[AGENT_ENTITY, slot] = np.float32(
            self.config.loadout.ability_cooldown_seconds[
                0,
                AGENT_ENTITY,
                slot,
            ]
        )

    def _execution_slot_for_root(
        self,
        root_slot: int,
        resources: np.ndarray,
        resource_available: np.ndarray,
    ) -> int:
        """Retain the deterministic child selected at native admission."""

        roots = jnp.full(
            self.config.loadout.weapon_family.shape,
            -1,
            dtype=jnp.int32,
        ).at[0, AGENT_ENTITY].set(jnp.int32(root_slot))
        selected = resolve_observed_execution_slot(
            self.config.loadout,
            roots,
            jnp.asarray(resources, dtype=jnp.float32),
            jnp.asarray(resource_available, dtype=jnp.bool_),
        )
        result = int(np.asarray(selected)[0, AGENT_ENTITY])
        if result < 0:
            raise ValueError(
                "native outer-root child selection lacks required resource evidence"
            )
        return result

    def _world_evidence(self, state, geometry, wire):
        world = empty_arsenal_commands(1, entity_count=ENTITY_COUNT).world
        tokens = empty_world_geometry_policy_tokens(
            1,
            self.world_geometry_config,
        )
        raw_tokens = None
        role_available = wire["role_opaque_cell_mask_available"]
        if geometry is not None:
            role_mask = (
                jnp.asarray(wire["role_opaque_cell_mask"])[None, :]
                if role_available
                else None
            )
            # Lazy import avoids the observation/environment package cycle.
            # This neutral path must not pull PPO/training into evidence use.
            from hytalegym.jax.combat.arsenal.environment import (
                geometry_arsenal_world_capabilities,
                geometry_arsenal_world_token_source,
            )
            from hytalegym.jax.world import (
                WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY,
            )

            world = geometry_arsenal_world_capabilities(
                state,
                self.params,
                geometry=geometry,
                config=self.config,
                role_opaque_mask=role_mask,
            )
            if role_available:
                raw_tokens = geometry_arsenal_world_token_source(
                    state,
                    self.params,
                    geometry=geometry,
                    role_opaque_mask=role_mask,
                    geometry_provenance=(
                        WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY
                    ),
                    policy_config=self.world_geometry_config,
                )
                tokens = encode_world_geometry_policy_tokens(
                    raw_tokens,
                    actor_index=0,
                    config=self.world_geometry_config,
                )

        agent = wire["actors"][AGENT_ENTITY]
        actor_values = agent["actor_world_values"]
        actor_available = agent["actor_world_available"]

        def prefer_native(native_value, existing, available):
            return jnp.where(
                jnp.asarray([available]),
                jnp.asarray([native_value]),
                existing,
            )

        target_perceptible = bool(
            wire["actors"][TARGET_ENTITY]["present"]
            and wire["actors"][TARGET_ENTITY]["perceptible"]
        )
        line_of_sight = world.line_of_sight.at[
            :, AGENT_ENTITY
        ].set(target_perceptible)
        line_of_sight_valid = world.line_of_sight_valid.at[
            :, AGENT_ENTITY
        ].set(True)
        direct_target_selected = world.direct_target_selected.at[
            :, AGENT_ENTITY
        ].set(target_perceptible)
        direct_target_selection_valid = (
            world.direct_target_selection_valid.at[
                :, AGENT_ENTITY
            ].set(target_perceptible)
        )
        world = world._replace(
            actor_world_state_available=(
                world.actor_world_state_available
                | jnp.asarray([np.any(actor_available)])
            ),
            actor_controller_medium_available=(
                world.actor_controller_medium_available
                | jnp.asarray([actor_available[0]])
            ),
            actor_submersion_available=(
                world.actor_submersion_available
                | jnp.asarray([actor_available[1]])
            ),
            actor_drop_available=(
                world.actor_drop_available
                | jnp.asarray([actor_available[2]])
            ),
            actor_controller_in_fluid=prefer_native(
                actor_values[0] > 0.5,
                world.actor_controller_in_fluid,
                actor_available[0],
            ),
            actor_feet_submerged=prefer_native(
                actor_values[1] > 0.5,
                world.actor_feet_submerged,
                actor_available[1],
            ),
            actor_eyes_submerged=prefer_native(
                actor_values[2] > 0.5,
                world.actor_eyes_submerged,
                actor_available[1],
            ),
            actor_drop_support_found=prefer_native(
                actor_values[3] > 0.5,
                world.actor_drop_support_found,
                actor_available[2],
            ),
            actor_drop_height=prefer_native(
                np.float32(max(0.0, actor_values[4])),
                world.actor_drop_height,
                actor_available[2],
            ),
            line_of_sight=line_of_sight,
            line_of_sight_valid=line_of_sight_valid,
            direct_target_selected=direct_target_selected,
            direct_target_selection_valid=(
                direct_target_selection_valid
            ),
        )
        return world, tokens, raw_tokens


def _scalar(value):
    return max(abs(float(np.asarray(value))), 1.0e-6)


__all__ = [
    "NATIVE_ACTOR_EVIDENCE_SCHEMA",
    "NATIVE_ACTOR_EVIDENCE_UNKNOWN_STATUS_FAILURE",
    "NATIVE_ACTOR_EVIDENCE_VERSION",
    "NativeActorEvidenceAssembler",
    "NativeActorEvidenceAssembly",
]
