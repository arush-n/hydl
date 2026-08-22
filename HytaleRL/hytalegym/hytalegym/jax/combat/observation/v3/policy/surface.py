"""Actor-safe masks and decoding for the published factored action surface."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.block_interactions.schema.contract import (
    INTERACTION_MOVEMENT_ALL_MASK,
    INTERACTION_MOVEMENT_DIRECTION_MASK,
    INTERACTION_MOVEMENT_DISABLE_BACKWARD,
    INTERACTION_MOVEMENT_DISABLE_FORWARD,
    INTERACTION_MOVEMENT_DISABLE_JUMP,
    INTERACTION_MOVEMENT_DISABLE_LEFT,
    INTERACTION_MOVEMENT_DISABLE_RIGHT,
)
from hytalegym.jax.combat.block_interactions.schema.types import (
    InteractionMovementConstraints,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.blocks import (
    BlockActionCandidatePolicyView,
    block_affordance_policy_staging_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_BLOCK_TRIGGER_NONE,
    ARSENAL_BLOCK_TRIGGER_PRIMARY,
    ARSENAL_GAIT_IDLE,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    ARSENAL_POLICY_BASE_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES,
    ARSENAL_POLICY_LOCOMOTION_DODGE_START,
    ARSENAL_POLICY_PLANNED_NON_TARGET_HEADS,
    arsenal_policy_contract_sha256,
    split_locomotion_choice,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.recipes import (
    RecipeCandidatePolicyView,
    recipe_candidate_policy_staging_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.schema.types import LearnerArsenalAction
from hytalegym.jax.combat.observation.v3.policy.look_deltas import (
    decode_look_delta,
)
from hytalegym.jax.combat.skills import (
    SKILL_APPROACH,
    SKILL_APPROACH_ATTACK,
    SKILL_ATTACK,
    SKILL_RETREAT,
    SKILL_RETREAT_ATTACK,
    SKILL_STRAFE_LEFT,
    SKILL_STRAFE_RIGHT,
)
from hytalegym.jax.world import ACTOR_RECIPE_TARGET_RADIX
from hytalegym.worldgen import native_item_interaction_contract_sha256


Array = jax.Array

_CURRENT_HEAD_NAMES = ARSENAL_POLICY_BASE_ACTION_HEAD_NAMES
_TARGET_HEAD_NAMES = ("block_none_plus_candidates",)
_NEW_HEAD_NAMES = (
    tuple(name for name, _ in ARSENAL_POLICY_PLANNED_NON_TARGET_HEADS)
    + _TARGET_HEAD_NAMES
)
_NEW_FIXED_HEAD_SIZES = tuple(
    size for _, size in ARSENAL_POLICY_PLANNED_NON_TARGET_HEADS
)
ACTION_SURFACE_STAGING_SCHEMA = "hytalerl_combat_action_surface_consumer_v3"
ACTION_SURFACE_STAGING_VERSION = 7


@dataclass(frozen=True)
class ActionSurfaceLayout:
    """Static explicit-factor layout; block capacity remains caller-owned."""

    block_candidate_capacity: int
    head_names: tuple[str, ...]
    head_sizes: tuple[int, ...]

    @property
    def head_count(self) -> int:
        return len(self.head_sizes)

    @property
    def head_offsets(self) -> dict[str, int]:
        offsets: dict[str, int] = {}
        offset = 0
        for name, size in zip(
            self.head_names,
            self.head_sizes,
            strict=True,
        ):
            offsets[name] = offset
            offset += size
        return offsets

    @property
    def yaw_neutral(self) -> int:
        return self.neutral_index("yaw_delta_bins")

    @property
    def body_yaw_neutral(self) -> int:
        return self.neutral_index("body_yaw_delta_bins")

    @property
    def pitch_neutral(self) -> int:
        return self.neutral_index("pitch_delta_bins")

    def head_index(self, name: str) -> int:
        try:
            return self.head_names.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown action-surface head: {name}") from exc

    def head_size(self, name: str) -> int:
        return self.head_sizes[self.head_index(name)]

    def neutral_index(self, name: str) -> int:
        size = self.head_size(name)
        if name in {"yaw_delta_bins", "body_yaw_delta_bins", "pitch_delta_bins"}:
            if size % 2 != 1:
                raise ValueError(f"{name} must have an odd number of bins")
            return size // 2
        return 0

    def split_mask(self, mask: Array) -> dict[str, Array]:
        values = jnp.asarray(mask)
        if values.dtype != jnp.dtype(jnp.bool_):
            raise TypeError("mask must have dtype bool")
        if values.ndim != 2 or values.shape[1] != sum(self.head_sizes):
            raise ValueError(f"mask must have shape (B, {sum(self.head_sizes)})")
        boundaries = _cumulative(self.head_sizes)[:-1]
        return dict(
            zip(
                self.head_names,
                jnp.split(values, boundaries, axis=1),
                strict=True,
            )
        )


class StagedActionSurfaceDecode(NamedTuple):
    """Mask-checked factor values before privileged runtime execution."""

    base: LearnerArsenalAction
    world_move_direction: Array
    yaw_delta_degrees: Array
    body_yaw_delta_degrees: Array
    pitch_delta_degrees: Array
    hotbar_slot: Array
    use_requested: Array
    block_interaction_trigger: Array
    recipe_candidate_index: Array
    block_candidate_index: Array
    action_legal: Array


def action_surface_layout(
    block_candidate_capacity: int,
) -> ActionSurfaceLayout:
    """Build the future layout without imposing a scalar radix budget."""

    if isinstance(block_candidate_capacity, bool):
        raise ValueError("block_candidate_capacity must be a positive integer")
    try:
        capacity = operator.index(block_candidate_capacity)
    except TypeError as exc:
        raise ValueError("block_candidate_capacity must be a positive integer") from exc
    if capacity < 1:
        raise ValueError("block_candidate_capacity must be a positive integer")
    return ActionSurfaceLayout(
        block_candidate_capacity=capacity,
        head_names=_CURRENT_HEAD_NAMES + _NEW_HEAD_NAMES,
        head_sizes=(
            ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES
            + _NEW_FIXED_HEAD_SIZES
            + (capacity + 1,)
        ),
    )


def action_surface_staging_contract_manifest() -> dict[str, object]:
    """Describe the public seam and its still fail-closed runtime boundaries."""

    return {
        "schema": ACTION_SURFACE_STAGING_SCHEMA,
        "version": ACTION_SURFACE_STAGING_VERSION,
        "current_public_policy_sha256": arsenal_policy_contract_sha256(),
        "current_public_head_sizes": list(ARSENAL_POLICY_ACTION_HEAD_SIZES),
        "future_head_order": list(_CURRENT_HEAD_NAMES + _NEW_HEAD_NAMES),
        "future_fixed_head_sizes": {
            name: size
            for name, size in zip(
                _NEW_HEAD_NAMES[:-1],
                _NEW_FIXED_HEAD_SIZES,
                strict=True,
            )
        },
        "block_target_radix": ("caller_owned_actor_candidate_capacity_plus_none"),
        "interaction_movement_constraints": {
            "source": "active_native_interaction_node",
            "missing_effects_while_active": "fail_closed",
            "directional_projection": (
                "neutral_world_move_only_when_any_local_direction_is_locked"
            ),
            "legacy_skill_projection": (
                "mask_each_actor_local_approach_retreat_and_strafe_skill_"
                "against_its_exact_native_direction_bit"
            ),
            "jump_projection": "disable_jump_on_native_bit",
            "runtime_enforcement": (
                "individual_local_direction_bits_after_factored_decode"
            ),
            "pending_heads": ["sprint", "crouch"],
        },
        "transport": "explicit_per_head_int32",
        "factor_dtype": "int32",
        "mask_dtype": "bool",
        "head_resolution": "declared_name_and_derived_size",
        "masking": ("actor_legal_views_before_sampling_selected_choice_rechecked"),
        "joint_legality": (
            "selected_block_and_recipe_targets_must_be_rechecked_before_commit"
        ),
        "standard_input_arbitration": {
            "source": (
                "InteractionTypeUtils.DEFAULT_INTERACTION_BLOCKED_BY_"
                "and_STANDARD_INPUT"
            ),
            "maximum_new_main_roots_per_control_tick": 1,
            "main_roots": [
                "legacy_attack",
                "ability",
                "guard",
                "use",
                "block_primary_or_secondary",
                "craft_recipe",
            ],
            "independently_concurrent": [
                "movement",
                "look",
                "jump",
                "dodge",
            ],
            "invalid_combination": "whole_factored_action_fail_closed",
            "integer_tick_conversion": "ceil(seconds_times_30)",
        },
        "block_trigger_semantics": (
            "primary_or_secondary_resolved_through_equipped_item_authored_root"
        ),
        "block_request_gate": (
            "none_plus_block_candidate_controls_presence_trigger_choice_"
            "remains_round_trip_exact_when_inactive"
        ),
        "shared_block_target_routing": (
            "use_on_routes_the_selected_block_candidate_to_Use_only_"
            "use_off_routes_it_to_the_selected_Primary_or_Secondary_root"
        ),
        "native_item_interaction_contract_sha256": (
            native_item_interaction_contract_sha256()
        ),
        "item_trigger_availability": (
            "active_hotbar_semantic_item_lookup_of_exact_native_root_presence_"
            "unknown_or_ambiguous_item_closed"
        ),
        "ability_slot_native_mapping": (
            "profile_local_slot_to_asset_authored_root_and_exact_"
            "InteractionType_not_raw_Ability1_to_Ability3_ordinal"
        ),
        "runtime_execution": (
            "injected_executor_with_persistent_block_chain_state_fail_closed"
        ),
        "block_consumer_sha256": (block_affordance_policy_staging_contract_sha256()),
        "recipe_consumer_sha256": (recipe_candidate_policy_staging_contract_sha256()),
        "publication": ("published_in_public_policy_v7_and_factored_PPO_rollouts"),
        "not_claimed": [
            "native_action_acceptance",
            "equipped_item_trigger_leaf_execution",
        ],
    }


def action_surface_staging_contract_sha256() -> str:
    """Return the canonical identity of the published consumer seam."""

    payload = json.dumps(
        action_surface_staging_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def staged_action_surface_mask(
    base_action_mask: Array,
    valid: Array,
    block_candidates: BlockActionCandidatePolicyView,
    recipe_candidates: RecipeCandidatePolicyView,
    *,
    use_available: Array,
    block_trigger_available: Array,
    interaction_movement: InteractionMovementConstraints | None = None,
    dodge_available: Array,
) -> Array:
    """Compose every action-surface head from actor-legal evidence."""

    if not isinstance(block_candidates, BlockActionCandidatePolicyView):
        raise TypeError("block_candidates must be BlockActionCandidatePolicyView")
    if not isinstance(recipe_candidates, RecipeCandidatePolicyView):
        raise TypeError("recipe_candidates must be RecipeCandidatePolicyView")
    base = jnp.asarray(base_action_mask)
    if base.dtype != jnp.dtype(jnp.bool_):
        raise TypeError("base_action_mask must have dtype bool")
    base_size = sum(ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES)
    if base.ndim != 2 or base.shape[1] != base_size:
        raise ValueError(f"base_action_mask must have shape (B, {base_size})")
    batch = base.shape[0]
    valid_mask = _batch_bool(valid, batch, "valid")
    base &= valid_mask[:, None]
    movement = _interaction_movement_constraints(
        interaction_movement,
        batch,
    )
    movement_known = (
        movement.movement_effects_present
        & (movement.movement_lock_mask >= 0)
        & (movement.movement_lock_mask <= jnp.int32(INTERACTION_MOVEMENT_ALL_MASK))
    )
    effective_movement_lock = jnp.where(
        movement.active,
        jnp.where(
            movement_known,
            jnp.where(
                movement.movement_disable_all,
                jnp.int32(INTERACTION_MOVEMENT_ALL_MASK),
                movement.movement_lock_mask,
            ),
            jnp.int32(INTERACTION_MOVEMENT_ALL_MASK),
        ),
        jnp.int32(0),
    )
    translation_available = (
        effective_movement_lock & jnp.int32(INTERACTION_MOVEMENT_DIRECTION_MASK)
    ) == 0
    forward_available = (
        effective_movement_lock & jnp.int32(INTERACTION_MOVEMENT_DISABLE_FORWARD)
    ) == 0
    backward_available = (
        effective_movement_lock & jnp.int32(INTERACTION_MOVEMENT_DISABLE_BACKWARD)
    ) == 0
    left_available = (
        effective_movement_lock & jnp.int32(INTERACTION_MOVEMENT_DISABLE_LEFT)
    ) == 0
    right_available = (
        effective_movement_lock & jnp.int32(INTERACTION_MOVEMENT_DISABLE_RIGHT)
    ) == 0
    for skill_id, available in (
        (SKILL_APPROACH, forward_available),
        (SKILL_APPROACH_ATTACK, forward_available),
        (SKILL_RETREAT, backward_available),
        (SKILL_RETREAT_ATTACK, backward_available),
        (SKILL_STRAFE_LEFT, left_available),
        (SKILL_STRAFE_RIGHT, right_available),
    ):
        base = base.at[:, skill_id].set(base[:, skill_id] & available)
    jump_available = (
        effective_movement_lock & jnp.int32(INTERACTION_MOVEMENT_DISABLE_JUMP)
    ) == 0
    jump_offset = sum(ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES[:-1])
    if ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES[-1] != 2:
        raise RuntimeError("jump action head must remain binary")
    base = base.at[:, jump_offset + 1].set(base[:, jump_offset + 1] & jump_available)
    use_mask = _batch_bool(use_available, batch, "use_available")
    trigger_mask = jnp.asarray(block_trigger_available)
    if trigger_mask.dtype != jnp.dtype(jnp.bool_) or trigger_mask.shape != (batch, 2):
        raise ValueError(
            "block_trigger_available must have shape (B, 2) and dtype bool"
        )
    if block_candidates.available.shape != (batch,):
        raise ValueError("block candidate batch does not match base mask")
    if jnp.asarray(block_candidates.available).dtype != jnp.dtype(jnp.bool_):
        raise TypeError("block candidate availability must have dtype bool")
    if (
        block_candidates.candidate_mask.ndim != 2
        or block_candidates.candidate_mask.shape[0] != batch
    ):
        raise ValueError("block candidate mask must have shape (B, capacity)")
    if jnp.asarray(block_candidates.candidate_mask).dtype != jnp.dtype(jnp.bool_):
        raise TypeError("block candidate mask must have dtype bool")
    if recipe_candidates.available.shape != (batch,):
        raise ValueError("recipe candidate batch does not match base mask")
    if jnp.asarray(recipe_candidates.available).dtype != jnp.dtype(jnp.bool_):
        raise TypeError("recipe candidate availability must have dtype bool")
    if (
        recipe_candidates.candidate_mask.ndim != 2
        or recipe_candidates.candidate_mask.shape[0] != batch
        or recipe_candidates.candidate_mask.shape[1] + 1 != ACTOR_RECIPE_TARGET_RADIX
    ):
        raise ValueError("recipe candidate mask does not match the World target radix")
    if jnp.asarray(recipe_candidates.candidate_mask).dtype != jnp.dtype(jnp.bool_):
        raise TypeError("recipe candidate mask must have dtype bool")

    any_trigger = jnp.any(trigger_mask, axis=1)
    block_available = (
        valid_mask
        & block_candidates.available
        & jnp.any(block_candidates.candidate_mask, axis=1)
        & any_trigger
    )
    target_available = (
        valid_mask
        & block_candidates.available
        & jnp.any(block_candidates.candidate_mask, axis=1)
        & (any_trigger | use_mask)
    )
    recipe_mask = (
        recipe_candidates.candidate_mask
        & recipe_candidates.available[:, None]
        & valid_mask[:, None]
    )
    block_mask = (
        block_candidates.candidate_mask
        & block_candidates.available[:, None]
        & valid_mask[:, None]
        & target_available[:, None]
    )
    layout = action_surface_layout(block_mask.shape[1])
    fixed_masks = {
        # One head, three legality rules: idle is always available, the gait
        # ring needs translation, and each dodge direction keeps the authored
        # per-direction legality it had when dodge was its own head.
        "locomotion_gait_compass": jnp.concatenate(
            (
                jnp.ones((batch, 1), dtype=jnp.bool_),
                jnp.broadcast_to(
                    (valid_mask & translation_available)[:, None],
                    (batch, ARSENAL_POLICY_LOCOMOTION_DODGE_START - 1),
                ),
                jnp.asarray(dodge_available, dtype=jnp.bool_)
                & valid_mask[:, None],
            ),
            axis=1,
        ),
        "yaw_delta_bins": _neutral_or_valid(
            valid_mask,
            layout.head_size("yaw_delta_bins"),
            layout.yaw_neutral,
        ),
        "body_yaw_delta_bins": _neutral_or_valid(
            valid_mask,
            layout.head_size("body_yaw_delta_bins"),
            layout.body_yaw_neutral,
        ),
        "pitch_delta_bins": _neutral_or_valid(
            valid_mask,
            layout.head_size("pitch_delta_bins"),
            layout.pitch_neutral,
        ),
        # Every slot is selectable, including empty ones: a player can press any
        # number key, and whether the slot HOLDS anything is the loadout's
        # business -- `effective_equipped` already reads the live stack. An
        # invalid row keeps choice 0, the no-switch neutral, so the row can never
        # collapse to an unsatisfiable all-false mask.
        "hotbar_none_plus_slots": _neutral_or_valid(
            valid_mask,
            layout.head_size("hotbar_none_plus_slots"),
            layout.neutral_index("hotbar_none_plus_slots"),
        ),
        "use_off_on": jnp.stack(
            (
                jnp.ones((batch,), dtype=jnp.bool_),
                valid_mask & use_mask,
            ),
            axis=1,
        ),
        "block_primary_secondary_trigger": jnp.where(
            block_available[:, None],
            trigger_mask & valid_mask[:, None],
            jnp.asarray((True, False), dtype=jnp.bool_)[None, :],
        ),
    }
    return jnp.concatenate(
        (
            base,
            *(fixed_masks[name] for name, _ in ARSENAL_POLICY_PLANNED_NON_TARGET_HEADS),
            # The recipe head is not published for a combat agent, so its mask
            # is validated above but contributes no columns.
            jnp.concatenate(
                (
                    jnp.ones((batch, 1), dtype=jnp.bool_),
                    block_mask,
                ),
                axis=1,
            ),
        ),
        axis=1,
    )


def decode_staged_action_surface_factors(
    action_factors: Array,
    action_mask: Array,
    *,
    layout: ActionSurfaceLayout,
    maximum_turn_degrees: float = 45.0,
) -> StagedActionSurfaceDecode:
    """Decode explicit factors only when every selected choice is legal."""

    if not isinstance(layout, ActionSurfaceLayout):
        raise TypeError("layout must be an ActionSurfaceLayout")
    maximum_turn = float(maximum_turn_degrees)
    if not math.isfinite(maximum_turn) or maximum_turn <= 0.0:
        raise ValueError("maximum_turn_degrees must be finite and positive")
    factors = jnp.asarray(action_factors)
    if factors.dtype != jnp.dtype(jnp.int32):
        raise TypeError("action_factors must have dtype int32")
    expected_heads = layout.head_count
    if factors.ndim != 2 or factors.shape[1] != expected_heads:
        raise ValueError(f"action_factors must have shape (B, {expected_heads})")
    mask = jnp.asarray(action_mask)
    if mask.dtype != jnp.dtype(jnp.bool_):
        raise TypeError("action_mask must have dtype bool")
    if mask.shape != (factors.shape[0], sum(layout.head_sizes)):
        raise ValueError("action_mask must match action_factors batch and layout size")

    in_range = jnp.ones((factors.shape[0],), dtype=jnp.bool_)
    selected_legal = jnp.ones_like(in_range)
    offset = 0
    for index, size in enumerate(layout.head_sizes):
        value = factors[:, index]
        value_in_range = (value >= 0) & (value < size)
        safe_value = jnp.clip(value, 0, size - 1)
        selected_legal &= jnp.take_along_axis(
            mask[:, offset : offset + size],
            safe_value[:, None],
            axis=1,
        )[:, 0]
        in_range &= value_in_range
        offset += size
    factor = {name: factors[:, layout.head_index(name)] for name in layout.head_names}
    legacy_attack = (
        (factor["base_action"] == SKILL_ATTACK)
        | (factor["base_action"] == SKILL_APPROACH_ATTACK)
        | (factor["base_action"] == SKILL_RETREAT_ATTACK)
    )
    use_requested = factor["use_off_on"] == 1
    block_requested = (
        (factor["block_none_plus_candidates"] > 0) & ~use_requested
    )
    standard_input_count = (
        legacy_attack.astype(jnp.int32)
        + (factor["ability_none_plus_slots"] > 0).astype(jnp.int32)
        + (factor["guard_off_on"] == 1).astype(jnp.int32)
        + use_requested.astype(jnp.int32)
        + block_requested.astype(jnp.int32)
        # No recipe head, so crafting can never be one of the competing roots.
    )
    legal = in_range & selected_legal & (standard_input_count <= 1)

    yaw_delta = _signed_delta(
        factor["yaw_delta_bins"],
        size=layout.head_size("yaw_delta_bins"),
        maximum=maximum_turn,
    )
    body_yaw_delta = _signed_delta(
        factor["body_yaw_delta_bins"],
        size=layout.head_size("body_yaw_delta_bins"),
        maximum=maximum_turn,
    )
    pitch_delta = _signed_delta(
        factor["pitch_delta_bins"],
        size=layout.head_size("pitch_delta_bins"),
        maximum=maximum_turn,
    )
    # The recipe head is no longer published for a combat agent, so no recipe
    # is ever requested. The field is retained because the world-action
    # executor and the native channels still read it.
    # Choice 0 is "no switch", so the slot is the choice minus one.
    hotbar_slot = factor["hotbar_none_plus_slots"] - jnp.int32(1)
    recipe_index = jnp.full_like(
        factor["locomotion_gait_compass"], -1, dtype=jnp.int32
    )
    block_index = factor["block_none_plus_candidates"] - jnp.int32(1)
    block_trigger = factor["block_primary_secondary_trigger"] + jnp.int32(
        ARSENAL_BLOCK_TRIGGER_PRIMARY
    )
    staged_gait, staged_direction, staged_dodge = split_locomotion_choice(
        factor["locomotion_gait_compass"]
    )
    return StagedActionSurfaceDecode(
        base=LearnerArsenalAction(
            skill_id=jnp.where(
                legal,
                factor["base_action"],
                jnp.int32(-1),
            ),
            ability_slot=jnp.where(
                legal,
                factor["ability_none_plus_slots"] - jnp.int32(1),
                jnp.int32(-1),
            ),
            guard_held=legal & (factor["guard_off_on"] == 1),
            dodge_direction=jnp.where(legal, staged_dodge, jnp.int32(0)),
            jump_held=legal & (factor["jump_off_on"] == 1),
            gait=jnp.where(legal, staged_gait, jnp.int32(ARSENAL_GAIT_IDLE)),
            world_move_direction=jnp.where(legal, staged_direction, jnp.int32(0)),
            yaw_delta_degrees=jnp.where(
                legal,
                yaw_delta,
                jnp.float32(0.0),
            ),
            body_yaw_delta_degrees=jnp.where(
                legal,
                body_yaw_delta,
                jnp.float32(0.0),
            ),
            pitch_delta_degrees=jnp.where(
                legal,
                pitch_delta,
                jnp.float32(0.0),
            ),
            hotbar_slot=jnp.where(legal, hotbar_slot, jnp.int32(-1)),
            use_requested=legal & (factor["use_off_on"] == 1),
            block_interaction_trigger=jnp.where(
                legal,
                block_trigger,
                jnp.int32(ARSENAL_BLOCK_TRIGGER_NONE),
            ),
            recipe_candidate_index=jnp.where(
                legal,
                recipe_index,
                jnp.int32(-1),
            ),
            block_candidate_index=jnp.where(
                legal,
                block_index,
                jnp.int32(-1),
            ),
        ),
        world_move_direction=jnp.where(legal, staged_direction, jnp.int32(0)),
        yaw_delta_degrees=jnp.where(
            legal,
            yaw_delta,
            jnp.float32(0.0),
        ),
        body_yaw_delta_degrees=jnp.where(
            legal,
            body_yaw_delta,
            jnp.float32(0.0),
        ),
        pitch_delta_degrees=jnp.where(
            legal,
            pitch_delta,
            jnp.float32(0.0),
        ),
        hotbar_slot=jnp.where(legal, hotbar_slot, jnp.int32(-1)),
        use_requested=legal & (factor["use_off_on"] == 1),
        block_interaction_trigger=jnp.where(
            legal,
            block_trigger,
            jnp.int32(ARSENAL_BLOCK_TRIGGER_NONE),
        ),
        recipe_candidate_index=jnp.where(
            legal,
            recipe_index,
            jnp.int32(-1),
        ),
        block_candidate_index=jnp.where(
            legal,
            block_index,
            jnp.int32(-1),
        ),
        action_legal=legal,
    )


def _neutral_or_valid(
    valid: Array,
    size: int,
    neutral: int,
) -> Array:
    indexes = jnp.arange(size, dtype=jnp.int32)
    return (indexes[None, :] == neutral) | valid[:, None]


def _batch_bool(value: Array, batch: int, name: str) -> Array:
    result = jnp.asarray(value)
    if result.dtype != jnp.dtype(jnp.bool_):
        raise TypeError(f"{name} must have dtype bool")
    if result.shape != (batch,):
        raise ValueError(f"{name} must have shape (B,)")
    return result


def _interaction_movement_constraints(
    value: InteractionMovementConstraints | None,
    batch: int,
) -> InteractionMovementConstraints:
    if value is None:
        return InteractionMovementConstraints(
            active=jnp.zeros((batch,), dtype=jnp.bool_),
            movement_effects_present=jnp.zeros(
                (batch,),
                dtype=jnp.bool_,
            ),
            movement_disable_all=jnp.zeros(
                (batch,),
                dtype=jnp.bool_,
            ),
            movement_lock_mask=jnp.zeros((batch,), dtype=jnp.int32),
            horizontal_speed_multiplier=jnp.ones(
                (batch,),
                dtype=jnp.float32,
            ),
        )
    if not isinstance(value, InteractionMovementConstraints):
        raise TypeError("interaction_movement must be InteractionMovementConstraints")
    active = _batch_bool(value.active, batch, "interaction_movement.active")
    present = _batch_bool(
        value.movement_effects_present,
        batch,
        "interaction_movement.movement_effects_present",
    )
    disable_all = _batch_bool(
        value.movement_disable_all,
        batch,
        "interaction_movement.movement_disable_all",
    )
    mask = jnp.asarray(value.movement_lock_mask)
    if mask.dtype != jnp.int32 or mask.shape != (batch,):
        raise ValueError("interaction_movement.movement_lock_mask must be int32[B]")
    multiplier = jnp.asarray(value.horizontal_speed_multiplier)
    if multiplier.dtype != jnp.float32 or multiplier.shape != (batch,):
        raise ValueError(
            "interaction_movement.horizontal_speed_multiplier must be float32[B]"
        )
    return InteractionMovementConstraints(
        active=active,
        movement_effects_present=present,
        movement_disable_all=disable_all,
        movement_lock_mask=mask,
        horizontal_speed_multiplier=multiplier,
    )


def _signed_delta(
    choice: Array,
    *,
    size: int,
    maximum: float,
) -> Array:
    return decode_look_delta(
        choice,
        size,
        maximum_degrees=maximum,
    )


def _cumulative(values: tuple[int, ...]) -> tuple[int, ...]:
    result = []
    total = 0
    for value in values:
        total += value
        result.append(total)
    return tuple(result)


__all__ = [
    "ActionSurfaceLayout",
    "StagedActionSurfaceDecode",
    "ACTION_SURFACE_STAGING_SCHEMA",
    "ACTION_SURFACE_STAGING_VERSION",
    "action_surface_layout",
    "action_surface_staging_contract_manifest",
    "action_surface_staging_contract_sha256",
    "decode_staged_action_surface_factors",
    "staged_action_surface_mask",
]
