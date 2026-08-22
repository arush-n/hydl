"""Actor-safe dense policy surface for learner observation version 3."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import OBSERVATION_CAPACITY
from hytalegym.jax.combat.arsenal.schema.spec import combat_arsenal_contract_sha256
from hytalegym.jax.combat.mechanics import (
    RESOURCE_COUNT,
    STATUS_CAPACITY,
)
from hytalegym.jax.combat.observation.v1.schema.contract import (
    LEARNER_ACTION_COUNT,
    COMBAT_FLOAT_FEATURES,
    SELF_FLOAT_FEATURES,
    TARGET_FLOAT_FEATURES,
)
from hytalegym.jax.combat.observation.v3.schema.contract import (
    ABILITY_FLOAT_SIZE,
    ACTOR_WORLD_FLOAT_SIZE,
    ACTOR_WORLD_MASK_SIZE,
    DEFENSE_FLOAT_SIZE,
    DODGE_ACTION_COUNT,
    MOVEMENT_STATE_SIZE,
    STATUS_FLOAT_SIZE,
)
from hytalegym.jax.combat.observation.v3.schema.spec import (
    learner_observation_v3_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerArsenalAction,
    LearnerCombatObservationV3,
)
from hytalegym.jax.combat.observation.v3.policy.look_deltas import (
    decode_look_delta,
    encode_look_delta,
    look_delta_values,
)
from hytalegym.jax.combat.observation.v3.policy.actions import (
    LearnerArsenalActionContext,
    learner_arsenal_action_context,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.blocks import (
    BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.contract import (
    RECIPE_CANDIDATE_EMBEDDING_SIZE,
    recipe_candidate_encoding_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.tokens.inventory import (
    INVENTORY_POLICY_FLOAT_FEATURES,
    INVENTORY_POLICY_SOURCE_FIELDS,
    INVENTORY_POLICY_TOKEN_CAPACITY,
    InventoryPolicyTokens,
    empty_inventory_policy_tokens,
    inventory_policy_flat_size,
    mask_inventory_policy_tokens,
)
from hytalegym.jax.combat.observation.v3.tokens.light import (
    ACTOR_LIGHT_POLICY_FEATURES,
    ActorLightPolicyTokens,
    actor_light_policy_flat_size,
    align_actor_light_policy_tokens,
    empty_actor_light_policy_tokens,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
    WorldGeometryPolicyConfig,
    normalize_world_geometry_policy_config,
    world_geometry_policy_config_manifest,
    world_geometry_policy_flat_size,
)
from hytalegym.jax.combat.skills import SKILL_COUNT, SKILL_IDLE
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    ENTITY_COUNT,
    GAIT_COUNT,
    GAIT_IDLE,
    GAIT_RUN,
    GAIT_SNEAK,
    GAIT_SPRINT,
    GAIT_WALK,
)
from hytalegym.jax.world import (
    ACTOR_RECIPE_CANDIDATE_CAPACITY,
    actor_inventory_token_contract_sha256,
    actor_block_light_token_contract_sha256,
    actor_block_action_candidate_contract_sha256,
    actor_recipe_candidate_contract_sha256,
    world_geometry_token_contract_sha256,
)
from hytalegym.jax.combat.inventory import (
    DEFAULT_CONTAINER_CAPACITIES,
    HOTBAR_CAPACITY,
    native_inventory_contract_sha256,
)


ARSENAL_POLICY_SCHEMA = "hytalerl_combat_arsenal_policy_v14"
ARSENAL_POLICY_VERSION = 14

ARSENAL_POLICY_EXCLUDED_LEGACY_COMBAT_FLOAT_FEATURES = (
    "target_phase_idle",
    "target_phase_windup",
    "target_phase_sweep",
    "target_phase_recovery",
    "target_phase_cooldown",
    "target_attack_progress",
)
ARSENAL_POLICY_COMBAT_FLOAT_FEATURES = tuple(
    name
    for name in COMBAT_FLOAT_FEATURES
    if name not in ARSENAL_POLICY_EXCLUDED_LEGACY_COMBAT_FLOAT_FEATURES
)
_ARSENAL_POLICY_COMBAT_FLOAT_INDICES = tuple(
    COMBAT_FLOAT_FEATURES.index(name) for name in ARSENAL_POLICY_COMBAT_FLOAT_FEATURES
)


#: Width the mixed-radix reservation is computed against. This was int32, which
#: reserved enough room to pack every factor combination into a SINGLE int32.
#: Nothing does that: the transport is `explicit_per_head_int32` -- one int32 per
#: head -- and `packed_radix_order` is published as ``None``. The only consumer
#: of the budget is a manifest field. Held at int32 the surface was full: with 11
#: heads the largest addition that still fit was size 2, which is not enough for
#: a 9-slot hotbar and would have left the agent permanently unable to swap
#: weapons. Widening to int64 keeps a real sanity bound on runaway target heads
#: while removing a reservation for an encoding that does not exist.
#:
#: The cost, stated plainly: a future single-int32 packed transport is no longer
#: possible and would need int64. Per-head int32 transport is unaffected, since
#: the binding constraint there is per-head size, not the product.
_ACTION_RADIX_RESERVATION_DTYPE = jnp.int64


def action_head_combination_count(head_sizes: Sequence[int]) -> int:
    """Return a checked mixed-radix product inside the radix reservation."""

    if not head_sizes:
        raise ValueError("action head sizes must not be empty")
    maximum = int(jnp.iinfo(_ACTION_RADIX_RESERVATION_DTYPE).max)
    count = 1
    for raw_size in head_sizes:
        if isinstance(raw_size, bool):
            raise ValueError("action head sizes must be positive integers")
        try:
            size = operator.index(raw_size)
        except TypeError as exc:
            raise ValueError("action head sizes must be positive integers") from exc
        if size < 1:
            raise ValueError("action head sizes must be positive integers")
        if count > maximum // size:
            raise OverflowError(
                "mixed-radix action combinations exceed the radix reservation"
            )
        count *= size
    return count


ARSENAL_SKILL_ACTION_START = 0
ARSENAL_ABILITY_ACTION_START = ARSENAL_SKILL_ACTION_START + LEARNER_ACTION_COUNT
ARSENAL_GUARD_ACTION_START = ARSENAL_ABILITY_ACTION_START + OBSERVATION_CAPACITY + 1
# Dodge is no longer a base action -- it is one branch of the locomotion head --
# so it drops out of this chain and jump follows guard directly.
ARSENAL_JUMP_ACTION_START = ARSENAL_GUARD_ACTION_START + 2
# Locomotion is one categorical because the gaits are physically disjoint: an
# actor is idle, walking, running, sprinting or sneaking, never two at once, and
# a dodge is a dash that replaces ordinary translation for its duration. Giving
# each its own head would let the policy emit combinations the engine cannot
# represent. Jump stays separate precisely because it is *not* disjoint -- a
# sprint-jump is a real move -- so it remains an independent two-choice head.
ARSENAL_POLICY_COMPASS_DIRECTIONS = 8
ARSENAL_GAIT_IDLE = GAIT_IDLE
ARSENAL_GAIT_WALK = GAIT_WALK
ARSENAL_GAIT_RUN = GAIT_RUN
ARSENAL_GAIT_SPRINT = GAIT_SPRINT
ARSENAL_GAIT_SNEAK = GAIT_SNEAK
#: Moving gaits only -- idle is the head's zero choice, not a ring entry.
ARSENAL_POLICY_GAIT_COUNT = GAIT_COUNT - 1
#: First locomotion choice that means "dodge", after idle and the gait ring.
ARSENAL_POLICY_LOCOMOTION_DODGE_START = 1 + (
    ARSENAL_POLICY_GAIT_COUNT * ARSENAL_POLICY_COMPASS_DIRECTIONS
)
ARSENAL_POLICY_LOCOMOTION_HEAD_SIZE = (
    ARSENAL_POLICY_LOCOMOTION_DODGE_START + DODGE_ACTION_COUNT
)

ARSENAL_POLICY_BASE_ACTION_HEAD_NAMES = (
    "base_action",
    "ability_none_plus_slots",
    "guard_off_on",
    "jump_off_on",
)
ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES = (
    LEARNER_ACTION_COUNT,
    OBSERVATION_CAPACITY + 1,
    2,
    2,
)
ARSENAL_POLICY_PLANNED_NON_TARGET_HEADS = (
    ("locomotion_gait_compass", ARSENAL_POLICY_LOCOMOTION_HEAD_SIZE),
    ("yaw_delta_bins", 9),
    # The body's own steer. `yaw_delta_bins` above is the camera; without this
    # head the body carries no command at all and an actor is frozen at whatever
    # heading it reset to, because travel rides the body. Same 9-bin geometric
    # grid as the head so the two levers quantise identically.
    ("body_yaw_delta_bins", 9),
    ("pitch_delta_bins", 5),
    # Weapon switching, the last thing a player could do that the agent could
    # not. Choice 0 is "leave the live slot alone"; 1..9 select hotbar slots
    # 0..8, matching a player's direct number-key access rather than a cycle.
    # `HOTBAR_CAPACITY` is 9 and the bridge already applies `hotbarSlot`
    # (`NativeEnvironmentSession` 3831), so only the head and the transport were
    # ever missing.
    ("hotbar_none_plus_slots", 10),
    ("use_off_on", 2),
    ("block_primary_secondary_trigger", 2),
)
ARSENAL_BLOCK_TRIGGER_NONE = 0
ARSENAL_BLOCK_TRIGGER_PRIMARY = 1
ARSENAL_BLOCK_TRIGGER_SECONDARY = 2
ARSENAL_POLICY_PLANNED_NON_TARGET_HEAD_SIZES = tuple(
    size for _, size in ARSENAL_POLICY_PLANNED_NON_TARGET_HEADS
)
ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY = 16
ARSENAL_POLICY_RECIPE_CANDIDATE_EMBEDDING_SIZE = RECIPE_CANDIDATE_EMBEDDING_SIZE
# Crafting is out of scope for a combat agent, so the recipe head is not
# published. ``LearnerArsenalAction.recipe_candidate_index`` still exists and
# decodes to a constant -1 (no recipe requested), which keeps the world-action
# executor and the native channels unchanged while freeing the int32 target
# radix budget the head consumed -- it was 289 of 324, leaving no room to grow.
ARSENAL_POLICY_TARGET_HEADS = (
    (
        "block_none_plus_candidates",
        ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY + 1,
    ),
)
ARSENAL_POLICY_ACTION_HEAD_NAMES = (
    ARSENAL_POLICY_BASE_ACTION_HEAD_NAMES
    + tuple(name for name, _ in ARSENAL_POLICY_PLANNED_NON_TARGET_HEADS)
    + tuple(name for name, _ in ARSENAL_POLICY_TARGET_HEADS)
)
ARSENAL_POLICY_ACTION_HEAD_SIZES = (
    ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES
    + ARSENAL_POLICY_PLANNED_NON_TARGET_HEAD_SIZES
    + tuple(size for _, size in ARSENAL_POLICY_TARGET_HEADS)
)


def action_target_radix_budget(
    base_head_sizes: Sequence[int],
    planned_non_target_head_sizes: Sequence[int],
) -> int:
    """Reserve target capacity against the complete planned factor set."""

    planned_count = action_head_combination_count(
        tuple(base_head_sizes) + tuple(planned_non_target_head_sizes)
    )
    reservation = int(jnp.iinfo(_ACTION_RADIX_RESERVATION_DTYPE).max)
    return reservation // planned_count


ARSENAL_POLICY_TARGET_RADIX_BUDGET = action_target_radix_budget(
    ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES,
    ARSENAL_POLICY_PLANNED_NON_TARGET_HEAD_SIZES,
)


def validate_action_target_head_sizes(
    target_head_sizes: Sequence[int],
    *,
    base_head_sizes: Sequence[int] = ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES,
    planned_non_target_head_sizes: Sequence[
        int
    ] = ARSENAL_POLICY_PLANNED_NON_TARGET_HEAD_SIZES,
) -> int:
    """Validate target radices without making landing order observable."""

    target_count = action_head_combination_count(target_head_sizes)
    budget = action_target_radix_budget(
        base_head_sizes,
        planned_non_target_head_sizes,
    )
    if target_count > budget:
        raise OverflowError(
            "action target head combinations exceed reserved int32 budget: "
            f"{target_count} > {budget}"
        )
    return target_count


ARSENAL_POLICY_ACTION_SIZE = sum(ARSENAL_POLICY_ACTION_HEAD_SIZES)
ARSENAL_POLICY_COMBINATION_COUNT = action_head_combination_count(
    ARSENAL_POLICY_ACTION_HEAD_SIZES
)

ARSENAL_POLICY_BASE_OBSERVATION_SIZE = (
    len(SELF_FLOAT_FEATURES)
    + len(TARGET_FLOAT_FEATURES)
    + 1
    + len(ARSENAL_POLICY_COMBAT_FLOAT_FEATURES)
    + 2 * RESOURCE_COUNT
    + DEFENSE_FLOAT_SIZE
    + STATUS_CAPACITY * STATUS_FLOAT_SIZE
    + STATUS_CAPACITY
    + OBSERVATION_CAPACITY * ABILITY_FLOAT_SIZE
    + 2 * OBSERVATION_CAPACITY
    + ACTOR_WORLD_FLOAT_SIZE
    + ACTOR_WORLD_MASK_SIZE
    + 2 * MOVEMENT_STATE_SIZE
    + SKILL_COUNT
    + 1
    + DODGE_ACTION_COUNT
    + 1
    + 1
    + inventory_policy_flat_size()
)
ARSENAL_POLICY_ACTION_SURFACE_OBSERVATION_SIZE = (
    ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY
    * BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE
    + ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY
    + 1
    + ACTOR_RECIPE_CANDIDATE_CAPACITY * ARSENAL_POLICY_RECIPE_CANDIDATE_EMBEDDING_SIZE
    + ACTOR_RECIPE_CANDIDATE_CAPACITY
    + 1
)
ARSENAL_POLICY_OBSERVATION_SIZE = (
    ARSENAL_POLICY_BASE_OBSERVATION_SIZE
    + world_geometry_policy_flat_size(DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG)
    + actor_light_policy_flat_size(DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG.token_capacity)
    + ARSENAL_POLICY_ACTION_SURFACE_OBSERVATION_SIZE
)


def arsenal_policy_observation_size(
    world_geometry_config: WorldGeometryPolicyConfig | None = None,
) -> int:
    """Return the dense policy width for one static token configuration."""

    return (
        ARSENAL_POLICY_BASE_OBSERVATION_SIZE
        + world_geometry_policy_flat_size(world_geometry_config)
        + actor_light_policy_flat_size(
            normalize_world_geometry_policy_config(world_geometry_config).token_capacity
        )
        + ARSENAL_POLICY_ACTION_SURFACE_OBSERVATION_SIZE
    )


def arsenal_policy_observation(
    observation: LearnerCombatObservationV3,
    block_candidates=None,
    recipe_candidate_encoding=None,
    *,
    inventory_tokens: InventoryPolicyTokens | None = None,
    light_tokens: ActorLightPolicyTokens | None = None,
):
    """Flatten only legal actor evidence; semantic integer IDs stay excluded."""

    batch = observation.valid.shape[0]
    resource_mask = observation.resource_mask[:, AGENT_ENTITY]
    status_mask = observation.status_mask[:, AGENT_ENTITY]
    ability_mask = observation.ability_mask[:, AGENT_ENTITY]
    actor_world_value_mask = jnp.stack(
        (
            observation.actor_world_mask[:, 0],
            observation.actor_world_mask[:, 1],
            observation.actor_world_mask[:, 1],
            observation.actor_world_mask[:, 2],
            observation.actor_world_mask[:, 2],
        ),
        axis=1,
    )
    movement_state_mask = jnp.asarray(
        observation.movement_state_mask,
        dtype=jnp.bool_,
    )
    if inventory_tokens is None:
        inventory_tokens = empty_inventory_policy_tokens(batch)
    inventory_tokens = mask_inventory_policy_tokens(
        inventory_tokens,
        observation.valid,
    )
    geometry_capacity = observation.world_geometry.token_mask.shape[1]
    if light_tokens is None:
        light_tokens = empty_actor_light_policy_tokens(
            batch,
            geometry_capacity,
        )
    light_tokens = align_actor_light_policy_tokens(
        light_tokens,
        observation.world_geometry.token_mask,
        observation.world_geometry.available,
        observation.valid,
    )
    if block_candidates is None:
        block_f32 = jnp.zeros(
            (
                batch,
                ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
                BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE,
            ),
            dtype=jnp.float32,
        )
        block_mask = jnp.zeros(
            (batch, ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY),
            dtype=jnp.bool_,
        )
        block_available = jnp.zeros((batch,), dtype=jnp.bool_)
    else:
        block_f32 = jnp.asarray(block_candidates.candidate_f32)
        block_mask = jnp.asarray(block_candidates.candidate_mask)
        block_available = jnp.asarray(block_candidates.available)
        if block_f32.shape != (
            batch,
            ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
            BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE,
        ):
            raise ValueError("block candidate view does not match public capacity")
        if block_mask.shape != block_f32.shape[:2]:
            raise ValueError("block candidate mask shape drift")
        if block_available.shape != (batch,):
            raise ValueError("block candidate availability shape drift")
    block_mask &= block_available[:, None] & observation.valid[:, None]
    block_f32 = jnp.where(
        block_mask[..., None],
        block_f32,
        jnp.float32(0.0),
    )

    if recipe_candidate_encoding is None:
        recipe_embedding = jnp.zeros(
            (
                batch,
                ACTOR_RECIPE_CANDIDATE_CAPACITY,
                ARSENAL_POLICY_RECIPE_CANDIDATE_EMBEDDING_SIZE,
            ),
            dtype=jnp.float32,
        )
        recipe_mask = jnp.zeros(
            (batch, ACTOR_RECIPE_CANDIDATE_CAPACITY),
            dtype=jnp.bool_,
        )
        recipe_available = jnp.zeros((batch,), dtype=jnp.bool_)
    else:
        recipe_embedding = jnp.asarray(recipe_candidate_encoding.candidate_embedding)
        recipe_mask = jnp.asarray(recipe_candidate_encoding.candidate_mask)
        recipe_available = jnp.asarray(recipe_candidate_encoding.available)
        if recipe_embedding.shape != (
            batch,
            ACTOR_RECIPE_CANDIDATE_CAPACITY,
            ARSENAL_POLICY_RECIPE_CANDIDATE_EMBEDDING_SIZE,
        ):
            raise ValueError("recipe candidate embedding shape drift")
        if recipe_mask.shape != recipe_embedding.shape[:2]:
            raise ValueError("recipe candidate mask shape drift")
        if recipe_available.shape != (batch,):
            raise ValueError("recipe candidate availability shape drift")
    recipe_mask &= recipe_available[:, None] & observation.valid[:, None]
    recipe_embedding = jnp.where(
        recipe_mask[..., None],
        recipe_embedding,
        jnp.float32(0.0),
    )

    groups = (
        observation.base.self_f32,
        observation.base.target_f32,
        observation.base.target_mask[:, None],
        jnp.take(
            observation.base.combat_f32,
            jnp.asarray(
                _ARSENAL_POLICY_COMBAT_FLOAT_INDICES,
                dtype=jnp.int32,
            ),
            axis=1,
        ),
        observation.resource_f32[:, AGENT_ENTITY] * resource_mask.astype(jnp.float32),
        resource_mask,
        observation.defense_f32[:, AGENT_ENTITY],
        (
            observation.status_f32[:, AGENT_ENTITY]
            * status_mask[..., None].astype(jnp.float32)
        ).reshape((batch, -1)),
        status_mask,
        (
            observation.ability_f32[:, AGENT_ENTITY]
            * ability_mask[..., None].astype(jnp.float32)
        ).reshape((batch, -1)),
        ability_mask,
        observation.ability_legal[:, AGENT_ENTITY],
        observation.actor_world_f32 * actor_world_value_mask.astype(jnp.float32),
        observation.actor_world_mask,
        observation.movement_state_f32 * movement_state_mask.astype(jnp.float32),
        movement_state_mask,
        observation.world_geometry.token_f32.reshape((batch, -1)),
        observation.world_geometry.token_mask,
        observation.world_geometry.available[:, None],
        light_tokens.token_f32.reshape((batch, -1)),
        light_tokens.available[:, None],
        inventory_tokens.container_f32,
        inventory_tokens.container_mask,
        inventory_tokens.token_f32.reshape((batch, -1)),
        inventory_tokens.token_mask,
        inventory_tokens.available[:, None],
        observation.skill_action_mask,
        observation.jump_action_mask[:, None],
        observation.guard_action_mask[:, None],
        observation.dodge_action_mask,
        observation.valid[:, None],
        block_f32.reshape((batch, -1)),
        block_mask,
        block_available[:, None] & observation.valid[:, None],
        recipe_embedding.reshape((batch, -1)),
        recipe_mask,
        recipe_available[:, None] & observation.valid[:, None],
    )
    flattened = jnp.concatenate(
        tuple(jnp.asarray(group, dtype=jnp.float32) for group in groups),
        axis=1,
    )
    expected_size = (
        ARSENAL_POLICY_BASE_OBSERVATION_SIZE
        + observation.world_geometry.token_f32.shape[1]
        * observation.world_geometry.token_f32.shape[2]
        + observation.world_geometry.token_mask.shape[1]
        + 1
        + actor_light_policy_flat_size(geometry_capacity)
        + ARSENAL_POLICY_ACTION_SURFACE_OBSERVATION_SIZE
    )
    if flattened.shape[1] != expected_size:
        raise RuntimeError(
            "arsenal policy observation size drift: "
            f"{flattened.shape[1]} != {expected_size}"
        )
    return flattened


def arsenal_policy_action_mask(
    observation: LearnerCombatObservationV3,
    block_candidates=None,
    recipe_candidates=None,
    *,
    use_available=None,
    block_trigger_available=None,
    interaction_movement=None,
):
    """Return all twelve masks, defaulting unbound World verbs closed."""

    return arsenal_policy_action_context_mask(
        learner_arsenal_action_context(observation),
        block_candidates,
        recipe_candidates,
        use_available=use_available,
        block_trigger_available=block_trigger_available,
        interaction_movement=interaction_movement,
    )


def arsenal_policy_action_context_mask(
    context: LearnerArsenalActionContext,
    block_candidates=None,
    recipe_candidates=None,
    *,
    use_available=None,
    block_trigger_available=None,
    interaction_movement=None,
):
    """Build all twelve masks from the compact environment context."""

    base_mask = jnp.concatenate(
        (
            context.base.action_mask[:, :SKILL_COUNT],
            context.base.action_mask[:, SKILL_COUNT:LEARNER_ACTION_COUNT],
            jnp.ones((context.valid.shape[0], 1), dtype=jnp.bool_),
            context.ability_action_mask,
            jnp.ones((context.valid.shape[0], 1), dtype=jnp.bool_),
            context.guard_action_mask[:, None],
            # Dodge is not a base head any more; its legality gates the dodge
            # branch of the locomotion head instead.
            jnp.ones((context.valid.shape[0], 1), dtype=jnp.bool_),
            context.jump_action_mask[:, None],
        ),
        axis=1,
    )
    if base_mask.shape[1] != sum(ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES):
        raise RuntimeError(
            "arsenal base action size drift: "
            f"{base_mask.shape[1]} != "
            f"{sum(ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES)}"
        )
    # Local imports avoid the policy/action-surface module cycle while keeping
    # the public function the single mask entry point.
    from hytalegym.jax.combat.observation.v3.policy.surface import (
        staged_action_surface_mask,
    )
    from hytalegym.jax.combat.observation.v3.policy.candidates.blocks import (
        empty_block_action_candidate_policy_view,
    )
    from hytalegym.jax.combat.observation.v3.policy.candidates.recipes import (
        empty_recipe_candidate_policy_view,
    )

    batch = context.valid.shape[0]
    blocks = (
        empty_block_action_candidate_policy_view(
            batch,
            ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
        )
        if block_candidates is None
        else block_candidates
    )
    recipes = (
        empty_recipe_candidate_policy_view(batch)
        if recipe_candidates is None
        else recipe_candidates
    )
    use = (
        jnp.zeros((batch,), dtype=jnp.bool_) if use_available is None else use_available
    )
    block_triggers = (
        jnp.zeros((batch, 2), dtype=jnp.bool_)
        if block_trigger_available is None
        else block_trigger_available
    )
    mask = staged_action_surface_mask(
        base_mask,
        context.valid,
        blocks,
        recipes,
        use_available=use,
        block_trigger_available=block_triggers,
        interaction_movement=interaction_movement,
        # Dodge left the base heads but keeps its authored per-direction
        # legality, which now gates the dodge branch of the locomotion head.
        dodge_available=context.dodge_action_mask,
    )
    if mask.shape[1] != ARSENAL_POLICY_ACTION_SIZE:
        raise RuntimeError(
            "arsenal policy action size drift: "
            f"{mask.shape[1]} != {ARSENAL_POLICY_ACTION_SIZE}"
        )
    return mask


def split_locomotion_choice(locomotion):
    """Fan one locomotion choice out into gait, compass and dodge.

    The published head is a single categorical because the gaits and a dodge
    occupy the same physical slot. Everything downstream still consumes the
    separate fields, so the split lives here and is shared by the factor decode
    and the staged action surface rather than being written twice.
    """

    choice = jnp.asarray(locomotion, dtype=jnp.int32)
    compass = jnp.int32(ARSENAL_POLICY_COMPASS_DIRECTIONS)
    dodging = choice >= jnp.int32(ARSENAL_POLICY_LOCOMOTION_DODGE_START)
    stepping = (choice >= jnp.int32(1)) & ~dodging
    ring = choice - jnp.int32(1)
    gait = jnp.where(
        stepping, ring // compass + jnp.int32(1), jnp.int32(ARSENAL_GAIT_IDLE)
    )
    direction = jnp.where(stepping, ring % compass + jnp.int32(1), jnp.int32(0))
    dodge = jnp.where(
        dodging,
        choice - jnp.int32(ARSENAL_POLICY_LOCOMOTION_DODGE_START) + jnp.int32(1),
        jnp.int32(0),
    )
    return gait, direction, dodge


def decode_arsenal_policy_actions(
    action_factors,
    *,
    maximum_turn_degrees: float = 45.0,
) -> LearnerArsenalAction:
    """Decode explicit ``int32[B, 12]`` factors into the structured action."""

    factors = jnp.asarray(action_factors)
    if factors.dtype != jnp.int32:
        raise TypeError("arsenal policy action factors must have dtype int32")
    if factors.ndim != 2 or factors.shape[1] != len(ARSENAL_POLICY_ACTION_HEAD_SIZES):
        raise ValueError(
            "arsenal policy action factors must have shape "
            f"(B, {len(ARSENAL_POLICY_ACTION_HEAD_SIZES)})"
        )
    bounded = jnp.stack(
        tuple(
            jnp.clip(factors[:, index], 0, size - 1)
            for index, size in enumerate(ARSENAL_POLICY_ACTION_HEAD_SIZES)
        ),
        axis=1,
    )
    yaw_delta = decode_look_delta(
        bounded[:, 5], 9, maximum_degrees=maximum_turn_degrees
    )
    body_yaw_delta = decode_look_delta(
        bounded[:, 6], 9, maximum_degrees=maximum_turn_degrees
    )
    pitch_delta = decode_look_delta(
        bounded[:, 7], 5, maximum_degrees=maximum_turn_degrees
    )
    # One locomotion choice fans back out into the fields the rest of the stack
    # already consumes, so consolidating the head does not ripple past here.
    locomotion = bounded[:, 4]
    gait, move_direction, dodge = split_locomotion_choice(locomotion)
    return LearnerArsenalAction(
        skill_id=bounded[:, 0],
        ability_slot=bounded[:, 1] - jnp.int32(1),
        guard_held=bounded[:, 2].astype(jnp.bool_),
        dodge_direction=dodge,
        jump_held=bounded[:, 3].astype(jnp.bool_),
        gait=gait,
        world_move_direction=move_direction,
        yaw_delta_degrees=yaw_delta,
        body_yaw_delta_degrees=body_yaw_delta,
        pitch_delta_degrees=pitch_delta,
        # Choice 0 means "no switch", so the slot is the choice minus one and
        # -1 reads as "leave the live hotbar slot where it is".
        hotbar_slot=bounded[:, 8] - jnp.int32(1),
        use_requested=bounded[:, 9].astype(jnp.bool_),
        block_interaction_trigger=(
            bounded[:, 10] + jnp.int32(ARSENAL_BLOCK_TRIGGER_PRIMARY)
        ),
        # Crafting is not published for a combat agent; see the target heads.
        recipe_candidate_index=jnp.full_like(locomotion, -1),
        block_candidate_index=bounded[:, 11] - jnp.int32(1),
    )


def block_interaction_request_mask(
    block_interaction_trigger,
    block_candidate_index,
):
    """Return where a trigger choice and actor-legal target form a request.

    The two-choice trigger head always preserves Primary versus Secondary.
    The independent none-plus-candidate head owns request presence. This
    keeps all published factors round-trippable without turning an inactive
    trigger choice into a world mutation request.
    """

    trigger = jnp.asarray(block_interaction_trigger)
    candidate = jnp.asarray(block_candidate_index)
    if trigger.shape != candidate.shape:
        raise ValueError("block trigger and candidate index must have matching shapes")
    return (
        (trigger >= jnp.int32(ARSENAL_BLOCK_TRIGGER_PRIMARY))
        & (trigger <= jnp.int32(ARSENAL_BLOCK_TRIGGER_SECONDARY))
        & (candidate >= jnp.int32(0))
    )


def neutral_arsenal_policy_action_factors(batch_size: int):
    """Return the physical no-op for every published action head.

    Most heads use choice zero as their neutral value. Signed yaw and pitch
    deltas instead use their centre bin, so deriving those positions from the
    published names prevents an "idle" caller from silently turning.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    factors = jnp.zeros(
        (batch_size, len(ARSENAL_POLICY_ACTION_HEAD_SIZES)),
        dtype=jnp.int32,
    )
    for name in ("yaw_delta_bins", "body_yaw_delta_bins", "pitch_delta_bins"):
        index = ARSENAL_POLICY_ACTION_HEAD_NAMES.index(name)
        factors = factors.at[:, index].set(
            jnp.int32(ARSENAL_POLICY_ACTION_HEAD_SIZES[index] // 2)
        )
    return factors


def encode_arsenal_policy_actions(
    actions: LearnerArsenalAction,
):
    """Encode a structured action as explicit ``int32[B, 12]`` factors."""

    skill_id = jnp.clip(
        actions.skill_id,
        SKILL_IDLE,
        LEARNER_ACTION_COUNT - 1,
    )
    ability_choice = jnp.clip(
        actions.ability_slot + jnp.int32(1),
        0,
        OBSERVATION_CAPACITY,
    )
    guard = jnp.asarray(actions.guard_held, dtype=jnp.int32)
    jump = jnp.asarray(actions.jump_held, dtype=jnp.int32)
    # Rebuild the single locomotion choice. A dodge wins over ordinary
    # translation because the two are the same physical slot, and a gait
    # without a direction is idle rather than a standing gait.
    dodge = jnp.clip(actions.dodge_direction, 0, DODGE_ACTION_COUNT)
    move_direction = jnp.clip(
        actions.world_move_direction, 0, ARSENAL_POLICY_COMPASS_DIRECTIONS
    )
    gait = jnp.clip(actions.gait, 0, ARSENAL_POLICY_GAIT_COUNT)
    stepping = (gait >= jnp.int32(1)) & (move_direction >= jnp.int32(1))
    locomotion = jnp.where(
        dodge >= jnp.int32(1),
        jnp.int32(ARSENAL_POLICY_LOCOMOTION_DODGE_START) + dodge - jnp.int32(1),
        jnp.where(
            stepping,
            jnp.int32(1)
            + (gait - jnp.int32(1))
            * jnp.int32(ARSENAL_POLICY_COMPASS_DIRECTIONS)
            + (move_direction - jnp.int32(1)),
            jnp.int32(0),
        ),
    )
    block_trigger = jnp.asarray(
        actions.block_interaction_trigger,
        dtype=jnp.int32,
    )
    block_candidate = jnp.asarray(
        actions.block_candidate_index,
        dtype=jnp.int32,
    )
    block_trigger_valid = (block_trigger >= ARSENAL_BLOCK_TRIGGER_PRIMARY) & (
        block_trigger <= ARSENAL_BLOCK_TRIGGER_SECONDARY
    )
    block_request_valid = block_trigger_valid & (
        (block_candidate >= 0)
        & (block_candidate < ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY)
    )
    return jnp.stack(
        (
            skill_id,
            ability_choice,
            guard,
            jump,
            locomotion,
            _signed_delta_to_choice(actions.yaw_delta_degrees, 9),
            _signed_delta_to_choice(actions.body_yaw_delta_degrees, 9),
            _signed_delta_to_choice(actions.pitch_delta_degrees, 5),
            jnp.clip(
                jnp.asarray(actions.hotbar_slot, dtype=jnp.int32) + jnp.int32(1),
                0,
                HOTBAR_CAPACITY,
            ),
            jnp.asarray(actions.use_requested, dtype=jnp.int32),
            jnp.clip(
                block_trigger - jnp.int32(ARSENAL_BLOCK_TRIGGER_PRIMARY),
                0,
                1,
            ),
            jnp.where(
                block_request_valid,
                block_candidate + jnp.int32(1),
                jnp.int32(0),
            ),
        ),
        axis=1,
    ).astype(jnp.int32)


def arsenal_policy_contract_manifest(
    world_geometry_config: WorldGeometryPolicyConfig | None = None,
    *,
    entity_count: int = ENTITY_COUNT,
) -> dict[str, Any]:
    """Return the complete dense actor adapter identity."""

    token_config = normalize_world_geometry_policy_config(world_geometry_config)
    return {
        "schema": ARSENAL_POLICY_SCHEMA,
        "version": ARSENAL_POLICY_VERSION,
        "learner_observation_v3_sha256": (
            learner_observation_v3_contract_sha256(
                entity_count=entity_count,
            )
        ),
        "arsenal_sha256": combat_arsenal_contract_sha256(
            entity_count=entity_count,
        ),
        "world_geometry_token_contract_sha256": (
            world_geometry_token_contract_sha256()
        ),
        "actor_inventory_token_contract_sha256": (
            actor_inventory_token_contract_sha256()
        ),
        "actor_block_light_token_contract_sha256": (
            actor_block_light_token_contract_sha256()
        ),
        "native_inventory_contract_sha256": (native_inventory_contract_sha256()),
        "world_geometry_policy_config": (
            world_geometry_policy_config_manifest(token_config)
        ),
        "inventory_policy_config": {
            "token_capacity": INVENTORY_POLICY_TOKEN_CAPACITY,
            "default_container_slot_capacities": list(DEFAULT_CONTAINER_CAPACITIES),
            "source_capacity": {
                "storage_armor_hotbar_utility_tools": "fixed_native_capacity",
                "backpack": "runtime_nonnegative_int32",
                "occupied_stack_overflow": "clear_complete_actor_row",
            },
            "container_capacity_normalization": {
                "fixed_containers": "capacity_divided_by_native_capacity",
                "backpack": "nonnegative_int32_divided_by_2147483647",
            },
            "token_float_features": list(INVENTORY_POLICY_FLOAT_FEATURES),
            "source_fields": list(INVENTORY_POLICY_SOURCE_FIELDS),
            "flat_size": inventory_policy_flat_size(),
            "actor_axis": "agent_entity_only_before_tokenization",
            "semantic_item_identity": (
                "positive_31_bit_FNV1a_of_exact_native_item_string"
            ),
            "backend_provenance": (
                "excluded_from_policy_to_avoid_native_vs_JAX_identity_leak"
            ),
            "unavailable_or_invalid": "zero_row_fail_closed",
        },
        "observation": {
            "size": arsenal_policy_observation_size(token_config),
            "base_combat_float_features": list(ARSENAL_POLICY_COMBAT_FLOAT_FEATURES),
            "excluded_legacy_combat_float_features": list(
                ARSENAL_POLICY_EXCLUDED_LEGACY_COMBAT_FLOAT_FEATURES
            ),
            "learner_v3_projection_size": (
                ARSENAL_POLICY_BASE_OBSERVATION_SIZE
                - inventory_policy_flat_size()
                + world_geometry_policy_flat_size(token_config)
            ),
            "inventory_policy_append_size": (inventory_policy_flat_size()),
            "light_policy_append_size": actor_light_policy_flat_size(
                token_config.token_capacity
            ),
            "policy_side_candidate_append_size": (
                ARSENAL_POLICY_ACTION_SURFACE_OBSERVATION_SIZE
            ),
            "width_boundary": (
                "inventory, actor-light, and candidate rows are policy-side inputs and "
                "do not change LearnerCombatObservationV3 or its native "
                "actor-evidence handshake"
            ),
            "dtype": "float32",
            "groups": [
                "base.self_f32",
                "base.target_f32",
                "base.target_mask",
                "base.combat_f32[arsenal_policy_features]",
                "agent.resource_f32_masked",
                "agent.resource_mask",
                "agent.defense_f32",
                "agent.status_f32_masked",
                "agent.status_mask",
                "agent.ability_f32_masked",
                "agent.ability_mask",
                "agent.ability_legal",
                "actor_world_f32_masked",
                "actor_world_mask",
                "movement_state_f32_masked",
                "movement_state_mask",
                "world_geometry.token_f32_masked",
                "world_geometry.token_mask",
                "world_geometry.available",
                "actor_light.token_f32_masked",
                "actor_light.available",
                "actor_inventory.container_f32_masked",
                "actor_inventory.container_mask",
                "actor_inventory.token_f32_masked",
                "actor_inventory.token_mask",
                "actor_inventory.available",
                "skill_action_mask",
                "jump_action_mask",
                "guard_action_mask",
                "dodge_action_mask",
                "valid",
                "block_candidates.candidate_f32_masked",
                "block_candidates.candidate_mask",
                "block_candidates.available",
                "recipe_candidates.candidate_embedding_masked",
                "recipe_candidates.candidate_mask",
                "recipe_candidates.available",
            ],
            "excluded": [
                "all combat semantic integer IDs",
                "target resources, statuses, and abilities",
                "legacy target attack phase/progress replaced by ability bank",
                "target inventories",
                "inventory metadata digests and source/backend provenance",
                "light diagnostics, provenance, and environment codes",
                "World capacity, source, tile, and source-index diagnostics",
                "all other privileged diagnostics",
            ],
        },
        "actor_light_policy": {
            "source_contract_sha256": actor_block_light_token_contract_sha256(),
            "token_capacity": token_config.token_capacity,
            "float_features": list(ACTOR_LIGHT_POLICY_FEATURES),
            "normalization": {
                "sky_and_block_light": "uint8_0_to_15_divide_15",
                "tint_rgb": "uint8_0_to_255_divide_255",
            },
            "token_mask": "reuses_exact_world_geometry_token_mask",
            "actor_axis": "agent_entity_only_before_policy_projection",
            "unavailable_or_misaligned": "zero_row_fail_closed",
        },
        "action_surface_observation": {
            "block_candidate_capacity": (
                ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY
            ),
            "block_candidate_feature_size": (
                BLOCK_ACTION_CANDIDATE_POLICY_FEATURE_SIZE
            ),
            "block_candidate_contract_sha256": (
                actor_block_action_candidate_contract_sha256()
            ),
            "recipe_candidate_capacity": ACTOR_RECIPE_CANDIDATE_CAPACITY,
            "recipe_candidate_embedding_size": (
                ARSENAL_POLICY_RECIPE_CANDIDATE_EMBEDDING_SIZE
            ),
            "recipe_candidate_contract_sha256": (
                actor_recipe_candidate_contract_sha256()
            ),
            "recipe_candidate_encoding_contract_sha256": (
                recipe_candidate_encoding_contract_sha256()
            ),
            "masked_values": "bit_exact_zero",
        },
        "action": {
            "logit_size": ARSENAL_POLICY_ACTION_SIZE,
            "head_sizes": list(ARSENAL_POLICY_ACTION_HEAD_SIZES),
            "heads": list(ARSENAL_POLICY_ACTION_HEAD_NAMES),
            "packed_combination_count": ARSENAL_POLICY_COMBINATION_COUNT,
            "transport": "explicit_per_head_int32",
            "factor_order": list(ARSENAL_POLICY_ACTION_HEAD_NAMES),
            "packed_radix_order": None,
            "look_delta_encoding": {
                "strategy": "symmetric_geometric_coarse_to_fine_v1",
                "yaw_degrees": list(look_delta_values(9)),
                "body_yaw_degrees": list(look_delta_values(9)),
                "pitch_degrees": list(look_delta_values(5)),
            },
            "composition": (
                "movement look jump and dodge may execute concurrently; at most one "
                "standard-input root may start per control tick; the shared block "
                "target is routed by use_off_on"
            ),
            "hotbar_factor": {
                "choices": "none_plus_slot_zero_through_eight",
                "neutral": "choice_zero_leaves_the_live_slot_alone",
                "low_level_channel": "slot_plus_one_so_zero_is_no_switch",
            },
            "block_trigger_factor": {
                "choices": ["primary", "secondary"],
                "request_gate": ("block_candidate_index_nonnegative_and_use_off"),
                "inactive_choice": (
                    "preserved_for_exact_factor_round_trip_not_a_request"
                ),
            },
            "use_target": (
                "use_on_plus_nonnegative_block_candidate_routes_that_actor_legal "
                "target to Use without also requesting Primary or Secondary"
            ),
            "block_trigger_values": {
                "none": ARSENAL_BLOCK_TRIGGER_NONE,
                "primary": ARSENAL_BLOCK_TRIGGER_PRIMARY,
                "secondary": ARSENAL_BLOCK_TRIGGER_SECONDARY,
            },
            "block_outcome_resolution": (
                "equipped_item_authored_root_then_selected_target_recheck"
            ),
            "ability_slot_native_mapping": (
                "profile_local_slot_to_asset_authored_root_and_exact_"
                "InteractionType_unbound_suffix_fails_closed"
            ),
        },
    }


def arsenal_policy_contract_sha256(
    world_geometry_config: WorldGeometryPolicyConfig | None = None,
    *,
    entity_count: int = ENTITY_COUNT,
) -> str:
    payload = json.dumps(
        arsenal_policy_contract_manifest(
            world_geometry_config,
            entity_count=entity_count,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _choice_to_signed_delta(choice, size: int):
    return decode_look_delta(choice, size)


def _signed_delta_to_choice(value, size: int):
    return encode_look_delta(value, size)


__all__ = [
    "ARSENAL_ABILITY_ACTION_START",
    "ARSENAL_BLOCK_TRIGGER_NONE",
    "ARSENAL_BLOCK_TRIGGER_PRIMARY",
    "ARSENAL_GAIT_IDLE",
    "ARSENAL_GAIT_WALK",
    "ARSENAL_GAIT_RUN",
    "ARSENAL_GAIT_SPRINT",
    "ARSENAL_GAIT_SNEAK",
    "ARSENAL_POLICY_GAIT_COUNT",
    "ARSENAL_POLICY_COMPASS_DIRECTIONS",
    "ARSENAL_POLICY_LOCOMOTION_DODGE_START",
    "ARSENAL_POLICY_LOCOMOTION_HEAD_SIZE",
    "split_locomotion_choice",
    "ARSENAL_BLOCK_TRIGGER_SECONDARY",
    "ARSENAL_POLICY_ACTION_HEAD_NAMES",
    "ARSENAL_GUARD_ACTION_START",
    "ARSENAL_JUMP_ACTION_START",
    "ARSENAL_POLICY_ACTION_HEAD_SIZES",
    "ARSENAL_POLICY_BASE_ACTION_HEAD_NAMES",
    "ARSENAL_POLICY_BASE_ACTION_HEAD_SIZES",
    "ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY",
    "ARSENAL_POLICY_ACTION_SIZE",
    "ARSENAL_POLICY_BASE_OBSERVATION_SIZE",
    "ARSENAL_POLICY_COMBINATION_COUNT",
    "ARSENAL_POLICY_COMBAT_FLOAT_FEATURES",
    "ARSENAL_POLICY_EXCLUDED_LEGACY_COMBAT_FLOAT_FEATURES",
    "ARSENAL_POLICY_OBSERVATION_SIZE",
    "ARSENAL_POLICY_PLANNED_NON_TARGET_HEADS",
    "ARSENAL_POLICY_PLANNED_NON_TARGET_HEAD_SIZES",
    "ARSENAL_POLICY_SCHEMA",
    "ARSENAL_POLICY_TARGET_RADIX_BUDGET",
    "ARSENAL_POLICY_VERSION",
    "action_target_radix_budget",
    "arsenal_policy_action_context_mask",
    "arsenal_policy_action_mask",
    "arsenal_policy_contract_manifest",
    "arsenal_policy_contract_sha256",
    "arsenal_policy_observation",
    "arsenal_policy_observation_size",
    "action_head_combination_count",
    "block_interaction_request_mask",
    "decode_arsenal_policy_actions",
    "encode_arsenal_policy_actions",
    "neutral_arsenal_policy_action_factors",
    "validate_action_target_head_sizes",
]
