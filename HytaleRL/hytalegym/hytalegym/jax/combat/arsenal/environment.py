"""Framework-neutral structured environment for the Hytale combat arsenal."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal.runtime import (
    combat_target_selection,
    reset_arsenal_batch,
    step_arsenal_batch,
)
from hytalegym.jax.combat.arsenal.compilation import (
    plan_arsenal_compilation,
)
from hytalegym.jax.combat.arsenal.effects.area_plan import (
    entity_only_area_source_support,
    static_area_placement_plan,
)
from hytalegym.jax.combat.arsenal.schema.contract import (
    ABILITY_CAPACITY,
    EF_ANGLED_ANGLE_DEGREES,
    EF_ANGLED_DISTANCE_DEGREES,
    EF_AIR_RESISTANCE,
    EF_AIR_RESISTANCE_MAX,
    EF_FORCE_MAGNITUDE,
    EF_GROUND_RESISTANCE,
    EF_GROUND_RESISTANCE_MAX,
    EF_PROJECTILE_HALF_EXTENT,
    EF_RESISTANCE_THRESHOLD,
    EI_FORCE_MODE,
    EI_RESISTANCE_STYLE,
    EI_TARGET_MODE,
    EVENT_MELEE_CONE,
    EVENT_FLAG_ANGLED_DAMAGE,
    EVENT_PROJECTILE,
    FORCE_SET,
    OBSERVATION_CAPACITY,
    REQUIRE_STATIC_AREA_PLACEMENT,
    TARGET_SELF,
)
from hytalegym.jax.combat.arsenal.effects.events import (
    damage_force_velocity,
    adjust_vertical_force,
    local_force_velocity,
)
from hytalegym.jax.combat.arsenal.programs.event_storage import ability_event_bank
from hytalegym.jax.combat.arsenal.projectiles.runtime import (
    projectile_entity_first_contact,
    projectile_look_direction,
)
from hytalegym.jax.combat.arsenal.factory import empty_arsenal_commands
from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalExplosionCandidates,
    ArsenalEnvironmentState,
    ArsenalInfo,
    ArsenalRuntimeConfig,
    ArsenalWorldCapabilities,
    ForceSweepPlan,
)
from hytalegym.jax.combat.environment import (
    ActionComponentSpec,
    EnvironmentSpec,
)
from hytalegym.jax.combat.inventory import (
    CONTAINER_HOTBAR,
    InventoryLayout,
    InventoryState,
    item_stack_at,
)
from hytalegym.jax.combat.block_interactions import (
    BlockInteractionState,
    apply_interaction_movement_constraints,
    empty_block_interaction_state,
    interaction_movement_constraints,
)
from hytalegym.jax.combat.env import (
    TargetNavigationProvider,
    _motion_delta,
)
from hytalegym.jax.combat.mechanics import (
    APPLIED_RESISTANCE_STYLE_FORCE_VELOCITY,
    CombatMechanicsRules,
    DODGE_EXECUTION_NATIVE_NPC_NULL_CONFIG,
    DODGE_VELOCITY_REMOVAL_SQUARED,
    damp_applied_velocity,
    resolve_dodge_launch_velocity,
)
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.observation.v1.runtime.factory import (
    empty_injected_world_features,
)
from hytalegym.jax.combat.observation.v1.schema.contract import LEARNER_ACTION_COUNT
from hytalegym.jax.combat.observation.v1.schema.types import DoorIntentRequest
from hytalegym.jax.combat.observation.v1.schema.types import InjectedWorldFeatures
from hytalegym.jax.combat.observation.v3.policy.actions import (
    LearnerArsenalActionContext,
    decode_learner_arsenal_action_context,
    learner_arsenal_action_context,
)
from hytalegym.jax.combat.observation.v3.policy.surface import (
    StagedActionSurfaceDecode,
    action_surface_layout,
    decode_staged_action_surface_factors,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.blocks import (
    BlockActionCandidatePolicyView,
    empty_block_action_candidate_policy_view,
    encode_block_action_candidate_policy_view,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.contract import (
    RECIPE_CANDIDATE_EMBEDDING_SIZE,
    recipe_candidate_encoding_parameter_seed,
)
from hytalegym.jax.combat.observation.v3.encoding.encoder import (
    encode_learner_observation_v3,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    ARSENAL_POLICY_LOCOMOTION_HEAD_SIZE,
    arsenal_policy_action_context_mask,
    block_interaction_request_mask,
    encode_arsenal_policy_actions,
)
from hytalegym.jax.combat.observation.v3.tokens.light import (
    ActorLightPolicyTokens,
    empty_actor_light_policy_tokens,
)
from hytalegym.jax.combat.observation.v3.tokens.inventory import (
    InventoryPolicyTokens,
    inventory_policy_tokens_from_state,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.recipes import (
    RecipeCandidatePolicyView,
    empty_recipe_candidate_policy_view,
    encode_recipe_candidate_policy_view,
)
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerArsenalAction,
    LearnerCombatObservationV3,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    WorldGeometryPolicyConfig,
    WorldGeometryPolicyTokens,
    empty_world_geometry_policy_tokens,
    encode_world_geometry_policy_tokens,
    normalize_world_geometry_policy_config,
)
from hytalegym.jax.combat.targeting import (
    default_combat_targeting_rules,
    select_combat_targets,
)
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    ENTITY_COUNT,
    TARGET_ENTITY,
    CombatInfo,
    CombatParams,
    CombatState,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.encoder import (
    RecipeCandidateEncoderParams,
    RecipeCandidateEncoding,
    encode_recipe_candidates,
    initialize_recipe_candidate_encoder,
)
from hytalegym.jax.world import (
    ACTOR_RECIPE_CANDIDATE_CAPACITY,
    ActorBlockActionCandidates,
    ActorRecipeCandidates,
    GeometryProvider,
    GeometryState,
    MutableBlockQueryResult,
    RegionActionRuntimeState,
    RegionGeometryState,
    TraversalTokenObservation,
    WorldGeometryTokenObservation,
    geometry_actor_world_state_result,
    geometry_hitbox_line_of_sight_result,
    geometry_perception_line_of_sight_result,
    geometry_projectile_first_contact_result,
    geometry_swept_volume_clearance_result,
    native_view_sector,
    produce_actor_block_action_candidates,
    produce_actor_world_geometry_tokens,
    produce_region_runtime_actor_block_action_candidates,
)
from hytalegym.jax.world.geometry.point_raycast import (
    geometry_point_ray_first_contact_result,
    native_deployable_surface_allowed,
)
from hytalegym.jax.world.region.actions import (
    NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY,
    NATIVE_CAMERA_BLOCK_RAY_DISTANCE,
    produce_region_runtime_camera_block_action_candidates,
)
from hytalegym.worldgen import (
    INTERACTION_TYPE_NAMES,
    NATIVE_ITEM_BLOCK_CHANGE_CAPACITY,
    NATIVE_ITEM_CHARGE_TIME_CAPACITY,
    NATIVE_ITEM_EDGE_CAPACITY,
    NATIVE_ITEM_INTERACTION_CAPACITY,
    NATIVE_ITEM_INTERACTION_SCHEMA,
    NATIVE_ITEM_INTERACTION_VERSION,
    NATIVE_ITEM_METADATA_CAPACITY,
    NATIVE_ITEM_TRIGGER_CAPACITY,
    NativeItemInteractionEvidence,
    native_item_interaction_contract_sha256,
)

DODGE_CLEARANCE_TICK_CAPACITY = 128
FORCE_CLEARANCE_TICK_CAPACITY = 96
_PRIMARY_TRIGGER_INDEX = INTERACTION_TYPE_NAMES.index("Primary")
_SECONDARY_TRIGGER_INDEX = INTERACTION_TYPE_NAMES.index("Secondary")
_USE_TRIGGER_INDEX = INTERACTION_TYPE_NAMES.index("Use")
NATIVE_ITEM_INTERACTION_RECORDING_SCHEMA = (
    "hytalerl_native_item_interaction_recording_v2"
)


class _EmptyArsenalActionSurfaceEvidence(NamedTuple):
    """Zero-storage view of an unbound action surface."""

    marker: jax.Array

    @property
    def block_candidates(self) -> BlockActionCandidatePolicyView:
        return empty_block_action_candidate_policy_view(
            self.marker.shape[0],
            ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
        )

    @property
    def recipe_candidates(self) -> RecipeCandidatePolicyView:
        return empty_recipe_candidate_policy_view(self.marker.shape[0])

    @property
    def recipe_encoding(self) -> RecipeCandidateEncoding:
        batch = self.marker.shape[0]
        shape = (batch, ACTOR_RECIPE_CANDIDATE_CAPACITY)
        return RecipeCandidateEncoding(
            available=jnp.zeros((batch,), dtype=jnp.bool_),
            candidate_mask=jnp.zeros(shape, dtype=jnp.bool_),
            candidate_embedding=jnp.zeros(
                shape + (RECIPE_CANDIDATE_EMBEDDING_SIZE,),
                dtype=jnp.float32,
            ),
        )

    @property
    def use_available(self) -> jax.Array:
        return jnp.zeros((self.marker.shape[0],), dtype=jnp.bool_)

    @property
    def block_trigger_available(self) -> jax.Array:
        return jnp.zeros((self.marker.shape[0], 2), dtype=jnp.bool_)


class ArsenalEnvironmentCarry(NamedTuple):
    """Compact simulator carry; observations remain explicit step outputs."""

    runtime: ArsenalEnvironmentState
    action_context: LearnerArsenalActionContext
    action_surface: "ArsenalActionSurfaceEvidence | _EmptyArsenalActionSurfaceEvidence"
    action_surface_runtime: "ArsenalActionSurfaceRuntime"
    light_tokens: ActorLightPolicyTokens | None = None
    inventory_tokens: InventoryPolicyTokens | None = None
    world_capabilities: ArsenalWorldCapabilities | None = None
    action_mask: jax.Array | None = None


class ArsenalPPOEnvironmentState(NamedTuple):
    """Compatibility state retained by the test-only PPO adapter."""

    runtime: ArsenalEnvironmentState
    learner_observation: LearnerCombatObservationV3
    action_surface: "ArsenalActionSurfaceEvidence | _EmptyArsenalActionSurfaceEvidence"
    action_surface_runtime: "ArsenalActionSurfaceRuntime"
    light_tokens: ActorLightPolicyTokens | None = None
    inventory_tokens: InventoryPolicyTokens | None = None
    world_capabilities: ArsenalWorldCapabilities | None = None
    action_mask: jax.Array | None = None


class ArsenalActionSurfaceViews(NamedTuple):
    """Actor-safe inputs for the published target and trigger heads."""

    block_candidates: BlockActionCandidatePolicyView
    recipe_candidates: RecipeCandidatePolicyView
    use_available: jax.Array
    block_trigger_available: jax.Array


class ArsenalItemInteractionTable(NamedTuple):
    """Fixed-capacity native root-presence rows keyed by semantic item ID."""

    item_id: jax.Array
    trigger_available: jax.Array
    valid: jax.Array


class ArsenalActorBlockCandidateSource(NamedTuple):
    """World-owned exact rows consumed by the actor-safe candidate producer."""

    geometry: GeometryProvider
    visible_position: jax.Array
    action_position: jax.Array
    targets: MutableBlockQueryResult
    source_complete: jax.Array
    role_opaque_mask: jax.Array
    maximum_distance: jax.Array
    view_sector_full_angle_radians: jax.Array


class ArsenalActionSurfaceEvidence(NamedTuple):
    """Views plus the checkpoint-stable recipe embedding."""

    block_candidates: BlockActionCandidatePolicyView
    recipe_candidates: RecipeCandidatePolicyView
    recipe_encoding: RecipeCandidateEncoding
    use_available: jax.Array
    block_trigger_available: jax.Array


def materialize_arsenal_action_surface_evidence(
    surface: ArsenalActionSurfaceEvidence | _EmptyArsenalActionSurfaceEvidence,
) -> ArsenalActionSurfaceEvidence:
    """Expand an internal zero-storage surface at an actor boundary."""

    if isinstance(surface, ArsenalActionSurfaceEvidence):
        return surface
    if not isinstance(surface, _EmptyArsenalActionSurfaceEvidence):
        raise TypeError("surface must be Arsenal action-surface evidence")
    return ArsenalActionSurfaceEvidence(
        block_candidates=surface.block_candidates,
        recipe_candidates=surface.recipe_candidates,
        recipe_encoding=surface.recipe_encoding,
        use_available=surface.use_available,
        block_trigger_available=surface.block_trigger_available,
    )


class ArsenalActionSurfaceRuntime(NamedTuple):
    """Persistent Combat state plus one caller-owned World PyTree."""

    block_interactions: BlockInteractionState
    world: object


ACTION_SURFACE_VERB_USE = 0
ACTION_SURFACE_VERB_BLOCK_INTERACTION = 1
ACTION_SURFACE_VERB_CRAFT = 2
ACTION_SURFACE_VERB_COUNT = 3

ACTION_SURFACE_REJECT_NONE = jnp.uint32(0)
ACTION_SURFACE_REJECT_EXECUTOR_UNAVAILABLE = jnp.uint32(1)
ACTION_SURFACE_REJECT_EXECUTOR_DENIED = jnp.uint32(2)
ACTION_SURFACE_REJECT_TARGET_RECHECK = jnp.uint32(3)


class ArsenalActionSurfaceLifecycle(NamedTuple):
    """Policy-external lifecycle evidence in stable surface-verb order."""

    requested: jax.Array
    accepted: jax.Array
    started: jax.Array
    finished: jax.Array
    rejected: jax.Array
    cancelled: jax.Array
    reject_reason: jax.Array


class ArsenalWorldRuntimeViews(NamedTuple):
    """All World-derived consumers projected from one persistent PyTree."""

    physical_geometry: GeometryProvider
    capabilities: ArsenalWorldCapabilities
    features: InjectedWorldFeatures
    tokens: WorldGeometryPolicyTokens
    light_tokens: ActorLightPolicyTokens | None = None


ArsenalBlockCandidateProvider = Callable[
    [ArsenalEnvironmentState, object, CombatParams],
    tuple[ActorBlockActionCandidates, jax.Array],
]
ArsenalBlockTriggerAvailabilityProvider = Callable[
    [ArsenalEnvironmentState, object, CombatParams],
    jax.Array,
]
ArsenalItemInteractionAvailabilityProvider = Callable[
    [ArsenalEnvironmentState, object, CombatParams],
    jax.Array,
]
ArsenalActorBlockCandidateSourceProvider = Callable[
    [ArsenalEnvironmentState, object, CombatParams],
    ArsenalActorBlockCandidateSource,
]
ArsenalRecipeCandidateProvider = Callable[
    [ArsenalEnvironmentState, object, CombatParams],
    ActorRecipeCandidates,
]
ArsenalUseAvailabilityProvider = Callable[
    [ArsenalEnvironmentState, object, CombatParams],
    jax.Array,
]
ArsenalActionSurfaceRuntimeInitializer = Callable[
    [jax.Array, ArsenalEnvironmentState, CombatParams, ArsenalRuntimeConfig],
    object,
]
ArsenalWorldRuntimeProvider = Callable[
    [ArsenalEnvironmentState, object, CombatParams],
    ArsenalWorldRuntimeViews,
]
ArsenalRuntimeExplosionCandidateProvider = Callable[
    [
        object,
        CombatState,
        CombatParams,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ],
    ArsenalExplosionCandidates,
]


class ArsenalActionSurfaceExecution(NamedTuple):
    """Executor result gated again by the public action-surface adapter."""

    state: ArsenalEnvironmentState
    runtime: ArsenalActionSurfaceRuntime
    request_legal: jax.Array
    selected_target_rechecked: jax.Array
    state_commit: jax.Array
    lifecycle: ArsenalActionSurfaceLifecycle


class ArsenalEnvironmentInfo(NamedTuple):
    """Step diagnostics kept outside the actor observation."""

    action_valid: jax.Array
    action_surface_legal: jax.Array
    world_verb_legal: jax.Array
    action_surface_lifecycle: ArsenalActionSurfaceLifecycle
    door: DoorIntentRequest
    combat_info: CombatInfo
    arsenal_info: ArsenalInfo
    terminated: jax.Array
    truncated: jax.Array


class ArsenalResetBatch(NamedTuple):
    """Exact actor/target coordinates selected from one reset-key batch."""

    agent_position: jax.Array
    target_position: jax.Array


@dataclass(frozen=True)
class ArsenalEnvironment:
    """Framework-neutral structured Arsenal reset/step interface."""

    spec: EnvironmentSpec
    reset: Callable
    step: Callable
    step_factors: Callable
    step_factors_with_params: Callable


ARSENAL_ENVIRONMENT_SPEC = EnvironmentSpec(
    observation_schema="hytalerl_learner_combat_observation_v3",
    # v5 added body_yaw_delta_bins: the body is steered separately from the
    # camera. v6 adds hotbar_none_plus_slots, so the factor list matches neither
    # of the earlier versions.
    action_schema="hytalerl_learner_arsenal_action_v6",
    action_components=(
        ActionComponentSpec("base_action", "discrete", LEARNER_ACTION_COUNT),
        ActionComponentSpec(
            "ability_none_plus_slots",
            "discrete",
            OBSERVATION_CAPACITY + 1,
        ),
        ActionComponentSpec("guard_held", "binary", 2),
        ActionComponentSpec("jump_held", "binary", 2),
        ActionComponentSpec(
            "locomotion_gait_compass",
            "discrete",
            ARSENAL_POLICY_LOCOMOTION_HEAD_SIZE,
        ),
        ActionComponentSpec("yaw_delta_bins", "discrete", 9),
        ActionComponentSpec("body_yaw_delta_bins", "discrete", 9),
        ActionComponentSpec("pitch_delta_bins", "discrete", 5),
        ActionComponentSpec("hotbar_none_plus_slots", "discrete", 10),
        ActionComponentSpec("use_off_on", "binary", 2),
        ActionComponentSpec(
            "block_primary_secondary_trigger",
            "discrete",
            2,
        ),
        ActionComponentSpec(
            "block_none_plus_candidates",
            "discrete",
            17,
        ),
    ),
)
REFERENCE_ARSENAL_POLICY_ACTION_HEAD_SIZES = ARSENAL_POLICY_ACTION_HEAD_SIZES
REFERENCE_SKILL_MAXIMUM_TURN_DEGREES = 45.0


ArsenalWorldCapabilityProvider = Callable[
    [ArsenalEnvironmentState, CombatParams],
    ArsenalWorldCapabilities,
]
ArsenalWorldTokenProvider = Callable[
    [ArsenalEnvironmentState, CombatParams],
    WorldGeometryPolicyTokens,
]
ArsenalWorldLightTokenProvider = Callable[
    [ArsenalEnvironmentState, CombatParams],
    ActorLightPolicyTokens,
]
ArsenalWorldFeatureProvider = Callable[
    [ArsenalEnvironmentState, CombatParams],
    InjectedWorldFeatures,
]
ArsenalResetProvider = Callable[[jax.Array], ArsenalResetBatch]
ArsenalInventoryResetProvider = Callable[
    [jax.Array, ArsenalRuntimeConfig],
    InventoryState,
]
ArsenalActionSurfaceProvider = Callable[
    [ArsenalEnvironmentState, ArsenalActionSurfaceRuntime, CombatParams],
    ArsenalActionSurfaceViews,
]
ArsenalActionSurfaceExecutor = Callable[
    [
        ArsenalEnvironmentState,
        ArsenalActionSurfaceRuntime,
        StagedActionSurfaceDecode,
        ArsenalActionSurfaceEvidence,
        jax.Array,
        CombatParams,
        ArsenalRuntimeConfig,
    ],
    ArsenalActionSurfaceExecution,
]
ArsenalOpponentAbilityProvider = Callable[
    [
        ArsenalEnvironmentState,
        ArsenalWorldCapabilities,
        ArsenalRuntimeConfig,
    ],
    jax.Array,
]


def item_interaction_table_from_native_evidence(
    evidence: Sequence[NativeItemInteractionEvidence],
    *,
    expected_bridge_sha256: str,
    capacity: int | None = None,
) -> ArsenalItemInteractionTable:
    """Compile native equipped-item roots into a deterministic device table.

    The bridge response is host-side evidence. This function validates its
    bridge identity and exact trigger vocabulary once, deduplicates repeated
    seed captures, and emits only fixed-shape numeric arrays for JAX. Unknown
    or hash-colliding items are rejected instead of being aliased.
    """

    expected_bridge = _required_sha256(
        expected_bridge_sha256,
        "expected_bridge_sha256",
    )
    rows = tuple(evidence)
    resolved: dict[int, tuple[str, tuple[bool, ...]]] = {}
    for row in rows:
        if not isinstance(row, NativeItemInteractionEvidence):
            raise TypeError("evidence must contain NativeItemInteractionEvidence rows")
        if _required_sha256(row.bridge_sha256, "evidence bridge_sha256") != (
            expected_bridge
        ):
            raise ValueError("item-interaction evidence bridge mismatch")
        if (
            len(row.triggers) != NATIVE_ITEM_TRIGGER_CAPACITY
            or tuple(trigger.interaction_type for trigger in row.triggers)
            != tuple(range(NATIVE_ITEM_TRIGGER_CAPACITY))
            or tuple(trigger.interaction_type_name for trigger in row.triggers)
            != INTERACTION_TYPE_NAMES
        ):
            raise ValueError("item-interaction evidence trigger vocabulary changed")
        if not row.equipped_slot_available:
            continue
        asset_ids = {
            trigger.item_asset_id for trigger in row.triggers if trigger.item_asset_id
        }
        if len(asset_ids) != 1:
            raise ValueError("equipped item evidence must identify exactly one asset")
        asset_id = asset_ids.pop()
        item_id = semantic_id(asset_id)
        trigger_mask = tuple(trigger.root is not None for trigger in row.triggers)
        previous = resolved.get(item_id)
        if previous is not None and previous[0] != asset_id:
            raise ValueError("semantic item ID collision")
        if previous is not None and previous[1] != trigger_mask:
            raise ValueError("native trigger availability changed across evidence rows")
        resolved[item_id] = (asset_id, trigger_mask)

    required_capacity = max(1, len(resolved))
    if capacity is None:
        table_capacity = required_capacity
    else:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity < required_capacity:
            raise ValueError("capacity cannot hold all native item-interaction rows")
        table_capacity = capacity

    item_ids = [-1] * table_capacity
    trigger_available = [
        [False] * NATIVE_ITEM_TRIGGER_CAPACITY for _ in range(table_capacity)
    ]
    valid = [False] * table_capacity
    for slot, (item_id, (_asset_id, trigger_mask)) in enumerate(
        sorted(resolved.items(), key=lambda item: item[1][0])
    ):
        item_ids[slot] = item_id
        trigger_available[slot] = list(trigger_mask)
        valid[slot] = True
    return ArsenalItemInteractionTable(
        item_id=jnp.asarray(item_ids, dtype=jnp.int32),
        trigger_available=jnp.asarray(
            trigger_available,
            dtype=jnp.bool_,
        ),
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )


def item_interaction_table_from_native_recording(
    recording: Mapping[str, object],
    *,
    expected_bridge_sha256: str,
    capacity: int | None = None,
) -> ArsenalItemInteractionTable:
    """Compile a fixed/control native recording into a device table.

    Recordings store dataclass manifests rather than raw bridge envelopes.
    Reconstructing the public envelope lets the typed response parser validate
    every graph row before either recorded trigger can reach an actor mask.
    """

    rows = item_interaction_evidence_from_native_recording(
        recording,
        expected_bridge_sha256=expected_bridge_sha256,
    )
    return item_interaction_table_from_native_evidence(
        rows,
        expected_bridge_sha256=expected_bridge_sha256,
        capacity=capacity,
    )


def item_interaction_evidence_from_native_recording(
    recording: Mapping[str, object],
    *,
    expected_bridge_sha256: str,
) -> tuple[NativeItemInteractionEvidence, ...]:
    """Replay and validate the complete fixed/control native item graphs."""

    if not isinstance(recording, Mapping):
        raise TypeError("recording must be an object")
    expected_bridge = _required_sha256(
        expected_bridge_sha256,
        "expected_bridge_sha256",
    )
    if recording.get("schema") != NATIVE_ITEM_INTERACTION_RECORDING_SCHEMA:
        raise ValueError("native item-interaction recording schema changed")
    if (
        _required_sha256(
            recording.get("bridge_sha256"),
            "recording bridge_sha256",
        )
        != expected_bridge
    ):
        raise ValueError("native item-interaction recording bridge mismatch")
    if (
        _required_sha256(
            recording.get("native_contract_sha256"),
            "recording native_contract_sha256",
        )
        != native_item_interaction_contract_sha256().upper()
    ):
        raise ValueError("native item-interaction recording contract changed")
    if (
        recording.get("cross_seed_semantic_match") is not True
        or recording.get("public_player_acceptance_certified") is not False
    ):
        raise ValueError(
            "native item-interaction recording lacks its fixed/control boundary"
        )

    fixed_seed = _recording_integer(recording, "fixed_seed")
    control_seed = _recording_integer(recording, "random_control_seed")
    if fixed_seed == control_seed:
        raise ValueError("native item-interaction control seed must be independent")
    fixed = _recording_mapping(recording, "fixed_seed_capture")
    if _recording_integer(fixed, "seed") != fixed_seed:
        raise ValueError("native item-interaction fixed seed changed")
    rows = tuple(
        _native_item_evidence_from_recording(
            _recording_mapping(fixed, name),
            expected_bridge,
        )
        for name in ("break_item", "place_item")
    )
    semantic_fields = (
        ("break_semantic_sha256", rows[0].semantic_sha256()),
        ("place_semantic_sha256", rows[1].semantic_sha256()),
    )
    control = _recording_mapping(recording, "random_control_semantics")
    if _recording_integer(control, "seed") != control_seed:
        raise ValueError("native item-interaction control seed changed")
    for name, actual in semantic_fields:
        recorded = _required_sha256(fixed.get(name), f"fixed {name}")
        controlled = _required_sha256(control.get(name), f"control {name}")
        if recorded != actual.upper() or controlled != recorded:
            raise ValueError("native item-interaction semantic evidence changed")
    return rows


def _native_item_evidence_from_recording(
    value: Mapping[str, object],
    expected_bridge_sha256: str,
) -> NativeItemInteractionEvidence:
    triggers = []
    for raw_trigger in _recording_sequence(value, "triggers"):
        if not isinstance(raw_trigger, Mapping):
            raise ValueError("triggers must contain objects")
        trigger = dict(raw_trigger)
        nodes = []
        for raw_node in _recording_sequence(trigger, "nodes"):
            if not isinstance(raw_node, Mapping):
                raise ValueError("nodes must contain objects")
            node = dict(raw_node)
            # Native evidence envelopes encode an absent typed payload as
            # ``{"kind": ""}``; dataclass recording manifests serialize
            # that same value as JSON null. Restore the public envelope form
            # before asking the typed parser to validate every node.
            if node.get("payload") is None:
                node["payload"] = {"kind": ""}
            elif isinstance(node.get("payload"), Mapping):
                payload = dict(node["payload"])
                if "kind" not in payload:
                    implementation = str(node.get("implementation_class", "")).rsplit(
                        ".", 1
                    )[-1]
                    kind = {
                        "BreakBlockInteraction": "break_block",
                        "PlaceBlockInteraction": "place_block",
                        "ChangeStateInteraction": "change_block",
                        "ModifyInventoryInteraction": "modify_inventory",
                    }.get(implementation)
                    if kind is None:
                        raise ValueError(
                            "recorded interaction payload has no typed kind"
                        )
                    payload["kind"] = kind
                node["payload"] = payload
            nodes.append(node)
        trigger.update(
            {
                "root_present": trigger.get("root") is not None,
                "node_count": len(nodes),
                "edge_count": len(_recording_sequence(trigger, "edges")),
                "nodes": nodes,
            }
        )
        triggers.append(trigger)
    response = dict(value)
    response.update(
        {
            "type": "item_interaction_evidence",
            # Recording v2 retains the exact evidence schema/version used by
            # the bridge (currently v3). Older recording rows omitted these
            # fields and replay through the compatible v2 defaults.
            "schema": value.get(
                "schema",
                NATIVE_ITEM_INTERACTION_SCHEMA,
            ),
            "version": value.get(
                "version",
                NATIVE_ITEM_INTERACTION_VERSION,
            ),
            "expected_trigger_count": NATIVE_ITEM_TRIGGER_CAPACITY,
            "trigger_count": len(triggers),
            "interaction_capacity_per_trigger": NATIVE_ITEM_INTERACTION_CAPACITY,
            "edge_capacity_per_trigger": NATIVE_ITEM_EDGE_CAPACITY,
            "charge_time_capacity": NATIVE_ITEM_CHARGE_TIME_CAPACITY,
            "block_change_capacity_per_node": (NATIVE_ITEM_BLOCK_CHANGE_CAPACITY),
            "item_metadata_capacity": NATIVE_ITEM_METADATA_CAPACITY,
            "evidence_scope": (
                "resolved_item_roots_rules_effects_and_reachable_authored_chain"
            ),
            "public_player_acceptance_certified": False,
            "triggers": triggers,
        }
    )
    return NativeItemInteractionEvidence.from_response(
        response,
        expected_bridge_sha256=expected_bridge_sha256,
    )


def _recording_mapping(
    value: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key} must be an object")
    return result


def _recording_sequence(
    value: Mapping[str, object],
    key: str,
) -> Sequence[object]:
    result = value.get(key)
    if not isinstance(result, (list, tuple)):
        raise ValueError(f"{key} must be an array")
    return result


def _recording_integer(
    value: Mapping[str, object],
    key: str,
) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"{key} must be an integer")
    return result


def item_interaction_trigger_availability(
    table: ArsenalItemInteractionTable,
    item_id: jax.Array,
) -> jax.Array:
    """Look up exact native root presence; unknown/ambiguous IDs stay closed."""

    table_item_id, trigger_available, table_valid = _validate_item_interaction_table(
        table
    )
    requested = jnp.asarray(item_id)
    if not jnp.issubdtype(requested.dtype, jnp.integer):
        raise TypeError("item_id must have an integer dtype")
    matches = table_valid & (table_item_id == requested[..., None])
    unique = jnp.sum(matches.astype(jnp.int32), axis=-1) == 1
    return (
        jnp.any(
            matches[..., None] & trigger_available,
            axis=-2,
        )
        & unique[..., None]
    )


def make_inventory_item_interaction_availability_provider(
    table: ArsenalItemInteractionTable,
    inventory_layout: InventoryLayout,
    *,
    actor_index: int = AGENT_ENTITY,
) -> ArsenalItemInteractionAvailabilityProvider:
    """Bind active-hotbar item roots to one controlled actor."""

    _validate_item_interaction_table(table)
    if isinstance(actor_index, bool) or not isinstance(actor_index, int):
        raise TypeError("actor_index must be an integer")
    if actor_index < 0:
        raise ValueError("actor_index must be non-negative")

    def provider(
        state: ArsenalEnvironmentState,
        _world: object,
        _params: CombatParams,
    ) -> jax.Array:
        entity_count = state.inventory.item_id.shape[1]
        if actor_index >= entity_count:
            raise ValueError("actor_index is outside the inventory entity axis")
        stack = item_stack_at(
            state.inventory,
            inventory_layout,
            CONTAINER_HOTBAR,
            state.inventory.active_hotbar_slot,
        )
        available = item_interaction_trigger_availability(
            table,
            stack.item_id[:, actor_index],
        )
        return available & (stack.quantity[:, actor_index, None] > 0)

    return provider


def _validate_item_interaction_table(
    table: ArsenalItemInteractionTable,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if not isinstance(table, ArsenalItemInteractionTable):
        raise TypeError("table must be ArsenalItemInteractionTable")
    item_id = jnp.asarray(table.item_id)
    trigger_available = jnp.asarray(table.trigger_available)
    valid = jnp.asarray(table.valid)
    if (
        item_id.ndim != 1
        or item_id.shape[0] < 1
        or item_id.dtype != jnp.dtype(jnp.int32)
    ):
        raise ValueError("table item_id must be int32[K] with K >= 1")
    if trigger_available.dtype != jnp.dtype(jnp.bool_) or trigger_available.shape != (
        item_id.shape[0],
        NATIVE_ITEM_TRIGGER_CAPACITY,
    ):
        raise ValueError("table trigger_available must be bool[K, trigger_capacity]")
    if valid.dtype != jnp.dtype(jnp.bool_) or valid.shape != item_id.shape:
        raise ValueError("table valid must be bool[K]")
    return item_id, trigger_available, valid


def _required_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{name} must be 64 hexadecimal characters")
    return value.upper()


def make_world_actor_block_candidate_provider(
    source_provider: ArsenalActorBlockCandidateSourceProvider,
) -> ArsenalBlockCandidateProvider:
    """Bind World's exact block producer to the controlled actor.

    The source provider owns exact geometry, mutable block queries, canonical
    action cells, source completeness, role opacity, authored reach, and view
    angle. Combat derives only the actor's current position, eye, and facing.
    Every surface refresh therefore re-queries the caller-owned World PyTree
    without allowing privileged block identity into policy code.
    """

    if not callable(source_provider):
        raise TypeError("source_provider must be callable")

    def provider(
        state: ArsenalEnvironmentState,
        world: object,
        params: CombatParams,
    ) -> tuple[ActorBlockActionCandidates, jax.Array]:
        source = source_provider(state, world, params)
        if not isinstance(source, ArsenalActorBlockCandidateSource):
            raise TypeError(
                "source_provider must return ArsenalActorBlockCandidateSource"
            )
        batch = state.combat.health.shape[0]
        actor_position = state.combat.position[
            :,
            AGENT_ENTITY : AGENT_ENTITY + 1,
            :,
        ]
        eye_position = (
            actor_position
            + jnp.asarray(params.agent_eye_offset, dtype=jnp.float32)[
                None,
                None,
                :,
            ]
        )
        actor_forward = projectile_look_direction(
            state.combat.desired_yaw,
            state.combat.desired_pitch,
        )[:, None, :]
        maximum_distance = _single_actor_parameter(
            source.maximum_distance,
            batch,
            "maximum_distance",
        )
        candidates = produce_actor_block_action_candidates(
            source.geometry,
            actor_position,
            eye_position,
            actor_forward,
            source.visible_position,
            source.action_position,
            source.targets,
            source_complete=source.source_complete,
            role_opaque_mask=source.role_opaque_mask,
            candidate_capacity=(ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY),
            maximum_distance=maximum_distance[:, None],
            view_sector_full_angle_radians=(source.view_sector_full_angle_radians),
        )
        return candidates, maximum_distance

    return provider


def make_region_runtime_block_candidate_provider(
    *,
    role_opaque_mask: jax.Array,
    maximum_distance: jax.Array | float,
    view_sector_full_angle_radians: jax.Array | float,
    cell_radius: int,
    max_los_cells: int | None = None,
) -> ArsenalBlockCandidateProvider:
    """Bind World's public mutable Region runtime to the controlled actor.

    Reach and view angle remain caller-owned because they are authored
    interaction semantics, not World constants. The dense stencil radius is
    static; World's completeness predicate closes the row if that stencil
    cannot contain every reachable cell.
    """

    if (
        isinstance(cell_radius, bool)
        or not isinstance(cell_radius, int)
        or cell_radius < 0
    ):
        raise ValueError("cell_radius must be a nonnegative integer")
    if max_los_cells is not None and (
        isinstance(max_los_cells, bool)
        or not isinstance(max_los_cells, int)
        or max_los_cells < 1
    ):
        raise ValueError("max_los_cells must be a positive integer or None")
    opaque = jnp.asarray(role_opaque_mask, dtype=jnp.bool_)
    maximum = jnp.asarray(maximum_distance, dtype=jnp.float32)
    full_angle = jnp.asarray(
        view_sector_full_angle_radians,
        dtype=jnp.float32,
    )

    def provider(
        state: ArsenalEnvironmentState,
        world: object,
        params: CombatParams,
    ) -> tuple[ActorBlockActionCandidates, jax.Array]:
        if not isinstance(world, RegionActionRuntimeState):
            raise TypeError("Region block candidates require RegionActionRuntimeState")
        batch = state.combat.health.shape[0]
        actor_position = state.combat.position[
            :,
            AGENT_ENTITY : AGENT_ENTITY + 1,
            :,
        ]
        eye_position = (
            actor_position
            + jnp.asarray(params.agent_eye_offset, dtype=jnp.float32)[
                None,
                None,
                :,
            ]
        )
        actor_forward = projectile_look_direction(
            state.combat.desired_yaw,
            state.combat.desired_pitch,
        )[:, None, :]
        actor_maximum = _single_actor_parameter(
            maximum,
            batch,
            "maximum_distance",
        )
        actor_full_angle = _single_actor_parameter(
            full_angle,
            batch,
            "view_sector_full_angle_radians",
        )
        atlas_shape = world.geometry.atlas.cell_flags.shape
        if opaque.shape == atlas_shape:
            actor_opaque = jnp.broadcast_to(
                opaque,
                (batch, 1) + atlas_shape,
            )
        elif opaque.shape == (batch, 1) + atlas_shape:
            actor_opaque = opaque
        else:
            raise ValueError(
                "role_opaque_mask must match the Region atlas or the "
                "explicit controlled-actor provider shape"
            )
        candidates = produce_region_runtime_actor_block_action_candidates(
            world,
            actor_position,
            eye_position,
            actor_forward,
            role_opaque_mask=actor_opaque,
            candidate_capacity=(ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY),
            maximum_distance=actor_maximum[:, None],
            view_sector_full_angle_radians=actor_full_angle[:, None],
            cell_radius=cell_radius,
            max_los_cells=max_los_cells,
        )
        return candidates, actor_maximum

    return provider


def make_region_runtime_camera_block_candidate_provider(
    *,
    maximum_interaction_distance: jax.Array | float,
    ray_distance: jax.Array | float = NATIVE_CAMERA_BLOCK_RAY_DISTANCE,
    ray_cell_capacity: int = NATIVE_CAMERA_BLOCK_RAY_CELL_CAPACITY,
) -> ArsenalBlockCandidateProvider:
    """Bind native camera targeting to World's mutable Region runtime.

    Native ``SimpleBlockInteraction`` selects the first non-air block along
    the actor's eye/head ray at eight blocks, then validates the selected cell
    centre against the held interaction's authored reach.  This provider
    preserves those as separate limits and fails the whole actor row closed
    when fixed ray or Region coverage is incomplete.
    """

    if (
        isinstance(ray_cell_capacity, bool)
        or not isinstance(ray_cell_capacity, int)
        or ray_cell_capacity < 1
    ):
        raise ValueError("ray_cell_capacity must be a positive integer")
    maximum = jnp.asarray(maximum_interaction_distance, dtype=jnp.float32)
    ray_limit = jnp.asarray(ray_distance, dtype=jnp.float32)

    def provider(
        state: ArsenalEnvironmentState,
        world: object,
        params: CombatParams,
    ) -> tuple[ActorBlockActionCandidates, jax.Array]:
        if not isinstance(world, RegionActionRuntimeState):
            raise TypeError("Region block candidates require RegionActionRuntimeState")
        batch = state.combat.health.shape[0]
        actor_position = state.combat.position[
            :,
            AGENT_ENTITY : AGENT_ENTITY + 1,
            :,
        ]
        eye_position = (
            actor_position
            + jnp.asarray(params.agent_eye_offset, dtype=jnp.float32)[
                None,
                None,
                :,
            ]
        )
        actor_forward = projectile_look_direction(
            state.combat.desired_yaw,
            state.combat.desired_pitch,
        )[:, None, :]
        actor_maximum = _single_actor_parameter(
            maximum,
            batch,
            "maximum_interaction_distance",
        )
        actor_ray_limit = _single_actor_parameter(
            ray_limit,
            batch,
            "ray_distance",
        )
        candidates = produce_region_runtime_camera_block_action_candidates(
            world,
            actor_position,
            eye_position,
            actor_forward,
            candidate_capacity=(ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY),
            maximum_interaction_distance=actor_maximum[:, None],
            ray_distance=actor_ray_limit[:, None],
            ray_cell_capacity=ray_cell_capacity,
        )
        return candidates, actor_maximum

    return provider


def _single_actor_parameter(
    value: jax.Array,
    batch: int,
    name: str,
) -> jax.Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        return jnp.broadcast_to(result, (batch,))
    if result.shape == (batch,):
        return result
    if result.shape == (batch, 1):
        return result[:, 0]
    raise ValueError(f"{name} must be scalar, [B], or [B, 1] for the controlled actor")


def make_arsenal_action_surface_provider(
    *,
    block_candidate_provider: ArsenalBlockCandidateProvider | None = None,
    recipe_candidate_provider: ArsenalRecipeCandidateProvider | None = None,
    item_interaction_availability_provider: (
        ArsenalItemInteractionAvailabilityProvider | None
    ) = None,
    use_availability_provider: ArsenalUseAvailabilityProvider | None = None,
    block_trigger_availability_provider: (
        ArsenalBlockTriggerAvailabilityProvider | None
    ) = None,
    actor_index: int = AGENT_ENTITY,
) -> ArsenalActionSurfaceProvider:
    """Adapt public actor-legal World rows into the policy action surface.

    Each capability is independently optional and defaults closed.  The block
    provider returns World's typed candidates plus the exact reach used to
    normalize their relative positions.  Recipe identity and block action
    positions remain outside the policy view; an executor must re-query and
    recheck the selected slot before committing it. Providers receive the
    same opaque World PyTree that the executor returns for the next step.
    """

    if isinstance(actor_index, bool) or not isinstance(actor_index, int):
        raise TypeError("actor_index must be an integer")
    if actor_index < 0:
        raise ValueError("actor_index must be non-negative")
    if item_interaction_availability_provider is not None and (
        use_availability_provider is not None
        or block_trigger_availability_provider is not None
    ):
        raise ValueError(
            "the combined item-interaction provider replaces separate "
            "use and block-trigger providers"
        )

    def provider(
        state: ArsenalEnvironmentState,
        runtime: ArsenalActionSurfaceRuntime,
        params: CombatParams,
    ) -> ArsenalActionSurfaceViews:
        batch = state.combat.health.shape[0]
        if block_candidate_provider is None:
            blocks = empty_block_action_candidate_policy_view(
                batch,
                ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
            )
        else:
            block_source, maximum_distance = block_candidate_provider(
                state,
                runtime.world,
                params,
            )
            blocks = encode_block_action_candidate_policy_view(
                block_source,
                maximum_distance,
                actor_index=actor_index,
            )
        if recipe_candidate_provider is None:
            recipes = empty_recipe_candidate_policy_view(batch)
        else:
            recipes = encode_recipe_candidate_policy_view(
                recipe_candidate_provider(state, runtime.world, params),
                actor_index=actor_index,
            )
        if item_interaction_availability_provider is None:
            use_available = (
                jnp.zeros((batch,), dtype=jnp.bool_)
                if use_availability_provider is None
                else jnp.asarray(
                    use_availability_provider(
                        state,
                        runtime.world,
                        params,
                    ),
                )
            )
            block_trigger_available = (
                jnp.zeros((batch, 2), dtype=jnp.bool_)
                if block_trigger_availability_provider is None
                else jnp.asarray(
                    block_trigger_availability_provider(
                        state,
                        runtime.world,
                        params,
                    ),
                )
            )
        else:
            trigger_available = jnp.asarray(
                item_interaction_availability_provider(
                    state,
                    runtime.world,
                    params,
                ),
            )
            if trigger_available.dtype != jnp.dtype(
                jnp.bool_
            ) or trigger_available.shape != (batch, NATIVE_ITEM_TRIGGER_CAPACITY):
                raise ValueError(
                    "item-interaction availability provider must return "
                    "bool[B, trigger_capacity]"
                )
            use_available = trigger_available[:, _USE_TRIGGER_INDEX]
            block_trigger_available = trigger_available[
                :,
                (
                    _PRIMARY_TRIGGER_INDEX,
                    _SECONDARY_TRIGGER_INDEX,
                ),
            ]
        if use_available.dtype != jnp.dtype(jnp.bool_) or use_available.shape != (
            batch,
        ):
            raise ValueError("use availability provider must return bool[B]")
        if block_trigger_available.dtype != jnp.dtype(
            jnp.bool_
        ) or block_trigger_available.shape != (batch, 2):
            raise ValueError(
                "block trigger availability provider must return bool[B, 2]"
            )
        return ArsenalActionSurfaceViews(
            block_candidates=blocks,
            recipe_candidates=recipes,
            use_available=use_available,
            block_trigger_available=block_trigger_available,
        )

    return provider


def make_arsenal_environment(
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    *,
    maximum_turn_degrees: float,
    geometry_provider: GeometryProvider | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
    world_capability_provider: ArsenalWorldCapabilityProvider | None = None,
    world_feature_provider: ArsenalWorldFeatureProvider | None = None,
    world_token_provider: ArsenalWorldTokenProvider | None = None,
    world_light_token_provider: ArsenalWorldLightTokenProvider | None = None,
    world_geometry_config: WorldGeometryPolicyConfig | None = None,
    reset_provider: ArsenalResetProvider | None = None,
    inventory_reset_provider: ArsenalInventoryResetProvider | None = None,
    action_surface_provider: ArsenalActionSurfaceProvider | None = None,
    action_surface_executor: ArsenalActionSurfaceExecutor | None = None,
    action_surface_runtime_initializer: (
        ArsenalActionSurfaceRuntimeInitializer | None
    ) = None,
    world_runtime_provider: ArsenalWorldRuntimeProvider | None = None,
    explosion_candidate_provider: (
        ArsenalRuntimeExplosionCandidateProvider | None
    ) = None,
    recipe_candidate_encoder_params: RecipeCandidateEncoderParams | None = None,
    opponent_ability_provider: ArsenalOpponentAbilityProvider | None = None,
) -> ArsenalEnvironment:
    """Build the structured Arsenal environment.

    ``geometry_provider`` is one fixed local or Region geometry PyTree used by
    reset and every physical microtick. World capability, feature, and token
    providers remain independent actor-evidence producers. ``reset_provider``
    may select exact `(batch, 3)` actor/target coordinates from reset keys; it
    requires physical geometry and leaves the fixed-spawn path unchanged when
    absent. ``action_surface_runtime_initializer`` may inject one arbitrary
    public World PyTree; Combat carries it unchanged except when the supplied
    executor returns an updated value. ``world_runtime_provider`` may project
    physical geometry, capabilities, actor features, and geometry tokens from
    that same value so a committed mutation reaches every World consumer.
    ``inventory_reset_provider`` may supply scenario inventory without
    embedding weapon- or scenario-specific stock quantities in Combat.
    Omitting ``opponent_ability_provider`` selects the deterministic
    first-legal production baseline.  A diagnostic that needs an inert
    opponent must pass ``inert_opponent_ability_slots`` explicitly.  The
    provider may fill only controlled non-actor ability rows; the actor command
    and shared legality checks remain authoritative.
    ``explosion_candidate_provider`` consumes the same persistent World value
    at the exact triggered-impact point; Combat never imports or copies the
    World's physical admission kernel.

    The episode-pinned loadout also specializes unreachable projectile and
    area banks. Direct ``reset_arsenal_batch`` callers retain full capacities.
    """

    if opponent_ability_provider is None:
        # Import lazily to keep the framework-neutral environment below the
        # opponent-policy package in the module dependency graph.
        from hytalegym.jax.combat.opponents.runtime.policy import (
            first_legal_opponent_ability_slots,
        )

        opponent_ability_provider = first_legal_opponent_ability_slots

    expected_batch = config.loadout.weapon_id.shape[0]
    compile_plan = plan_arsenal_compilation(params, config)
    compiled_microticks = compile_plan.microticks
    compiled_has_item_programs = compile_plan.has_item_programs
    compiled_has_projectiles = compile_plan.has_projectiles
    compiled_has_areas = compile_plan.has_areas
    compiled_projectile_capacity = compile_plan.projectile_capacity
    compiled_area_capacity = compile_plan.area_capacity
    compiled_null_loadout = compile_plan.null_loadout
    capability_provider = (
        fail_closed_arsenal_world_capabilities
        if world_capability_provider is None
        else jax.jit(world_capability_provider, inline=False)
    )
    compiled_world_runtime_provider = (
        None
        if world_runtime_provider is None
        else jax.jit(world_runtime_provider, inline=False)
    )
    if world_runtime_provider is not None and any(
        provider is not None
        for provider in (
            world_capability_provider,
            world_feature_provider,
            world_token_provider,
            world_light_token_provider,
        )
    ):
        raise ValueError(
            "world_runtime_provider replaces capability, feature, and token providers"
        )
    if world_runtime_provider is not None and geometry_provider is None:
        raise ValueError(
            "world_runtime_provider requires the exact base "
            "geometry_provider used by reset"
        )
    token_config = normalize_world_geometry_policy_config(world_geometry_config)
    surface_layout = action_surface_layout(
        ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY
    )
    recipe_encoder = (
        _default_recipe_candidate_encoder()
        if recipe_candidate_encoder_params is None
        else recipe_candidate_encoder_params
    )

    def action_surface_evidence(
        state: ArsenalEnvironmentState,
        runtime: ArsenalActionSurfaceRuntime,
        combat_params: CombatParams,
    ) -> ArsenalActionSurfaceEvidence | _EmptyArsenalActionSurfaceEvidence:
        batch = state.combat.health.shape[0]
        if action_surface_provider is None and action_surface_executor is None:
            return _EmptyArsenalActionSurfaceEvidence(
                marker=jnp.empty((batch, 0), dtype=jnp.bool_),
            )
        views = (
            _empty_action_surface_views(batch)
            if action_surface_provider is None
            else action_surface_provider(state, runtime, combat_params)
        )
        _validate_action_surface_views(views, batch)
        return ArsenalActionSurfaceEvidence(
            block_candidates=views.block_candidates,
            recipe_candidates=views.recipe_candidates,
            recipe_encoding=encode_recipe_candidates(
                recipe_encoder,
                views.recipe_candidates,
            ),
            use_available=views.use_available,
            block_trigger_available=views.block_trigger_available,
        )

    def policy_tokens(
        state: ArsenalEnvironmentState,
        combat_params: CombatParams,
    ) -> WorldGeometryPolicyTokens:
        if world_token_provider is None:
            return empty_world_geometry_policy_tokens(
                state.combat.health.shape[0],
                token_config,
            )
        return world_token_provider(state, combat_params)

    def actor_world_features(
        state: ArsenalEnvironmentState,
        combat_params: CombatParams,
    ) -> InjectedWorldFeatures:
        if world_feature_provider is None:
            return empty_injected_world_features(state.combat.health.shape[0])
        return world_feature_provider(state, combat_params)

    def policy_light_tokens(
        state: ArsenalEnvironmentState,
        combat_params: CombatParams,
    ) -> ActorLightPolicyTokens:
        if world_light_token_provider is None:
            return empty_actor_light_policy_tokens(
                state.combat.health.shape[0],
                token_config.token_capacity,
            )
        result = world_light_token_provider(state, combat_params)
        if not isinstance(result, ActorLightPolicyTokens):
            raise TypeError(
                "world_light_token_provider must return ActorLightPolicyTokens"
            )
        return result

    def runtime_world_views(
        state: ArsenalEnvironmentState,
        action_runtime: ArsenalActionSurfaceRuntime,
        combat_params: CombatParams,
    ) -> ArsenalWorldRuntimeViews:
        if world_runtime_provider is None:
            return ArsenalWorldRuntimeViews(
                physical_geometry=geometry_provider,
                capabilities=capability_provider(state, combat_params),
                features=actor_world_features(state, combat_params),
                tokens=policy_tokens(state, combat_params),
                light_tokens=policy_light_tokens(state, combat_params),
            )
        views = compiled_world_runtime_provider(
            state,
            action_runtime.world,
            combat_params,
        )
        if not isinstance(views, ArsenalWorldRuntimeViews):
            raise TypeError(
                "world_runtime_provider must return ArsenalWorldRuntimeViews"
            )
        if not isinstance(
            views.physical_geometry,
            (GeometryState, RegionGeometryState),
        ):
            raise TypeError(
                "runtime physical geometry must be exact GeometryState or "
                "RegionGeometryState"
            )
        if views.light_tokens is None:
            views = views._replace(
                light_tokens=empty_actor_light_policy_tokens(
                    state.combat.health.shape[0],
                    token_config.token_capacity,
                )
            )
        elif not isinstance(views.light_tokens, ActorLightPolicyTokens):
            raise TypeError(
                "runtime light_tokens must be ActorLightPolicyTokens or None"
            )
        return views

    def current_world_physics(
        state: ArsenalEnvironmentState,
        action_runtime: ArsenalActionSurfaceRuntime,
        combat_params: CombatParams,
        cached_capabilities: ArsenalWorldCapabilities | None,
    ):
        """Project only the consumers required before the transition."""

        if world_runtime_provider is None:
            capabilities = (
                capability_provider(state, combat_params)
                if cached_capabilities is None
                else cached_capabilities
            )
            return geometry_provider, capabilities
        if cached_capabilities is not None:
            current_world = action_runtime.world
            if isinstance(current_world, RegionActionRuntimeState):
                return current_world.geometry, cached_capabilities
            if isinstance(current_world, (GeometryState, RegionGeometryState)):
                return current_world, cached_capabilities
        views = runtime_world_views(state, action_runtime, combat_params)
        return views.physical_geometry, views.capabilities

    def compose_environment_state(
        runtime: ArsenalEnvironmentState,
        observation: LearnerCombatObservationV3,
        action_runtime: ArsenalActionSurfaceRuntime,
        world_views: ArsenalWorldRuntimeViews,
        combat_params: CombatParams,
    ) -> ArsenalEnvironmentCarry:
        """Carry every projection derived from one exact runtime snapshot."""

        surface = action_surface_evidence(
            runtime,
            action_runtime,
            combat_params,
        )
        action_context = learner_arsenal_action_context(observation)
        action_mask = arsenal_policy_action_context_mask(
            action_context,
            surface.block_candidates,
            surface.recipe_candidates,
            use_available=surface.use_available,
            block_trigger_available=surface.block_trigger_available,
            interaction_movement=interaction_movement_constraints(
                action_runtime.block_interactions,
                actor_index=AGENT_ENTITY,
            ),
        )
        return ArsenalEnvironmentCarry(
            runtime=runtime,
            action_context=action_context,
            action_surface=surface,
            action_surface_runtime=action_runtime,
            light_tokens=world_views.light_tokens,
            inventory_tokens=inventory_policy_tokens_from_state(
                runtime.inventory,
                config.inventory_layout,
                actor_valid=observation.valid,
            ),
            world_capabilities=world_views.capabilities,
            action_mask=action_mask,
        )

    def reset(keys):
        if keys.shape[0] != expected_batch:
            raise ValueError("PPO environment batch does not match Arsenal loadout")
        reset_batch = None if reset_provider is None else reset_provider(keys)
        initial_inventory = (
            None
            if inventory_reset_provider is None
            else inventory_reset_provider(keys, config)
        )
        runtime, _ = reset_arsenal_batch(
            keys,
            params,
            config,
            geometry_provider,
            agent_position=(
                None if reset_batch is None else reset_batch.agent_position
            ),
            target_position=(
                None if reset_batch is None else reset_batch.target_position
            ),
            initial_inventory=initial_inventory,
            projectile_capacity=compiled_projectile_capacity,
            area_capacity=compiled_area_capacity,
        )
        action_runtime = ArsenalActionSurfaceRuntime(
            block_interactions=empty_block_interaction_state(
                expected_batch,
                entity_count=runtime.combat.health.shape[1],
            ),
            world=(
                ()
                if action_surface_runtime_initializer is None
                else action_surface_runtime_initializer(
                    keys,
                    runtime,
                    params,
                    config,
                )
            ),
        )
        world_views = runtime_world_views(runtime, action_runtime, params)
        observation = encode_learner_observation_v3(
            runtime,
            params,
            world_views.features,
            world_views.capabilities,
            config,
            world_geometry=world_views.tokens,
        )
        return (
            compose_environment_state(
                runtime,
                observation,
                action_runtime,
                world_views,
                params,
            ),
            observation,
        )

    def decode_step_factors(
        state: ArsenalEnvironmentCarry,
        action_factors: jax.Array,
        current_world: ArsenalWorldCapabilities,
    ):
        """Decode one policy action without entering the mechanics graph."""

        surface_mask = state.action_mask
        if surface_mask is None:
            surface_mask = arsenal_policy_action_context_mask(
                state.action_context,
                state.action_surface.block_candidates,
                state.action_surface.recipe_candidates,
                use_available=state.action_surface.use_available,
                block_trigger_available=(state.action_surface.block_trigger_available),
                interaction_movement=interaction_movement_constraints(
                    state.action_surface_runtime.block_interactions,
                    actor_index=AGENT_ENTITY,
                ),
            )
        surface = decode_staged_action_surface_factors(
            action_factors,
            surface_mask,
            layout=surface_layout,
            maximum_turn_degrees=maximum_turn_degrees,
        )
        decoded = decode_learner_arsenal_action_context(
            state.action_context,
            surface.base,
            current_world,
            maximum_turn_degrees=maximum_turn_degrees,
            desired_body_yaw_degrees=state.runtime.combat.desired_body_yaw,
        )
        decoded = decoded._replace(
            low_level_action=apply_interaction_movement_constraints(
                state.action_surface_runtime.block_interactions,
                decoded.low_level_action,
                actor_index=AGENT_ENTITY,
            )
        )
        if opponent_ability_provider is not None:
            opponent_slots = jnp.asarray(
                opponent_ability_provider(
                    state.runtime,
                    current_world,
                    config,
                )
            )
            if (
                opponent_slots.shape != decoded.commands.ability_slot.shape
                or opponent_slots.dtype != jnp.dtype(jnp.int32)
            ):
                raise ValueError(
                    "opponent ability provider must return int32[batch, entity]"
                )
            decoded = decoded._replace(
                commands=decoded.commands._replace(
                    ability_slot=jnp.where(
                        config.opponent_controller_mask,
                        opponent_slots,
                        decoded.commands.ability_slot,
                    )
                )
            )
        return surface, decoded

    compiled_decode_step_factors = jax.jit(
        decode_step_factors,
        inline=False,
    )

    def fixed_runtime_step(
        runtime: ArsenalEnvironmentState,
        action_surface_world,
        decoded,
        keys: jax.Array,
        current_geometry,
    ):
        """Run fixed-parameter mechanics behind a stable XLA call boundary."""

        runtime_explosion_provider = None
        if explosion_candidate_provider is not None:

            def runtime_explosion_provider(
                combat_state,
                impact,
                block_damage_radius,
                entity_damage_radius,
                damage_blocks,
                query_mask,
                candidate_mask,
            ):
                return explosion_candidate_provider(
                    action_surface_world,
                    combat_state,
                    params,
                    impact,
                    block_damage_radius,
                    entity_damage_radius,
                    damage_blocks,
                    query_mask,
                    candidate_mask,
                )

        return step_arsenal_batch(
            runtime,
            decoded.low_level_action,
            keys,
            params,
            decoded.commands,
            config,
            current_geometry,
            target_navigation_provider,
            runtime_explosion_provider,
            None,
            compiled_microticks,
            compiled_null_loadout,
            compiled_has_item_programs,
            compiled_has_projectiles,
            compiled_has_areas,
        )

    compiled_fixed_runtime_step = jax.jit(
        fixed_runtime_step,
        inline=False,
    )

    def step_factors_with_params(
        state: ArsenalEnvironmentCarry,
        action_factors: jax.Array,
        keys: jax.Array,
        step_params: CombatParams,
        motion_delta_seconds: jax.Array | None = None,
    ):
        current_geometry, current_world = current_world_physics(
            state.runtime,
            state.action_surface_runtime,
            step_params,
            state.world_capabilities if step_params is params else None,
        )
        fixed_compilation_boundary = (
            step_params is params and motion_delta_seconds is None
        )
        decode = (
            compiled_decode_step_factors
            if fixed_compilation_boundary
            else decode_step_factors
        )
        surface, decoded = decode(
            state,
            action_factors,
            current_world,
        )
        if fixed_compilation_boundary:
            transition = compiled_fixed_runtime_step(
                state.runtime,
                state.action_surface_runtime.world,
                decoded,
                keys,
                current_geometry,
            )
        else:
            runtime_explosion_provider = None
            if explosion_candidate_provider is not None:

                def runtime_explosion_provider(
                    combat_state,
                    impact,
                    block_damage_radius,
                    entity_damage_radius,
                    damage_blocks,
                    query_mask,
                    candidate_mask,
                ):
                    return explosion_candidate_provider(
                        state.action_surface_runtime.world,
                        combat_state,
                        step_params,
                        impact,
                        block_damage_radius,
                        entity_damage_radius,
                        damage_blocks,
                        query_mask,
                        candidate_mask,
                    )

            transition = step_arsenal_batch(
                state.runtime,
                decoded.low_level_action,
                keys,
                step_params,
                decoded.commands,
                config,
                current_geometry,
                target_navigation_provider,
                runtime_explosion_provider,
                motion_delta_seconds,
                compiled_microticks if step_params is params else None,
                compiled_null_loadout if step_params is params else None,
                compiled_has_item_programs if step_params is params else None,
                compiled_has_projectiles if step_params is params else None,
                compiled_has_areas if step_params is params else None,
            )
        verb_requests = action_surface_verb_requests(surface)
        verb_requested = jnp.any(verb_requests, axis=1)
        if action_surface_executor is None:
            executed_state = transition.state
            verb_legal = ~verb_requested
            next_surface_runtime = state.action_surface_runtime
            lifecycle = action_surface_lifecycle_evidence(
                surface,
                reject_reason=jnp.full(
                    verb_requests.shape,
                    ACTION_SURFACE_REJECT_EXECUTOR_UNAVAILABLE,
                    dtype=jnp.uint32,
                ),
            )
        else:
            execution = action_surface_executor(
                transition.state,
                state.action_surface_runtime,
                surface,
                state.action_surface,
                keys,
                step_params,
                config,
            )
            _validate_action_surface_execution(
                execution,
                verb_requested.shape[0],
            )
            target_requested = (
                (
                    block_interaction_request_mask(
                        surface.block_interaction_trigger,
                        surface.block_candidate_index,
                    )
                    & ~surface.use_requested
                )
                | (surface.use_requested & (surface.block_candidate_index >= 0))
                | (surface.recipe_candidate_index >= 0)
            )
            recheck_satisfied = ~target_requested | execution.selected_target_rechecked
            lifecycle_accepted = jnp.all(
                ~verb_requests | execution.lifecycle.accepted,
                axis=1,
            )
            executor_legal = (
                execution.request_legal & recheck_satisfied & lifecycle_accepted
            )
            lifecycle = _canonicalize_action_surface_lifecycle(
                execution.lifecycle,
                verb_requests,
                executor_legal,
                recheck_satisfied,
            )
            commit = (
                execution.state_commit
                & surface.action_legal
                & (~verb_requested | executor_legal)
            )
            executed_state = _batch_select_state(
                commit,
                execution.state,
                transition.state,
            )
            next_surface_runtime = _batch_select_state(
                commit,
                execution.runtime,
                state.action_surface_runtime,
            )
            verb_legal = ~verb_requested | executor_legal
        next_world_views = runtime_world_views(
            executed_state,
            next_surface_runtime,
            step_params,
        )
        next_observation = encode_learner_observation_v3(
            executed_state,
            step_params,
            next_world_views.features,
            next_world_views.capabilities,
            config,
            world_geometry=next_world_views.tokens,
        )
        done = transition.terminated | transition.truncated
        return (
            compose_environment_state(
                executed_state,
                next_observation,
                next_surface_runtime,
                next_world_views,
                step_params,
            ),
            next_observation,
            transition.reward,
            done,
            ArsenalEnvironmentInfo(
                action_valid=(decoded.valid & surface.action_legal & verb_legal),
                action_surface_legal=surface.action_legal,
                world_verb_legal=verb_legal,
                action_surface_lifecycle=lifecycle,
                door=decoded.door,
                combat_info=transition.combat_info,
                arsenal_info=transition.arsenal_info,
                terminated=transition.terminated,
                truncated=transition.truncated,
            ),
        )

    def step_factors(
        state: ArsenalEnvironmentCarry,
        action_factors: jax.Array,
        keys: jax.Array,
    ):
        return step_factors_with_params(
            state,
            action_factors,
            keys,
            params,
        )

    # Preserve the framework-neutral environment transition as one reusable
    # device computation when a learner embeds it in an outer scan. Exact World
    # projection plus Combat otherwise forms a single oversized CUDA compiler
    # graph at otherwise-valid batch widths.
    compiled_step_factors = jax.jit(step_factors, inline=False)

    def step(
        state: ArsenalEnvironmentCarry,
        action: LearnerArsenalAction,
        keys: jax.Array,
    ):
        return compiled_step_factors(
            state,
            encode_arsenal_policy_actions(action),
            keys,
        )

    compiled_reset = jax.jit(reset, inline=False)
    return ArsenalEnvironment(
        spec=ARSENAL_ENVIRONMENT_SPEC,
        reset=compiled_reset,
        step=step,
        step_factors=compiled_step_factors,
        step_factors_with_params=step_factors_with_params,
    )


def _empty_action_surface_views(batch: int) -> ArsenalActionSurfaceViews:
    return ArsenalActionSurfaceViews(
        block_candidates=empty_block_action_candidate_policy_view(
            batch,
            ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
        ),
        recipe_candidates=empty_recipe_candidate_policy_view(batch),
        use_available=jnp.zeros((batch,), dtype=jnp.bool_),
        block_trigger_available=jnp.zeros(
            (batch, 2),
            dtype=jnp.bool_,
        ),
    )


def _validate_action_surface_views(
    views: ArsenalActionSurfaceViews,
    batch: int,
) -> None:
    if not isinstance(views, ArsenalActionSurfaceViews):
        raise TypeError("action_surface_provider must return ArsenalActionSurfaceViews")
    if views.block_candidates.available.shape != (batch,):
        raise ValueError("block candidate view batch mismatch")
    if views.block_candidates.candidate_mask.shape != (
        batch,
        ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
    ):
        raise ValueError("block candidate capacity does not match policy")
    if views.recipe_candidates.available.shape != (batch,):
        raise ValueError("recipe candidate view batch mismatch")
    use = jnp.asarray(views.use_available)
    if use.dtype != jnp.bool_ or use.shape != (batch,):
        raise ValueError("use_available must be boolean with shape [B]")
    block_triggers = jnp.asarray(views.block_trigger_available)
    if block_triggers.dtype != jnp.bool_ or block_triggers.shape != (batch, 2):
        raise ValueError("block_trigger_available must be boolean with shape [B, 2]")


def _validate_action_surface_execution(
    execution: ArsenalActionSurfaceExecution,
    batch: int,
) -> None:
    if not isinstance(execution, ArsenalActionSurfaceExecution):
        raise TypeError(
            "action_surface_executor must return ArsenalActionSurfaceExecution"
        )
    for name in (
        "request_legal",
        "selected_target_rechecked",
        "state_commit",
    ):
        value = jnp.asarray(getattr(execution, name))
        if value.dtype != jnp.bool_ or value.shape != (batch,):
            raise ValueError(f"{name} must be boolean with shape [B]")
    lifecycle = execution.lifecycle
    if not isinstance(lifecycle, ArsenalActionSurfaceLifecycle):
        raise TypeError("execution.lifecycle must be ArsenalActionSurfaceLifecycle")
    shape = (batch, ACTION_SURFACE_VERB_COUNT)
    for name in (
        "requested",
        "accepted",
        "started",
        "finished",
        "rejected",
        "cancelled",
    ):
        value = jnp.asarray(getattr(lifecycle, name))
        if value.dtype != jnp.bool_ or value.shape != shape:
            raise ValueError(f"lifecycle.{name} must be boolean with shape {shape}")
    reason = jnp.asarray(lifecycle.reject_reason)
    if reason.dtype != jnp.uint32 or reason.shape != shape:
        raise ValueError(f"lifecycle.reject_reason must be uint32 with shape {shape}")


def action_surface_verb_requests(
    surface: StagedActionSurfaceDecode,
) -> jax.Array:
    """Return requests in stable Use, block-interaction, craft order."""

    if not isinstance(surface, StagedActionSurfaceDecode):
        raise TypeError("surface must be StagedActionSurfaceDecode")
    return jnp.stack(
        (
            surface.use_requested,
            (
                block_interaction_request_mask(
                    surface.block_interaction_trigger,
                    surface.block_candidate_index,
                )
                & ~surface.use_requested
            ),
            surface.recipe_candidate_index >= 0,
        ),
        axis=1,
    )


def action_surface_lifecycle_evidence(
    surface: StagedActionSurfaceDecode,
    *,
    accepted: jax.Array | None = None,
    started: jax.Array | None = None,
    finished: jax.Array | None = None,
    cancelled: jax.Array | None = None,
    reject_reason: jax.Array | None = None,
) -> ArsenalActionSurfaceLifecycle:
    """Build complete per-verb evidence without a permissive default.

    Every current request that is not explicitly accepted is rejected.
    Completion and cancellation may be emitted on a later tick without a new
    request, which is required by multi-tick block and crafting chains.
    """

    requested = action_surface_verb_requests(surface)
    shape = requested.shape
    accepted_value = (
        _surface_lifecycle_bool(
            accepted,
            shape,
            "accepted",
        )
        & requested
    )
    started_value = _surface_lifecycle_bool(
        started,
        shape,
        "started",
    )
    finished_value = _surface_lifecycle_bool(
        finished,
        shape,
        "finished",
    )
    cancelled_value = _surface_lifecycle_bool(
        cancelled,
        shape,
        "cancelled",
    )
    rejected = requested & ~accepted_value
    reason = (
        jnp.full(
            shape,
            ACTION_SURFACE_REJECT_EXECUTOR_DENIED,
            dtype=jnp.uint32,
        )
        if reject_reason is None
        else jnp.asarray(reject_reason)
    )
    if reason.dtype != jnp.uint32 or reason.shape != shape:
        raise ValueError(f"reject_reason must be uint32 with shape {shape}")
    return ArsenalActionSurfaceLifecycle(
        requested=requested,
        accepted=accepted_value,
        started=started_value,
        finished=finished_value,
        rejected=rejected,
        cancelled=cancelled_value,
        reject_reason=jnp.where(
            rejected,
            jnp.where(
                reason == ACTION_SURFACE_REJECT_NONE,
                ACTION_SURFACE_REJECT_EXECUTOR_DENIED,
                reason,
            ),
            ACTION_SURFACE_REJECT_NONE,
        ),
    )


def _canonicalize_action_surface_lifecycle(
    lifecycle: ArsenalActionSurfaceLifecycle,
    requested: jax.Array,
    executor_legal: jax.Array,
    recheck_satisfied: jax.Array,
) -> ArsenalActionSurfaceLifecycle:
    accepted = lifecycle.accepted & requested & executor_legal[:, None]
    rejected = requested & ~accepted
    default_reason = jnp.where(
        recheck_satisfied[:, None],
        ACTION_SURFACE_REJECT_EXECUTOR_DENIED,
        ACTION_SURFACE_REJECT_TARGET_RECHECK,
    )
    supplied_reason = jnp.where(
        lifecycle.reject_reason == ACTION_SURFACE_REJECT_NONE,
        default_reason,
        lifecycle.reject_reason,
    )
    return ArsenalActionSurfaceLifecycle(
        requested=requested,
        accepted=accepted,
        started=(lifecycle.started & (~requested | accepted)),
        finished=lifecycle.finished,
        rejected=rejected,
        cancelled=lifecycle.cancelled,
        reject_reason=jnp.where(
            rejected,
            supplied_reason,
            ACTION_SURFACE_REJECT_NONE,
        ),
    )


def _surface_lifecycle_bool(
    value: jax.Array | None,
    shape: tuple[int, int],
    name: str,
) -> jax.Array:
    result = jnp.zeros(shape, dtype=jnp.bool_) if value is None else jnp.asarray(value)
    if result.dtype != jnp.bool_ or result.shape != shape:
        raise ValueError(f"{name} must be boolean with shape {shape}")
    return result


def _default_recipe_candidate_encoder() -> RecipeCandidateEncoderParams:
    return initialize_recipe_candidate_encoder(
        jax.random.key(recipe_candidate_encoding_parameter_seed())
    )


def _batch_select_state(mask, when_true, when_false):
    """Commit a PyTree result only on selected, revalidated batch rows."""

    selected = jnp.asarray(mask, dtype=jnp.bool_)

    def choose(true_value, false_value):
        true_array = jnp.asarray(true_value)
        false_array = jnp.asarray(false_value)
        if true_array.shape != false_array.shape:
            raise ValueError("executor PyTree leaf shape mismatch")
        if true_array.ndim == 0 or true_array.shape[0] != selected.shape[0]:
            if true_value is false_value:
                return false_value
            raise ValueError(
                "non-batch executor PyTree leaves must be passed through unchanged"
            )
        expanded = selected.reshape((selected.shape[0],) + (1,) * (true_array.ndim - 1))
        return jnp.where(expanded, true_array, false_array)

    return jax.tree.map(choose, when_true, when_false)


def fail_closed_arsenal_world_capabilities(
    state: ArsenalEnvironmentState,
    params: CombatParams,
) -> ArsenalWorldCapabilities:
    """Return state transforms while every unproduced world query stays false."""

    combat = state.combat
    entity_shape = combat.health.shape
    entity_count = entity_shape[1]
    eye_offsets = (
        jnp.broadcast_to(
            params.target_eye_offset[None, :],
            (entity_count, 3),
        )
        .at[AGENT_ENTITY]
        .set(params.agent_eye_offset)
    )
    muzzle_position = combat.position + eye_offsets[None, :, :]
    # A projectile leaves along the shooter's LOOK direction, so both muzzles
    # read a head, not a chest. The target has always done this; before the
    # agent had a head of its own its slot fell back to the body orientation,
    # which is the same value only while nothing slows the head's rotation.
    muzzle_pitch = jnp.zeros(entity_shape, dtype=jnp.float32)
    muzzle_pitch = muzzle_pitch.at[:, AGENT_ENTITY].set(combat.agent_head_pitch)
    muzzle_pitch = muzzle_pitch.at[:, TARGET_ENTITY].set(combat.target_head_pitch)
    muzzle_yaw = combat.yaw.at[:, AGENT_ENTITY].set(combat.agent_head_yaw)
    muzzle_yaw = muzzle_yaw.at[:, TARGET_ENTITY].set(combat.target_head_yaw)
    opposite = jnp.zeros((entity_count,), dtype=jnp.int32)
    opposite = opposite.at[AGENT_ENTITY].set(jnp.int32(TARGET_ENTITY))
    finite_muzzle = (
        jnp.all(jnp.isfinite(muzzle_position), axis=2)
        & jnp.isfinite(muzzle_yaw)
        & jnp.isfinite(muzzle_pitch)
    )
    return empty_arsenal_commands(
        entity_shape[0],
        entity_count=entity_count,
    ).world._replace(
        muzzle_position=muzzle_position,
        muzzle_yaw_degrees=muzzle_yaw,
        muzzle_pitch_degrees=muzzle_pitch,
        muzzle_valid=finite_muzzle,
        area_center=jnp.take(combat.position, opposite, axis=1),
    )


def open_flat_arsenal_world_capabilities(
    state: ArsenalEnvironmentState,
    params: CombatParams,
    *,
    config: ArsenalRuntimeConfig | None = None,
) -> ArsenalWorldCapabilities:
    """Explicit permissive control fixture; never the production default."""

    world = fail_closed_arsenal_world_capabilities(state, params)
    entity_shape = state.combat.health.shape
    clear = jnp.ones(entity_shape, dtype=jnp.bool_)
    if entity_shape[1] == ENTITY_COUNT:
        area_center = world.area_center
    else:
        targeting = (
            combat_target_selection(state.combat, params, config)
            if config is not None
            else select_combat_targets(
                state.combat.position,
                state.combat.health,
                default_combat_targeting_rules(
                    entity_shape[0],
                    entity_shape[1],
                    sensor_range=params.sensor_range,
                ),
                candidate_capacity=1,
            )
        )
        area_center = jnp.where(
            targeting.engagement_target_mask[..., None],
            _gather_entity(
                state.combat.position,
                targeting.engagement_target_id,
            ),
            state.combat.position,
        )
    return world._replace(
        actor_world_state_available=jnp.ones((entity_shape[0],), dtype=jnp.bool_),
        actor_controller_medium_available=jnp.ones(
            (entity_shape[0],),
            dtype=jnp.bool_,
        ),
        actor_submersion_available=jnp.ones((entity_shape[0],), dtype=jnp.bool_),
        actor_drop_available=jnp.ones((entity_shape[0],), dtype=jnp.bool_),
        actor_drop_support_found=jnp.ones((entity_shape[0],), dtype=jnp.bool_),
        target_candidate_perceptible=jnp.ones(
            (entity_shape[0], entity_shape[1], entity_shape[1]),
            dtype=jnp.bool_,
        ),
        target_candidate_perception_valid=jnp.ones(
            (entity_shape[0], entity_shape[1], entity_shape[1]),
            dtype=jnp.bool_,
        ),
        line_of_sight=clear,
        line_of_sight_valid=clear,
        selector_line_of_sight=clear,
        selector_line_of_sight_valid=clear,
        direct_target_selected=clear,
        direct_target_selection_valid=clear,
        clear_projectile_flight=clear,
        clear_force_path=jnp.ones(
            entity_shape + (ABILITY_CAPACITY,),
            dtype=jnp.bool_,
        ),
        applied_force_collision_available=jnp.ones(
            entity_shape + (ABILITY_CAPACITY,),
            dtype=jnp.bool_,
        ),
        dodge_corridor_clear=jnp.ones(
            entity_shape + (4,),
            dtype=jnp.bool_,
        ),
        entity_only_area=clear,
        static_area_placement=clear,
        area_center=area_center,
    )


def geometry_arsenal_world_capabilities(
    state: ArsenalEnvironmentState,
    params: CombatParams,
    *,
    geometry: GeometryProvider,
    config: ArsenalRuntimeConfig,
    role_opaque_mask: jax.Array | None = None,
    view_sector_full_angle_radians: jax.Array | float | None = None,
) -> ArsenalWorldCapabilities:
    """Bind candidate perception, projectile contact, force, and dodge.

    ``role_opaque_mask=None`` intentionally requests base block opacity only.
    Production callers that need native role-specific opacity must supply the
    world-owned mask instead of treating the base-opacity result as parity.
    Multi-entity callers must supply one native full-width horizontal view
    angle per source. Missing view evidence fails every candidate closed.
    Candidate FOV and perception LOS are both evaluated before distance
    ranking; the certified two-entity path stays unchanged.

    Entity-only area support is derived from independent per-event asset
    flags; a future unflagged area program keeps its entire source closed.
    Force clearance is ability-specific. Only direct force
    programs whose complete authored lifetime fits the static integration
    capacity are certified. Projectile flight is admitted only when the
    full first-contact query certifies no solid contact before the locked
    target; every admitted projectile retains dynamic contact handling.
    Projectile-origin force paths remain false.
    """

    world = fail_closed_arsenal_world_capabilities(state, params)
    combat = state.combat
    batch, entity_count = combat.health.shape
    agent_offset = _broadcast_los_offset(
        geometry.agent_los_offset,
        batch,
        "agent_los_offset",
    )
    target_offset = _broadcast_los_offset(
        geometry.target_los_offset,
        batch,
        "target_los_offset",
    )
    source_offset = (
        jnp.broadcast_to(
            target_offset[:, None, :],
            combat.position.shape,
        )
        .at[:, AGENT_ENTITY]
        .set(agent_offset)
    )
    source_eye_position = combat.position + source_offset
    if entity_count == ENTITY_COUNT:
        targeting = combat_target_selection(combat, params, config)
        candidate_perception = None
        candidate_in_view = None
    else:
        view_angle = (
            jnp.full(
                (batch, entity_count),
                jnp.nan,
                dtype=jnp.float32,
            )
            if view_sector_full_angle_radians is None
            else _broadcast_view_angle(
                view_sector_full_angle_radians,
                batch,
                entity_count,
            )
        )
        head_yaw = combat.yaw.at[:, TARGET_ENTITY].set(combat.target_head_yaw)
        head_yaw_radians = jnp.deg2rad(head_yaw)
        forward_xz = jnp.stack(
            (
                -jnp.sin(head_yaw_radians),
                -jnp.cos(head_yaw_radians),
            ),
            axis=2,
        )
        horizontal_position = source_eye_position[
            ..., jnp.asarray((0, 2), dtype=jnp.int32)
        ]
        horizontal_delta = (
            horizontal_position[:, None, :, :] - horizontal_position[:, :, None, :]
        )
        candidate_in_view = native_view_sector(
            horizontal_delta,
            forward_xz[:, :, None, :],
            view_angle[:, :, None],
        )

        def source_perception(source: jax.Array):
            return jax.vmap(
                lambda target: geometry_perception_line_of_sight_result(
                    geometry,
                    source,
                    target,
                    role_opaque_mask=role_opaque_mask,
                ),
                in_axes=1,
                out_axes=1,
            )(source_eye_position)

        candidate_perception = jax.vmap(
            source_perception,
            in_axes=1,
            out_axes=1,
        )(source_eye_position)
        candidate_valid = (
            ~(
                candidate_perception.geometry_exhausted
                | candidate_perception.capacity_exceeded
                | candidate_perception.invalid
            )
            & jnp.isfinite(view_angle)[:, :, None]
            & (view_angle[:, :, None] >= jnp.float32(0.0))
            & (view_angle[:, :, None] <= jnp.float32(math.tau))
            & ~jnp.eye(entity_count, dtype=jnp.bool_)[None, :, :]
        )
        candidate_perceptible = (
            candidate_perception.visible & candidate_in_view & candidate_valid
        )
        targeting = combat_target_selection(
            combat,
            params,
            config,
            candidate_evidence=candidate_perceptible,
        )
    target_id = targeting.engagement_target_id
    target_valid = targeting.engagement_target_mask
    target_offset_by_id = jnp.where(
        (target_id == AGENT_ENTITY)[..., None],
        agent_offset[:, None, :],
        target_offset[:, None, :],
    )
    target_position = _gather_entity(combat.position, target_id)
    entity_bounds = (
        jnp.broadcast_to(
            params.target_bounds[None, :],
            (entity_count, 6),
        )
        .at[AGENT_ENTITY]
        .set(params.agent_bounds)
    )
    start = source_eye_position
    end = target_position + target_offset_by_id
    muzzle_direction = None
    projectile_sources = _concrete_projectile_sources(config.loadout)
    if projectile_sources is not None and projectile_sources.size == 0:
        projectile_world_collision_available = jnp.zeros_like(target_valid)
        clear_projectile_flight = jnp.zeros_like(target_valid)
    else:
        target_bounds = _gather_entity(
            jnp.broadcast_to(
                entity_bounds[None, :, :],
                (batch, entity_count, 6),
            ),
            target_id,
        )
        projectile_extent, projectile_authored = _projectile_preflight_extent(
            config.loadout
        )
        safe_projectile_extent = jnp.broadcast_to(
            jnp.where(
                projectile_authored,
                projectile_extent,
                jnp.float32(0.1),
            )[..., None],
            combat.position.shape,
        )
        target_delta = target_position - world.muzzle_position
        muzzle_direction = projectile_look_direction(
            world.muzzle_yaw_degrees,
            world.muzzle_pitch_degrees,
        )
        projectile_query = projectile_authored & target_valid & world.muzzle_valid
        projectile_displacement = muzzle_direction * jnp.linalg.norm(
            target_delta,
            axis=2,
            keepdims=True,
        )
        projectile_displacement = jnp.where(
            projectile_query[..., None],
            projectile_displacement,
            jnp.float32(0.0),
        )
        if isinstance(geometry, GeometryState) and geometry.origin.shape[0] == 1:
            if projectile_sources is None:
                projectile_sources = np.arange(
                    batch * entity_count,
                    dtype=np.int32,
                )
            source_index = jnp.asarray(projectile_sources, dtype=jnp.int32)
            flat_muzzle = world.muzzle_position.reshape((-1, 3))[source_index]
            flat_displacement = projectile_displacement.reshape((-1, 3))[source_index]
            flat_extent = safe_projectile_extent.reshape((-1, 3))[source_index]
            flat_target = target_position.reshape((-1, 3))[source_index]
            flat_target_bounds = target_bounds.reshape((-1, 6))[source_index]
            flat_query = projectile_query.reshape((-1,))[source_index]
            entity_hit, entity_fraction = projectile_entity_first_contact(
                flat_muzzle,
                flat_displacement,
                flat_extent,
                flat_target,
                flat_target_bounds,
                flat_query,
            )
            contact = geometry_projectile_first_contact_result(
                geometry,
                flat_muzzle,
                flat_displacement,
                jnp.concatenate((-flat_extent, flat_extent), axis=1),
                entity_fraction,
                entity_hit,
            )
            packed_available = contact.available & flat_query
            flat_available = (
                jnp.zeros(
                    (batch * entity_count,),
                    dtype=jnp.bool_,
                )
                .at[source_index]
                .set(packed_available)
            )
            flat_clear = (
                jnp.zeros_like(flat_available)
                .at[source_index]
                .set(contact.clear & packed_available)
            )
            projectile_world_collision_available = flat_available.reshape(
                (batch, entity_count)
            )
            clear_projectile_flight = flat_clear.reshape((batch, entity_count))
        else:
            projectile_entity_hit, projectile_entity_fraction = (
                projectile_entity_first_contact(
                    world.muzzle_position,
                    projectile_displacement,
                    safe_projectile_extent,
                    target_position,
                    target_bounds,
                    projectile_query,
                )
            )
            projectile_bounds = jnp.concatenate(
                (-safe_projectile_extent, safe_projectile_extent),
                axis=2,
            )
            projectile_contact = jax.lax.map(
                lambda values: geometry_projectile_first_contact_result(
                    geometry,
                    *values,
                ),
                (
                    jnp.moveaxis(world.muzzle_position, 1, 0),
                    jnp.moveaxis(projectile_displacement, 1, 0),
                    jnp.moveaxis(projectile_bounds, 1, 0),
                    jnp.moveaxis(projectile_entity_fraction, 1, 0),
                    jnp.moveaxis(projectile_entity_hit, 1, 0),
                ),
            )
            projectile_contact = jax.tree.map(
                lambda value: jnp.moveaxis(value, 0, 1),
                projectile_contact,
            )
            projectile_world_collision_available = (
                projectile_contact.available & projectile_query
            )
            clear_projectile_flight = (
                projectile_contact.clear & projectile_world_collision_available
            )
    actor_state = geometry_actor_world_state_result(
        geometry,
        combat.position[:, AGENT_ENTITY],
        combat.position[:, AGENT_ENTITY] + params.agent_eye_offset,
        params.agent_bounds,
        maximum_drop_distance=params.agent_walk_max_drop_height,
    )
    # The public actor-world-state ABI measures AGENT_ENTITY only. Keep the
    # deployable evidence matrix actor-shaped without reinterpreting that one
    # measured row as evidence for autonomous/NPC entities.
    actor_medium_ok = (
        jnp.zeros_like(
            config.player_backed_actor_mask,
            dtype=jnp.bool_,
        )
        .at[:, AGENT_ENTITY]
        .set(
            actor_state.controller_medium_available
            & actor_state.submersion_available
            & ~actor_state.controller_in_fluid
            & ~actor_state.feet_submerged
            & ~actor_state.eyes_submerged
        )
    )
    if entity_count == ENTITY_COUNT:
        # Preserve the certified pair's exact traversal and broadcast shape.
        pair_perception = geometry_perception_line_of_sight_result(
            geometry,
            start[:, AGENT_ENTITY],
            end[:, AGENT_ENTITY],
            role_opaque_mask=role_opaque_mask,
        )
        pair_selector = geometry_hitbox_line_of_sight_result(
            geometry,
            start[:, AGENT_ENTITY],
            end[:, AGENT_ENTITY],
        )

        def paired_result(value: jax.Array) -> jax.Array:
            return jnp.broadcast_to(value[:, None], combat.health.shape)

        perception = jax.tree_util.tree_map(
            paired_result,
            pair_perception,
        )
        selector = jax.tree_util.tree_map(
            paired_result,
            pair_selector,
        )
        pair_perception_valid = ~(
            pair_perception.geometry_exhausted
            | pair_perception.capacity_exceeded
            | pair_perception.invalid
        )
        candidate_perceptible = jnp.zeros(
            (batch, entity_count, entity_count),
            dtype=jnp.bool_,
        )
        candidate_perceptible = candidate_perceptible.at[
            :, AGENT_ENTITY, TARGET_ENTITY
        ].set(pair_perception.visible)
        candidate_perceptible = candidate_perceptible.at[
            :, TARGET_ENTITY, AGENT_ENTITY
        ].set(pair_perception.visible)
        candidate_valid = jnp.zeros_like(candidate_perceptible)
        candidate_valid = candidate_valid.at[:, AGENT_ENTITY, TARGET_ENTITY].set(
            pair_perception_valid
        )
        candidate_valid = candidate_valid.at[:, TARGET_ENTITY, AGENT_ENTITY].set(
            pair_perception_valid
        )
    else:

        def selected_candidate_result(value: jax.Array) -> jax.Array:
            selected = jnp.take_along_axis(
                value,
                jnp.clip(target_id, 0, entity_count - 1)[..., None],
                axis=2,
            )[..., 0]
            return jnp.where(
                target_valid,
                selected,
                jnp.zeros_like(selected),
            )

        perception = jax.tree_util.tree_map(
            selected_candidate_result,
            candidate_perception,
        )
        selector = jax.vmap(
            lambda source, target: geometry_hitbox_line_of_sight_result(
                geometry,
                source,
                target,
            ),
            in_axes=(1, 1),
            out_axes=1,
        )(start, end)
    perception_valid = ~(
        perception.geometry_exhausted
        | perception.capacity_exceeded
        | perception.invalid
    )
    # Unavailable cells are intentionally non-blocking for native selector
    # LOS. Its geometry_exhausted bit is a diagnostic, not invalidity.
    selector_valid = ~(selector.capacity_exceeded | selector.invalid)
    dodge_displacement, dodge_complete = _dodge_corridor_displacements(
        state,
        params,
        config.mechanics_rules,
    )
    dodge_position = jnp.broadcast_to(
        combat.position[:, :, None, :],
        dodge_displacement.shape,
    )
    dodge_bounds = jnp.broadcast_to(
        entity_bounds[None, :, None, :],
        dodge_displacement.shape[:-1] + (6,),
    )
    dodge_corridor_clear = (
        _swept_clearance(
            geometry,
            dodge_position,
            dodge_displacement,
            dodge_bounds,
        )
        & dodge_complete
    )
    clear_force_path = _force_corridor_clearance(
        state,
        params,
        config,
        geometry,
    )
    entity_only_area = entity_only_area_source_support(config.loadout)
    static_plan = static_area_placement_plan(config.loadout)
    static_sources = _concrete_static_area_sources(config.loadout)
    if static_sources is not None and static_sources.size == 0:
        static_area_placement = jnp.zeros_like(target_valid)
        area_center = target_position
    else:
        if muzzle_direction is None:
            muzzle_direction = projectile_look_direction(
                world.muzzle_yaw_degrees,
                world.muzzle_pitch_degrees,
            )
        static_query = static_plan.unique & world.muzzle_valid
        static_distance = jnp.where(
            static_query,
            static_plan.maximum_distance,
            jnp.float32(0.0),
        )
        if isinstance(geometry, GeometryState) and geometry.origin.shape[0] == 1:
            if static_sources is None:
                static_sources = np.arange(
                    batch * entity_count,
                    dtype=np.int32,
                )
            source_index = jnp.asarray(static_sources, dtype=jnp.int32)
            static_contact = geometry_point_ray_first_contact_result(
                geometry,
                world.muzzle_position.reshape((-1, 3))[source_index],
                muzzle_direction.reshape((-1, 3))[source_index],
                static_distance.reshape((-1,))[source_index],
            )
            packed_surface = native_deployable_surface_allowed(
                static_contact,
                static_plan.allow_walls.reshape((-1,))[source_index],
            )
            flat_placement = (
                jnp.zeros(
                    (batch * entity_count,),
                    dtype=jnp.bool_,
                )
                .at[source_index]
                .set(static_query.reshape((-1,))[source_index] & packed_surface)
            )
            flat_hit_point = (
                jnp.zeros(
                    (batch * entity_count, 3),
                    dtype=jnp.float32,
                )
                .at[source_index]
                .set(static_contact.hit_point)
            )
            static_area_placement = flat_placement.reshape((batch, entity_count))
            static_hit_point = flat_hit_point.reshape((batch, entity_count, 3))
        else:
            static_contact = jax.lax.map(
                lambda values: geometry_point_ray_first_contact_result(
                    geometry,
                    *values,
                ),
                (
                    jnp.moveaxis(world.muzzle_position, 1, 0),
                    jnp.moveaxis(muzzle_direction, 1, 0),
                    jnp.moveaxis(static_distance, 1, 0),
                ),
            )
            static_contact = jax.tree.map(
                lambda value: jnp.moveaxis(value, 0, 1),
                static_contact,
            )
            static_surface = jax.vmap(
                native_deployable_surface_allowed,
                in_axes=(1, 1),
                out_axes=1,
            )(static_contact, static_plan.allow_walls)
            static_area_placement = static_query & static_surface
            static_hit_point = static_contact.hit_point
        area_center = jnp.where(
            static_area_placement[..., None],
            static_hit_point,
            target_position,
        )

    return world._replace(
        actor_world_state_available=actor_state.available,
        actor_controller_medium_available=(actor_state.controller_medium_available),
        actor_submersion_available=actor_state.submersion_available,
        actor_drop_available=actor_state.drop_available,
        actor_controller_in_fluid=actor_state.controller_in_fluid,
        actor_feet_submerged=actor_state.feet_submerged,
        actor_eyes_submerged=actor_state.eyes_submerged,
        actor_drop_support_found=actor_state.drop_support_found,
        actor_drop_height=actor_state.drop_height,
        target_candidate_perceptible=candidate_perceptible,
        target_candidate_perception_valid=candidate_valid,
        line_of_sight=perception.visible & target_valid,
        line_of_sight_valid=perception_valid & target_valid,
        selector_line_of_sight=selector.clear & target_valid,
        selector_line_of_sight_valid=selector_valid & target_valid,
        clear_projectile_flight=clear_projectile_flight,
        projectile_world_collision_available=(projectile_world_collision_available),
        deployable_intended_graph_available=(
            config.player_proxy_deployable_launch_and_contact_lifecycle_timing_attested
            & config.player_backed_actor_mask
            # This bounded terminal graph carries the authored air trajectory
            # only. Native water drag uses a distinct authored terminal
            # velocity, so the provider must attest that the complete
            # projectile path/lifetime (not merely the launch actor) is dry.
            & config.deployable_projectile_dry_air_path_attested
            # Omitted Stats still acquire native default health and the
            # deployable is damageable/tangible. This bounded AreaState does
            # not model arbitrary attacks against deployable ECS entities, so
            # a provider must attest full-life noninterference. The projectile
            # runtime also rejects any contact it can detect against an active
            # typed deployable AABB before committing the step.
            & config.deployable_full_life_owner_valid_and_noninterference_attested[
                :, None
            ]
            # The authored Cooldown.Id is shared across roots. AbilityLoadout
            # currently stores slot-local cooldowns only, so this slice admits
            # exactly one fixed deployable profile with no item/root swap.
            & config.deployable_single_profile_no_swap_cooldown_group_attested[:, None]
            # Current actor-medium evidence is a defense-in-depth consistency
            # check on that episode-pinned full-path attestation.
            & actor_medium_ok
            & config.deployable_spatial_roster_and_group_equivalence_attested[:, None]
            & config.deployable_area_effect_candidates_available[:, None]
            & config.entity_projectile_collidable_available[:, None]
            & ~jnp.any(
                state.arsenal.areas.active[:, None, :]
                & state.arsenal.areas.deployable_count_towards_global_limit[:, None, :]
                & (
                    state.arsenal.areas.owner_entity_id[:, None, :]
                    == jnp.arange(entity_count, dtype=jnp.int32)[None, :, None]
                ),
                axis=2,
            )
        ),
        clear_force_path=clear_force_path,
        applied_force_collision_available=(config.applied_force_collision_support),
        dodge_corridor_clear=dodge_corridor_clear,
        entity_only_area=entity_only_area,
        static_area_placement=static_area_placement,
        area_center=area_center,
    )


def geometry_arsenal_world_token_source(
    state: ArsenalEnvironmentState,
    params: CombatParams,
    *,
    geometry: GeometryState,
    role_opaque_mask: jax.Array,
    geometry_provenance: int,
    traversal: TraversalTokenObservation | None = None,
    traversal_dynamic_state_visible: jax.Array | None = None,
    traversal_edge_state_visible: jax.Array | None = None,
    geometry_tile_index: jax.Array | None = None,
    view_sector_full_angle_radians: jax.Array | float = math.tau,
    max_los_cells: int | None = None,
    policy_config: WorldGeometryPolicyConfig | None = None,
) -> WorldGeometryTokenObservation:
    """Produce the actor-legal raw token row shared by policy companions."""

    token_config = normalize_world_geometry_policy_config(policy_config)
    combat = state.combat
    actor_position = combat.position[:, AGENT_ENTITY : AGENT_ENTITY + 1, :]
    actor_eye_position = actor_position + params.agent_eye_offset[None, None, :]
    yaw_radians = jnp.deg2rad(combat.yaw[:, AGENT_ENTITY])
    actor_forward = jnp.stack(
        (
            -jnp.sin(yaw_radians),
            jnp.zeros_like(yaw_radians),
            -jnp.cos(yaw_radians),
        ),
        axis=1,
    )[:, None, :]
    return produce_actor_world_geometry_tokens(
        geometry,
        actor_position,
        actor_eye_position,
        actor_forward,
        role_opaque_mask=role_opaque_mask,
        geometry_provenance=geometry_provenance,
        traversal=traversal,
        traversal_dynamic_state_visible=traversal_dynamic_state_visible,
        traversal_edge_state_visible=traversal_edge_state_visible,
        geometry_tile_index=geometry_tile_index,
        token_capacity=token_config.token_capacity,
        maximum_distance=token_config.maximum_distance,
        view_sector_full_angle_radians=view_sector_full_angle_radians,
        max_los_cells=max_los_cells,
    )


def geometry_arsenal_world_tokens(
    state: ArsenalEnvironmentState,
    params: CombatParams,
    *,
    geometry: GeometryState,
    role_opaque_mask: jax.Array,
    geometry_provenance: int,
    traversal: TraversalTokenObservation | None = None,
    traversal_dynamic_state_visible: jax.Array | None = None,
    traversal_edge_state_visible: jax.Array | None = None,
    geometry_tile_index: jax.Array | None = None,
    view_sector_full_angle_radians: jax.Array | float = math.tau,
    max_los_cells: int | None = None,
    policy_config: WorldGeometryPolicyConfig | None = None,
) -> WorldGeometryPolicyTokens:
    """Produce and sanitize the agent's actor-legal World geometry tokens."""

    token_config = normalize_world_geometry_policy_config(policy_config)
    source = geometry_arsenal_world_token_source(
        state,
        params,
        geometry=geometry,
        role_opaque_mask=role_opaque_mask,
        geometry_provenance=geometry_provenance,
        traversal=traversal,
        traversal_dynamic_state_visible=traversal_dynamic_state_visible,
        traversal_edge_state_visible=traversal_edge_state_visible,
        geometry_tile_index=geometry_tile_index,
        view_sector_full_angle_radians=view_sector_full_angle_radians,
        max_los_cells=max_los_cells,
        policy_config=token_config,
    )
    return encode_world_geometry_policy_tokens(
        source,
        actor_index=0,
        config=token_config,
    )


def _projectile_preflight_extent(loadout) -> tuple[jax.Array, jax.Array]:
    """Return each entity's conservative authored projectile half extent."""

    projectile = loadout.event_mask & (loadout.event_kind == EVENT_PROJECTILE)
    reduction_axes = tuple(range(2, projectile.ndim))
    extent = jnp.max(
        jnp.where(
            projectile,
            loadout.event_f32[..., EF_PROJECTILE_HALF_EXTENT],
            jnp.float32(0.0),
        ),
        axis=reduction_axes,
    )
    return extent, jnp.any(projectile, axis=reduction_axes)


def _concrete_projectile_sources(loadout) -> np.ndarray | None:
    """Return packed projectile sources when the episode loadout is static."""

    return _concrete_event_sources(loadout, EVENT_PROJECTILE)


def _concrete_event_sources(loadout, kind: int) -> np.ndarray | None:
    """Return sources with one event kind, or ``None`` for dynamic loadouts."""

    try:
        event_mask = np.asarray(loadout.event_mask, dtype=np.bool_)
        event_kind = np.asarray(loadout.event_kind)
    except (TypeError, ValueError):
        return None
    selected = event_mask & (event_kind == kind)
    authored = np.any(selected, axis=tuple(range(2, selected.ndim)))
    return np.flatnonzero(authored.reshape(-1)).astype(np.int32, copy=False)


def _concrete_has_angled_force(loadout) -> bool | None:
    """Return whether a static loadout has angled force events."""

    try:
        event_mask = np.asarray(loadout.event_mask, dtype=np.bool_)
        event_f32 = np.asarray(loadout.event_f32)
        event_flags = np.asarray(loadout.event_flags, dtype=np.uint32)
    except (TypeError, ValueError, jax.errors.TracerArrayConversionError):
        return None
    return bool(
        np.any(
            event_mask
            & (np.abs(event_f32[..., EF_FORCE_MAGNITUDE]) > 0.0)
            & ((event_flags & np.uint32(EVENT_FLAG_ANGLED_DAMAGE)) != 0)
        )
    )


def _concrete_static_area_sources(loadout) -> np.ndarray | None:
    """Return packed unambiguous static-placement sources for a static loadout."""

    try:
        ability_mask = np.asarray(loadout.ability_mask, dtype=np.bool_)
        requirements = np.asarray(loadout.ability_requirements, dtype=np.uint32)
    except (TypeError, ValueError):
        return None
    required = ability_mask & (
        (requirements & np.uint32(REQUIRE_STATIC_AREA_PLACEMENT)) != 0
    )
    unique = np.sum(required, axis=2) == 1
    return np.flatnonzero(unique.reshape(-1)).astype(np.int32, copy=False)


def _broadcast_view_angle(
    value: jax.Array | float,
    batch: int,
    entity_count: int,
) -> jax.Array:
    angle = jnp.asarray(value, dtype=jnp.float32)
    if angle.ndim == 0:
        return jnp.broadcast_to(angle, (batch, entity_count))
    if angle.shape == (entity_count,):
        return jnp.broadcast_to(angle[None, :], (batch, entity_count))
    if angle.shape == (1, entity_count):
        return jnp.broadcast_to(angle, (batch, entity_count))
    if angle.shape == (batch, entity_count):
        return angle
    raise ValueError(
        "view_sector_full_angle_radians must be scalar or have shape "
        f"[{entity_count}] or [{batch},{entity_count}]"
    )


def _broadcast_los_offset(
    value: jax.Array,
    batch: int,
    name: str,
) -> jax.Array:
    if value.shape == (batch, 3):
        return value
    if value.shape == (1, 3):
        return jnp.broadcast_to(value, (batch, 3))
    raise ValueError(f"{name} must have shape [1,3] or [{batch},3]")


def _gather_entity(array: jax.Array, entity_id: jax.Array) -> jax.Array:
    batch_index = jnp.arange(array.shape[0]).reshape(
        (array.shape[0],) + (1,) * (entity_id.ndim - 1)
    )
    return array[
        batch_index,
        jnp.clip(entity_id, 0, array.shape[1] - 1),
    ]


def _swept_clearance(
    geometry: GeometryProvider,
    position: jax.Array,
    displacement: jax.Array,
    bounds: jax.Array,
) -> jax.Array:
    """Evaluate a fixed query axis, sharing immutable local geometry when possible."""

    batch = position.shape[0]
    query_shape = position.shape[1:-1]
    query_count = 1
    for size in query_shape:
        query_count *= size

    if isinstance(geometry, GeometryState) and geometry.origin.shape[0] == 1:
        result = geometry_swept_volume_clearance_result(
            geometry,
            position.reshape((batch * query_count, 3)),
            displacement.reshape((batch * query_count, 3)),
            bounds.reshape((batch * query_count, 6)),
        )
        return result.clear.reshape((batch,) + query_shape)

    def query_major(value: jax.Array) -> jax.Array:
        return jnp.moveaxis(value, 0, -2).reshape((query_count, batch, value.shape[-1]))

    result = jax.vmap(
        lambda query_position, query_displacement, query_bounds: (
            geometry_swept_volume_clearance_result(
                geometry,
                query_position,
                query_displacement,
                query_bounds,
            )
        )
    )(
        query_major(position),
        query_major(displacement),
        query_major(bounds),
    )
    return jnp.swapaxes(result.clear, 0, 1).reshape((batch,) + query_shape)


def _force_corridor_clearance(
    state: ArsenalEnvironmentState,
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    geometry: GeometryProvider,
) -> jax.Array:
    """Certify authored force programs through a batch-wide static plan."""

    plan = config.force_sweep_plan
    loadout = config.loadout
    batch = state.combat.position.shape[0]
    if plan.ability_valid.shape[0] != batch:
        raise ValueError("force sweep plan batch does not match combat batch")
    entity_count = plan.program_affected.shape[1]
    if entity_count != state.combat.position.shape[1]:
        raise ValueError("force sweep plan entity axis does not match combat")

    (
        event_mask,
        event_time,
        event_kind,
        event_f32,
        event_i32,
        event_flags,
    ) = jax.tree_util.tree_map(
        lambda value: _gather_force_program(value, plan),
        ability_event_bank(loadout),
    )
    force_event = event_mask & (
        jnp.abs(event_f32[..., EF_FORCE_MAGNITUDE]) > jnp.float32(0.0)
    )
    target_mode = event_i32[..., EI_TARGET_MODE]
    has_angled_force = _concrete_has_angled_force(loadout)
    program_count = plan.program_valid.shape[0]
    victim_ids = jnp.arange(entity_count, dtype=jnp.int32)[None, :]
    source_victim = plan.program_source_index[:, None] == victim_ids
    source_victim_f32 = source_victim[..., None].astype(jnp.float32)
    other_victim = ~source_victim
    affected = plan.program_affected & plan.program_valid[:, None]
    base_position = state.combat.position[plan.program_environment_index]
    velocity = state.mechanics.applied_velocity[plan.program_environment_index]

    def gather_victims(value: jax.Array) -> jax.Array:
        return value[plan.program_environment_index]

    resistance = (
        gather_victims(state.mechanics.applied_air_resistance),
        gather_victims(state.mechanics.applied_air_resistance_max),
        gather_victims(state.mechanics.applied_ground_resistance),
        gather_victims(state.mechanics.applied_ground_resistance_max),
        gather_victims(state.mechanics.applied_resistance_threshold),
        gather_victims(state.mechanics.applied_resistance_style),
    )
    dampen_y = gather_victims(state.mechanics.applied_dampen_y)
    grounded_by_environment = (
        jnp.ones(
            (batch, entity_count),
            dtype=jnp.bool_,
        )
        .at[:, AGENT_ENTITY]
        .set(state.combat.agent_grounded)
    )
    grounded = grounded_by_environment[plan.program_environment_index]
    duration = _gather_force_program(
        loadout.ability_duration_seconds,
        plan,
    )
    duration = jnp.where(plan.program_valid, duration, jnp.float32(0.0))
    fired = jnp.zeros_like(force_event)
    elapsed = jnp.zeros_like(duration)
    offset = jnp.zeros(
        (program_count, entity_count, 3),
        dtype=jnp.float32,
    )
    source_yaw = state.combat.yaw[
        plan.program_environment_index,
        plan.program_source_index,
    ]
    direct_velocity_bank = adjust_vertical_force(
        local_force_velocity(
            event_f32,
            source_yaw[:, None],
        ),
        event_f32,
        event_i32,
    )
    target_yaw = state.combat.yaw[plan.program_environment_index]

    def simulate_tick(carry, index):
        (
            current_velocity,
            current_offset,
            current_elapsed,
            current_fired,
            air_min,
            air_max,
            ground_min,
            ground_max,
            threshold,
            style,
        ) = carry
        dt = _motion_delta(
            state.combat.tick_count[plan.program_environment_index]
            + jnp.int32(index + 1),
            state.combat.motion_timing_profile[plan.program_environment_index],
            params,
        )
        event_after = current_elapsed[:, None] + dt[:, None]
        due = (
            force_event
            & ~current_fired
            & (event_time >= current_elapsed[:, None])
            & (event_time < event_after)
            & (event_time < duration[:, None])
            & plan.program_valid[:, None]
        )
        positions = base_position + current_offset
        source_position = jnp.sum(
            positions * source_victim_f32,
            axis=1,
        )
        if has_angled_force is False:
            source_relative_yaw = None
        else:
            source_offset = source_position[:, None, :] - positions
            source_bearing = jnp.rad2deg(
                jnp.arctan2(
                    source_offset[..., 0],
                    source_offset[..., 2],
                )
            )
            source_relative_yaw = _normalize_degrees(
                source_bearing + jnp.float32(180.0) - target_yaw
            )

        def apply_event(slot, event_carry):
            (
                event_velocity_state,
                event_air_min,
                event_air_max,
                event_ground_min,
                event_ground_max,
                event_threshold,
                event_style,
            ) = event_carry
            enabled = due[:, slot]
            f32 = event_f32[:, slot, :]
            i32 = event_i32[:, slot, :]
            if source_relative_yaw is None:
                angled = jnp.zeros_like(source_victim)
            else:
                angled = (
                    (event_flags[:, slot] & jnp.uint32(EVENT_FLAG_ANGLED_DAMAGE)) != 0
                )[:, None] & (
                    jnp.abs(
                        _normalize_degrees(
                            source_relative_yaw - f32[:, EF_ANGLED_ANGLE_DEGREES, None]
                        )
                    )
                    < f32[:, EF_ANGLED_DISTANCE_DEGREES, None]
                )
            melee_velocity = damage_force_velocity(
                f32[:, None, :],
                i32[:, None, :],
                source_position[:, None, :],
                positions,
                source_yaw[:, None],
                angled,
            )
            direct_velocity = direct_velocity_bank[:, slot]
            authored_velocity = jnp.where(
                (event_kind[:, slot] == EVENT_MELEE_CONE)[:, None, None],
                melee_velocity,
                direct_velocity[:, None, :],
            )
            targets = jnp.where(
                (target_mode[:, slot] == TARGET_SELF)[:, None],
                source_victim,
                other_victim,
            )
            applies = enabled[..., None] & targets
            candidate_velocity = jnp.where(
                (i32[:, EI_FORCE_MODE] == FORCE_SET)[:, None, None],
                authored_velocity,
                event_velocity_state + authored_velocity,
            )
            event_velocity_state = jnp.where(
                applies[..., None],
                candidate_velocity,
                event_velocity_state,
            )

            def replace(current: jax.Array, value: jax.Array) -> jax.Array:
                return jnp.where(applies, value[:, None], current)

            return (
                event_velocity_state,
                replace(event_air_min, f32[:, EF_AIR_RESISTANCE]),
                replace(event_air_max, f32[:, EF_AIR_RESISTANCE_MAX]),
                replace(event_ground_min, f32[:, EF_GROUND_RESISTANCE]),
                replace(event_ground_max, f32[:, EF_GROUND_RESISTANCE_MAX]),
                replace(event_threshold, f32[:, EF_RESISTANCE_THRESHOLD]),
                replace(event_style, i32[:, EI_RESISTANCE_STYLE]),
            )

        (
            current_velocity,
            air_min,
            air_max,
            ground_min,
            ground_max,
            threshold,
            style,
        ) = jax.lax.fori_loop(
            0,
            event_mask.shape[1],
            apply_event,
            (
                current_velocity,
                air_min,
                air_max,
                ground_min,
                ground_max,
                threshold,
                style,
            ),
        )
        current_fired |= due
        active = (current_elapsed < duration) & plan.program_valid
        speed_squared = jnp.sum(current_velocity * current_velocity, axis=2)
        moving = active[..., None] & (
            speed_squared >= jnp.float32(DODGE_VELOCITY_REMOVAL_SQUARED)
        )
        current_offset = current_offset + jnp.where(
            moving[..., None],
            current_velocity * dt[:, None, None],
            jnp.float32(0.0),
        )
        threshold_squared = jnp.maximum(
            threshold * threshold,
            jnp.finfo(jnp.float32).tiny,
        )
        linear_blend = jnp.clip(
            jnp.sqrt(speed_squared) / jnp.sqrt(threshold_squared),
            0.0,
            1.0,
        )
        exponential_blend = jnp.clip(
            speed_squared / threshold_squared,
            0.0,
            1.0,
        )
        blend = jnp.where(style == 1, exponential_blend, linear_blend)
        minimum = jnp.where(grounded, ground_min, air_min)
        maximum = jnp.where(grounded, ground_max, air_max)
        scale = (minimum * blend + maximum * (1.0 - blend)) ** (
            jnp.float32(60.0) / config.mechanics_rules.server_ticks_per_second
        )
        damped = current_velocity.at[..., 0].set(current_velocity[..., 0] * scale)
        damped = damped.at[..., 2].set(current_velocity[..., 2] * scale)
        damped = damped.at[..., 1].set(
            jnp.where(
                dampen_y,
                current_velocity[..., 1] * scale,
                current_velocity[..., 1],
            )
        )
        current_velocity = jnp.where(
            moving[..., None],
            damped,
            jnp.float32(0.0),
        )
        current_elapsed = current_elapsed + jnp.where(
            active,
            dt,
            jnp.float32(0.0),
        )
        return (
            (
                current_velocity,
                current_offset,
                current_elapsed,
                current_fired,
                air_min,
                air_max,
                ground_min,
                ground_max,
                threshold,
                style,
            ),
            current_offset,
        )

    force_state = (
        velocity,
        offset,
        elapsed,
        fired,
        *resistance,
    )
    # Unwritten suffix rows stay neutral to the zero-clamped deviation bounds.
    trajectory = jnp.zeros(
        (FORCE_CLEARANCE_TICK_CAPACITY,) + offset.shape,
        dtype=offset.dtype,
    )

    def force_pending(loop):
        index, current, _ = loop
        return (index < FORCE_CLEARANCE_TICK_CAPACITY) & jnp.any(
            plan.program_valid & (current[2] < duration)
        )

    def simulate_force(loop):
        index, current, current_trajectory = loop
        current, current_offset = simulate_tick(current, index)
        return (
            index + jnp.int32(1),
            current,
            current_trajectory.at[index].set(current_offset),
        )

    _, force_state, trajectory = jax.lax.while_loop(
        force_pending,
        simulate_force,
        (jnp.int32(0), force_state, trajectory),
    )
    (
        _,
        final_offset,
        final_elapsed,
        final_fired,
        *_,
    ) = force_state
    complete = (
        plan.program_valid
        & (final_elapsed >= duration)
        & jnp.all(~force_event | final_fired, axis=1)
    )
    final_squared = jnp.sum(final_offset * final_offset, axis=2)
    projection = jnp.sum(
        trajectory * final_offset[None, ...],
        axis=3,
    ) / jnp.maximum(final_squared[None, ...], jnp.finfo(jnp.float32).tiny)
    projection = jnp.clip(projection, 0.0, 1.0)
    deviation = trajectory - projection[..., None] * final_offset[None, ...]
    minimum_deviation = jnp.minimum(
        jnp.min(deviation, axis=0),
        jnp.float32(0.0),
    )
    maximum_deviation = jnp.maximum(
        jnp.max(deviation, axis=0),
        jnp.float32(0.0),
    )
    victim_bounds = (
        jnp.broadcast_to(
            params.target_bounds[None, :],
            (entity_count, 6),
        )
        .at[AGENT_ENTITY]
        .set(params.agent_bounds)
    )
    victim_bounds = jnp.broadcast_to(
        victim_bounds[None, :, :],
        (program_count, entity_count, 6),
    )
    expanded_bounds = jnp.concatenate(
        (
            victim_bounds[..., :3] + minimum_deviation,
            victim_bounds[..., 3:] + maximum_deviation,
        ),
        axis=2,
    )
    safe_delta = jnp.asarray((2.0e-6, 0.0, 0.0), dtype=jnp.float32)
    query_displacement = jnp.where(
        (final_squared > jnp.float32(1.0e-12))[..., None],
        final_offset,
        safe_delta,
    )
    sweep_program = plan.sweep_program_index
    sweep_victim = plan.sweep_victim_index
    sweep_geometry = _force_sweep_geometry(
        geometry,
        plan,
        batch,
    )
    packed_clear = _swept_clearance(
        sweep_geometry,
        base_position[sweep_program, sweep_victim],
        query_displacement[sweep_program, sweep_victim],
        expanded_bounds[sweep_program, sweep_victim],
    )
    packed_clear &= plan.sweep_valid
    clear = jnp.zeros(
        (program_count, entity_count),
        dtype=jnp.uint8,
    )
    clear = clear.at[
        sweep_program,
        sweep_victim,
    ].max(packed_clear.astype(jnp.uint8))
    clear = clear.astype(jnp.bool_)
    ability_clear = plan.program_valid & complete & jnp.all(~affected | clear, axis=1)
    output_capacity = entity_count * ABILITY_CAPACITY
    output_index = (
        plan.program_source_index * ABILITY_CAPACITY + plan.program_slot_index
    )
    output = jnp.zeros(
        (batch, output_capacity),
        dtype=jnp.uint8,
    )
    output = output.at[
        plan.program_environment_index,
        output_index,
    ].max(ability_clear.astype(jnp.uint8))
    return output.astype(jnp.bool_).reshape((batch, entity_count, ABILITY_CAPACITY))


def _force_sweep_geometry(
    geometry: GeometryProvider,
    plan: ForceSweepPlan,
    batch: int,
) -> GeometryProvider:
    """Align per-environment geometry with batch-flattened sweep lanes."""

    if isinstance(geometry, RegionGeometryState):
        if geometry.environment_world_id.shape != (batch,):
            raise ValueError("Region geometry batch does not match combat")
        sweep_environment_index = plan.sweep_environment_index
        mutable_blocks = geometry.mutable_blocks
        if mutable_blocks is not None:
            for leaf in jax.tree_util.tree_leaves(mutable_blocks):
                if leaf.ndim == 0 or leaf.shape[0] != batch:
                    raise ValueError("Region mutable-block batch does not match combat")
            mutable_blocks = jax.tree_util.tree_map(
                lambda value: value[sweep_environment_index],
                mutable_blocks,
            )
        return geometry._replace(
            environment_world_id=geometry.environment_world_id[sweep_environment_index],
            mutable_blocks=mutable_blocks,
        )
    if isinstance(geometry, GeometryState):
        geometry_batch = geometry.origin.shape[0]
        if geometry_batch == 1:
            return geometry
        if geometry_batch != batch:
            raise ValueError("local geometry batch does not match combat")
        return jax.tree_util.tree_map(
            lambda value: (
                None if value is None else value[plan.sweep_environment_index]
            ),
            geometry,
        )
    return geometry


def _gather_force_program(
    value: jax.Array,
    plan: ForceSweepPlan,
) -> jax.Array:
    """Gather a source/ability tensor through the flattened runtime plan."""

    return value[
        plan.program_environment_index,
        plan.program_source_index,
        plan.program_slot_index,
    ]


def _normalize_degrees(value: jax.Array) -> jax.Array:
    return (value + jnp.float32(180.0)) % jnp.float32(360.0) - jnp.float32(180.0)


def _dodge_corridor_displacements(
    state: ArsenalEnvironmentState,
    params: CombatParams,
    rules: CombatMechanicsRules,
) -> tuple[jax.Array, jax.Array]:
    """Integrate the same open-space dodge recurrence used by the runtime."""

    combat = state.combat
    batch, entity_count = combat.health.shape
    radians = jnp.deg2rad(combat.yaw)
    authored_velocity = (
        jnp.stack(
            (
                -jnp.sin(radians),
                jnp.zeros_like(radians),
                -jnp.cos(radians),
            ),
            axis=2,
        )
        * rules.dodge_force[..., None]
    )
    native_profile = (
        rules.dodge_execution_profile == DODGE_EXECUTION_NATIVE_NPC_NULL_CONFIG
    )
    velocity = resolve_dodge_launch_velocity(
        authored_velocity,
        rules.dodge_execution_profile,
        params,
    )
    grounded = (
        jnp.ones(
            (batch, entity_count),
            dtype=jnp.bool_,
        )
        .at[:, AGENT_ENTITY]
        .set(combat.agent_grounded)
    )
    air_resistance = jnp.where(
        native_profile,
        params.agent_force_air_drag_min,
        rules.dodge_air_resistance,
    )
    air_resistance_max = jnp.where(
        native_profile,
        params.agent_force_air_drag_max,
        rules.dodge_air_resistance_max,
    )
    ground_resistance = jnp.where(
        native_profile,
        params.agent_force_ground_drag_base,
        rules.dodge_ground_resistance,
    )
    ground_resistance_max = jnp.where(
        native_profile,
        params.agent_force_ground_drag_base,
        rules.dodge_ground_resistance_max,
    )
    resistance_threshold = jnp.where(
        native_profile,
        params.agent_force_air_drag_max_speed,
        rules.dodge_resistance_threshold,
    )
    resistance_style = jnp.where(
        native_profile,
        jnp.int32(APPLIED_RESISTANCE_STYLE_FORCE_VELOCITY),
        jnp.int32(1),
    )
    dampen_y = jnp.zeros_like(resistance_style, dtype=jnp.bool_)

    def moving(current_velocity: jax.Array) -> jax.Array:
        speed_squared = jnp.sum(current_velocity * current_velocity, axis=2)
        return jnp.where(
            native_profile,
            jnp.any(current_velocity != jnp.float32(0.0), axis=2),
            speed_squared >= jnp.float32(DODGE_VELOCITY_REMOVAL_SQUARED),
        )

    def continue_dodge(carry) -> jax.Array:
        index, current_velocity, _ = carry
        return (index < jnp.int32(DODGE_CLEARANCE_TICK_CAPACITY)) & jnp.any(
            moving(current_velocity)
        )

    def advance(carry):
        index, current_velocity, displacement = carry
        active = moving(current_velocity)
        dt = _motion_delta(
            combat.tick_count + jnp.int32(index + 1),
            combat.motion_timing_profile,
            params,
        )[:, None]
        displacement = displacement + jnp.where(
            active[..., None],
            current_velocity * dt[..., None],
            jnp.float32(0.0),
        )
        damped = damp_applied_velocity(
            current_velocity,
            grounded,
            air_resistance,
            air_resistance_max,
            ground_resistance,
            ground_resistance_max,
            resistance_threshold,
            resistance_style,
            dampen_y,
            rules.server_ticks_per_second,
            params,
        )
        return index + jnp.int32(1), damped, displacement

    _, final_velocity, displacement = jax.lax.while_loop(
        continue_dodge,
        advance,
        (jnp.int32(0), velocity, jnp.zeros_like(velocity)),
    )
    configured_complete = jnp.sum(
        final_velocity * final_velocity, axis=2
    ) < jnp.float32(DODGE_VELOCITY_REMOVAL_SQUARED)
    complete = jnp.where(
        native_profile,
        ~jnp.any(final_velocity != jnp.float32(0.0), axis=2),
        configured_complete,
    )
    forward_x = displacement[..., 0]
    forward_y = displacement[..., 1]
    forward_z = displacement[..., 2]
    backward = jnp.stack(
        (-forward_x, forward_y, -forward_z),
        axis=2,
    )
    left = jnp.stack(
        (
            forward_z,
            forward_y,
            -forward_x,
        ),
        axis=2,
    )
    right = jnp.stack(
        (-forward_z, forward_y, forward_x),
        axis=2,
    )
    return (
        jnp.stack((displacement, backward, left, right), axis=2),
        jnp.broadcast_to(complete[..., None], (batch, entity_count, 4)),
    )


__all__ = [
    "ACTION_SURFACE_REJECT_EXECUTOR_DENIED",
    "ACTION_SURFACE_REJECT_EXECUTOR_UNAVAILABLE",
    "ACTION_SURFACE_REJECT_NONE",
    "ACTION_SURFACE_REJECT_TARGET_RECHECK",
    "ACTION_SURFACE_VERB_BLOCK_INTERACTION",
    "ACTION_SURFACE_VERB_COUNT",
    "ACTION_SURFACE_VERB_CRAFT",
    "ACTION_SURFACE_VERB_USE",
    "ARSENAL_ENVIRONMENT_SPEC",
    "ArsenalActionSurfaceEvidence",
    "ArsenalActionSurfaceExecution",
    "ArsenalActionSurfaceExecutor",
    "ArsenalActionSurfaceLifecycle",
    "ArsenalActionSurfaceProvider",
    "ArsenalActionSurfaceRuntime",
    "ArsenalActionSurfaceRuntimeInitializer",
    "ArsenalActionSurfaceViews",
    "ArsenalActorBlockCandidateSource",
    "ArsenalActorBlockCandidateSourceProvider",
    "ArsenalBlockCandidateProvider",
    "ArsenalBlockTriggerAvailabilityProvider",
    "ArsenalItemInteractionAvailabilityProvider",
    "ArsenalItemInteractionTable",
    "ArsenalOpponentAbilityProvider",
    "ArsenalEnvironment",
    "ArsenalEnvironmentCarry",
    "ArsenalEnvironmentInfo",
    "ArsenalPPOEnvironmentState",
    "ArsenalRecipeCandidateProvider",
    "ArsenalResetBatch",
    "ArsenalResetProvider",
    "ArsenalInventoryResetProvider",
    "ArsenalRuntimeExplosionCandidateProvider",
    "ArsenalUseAvailabilityProvider",
    "ArsenalWorldCapabilityProvider",
    "ArsenalWorldFeatureProvider",
    "ArsenalWorldLightTokenProvider",
    "ArsenalWorldRuntimeProvider",
    "ArsenalWorldRuntimeViews",
    "ArsenalWorldTokenProvider",
    "NATIVE_ITEM_INTERACTION_RECORDING_SCHEMA",
    "REFERENCE_ARSENAL_POLICY_ACTION_HEAD_SIZES",
    "REFERENCE_SKILL_MAXIMUM_TURN_DEGREES",
    "action_surface_lifecycle_evidence",
    "action_surface_verb_requests",
    "fail_closed_arsenal_world_capabilities",
    "geometry_arsenal_world_capabilities",
    "geometry_arsenal_world_token_source",
    "geometry_arsenal_world_tokens",
    "item_interaction_table_from_native_evidence",
    "item_interaction_evidence_from_native_recording",
    "item_interaction_table_from_native_recording",
    "item_interaction_trigger_availability",
    "materialize_arsenal_action_surface_evidence",
    "make_arsenal_action_surface_provider",
    "make_arsenal_environment",
    "make_inventory_item_interaction_availability_provider",
    "make_region_runtime_block_candidate_provider",
    "make_region_runtime_camera_block_candidate_provider",
    "make_world_actor_block_candidate_provider",
    "open_flat_arsenal_world_capabilities",
]
