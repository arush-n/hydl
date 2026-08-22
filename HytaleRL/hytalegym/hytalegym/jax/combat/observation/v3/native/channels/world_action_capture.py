"""Strict socket composition for atomic native World-action candidates.

This module is the host half of a versioned bridge endpoint.  One ``observe``
request returns actor-safe bounded candidates and their privileged identities
from a single World-thread capture.  A later ``commit`` request names that
opaque candidate generation and returns freshly captured evidence.  No native
String ID or absolute cell enters the learner row.

Version 2 routes every capture through an explicit policy actor slot plus its
reset-negotiated UUID.  It intentionally opens only the surfaces it can
represent without an
invented value.  Place is a deliberate rotation-zero policy subset, but stays
closed until the bridge can prove its exact empty destination and pair it with
the matching Region placed-geometry identity.  Recipe stays closed unless the
bridge returns the complete ordinary-bench/fieldcraft context and the host has
the exact packed table and encoder parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import uuid
from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.inventory import parse_native_inventory_frame
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.blocks import (
    empty_block_action_candidate_policy_view,
    encode_block_action_candidate_policy_view,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.contract import (
    RECIPE_CANDIDATE_EMBEDDING_SIZE,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.encoder import (
    RecipeCandidateEncoderParams,
    RecipeCandidateEncoding,
    encode_recipe_candidates,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.recipes import (
    empty_recipe_candidate_policy_view,
    encode_recipe_candidate_policy_view,
)
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerArsenalAction,
)
from hytalegym.jax.crafting import PackedRecipeTable
from hytalegym.jax.crafting.contract import (
    RECIPE_TABLE_CAPACITY,
    identity_sha256_words,
)
from hytalegym.jax.world import (
    ACTOR_RECIPE_CANDIDATE_CAPACITY,
    ActorBlockActionCandidates,
    produce_actor_recipe_candidates_from_legal_mask,
)
from hytalegym.worldgen import (
    NativeBlockUseEvidence,
    NativeCraftingRecipe,
    NativeItemInteractionEvidence,
    NativeWorldVerbTransportSession,
)
from hytalegym.worldgen.native_action_resolution import (
    NativeCraftingExecutionContext,
)
from hytalegym.worldgen.native_mutable_blocks import (
    NativeMutableBlockCells,
    NativeMutableBlockRow,
)

from .world_actions import (
    NativeBlockCandidateBinding,
    NativeBlockCommitBinding,
    NativePolicyActionCandidateSnapshot,
    NativePolicyActionCommitEvidence,
    NativePolicyActionSurface,
    NativePolicyWorldActionAdapter,
    NativeRecipeCandidateBinding,
    NativeUseCandidateBinding,
    native_inventory_semantic_sha256,
)


NATIVE_POLICY_WORLD_ACTION_CAPTURE_SCHEMA = (
    "hytalerl_native_policy_world_action_capture_v2"
)
NATIVE_POLICY_WORLD_ACTION_CAPTURE_VERSION = 2
NATIVE_POLICY_WORLD_ACTION_CAPTURE_TYPE = "policy_world_action_capture"

BridgeRequest = Callable[[dict[str, object]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class _CaptureIdentity:
    phase: str
    bridge_sha256: str
    world: str
    world_epoch: str
    world_tick: int
    environment_step: int
    actor_slot: int
    actor_identity: str
    inventory_identity_sha256: str
    recipe_table_identity_sha256: str
    candidate_generation_sha256: str
    expected_candidate_generation_sha256: str
    expected_generation_matched: bool
    inventory: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _DecodedSurface:
    identity: _CaptureIdentity
    surface: NativePolicyActionSurface


class NativePolicyWorldActionCaptureAdapter(NativePolicyWorldActionAdapter):
    """Bind the generic resolver to one same-connection socket endpoint.

    The adapter is opt-in.  :class:`~hytalegym.envs.hytale_env.HytaleEnv`
    binds its already-open ``BridgeConnection.send_and_recv`` method at reset;
    a second socket is therefore never used for candidate or commit evidence.
    """

    def __init__(
        self,
        *,
        packed_recipes: PackedRecipeTable | None = None,
        recipe_encoder_params: RecipeCandidateEncoderParams | None = None,
        actor_slot: int = 0,
        expected_actor_identity: str | None = None,
        initial_request_id: int = 0,
    ) -> None:
        if packed_recipes is not None and not isinstance(
            packed_recipes,
            PackedRecipeTable,
        ):
            raise TypeError("packed_recipes must be PackedRecipeTable or None")
        if recipe_encoder_params is not None and not isinstance(
            recipe_encoder_params,
            RecipeCandidateEncoderParams,
        ):
            raise TypeError(
                "recipe_encoder_params must be RecipeCandidateEncoderParams "
                "or None"
            )
        self._bridge_request: BridgeRequest | None = None
        self._bridge_request_key: tuple[object, object] | None = None
        self._packed_recipes = packed_recipes
        self._recipe_encoder_params = recipe_encoder_params
        self._last_surface_inventory: Mapping[str, object] | None = None
        if (
            isinstance(actor_slot, bool)
            or not isinstance(actor_slot, int)
            or actor_slot < 0
        ):
            raise ValueError("actor_slot must be a nonnegative integer")
        self._actor_slot = actor_slot
        self._expected_actor_identity = (
            None
            if expected_actor_identity is None
            else _canonical_expected_actor_identity(expected_actor_identity)
        )
        super().__init__(
            self._capture_surface,
            self._capture_commit,
            initial_request_id=initial_request_id,
        )

    @property
    def actor_slot(self) -> int:
        """Reset-negotiated policy actor owned by this adapter instance."""

        return self._actor_slot

    @property
    def last_surface_inventory(self) -> Mapping[str, object] | None:
        """Exact inventory captured with the most recent actor surface.

        Actor-major hosts need the inventory belonging to each independently
        captured policy row.  The ordinary observation carries only the
        single-actor compatibility inventory, so exposing this already-
        validated capture half prevents a nonzero row from borrowing actor
        zero's tokens.  The mapping remains host-only and is replaced on each
        successful observe capture.
        """

        return self._last_surface_inventory

    def reset(self) -> None:
        """Reset the request sequence and discard the prior episode capture."""

        super().reset()
        self._last_surface_inventory = None

    def bind_bridge_request(self, request: BridgeRequest) -> None:
        """Bind exactly the environment's live request/response function."""

        if not callable(request):
            raise TypeError("request must be callable")
        key = _callable_key(request)
        if self._bridge_request is not None and key != self._bridge_request_key:
            raise RuntimeError(
                "native World-action capture adapter is already bound to "
                "another environment connection"
            )
        self._bridge_request = request
        self._bridge_request_key = key

    def unbind_bridge_request(self, request: BridgeRequest | None = None) -> None:
        """Drop the borrowed connection callback without closing its socket."""

        if request is not None and self._bridge_request_key != _callable_key(request):
            raise RuntimeError(
                "cannot unbind a World-action adapter owned by another "
                "environment connection"
            )
        self._bridge_request = None
        self._bridge_request_key = None

    def _request(self, payload: dict[str, object]) -> Mapping[str, Any]:
        if self._bridge_request is None:
            raise RuntimeError(
                "native World-action capture adapter is not bound to the "
                "environment BridgeConnection"
            )
        response = self._bridge_request(payload)
        if not isinstance(response, Mapping):
            raise TypeError("native World-action capture response must be an object")
        return response

    def _capture_surface(
        self,
        _assembly: object,
        info: Mapping[str, Any],
    ) -> NativePolicyActionSurface:
        expected_actor = self._expected_actor_identity or (
            _policy_actor_identity(info, self._actor_slot)
        )
        if expected_actor is None:
            raise RuntimeError(
                "native World-action capture requires the policy actor "
                "identity from reset negotiation or adapter construction"
            )
        expected_bridge = _optional_string(info, "bridge_sha256")
        expected_epoch = _optional_string(info, "native_world_verb_epoch")
        expected_world = _optional_string(info, "world")
        expected_step = _optional_integer(info, "step_count")
        if (
            expected_bridge is None
            or expected_epoch is None
            or expected_world is None
            or expected_step is None
        ):
            raise RuntimeError(
                "native World-action capture requires reset-negotiated "
                "bridge, World, epoch, and environment-step identity"
            )
        response = self._request(
            native_policy_world_action_capture_request(
                actor_slot=self._actor_slot,
                expected_actor_identity=expected_actor,
            )
        )
        decoded = decode_native_policy_world_action_surface(
            response,
            expected_bridge_sha256=expected_bridge,
            expected_world_epoch=expected_epoch,
            expected_world_identity=expected_world,
            expected_environment_step=expected_step,
            expected_actor_slot=self._actor_slot,
            expected_actor_identity=expected_actor,
            packed_recipes=self._packed_recipes,
            recipe_encoder_params=self._recipe_encoder_params,
        )
        self._last_surface_inventory = decoded.identity.inventory
        return decoded.surface

    def _capture_commit(
        self,
        action: LearnerArsenalAction,
        snapshot: NativePolicyActionCandidateSnapshot,
        info: Mapping[str, Any],
    ) -> NativePolicyActionCommitEvidence:
        response = self._request(
            native_policy_world_action_capture_request(
                phase="commit",
                actor_slot=snapshot.actor_slot,
                expected_actor_identity=snapshot.actor_identity,
                expected_candidate_generation_sha256=(
                    snapshot.candidate_generation_sha256
                ),
                action=action,
            )
        )
        return decode_native_policy_world_action_commit(
            response,
            snapshot=snapshot,
            info=info,
            packed_recipes=self._packed_recipes,
            expected_action=action,
        )


def native_policy_world_action_capture_request(
    *,
    phase: str = "observe",
    actor_slot: int,
    expected_actor_identity: str,
    expected_candidate_generation_sha256: str = "",
    action: LearnerArsenalAction | None = None,
) -> dict[str, object]:
    """Build one versioned observe or commit request.

    Commit carries only bounded policy selections plus the opaque generation;
    it never sends the privileged target back to the server as an authority.
    """

    if phase not in {"observe", "commit"}:
        raise ValueError("phase must be observe or commit")
    if (
        isinstance(actor_slot, bool)
        or not isinstance(actor_slot, int)
        or actor_slot < 0
    ):
        raise ValueError("actor_slot must be a nonnegative integer")
    actor_identity = _canonical_expected_actor_identity(
        expected_actor_identity
    )
    payload: dict[str, object] = {
        "type": NATIVE_POLICY_WORLD_ACTION_CAPTURE_TYPE,
        "schema": NATIVE_POLICY_WORLD_ACTION_CAPTURE_SCHEMA,
        "version": NATIVE_POLICY_WORLD_ACTION_CAPTURE_VERSION,
        "contract_sha256": native_policy_world_action_capture_contract_sha256(),
        "phase": phase,
        "actor_slot": actor_slot,
        "expected_actor_identity": actor_identity,
    }
    if phase == "observe":
        if expected_candidate_generation_sha256 or action is not None:
            raise ValueError("observe cannot carry commit fields")
        return payload
    generation = _sha256(
        expected_candidate_generation_sha256,
        "expected_candidate_generation_sha256",
    )
    if not isinstance(action, LearnerArsenalAction):
        raise TypeError("commit requires LearnerArsenalAction")
    payload.update(
        {
            "expected_candidate_generation_sha256": generation,
            "selection": {
                "use_requested": bool(
                    _lane_scalar(action.use_requested, "use_requested")
                ),
                "block_interaction_trigger": int(
                    _lane_scalar(
                        action.block_interaction_trigger,
                        "block_interaction_trigger",
                    )
                ),
                "block_candidate_index": int(
                    _lane_scalar(
                        action.block_candidate_index,
                        "block_candidate_index",
                    )
                ),
                "recipe_candidate_index": int(
                    _lane_scalar(
                        action.recipe_candidate_index,
                        "recipe_candidate_index",
                    )
                ),
            },
        }
    )
    return payload


def decode_native_policy_world_action_surface(
    value: Mapping[str, Any],
    *,
    expected_bridge_sha256: str | None = None,
    expected_world_epoch: str | None = None,
    expected_world_identity: str | None = None,
    expected_environment_step: int | None = None,
    expected_actor_slot: int | None = None,
    expected_actor_identity: str | None = None,
    packed_recipes: PackedRecipeTable | None = None,
    recipe_encoder_params: RecipeCandidateEncoderParams | None = None,
) -> _DecodedSurface:
    """Decode an atomic observe capture into actor and privileged halves."""

    identity = _capture_identity(
        value,
        expected_phase="observe",
        expected_bridge_sha256=expected_bridge_sha256,
        expected_world_epoch=expected_world_epoch,
        expected_world_identity=expected_world_identity,
        expected_environment_step=expected_environment_step,
        expected_actor_slot=expected_actor_slot,
        expected_actor_identity=expected_actor_identity,
    )
    _capture_selection(value.get("selection"), expected_phase="observe")
    camera = _camera(_mapping(value, "camera"))
    blocks, block_bindings, trigger_available = _block_surface(
        _mapping(value, "blocks"),
        camera=camera,
        inventory=identity.inventory,
    )
    recipes, recipe_encoding, recipe_bindings, context = _recipe_surface(
        _mapping(value, "recipes"),
        native_recipe_table_identity=identity.recipe_table_identity_sha256,
        packed_recipes=packed_recipes,
        recipe_encoder_params=recipe_encoder_params,
    )
    use_binding = _use_surface(_mapping(value, "use"), camera=camera)
    _validate_use_inventory(use_binding, identity.inventory)
    snapshot = NativePolicyActionCandidateSnapshot(
        bridge_sha256=identity.bridge_sha256,
        world_epoch=identity.world_epoch,
        inventory_sha256=identity.inventory_identity_sha256,
        recipe_table_content_sha256=(
            "" if packed_recipes is None else packed_recipes.content_sha256
        ),
        block_candidate_mask=tuple(row is not None for row in block_bindings),
        block_candidates=block_bindings,
        recipe_candidate_mask=tuple(row is not None for row in recipe_bindings),
        recipe_candidates=recipe_bindings,
        block_trigger_available=trigger_available,
        use_available=use_binding is not None,
        use=use_binding,
        crafting_context=context,
        candidate_generation_sha256=identity.candidate_generation_sha256,
        actor_identity=identity.actor_identity,
        recipe_table_identity_sha256=(
            identity.recipe_table_identity_sha256
        ),
        world_tick=identity.world_tick,
        environment_step=identity.environment_step,
        actor_slot=identity.actor_slot,
        world_identity=identity.world,
    )
    return _DecodedSurface(
        identity=identity,
        surface=NativePolicyActionSurface(
            block_candidates=blocks,
            recipe_candidates=recipes,
            recipe_encoding=recipe_encoding,
            use_available=np.asarray(
                [use_binding is not None],
                dtype=np.bool_,
            ),
            block_trigger_available=np.asarray(
                [trigger_available],
                dtype=np.bool_,
            ),
            snapshot=snapshot,
        ),
    )


def decode_native_policy_world_action_commit(
    value: Mapping[str, Any],
    *,
    snapshot: NativePolicyActionCandidateSnapshot,
    info: Mapping[str, Any],
    packed_recipes: PackedRecipeTable | None = None,
    expected_action: LearnerArsenalAction | None = None,
) -> NativePolicyActionCommitEvidence:
    """Decode fresh commit evidence or one typed stale-generation rejection."""

    if not isinstance(snapshot, NativePolicyActionCandidateSnapshot):
        raise TypeError("snapshot must be NativePolicyActionCandidateSnapshot")
    identity = _capture_identity(
        value,
        expected_phase="commit",
        expected_bridge_sha256=snapshot.bridge_sha256,
        expected_world_epoch=snapshot.world_epoch,
        expected_world_identity=None,
        expected_environment_step=None,
        expected_actor_slot=snapshot.actor_slot,
        expected_actor_identity=None,
    )
    selection = _capture_selection(
        value.get("selection"),
        expected_phase="commit",
    )
    if expected_action is not None:
        expected_selection = (
            bool(_lane_scalar(expected_action.use_requested, "use_requested")),
            int(
                _lane_scalar(
                    expected_action.block_interaction_trigger,
                    "block_interaction_trigger",
                )
            ),
            int(
                _lane_scalar(
                    expected_action.block_candidate_index,
                    "block_candidate_index",
                )
            ),
            int(
                _lane_scalar(
                    expected_action.recipe_candidate_index,
                    "recipe_candidate_index",
                )
            ),
        )
        if selection != expected_selection:
            raise ValueError("commit selection echo differs from the request")
    session = NativeWorldVerbTransportSession.from_info(
        info,
        expected_bridge_sha256=snapshot.bridge_sha256,
    )
    common = {
        "session": session,
        "inventory": identity.inventory,
        "candidate_generation_sha256": (
            identity.candidate_generation_sha256
        ),
        "actor_identity": identity.actor_identity,
        "recipe_table_identity_sha256": (
            identity.recipe_table_identity_sha256
        ),
        "world_tick": identity.world_tick,
        "environment_step": identity.environment_step,
        "actor_slot": identity.actor_slot,
        "world_identity": identity.world,
    }
    if (
        identity.expected_candidate_generation_sha256
        != snapshot.candidate_generation_sha256
    ):
        return NativePolicyActionCommitEvidence(
            **common,
            unavailable_reason="native_commit_expected_generation_mismatch",
        )
    if identity.environment_step != snapshot.environment_step:
        return NativePolicyActionCommitEvidence(
            **common,
            unavailable_reason="native_candidate_environment_step_changed",
        )
    if not identity.expected_generation_matched:
        return NativePolicyActionCommitEvidence(
            **common,
            unavailable_reason="native_candidate_generation_changed",
        )

    camera = _camera(_mapping(value, "camera"))
    _blocks, _bindings, _triggers = _block_surface(
        _mapping(value, "blocks"),
        camera=camera,
        inventory=identity.inventory,
    )
    _recipes, _encoding, _recipe_bindings, context = _recipe_surface(
        _mapping(value, "recipes"),
        native_recipe_table_identity=identity.recipe_table_identity_sha256,
        packed_recipes=packed_recipes,
        recipe_encoder_params=None,
        commit_only=True,
    )
    use = _use_surface(_mapping(value, "use"), camera=camera)
    _validate_use_inventory(use, identity.inventory)
    block_cells_raw = value.get("commit_block_cells")
    interactions_raw = value.get("commit_item_interactions")
    block_cells = (
        None
        if block_cells_raw is None
        else NativeMutableBlockCells.from_response(_mapping_value(block_cells_raw))
    )
    interactions = (
        None
        if interactions_raw is None
        else NativeItemInteractionEvidence.from_response(
            _mapping_value(interactions_raw),
            expected_bridge_sha256=identity.bridge_sha256,
        )
    )
    needs_block_evidence = selection[0] or (
        selection[2] >= 0 and not selection[0]
    )
    partial_block_evidence = (block_cells is None) != (interactions is None)
    if partial_block_evidence or needs_block_evidence != (
        block_cells is not None and interactions is not None
    ):
        return NativePolicyActionCommitEvidence(
            **common,
            block_candidates=_bindings,
            use_binding=use,
            block_use=(
                None
                if use is None
                else _block_use_evidence(use, identity.bridge_sha256, camera)
            ),
            packed_recipes=packed_recipes,
            crafting_context=context,
            unavailable_reason="native_commit_evidence_incomplete",
        )
    return NativePolicyActionCommitEvidence(
        **common,
        block_cells=block_cells,
        item_interactions=interactions,
        block_use=(
            None
            if use is None
            else _block_use_evidence(use, identity.bridge_sha256, camera)
        ),
        packed_recipes=packed_recipes,
        crafting_context=context,
        block_candidates=_bindings,
        use_binding=use,
    )


def native_policy_world_action_capture_contract() -> dict[str, object]:
    """Describe the no-inference socket boundary and its v2 limitations."""

    return {
        "schema": NATIVE_POLICY_WORLD_ACTION_CAPTURE_SCHEMA,
        "version": NATIVE_POLICY_WORLD_ACTION_CAPTURE_VERSION,
        "request_type": NATIVE_POLICY_WORLD_ACTION_CAPTURE_TYPE,
        "phases": ["observe", "commit"],
        "atomicity": "one_World_thread_capture_per_response",
        "identity": [
            "bridge_sha256",
            "world",
            "world_epoch",
            "world_tick",
            "environment_step",
            "actor_slot",
            "actor_identity",
            "inventory_identity_sha256",
            "recipe_table_identity_sha256",
            "candidate_generation_sha256",
        ],
        "recipe_table_identity": (
            "NativeCraftingCatalogEvidence.semantic_sha256_equal_to_"
            "PackedRecipeTable.source_sha256"
        ),
        "candidate_capacity": {
            "block": ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
            "recipe": ACTOR_RECIPE_CANDIDATE_CAPACITY,
        },
        "observe": {
            "request_actor": "explicit_slot_plus_reset_negotiated_UUID",
            "camera": [
                "actor_position",
                "eye_position",
                "direction",
                "maximum_distance",
                "raw_hit_cell",
                "canonical_action_cell",
                "block_face",
            ],
            "block_candidate": [
                "visible_cell",
                "action_cell",
                "visible_relative_position",
                "affordance_tags",
                "gather_type_index",
                "required_tool_quality",
                "per_trigger_privileged_binding",
            ],
            "recipe_candidate": (
                "complete_actor_legal_native_index_plus_packed_policy_row"
            ),
            "use": "one_exact_current_block_target_edge",
        },
        "commit": {
            "request": (
                "actor_slot_plus_UUID_plus_observed_generation_plus_"
                "bounded_policy_selection"
            ),
            "selection_echo": "exact_bounded_request_selection",
            "fresh": [
                "inventory",
                "camera",
                "block_cells",
                "item_interaction_roots",
                "use_binding",
                "recipe_context",
            ],
            "stale": "typed_rejection_never_target_reconstruction",
        },
        "candidate_generation": {
            "algorithm": "sha256_of_canonical_length_prefixed_fields",
            "includes": [
                "bridge_sha256",
                "world",
                "world_epoch",
                "actor_slot",
                "actor_identity",
                "inventory_identity_and_payload",
                "recipe_table_identity",
                "camera_actor_eye_direction_reach_raw_action_face",
                "ordered_block_candidates_and_bindings",
                "ordered_recipe_candidates_and_legality",
                "recipe_bench_knowledge_memory_manager_queue_context",
                "use_binding",
                "availability_reasons_counts_and_overflow",
            ],
            "excludes": [
                "phase",
                "expected_candidate_generation_sha256",
                "expected_generation_matched",
                "world_tick",
                "environment_step",
            ],
            "rationale": (
                "volatile_provenance_is_checked_separately_and_must_not_"
                "invalidate_byte_identical_candidates_at_30_tps"
            ),
        },
        "actor_ownership": (
            "request_and_response_slot_UUID_equal_reset_negotiated_actor"
        ),
        "v2_fail_closed": {
            "place": (
                "rotation_zero_subset_requires_exact_empty_destination_and_"
                "Region_placed_geometry_identity_not_yet_joined"
            ),
            "recipe": (
                "complete_actor_bench_knowledge_memory_context_required"
            ),
            "overflow": "complete_surface_unavailable",
            "partial_field": "complete_surface_unavailable",
        },
        "policy_excludes": [
            "absolute_cells",
            "native_String_IDs",
            "inventory_container_and_slot",
            "candidate_generation",
            "recipe_global_index",
        ],
    }


def native_policy_world_action_capture_contract_sha256() -> str:
    payload = json.dumps(
        native_policy_world_action_capture_contract(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _capture_selection(
    raw: object,
    *,
    expected_phase: str,
) -> tuple[bool, int, int, int] | None:
    if expected_phase == "observe":
        if raw is not None:
            raise ValueError("observe capture cannot echo a selection")
        return None
    value = _mapping_value(raw)
    selection = (
        _boolean(value, "use_requested"),
        _integer(value, "block_interaction_trigger", minimum=0, maximum=2),
        _integer(
            value,
            "block_candidate_index",
            minimum=-1,
            maximum=ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY - 1,
        ),
        _integer(
            value,
            "recipe_candidate_index",
            minimum=-1,
            maximum=ACTOR_RECIPE_CANDIDATE_CAPACITY - 1,
        ),
    )
    use, trigger, block, recipe = selection
    selected_count = int(use) + int(block >= 0 and not use) + int(recipe >= 0)
    if selected_count != 1:
        raise ValueError("commit selection must name exactly one World verb")
    if block >= 0 and not use and trigger not in (1, 2):
        raise ValueError("block selection requires Primary or Secondary")
    return selection


def _capture_identity(
    value: Mapping[str, Any],
    *,
    expected_phase: str,
    expected_bridge_sha256: str | None,
    expected_world_epoch: str | None,
    expected_world_identity: str | None,
    expected_environment_step: int | None,
    expected_actor_slot: int | None,
    expected_actor_identity: str | None,
) -> _CaptureIdentity:
    if not isinstance(value, Mapping):
        raise TypeError("native World-action capture must be an object")
    _equal(value, "type", NATIVE_POLICY_WORLD_ACTION_CAPTURE_TYPE)
    _equal(value, "schema", NATIVE_POLICY_WORLD_ACTION_CAPTURE_SCHEMA)
    _equal(value, "version", NATIVE_POLICY_WORLD_ACTION_CAPTURE_VERSION)
    _equal(
        value,
        "contract_sha256",
        native_policy_world_action_capture_contract_sha256(),
    )
    phase = _string(value, "phase")
    if phase != expected_phase:
        raise ValueError("native World-action capture returned the wrong phase")
    bridge = _sha256(_string(value, "bridge_sha256"), "bridge_sha256")
    if expected_bridge_sha256 is not None and bridge != _sha256(
        expected_bridge_sha256,
        "expected_bridge_sha256",
    ):
        raise ValueError("native World-action capture came from another bridge")
    epoch = _string(value, "world_epoch")
    if expected_world_epoch is not None and epoch != expected_world_epoch:
        raise ValueError("native World-action capture came from another World epoch")
    world = _string(value, "world")
    if expected_world_identity is not None and world != expected_world_identity:
        raise ValueError("native World-action capture came from another World")
    inventory = _mapping(value, "inventory")
    parsed_inventory = parse_native_inventory_frame(inventory)
    inventory_identity = _sha256(
        _string(value, "inventory_identity_sha256"),
        "inventory_identity_sha256",
    )
    if native_inventory_semantic_sha256(inventory) != inventory_identity:
        raise ValueError("native inventory identity does not match its payload")
    expected_generation = _string(
        value,
        "expected_candidate_generation_sha256",
        allow_empty=True,
    )
    matched = _boolean(value, "expected_generation_matched")
    generation = _sha256(
        _string(value, "candidate_generation_sha256"),
        "candidate_generation_sha256",
    )
    if phase == "observe":
        if expected_generation or not matched:
            raise ValueError("observe generation fields are inconsistent")
    else:
        _sha256(
            expected_generation,
            "expected_candidate_generation_sha256",
        )
        if matched != (expected_generation == generation):
            raise ValueError("commit generation comparison is inconsistent")
    if not parsed_inventory["available"] and any(
        _boolean(_mapping(value, name), "available")
        for name in ("blocks", "recipes", "use")
    ):
        raise ValueError("unavailable inventory cannot publish World actions")
    environment_step = _integer(value, "environment_step", minimum=0)
    if (
        expected_environment_step is not None
        and environment_step != expected_environment_step
    ):
        raise ValueError(
            "native World-action capture came from another environment step"
        )
    actor_identity = _canonical_expected_actor_identity(
        _string(value, "actor_identity")
    )
    actor_slot = _integer(value, "actor_slot", minimum=0)
    if expected_actor_slot is not None and actor_slot != expected_actor_slot:
        raise ValueError(
            "native World-action capture came from another policy actor slot"
        )
    if (
        expected_actor_identity is not None
        and actor_identity
        != _canonical_expected_actor_identity(expected_actor_identity)
    ):
        raise ValueError(
            "native World-action capture came from another policy actor"
        )
    return _CaptureIdentity(
        phase=phase,
        bridge_sha256=bridge,
        world=world,
        world_epoch=epoch,
        world_tick=_integer(value, "world_tick", minimum=0),
        environment_step=environment_step,
        actor_slot=actor_slot,
        actor_identity=actor_identity,
        inventory_identity_sha256=inventory_identity,
        recipe_table_identity_sha256=_sha256(
            _string(value, "recipe_table_identity_sha256"),
            "recipe_table_identity_sha256",
        ),
        candidate_generation_sha256=generation,
        expected_candidate_generation_sha256=expected_generation,
        expected_generation_matched=matched,
        inventory=inventory,
    )


def _camera(value: Mapping[str, Any]) -> dict[str, object]:
    available = _boolean(value, "available")
    reason = _string(value, "unavailable_reason", allow_empty=True)
    eye = _float_vector(value, "eye_position", 3 if available else 0)
    actor = _float_vector(value, "actor_position", 3 if available else 0)
    direction = _float_vector(value, "direction", 3 if available else 0)
    raw = _int_vector(value, "raw_hit_cell", 3 if available else 0)
    action = _int_vector(value, "canonical_action_cell", 3 if available else 0)
    maximum = _number(value, "maximum_distance", minimum=0.0)
    face = _integer(value, "block_face", minimum=0, maximum=6)
    if available:
        if (
            reason
            or maximum <= 0.0
            or face == 0
            or not np.isclose(
                np.linalg.norm(np.asarray(direction, dtype=np.float64)),
                1.0,
                rtol=0.0,
                atol=1.0e-5,
            )
            or not _ray_hits_cell(eye, direction, raw, maximum)
        ):
            raise ValueError("available camera row is incomplete")
        derived_face = _ray_entry_face(eye, direction, raw, maximum)
        if derived_face is None:
            raise ValueError("camera ray has an ambiguous block entry face")
        if face != derived_face:
            raise ValueError("camera block face differs from its ray traversal")
    elif not reason or maximum != 0.0 or face != 0:
        raise ValueError("unavailable camera row must be structurally empty")
    return {
        "available": available,
        "unavailable_reason": reason,
        "eye_position": eye,
        "actor_position": actor,
        "direction": direction,
        "maximum_distance": maximum,
        "raw_hit_cell": raw,
        "canonical_action_cell": action,
        "block_face": face,
    }


def _block_surface(
    value: Mapping[str, Any],
    *,
    camera: Mapping[str, object],
    inventory: Mapping[str, object],
):
    capacity = ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY
    available = _boolean(value, "available")
    reason = _string(value, "unavailable_reason", allow_empty=True)
    source_count = _integer(value, "source_count", minimum=0)
    emitted_count = _integer(value, "emitted_count", minimum=0, maximum=capacity)
    overflow = _boolean(value, "capacity_exceeded")
    if overflow != (source_count > capacity):
        raise ValueError("block candidate overflow flag is inconsistent")
    raw_candidates = _array(value, "candidates")
    if emitted_count != len(raw_candidates):
        raise ValueError("block candidate emitted count is inconsistent")
    if available:
        if (
            reason
            or overflow
            or source_count != emitted_count
            or not camera["available"]
        ):
            raise ValueError("available block surface is incomplete")
    elif not reason or emitted_count or raw_candidates:
        raise ValueError("unavailable block surface must fail closed")

    source = _empty_block_candidate_source()
    masks = np.zeros((capacity,), dtype=np.bool_)
    visible = np.zeros((capacity, 3), dtype=np.float32)
    action = np.zeros((capacity, 3), dtype=np.float32)
    relative = np.zeros((capacity, 3), dtype=np.float32)
    tags = np.zeros((capacity,), dtype=np.uint16)
    gather = np.zeros((capacity,), dtype=np.uint8)
    quality = np.zeros((capacity,), dtype=np.int16)
    bindings: list[NativeBlockCandidateBinding | None] = [None] * capacity
    primary_count = 0
    secondary_count = 0
    raw_primary_count = 0
    raw_secondary_count = 0
    for expected_slot, raw in enumerate(raw_candidates):
        row = _mapping_value(raw)
        slot = _integer(row, "slot", minimum=0, maximum=capacity - 1)
        if slot != expected_slot:
            raise ValueError("block candidates must be contiguous and ordered")
        visible_cell = _int_vector(row, "visible_cell", 3)
        action_cell = _int_vector(row, "action_cell", 3)
        relative_position = _float_vector(
            row,
            "visible_relative_position",
            3,
        )
        expected_relative = tuple(
            float(cell) + 0.5 - float(origin)
            for cell, origin in zip(
                visible_cell,
                camera["actor_position"],
                strict=True,
            )
        )
        if not np.allclose(
            relative_position,
            expected_relative,
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError(
                "block candidate relative position differs from actor and cell"
            )
        action_center = np.asarray(action_cell, dtype=np.float64) + 0.5
        if np.linalg.norm(
            action_center
            - np.asarray(camera["eye_position"], dtype=np.float64)
        ) > (float(camera["maximum_distance"]) + 1.0e-5):
            raise ValueError("block candidate exceeds authored reach")
        semantics = NativeMutableBlockRow.from_response(
            _mapping(row, "semantics")
        )
        if (
            not semantics.block_present
            or not semantics.semantic_key_valid
            or not semantics.affordance_valid
        ):
            raise ValueError("block candidate semantics are incomplete")
        raw_primary = row.get("primary")
        raw_secondary = row.get("secondary")
        raw_primary_count += int(raw_primary is not None)
        raw_secondary_count += int(raw_secondary is not None)
        primary = _optional_block_binding(
            raw_primary,
            expected_verb="break_block",
            expected_interaction_type=0,
            semantics=semantics,
        )
        secondary = _optional_block_binding(
            raw_secondary,
            expected_verb="place_block",
            expected_interaction_type=1,
            semantics=semantics,
        )
        if primary is not None:
            if primary.target != action_cell:
                raise ValueError(
                    "Break target differs from its canonical action cell"
                )
            candidate_direction = (
                np.asarray(visible_cell, dtype=np.float64)
                + 0.5
                - np.asarray(camera["eye_position"], dtype=np.float64)
            )
            candidate_distance = float(np.linalg.norm(candidate_direction))
            if candidate_distance <= 1.0e-9:
                raise ValueError("Break candidate has no camera ray")
            candidate_face = _ray_entry_face(
                camera["eye_position"],
                candidate_direction / candidate_distance,
                visible_cell,
                candidate_distance + 1.0e-5,
            )
            if candidate_face is None or primary.block_face != candidate_face:
                raise ValueError(
                    "Break face differs from its candidate camera ray"
                )
            if visible_cell == tuple(camera["raw_hit_cell"]) and (
                action_cell != tuple(camera["canonical_action_cell"])
                or primary.block_face != int(camera["block_face"])
            ):
                raise ValueError(
                    "camera-hit Break binding differs from camera evidence"
                )
            source_reason = _inventory_binding_reason(inventory, primary)
            if source_reason:
                raise ValueError(
                    f"block candidate source inventory differs: {source_reason}"
                )
        if primary is None and secondary is None:
            # A native row can be complete while every trigger is locally
            # unrepresentable (currently Place without an exact rotation-zero
            # destination plus its Region placed-geometry identity).
            # Keep its slot masked rather than publishing a partial binding.
            continue
        masks[slot] = True
        visible[slot] = visible_cell
        action[slot] = action_cell
        relative[slot] = relative_position
        tags[slot] = semantics.affordance_tags
        gather[slot] = semantics.gather_type_index
        quality[slot] = semantics.required_tool_quality
        bindings[slot] = NativeBlockCandidateBinding(primary, secondary)
        primary_count += int(primary is not None)
        secondary_count += int(secondary is not None)

    primary_reason = _string(
        value,
        "primary_unavailable_reason",
        allow_empty=True,
    )
    secondary_reason = _string(
        value,
        "secondary_unavailable_reason",
        allow_empty=True,
    )
    active_bindings = tuple(row for row in bindings if row is not None)
    primary_rectangular = bool(active_bindings) and all(
        row.primary is not None for row in active_bindings
    )
    secondary_rectangular = bool(active_bindings) and all(
        row.secondary is not None for row in active_bindings
    )
    trigger_available = (
        bool(available and primary_rectangular),
        bool(available and secondary_rectangular),
    )
    if active_bindings and (
        not any(trigger_available) or len(active_bindings) != source_count
    ):
        masks[:] = False
        bindings = [None] * capacity
        trigger_available = (False, False)
    if (not primary_reason) != bool(available and raw_primary_count):
        raise ValueError("Primary availability reason is inconsistent")
    if (not secondary_reason) != bool(available and raw_secondary_count):
        raise ValueError("Secondary availability reason is inconsistent")
    if available:
        source = ActorBlockActionCandidates(
            available=jnp.asarray([[True]], dtype=jnp.bool_),
            capacity_exceeded=jnp.asarray([[False]], dtype=jnp.bool_),
            diagnostics=jnp.asarray([[0]], dtype=jnp.uint32),
            candidate_mask=jnp.asarray(masks[None, None], dtype=jnp.bool_),
            visible_position=jnp.asarray(visible[None, None]),
            action_position=jnp.asarray(action[None, None]),
            visible_relative_position=jnp.asarray(relative[None, None]),
            affordance_tags=jnp.asarray(tags[None, None]),
            gather_type_index=jnp.asarray(gather[None, None]),
            required_tool_quality=jnp.asarray(quality[None, None]),
            provenance=jnp.zeros((1, 1, capacity), dtype=jnp.uint32),
        )
        policy = encode_block_action_candidate_policy_view(
            source,
            float(camera["maximum_distance"]),
        )
    else:
        policy = empty_block_action_candidate_policy_view(1, capacity)
        bindings = [None] * capacity
        trigger_available = (False, False)
    return policy, tuple(bindings), trigger_available


def _optional_block_binding(
    raw: object,
    *,
    expected_verb: str,
    expected_interaction_type: int,
    semantics: NativeMutableBlockRow,
) -> NativeBlockCommitBinding | None:
    if raw is None:
        return None
    value = _mapping_value(raw)
    _equal(value, "verb", expected_verb)
    interaction_type = _integer(
        value,
        "interaction_type",
        minimum=0,
        maximum=5,
    )
    if interaction_type != expected_interaction_type:
        raise ValueError("block binding interaction type is inconsistent")
    semantic_bytes = _byte_payload(value, "expected_semantic_key_sha256", 32)
    expected_words = tuple(
        int(row)
        for row in np.frombuffer(semantic_bytes, dtype=">u4").astype(np.uint32)
    )
    current_words = tuple(int(row) for row in semantics.semantic_key)
    if expected_verb == "break_block" and expected_words != current_words:
        raise ValueError("Break binding and candidate semantics differ")
    expected_block_id = _string(value, "expected_block_id")
    expected_tool = _string(
        value,
        "expected_interaction_tool_id",
        allow_empty=True,
    )
    if (
        expected_verb == "break_block"
        and expected_block_id != semantics.block_asset_id
    ):
        raise ValueError("Break binding and candidate block identity differ")
    if expected_verb == "break_block" and not expected_tool:
        raise ValueError("Break binding lacks its authored tool category")
    source_container = _string(value, "source_container")
    if expected_verb == "break_block" and source_container != "hotbar":
        raise ValueError("v1 Break source must be the native hotbar")
    rotation_applicable = _boolean(value, "rotation_applicable")
    rotation = _int_vector(value, "rotation", 3)
    if expected_verb == "place_block" and not rotation_applicable:
        # The current policy deliberately represents the native rotation-zero
        # placement subset.  It still requires an exact adjacent empty cell
        # and the matching Region placed-geometry identity before opening.
        return None
    if expected_verb == "place_block":
        # Capture v1 does not yet carry that destination/geometry join.  Do not
        # turn a server-provided rotation into a structurally valid Place row.
        return None
    binding_semantic_sha256 = _string(
        value,
        "binding_semantic_sha256",
    )
    _sha256(binding_semantic_sha256, "binding_semantic_sha256")
    return NativeBlockCommitBinding(
        target=_int_vector(value, "target", 3),
        block_face=_integer(value, "block_face", minimum=1, maximum=6),
        rotation=rotation,
        expected_block_id=expected_block_id,
        expected_semantic_key=(
            None if expected_block_id == "Empty" else expected_words
        ),
        source_container=source_container,
        source_slot=_integer(value, "source_slot", minimum=0),
        expected_source_quantity=_integer(
            value,
            "source_quantity",
            minimum=1,
        ),
        source_item_id=_string(value, "source_item_id"),
        source_block_id=_string(value, "source_block_id", allow_empty=True),
        expected_interaction_tool_id=expected_tool,
        interaction_type=interaction_type,
        interaction_id=_string(value, "interaction_id"),
        block_interaction_id=_string(
            value,
            "block_interaction_id",
            allow_empty=True,
        ),
        rotation_applicable=rotation_applicable,
        semantic_sha256=binding_semantic_sha256,
    )


def _recipe_surface(
    value: Mapping[str, Any],
    *,
    native_recipe_table_identity: str,
    packed_recipes: PackedRecipeTable | None,
    recipe_encoder_params: RecipeCandidateEncoderParams | None,
    commit_only: bool = False,
):
    capacity = ACTOR_RECIPE_CANDIDATE_CAPACITY
    available = _boolean(value, "available")
    reason = _string(value, "unavailable_reason", allow_empty=True)
    source_count = _integer(value, "source_count", minimum=0)
    emitted_count = _integer(value, "emitted_count", minimum=0, maximum=capacity)
    overflow = _boolean(value, "capacity_exceeded")
    candidates = _array(value, "candidates")
    if overflow != (source_count > capacity) or emitted_count != len(candidates):
        raise ValueError("recipe candidate counts are inconsistent")
    if available:
        if reason or overflow or source_count != emitted_count:
            raise ValueError("available recipe surface is incomplete")
    elif not reason or emitted_count or candidates:
        raise ValueError("unavailable recipe surface must fail closed")

    memories_level = _integer(value, "memories_level", minimum=0)
    knowledge_available = _boolean(value, "knowledge_available")
    bench_context_available = _boolean(value, "bench_context_available")
    manager_available = _boolean(value, "manager_available")
    manager_has_bench = _boolean(value, "manager_has_bench")
    manager_queue_size = _integer(value, "manager_queue_size", minimum=0)
    manager_queue_recipe_id = _string(
        value,
        "manager_queue_recipe_id",
        allow_empty=True,
    )
    manager_queue_identity = _string(
        value,
        "manager_queue_identity_sha256",
        allow_empty=True,
    )
    if manager_queue_identity:
        _sha256(manager_queue_identity, "manager_queue_identity_sha256")
    queue_fields_empty = (
        manager_queue_recipe_id == "" and manager_queue_identity == ""
    )
    queue_fields_full = (
        manager_queue_recipe_id != "" and manager_queue_identity != ""
    )
    if not (
        (manager_queue_size == 0 and queue_fields_empty)
        or (manager_queue_size > 0 and (queue_fields_empty or queue_fields_full))
    ):
        raise ValueError("recipe manager queue identity is inconsistent")
    raw_context = value.get("crafting_context")
    context = _crafting_context(raw_context)
    if raw_context is not None and context is None:
        raise ValueError("recipe crafting context is malformed")
    if bench_context_available != (context is not None):
        raise ValueError("recipe bench context availability is inconsistent")
    if available and (
        not knowledge_available
        or not bench_context_available
        or not manager_available
        or manager_has_bench
        or manager_queue_size != 0
        or context is None
    ):
        raise ValueError("available recipe context is incomplete")
    host_ready = (
        available
        and context is not None
        and packed_recipes is not None
        and packed_recipes.source_sha256.upper()
        == native_recipe_table_identity.upper()
    )
    if not host_ready:
        return (
            empty_recipe_candidate_policy_view(1),
            _empty_recipe_encoding(),
            (None,) * capacity,
            None,
        )
    assert packed_recipes is not None
    indexes: list[int] = []
    native_indexes: list[int] = []
    host_index_by_id = {
        recipe_id: index
        for index, recipe_id in enumerate(packed_recipes.recipe_ids)
    }
    bindings: list[NativeRecipeCandidateBinding | None] = [None] * capacity
    for expected_slot, raw in enumerate(candidates):
        row = _mapping_value(raw)
        slot = _integer(row, "slot", minimum=0, maximum=capacity - 1)
        if slot != expected_slot:
            raise ValueError("recipe candidates must be contiguous and ordered")
        native_index = _integer(
            row,
            "native_recipe_index",
            minimum=0,
            maximum=RECIPE_TABLE_CAPACITY - 1,
        )
        recipe_id = _string(row, "recipe_id")
        digest = _byte_payload(row, "recipe_id_sha256", 32)
        if hashlib.sha256(recipe_id.encode("utf-8")).digest() != digest:
            raise ValueError("recipe String ID hash is inconsistent")
        recipe = NativeCraftingRecipe.from_response(_mapping(row, "recipe"))
        if recipe.recipe_id != recipe_id:
            raise ValueError("recipe candidate payload identity is inconsistent")
        if not all(
            _boolean(row, name)
            for name in (
                "knowledge_satisfied",
                "memory_satisfied",
                "bench_satisfied",
            )
        ):
            raise ValueError("published recipe candidate is not actor-legal")
        knowledge_key = _string(
            row,
            "knowledge_key",
            allow_empty=True,
        )
        expected_knowledge_key = (
            recipe.primary_output_item_asset_id
            if recipe.knowledge_required
            else ""
        )
        if knowledge_key != expected_knowledge_key:
            raise ValueError("recipe candidate knowledge identity changed")
        if (
            recipe.required_memories_level > 1
            and memories_level < recipe.required_memories_level
        ):
            raise ValueError("recipe candidate exceeds the captured memory level")
        if not _recipe_bench_matches_context(recipe, context):
            raise ValueError("recipe candidate differs from captured bench context")
        host_index = host_index_by_id.get(recipe_id)
        if host_index is None:
            # The native legal set cannot be partially projected through a
            # smaller host table; one omitted row changes candidate ordering.
            return (
                empty_recipe_candidate_policy_view(1),
                _empty_recipe_encoding(),
                (None,) * capacity,
                None,
            )
        indexes.append(host_index)
        native_indexes.append(native_index)
        bindings[slot] = NativeRecipeCandidateBinding(
            recipe_index=host_index,
            recipe_id_hash=identity_sha256_words(recipe_id),
            semantic_sha256=_sha256(
                _string(row, "candidate_semantic_sha256"),
                "candidate_semantic_sha256",
            ),
        )
    if native_indexes != sorted(set(native_indexes)):
        raise ValueError("recipe candidates are not in canonical table order")
    if indexes != sorted(set(indexes)):
        # The policy producer compacts in packed-table order.  Reordering only
        # the actor view would make the bounded commit slot name another
        # native candidate, so a divergent catalog order fails closed.
        return (
            empty_recipe_candidate_policy_view(1),
            _empty_recipe_encoding(),
            (None,) * capacity,
            None,
        )
    if commit_only:
        return (
            empty_recipe_candidate_policy_view(1),
            _empty_recipe_encoding(),
            tuple(bindings),
            context,
        )
    if recipe_encoder_params is None:
        return (
            empty_recipe_candidate_policy_view(1),
            _empty_recipe_encoding(),
            (None,) * capacity,
            None,
        )
    legal = np.zeros((1, 1, RECIPE_TABLE_CAPACITY), dtype=np.bool_)
    legal[0, 0, indexes] = True
    source = produce_actor_recipe_candidates_from_legal_mask(
        packed_recipes.table,
        jnp.asarray(legal),
        catalog_available=True,
        legal_evidence_available=jnp.asarray([[True]], dtype=jnp.bool_),
    )
    policy = encode_recipe_candidate_policy_view(source)
    encoding = encode_recipe_candidates(recipe_encoder_params, policy)
    return policy, encoding, tuple(bindings), context


def _crafting_context(raw: object) -> NativeCraftingExecutionContext | None:
    if not isinstance(raw, Mapping):
        return None
    kind = _string(raw, "kind")
    if kind == "fieldcraft":
        return NativeCraftingExecutionContext.fieldcraft()
    if kind != "bench":
        return None
    try:
        return NativeCraftingExecutionContext.ordinary_bench(
            position=_int_vector(raw, "position", 3),
            block_id=_string(raw, "block_id"),
            bench_id=_string(raw, "bench_id"),
            tier=_integer(raw, "tier", minimum=1),
        )
    except (TypeError, ValueError):
        return None


def _use_surface(
    value: Mapping[str, Any],
    *,
    camera: Mapping[str, object],
) -> NativeUseCandidateBinding | None:
    available = _boolean(value, "available")
    reason = _string(value, "unavailable_reason", allow_empty=True)
    raw_binding = value.get("binding")
    if not available:
        if not reason or raw_binding is not None:
            raise ValueError("unavailable Use surface must fail closed")
        return None
    if reason or not camera["available"] or not isinstance(raw_binding, Mapping):
        raise ValueError("available Use surface is incomplete")
    value = raw_binding
    _equal(value, "verb", "use")
    interaction_type = _integer(
        value,
        "interaction_type",
        minimum=0,
        maximum=5,
    )
    if interaction_type != 5:
        raise ValueError("Use binding carries the wrong interaction type")
    rotation_applicable = _boolean(value, "rotation_applicable")
    rotation = _int_vector(value, "rotation", 3)
    if not rotation_applicable and any(rotation):
        raise ValueError("non-applicable Use rotation must be zero")
    semantic_bytes = _byte_payload(
        value,
        "expected_semantic_key_sha256",
        32,
    )
    semantic_words = tuple(
        int(row)
        for row in np.frombuffer(semantic_bytes, dtype=">u4").astype(np.uint32)
    )
    target = _int_vector(value, "target", 3)
    if target != tuple(camera["canonical_action_cell"]):
        raise ValueError(
            "Use target differs from the authoritative base-block action cell"
        )
    block_face = _integer(value, "block_face", minimum=1, maximum=6)
    if block_face != int(camera["block_face"]):
        raise ValueError("Use face differs from the authoritative camera hit")
    source_container = _string(value, "source_container")
    source_slot = _integer(value, "source_slot", minimum=-1)
    source_quantity = _integer(value, "source_quantity", minimum=0)
    source_item_id = _string(value, "source_item_id")
    if source_container not in {"unarmed", "interaction_context"}:
        raise ValueError("Use source is not a native block-Use source")
    resolved_source_container = _string(
        value,
        "resolved_source_container",
    )
    if resolved_source_container not in {"hotbar", "tools"}:
        raise ValueError("Use resolved source container is unsupported")
    if (
        source_container == "unarmed"
        and (source_item_id != "Empty" or source_slot != -1 or source_quantity != 0)
    ) or (
        source_container == "interaction_context"
        and (source_item_id == "Empty" or source_slot < 0 or source_quantity <= 0)
    ):
        raise ValueError("Use source binding is inconsistent")
    return NativeUseCandidateBinding(
        target=target,
        block_face=block_face,
        rotation=rotation,
        interaction_id=_string(value, "interaction_id"),
        block_interaction_id=_string(value, "block_interaction_id"),
        item_id=source_item_id,
        source_container=source_container,
        source_slot=source_slot,
        source_quantity=source_quantity,
        expected_block_id=_string(value, "expected_block_id"),
        interaction_type=interaction_type,
        maximum_distance=float(camera["maximum_distance"]),
        expected_semantic_key=semantic_words,
        resolved_source_container=resolved_source_container,
        semantic_sha256=_sha256(
            _string(value, "binding_semantic_sha256"),
            "binding_semantic_sha256",
        ),
    )


def _block_use_evidence(
    binding: NativeUseCandidateBinding,
    bridge_sha256: str,
    camera: Mapping[str, object],
) -> NativeBlockUseEvidence:
    return NativeBlockUseEvidence(
        bridge_sha256=bridge_sha256,
        available=True,
        available_interaction_id=binding.interaction_id,
        available_block_interaction_id=binding.block_interaction_id,
        available_item_id=binding.item_id,
        available_source_container=binding.source_container,
        available_source_slot=binding.source_slot,
        available_source_quantity=binding.source_quantity,
        available_target=binding.target,
        available_maximum_distance=float(camera["maximum_distance"]),
        unavailable_reason="",
        requested=False,
        accepted=False,
        started=False,
        active=False,
        finished=False,
        failed=False,
        reject_reason="",
        interaction_id="",
        block_interaction_id="",
        target=None,
        maximum_distance=0.0,
        chain_start_world_tick=-1,
        chain_finish_world_tick=-1,
    )


def _recipe_bench_matches_context(
    recipe: NativeCraftingRecipe,
    context: NativeCraftingExecutionContext | None,
) -> bool:
    """Mirror CraftingManager's any-requirement bench predicate exactly."""

    if context is None:
        return False
    return any(
        requirement.bench_type == context.expected_bench_type
        and requirement.bench_id == context.expected_bench_id
        and requirement.required_tier_level <= context.expected_bench_tier
        for requirement in recipe.bench_requirements
    )


def _inventory_binding_reason(
    inventory: Mapping[str, object],
    binding: NativeBlockCommitBinding,
) -> str:
    parsed = parse_native_inventory_frame(inventory)
    containers = {row["name"]: row for row in parsed["containers"]}
    container = containers.get(binding.source_container)
    if container is None or not container["available"]:
        return "source_container_unavailable"
    slots = {row["slot"]: row for row in container["occupied_slots"]}
    slot = slots.get(binding.source_slot)
    if slot is None:
        return "source_slot_empty"
    if slot["item_id"] != binding.source_item_id:
        return "source_item_changed"
    if slot["quantity"] != binding.expected_source_quantity:
        return "source_quantity_changed"
    return ""


def _validate_use_inventory(
    binding: NativeUseCandidateBinding | None,
    inventory: Mapping[str, object],
) -> None:
    if binding is None:
        return
    parsed = parse_native_inventory_frame(inventory)
    source_container = binding.resolved_source_container
    container = next(
        (row for row in parsed["containers"] if row["name"] == source_container),
        None,
    )
    if container is None or not container["available"]:
        raise ValueError("Use source inventory is unavailable")
    active_slot = parsed["active_slots"][source_container]
    slots = {row["slot"]: row for row in container["occupied_slots"]}
    if binding.source_container == "unarmed":
        if (
            binding.item_id != "Empty"
            or binding.source_slot != -1
            or binding.source_quantity != 0
            or active_slot in slots
        ):
            raise ValueError(
                "unarmed Use disagrees with active resolved inventory"
            )
        return
    if binding.source_container != "interaction_context":
        raise ValueError("Use interaction-context source is inconsistent")
    if active_slot != binding.source_slot:
        raise ValueError(
            "Use source differs from the active resolved inventory slot"
        )
    slot = slots.get(binding.source_slot)
    if slot is None or (
        slot["item_id"] != binding.item_id
        or slot["quantity"] != binding.source_quantity
    ):
        raise ValueError("Use source differs from active inventory evidence")


def _callable_key(value: BridgeRequest) -> tuple[object, object]:
    return (
        getattr(value, "__self__", None),
        getattr(value, "__func__", value),
    )


def _canonical_expected_actor_identity(value: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or "\0" in value
    ):
        raise ValueError("expected_actor_identity must be a canonical string")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as error:
        raise ValueError(
            "expected_actor_identity must be a canonical UUID"
        ) from error
    canonical = str(parsed)
    if value != canonical:
        raise ValueError("expected_actor_identity must be canonical UUID text")
    return canonical


def _ray_hits_cell(
    origin: Sequence[float],
    direction: Sequence[float],
    cell: Sequence[int],
    maximum_distance: float,
) -> bool:
    """Return whether the authored camera ray intersects the unit cell."""

    ray_origin = np.asarray(origin, dtype=np.float64)
    ray_direction = np.asarray(direction, dtype=np.float64)
    lower = np.asarray(cell, dtype=np.float64)
    upper = lower + 1.0
    t_min = 0.0
    t_max = float(maximum_distance)
    for axis in range(3):
        if abs(ray_direction[axis]) <= 1.0e-12:
            if ray_origin[axis] < lower[axis] or ray_origin[axis] > upper[axis]:
                return False
            continue
        first = (lower[axis] - ray_origin[axis]) / ray_direction[axis]
        second = (upper[axis] - ray_origin[axis]) / ray_direction[axis]
        near, far = min(first, second), max(first, second)
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max:
            return False
    return t_max >= 0.0


def _ray_entry_face(
    origin: Sequence[float],
    direction: Sequence[float],
    cell: Sequence[int],
    maximum_distance: float,
) -> int | None:
    """Derive the unique protocol face entered by a BlockIterator-style ray.

    Edge/corner entry crosses more than one voxel boundary at the same
    parameter. Hytale's iterator changes every tied axis in one transition,
    so no single protocol face is source-authoritative; v1 fails that row
    closed instead of inventing a tie-break.
    """

    ray_origin = np.asarray(origin, dtype=np.float64)
    ray_direction = np.asarray(direction, dtype=np.float64)
    lower = np.asarray(cell, dtype=np.float64)
    upper = lower + 1.0
    near_values = np.full((3,), -np.inf, dtype=np.float64)
    far = float(maximum_distance)
    for axis in range(3):
        component = ray_direction[axis]
        if abs(component) <= 1.0e-12:
            if ray_origin[axis] < lower[axis] or ray_origin[axis] > upper[axis]:
                return None
            continue
        first = (lower[axis] - ray_origin[axis]) / component
        second = (upper[axis] - ray_origin[axis]) / component
        near_values[axis] = min(first, second)
        far = min(far, max(first, second))
    entry = max(0.0, float(np.max(near_values)))
    if entry > far:
        return None
    tied = np.flatnonzero(np.isclose(near_values, entry, rtol=0.0, atol=1.0e-9))
    if tied.size != 1:
        return None
    axis = int(tied[0])
    positive = ray_direction[axis] > 0.0
    # Protocol: Up=1, Down=2, North=-Z=3, South=+Z=4,
    # East=+X=5, West=-X=6. The entered face opposes travel.
    return (
        (6 if positive else 5)
        if axis == 0
        else (2 if positive else 1)
        if axis == 1
        else (3 if positive else 4)
    )


def _empty_block_candidate_source() -> ActorBlockActionCandidates:
    capacity = ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY
    return ActorBlockActionCandidates(
        available=jnp.zeros((1, 1), dtype=jnp.bool_),
        capacity_exceeded=jnp.zeros((1, 1), dtype=jnp.bool_),
        diagnostics=jnp.zeros((1, 1), dtype=jnp.uint32),
        candidate_mask=jnp.zeros((1, 1, capacity), dtype=jnp.bool_),
        visible_position=jnp.zeros((1, 1, capacity, 3), dtype=jnp.float32),
        action_position=jnp.zeros((1, 1, capacity, 3), dtype=jnp.float32),
        visible_relative_position=jnp.zeros(
            (1, 1, capacity, 3),
            dtype=jnp.float32,
        ),
        affordance_tags=jnp.zeros((1, 1, capacity), dtype=jnp.uint16),
        gather_type_index=jnp.zeros((1, 1, capacity), dtype=jnp.uint8),
        required_tool_quality=jnp.zeros((1, 1, capacity), dtype=jnp.int16),
        provenance=jnp.zeros((1, 1, capacity), dtype=jnp.uint32),
    )


def _empty_recipe_encoding() -> RecipeCandidateEncoding:
    return RecipeCandidateEncoding(
        available=jnp.zeros((1,), dtype=jnp.bool_),
        candidate_mask=jnp.zeros(
            (1, ACTOR_RECIPE_CANDIDATE_CAPACITY),
            dtype=jnp.bool_,
        ),
        candidate_embedding=jnp.zeros(
            (
                1,
                ACTOR_RECIPE_CANDIDATE_CAPACITY,
                RECIPE_CANDIDATE_EMBEDDING_SIZE,
            ),
            dtype=jnp.float32,
        ),
    )


def _mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping_value(value.get(name))


def _mapping_value(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("expected an object")
    return value


def _array(value: Mapping[str, Any], name: str) -> Sequence[object]:
    result = value.get(name)
    if isinstance(result, (str, bytes, bytearray)) or not isinstance(
        result,
        Sequence,
    ):
        raise TypeError(f"{name} must be an array")
    return result


def _string(
    value: Mapping[str, Any],
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    result = value.get(name)
    if (
        not isinstance(result, str)
        or result != result.strip()
        or "\0" in result
        or (not allow_empty and not result)
    ):
        raise ValueError(f"{name} must be a canonical string")
    return result


def _optional_string(value: Mapping[str, Any], name: str) -> str | None:
    result = value.get(name)
    return result if isinstance(result, str) and result else None


def _optional_integer(value: Mapping[str, Any], name: str) -> int | None:
    result = value.get(name)
    if result is None:
        return None
    return _integer({name: result}, name, minimum=0)


def _policy_actor_identity(
    info: Mapping[str, Any],
    actor_slot: int,
) -> str | None:
    """Return one exact reset-negotiated actor UUID without slot fallback."""

    values = info.get("native_policy_actor_identities")
    if isinstance(values, Sequence) and not isinstance(
        values,
        (str, bytes, bytearray),
    ):
        if actor_slot >= len(values):
            return None
        value = values[actor_slot]
        return value if isinstance(value, str) and value else None
    if actor_slot == 0:
        return _optional_string(info, "native_policy_actor_identity")
    return None


def _integer(
    value: Mapping[str, Any],
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(result)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} is below its minimum")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} exceeds its maximum")
    return result


def _number(
    value: Mapping[str, Any],
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    result = value.get(name)
    if (
        isinstance(result, bool)
        or not isinstance(result, (int, float, np.integer, np.floating))
        or not math.isfinite(float(result))
    ):
        raise TypeError(f"{name} must be finite")
    result = float(result)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} is below its minimum")
    return result


def _boolean(value: Mapping[str, Any], name: str) -> bool:
    result = value.get(name)
    if not isinstance(result, bool):
        raise TypeError(f"{name} must be boolean")
    return result


def _int_vector(
    value: Mapping[str, Any],
    name: str,
    size: int,
) -> tuple[int, ...]:
    raw = _array(value, name)
    if len(raw) != size:
        raise ValueError(f"{name} must contain {size} integers")
    result = tuple(
        _integer({"value": item}, "value")
        for item in raw
    )
    if any(item < -(2**31) or item > 2**31 - 1 for item in result):
        raise ValueError(f"{name} must fit int32")
    return result


def _float_vector(
    value: Mapping[str, Any],
    name: str,
    size: int,
) -> tuple[float, ...]:
    raw = _array(value, name)
    if len(raw) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    return tuple(_number({"value": item}, "value") for item in raw)


def _byte_payload(
    value: Mapping[str, Any],
    name: str,
    size: int,
) -> bytes:
    raw = value.get(name)
    if isinstance(raw, bytes):
        result = raw
    elif isinstance(raw, bytearray):
        result = bytes(raw)
    elif isinstance(raw, Sequence) and not isinstance(raw, str):
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, np.integer))
            or not 0 <= int(item) <= 255
            for item in raw
        ):
            raise ValueError(f"{name} contains a non-byte value")
        result = bytes(int(item) for item in raw)
    else:
        raise TypeError(f"{name} must be bytes")
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} bytes")
    return result


def _sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{name} must be SHA-256")
    return value.upper()


def _equal(value: Mapping[str, Any], name: str, expected: object) -> None:
    if value.get(name) != expected:
        raise ValueError(f"{name} does not match the capture contract")


def _lane_scalar(value: object, name: str) -> object:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.shape != (1,):
        raise ValueError(f"{name} must be scalar or have shape (1,)")
    return array[0].item()


__all__ = [
    "NATIVE_POLICY_WORLD_ACTION_CAPTURE_SCHEMA",
    "NATIVE_POLICY_WORLD_ACTION_CAPTURE_TYPE",
    "NATIVE_POLICY_WORLD_ACTION_CAPTURE_VERSION",
    "NativePolicyWorldActionCaptureAdapter",
    "decode_native_policy_world_action_commit",
    "decode_native_policy_world_action_surface",
    "native_policy_world_action_capture_contract",
    "native_policy_world_action_capture_contract_sha256",
    "native_policy_world_action_capture_request",
]
