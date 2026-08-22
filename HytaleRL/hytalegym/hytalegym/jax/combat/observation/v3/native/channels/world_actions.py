"""Privileged policy-slot resolution for native World actions.

The learner sees bounded block/recipe slots, never absolute targets or native
String IDs.  This module retains that identity outside the observation and
joins one selected slot to the existing typed bridge request only after a
fresh evidence check.  It is deliberately host-side Python: neither the JAX
policy nor the Java bridge is allowed to infer a stale slot mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_BLOCK_TRIGGER_PRIMARY,
    ARSENAL_BLOCK_TRIGGER_SECONDARY,
    ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.blocks import (
    BlockActionCandidatePolicyView,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.encoder import (
    RecipeCandidateEncoding,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.recipes import (
    RecipeCandidatePolicyView,
)
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerArsenalAction,
)
from hytalegym.jax.crafting import IDENTITY_HASH_WORDS, PackedRecipeTable
from hytalegym.jax.world.mutable_blocks import BLOCK_SEMANTIC_KEY_WORDS
from hytalegym.jax.world.interactions.crafting import (
    ACTOR_RECIPE_CANDIDATE_CAPACITY,
)
from hytalegym.jax.combat.inventory import parse_native_inventory_frame
from hytalegym.worldgen import (
    NativeBlockUseEvidence,
    NativeItemInteractionEvidence,
    NativeWorldVerbRequest,
    NativeWorldVerbTransportSession,
)
from hytalegym.worldgen.native_mutable_blocks import NativeMutableBlockCells
from hytalegym.worldgen.native_action_resolution import (
    NativeCraftingExecutionContext,
    build_native_break_request,
    build_native_craft_request,
    build_native_place_request,
    build_native_use_request,
)


@dataclass(frozen=True, slots=True)
class NativeBlockCommitBinding:
    """Host-only exact inputs for one Primary or Secondary candidate."""

    target: tuple[int, int, int]
    block_face: int
    rotation: tuple[int, int, int]
    expected_block_id: str
    expected_semantic_key: tuple[int, ...] | None
    source_container: str
    source_slot: int
    expected_source_quantity: int
    source_item_id: str
    source_block_id: str = ""
    expected_interaction_tool_id: str = ""
    interaction_type: int = -1
    interaction_id: str = ""
    block_interaction_id: str = ""
    rotation_applicable: bool = False
    semantic_sha256: str = ""

    def __post_init__(self) -> None:
        _target(self.target, "target")
        _rotation(self.rotation)
        if isinstance(self.block_face, bool) or self.block_face not in range(1, 7):
            raise ValueError("block_face must be in [1, 6]")
        for name, value, allow_empty in (
            ("expected_block_id", self.expected_block_id, False),
            ("source_container", self.source_container, False),
            ("source_item_id", self.source_item_id, False),
            ("source_block_id", self.source_block_id, True),
            (
                "expected_interaction_tool_id",
                self.expected_interaction_tool_id,
                True,
            ),
            ("interaction_id", self.interaction_id, True),
            (
                "block_interaction_id",
                self.block_interaction_id,
                True,
            ),
        ):
            _canonical_string(value, name, allow_empty=allow_empty)
        if (
            isinstance(self.interaction_type, bool)
            or not isinstance(self.interaction_type, int)
            or self.interaction_type not in {-1, 0, 1}
        ):
            raise ValueError("interaction_type must be -1, Primary, or Secondary")
        if not isinstance(self.rotation_applicable, bool):
            raise TypeError("rotation_applicable must be boolean")
        if self.semantic_sha256:
            _sha256(self.semantic_sha256, "semantic_sha256")
        if not self.rotation_applicable and any(self.rotation):
            raise ValueError(
                "non-applicable block rotation must be the zero sentinel"
            )
        if self.source_container != "hotbar":
            raise ValueError("v1 block actions require a hotbar source")
        if (
            isinstance(self.source_slot, bool)
            or not isinstance(self.source_slot, int)
            or self.source_slot < 0
        ):
            raise ValueError("source_slot must be nonnegative")
        if (
            isinstance(self.expected_source_quantity, bool)
            or not isinstance(self.expected_source_quantity, int)
            or self.expected_source_quantity <= 0
        ):
            raise ValueError("expected_source_quantity must be positive")
        if self.expected_semantic_key is not None:
            _identity_words(
                self.expected_semantic_key,
                BLOCK_SEMANTIC_KEY_WORDS,
                "expected_semantic_key",
            )


@dataclass(frozen=True, slots=True)
class NativeBlockCandidateBinding:
    """Trigger-specific exact bindings retained behind one visible slot."""

    primary: NativeBlockCommitBinding | None = None
    secondary: NativeBlockCommitBinding | None = None

    def __post_init__(self) -> None:
        if self.primary is not None and self.primary.interaction_type not in {
            -1,
            0,
        }:
            raise ValueError("Primary binding carries the wrong interaction type")
        if self.secondary is not None and self.secondary.interaction_type not in {
            -1,
            1,
        }:
            raise ValueError("Secondary binding carries the wrong interaction type")
        if self.secondary is not None and not self.secondary.rotation_applicable:
            raise ValueError("Place binding requires an explicit rotation")

    def for_trigger(self, trigger: int) -> NativeBlockCommitBinding | None:
        if trigger == ARSENAL_BLOCK_TRIGGER_PRIMARY:
            return self.primary
        if trigger == ARSENAL_BLOCK_TRIGGER_SECONDARY:
            return self.secondary
        return None


@dataclass(frozen=True, slots=True)
class NativeRecipeCandidateBinding:
    """Host table identity retained behind one visible recipe slot."""

    recipe_index: int
    recipe_id_hash: tuple[int, ...]
    semantic_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.recipe_index, bool)
            or not isinstance(self.recipe_index, int)
            or self.recipe_index < 0
        ):
            raise ValueError("recipe_index must be nonnegative")
        _identity_words(
            self.recipe_id_hash,
            IDENTITY_HASH_WORDS,
            "recipe_id_hash",
        )
        if self.semantic_sha256:
            _sha256(self.semantic_sha256, "semantic_sha256")


@dataclass(frozen=True, slots=True)
class NativeUseCandidateBinding:
    """Exact actor-visible Use row retained outside the learner vector."""

    target: tuple[int, int, int]
    block_face: int
    rotation: tuple[int, int, int]
    interaction_id: str
    block_interaction_id: str
    item_id: str
    source_container: str
    source_slot: int
    source_quantity: int
    expected_block_id: str
    interaction_type: int = 5
    maximum_distance: float = 0.0
    expected_semantic_key: tuple[int, ...] | None = None
    resolved_source_container: str = "hotbar"
    semantic_sha256: str = ""

    def __post_init__(self) -> None:
        _target(self.target, "target")
        _rotation(self.rotation)
        if isinstance(self.block_face, bool) or self.block_face not in range(1, 7):
            raise ValueError("block_face must be in [1, 6]")
        for name in (
            "interaction_id",
            "block_interaction_id",
            "item_id",
            "source_container",
            "expected_block_id",
            "resolved_source_container",
        ):
            _canonical_string(getattr(self, name), name, allow_empty=False)
        if self.resolved_source_container not in {"hotbar", "tools"}:
            raise ValueError("resolved_source_container is unsupported")
        if isinstance(self.source_slot, bool) or not isinstance(
            self.source_slot,
            int,
        ):
            raise TypeError("source_slot must be an integer")
        if (
            isinstance(self.source_quantity, bool)
            or not isinstance(self.source_quantity, int)
            or self.source_quantity < 0
        ):
            raise ValueError("source_quantity must be nonnegative")
        if (
            isinstance(self.interaction_type, bool)
            or not isinstance(self.interaction_type, int)
            or self.interaction_type != 5
        ):
            raise ValueError("Use binding must carry interaction type 5")
        if (
            isinstance(self.maximum_distance, bool)
            or not isinstance(self.maximum_distance, (int, float))
            or not np.isfinite(float(self.maximum_distance))
            or float(self.maximum_distance) < 0.0
        ):
            raise ValueError("maximum_distance must be finite and nonnegative")
        if self.expected_semantic_key is not None:
            _identity_words(
                self.expected_semantic_key,
                BLOCK_SEMANTIC_KEY_WORDS,
                "expected_semantic_key",
            )
        if self.semantic_sha256:
            _sha256(self.semantic_sha256, "semantic_sha256")


@dataclass(frozen=True, slots=True)
class NativePolicyActionCandidateSnapshot:
    """The privileged counterpart of one immediately preceding policy row."""

    bridge_sha256: str
    world_epoch: str
    inventory_sha256: str
    recipe_table_content_sha256: str
    block_candidate_mask: tuple[bool, ...]
    block_candidates: tuple[NativeBlockCandidateBinding | None, ...]
    recipe_candidate_mask: tuple[bool, ...]
    recipe_candidates: tuple[NativeRecipeCandidateBinding | None, ...]
    block_trigger_available: tuple[bool, bool]
    use_available: bool
    use: NativeUseCandidateBinding | None
    crafting_context: NativeCraftingExecutionContext | None
    candidate_generation_sha256: str = ""
    actor_slot: int = -1
    actor_identity: str = ""
    recipe_table_identity_sha256: str = ""
    world_tick: int = -1
    environment_step: int = -1
    world_identity: str = ""

    def __post_init__(self) -> None:
        _sha256(self.bridge_sha256, "bridge_sha256")
        _canonical_string(self.world_epoch, "world_epoch", allow_empty=False)
        _sha256(self.inventory_sha256, "inventory_sha256")
        if self.recipe_table_content_sha256:
            _sha256(
                self.recipe_table_content_sha256,
                "recipe_table_content_sha256",
            )
        if self.candidate_generation_sha256:
            _sha256(
                self.candidate_generation_sha256,
                "candidate_generation_sha256",
            )
        if self.actor_identity:
            _canonical_string(
                self.actor_identity,
                "actor_identity",
                allow_empty=False,
            )
        if (
            isinstance(self.actor_slot, bool)
            or not isinstance(self.actor_slot, int)
            or self.actor_slot < -1
        ):
            raise ValueError("actor_slot must be -1 or nonnegative")
        if (self.actor_slot >= 0) != bool(self.actor_identity):
            raise ValueError(
                "actor slot and identity must either both be present or absent"
            )
        if self.recipe_table_identity_sha256:
            _sha256(
                self.recipe_table_identity_sha256,
                "recipe_table_identity_sha256",
            )
        if self.world_identity:
            _canonical_string(
                self.world_identity,
                "world_identity",
                allow_empty=False,
            )
        for name, value in (
            ("world_tick", self.world_tick),
            ("environment_step", self.environment_step),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < -1
            ):
                raise ValueError(f"{name} must be -1 or nonnegative")
        if len(self.block_candidate_mask) != (
            ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY
        ) or len(self.block_candidates) != len(self.block_candidate_mask):
            raise ValueError("block candidate snapshot capacity drift")
        if len(self.recipe_candidate_mask) != (
            ACTOR_RECIPE_CANDIDATE_CAPACITY
        ) or len(self.recipe_candidates) != len(self.recipe_candidate_mask):
            raise ValueError("recipe candidate snapshot capacity drift")
        if len(self.block_trigger_available) != 2:
            raise ValueError("block trigger availability must have two values")
        for mask, rows, name in (
            (
                self.block_candidate_mask,
                self.block_candidates,
                "block",
            ),
            (
                self.recipe_candidate_mask,
                self.recipe_candidates,
                "recipe",
            ),
        ):
            if any(not isinstance(value, bool) for value in mask):
                raise TypeError(f"{name} candidate mask must be boolean")
            if any(enabled != (row is not None) for enabled, row in zip(mask, rows)):
                raise ValueError(f"{name} candidate mask and bindings differ")
        if self.use_available != (self.use is not None):
            raise ValueError("Use availability and binding differ")
        if any(self.recipe_candidate_mask) and (
            not self.recipe_table_content_sha256
            or self.crafting_context is None
        ):
            raise ValueError("recipe candidates require table and context identity")


@dataclass(frozen=True, slots=True)
class NativePolicyActionSurface:
    """Actor-safe views and their parallel privileged commit snapshot."""

    block_candidates: BlockActionCandidatePolicyView
    recipe_candidates: RecipeCandidatePolicyView
    recipe_encoding: RecipeCandidateEncoding
    use_available: np.ndarray
    block_trigger_available: np.ndarray
    snapshot: NativePolicyActionCandidateSnapshot

    def __post_init__(self) -> None:
        if not isinstance(
            self.block_candidates,
            BlockActionCandidatePolicyView,
        ):
            raise TypeError(
                "block_candidates must be BlockActionCandidatePolicyView"
            )
        if not isinstance(self.recipe_candidates, RecipeCandidatePolicyView):
            raise TypeError(
                "recipe_candidates must be RecipeCandidatePolicyView"
            )
        if not isinstance(self.recipe_encoding, RecipeCandidateEncoding):
            raise TypeError("recipe_encoding must be RecipeCandidateEncoding")
        use = np.asarray(self.use_available)
        triggers = np.asarray(self.block_trigger_available)
        if use.dtype != np.bool_ or use.shape != (1,):
            raise ValueError("native Use availability must be bool[1]")
        if triggers.dtype != np.bool_ or triggers.shape != (1, 2):
            raise ValueError(
                "native block trigger availability must be bool[1,2]"
            )
        if bool(use[0]) != self.snapshot.use_available or tuple(
            bool(value) for value in triggers[0]
        ) != self.snapshot.block_trigger_available:
            raise ValueError(
                "actor-safe availability and privileged snapshot differ"
            )
        block_mask = np.asarray(self.block_candidates.candidate_mask)
        recipe_mask = np.asarray(self.recipe_candidates.candidate_mask)
        encoded_recipe_mask = np.asarray(self.recipe_encoding.candidate_mask)
        expected_block_mask = np.asarray(
            [self.snapshot.block_candidate_mask],
            dtype=np.bool_,
        )
        expected_recipe_mask = np.asarray(
            [self.snapshot.recipe_candidate_mask],
            dtype=np.bool_,
        )
        if (
            block_mask.shape != expected_block_mask.shape
            or not np.array_equal(block_mask, expected_block_mask)
        ):
            raise ValueError(
                "actor-safe block slots and privileged bindings differ"
            )
        if (
            recipe_mask.shape != expected_recipe_mask.shape
            or not np.array_equal(recipe_mask, expected_recipe_mask)
            or encoded_recipe_mask.shape != expected_recipe_mask.shape
            or not np.array_equal(encoded_recipe_mask, expected_recipe_mask)
        ):
            raise ValueError(
                "actor-safe recipe slots and privileged bindings differ"
            )
        if np.any(expected_block_mask) and not bool(
            np.asarray(self.block_candidates.available)[0]
        ):
            raise ValueError("block row availability and slots differ")
        recipe_available = bool(
            np.asarray(self.recipe_candidates.available)[0]
        )
        encoding_available = bool(np.asarray(self.recipe_encoding.available)[0])
        if recipe_available != encoding_available or (
            np.any(expected_recipe_mask) and not recipe_available
        ):
            raise ValueError("recipe row availability and slots differ")


@dataclass(frozen=True, slots=True)
class NativePolicyActionCommitEvidence:
    """Fresh evidence captured immediately before typed-request construction."""

    session: NativeWorldVerbTransportSession
    inventory: Mapping[str, object]
    block_cells: NativeMutableBlockCells | None = None
    item_interactions: NativeItemInteractionEvidence | None = None
    block_use: NativeBlockUseEvidence | None = None
    packed_recipes: PackedRecipeTable | None = None
    crafting_context: NativeCraftingExecutionContext | None = None
    candidate_generation_sha256: str = ""
    actor_slot: int = -1
    actor_identity: str = ""
    recipe_table_identity_sha256: str = ""
    world_tick: int = -1
    environment_step: int = -1
    unavailable_reason: str = ""
    world_identity: str = ""
    block_candidates: tuple[NativeBlockCandidateBinding | None, ...] | None = None
    use_binding: NativeUseCandidateBinding | None = None

    def __post_init__(self) -> None:
        if self.candidate_generation_sha256:
            _sha256(
                self.candidate_generation_sha256,
                "candidate_generation_sha256",
            )
        if self.actor_identity:
            _canonical_string(
                self.actor_identity,
                "actor_identity",
                allow_empty=False,
            )
        if (
            isinstance(self.actor_slot, bool)
            or not isinstance(self.actor_slot, int)
            or self.actor_slot < -1
        ):
            raise ValueError("actor_slot must be -1 or nonnegative")
        if (self.actor_slot >= 0) != bool(self.actor_identity):
            raise ValueError(
                "actor slot and identity must either both be present or absent"
            )
        if self.recipe_table_identity_sha256:
            _sha256(
                self.recipe_table_identity_sha256,
                "recipe_table_identity_sha256",
            )
        if self.unavailable_reason:
            _canonical_string(
                self.unavailable_reason,
                "unavailable_reason",
                allow_empty=False,
            )
        if self.world_identity:
            _canonical_string(
                self.world_identity,
                "world_identity",
                allow_empty=False,
            )
        if self.block_candidates is not None and len(self.block_candidates) != (
            ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY
        ):
            raise ValueError("fresh block candidate capacity drift")
        for name, value in (
            ("world_tick", self.world_tick),
            ("environment_step", self.environment_step),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < -1
            ):
                raise ValueError(f"{name} must be -1 or nonnegative")


@dataclass(frozen=True, slots=True)
class NativePolicyWorldActionResolution:
    """One resolver result consumed atomically by native translation."""

    request: NativeWorldVerbRequest | None
    legal: bool
    reject_reasons: tuple[str, ...]


NativePolicyActionSurfaceProvider = Callable[
    [object, Mapping[str, Any]],
    NativePolicyActionSurface,
]
NativePolicyActionCommitEvidenceProvider = Callable[
    [
        LearnerArsenalAction,
        NativePolicyActionCandidateSnapshot,
        Mapping[str, Any],
    ],
    NativePolicyActionCommitEvidence,
]


class NativePolicyWorldActionAdapter:
    """Own one observation-to-commit privileged candidate lifecycle.

    ``surface_provider`` derives actor-safe views and parallel host-only
    bindings from one native response. ``commit_evidence_provider`` performs
    the mandatory fresh bridge queries only when a policy selects a World
    verb. The adapter supplies monotonic request IDs and delegates the exact
    stale-evidence checks to :func:`resolve_native_policy_world_action`.

    The callback boundary is intentional: Region evidence production belongs
    to World, while the policy host owns when that evidence is sampled. Native
    IDs and coordinates never enter the learner row.
    """

    def __init__(
        self,
        surface_provider: NativePolicyActionSurfaceProvider,
        commit_evidence_provider: NativePolicyActionCommitEvidenceProvider,
        *,
        initial_request_id: int = 0,
    ) -> None:
        if not callable(surface_provider):
            raise TypeError("surface_provider must be callable")
        if not callable(commit_evidence_provider):
            raise TypeError("commit_evidence_provider must be callable")
        if (
            isinstance(initial_request_id, bool)
            or not isinstance(initial_request_id, int)
            or not 0 <= initial_request_id <= 2**63 - 1
        ):
            raise ValueError("initial_request_id must be a nonnegative int64")
        self._surface_provider = surface_provider
        self._commit_evidence_provider = commit_evidence_provider
        self._initial_request_id = initial_request_id
        self._next_request_id = initial_request_id

    def reset(self) -> None:
        """Reset the per-episode request sequence without retaining a slot."""

        self._next_request_id = self._initial_request_id

    def surface(
        self,
        assembly: object,
        info: Mapping[str, Any],
    ) -> NativePolicyActionSurface:
        """Build and validate the surface paired with this native row."""

        result = self._surface_provider(assembly, info)
        if not isinstance(result, NativePolicyActionSurface):
            raise TypeError(
                "surface_provider must return NativePolicyActionSurface"
            )
        return result

    def resolve(
        self,
        action: LearnerArsenalAction,
        snapshot: NativePolicyActionCandidateSnapshot,
        info: Mapping[str, Any],
    ) -> NativePolicyWorldActionResolution:
        """Freshly revalidate and resolve one selected policy World verb."""

        if not isinstance(snapshot, NativePolicyActionCandidateSnapshot):
            raise TypeError(
                "snapshot must be NativePolicyActionCandidateSnapshot"
            )
        if _selected_world_verb_count(action) == 0:
            return NativePolicyWorldActionResolution(None, True, ())
        evidence = self._commit_evidence_provider(action, snapshot, info)
        if not isinstance(evidence, NativePolicyActionCommitEvidence):
            raise TypeError(
                "commit_evidence_provider must return "
                "NativePolicyActionCommitEvidence"
            )
        request_id = self._next_request_id
        if request_id > 2**63 - 1:
            return _reject("native_world_verb_request_id_exhausted")
        result = resolve_native_policy_world_action(
            action,
            snapshot,
            evidence,
            request_id=request_id,
        )
        if result.request is not None:
            self._next_request_id += 1
        return result


def native_inventory_semantic_sha256(frame: Mapping[str, object]) -> str:
    """Hash validated quantity-aware semantics, excluding no represented field."""

    parsed = parse_native_inventory_frame(frame)
    payload = json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _selected_world_verb_count(action: LearnerArsenalAction) -> int:
    """Count executable verbs, not the block target shared by ``Use``."""

    if not isinstance(action, LearnerArsenalAction):
        raise TypeError("action must be LearnerArsenalAction")
    block = int(_lane_scalar(action.block_candidate_index, 0, "block_candidate_index"))
    recipe = int(
        _lane_scalar(action.recipe_candidate_index, 0, "recipe_candidate_index")
    )
    use = bool(_lane_scalar(action.use_requested, 0, "use_requested"))
    return int(use) + int(block >= 0 and not use) + int(recipe >= 0)


def resolve_native_policy_world_action(
    action: LearnerArsenalAction,
    snapshot: NativePolicyActionCandidateSnapshot,
    current: NativePolicyActionCommitEvidence,
    *,
    request_id: int,
    lane: int = 0,
) -> NativePolicyWorldActionResolution:
    """Resolve one selected learner factor after exact current revalidation."""

    if not isinstance(action, LearnerArsenalAction):
        raise TypeError("action must be LearnerArsenalAction")
    if not isinstance(snapshot, NativePolicyActionCandidateSnapshot):
        raise TypeError("snapshot must be NativePolicyActionCandidateSnapshot")
    if not isinstance(current, NativePolicyActionCommitEvidence):
        raise TypeError("current must be NativePolicyActionCommitEvidence")
    if (
        isinstance(request_id, bool)
        or not isinstance(request_id, int)
        or request_id < 0
    ):
        raise ValueError("request_id must be nonnegative")
    semantic = {
        name: _lane_scalar(getattr(action, name), lane, name)
        for name in LearnerArsenalAction._fields
    }
    block_index = int(semantic["block_candidate_index"])
    recipe_index = int(semantic["recipe_candidate_index"])
    use_requested = bool(semantic["use_requested"])
    requested_count = (
        int(use_requested)
        + int(block_index >= 0 and not use_requested)
        + int(recipe_index >= 0)
    )
    if requested_count == 0:
        return NativePolicyWorldActionResolution(None, True, ())
    if requested_count != 1:
        return _reject("native_world_verb_selection_conflict")
    if current.unavailable_reason:
        return _reject(current.unavailable_reason)
    identity_reason = _session_reason(snapshot, current.session)
    if identity_reason:
        return _reject(identity_reason)
    if (
        snapshot.candidate_generation_sha256
        and current.candidate_generation_sha256
        != snapshot.candidate_generation_sha256
    ):
        return _reject("native_candidate_generation_changed")
    if (
        snapshot.actor_slot >= 0
        and current.actor_slot != snapshot.actor_slot
    ):
        return _reject("native_candidate_actor_slot_changed")
    if (
        snapshot.actor_identity
        and current.actor_identity != snapshot.actor_identity
    ):
        return _reject("native_candidate_actor_changed")
    if (
        snapshot.world_identity
        and current.world_identity != snapshot.world_identity
    ):
        return _reject("native_candidate_world_changed")
    if (
        snapshot.world_tick >= 0
        and current.world_tick < snapshot.world_tick
    ):
        return _reject("native_candidate_world_tick_regressed")
    if (
        snapshot.environment_step >= 0
        and current.environment_step != snapshot.environment_step
    ):
        return _reject("native_candidate_environment_step_changed")
    if (
        snapshot.recipe_table_identity_sha256
        and current.recipe_table_identity_sha256
        != snapshot.recipe_table_identity_sha256
    ):
        return _reject("native_recipe_catalog_identity_changed")
    try:
        current_inventory_sha = native_inventory_semantic_sha256(
            current.inventory
        )
    except (TypeError, ValueError) as error:
        return _reject(_failure("native_inventory_evidence_invalid", error))
    if current_inventory_sha != snapshot.inventory_sha256:
        return _reject("native_inventory_snapshot_changed")

    if use_requested:
        return _resolve_use(snapshot, current, request_id=request_id)
    if block_index >= 0:
        return _resolve_block(
            semantic,
            snapshot,
            current,
            block_index,
            request_id=request_id,
        )
    return _resolve_recipe(
        snapshot,
        current,
        recipe_index,
        request_id=request_id,
    )


def _resolve_block(
    semantic: Mapping[str, object],
    snapshot: NativePolicyActionCandidateSnapshot,
    current: NativePolicyActionCommitEvidence,
    index: int,
    *,
    request_id: int,
) -> NativePolicyWorldActionResolution:
    if not 0 <= index < len(snapshot.block_candidates):
        return _reject("native_block_candidate_out_of_range")
    if not snapshot.block_candidate_mask[index]:
        return _reject("native_block_candidate_masked")
    trigger = int(semantic["block_interaction_trigger"])
    trigger_offset = trigger - ARSENAL_BLOCK_TRIGGER_PRIMARY
    if not 0 <= trigger_offset < len(snapshot.block_trigger_available):
        return _reject("native_block_trigger_invalid")
    if not snapshot.block_trigger_available[trigger_offset]:
        return _reject("native_block_trigger_masked")
    pair = snapshot.block_candidates[index]
    binding = None if pair is None else pair.for_trigger(trigger)
    if binding is None:
        return _reject("native_block_trigger_binding_missing")
    if current.block_candidates is not None:
        fresh_pair = current.block_candidates[index]
        fresh_binding = (
            None if fresh_pair is None else fresh_pair.for_trigger(trigger)
        )
        if fresh_binding != binding:
            return _reject("native_block_candidate_changed")
    if current.block_cells is None:
        return _reject("native_block_commit_evidence_missing")
    if current.item_interactions is None:
        return _reject("native_item_interaction_evidence_missing")
    native_interaction_type = (
        0 if trigger == ARSENAL_BLOCK_TRIGGER_PRIMARY else 1
    )
    if binding.interaction_type not in {-1, native_interaction_type}:
        return _reject("native_block_interaction_type_changed")
    native_trigger = current.item_interactions.triggers[
        native_interaction_type
    ]
    if (
        native_trigger.root is None
        or (
            binding.interaction_id
            and native_trigger.root.root_id != binding.interaction_id
        )
        or native_trigger.item_asset_id != binding.source_item_id
        or native_trigger.held_item_slot != binding.source_slot
    ):
        return _reject("native_block_interaction_root_changed")
    row = _current_block_row(
        current.block_cells,
        binding,
        current.session,
    )
    if isinstance(row, str):
        return _reject(row)
    inventory_reason = _inventory_source_reason(current.inventory, binding)
    if inventory_reason:
        return _reject(inventory_reason)
    expectation_reason = _candidate_expectation_reason(
        snapshot.candidate_generation_sha256,
        binding.semantic_sha256,
    )
    if expectation_reason:
        return _reject(expectation_reason)
    try:
        if trigger == ARSENAL_BLOCK_TRIGGER_PRIMARY:
            request = build_native_break_request(
                current.item_interactions,
                current.session,
                request_id=request_id,
                target=binding.target,
                block_face=binding.block_face,
                rotation=binding.rotation,
                expected_source_quantity=binding.expected_source_quantity,
                expected_block_id=binding.expected_block_id,
                expected_interaction_tool_id=(
                    binding.expected_interaction_tool_id
                ),
                expected_candidate_generation_sha256=(
                    snapshot.candidate_generation_sha256
                ),
                expected_selected_semantic_sha256=binding.semantic_sha256,
            )
        else:
            request = build_native_place_request(
                current.item_interactions,
                current.session,
                request_id=request_id,
                target=binding.target,
                block_face=binding.block_face,
                rotation=binding.rotation,
                expected_source_quantity=binding.expected_source_quantity,
                expected_block_id=binding.expected_block_id,
                source_block_id=binding.source_block_id,
                expected_candidate_generation_sha256=(
                    snapshot.candidate_generation_sha256
                ),
                expected_selected_semantic_sha256=binding.semantic_sha256,
            )
    except (TypeError, ValueError) as error:
        return _reject(_failure("native_block_request_revalidation_failed", error))
    return NativePolicyWorldActionResolution(request, True, ())


def _resolve_recipe(
    snapshot: NativePolicyActionCandidateSnapshot,
    current: NativePolicyActionCommitEvidence,
    index: int,
    *,
    request_id: int,
) -> NativePolicyWorldActionResolution:
    if not 0 <= index < len(snapshot.recipe_candidates):
        return _reject("native_recipe_candidate_out_of_range")
    if not snapshot.recipe_candidate_mask[index]:
        return _reject("native_recipe_candidate_masked")
    binding = snapshot.recipe_candidates[index]
    if binding is None:
        return _reject("native_recipe_candidate_binding_missing")
    if current.packed_recipes is None:
        return _reject("native_recipe_table_missing")
    if (
        current.packed_recipes.content_sha256.upper()
        != snapshot.recipe_table_content_sha256.upper()
    ):
        return _reject("native_recipe_table_changed")
    if current.crafting_context != snapshot.crafting_context:
        return _reject("native_crafting_context_changed")
    expectation_reason = _candidate_expectation_reason(
        snapshot.candidate_generation_sha256,
        binding.semantic_sha256,
    )
    if expectation_reason:
        return _reject(expectation_reason)
    assert current.crafting_context is not None
    try:
        request = build_native_craft_request(
            current.packed_recipes,
            current.session,
            current.crafting_context,
            request_id=request_id,
            recipe_index=binding.recipe_index,
            recipe_id_hash=binding.recipe_id_hash,
            expected_candidate_generation_sha256=(
                snapshot.candidate_generation_sha256
            ),
            expected_selected_semantic_sha256=binding.semantic_sha256,
        )
    except (TypeError, ValueError) as error:
        return _reject(_failure("native_recipe_revalidation_failed", error))
    return NativePolicyWorldActionResolution(request, True, ())


def _resolve_use(
    snapshot: NativePolicyActionCandidateSnapshot,
    current: NativePolicyActionCommitEvidence,
    *,
    request_id: int,
) -> NativePolicyWorldActionResolution:
    binding = snapshot.use
    if not snapshot.use_available or binding is None:
        return _reject("native_use_candidate_masked")
    if current.use_binding is not None and current.use_binding != binding:
        return _reject("native_use_candidate_changed")
    evidence = current.block_use
    if evidence is None:
        return _reject("native_use_evidence_missing")
    expected = (
        binding.target,
        binding.interaction_id,
        binding.block_interaction_id,
        binding.item_id,
        binding.source_container,
        binding.source_slot,
        binding.source_quantity,
    )
    actual = (
        evidence.available_target,
        evidence.available_interaction_id,
        evidence.available_block_interaction_id,
        evidence.available_item_id,
        evidence.available_source_container,
        evidence.available_source_slot,
        evidence.available_source_quantity,
        evidence.available_maximum_distance,
    )
    expected_maximum_distance = (
        evidence.available_maximum_distance
        if binding.maximum_distance == 0.0
        else float(binding.maximum_distance)
    )
    expected = expected + (expected_maximum_distance,)
    if not evidence.available or actual != expected:
        return _reject("native_use_candidate_changed")
    expectation_reason = _candidate_expectation_reason(
        snapshot.candidate_generation_sha256,
        binding.semantic_sha256,
    )
    if expectation_reason:
        return _reject(expectation_reason)
    try:
        request = build_native_use_request(
            evidence,
            current.session,
            request_id=request_id,
            block_face=binding.block_face,
            rotation=binding.rotation,
            expected_block_id=binding.expected_block_id,
            expected_candidate_generation_sha256=(
                snapshot.candidate_generation_sha256
            ),
            expected_selected_semantic_sha256=binding.semantic_sha256,
        )
    except (TypeError, ValueError) as error:
        return _reject(_failure("native_use_revalidation_failed", error))
    return NativePolicyWorldActionResolution(request, True, ())


def _current_block_row(
    capture: NativeMutableBlockCells,
    binding: NativeBlockCommitBinding,
    session: NativeWorldVerbTransportSession,
):
    if capture.bridge_sha256 != session.bridge_sha256.upper():
        return "native_block_evidence_bridge_changed"
    matches = [
        cell
        for cell in capture.cells
        if tuple(int(value) for value in cell.position) == binding.target
    ]
    if len(matches) != 1 or not matches[0].available or matches[0].row is None:
        return "native_block_target_missing"
    row = matches[0].row
    if binding.expected_block_id == "Empty":
        if row.block_present:
            return "native_block_semantic_identity_changed"
    else:
        semantic = tuple(int(value) for value in row.semantic_key)
        if (
            not row.block_present
            or row.block_asset_id != binding.expected_block_id
            or not row.semantic_key_valid
            or binding.expected_semantic_key is None
            or semantic != binding.expected_semantic_key
        ):
            return "native_block_semantic_identity_changed"
    return row


def _inventory_source_reason(
    frame: Mapping[str, object],
    binding: NativeBlockCommitBinding,
) -> str | None:
    try:
        parsed = parse_native_inventory_frame(frame)
    except (TypeError, ValueError):
        return "native_inventory_evidence_invalid"
    containers = {
        row["name"]: row for row in parsed["containers"]
    }
    container = containers.get(binding.source_container)
    if container is None or not container["available"]:
        return "native_inventory_source_unavailable"
    slots = {
        row["slot"]: row for row in container["occupied_slots"]
    }
    slot = slots.get(binding.source_slot)
    if slot is None or (
        slot["item_id"] != binding.source_item_id
        or slot["quantity"] != binding.expected_source_quantity
    ):
        return "native_inventory_source_changed"
    return None


def _session_reason(
    snapshot: NativePolicyActionCandidateSnapshot,
    session: NativeWorldVerbTransportSession,
) -> str | None:
    if session.bridge_sha256.upper() != snapshot.bridge_sha256.upper():
        return "native_candidate_snapshot_bridge_changed"
    if session.world_epoch != snapshot.world_epoch:
        return "native_candidate_snapshot_epoch_changed"
    return None


def _candidate_expectation_reason(
    generation_sha256: str,
    selected_semantic_sha256: str,
) -> str | None:
    if bool(generation_sha256) != bool(selected_semantic_sha256):
        return "native_policy_candidate_expectation_incomplete"
    return None


def _reject(reason: str) -> NativePolicyWorldActionResolution:
    return NativePolicyWorldActionResolution(None, False, (reason,))


def _failure(prefix: str, error: Exception) -> str:
    detail = re.sub(r"[^a-z0-9]+", "_", str(error).lower()).strip("_")
    return prefix if not detail else f"{prefix}:{detail}"


def _lane_scalar(value, lane: int, name: str):
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.ndim != 1 or lane < 0 or lane >= array.shape[0]:
        raise ValueError(f"{name} must be scalar or have shape (B,)")
    return array[lane].item()


def _identity_words(
    values: Sequence[int],
    expected: int,
    name: str,
) -> tuple[int, ...]:
    result = tuple(values)
    if len(result) != expected or any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 0 <= int(value) <= 0xFFFFFFFF
        for value in result
    ):
        raise ValueError(f"{name} must contain {expected} uint32 words")
    return tuple(int(value) for value in result)


def _target(value: tuple[int, int, int], name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3 or any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not -(2**31) <= item < 2**31
        for item in value
    ):
        raise ValueError(f"{name} must contain three int32 coordinates")


def _rotation(value: tuple[int, int, int]) -> None:
    if not isinstance(value, tuple) or len(value) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) or item not in range(4)
        for item in value
    ):
        raise ValueError("rotation must contain three values in [0, 3]")


def _canonical_string(value: str, name: str, *, allow_empty: bool) -> None:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or "\0" in value
        or (not allow_empty and not value)
    ):
        raise ValueError(f"{name} must be a canonical string")


def _sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{name} must be one SHA-256")
    return value.upper()


__all__ = [
    "NativeBlockCandidateBinding",
    "NativeBlockCommitBinding",
    "NativePolicyActionCandidateSnapshot",
    "NativePolicyActionCommitEvidence",
    "NativePolicyActionCommitEvidenceProvider",
    "NativePolicyActionSurface",
    "NativePolicyActionSurfaceProvider",
    "NativePolicyWorldActionAdapter",
    "NativePolicyWorldActionResolution",
    "NativeRecipeCandidateBinding",
    "NativeUseCandidateBinding",
    "native_inventory_semantic_sha256",
    "resolve_native_policy_world_action",
]
