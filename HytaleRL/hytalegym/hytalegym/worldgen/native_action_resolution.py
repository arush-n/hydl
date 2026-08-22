"""Lossless host resolution from bounded policy slots to native World verbs.

Policy tensors keep hashes and table-local indexes on purpose.  The bridge,
however, accepts Hytale String asset IDs.  This module is the privileged seam:
it resolves those values from pinned host catalogs, revalidates current native
interaction evidence, and only then constructs a typed v2 request.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from hytalegym.jax.crafting.contract import (
    IDENTITY_HASH_WORDS,
    identity_sha256_words,
)
from hytalegym.jax.crafting.packing import PackedRecipeTable
from hytalegym.worldgen.block_actions import (
    block_asset_key,
    block_semantic_key,
)
from hytalegym.worldgen.native_interactions import (
    NativeBreakBlockPayload,
    NativeItemInteractionEvidence,
    NativePlaceBlockPayload,
)
from hytalegym.worldgen.native_world_actions import (
    NativeBlockUseEvidence,
    NativeWorldVerbRequest,
    NativeWorldVerbTransportSession,
)
from hytalegym.worldgen.region.block_action_catalog import (
    resolve_region_block_asset_references,
)
from hytalegym.worldgen.region.block_semantics import (
    RegionBlockSemanticEntry,
)
from hytalegym.worldgen.surrogate.assets import HytaleAssetArchive
from hytalegym.worldgen.surrogate.semantics import LocalBlockSemanticsResolver


NATIVE_ACTION_RESOLUTION_SCHEMA = "hytalerl_native_action_resolution_v1"
NATIVE_ACTION_RESOLUTION_VERSION = 1


@dataclass(frozen=True, slots=True)
class RegionNativeBlockIdentity:
    """One exact Region semantic key and its native String block ID."""

    semantic_key: tuple[int, ...]
    asset_key: tuple[int, ...]
    block_id: str
    rotation_index: int

    def __post_init__(self) -> None:
        block_id = _native_id(self.block_id, "block_id")
        rotation = _nonnegative_int(self.rotation_index, "rotation_index")
        semantic_key = _key(self.semantic_key, "semantic_key")
        asset_key = _key(self.asset_key, "asset_key")
        if semantic_key != block_semantic_key(block_id, rotation):
            raise ValueError("native block ID does not match semantic_key")
        if asset_key != block_asset_key(block_id):
            raise ValueError("native block ID does not match asset_key")
        object.__setattr__(self, "block_id", block_id)
        object.__setattr__(self, "rotation_index", rotation)
        object.__setattr__(self, "semantic_key", semantic_key)
        object.__setattr__(self, "asset_key", asset_key)


@dataclass(frozen=True, slots=True)
class RegionNativeBlockIdentityCatalog:
    """Canonical host-only reverse mapping for one Region palette."""

    entries: tuple[RegionNativeBlockIdentity, ...]
    source_block_semantic_sha256: str

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries or any(
            not isinstance(entry, RegionNativeBlockIdentity) for entry in entries
        ):
            raise TypeError("entries must contain native block identities")
        if entries != tuple(sorted(entries, key=lambda row: row.semantic_key)):
            raise ValueError("native block identities are not canonical")
        if len({row.semantic_key for row in entries}) != len(entries):
            raise ValueError("native block semantic keys must be unique")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self,
            "source_block_semantic_sha256",
            _sha256(
                self.source_block_semantic_sha256,
                "source_block_semantic_sha256",
            ),
        )

    def semantic_sha256(self) -> str:
        """Hash the exact reverse mapping, excluding replaceable provenance."""

        return _canonical_sha256(
            {
                "schema": NATIVE_ACTION_RESOLUTION_SCHEMA,
                "version": NATIVE_ACTION_RESOLUTION_VERSION,
                "contract_sha256": native_action_resolution_contract_sha256(),
                "source_block_semantic_sha256": (self.source_block_semantic_sha256),
                "entries": [asdict(entry) for entry in self.entries],
            }
        )


@dataclass(frozen=True, slots=True)
class NativeCraftingExecutionContext:
    """Exact current fieldcraft or ordinary-Crafting-bench evidence."""

    crafting_context: str
    bench: tuple[int, int, int]
    expected_bench_block_id: str
    expected_bench_id: str
    expected_bench_type: int
    expected_bench_tier: int

    @classmethod
    def fieldcraft(cls) -> NativeCraftingExecutionContext:
        return cls("fieldcraft", (0, 0, 0), "", "Fieldcraft", 0, 0)

    @classmethod
    def ordinary_bench(
        cls,
        *,
        position: tuple[int, int, int],
        block_id: str,
        bench_id: str,
        tier: int,
    ) -> NativeCraftingExecutionContext:
        return cls(
            "bench",
            _target(position, "bench position"),
            _native_id(block_id, "bench block_id"),
            _native_id(bench_id, "bench_id", maximum=128),
            0,
            _positive_int(tier, "bench tier"),
        )

    def __post_init__(self) -> None:
        if self.crafting_context == "fieldcraft":
            if (
                self.bench != (0, 0, 0)
                or self.expected_bench_block_id
                or self.expected_bench_id != "Fieldcraft"
                or self.expected_bench_type != 0
                or self.expected_bench_tier != 0
            ):
                raise ValueError("fieldcraft context carries bench-only fields")
            return
        if self.crafting_context != "bench":
            raise ValueError("only fieldcraft and ordinary benches are supported")
        _target(self.bench, "bench")
        _native_id(self.expected_bench_block_id, "expected_bench_block_id")
        _native_id(
            self.expected_bench_id,
            "expected_bench_id",
            maximum=128,
        )
        if self.expected_bench_type != 0:
            raise ValueError("only ordinary Crafting benches are lossless")
        _positive_int(self.expected_bench_tier, "expected_bench_tier")


def compile_region_native_block_identity_catalog(
    archive: HytaleAssetArchive,
    palette: Sequence[RegionBlockSemanticEntry],
    *,
    source_block_semantic_sha256: str,
) -> RegionNativeBlockIdentityCatalog:
    """Compile a collision-checked native String mapping for a Region."""

    references = resolve_region_block_asset_references(archive, palette)
    return RegionNativeBlockIdentityCatalog(
        entries=tuple(
            RegionNativeBlockIdentity(
                semantic_key=entry.semantic_key,
                asset_key=entry.asset_key,
                block_id=reference,
                rotation_index=entry.rotation_index,
            )
            for entry, reference in references
        ),
        source_block_semantic_sha256=source_block_semantic_sha256,
    )


def resolve_region_native_block_id(
    catalog: RegionNativeBlockIdentityCatalog,
    semantic_key: Sequence[int],
) -> str:
    """Resolve one current semantic key without accepting a guessed ID."""

    if not isinstance(catalog, RegionNativeBlockIdentityCatalog):
        raise TypeError("catalog must be a RegionNativeBlockIdentityCatalog")
    key = _key(semantic_key, "semantic_key")
    matches = tuple(row.block_id for row in catalog.entries if row.semantic_key == key)
    if len(matches) != 1:
        raise ValueError("semantic key has no unique native block identity")
    return matches[0]


def resolve_native_place_item_block_id(
    archive: HytaleAssetArchive,
    item_asset_id: str,
) -> str:
    """Resolve the held ItemStack block key for an ordinary block item.

    Hytale 0.5.7 ``Item.processConfig`` assigns the containing Item ID to the
    embedded ``BlockType``.  Requiring the resolved inherited BlockType here
    distinguishes that source-backed rule from merely assuming item == block.
    """

    if not isinstance(archive, HytaleAssetArchive):
        raise TypeError("archive must be a HytaleAssetArchive")
    item_id = _native_id(item_asset_id, "item_asset_id")
    item, _ = LocalBlockSemanticsResolver(archive).resolve_item_asset(item_id)
    if not isinstance(item.get("BlockType"), Mapping):
        raise ValueError("selected Place item has no resolved BlockType")
    return item_id


def resolve_native_recipe_id(
    packed: PackedRecipeTable,
    recipe_index: int,
    recipe_id_hash: Sequence[int],
) -> str:
    """Resolve one selected recipe index and verify both identity hashes."""

    if not isinstance(packed, PackedRecipeTable):
        raise TypeError("packed must be a PackedRecipeTable")
    index = _nonnegative_int(recipe_index, "recipe_index")
    if index >= len(packed.recipe_ids):
        raise ValueError("recipe_index is outside the resolved catalog")
    recipe_id = _native_id(packed.recipe_ids[index], "recipe_id")
    expected = identity_sha256_words(recipe_id)
    selected_hash = _identity_words(recipe_id_hash, "recipe_id_hash")
    table_mask = np.asarray(packed.table.recipe_mask)
    table_hash = np.asarray(packed.table.recipe_id_hash)
    if (
        table_mask.ndim != 1
        or table_hash.shape != (table_mask.shape[0], IDENTITY_HASH_WORDS)
        or index >= table_mask.shape[0]
        or not bool(table_mask[index])
        or _identity_words(table_hash[index], "table recipe_id_hash") != expected
        or selected_hash != expected
    ):
        raise ValueError("selected recipe identity is stale or inconsistent")
    return recipe_id


def build_native_place_request(
    evidence: NativeItemInteractionEvidence,
    session: NativeWorldVerbTransportSession,
    *,
    request_id: int,
    target: tuple[int, int, int],
    block_face: int,
    rotation: tuple[int, int, int],
    expected_source_quantity: int,
    expected_block_id: str,
    source_block_id: str,
    expected_candidate_generation_sha256: str = "",
    expected_selected_semantic_sha256: str = "",
) -> NativeWorldVerbRequest:
    """Build one exact Secondary Place request from current item evidence."""

    trigger = _item_trigger(evidence, session, 1)
    payloads = tuple(
        node.payload
        for node in trigger.nodes
        if isinstance(node.payload, NativePlaceBlockPayload)
    )
    explicit_block_ids = {
        _native_id(payload.block_asset_id, "place payload block_id")
        for payload in payloads
        if payload.block_asset_id
    }
    source_block = _native_id(source_block_id, "source_block_id")
    if source_block != trigger.item_asset_id or (
        explicit_block_ids and explicit_block_ids != {source_block}
    ):
        raise ValueError("Place root and held-item block identities differ")
    expected = _native_id(expected_block_id, "expected_block_id")
    if expected != "Empty":
        raise ValueError("v1 placement candidates must target exact empty cells")
    return NativeWorldVerbRequest(
        request_id=_request_id(request_id),
        verb="place_block",
        interaction_type=1,
        target_kind="block",
        interaction_id=trigger.root.root_id,
        recipe_id="",
        item_id=_native_id(trigger.item_asset_id, "place item_id"),
        block_id=source_block,
        target=_target(target, "target"),
        block_face=_block_face(block_face),
        rotation=_rotation(rotation),
        source_container="hotbar",
        source_slot=_nonnegative_int(trigger.held_item_slot, "source_slot"),
        expected_source_quantity=_positive_int(
            expected_source_quantity,
            "expected_source_quantity",
        ),
        destination_container="",
        expected_block_id=expected,
        world_epoch=session.world_epoch,
        quantity=1,
        placement_variant="default",
        expected_candidate_generation_sha256=(
            expected_candidate_generation_sha256
        ),
        expected_selected_semantic_sha256=(
            expected_selected_semantic_sha256
        ),
    )


def build_native_break_request(
    evidence: NativeItemInteractionEvidence,
    session: NativeWorldVerbTransportSession,
    *,
    request_id: int,
    target: tuple[int, int, int],
    block_face: int,
    rotation: tuple[int, int, int],
    expected_source_quantity: int,
    expected_block_id: str,
    expected_interaction_tool_id: str,
    expected_candidate_generation_sha256: str = "",
    expected_selected_semantic_sha256: str = "",
) -> NativeWorldVerbRequest:
    """Build Primary Break and bind its authored server tool key."""

    trigger = _item_trigger(evidence, session, 0)
    payloads = tuple(
        node.payload
        for node in trigger.nodes
        if isinstance(node.payload, NativeBreakBlockPayload)
    )
    tool_id = _native_id(
        expected_interaction_tool_id,
        "expected_interaction_tool_id",
    )
    authored_ids = {payload.tool_id for payload in payloads if payload.tool_id}
    if tool_id not in authored_ids or any(
        payload.match_tool and payload.tool_id != tool_id for payload in payloads
    ):
        raise ValueError("Break root does not carry the selected native tool key")
    expected = _native_id(expected_block_id, "expected_block_id")
    if expected == "Empty":
        raise ValueError("Break requires an exact present block identity")
    return NativeWorldVerbRequest(
        request_id=_request_id(request_id),
        verb="break_block",
        interaction_type=0,
        target_kind="block",
        interaction_id=trigger.root.root_id,
        recipe_id="",
        item_id=_native_id(trigger.item_asset_id, "break item_id"),
        block_id="",
        target=_target(target, "target"),
        block_face=_block_face(block_face),
        rotation=_rotation(rotation),
        source_container="hotbar",
        source_slot=_nonnegative_int(trigger.held_item_slot, "source_slot"),
        expected_source_quantity=_positive_int(
            expected_source_quantity,
            "expected_source_quantity",
        ),
        destination_container="",
        expected_block_id=expected,
        world_epoch=session.world_epoch,
        quantity=1,
        placement_variant="",
        expected_candidate_generation_sha256=(
            expected_candidate_generation_sha256
        ),
        expected_selected_semantic_sha256=(
            expected_selected_semantic_sha256
        ),
    )


def build_native_use_request(
    evidence: NativeBlockUseEvidence,
    session: NativeWorldVerbTransportSession,
    *,
    request_id: int,
    block_face: int,
    rotation: tuple[int, int, int],
    expected_block_id: str,
    expected_candidate_generation_sha256: str = "",
    expected_selected_semantic_sha256: str = "",
) -> NativeWorldVerbRequest:
    """Build exact block Use from the bridge's current availability row."""

    if not isinstance(evidence, NativeBlockUseEvidence):
        raise TypeError("evidence must be NativeBlockUseEvidence")
    _same_bridge(evidence.bridge_sha256, session)
    if (
        not evidence.available
        or evidence.available_target is None
        or not evidence.available_interaction_id
        or not evidence.available_item_id
        or not evidence.available_source_container
    ):
        raise ValueError("native block Use is not currently available")
    return NativeWorldVerbRequest(
        request_id=_request_id(request_id),
        verb="use",
        interaction_type=5,
        target_kind="block",
        interaction_id=evidence.available_interaction_id,
        recipe_id="",
        item_id=evidence.available_item_id,
        block_id="",
        target=_target(evidence.available_target, "Use target"),
        block_face=_block_face(block_face),
        rotation=_rotation(rotation),
        source_container=evidence.available_source_container,
        source_slot=evidence.available_source_slot,
        expected_source_quantity=evidence.available_source_quantity,
        destination_container="",
        expected_block_id=_native_id(
            expected_block_id,
            "expected_block_id",
        ),
        world_epoch=session.world_epoch,
        quantity=1,
        placement_variant="",
        expected_candidate_generation_sha256=(
            expected_candidate_generation_sha256
        ),
        expected_selected_semantic_sha256=(
            expected_selected_semantic_sha256
        ),
    )


def build_native_craft_request(
    packed: PackedRecipeTable,
    session: NativeWorldVerbTransportSession,
    context: NativeCraftingExecutionContext,
    *,
    request_id: int,
    recipe_index: int,
    recipe_id_hash: Sequence[int],
    quantity: int = 1,
    expected_candidate_generation_sha256: str = "",
    expected_selected_semantic_sha256: str = "",
) -> NativeWorldVerbRequest:
    """Build String-ID Craft after exact selected-slot revalidation."""

    if not isinstance(context, NativeCraftingExecutionContext):
        raise TypeError("context must be NativeCraftingExecutionContext")
    return NativeWorldVerbRequest(
        request_id=_request_id(request_id),
        verb="craft_recipe",
        interaction_type=-1,
        target_kind="recipe",
        interaction_id="",
        recipe_id=resolve_native_recipe_id(
            packed,
            recipe_index,
            recipe_id_hash,
        ),
        item_id="",
        block_id="",
        target=(0, 0, 0),
        block_face=0,
        rotation=(0, 0, 0),
        source_container="player_inventory",
        source_slot=-1,
        expected_source_quantity=-1,
        destination_container="player_inventory_or_world_drop",
        expected_block_id="",
        world_epoch=session.world_epoch,
        quantity=_positive_int(quantity, "quantity"),
        placement_variant="",
        crafting_context=context.crafting_context,
        bench=context.bench,
        expected_bench_block_id=context.expected_bench_block_id,
        expected_bench_id=context.expected_bench_id,
        expected_bench_type=context.expected_bench_type,
        expected_bench_tier=context.expected_bench_tier,
        expected_candidate_generation_sha256=(
            expected_candidate_generation_sha256
        ),
        expected_selected_semantic_sha256=(
            expected_selected_semantic_sha256
        ),
    )


def native_action_resolution_contract() -> dict[str, object]:
    """Publish the privileged identity and request-resolution boundary."""

    return {
        "schema": NATIVE_ACTION_RESOLUTION_SCHEMA,
        "version": NATIVE_ACTION_RESOLUTION_VERSION,
        "server_version": "0.5.7",
        "policy_boundary": {
            "visible": "bounded_candidate_slot_and_policy_fields",
            "privileged": [
                "Region_semantic_SHA256_words",
                "recipe_table_index_and_recipe_String_SHA256_words",
                "current_native_item_root_and_block_Use_evidence",
            ],
            "native": "exact_String_asset_IDs_only_after_selection",
        },
        "block_identity": {
            "source": "installed_inherited_item_and_state_definition_assets",
            "ambiguity": "reject",
            "break": "current_semantic_key_to_exact_BlockType.getId",
            "place": (
                "resolved_held_Item.processConfig_block_ID_with_optional_"
                "typed_PlaceBlock_override_plus_exact_empty_destination"
            ),
            "face": "actor_aim_Hytale_protocol_BlockFace_never_defaulted",
            "tool": "selected_native_BreakBlock_tool_key_must_be_authored",
        },
        "recipe_identity": {
            "source": "PackedRecipeTable.recipe_ids",
            "checks": [
                "selected_index_active",
                "selected_hash_matches_String_ID",
                "packed_table_hash_matches_String_ID",
            ],
            "supported_contexts": ["fieldcraft", "ordinary_Crafting_bench"],
            "future_contexts": ["Diagram", "Structural", "Processing"],
        },
        "revalidation": (
            "bridge_identity_item_root_source_slot_quantity_target_"
            "block_identity_and_recipe_identity_fail_closed"
        ),
        "policy_abi": "unchanged_host_only_resolver",
    }


def native_action_resolution_contract_sha256() -> str:
    return _canonical_sha256(native_action_resolution_contract())


def _item_trigger(
    evidence: NativeItemInteractionEvidence,
    session: NativeWorldVerbTransportSession,
    interaction_type: int,
):
    if not isinstance(evidence, NativeItemInteractionEvidence):
        raise TypeError("evidence must be NativeItemInteractionEvidence")
    _same_bridge(evidence.bridge_sha256, session)
    if interaction_type < 0 or interaction_type >= len(evidence.triggers):
        raise ValueError("selected native item trigger is unavailable")
    trigger = evidence.triggers[interaction_type]
    if (
        trigger.interaction_type != interaction_type
        or trigger.root is None
        or not trigger.root.root_id
        or not trigger.item_asset_id
        or trigger.held_item_slot < 0
    ):
        raise ValueError("selected native item root is unavailable")
    return trigger


def _same_bridge(
    evidence_bridge_sha256: str,
    session: NativeWorldVerbTransportSession,
) -> None:
    if not isinstance(session, NativeWorldVerbTransportSession):
        raise TypeError("session must be NativeWorldVerbTransportSession")
    if _sha256(evidence_bridge_sha256, "evidence bridge") != _sha256(
        session.bridge_sha256,
        "session bridge",
    ):
        raise ValueError("native action evidence came from another bridge")


def _request_id(value: object) -> int:
    result = _nonnegative_int(value, "request_id")
    if result > 2**63 - 1:
        raise ValueError("request_id must fit int64")
    return result


def _target(value: object, label: str) -> tuple[int, int, int]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{label} must be a coordinate tuple")
    result = tuple(_integer(row, label) for row in value)
    if any(row < -(2**31) or row > 2**31 - 1 for row in result):
        raise ValueError(f"{label} must fit int32")
    return result


def _rotation(value: object) -> tuple[int, int, int]:
    result = _target(value, "rotation")
    if any(row not in range(4) for row in result):
        raise ValueError("rotation values must be in [0, 3]")
    return result


def _block_face(value: object) -> int:
    face = _integer(value, "block_face")
    if face not in range(1, 7):
        raise ValueError("actor block face must be in [1, 6]")
    return face


def _identity_words(value: object, label: str) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        source = value.tolist()
    elif not isinstance(value, (str, bytes, bytearray)) and isinstance(
        value,
        Sequence,
    ):
        source = value
    else:
        raise TypeError(f"{label} must be a word sequence")
    words = tuple(_integer(row, label) for row in source)
    if len(words) != IDENTITY_HASH_WORDS or any(
        row < 0 or row > 0xFFFFFFFF for row in words
    ):
        raise ValueError(f"{label} must contain eight uint32 words")
    return words


def _key(value: object, label: str) -> tuple[int, ...]:
    return _identity_words(value, label)


def _native_id(value: object, label: str, *, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\0" in value
        or len(value) > maximum
    ):
        raise ValueError(f"{label} must be a bounded native String ID")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _nonnegative_int(value: object, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _positive_int(value: object, label: str) -> int:
    result = _integer(value, label)
    if result <= 0 or result > 2**31 - 1:
        raise ValueError(f"{label} must be a positive int32")
    return result


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "NATIVE_ACTION_RESOLUTION_SCHEMA",
    "NATIVE_ACTION_RESOLUTION_VERSION",
    "NativeCraftingExecutionContext",
    "RegionNativeBlockIdentity",
    "RegionNativeBlockIdentityCatalog",
    "build_native_break_request",
    "build_native_craft_request",
    "build_native_place_request",
    "build_native_use_request",
    "compile_region_native_block_identity_catalog",
    "native_action_resolution_contract",
    "native_action_resolution_contract_sha256",
    "resolve_native_recipe_id",
    "resolve_native_place_item_block_id",
    "resolve_region_native_block_id",
]
