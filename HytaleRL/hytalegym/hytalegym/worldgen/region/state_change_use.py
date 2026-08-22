"""Exact, bounded Region block-state transitions for generic Use."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json

from hytalegym.worldgen.block_actions import block_semantic_key
from hytalegym.worldgen.native_world_actions import (
    native_block_use_evidence_contract_sha256,
)
from hytalegym.worldgen.region.block_action_catalog import (
    resolve_region_block_asset_references,
)
from hytalegym.worldgen.region.block_semantics import (
    RegionBlockSemanticEntry,
    region_block_semantic_contract_sha256,
)
from hytalegym.worldgen.surrogate.assets import (
    HytaleAssetArchive,
    HytaleAssetError,
)
from hytalegym.worldgen.surrogate.semantics import (
    LocalBlockSemanticsResolver,
)


REGION_STATE_CHANGE_USE_SCHEMA = "hytalerl_region_state_change_use_v1"
REGION_STATE_CHANGE_USE_VERSION = 1
_STATE_SEPARATOR = "_State_Definitions_"


@dataclass(frozen=True)
class RegionStateChangeUseDefinition:
    """One exact current-state-to-next-state mapping from installed assets."""

    source_reference: str
    target_reference: str
    rotation_index: int
    source_semantic_key: tuple[int, ...]
    target_semantic_key: tuple[int, ...]


def compile_region_state_change_use_definitions(
    archive: HytaleAssetArchive,
    palette: Sequence[RegionBlockSemanticEntry],
) -> tuple[RegionStateChangeUseDefinition, ...]:
    """Compile only exact one-node, side-independent ``ChangeState`` uses.

    The source must already be a native-captured Region semantic row. The
    target is resolved from the installed 0.5.7 asset graph, but this host
    compiler does not claim target geometry. Device publication separately
    requires an exact target-state geometry row in the same loaded Region.
    """

    if not isinstance(archive, HytaleAssetArchive):
        raise TypeError("archive must be a HytaleAssetArchive")
    resolver = LocalBlockSemanticsResolver(archive)
    definitions: dict[tuple[int, ...], RegionStateChangeUseDefinition] = {}
    for entry, reference in resolve_region_block_asset_references(
        archive,
        palette,
    ):
        try:
            source = resolver.resolve(
                reference,
                rotation=entry.rotation_index,
            )
        except HytaleAssetError:
            continue
        if source.is_door or source.filler != 0 or not source.use_state_changes:
            continue
        asset_id, source_state = _reference_state(reference)
        targets = tuple(
            target
            for current, target in source.use_state_changes
            if current == source_state
        )
        if len(targets) != 1:
            continue
        target_reference = _state_reference(asset_id, targets[0])
        try:
            target = resolver.resolve(
                target_reference,
                rotation=entry.rotation_index,
            )
        except HytaleAssetError:
            continue
        if (
            target.is_door
            or target.filler != 0
            or target.use_state_changes != source.use_state_changes
        ):
            continue
        source_key = block_semantic_key(reference, entry.rotation_index)
        target_key = block_semantic_key(
            target_reference,
            entry.rotation_index,
        )
        if source_key != entry.semantic_key or source_key == target_key:
            continue
        definition = RegionStateChangeUseDefinition(
            source_reference=reference,
            target_reference=target_reference,
            rotation_index=entry.rotation_index,
            source_semantic_key=source_key,
            target_semantic_key=target_key,
        )
        previous = definitions.setdefault(source_key, definition)
        if previous != definition:
            raise HytaleAssetError(
                "Region state-change source has conflicting targets"
            )
    return tuple(
        definitions[key]
        for key in sorted(definitions)
    )


def region_state_change_use_contract() -> dict[str, object]:
    """Return the portable host-side transition contract."""

    return {
        "schema": REGION_STATE_CHANGE_USE_SCHEMA,
        "version": REGION_STATE_CHANGE_USE_VERSION,
        "dependencies": {
            "region_block_semantic_contract_sha256": (
                region_block_semantic_contract_sha256()
            ),
            "native_block_use_evidence_contract_sha256": (
                native_block_use_evidence_contract_sha256()
            ),
        },
        "source_state": "native_captured_region_semantic_key",
        "transition": (
            "installed_asset_exact_one_node_ChangeState_nonempty_mapping_"
            "UpdateBlockState_false"
        ),
        "excluded": (
            "doors_fillers_multi_node_empty_mapping_UpdateBlockState_true_"
            "unresolved_or_ambiguous_state"
        ),
        "target_geometry": "not_claimed_by_host_contract",
        "published_provenance": (
            "source_exact_with_representative_native_outcome_calibration"
        ),
        "representative_native_calibration": {
            "source": "Deco_Lantern",
            "target": "*Deco_Lantern_State_Definitions_Off",
            "fixed_seed": 570057,
            "random_control_required": True,
            "scope": "semantic_outcome_not_dynamic_light_propagation",
        },
        "fail_closed": True,
    }


def region_state_change_use_contract_sha256() -> str:
    payload = json.dumps(
        region_state_change_use_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _reference_state(reference: str) -> tuple[str, str]:
    if not reference.startswith("*"):
        return reference, "default"
    asset_id, separator, state = reference[1:].partition(_STATE_SEPARATOR)
    if not separator or not asset_id or not state:
        raise HytaleAssetError("malformed block state reference")
    return asset_id, state


def _state_reference(asset_id: str, state: str) -> str:
    if state == "default":
        return asset_id
    return f"*{asset_id}{_STATE_SEPARATOR}{state}"


__all__ = [
    "REGION_STATE_CHANGE_USE_SCHEMA",
    "REGION_STATE_CHANGE_USE_VERSION",
    "RegionStateChangeUseDefinition",
    "compile_region_state_change_use_definitions",
    "region_state_change_use_contract",
    "region_state_change_use_contract_sha256",
]
