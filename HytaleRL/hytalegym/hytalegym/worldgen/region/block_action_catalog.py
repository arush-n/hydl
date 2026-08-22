"""Compile Region block identities into exact host block-action semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json

from hytalegym.worldgen.block_actions import (
    LocalBlockActionSemantics,
    LocalToolSpec,
    block_action_contract_sha256,
    block_asset_key,
    resolve_default_gather_spec,
    resolve_local_block_action_semantics,
)
from hytalegym.worldgen.block_affordances import (
    GATHER_TYPES,
    block_affordance_dictionary_sha256,
    resolve_local_block_affordance,
)
from hytalegym.worldgen.drop_programs import (
    BLOCK_DROP_PROGRAM_CAPACITY,
    LocalDropProgram,
    LocalDropProgramCompiler,
    block_drop_program_contract_sha256,
)
from hytalegym.worldgen.region.block_semantics import (
    RegionBlockSemanticEntry,
    region_block_semantic_contract_sha256,
)
from hytalegym.worldgen.surrogate.assets import (
    UNARMED_GATHERING_PREFIX,
    AssetProvenance,
    HytaleAssetArchive,
    HytaleAssetError,
)
from hytalegym.worldgen.surrogate.semantics import (
    LocalBlockSemanticsResolver,
)


REGION_BLOCK_ACTION_CATALOG_SCHEMA = (
    "hytalerl_region_block_action_catalog_v4"
)
REGION_BLOCK_ACTION_CATALOG_VERSION = 4
_STATE_SEPARATOR = "_State_Definitions_"


@dataclass(frozen=True)
class RegionBlockActionCatalog:
    """Canonical host catalog plus non-semantic installed-asset provenance."""

    entries: tuple[LocalBlockActionSemantics, ...]
    gather_defaults: tuple[LocalToolSpec | None, ...]
    drop_programs: tuple[LocalDropProgram, ...]
    source_block_semantic_sha256: str
    asset_evidence_sha256: str

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        defaults = tuple(self.gather_defaults)
        programs = tuple(self.drop_programs)
        if not entries:
            raise ValueError("Region block-action catalog cannot be empty")
        if any(
            not isinstance(entry, LocalBlockActionSemantics)
            for entry in entries
        ):
            raise TypeError("Region block-action entries have wrong type")
        if entries != tuple(sorted(entries, key=lambda item: item.semantic_key)):
            raise ValueError("Region block-action entries are not canonical")
        if len({entry.semantic_key for entry in entries}) != len(entries):
            raise ValueError("Region block-action semantic keys must be unique")
        if len(defaults) != len(GATHER_TYPES) or defaults[0] is not None:
            raise ValueError("Region gather defaults do not match dictionary")
        for index, spec in enumerate(defaults[1:], start=1):
            if spec is not None and not isinstance(spec, LocalToolSpec):
                raise TypeError("Region gather default has wrong type")
            if spec is not None and spec.gather_type_index != index:
                raise ValueError("installed gather default has wrong index")
        if len(programs) > BLOCK_DROP_PROGRAM_CAPACITY:
            raise ValueError("Region drop-program capacity exceeded")
        if programs != tuple(
            sorted(programs, key=lambda item: item.semantic_sha256)
        ):
            raise ValueError("Region drop programs are not canonical")
        if len({item.semantic_sha256 for item in programs}) != len(programs):
            raise ValueError("Region drop programs must be unique")
        published_programs = {
            item.semantic_sha256 for item in programs
        }
        route_programs = {
            route.program_sha256
            for entry in entries
            for route in (
                entry.interaction_tool_route.drop,
                entry.tool_drop,
                entry.soft_drop,
                entry.harvest_drop,
            )
            if route.available and route.program_sha256 is not None
        }
        if route_programs != published_programs:
            raise ValueError(
                "Region block routes and drop-program library differ"
            )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "gather_defaults", defaults)
        object.__setattr__(self, "drop_programs", programs)
        object.__setattr__(
            self,
            "source_block_semantic_sha256",
            _sha256(self.source_block_semantic_sha256, "source semantics"),
        )
        object.__setattr__(
            self,
            "asset_evidence_sha256",
            _sha256(self.asset_evidence_sha256, "asset evidence"),
        )

    def semantic_sha256(self) -> str:
        """Hash compiled meaning while excluding replaceable provenance."""

        return _canonical_sha256(
            {
                "schema": REGION_BLOCK_ACTION_CATALOG_SCHEMA,
                "version": REGION_BLOCK_ACTION_CATALOG_VERSION,
                "contract_sha256": (
                    region_block_action_catalog_contract_sha256()
                ),
                "source_block_semantic_sha256": (
                    self.source_block_semantic_sha256
                ),
                "entries": [asdict(entry) for entry in self.entries],
                "gather_defaults": [
                    None if spec is None else asdict(spec)
                    for spec in self.gather_defaults
                ],
                "drop_programs": [
                    asdict(program) for program in self.drop_programs
                ],
            }
        )


def compile_region_block_action_catalog(
    archive: HytaleAssetArchive,
    palette: Sequence[RegionBlockSemanticEntry],
    *,
    source_block_semantic_sha256: str,
) -> RegionBlockActionCatalog:
    """Resolve one complete Region semantic palette against installed assets."""

    if not isinstance(archive, HytaleAssetArchive):
        raise TypeError("archive must be a HytaleAssetArchive")
    source_sha256 = _sha256(
        source_block_semantic_sha256,
        "source semantics",
    )
    unique = _unique_valid_entries(palette)
    resolver = LocalBlockSemanticsResolver(archive)
    references = _asset_references(
        archive,
        resolver,
        {entry.asset_key for entry in unique},
    )
    evidence: dict[tuple[str, str], None] = {}
    drop_list_ids = frozenset(archive.item_drop_list_asset_ids())

    def record(rows: Sequence[AssetProvenance]) -> None:
        for row in rows:
            evidence[(row.entry_path, row.content_sha256)] = None

    def item_lookup(asset_id: str) -> Mapping[str, object]:
        item, provenance = resolver.load_inherited_item_asset(asset_id)
        record(provenance)
        return item

    def named_drop_lookup(
        asset_id: str,
    ) -> Mapping[str, object] | None:
        if asset_id not in drop_list_ids:
            return None
        value, provenance = archive.load_item_drop_list_asset(asset_id)
        record((provenance,))
        return value

    program_compiler = LocalDropProgramCompiler(
        named_drop_lookup,
        item_lookup,
    )
    program_inputs: dict[str, LocalDropProgram] = {}
    programs: dict[str, LocalDropProgram] = {}

    def drop_program_lookup(
        value: str | Mapping[str, object],
    ) -> LocalDropProgram:
        key = (
            f"named\0{value}"
            if isinstance(value, str)
            else "embedded\0"
            + json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
        program = program_inputs.get(key)
        if program is None:
            program = program_compiler.compile(value)
            program_inputs[key] = program
            previous = programs.setdefault(program.semantic_sha256, program)
            if previous != program:
                raise HytaleAssetError(
                    "drop-program semantic SHA-256 collision"
                )
        return program

    compiled: list[LocalBlockActionSemantics] = []
    for entry in unique:
        reference = references[entry.asset_key]
        item, provenance = resolver.resolve_item_asset(reference)
        record(provenance)
        block = item["BlockType"]
        affordance = resolve_local_block_affordance(item, block)
        if (
            not affordance.valid
            or affordance.tags != entry.affordance_tags
            or affordance.gather_type_index != entry.gather_type_index
            or affordance.required_tool_quality
            != entry.required_tool_quality
        ):
            raise HytaleAssetError(
                f"Region affordance differs from installed asset {reference!r}"
            )
        action = resolve_local_block_action_semantics(
            reference,
            item,
            block,
            rotation_index=entry.rotation_index,
            item_lookup=item_lookup,
            drop_program_lookup=drop_program_lookup,
        )
        if action.semantic_key != entry.semantic_key:
            raise HytaleAssetError(
                f"Region semantic key differs from installed asset {reference!r}"
            )
        compiled.append(action)

    authored_defaults = {
        path.removeprefix(UNARMED_GATHERING_PREFIX).removesuffix(".json")
        for path in archive.entry_paths(
            prefix=UNARMED_GATHERING_PREFIX,
            suffix=".json",
        )
    }
    unknown_defaults = authored_defaults.difference(GATHER_TYPES[1:])
    if unknown_defaults:
        raise HytaleAssetError(
            f"unknown unarmed gather assets: {sorted(unknown_defaults)}"
        )
    defaults: list[LocalToolSpec | None] = [None]
    for gather_type in GATHER_TYPES[1:]:
        if gather_type not in authored_defaults:
            defaults.append(None)
            continue
        value, provenance = archive.load_unarmed_gathering_asset(gather_type)
        record((provenance,))
        defaults.append(resolve_default_gather_spec(gather_type, value))

    return RegionBlockActionCatalog(
        entries=tuple(sorted(compiled, key=lambda item: item.semantic_key)),
        gather_defaults=tuple(defaults),
        drop_programs=tuple(
            sorted(programs.values(), key=lambda item: item.semantic_sha256)
        ),
        source_block_semantic_sha256=source_sha256,
        asset_evidence_sha256=_canonical_sha256(
            {
                "entries": [
                    {"entry_path": path, "content_sha256": content_sha256}
                    for path, content_sha256 in sorted(evidence)
                ],
                "drop_list_index_sha256": _canonical_sha256(
                    sorted(drop_list_ids)
                ),
            }
        ),
    )


def region_block_action_catalog_contract() -> dict[str, object]:
    """Publish the host boundary consumed by fixed-shape JAX converters."""

    return {
        "schema": REGION_BLOCK_ACTION_CATALOG_SCHEMA,
        "version": REGION_BLOCK_ACTION_CATALOG_VERSION,
        "server_version": "0.5.7",
        "source": {
            "region_block_semantic_contract_sha256": (
                region_block_semantic_contract_sha256()
            ),
            "block_action_contract_sha256": block_action_contract_sha256(),
            "affordance_dictionary_sha256": (
                block_affordance_dictionary_sha256()
            ),
            "assets": "installed_inherited_item_and_unarmed_gathering_assets",
        },
        "identity": {
            "lookup": "reverse_sha256_asset_key_against_installed_assets",
            "state_reference": "*asset_State_Definitions_state",
            "semantic_key": "must_equal_native_Region_palette_entry",
            "runtime_ordinals": "never_published",
        },
        "outputs": {
            "block_actions": (
                "canonical_unique_LocalBlockActionSemantics_including_"
                "support_dependency_and_exact_Gathering_Tools_route_by_"
                "semantic_key"
            ),
            "gather_defaults": (
                "complete_GATHER_TYPES_indexed_optional_LocalToolSpec_tuple"
            ),
            "drop_programs": (
                "deduplicated_semantic_sha256_sorted_exact_distributions"
            ),
            "jax_adapters": [
                "block_action_catalog_from_local",
                "block_gather_defaults_from_local",
            ],
        },
        "drops": {
            "direct_or_default": "resolved",
            "named_or_embedded_random_program": (
                "exact_distribution_requires_explicit_device_sample"
            ),
            "drop_program_contract_sha256": (
                block_drop_program_contract_sha256()
            ),
            "entity_death": "outside_world_block_catalog",
        },
        "missing_unarmed_gather_asset": (
            "explicit_unavailable_default_never_fabricated"
        ),
        "hashing": {
            "semantic": (
                "compiled_meaning_plus_source_semantic_identity_"
                "asset_provenance_excluded"
            ),
            "asset_evidence": (
                "sorted_relevant_entry_path_and_content_sha256_separate"
            ),
        },
        "unknown_missing_ambiguous_or_disagreeing_block_asset": (
            "reject_catalog"
        ),
        "policy_abi": "unchanged_until_consumer_announces_batched_move",
    }


def region_block_action_catalog_contract_sha256() -> str:
    return _canonical_sha256(region_block_action_catalog_contract())


def resolve_region_block_asset_references(
    archive: HytaleAssetArchive,
    palette: Sequence[RegionBlockSemanticEntry],
) -> tuple[tuple[RegionBlockSemanticEntry, str], ...]:
    """Reverse exact Region keys to installed native block String IDs.

    This is a host-only identity projection.  It deliberately reuses the
    catalog compiler's ambiguity and state-definition checks instead of
    teaching an action consumer to guess a String ID from a digest.
    """

    if not isinstance(archive, HytaleAssetArchive):
        raise TypeError("archive must be a HytaleAssetArchive")
    entries = _unique_valid_entries(palette)
    references = _asset_references(
        archive,
        LocalBlockSemanticsResolver(archive),
        {entry.asset_key for entry in entries},
    )
    return tuple(
        (entry, references[entry.asset_key])
        for entry in entries
    )


def _unique_valid_entries(
    palette: Sequence[RegionBlockSemanticEntry],
) -> tuple[RegionBlockSemanticEntry, ...]:
    if isinstance(palette, (str, bytes, bytearray)) or not isinstance(
        palette,
        Sequence,
    ):
        raise TypeError("palette must be a sequence")
    entries: dict[tuple[int, ...], RegionBlockSemanticEntry] = {}
    for entry in palette:
        if not isinstance(entry, RegionBlockSemanticEntry):
            raise TypeError("palette entries must be RegionBlockSemanticEntry")
        if not entry.valid:
            continue
        previous = entries.setdefault(entry.semantic_key, entry)
        if previous != entry:
            raise HytaleAssetError(
                "Region semantic key has conflicting palette meanings"
            )
    if not entries:
        raise HytaleAssetError("Region semantic palette has no valid blocks")
    return tuple(sorted(entries.values(), key=lambda item: item.semantic_key))


def _asset_references(
    archive: HytaleAssetArchive,
    resolver: LocalBlockSemanticsResolver,
    required: set[tuple[int, ...]],
) -> dict[tuple[int, ...], str]:
    references: dict[tuple[int, ...], str] = {}
    asset_ids = archive.item_asset_ids()
    for asset_id in asset_ids:
        _index_reference(references, required, asset_id)
    if required.difference(references):
        for asset_id in asset_ids:
            item, _ = resolver.load_inherited_item_asset(asset_id)
            block = item.get("BlockType")
            if not isinstance(block, Mapping):
                continue
            state = block.get("State")
            if not isinstance(state, Mapping):
                continue
            definitions = state.get("Definitions")
            if definitions is None:
                continue
            if not isinstance(definitions, Mapping):
                raise HytaleAssetError(
                    f"block asset {asset_id!r} has malformed state definitions"
                )
            for state_id in definitions:
                if not isinstance(state_id, str) or not state_id or any(
                    character in state_id for character in "/\\\0"
                ):
                    continue
                _index_reference(
                    references,
                    required,
                    f"*{asset_id}{_STATE_SEPARATOR}{state_id}",
                )
    missing = sorted(required.difference(references))
    if missing:
        raise HytaleAssetError(
            f"{len(missing)} Region block asset hashes are unavailable"
        )
    return references


def _index_reference(
    references: dict[tuple[int, ...], str],
    required: set[tuple[int, ...]],
    reference: str,
) -> None:
    key = block_asset_key(reference)
    if key not in required:
        return
    previous = references.setdefault(key, reference)
    if previous != reference:
        raise HytaleAssetError("Region block asset hash is ambiguous")


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} SHA-256 must be 64 hexadecimal characters")
    return value.lower()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "REGION_BLOCK_ACTION_CATALOG_SCHEMA",
    "REGION_BLOCK_ACTION_CATALOG_VERSION",
    "RegionBlockActionCatalog",
    "compile_region_block_action_catalog",
    "resolve_region_block_asset_references",
    "region_block_action_catalog_contract",
    "region_block_action_catalog_contract_sha256",
]
