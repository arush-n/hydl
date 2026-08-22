"""Exact installed-asset support rules for bounded block-physics cascades."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from itertools import combinations

from hytalegym.worldgen.block_actions import (
    SUPPORT_DROP_BREAK,
    SUPPORT_DROP_DESTROY,
    SUPPORT_DROP_FALL,
    SUPPORT_DROP_NONE,
    block_asset_key,
    block_semantic_key,
)
from hytalegym.worldgen.surrogate.assets import (
    HytaleAssetArchive,
    HytaleAssetError,
)
from hytalegym.worldgen.surrogate.semantics import (
    LocalBlockSemanticsResolver,
)


BLOCK_SUPPORT_SCHEMA = "hytalerl_block_support_physics_v1"
BLOCK_SUPPORT_VERSION = 1

SUPPORT_MATCH_IGNORED = 0
SUPPORT_MATCH_REQUIRED = 1
SUPPORT_MATCH_DISALLOWED = 2

SUPPORT_FACE_COUNT = 26
SUPPORT_ROTATION_COUNT = 64
SUPPORT_GROUP_CAPACITY_0_5_7 = 98
SUPPORT_CLAUSE_CAPACITY_0_5_7 = 12
SUPPORT_OFFER_CAPACITY_0_5_7 = 26
SUPPORT_FACE_TYPE_CAPACITY_0_5_7 = 17
SUPPORT_TAG_CAPACITY_0_5_7 = 13
SUPPORT_BLOCK_SET_CAPACITY_0_5_7 = 1
SUPPORT_FLUID_CAPACITY_0_5_7 = 2

_STATE_SEPARATOR = "_State_Definitions_"
_FULL_CUBE_DRAWS = {"Cube", "CubeWithModel", "GizmoCube"}
_FACE_NAMES = (
    "Up",
    "Down",
    "North",
    "East",
    "South",
    "West",
    "UpNorth",
    "UpSouth",
    "UpEast",
    "UpWest",
    "DownNorth",
    "DownSouth",
    "DownEast",
    "DownWest",
    "NorthEast",
    "SouthEast",
    "SouthWest",
    "NorthWest",
    "UpNorthEast",
    "UpSouthEast",
    "UpSouthWest",
    "UpNorthWest",
    "DownNorthEast",
    "DownSouthEast",
    "DownSouthWest",
    "DownNorthWest",
)
_FACE_DIRECTIONS = (
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, -1),
    (1, 0, 0),
    (0, 0, 1),
    (-1, 0, 0),
    (0, 1, -1),
    (0, 1, 1),
    (1, 1, 0),
    (-1, 1, 0),
    (0, -1, -1),
    (0, -1, 1),
    (1, -1, 0),
    (-1, -1, 0),
    (1, 0, -1),
    (1, 0, 1),
    (-1, 0, 1),
    (-1, 0, -1),
    (1, 1, -1),
    (1, 1, 1),
    (-1, 1, 1),
    (-1, 1, -1),
    (1, -1, -1),
    (1, -1, 1),
    (-1, -1, 1),
    (-1, -1, -1),
)
_FACE_INDEX = {
    direction: index for index, direction in enumerate(_FACE_DIRECTIONS)
}
SUPPORT_NEIGHBOUR_OFFSETS = tuple(
    (x, y, z)
    for y in (-1, 0, 1)
    for z in (-1, 0, 1)
    for x in (-1, 0, 1)
    if (x, y, z) != (0, 0, 0)
)
_NEIGHBOUR_INDEX = {
    offset: index for index, offset in enumerate(SUPPORT_NEIGHBOUR_OFFSETS)
}
_MERGED_FACES = {
    "All": tuple(range(SUPPORT_FACE_COUNT)),
    "BlockSides": tuple(range(6)),
    "CardinalDirections": (2, 3, 4, 5),
    "Horizontal": (2, 14, 3, 15, 4, 16, 5, 17),
    "UpCardinalDirections": (6, 8, 7, 9),
    "DownCardinalDirections": (10, 12, 11, 13),
}


@dataclass(frozen=True)
class LocalSupportClause:
    """One decoded ``RequiredBlockFaceSupport`` row."""

    face_type: str | None
    self_face_type: str | None
    block_set_id: str | None
    block_type_id: str | None
    fluid_id: str | None
    tag_id: str | None
    match_self: int
    support: int
    allow_support_propagation: bool
    filler: tuple[tuple[int, int, int], ...] | None


@dataclass(frozen=True)
class LocalSupportGroup:
    """Clauses tested against one native connecting-face neighbour."""

    neighbour_slot: int
    block_face: int
    neighbour_face: int
    clauses: tuple[LocalSupportClause, ...]


@dataclass(frozen=True)
class LocalSupportingFace:
    """One rotated face type offered by a candidate supporting block."""

    face: int
    face_type: str
    filler: tuple[tuple[int, int, int], ...] | None


@dataclass(frozen=True)
class LocalBlockSupportSemantics:
    """One block/rotation's exact support rule and matching metadata."""

    reference: str
    rotation_index: int
    semantic_key: tuple[int, ...]
    asset_key: tuple[int, ...]
    material_empty: bool
    player_placement_marks_deco: bool
    support_dependent: bool
    support_drop_type: int
    max_support_distance: int
    groups: tuple[LocalSupportGroup, ...]
    supporting_by_rotation: tuple[tuple[LocalSupportingFace, ...], ...]
    tag_ids: tuple[str, ...]
    block_set_ids: tuple[str, ...]


def compile_local_block_support_semantics(
    archive: HytaleAssetArchive,
    references: Sequence[tuple[str, int]],
) -> tuple[LocalBlockSupportSemantics, ...]:
    """Resolve a bounded reference set against installed inherited assets."""

    if not isinstance(archive, HytaleAssetArchive):
        raise TypeError("archive must be a HytaleAssetArchive")
    if isinstance(references, (str, bytes, bytearray)):
        raise TypeError("references must be a sequence")
    source = tuple(
        (_reference(reference), _rotation(rotation))
        for reference, rotation in references
    )
    if not source or len(set(source)) != len(source):
        raise ValueError("references must be nonempty and unique")

    resolver = LocalBlockSemanticsResolver(archive)
    resolved: list[
        tuple[str, int, Mapping[str, object], Mapping[str, object]]
    ] = []
    required_sets: set[str] = set()
    for reference, rotation in source:
        item, _ = resolver.resolve_item_asset(reference)
        block = _mapping(item.get("BlockType"), "BlockType")
        required_sets.update(_support_block_sets(block))
        resolved.append((reference, rotation, item, block))

    set_members: dict[str, frozenset[str]] = {}
    unique_references = tuple(sorted({reference for reference, _ in source}))
    for set_id in sorted(required_sets):
        value = archive.resolve_block_set_block_types(
            set_id,
            unique_references,
        )
        set_members[set_id] = frozenset(value.block_type_ids)

    entries = []
    tag_cache: dict[str, tuple[str, ...]] = {}
    for reference, rotation, item, block in resolved:
        asset_id = _reference_asset_id(reference)
        if asset_id not in tag_cache:
            tag_cache[asset_id] = _inherited_tag_ids(
                archive,
                asset_id,
                (),
            )
        tags = tag_cache[asset_id]
        memberships = tuple(
            set_id
            for set_id in sorted(required_sets)
            if reference in set_members[set_id]
        )
        entries.append(
            resolve_local_block_support_semantics(
                reference,
                item,
                block,
                rotation_index=rotation,
                tag_ids=tags,
                block_set_ids=memberships,
            )
        )
    return tuple(sorted(entries, key=lambda value: value.semantic_key))


def resolve_local_block_support_semantics(
    reference: str,
    item: Mapping[str, object],
    block: Mapping[str, object],
    *,
    rotation_index: int = 0,
    tag_ids: Iterable[str] = (),
    block_set_ids: Iterable[str] = (),
) -> LocalBlockSupportSemantics:
    """Resolve native post-decode defaults and all 64 rotated offerings."""

    identifier = _reference(reference)
    _mapping(item, "item")
    block_data = _mapping(block, "BlockType")
    rotation = _rotation(rotation_index)
    maximum = _integer(
        block_data.get("MaxSupportDistance", 0),
        "MaxSupportDistance",
        minimum=0,
        maximum=14,
    )
    support = _effective_support(block_data)
    groups = _support_groups(support, rotation)
    dependent = bool(support) or maximum > 0
    drop_type = _support_drop_type(
        block_data.get("SupportDropType", "BREAK"),
        dependent,
    )
    supporting = tuple(
        _supporting_faces(block_data, candidate_rotation)
        for candidate_rotation in range(SUPPORT_ROTATION_COUNT)
    )
    tags = tuple(sorted({_text(value, "tag ID") for value in tag_ids}))
    sets = tuple(
        sorted({_text(value, "block set ID") for value in block_set_ids})
    )
    _require_capacity(len(groups), SUPPORT_GROUP_CAPACITY_0_5_7, "groups")
    _require_capacity(
        max((len(group.clauses) for group in groups), default=0),
        SUPPORT_CLAUSE_CAPACITY_0_5_7,
        "clauses per group",
    )
    _require_capacity(
        max((len(value) for value in supporting), default=0),
        SUPPORT_OFFER_CAPACITY_0_5_7,
        "supporting faces",
    )
    return LocalBlockSupportSemantics(
        reference=identifier,
        rotation_index=rotation,
        semantic_key=block_semantic_key(identifier, rotation),
        asset_key=block_asset_key(identifier),
        material_empty=block_data.get("Material", "Empty") == "Empty",
        player_placement_marks_deco=_player_placement_marks_deco(block_data),
        support_dependent=dependent,
        support_drop_type=drop_type,
        max_support_distance=maximum,
        groups=groups,
        supporting_by_rotation=supporting,
        tag_ids=tags,
        block_set_ids=sets,
    )


def block_support_contract() -> dict[str, object]:
    """Return the fixed-capacity exact dry/single-cell support boundary."""

    return {
        "schema": BLOCK_SUPPORT_SCHEMA,
        "version": BLOCK_SUPPORT_VERSION,
        "server_version": "0.5.7",
        "native_source": {
            "decode_defaults": "BlockType.afterDecode",
            "predicate": "BlockPhysicsUtil.testBlockPhysics",
            "dispatch": "BlockPhysicsUtil.applyBlockPhysics",
            "neighbourhood": "BlockFace.connectingFaceOffsets",
            "radius": "BlockPhysicsSystems.MAX_SUPPORT_RADIUS_14",
        },
        "installed_capacities": {
            "faces": SUPPORT_FACE_COUNT,
            "rotations": SUPPORT_ROTATION_COUNT,
            "groups": SUPPORT_GROUP_CAPACITY_0_5_7,
            "clauses_per_group": SUPPORT_CLAUSE_CAPACITY_0_5_7,
            "supporting_faces": SUPPORT_OFFER_CAPACITY_0_5_7,
            "face_types": SUPPORT_FACE_TYPE_CAPACITY_0_5_7,
            "tags": SUPPORT_TAG_CAPACITY_0_5_7,
            "block_sets": SUPPORT_BLOCK_SET_CAPACITY_0_5_7,
            "fluids": SUPPORT_FLUID_CAPACITY_0_5_7,
        },
        "semantics": {
            "implicit_support": (
                "empty_material_non_tech_without_Support_requires_Down_Full"
            ),
            "implicit_supporting": (
                "solid_Cube_CubeWithModel_or_GizmoCube_offers_Full_all_faces"
            ),
            "rotation": "RotationTuple_yaw_pitch_roll_all_64_indices",
            "conditions": [
                "FaceType",
                "SelfFaceType_using_native_neighbour_rotation_quirk",
                "BlockSetId",
                "BlockTypeId",
                "FluidId",
                "MatchSelf",
                "Support",
                "AllowSupportPropagation",
                "Filler",
                "TagId",
            ],
            "propagation": (
                "minimum_neighbour_support_plus_one_bounded_by_"
                "MaxSupportDistance_with_15_anchor_special_case"
            ),
            "player_placement": (
                "normal_non_override_player_marks_support_15_when_"
                "IgnoreSupportWhenPlaced_or_UseDefaultDropWhenPlaced"
            ),
            "deco_tick": "current_support_15_is_ignored",
            "drop": "BREAK_DESTROY_or_FALL",
        },
        "exact_runtime_scope": "covered_dry_single_cell_root_blocks",
        "unsupported": {
            "fluid_identity_missing": "fail_closed",
            "filler_or_protruding_block": "fail_closed",
            "uncovered_neighbour": "fail_closed_waiting_chunk",
            "capacity": "fail_closed",
        },
        "provenance": "installed_inherited_assets_and_0_5_7_server_source",
    }


def block_support_contract_sha256() -> str:
    payload = json.dumps(
        block_support_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _effective_support(
    block: Mapping[str, object],
) -> Mapping[str, object]:
    value = block.get("Support")
    if value is None:
        if (
            block.get("Material", "Empty") == "Empty"
            and block.get("Group") != "@Tech"
        ):
            return {"Down": [{"FaceType": "Full"}]}
        return {}
    return _mapping(value, "BlockType.Support")


def _player_placement_marks_deco(
    block: Mapping[str, object],
) -> bool:
    ignored = _boolean(
        block.get("IgnoreSupportWhenPlaced", False),
        "BlockType.IgnoreSupportWhenPlaced",
    )
    gathering = block.get("Gathering")
    if gathering is None:
        return ignored
    data = _mapping(gathering, "BlockType.Gathering")
    return ignored or _boolean(
        data.get("UseDefaultDropWhenPlaced", False),
        "BlockType.Gathering.UseDefaultDropWhenPlaced",
    )


def _support_groups(
    support: Mapping[str, object],
    rotation: int,
) -> tuple[LocalSupportGroup, ...]:
    by_face: dict[int, list[LocalSupportClause]] = {}
    for face_name, rows in support.items():
        faces = _faces(face_name)
        if not isinstance(rows, list):
            raise ValueError("BlockType.Support rows must be an array")
        for value in rows:
            data = _mapping(value, "RequiredBlockFaceSupport")
            rotate = _boolean(data.get("Rotate", True), "Support.Rotate")
            clause = _support_clause(data, rotation if rotate else 0)
            for face in faces:
                selected = _rotate_face(face, rotation) if rotate else face
                by_face.setdefault(selected, []).append(clause)

    groups = []
    for face in range(SUPPORT_FACE_COUNT):
        clauses = tuple(by_face.get(face, ()))
        if not clauses:
            continue
        for slot, neighbour_face in _connecting_faces(face):
            groups.append(
                LocalSupportGroup(
                    neighbour_slot=slot,
                    block_face=face,
                    neighbour_face=neighbour_face,
                    clauses=clauses,
                )
            )
    return tuple(groups)


def _support_clause(
    data: Mapping[str, object],
    rotation: int,
) -> LocalSupportClause:
    filler = _filler(data.get("Filler"), "Support.Filler")
    return LocalSupportClause(
        face_type=_optional_text(data.get("FaceType"), "Support.FaceType"),
        self_face_type=_optional_text(
            data.get("SelfFaceType"),
            "Support.SelfFaceType",
        ),
        block_set_id=_optional_text(
            data.get("BlockSetId"),
            "Support.BlockSetId",
        ),
        block_type_id=_optional_text(
            data.get("BlockTypeId"),
            "Support.BlockTypeId",
        ),
        fluid_id=_optional_text(data.get("FluidId"), "Support.FluidId"),
        tag_id=_optional_text(data.get("TagId"), "Support.TagId"),
        match_self=_match(data.get("MatchSelf", "Ignored")),
        support=_match(data.get("Support", "Required")),
        allow_support_propagation=_boolean(
            data.get("AllowSupportPropagation", True),
            "Support.AllowSupportPropagation",
        ),
        filler=(
            None
            if filler is None
            else tuple(_rotate_vector(value, rotation) for value in filler)
        ),
    )


def _supporting_faces(
    block: Mapping[str, object],
    rotation: int,
) -> tuple[LocalSupportingFace, ...]:
    value = block.get("Supporting")
    if value is None:
        if (
            block.get("Material", "Empty") == "Solid"
            and block.get("DrawType", "Cube") in _FULL_CUBE_DRAWS
        ):
            return tuple(
                LocalSupportingFace(face, "Full", None)
                for face in range(SUPPORT_FACE_COUNT)
            )
        return ()
    supporting = _mapping(value, "BlockType.Supporting")
    rows = []
    for face_name, offers in supporting.items():
        if not isinstance(offers, list):
            raise ValueError("BlockType.Supporting rows must be an array")
        for face in _faces(face_name):
            selected = _rotate_face(face, rotation)
            for raw in offers:
                data = _mapping(raw, "BlockFaceSupport")
                filler = _filler(data.get("Filler"), "Supporting.Filler")
                rows.append(
                    LocalSupportingFace(
                        face=selected,
                        face_type=_text(
                            data.get("FaceType", "Full"),
                            "Supporting.FaceType",
                        ),
                        filler=(
                            None
                            if filler is None
                            else tuple(
                                _rotate_vector(item, rotation)
                                for item in filler
                            )
                        ),
                    )
                )
    return tuple(rows)


def _connecting_faces(face: int) -> tuple[tuple[int, int], ...]:
    direction = _FACE_DIRECTIONS[face]
    axes = tuple(index for index, value in enumerate(direction) if value)
    rows = []
    for size in range(1, len(axes) + 1):
        for selected in combinations(axes, size):
            neighbour = list(direction)
            offset = [0, 0, 0]
            for axis in selected:
                neighbour[axis] *= -1
                offset[axis] = direction[axis]
            rows.append(
                (
                    _NEIGHBOUR_INDEX[tuple(offset)],
                    _FACE_INDEX[tuple(neighbour)],
                )
            )
    return tuple(rows)


def _support_block_sets(block: Mapping[str, object]) -> set[str]:
    result = set()
    for rows in _effective_support(block).values():
        if not isinstance(rows, list):
            raise ValueError("BlockType.Support rows must be an array")
        for value in rows:
            data = _mapping(value, "RequiredBlockFaceSupport")
            set_id = _optional_text(data.get("BlockSetId"), "BlockSetId")
            if set_id is not None:
                result.add(set_id)
    return result


def _inherited_tag_ids(
    archive: HytaleAssetArchive,
    asset_id: str,
    stack: tuple[str, ...],
) -> tuple[str, ...]:
    if asset_id in stack:
        raise HytaleAssetError(
            "item tag inheritance cycle: "
            + " -> ".join(stack + (asset_id,))
        )
    if len(stack) >= 32:
        raise HytaleAssetError("item tag inheritance exceeds capacity")
    data, _ = archive.load_item_asset(asset_id)
    result: set[str] = set()
    parent = data.get("Parent")
    if parent is not None:
        result.update(
            _inherited_tag_ids(
                archive,
                _text(parent, "item Parent"),
                stack + (asset_id,),
            )
        )
    tags = data.get("Tags")
    if tags is not None:
        for key, values in _mapping(tags, "item Tags").items():
            tag = _text(key, "tag key")
            result.add(tag)
            if not isinstance(values, list):
                raise ValueError("tag values must be an array")
            for raw in values:
                value = _text(raw, "tag value")
                result.add(value)
                result.add(f"{tag}={value}")
    return tuple(sorted(result))


def _faces(value: object) -> tuple[int, ...]:
    name = _text(value, "block face")
    if name in _MERGED_FACES:
        return _MERGED_FACES[name]
    try:
        return (_FACE_NAMES.index(name),)
    except ValueError as error:
        raise ValueError(f"unknown block face {name!r}") from error


def _rotate_face(face: int, rotation: int) -> int:
    return _FACE_INDEX[_rotate_vector(_FACE_DIRECTIONS[face], rotation)]


def _rotate_vector(
    value: tuple[int, int, int],
    rotation: int,
) -> tuple[int, int, int]:
    x, y, z = value
    yaw = rotation & 3
    pitch = (rotation >> 2) & 3
    roll = (rotation >> 4) & 3
    if roll == 1:
        x, y = -y, x
    elif roll == 2:
        x, y = -x, -y
    elif roll == 3:
        x, y = y, -x
    if pitch == 1:
        y, z = -z, y
    elif pitch == 2:
        y, z = -y, -z
    elif pitch == 3:
        y, z = z, -y
    if yaw == 1:
        x, z = z, -x
    elif yaw == 2:
        x, z = -x, -z
    elif yaw == 3:
        x, z = -z, x
    return x, y, z


def _filler(
    value: object,
    label: str,
) -> tuple[tuple[int, int, int], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    rows = tuple(_vector(item, label) for item in value)
    _require_capacity(len(rows), 1, label)
    return rows


def _vector(value: object, label: str) -> tuple[int, int, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} vectors must be objects")
    return tuple(
        _integer(value.get(axis.upper()), f"{label}.{axis.upper()}")
        for axis in ("x", "y", "z")
    )


def _match(value: object) -> int:
    name = _text(value, "support match").casefold()
    result = {
        "ignored": SUPPORT_MATCH_IGNORED,
        "required": SUPPORT_MATCH_REQUIRED,
        "disallowed": SUPPORT_MATCH_DISALLOWED,
    }.get(name)
    if result is None:
        raise ValueError(f"unknown support match {value!r}")
    return result


def _support_drop_type(value: object, dependent: bool) -> int:
    if not dependent:
        return SUPPORT_DROP_NONE
    name = _text(value, "SupportDropType").casefold()
    result = {
        "break": SUPPORT_DROP_BREAK,
        "destroy": SUPPORT_DROP_DESTROY,
        "fall": SUPPORT_DROP_FALL,
    }.get(name)
    if result is None:
        raise ValueError(f"unknown SupportDropType {value!r}")
    return result


def _reference(value: object) -> str:
    result = _text(value, "block reference")
    if any(character in result for character in "/\\\0"):
        raise ValueError("block reference is unsafe")
    return result


def _reference_asset_id(reference: str) -> str:
    if not reference.startswith("*"):
        return reference
    value = reference[1:]
    asset_id, separator, state = value.partition(_STATE_SEPARATOR)
    if not separator or not asset_id or not state:
        raise ValueError("state block reference is malformed")
    return asset_id


def _rotation(value: object) -> int:
    return _integer(value, "rotation_index", minimum=0, maximum=63)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _optional_text(value: object, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} is below its minimum")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds its maximum")
    return value


def _require_capacity(value: int, capacity: int, label: str) -> None:
    if value > capacity:
        raise ValueError(f"{label} exceeds installed 0.5.7 capacity")


__all__ = [
    "BLOCK_SUPPORT_SCHEMA",
    "BLOCK_SUPPORT_VERSION",
    "LocalBlockSupportSemantics",
    "LocalSupportClause",
    "LocalSupportGroup",
    "LocalSupportingFace",
    "SUPPORT_BLOCK_SET_CAPACITY_0_5_7",
    "SUPPORT_CLAUSE_CAPACITY_0_5_7",
    "SUPPORT_FACE_COUNT",
    "SUPPORT_FACE_TYPE_CAPACITY_0_5_7",
    "SUPPORT_FLUID_CAPACITY_0_5_7",
    "SUPPORT_GROUP_CAPACITY_0_5_7",
    "SUPPORT_MATCH_DISALLOWED",
    "SUPPORT_MATCH_IGNORED",
    "SUPPORT_MATCH_REQUIRED",
    "SUPPORT_NEIGHBOUR_OFFSETS",
    "SUPPORT_OFFER_CAPACITY_0_5_7",
    "SUPPORT_ROTATION_COUNT",
    "SUPPORT_TAG_CAPACITY_0_5_7",
    "block_support_contract",
    "block_support_contract_sha256",
    "compile_local_block_support_semantics",
    "resolve_local_block_support_semantics",
]
