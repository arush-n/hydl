"""Local block support catalog construction and validation."""
from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.worldgen.block_actions import (
    BLOCK_SEMANTIC_KEY_WORDS,
    block_asset_key,
)
from hytalegym.worldgen.block_support import (
    LocalBlockSupportSemantics,
    SUPPORT_BLOCK_SET_CAPACITY_0_5_7,
    SUPPORT_CLAUSE_CAPACITY_0_5_7,
    SUPPORT_FACE_COUNT,
    SUPPORT_FACE_TYPE_CAPACITY_0_5_7,
    SUPPORT_FLUID_CAPACITY_0_5_7,
    SUPPORT_GROUP_CAPACITY_0_5_7,
    SUPPORT_ROTATION_COUNT,
    SUPPORT_TAG_CAPACITY_0_5_7,
)

from hytalegym.jax.world.block_support._primitives import (
    BlockSupportCatalog,
    _bits,
    _capacity,
    _face_types,
    _vocabulary,
)


Array = jax.Array
BLOCK_SUPPORT_CASCADE_SCHEMA = "hytalerl_jax_block_support_cascade_v1"
BLOCK_SUPPORT_CASCADE_VERSION = 1

BLOCK_SUPPORT_DIAGNOSTIC_CATALOG = 1 << 0
BLOCK_SUPPORT_DIAGNOSTIC_NEIGHBOUR = 1 << 1
BLOCK_SUPPORT_DIAGNOSTIC_FLUID = 1 << 2
BLOCK_SUPPORT_DIAGNOSTIC_SINGLE_CELL = 1 << 3
BLOCK_SUPPORT_DIAGNOSTIC_CAPACITY = 1 << 4
BLOCK_SUPPORT_DIAGNOSTIC_INPUT = 1 << 5


def _validate_catalog(catalog: BlockSupportCatalog) -> int:
    if not isinstance(catalog, BlockSupportCatalog):
        raise TypeError("catalog must be BlockSupportCatalog")
    if catalog.entry_mask.ndim != 1:
        raise ValueError("catalog.entry_mask has the wrong shape")
    capacity = catalog.entry_mask.shape[0]
    if capacity < 1:
        raise ValueError("catalog must have positive capacity")
    expected = {
        "semantic_key": (capacity, BLOCK_SEMANTIC_KEY_WORDS),
        "asset_key": (capacity, BLOCK_SEMANTIC_KEY_WORDS),
        "rotation_index": (capacity,),
        "material_empty": (capacity,),
        "player_placement_marks_deco": (capacity,),
        "support_dependent": (capacity,),
        "support_drop_type": (capacity,),
        "max_support_distance": (capacity,),
        "tag_bits": (capacity,),
        "block_set_bits": (capacity,),
        "supporting_face_types": (
            capacity,
            SUPPORT_ROTATION_COUNT,
            SUPPORT_FACE_COUNT,
        ),
        "group_mask": (capacity, SUPPORT_GROUP_CAPACITY_0_5_7),
        "group_neighbour_slot": (
            capacity,
            SUPPORT_GROUP_CAPACITY_0_5_7,
        ),
        "group_block_face": (
            capacity,
            SUPPORT_GROUP_CAPACITY_0_5_7,
        ),
        "group_neighbour_face": (
            capacity,
            SUPPORT_GROUP_CAPACITY_0_5_7,
        ),
    }
    clause_shape = (
        capacity,
        SUPPORT_GROUP_CAPACITY_0_5_7,
        SUPPORT_CLAUSE_CAPACITY_0_5_7,
    )
    for field in (
        "clause_mask",
        "clause_face_type_mask",
        "clause_self_face_type_mask",
        "clause_block_type_valid",
        "clause_fluid_id",
        "clause_tag_mask",
        "clause_block_set_mask",
        "clause_match_self",
        "clause_support",
        "clause_allow_support_propagation",
    ):
        expected[field] = clause_shape
    expected["clause_block_type_key"] = clause_shape + (
        BLOCK_SEMANTIC_KEY_WORDS,
    )
    dtypes = {
        "entry_mask": jnp.bool_,
        "semantic_key": jnp.uint32,
        "asset_key": jnp.uint32,
        "rotation_index": jnp.uint8,
        "material_empty": jnp.bool_,
        "player_placement_marks_deco": jnp.bool_,
        "support_dependent": jnp.bool_,
        "support_drop_type": jnp.uint8,
        "max_support_distance": jnp.uint8,
        "tag_bits": jnp.uint16,
        "block_set_bits": jnp.uint8,
        "supporting_face_types": jnp.uint32,
        "group_mask": jnp.bool_,
        "group_neighbour_slot": jnp.uint8,
        "group_block_face": jnp.uint8,
        "group_neighbour_face": jnp.uint8,
        "clause_mask": jnp.bool_,
        "clause_face_type_mask": jnp.uint32,
        "clause_self_face_type_mask": jnp.uint32,
        "clause_block_type_valid": jnp.bool_,
        "clause_block_type_key": jnp.uint32,
        "clause_fluid_id": jnp.uint8,
        "clause_tag_mask": jnp.uint16,
        "clause_block_set_mask": jnp.uint8,
        "clause_match_self": jnp.uint8,
        "clause_support": jnp.uint8,
        "clause_allow_support_propagation": jnp.bool_,
    }
    if catalog.entry_mask.dtype != dtypes["entry_mask"]:
        raise ValueError(
            "catalog.entry_mask has the wrong dtype"
        )
    for field, shape in expected.items():
        value = getattr(catalog, field)
        if value.shape != shape:
            raise ValueError(f"catalog.{field} has the wrong shape")
        if value.dtype != dtypes[field]:
            raise ValueError(f"catalog.{field} has the wrong dtype")
    return capacity


def block_support_catalog_from_local(
    entries: Sequence[LocalBlockSupportSemantics],
    *,
    entry_capacity: int | None = None,
) -> BlockSupportCatalog:
    """Pack exact installed semantics and reject every census overflow."""

    source = tuple(entries)
    if not source or any(
        not isinstance(value, LocalBlockSupportSemantics) for value in source
    ):
        raise TypeError(
            "entries must be a nonempty LocalBlockSupportSemantics sequence"
        )
    capacity = len(source) if entry_capacity is None else _capacity(
        entry_capacity,
        "entry",
    )
    if capacity < len(source):
        raise ValueError("entry capacity is smaller than the source")
    if len({value.semantic_key for value in source}) != len(source):
        raise ValueError("support catalog semantic keys must be unique")

    face_types = _vocabulary(
        (
            text
            for entry in source
            for text in _face_types(entry)
        ),
        SUPPORT_FACE_TYPE_CAPACITY_0_5_7,
        "face type",
    )
    tags = _vocabulary(
        (
            clause.tag_id
            for entry in source
            for group in entry.groups
            for clause in group.clauses
            if clause.tag_id is not None
        ),
        SUPPORT_TAG_CAPACITY_0_5_7,
        "support tag",
    )
    block_sets = _vocabulary(
        (
            text
            for entry in source
            for text in (
                *entry.block_set_ids,
                *(
                    clause.block_set_id
                    for group in entry.groups
                    for clause in group.clauses
                    if clause.block_set_id is not None
                ),
            )
        ),
        SUPPORT_BLOCK_SET_CAPACITY_0_5_7,
        "block set",
    )
    fluids = _vocabulary(
        (
            clause.fluid_id
            for entry in source
            for group in entry.groups
            for clause in group.clauses
            if clause.fluid_id is not None
        ),
        SUPPORT_FLUID_CAPACITY_0_5_7,
        "fluid",
    )
    face_index = {value: index for index, value in enumerate(face_types)}
    tag_index = {value: index for index, value in enumerate(tags)}
    set_index = {value: index for index, value in enumerate(block_sets)}
    fluid_index = {
        value: index + 1 for index, value in enumerate(fluids)
    }

    shape = (capacity,)
    groups = (capacity, SUPPORT_GROUP_CAPACITY_0_5_7)
    clauses = groups + (SUPPORT_CLAUSE_CAPACITY_0_5_7,)
    entry_mask = np.zeros(shape, dtype=np.bool_)
    semantic_key = np.zeros(
        shape + (BLOCK_SEMANTIC_KEY_WORDS,),
        dtype=np.uint32,
    )
    asset_key = np.zeros_like(semantic_key)
    rotation = np.zeros(shape, dtype=np.uint8)
    material_empty = np.zeros(shape, dtype=np.bool_)
    placement_deco = np.zeros(shape, dtype=np.bool_)
    dependent = np.zeros(shape, dtype=np.bool_)
    drop_type = np.zeros(shape, dtype=np.uint8)
    maximum = np.zeros(shape, dtype=np.uint8)
    tag_bits = np.zeros(shape, dtype=np.uint16)
    set_bits = np.zeros(shape, dtype=np.uint8)
    supporting = np.zeros(
        (capacity, SUPPORT_ROTATION_COUNT, SUPPORT_FACE_COUNT),
        dtype=np.uint32,
    )
    group_mask = np.zeros(groups, dtype=np.bool_)
    neighbour_slot = np.zeros(groups, dtype=np.uint8)
    block_face = np.zeros(groups, dtype=np.uint8)
    neighbour_face = np.zeros(groups, dtype=np.uint8)
    clause_mask = np.zeros(clauses, dtype=np.bool_)
    clause_face = np.zeros(clauses, dtype=np.uint32)
    clause_self_face = np.zeros(clauses, dtype=np.uint32)
    clause_block_valid = np.zeros(clauses, dtype=np.bool_)
    clause_block_key = np.zeros(
        clauses + (BLOCK_SEMANTIC_KEY_WORDS,),
        dtype=np.uint32,
    )
    clause_fluid = np.zeros(clauses, dtype=np.uint8)
    clause_tag = np.zeros(clauses, dtype=np.uint16)
    clause_set = np.zeros(clauses, dtype=np.uint8)
    clause_self = np.zeros(clauses, dtype=np.uint8)
    clause_support = np.zeros(clauses, dtype=np.uint8)
    clause_propagation = np.zeros(clauses, dtype=np.bool_)

    for index, entry in enumerate(source):
        if (
            not 0 <= entry.rotation_index < SUPPORT_ROTATION_COUNT
            or len(entry.groups) > SUPPORT_GROUP_CAPACITY_0_5_7
            or len(entry.supporting_by_rotation)
            != SUPPORT_ROTATION_COUNT
        ):
            raise ValueError("support entry exceeds its fixed shape")
        entry_mask[index] = True
        semantic_key[index] = entry.semantic_key
        asset_key[index] = entry.asset_key
        rotation[index] = entry.rotation_index
        material_empty[index] = entry.material_empty
        placement_deco[index] = entry.player_placement_marks_deco
        dependent[index] = entry.support_dependent
        drop_type[index] = entry.support_drop_type
        maximum[index] = entry.max_support_distance
        tag_bits[index] = _bits(entry.tag_ids, tag_index)
        set_bits[index] = _bits(entry.block_set_ids, set_index)
        for candidate_rotation, offers in enumerate(
            entry.supporting_by_rotation
        ):
            for offer in offers:
                if offer.filler is not None and (0, 0, 0) not in offer.filler:
                    continue
                supporting[
                    index,
                    candidate_rotation,
                    offer.face,
                ] |= np.uint32(1 << face_index[offer.face_type])
        for group_index, group in enumerate(entry.groups):
            if len(group.clauses) > SUPPORT_CLAUSE_CAPACITY_0_5_7:
                raise ValueError("support group exceeds clause capacity")
            group_mask[index, group_index] = True
            neighbour_slot[index, group_index] = group.neighbour_slot
            block_face[index, group_index] = group.block_face
            neighbour_face[index, group_index] = group.neighbour_face
            for clause_index, clause in enumerate(group.clauses):
                if (
                    clause.filler is not None
                    and (0, 0, 0) not in clause.filler
                ):
                    continue
                target = (index, group_index, clause_index)
                clause_mask[target] = True
                if clause.face_type is not None:
                    clause_face[target] = np.uint32(
                        1 << face_index[clause.face_type]
                    )
                if clause.self_face_type is not None:
                    clause_self_face[target] = np.uint32(
                        1 << face_index[clause.self_face_type]
                    )
                if clause.block_type_id is not None:
                    clause_block_valid[target] = True
                    clause_block_key[target] = block_asset_key(
                        clause.block_type_id
                    )
                if clause.fluid_id is not None:
                    clause_fluid[target] = fluid_index[clause.fluid_id]
                if clause.tag_id is not None:
                    clause_tag[target] = np.uint16(
                        1 << tag_index[clause.tag_id]
                    )
                if clause.block_set_id is not None:
                    clause_set[target] = np.uint8(
                        1 << set_index[clause.block_set_id]
                    )
                clause_self[target] = clause.match_self
                clause_support[target] = clause.support
                clause_propagation[
                    target
                ] = clause.allow_support_propagation

    return BlockSupportCatalog(
        entry_mask=jnp.asarray(entry_mask),
        semantic_key=jnp.asarray(semantic_key),
        asset_key=jnp.asarray(asset_key),
        rotation_index=jnp.asarray(rotation),
        material_empty=jnp.asarray(material_empty),
        player_placement_marks_deco=jnp.asarray(placement_deco),
        support_dependent=jnp.asarray(dependent),
        support_drop_type=jnp.asarray(drop_type),
        max_support_distance=jnp.asarray(maximum),
        tag_bits=jnp.asarray(tag_bits),
        block_set_bits=jnp.asarray(set_bits),
        supporting_face_types=jnp.asarray(supporting),
        group_mask=jnp.asarray(group_mask),
        group_neighbour_slot=jnp.asarray(neighbour_slot),
        group_block_face=jnp.asarray(block_face),
        group_neighbour_face=jnp.asarray(neighbour_face),
        clause_mask=jnp.asarray(clause_mask),
        clause_face_type_mask=jnp.asarray(clause_face),
        clause_self_face_type_mask=jnp.asarray(clause_self_face),
        clause_block_type_valid=jnp.asarray(clause_block_valid),
        clause_block_type_key=jnp.asarray(clause_block_key),
        clause_fluid_id=jnp.asarray(clause_fluid),
        clause_tag_mask=jnp.asarray(clause_tag),
        clause_block_set_mask=jnp.asarray(clause_set),
        clause_match_self=jnp.asarray(clause_self),
        clause_support=jnp.asarray(clause_support),
        clause_allow_support_propagation=jnp.asarray(
            clause_propagation
        ),
    )
