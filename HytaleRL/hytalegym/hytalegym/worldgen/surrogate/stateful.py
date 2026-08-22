"""Physical door state tables derived from local 0.5.7 assets."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from hytalegym.worldgen.region import CAPTURE_BLOCKS_PER_AXIS, WORLD_HEIGHT
from hytalegym.worldgen.surrogate.assets import (
    HytaleAssetError,
    decode_prefab_filler,
)
from hytalegym.worldgen.surrogate.contract import SurrogateWorldCapacity
from hytalegym.worldgen.surrogate.palette import (
    CompiledSemanticPalette,
    compile_local_semantic_palette,
)
from hytalegym.worldgen.surrogate.prefab import CompiledPrefabOverlay
from hytalegym.worldgen.surrogate.semantics import (
    LocalBlockSemantics,
    LocalBlockSemanticsResolver,
)

DOOR_CLOSED = 0
DOOR_OPEN_IN = 1
DOOR_OPEN_OUT = 2
DOOR_PHYSICAL_STATE_COUNT = 3
DOOR_SIDE_BEHIND = 0
DOOR_SIDE_FRONT = 1
DOOR_SIDE_COUNT = 2


@dataclass(frozen=True)
class CompiledStatefulBlocks:
    """Padded physical state and interaction tables for one tile."""

    block_mask: np.ndarray
    block_key: np.ndarray
    identity_key: np.ndarray
    partner_identity_key: np.ndarray
    state_count: np.ndarray
    initial_state: np.ndarray
    yaw: np.ndarray
    variant_cell_code: np.ndarray
    success_target: np.ndarray
    blocked_target: np.ndarray
    transition_mask: np.ndarray
    asset_ids: tuple[str, ...]
    state_names: tuple[tuple[str, ...], ...]

    @property
    def block_count(self) -> int:
        return int(np.count_nonzero(self.block_mask))

    @property
    def logical_bytes(self) -> int:
        return sum(
            value.nbytes
            for value in (
                self.block_mask,
                self.block_key,
                self.identity_key,
                self.partner_identity_key,
                self.state_count,
                self.initial_state,
                self.yaw,
                self.variant_cell_code,
                self.success_target,
                self.blocked_target,
                self.transition_mask,
            )
        )


@dataclass(frozen=True)
class ResolvedStatefulPrefabOverlay:
    """Sparse structure geometry plus per-cell physical state slots."""

    source: CompiledPrefabOverlay
    cell_code: np.ndarray
    state_slot: np.ndarray
    palette: CompiledSemanticPalette
    stateful: CompiledStatefulBlocks

    @property
    def cell_count(self) -> int:
        return self.source.cell_count


@dataclass(frozen=True)
class _SimpleStateChange:
    index: int
    default: LocalBlockSemantics
    names: tuple[str, ...]
    variants: tuple[LocalBlockSemantics, ...]
    target_by_state: tuple[int, ...]
    initial_state: int


def resolve_stateful_prefab_overlay(
    overlay: CompiledPrefabOverlay,
    resolver: LocalBlockSemanticsResolver,
    *,
    capacity: SurrogateWorldCapacity | None = None,
) -> ResolvedStatefulPrefabOverlay:
    """Resolve static cells, vertical doors, and exact ChangeState uses."""

    if not isinstance(overlay, CompiledPrefabOverlay):
        raise TypeError("overlay must be a CompiledPrefabOverlay")
    if not isinstance(resolver, LocalBlockSemanticsResolver):
        raise TypeError("resolver must be a LocalBlockSemanticsResolver")
    layout = SurrogateWorldCapacity() if capacity is None else capacity
    if not isinstance(layout, SurrogateWorldCapacity):
        raise TypeError("capacity must be a SurrogateWorldCapacity")

    semantics: list[LocalBlockSemantics] = []
    active: list[tuple[int, LocalBlockSemantics]] = []
    doors: list[
        tuple[
            int,
            int,
            LocalBlockSemantics,
            LocalBlockSemantics,
            LocalBlockSemantics,
        ]
    ] = []
    simple_changes: list[_SimpleStateChange] = []
    active_indices = [int(value) for value in np.flatnonzero(overlay.cell_mask)]
    key_to_index = {int(overlay.block_key[index]): index for index in active_indices}
    for index in active_indices:
        asset_code = int(overlay.asset_code[index])
        if not 0 <= asset_code < len(overlay.asset_ids):
            raise HytaleAssetError("prefab overlay asset code is invalid")
        rotation_value = int(overlay.block_rotation[index])
        filler_value = int(overlay.filler[index])
        rotation = None if rotation_value < 0 else rotation_value
        filler = None if filler_value < 0 else filler_value
        default = resolver.resolve(
            overlay.asset_ids[asset_code],
            rotation=rotation,
            filler=filler,
        )
        support = int(overlay.support[index])
        default = replace(default, support=support)
        semantics.append(default)
        active.append((index, default))
        if default.is_door:
            if default.interaction_id != "Door":
                raise HytaleAssetError(
                    "only the vertical 0.5.7 Door interaction is supported"
                )
            if default.rotation >= 4:
                raise HytaleAssetError("vertical doors support yaw rotation only")
            owner_index = _filler_owner_index(
                overlay,
                index,
                key_to_index,
            )
            open_in = resolver.resolve(
                f"*{default.asset_id}_State_Definitions_OpenDoorIn",
                rotation=rotation,
                filler=filler,
            )
            open_out = resolver.resolve(
                f"*{default.asset_id}_State_Definitions_OpenDoorOut",
                rotation=rotation,
                filler=filler,
            )
            open_in = replace(open_in, support=support)
            open_out = replace(open_out, support=support)
            if not open_in.is_door or not open_out.is_door:
                raise HytaleAssetError("door states must remain door block types")
            semantics.extend((open_in, open_out))
            doors.append((index, owner_index, default, open_in, open_out))
        elif default.use_state_changes and default.filler == 0:
            simple = _resolve_simple_state_change(
                index,
                default,
                resolver,
                support=support,
            )
            if simple is not None:
                semantics.extend(simple.variants)
                simple_changes.append(simple)
        if len(doors) + len(simple_changes) > layout.stateful_block_capacity:
            raise HytaleAssetError(
                "prefab stateful blocks exceed stateful block capacity "
                f"{layout.stateful_block_capacity}"
            )

    partners = _double_door_partners(overlay, doors, resolver)
    if DOOR_PHYSICAL_STATE_COUNT > layout.states_per_block:
        raise HytaleAssetError(
            "door physical states exceed states-per-block capacity "
            f"{layout.states_per_block}"
        )
    if any(
        len(simple.names) > layout.states_per_block for simple in simple_changes
    ):
        raise HytaleAssetError(
            "ChangeState physical states exceed states-per-block capacity "
            f"{layout.states_per_block}"
        )
    palette = compile_local_semantic_palette(semantics, capacity=layout)
    cell_code = np.zeros(overlay.cell_mask.shape, dtype=np.uint16)
    state_slot = np.full(overlay.cell_mask.shape, -1, dtype=np.int32)
    for index, semantic in active:
        cell_code[index] = palette.cell_code(
            semantic.reference,
            semantic.rotation,
            semantic.support,
            semantic.filler,
        )

    block_capacity = layout.stateful_block_capacity
    states = layout.states_per_block
    block_mask = np.zeros(block_capacity, dtype=np.bool_)
    block_key = np.full(block_capacity, np.iinfo(np.int32).max, dtype=np.int32)
    identity_key = np.full(
        block_capacity,
        np.iinfo(np.int32).max,
        dtype=np.int32,
    )
    partner_identity_key = np.full(
        block_capacity,
        np.iinfo(np.int32).max,
        dtype=np.int32,
    )
    state_count = np.zeros(block_capacity, dtype=np.uint8)
    initial_state = np.zeros(block_capacity, dtype=np.uint8)
    yaw = np.zeros(block_capacity, dtype=np.uint8)
    variant_code = np.zeros((block_capacity, states), dtype=np.uint16)
    success_target = np.zeros(
        (block_capacity, states, DOOR_SIDE_COUNT),
        dtype=np.uint8,
    )
    blocked_target = np.zeros_like(success_target)
    transition_mask = np.zeros_like(success_target, dtype=np.bool_)
    asset_ids: list[str] = []
    state_names: list[tuple[str, ...]] = []
    for slot, (index, owner_index, default, open_in, open_out) in enumerate(doors):
        block_mask[slot] = True
        block_key[slot] = overlay.block_key[index]
        identity_key[slot] = overlay.block_key[owner_index]
        partner_owner = partners.get(owner_index)
        if partner_owner is not None:
            partner_identity_key[slot] = overlay.block_key[partner_owner]
        state_count[slot] = DOOR_PHYSICAL_STATE_COUNT
        yaw[slot] = default.rotation
        variant_code[slot, :DOOR_PHYSICAL_STATE_COUNT] = [
            palette.cell_code(
                default.reference,
                default.rotation,
                default.support,
                default.filler,
            ),
            palette.cell_code(
                open_in.reference,
                open_in.rotation,
                open_in.support,
                open_in.filler,
            ),
            palette.cell_code(
                open_out.reference,
                open_out.rotation,
                open_out.support,
                open_out.filler,
            ),
        ]
        success_target[slot, DOOR_CLOSED] = [DOOR_OPEN_OUT, DOOR_OPEN_IN]
        success_target[slot, DOOR_OPEN_IN] = DOOR_CLOSED
        success_target[slot, DOOR_OPEN_OUT] = DOOR_CLOSED
        blocked_target[slot, DOOR_CLOSED] = DOOR_CLOSED
        blocked_target[slot, DOOR_OPEN_IN] = DOOR_OPEN_IN
        blocked_target[slot, DOOR_OPEN_OUT] = DOOR_OPEN_OUT
        transition_mask[slot, :DOOR_PHYSICAL_STATE_COUNT] = True
        state_slot[index] = slot
        asset_ids.append(default.asset_id)
        state_names.append(("default", "OpenDoorIn", "OpenDoorOut"))
    for offset, simple in enumerate(simple_changes, start=len(doors)):
        block_mask[offset] = True
        block_key[offset] = overlay.block_key[simple.index]
        identity_key[offset] = overlay.block_key[simple.index]
        state_count[offset] = len(simple.names)
        initial_state[offset] = simple.initial_state
        yaw[offset] = simple.default.rotation
        variant_code[offset, : len(simple.names)] = [
            palette.cell_code(
                variant.reference,
                variant.rotation,
                variant.support,
                variant.filler,
            )
            for variant in simple.variants
        ]
        for source, target in enumerate(simple.target_by_state):
            success_target[offset, source] = source
            blocked_target[offset, source] = source
            if target >= 0:
                success_target[offset, source] = target
                transition_mask[offset, source] = True
        state_slot[simple.index] = offset
        asset_ids.append(simple.default.asset_id)
        state_names.append(simple.names)

    arrays = (
        cell_code,
        state_slot,
        block_mask,
        block_key,
        identity_key,
        partner_identity_key,
        state_count,
        initial_state,
        yaw,
        variant_code,
        success_target,
        blocked_target,
        transition_mask,
    )
    for value in arrays:
        value.flags.writeable = False
    return ResolvedStatefulPrefabOverlay(
        source=overlay,
        cell_code=cell_code,
        state_slot=state_slot,
        palette=palette,
        stateful=CompiledStatefulBlocks(
            block_mask=block_mask,
            block_key=block_key,
            identity_key=identity_key,
            partner_identity_key=partner_identity_key,
            state_count=state_count,
            initial_state=initial_state,
            yaw=yaw,
            variant_cell_code=variant_code,
            success_target=success_target,
            blocked_target=blocked_target,
            transition_mask=transition_mask,
            asset_ids=tuple(asset_ids),
            state_names=tuple(state_names),
        ),
    )


def _resolve_simple_state_change(
    index: int,
    default: LocalBlockSemantics,
    resolver: LocalBlockSemanticsResolver,
    *,
    support: int,
) -> _SimpleStateChange | None:
    names = ["default"]
    for source, target in default.use_state_changes:
        for name in (source, target):
            if name not in names:
                names.append(name)
    initial_name = "default" if default.state is None else default.state
    if initial_name not in names:
        return None

    variants = []
    for name in names:
        reference = (
            default.asset_id
            if name == "default"
            else f"*{default.asset_id}_State_Definitions_{name}"
        )
        variant = replace(
            resolver.resolve(
                reference,
                rotation=default.rotation,
            ),
            support=support,
        )
        if (
            variant.is_door
            or variant.filler != 0
            or variant.use_state_changes != default.use_state_changes
        ):
            return None
        variants.append(variant)

    state_index = {name: slot for slot, name in enumerate(names)}
    targets = [-1] * len(names)
    for source, target in default.use_state_changes:
        targets[state_index[source]] = state_index[target]
    return _SimpleStateChange(
        index=index,
        default=default,
        names=tuple(names),
        variants=tuple(variants),
        target_by_state=tuple(targets),
        initial_state=state_index[initial_name],
    )


def _double_door_partners(
    overlay: CompiledPrefabOverlay,
    doors: list[
        tuple[
            int,
            int,
            LocalBlockSemantics,
            LocalBlockSemantics,
            LocalBlockSemantics,
        ]
    ],
    resolver: LocalBlockSemanticsResolver,
) -> dict[int, int]:
    roots = {
        owner_index: default
        for index, owner_index, default, _open_in, _open_out in doors
        if index == owner_index
    }
    owner_by_position = {
        _decode_key(int(overlay.block_key[owner])): owner for owner in roots
    }
    partners: dict[int, int] = {}
    for owner, default in roots.items():
        canonical = resolver.resolve(default.asset_id, rotation=0)
        if not canonical.collision_boxes:
            raise HytaleAssetError("door base hitbox contains no collision boxes")
        maximum_x = max(
            box.maximum[0] for box in canonical.collision_boxes
        )
        # DoorInteraction casts maxX before doubling. Python's int() has the
        # same finite-value truncation-toward-zero behavior as the Java cast.
        offset = int(maximum_x) * 2 - 1
        position = _decode_key(int(overlay.block_key[owner]))
        rotated = _rotate_y((offset, 0, 0), default.rotation)
        target = tuple(position[axis] + rotated[axis] for axis in range(3))
        partner = owner_by_position.get(target)
        if partner is None:
            continue
        candidate = roots[partner]
        if (
            candidate.rotation != (default.rotation + 2) % 4
            or candidate.hitbox_type != default.hitbox_type
        ):
            continue
        partners[owner] = partner
    for owner, partner in partners.items():
        if partners.get(partner) != owner:
            raise HytaleAssetError("double-door pairing is not reciprocal")
    return partners


def _rotate_y(
    value: tuple[int, int, int],
    quarter_turns: int,
) -> tuple[int, int, int]:
    x, y, z = value
    return (
        (x, y, z),
        (z, y, -x),
        (-x, y, -z),
        (-z, y, x),
    )[quarter_turns]


def _filler_owner_index(
    overlay: CompiledPrefabOverlay,
    index: int,
    key_to_index: dict[int, int],
) -> int:
    filler = int(overlay.filler[index])
    if filler < 0:
        return index
    offset = decode_prefab_filler(filler)
    cell = _decode_key(int(overlay.block_key[index]))
    owner = tuple(cell[axis] - offset[axis] for axis in range(3))
    if any(
        value < 0 or value >= (CAPTURE_BLOCKS_PER_AXIS if axis != 1 else WORLD_HEIGHT)
        for axis, value in enumerate(owner)
    ):
        raise HytaleAssetError("filler owner leaves the surrogate tile")
    owner_index = key_to_index.get(_encode_key(owner))
    if owner_index is None:
        raise HytaleAssetError("filler owner is absent from the prefab overlay")
    if int(overlay.asset_code[owner_index]) != int(overlay.asset_code[index]) or int(
        overlay.block_rotation[owner_index]
    ) != int(overlay.block_rotation[index]):
        raise HytaleAssetError("filler owner block or rotation disagrees")
    if int(overlay.filler[owner_index]) >= 0:
        raise HytaleAssetError("filler must point directly to a root block")
    return owner_index


def _decode_key(key: int) -> tuple[int, int, int]:
    area = CAPTURE_BLOCKS_PER_AXIS**2
    y, remainder = divmod(key, area)
    z, x = divmod(remainder, CAPTURE_BLOCKS_PER_AXIS)
    return x, y, z


def _encode_key(position: tuple[int, int, int]) -> int:
    return (
        position[1] * CAPTURE_BLOCKS_PER_AXIS * CAPTURE_BLOCKS_PER_AXIS
        + position[2] * CAPTURE_BLOCKS_PER_AXIS
        + position[0]
    )


__all__ = [
    "DOOR_CLOSED",
    "DOOR_OPEN_IN",
    "DOOR_OPEN_OUT",
    "DOOR_PHYSICAL_STATE_COUNT",
    "DOOR_SIDE_BEHIND",
    "DOOR_SIDE_FRONT",
    "CompiledStatefulBlocks",
    "ResolvedStatefulPrefabOverlay",
    "resolve_stateful_prefab_overlay",
]
