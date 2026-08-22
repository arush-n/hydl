"""Host compiler from World's public native interaction evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp

from hytalegym.jax.combat.block_interactions.schema.contract import (
    BLOCK_INTERACTION_BREAK,
    BLOCK_INTERACTION_PLACE,
    INTERACTION_MOVEMENT_ALL_MASK,
    NO_INTERACTION_PROGRAM,
)
from hytalegym.jax.combat.block_interactions.execution.runtime import (
    empty_block_interaction_program,
)
from hytalegym.jax.combat.block_interactions.schema.types import (
    BlockInteractionProgram,
)
from hytalegym.worldgen import (
    NATIVE_ITEM_INTERACTION_VERSION,
    NATIVE_ITEM_INTERACTION_VERSION_V3,
    NativeBreakBlockPayload,
    NativeInteractionNode,
    NativePlaceBlockPayload,
    native_item_interaction_contract_sha256,
    native_item_interaction_v3_contract_sha256,
)


_BREAK_CLASS = "BreakBlockInteraction"
_PLACE_CLASS = "PlaceBlockInteraction"


@dataclass(frozen=True)
class CompiledNativeBlockInteraction:
    """Device program plus host-only authored placement identity."""

    program: BlockInteractionProgram
    implementation_class: str
    use_latest_target: bool
    break_tool_id: str
    break_match_tool: bool
    place_block_asset_id: str
    allow_drag_placement: bool
    native_evidence_version: int
    native_contract_sha256: str


def block_interaction_program_from_native_node(
    node: NativeInteractionNode,
    *,
    batch_size: int,
    entity_count: int,
    animation_duration_seconds: float | None = None,
    next_program_id: int = NO_INTERACTION_PROGRAM,
    failed_program_id: int = NO_INTERACTION_PROGRAM,
    native_evidence_version: int | None = None,
) -> CompiledNativeBlockInteraction:
    """Compile one resolved Break/Place node without content-name branches.

    The public World row is host evidence. Strings required by the eventual
    World executor remain host-only; all scheduler and movement fields become
    fixed-shape JAX arrays. Other interaction classes are rejected instead of
    being misclassified as block mutations.
    """

    if NATIVE_ITEM_INTERACTION_VERSION < 2:
        raise RuntimeError(
            "native item interaction v2 is required for movement and payloads"
        )
    if not isinstance(node, NativeInteractionNode):
        raise TypeError("node must be NativeInteractionNode")
    if isinstance(node.wait_for_data_from, bool) or node.wait_for_data_from != 0:
        raise ValueError(
            "Break/Place compilation requires native Client data semantics"
        )
    effects_present = _boolean(node.effects_present, "effects_present")
    wait_for_animation = _boolean(
        node.wait_for_animation_to_finish,
        "wait_for_animation_to_finish",
    )
    movement_present = _boolean(
        node.movement_effects_present,
        "movement_effects_present",
    )
    movement_disable_all = _boolean(
        node.movement_disable_all,
        "movement_disable_all",
    )
    cancel_on_item_change = _boolean(
        node.cancel_on_item_change,
        "cancel_on_item_change",
    )
    harvest = _boolean(node.harvest, "harvest")
    use_latest_target = _boolean(
        node.use_latest_target,
        "use_latest_target",
    )
    leaf = node.implementation_class.rsplit(".", 1)[-1]
    evidence_version = _evidence_version(
        native_evidence_version,
        node.payload,
    )
    break_tool_id = ""
    break_match_tool = False
    place_asset_id = ""
    allow_drag = False
    remove_item = True
    if leaf == _BREAK_CLASS:
        kind = BLOCK_INTERACTION_BREAK
        if node.payload is None:
            if evidence_version >= NATIVE_ITEM_INTERACTION_VERSION_V3:
                raise ValueError(
                    "native v3 BreakBlockInteraction requires "
                    "NativeBreakBlockPayload"
                )
        elif isinstance(node.payload, NativeBreakBlockPayload):
            if evidence_version < NATIVE_ITEM_INTERACTION_VERSION_V3:
                raise ValueError(
                    "NativeBreakBlockPayload requires native evidence v3"
                )
            break_tool_id = node.payload.tool_id
            break_match_tool = _boolean(
                node.payload.match_tool,
                "payload.match_tool",
            )
            if break_match_tool and not break_tool_id:
                raise ValueError(
                    "exact native Break tool matching requires a tool ID"
                )
        else:
            raise ValueError(
                "BreakBlockInteraction requires NativeBreakBlockPayload "
                "or a v2 empty payload"
            )
    elif leaf == _PLACE_CLASS:
        kind = BLOCK_INTERACTION_PLACE
        if not isinstance(node.payload, NativePlaceBlockPayload):
            raise ValueError("PlaceBlockInteraction requires NativePlaceBlockPayload")
        # Native PlaceBlockInteraction.BlockTypeToPlace is an optional
        # override. An empty identity means "use the held item's block key";
        # it is valid authored evidence, not a missing payload.
        place_asset_id = node.payload.block_asset_id
        remove_item = _boolean(
            node.payload.remove_item_in_hand,
            "payload.remove_item_in_hand",
        )
        allow_drag = _boolean(
            node.payload.allow_drag_placement,
            "payload.allow_drag_placement",
        )
    else:
        raise ValueError(
            "native node is not a BreakBlockInteraction or PlaceBlockInteraction"
        )

    run_time = _finite_nonnegative(node.run_time, "run_time")
    start_delay = _finite_nonnegative(node.start_delay, "start_delay")
    speed = _finite_nonnegative(
        node.horizontal_speed_multiplier,
        "horizontal_speed_multiplier",
    )
    animation = (
        0.0
        if animation_duration_seconds is None
        else _finite_nonnegative(
            animation_duration_seconds,
            "animation_duration_seconds",
        )
    )
    if wait_for_animation and animation_duration_seconds is None:
        raise ValueError(
            "animation duration evidence is required when native waits for it"
        )
    movement_mask = _movement_mask(node.movement_lock_mask)
    if not effects_present and (
        start_delay != 0.0 or wait_for_animation or movement_present
    ):
        raise ValueError("effect fields cannot exist without effects")
    if not movement_present and (movement_disable_all or movement_mask != 0):
        raise ValueError("movement fields cannot exist without movement evidence")
    if movement_disable_all and (
        not movement_present or movement_mask != INTERACTION_MOVEMENT_ALL_MASK
    ):
        raise ValueError(
            "native disable-all requires present movement effects and all bits"
        )
    _mapped_child(node.next_interaction_id, next_program_id, "next")
    _mapped_child(node.failed_interaction_id, failed_program_id, "failed")

    shape = (batch_size, entity_count)
    program = empty_block_interaction_program(
        batch_size,
        entity_count=entity_count,
    )._replace(
        kind=jnp.full(shape, kind, dtype=jnp.int32),
        run_time_seconds=jnp.full(shape, run_time, dtype=jnp.float32),
        animation_duration_seconds=jnp.full(
            shape,
            animation,
            dtype=jnp.float32,
        ),
        start_delay_seconds=jnp.full(
            shape,
            start_delay,
            dtype=jnp.float32,
        ),
        wait_for_animation_to_finish=jnp.full(
            shape,
            wait_for_animation,
            dtype=jnp.bool_,
        ),
        horizontal_speed_multiplier=jnp.full(
            shape,
            speed,
            dtype=jnp.float32,
        ),
        movement_effects_present=jnp.full(
            shape,
            movement_present,
            dtype=jnp.bool_,
        ),
        movement_disable_all=jnp.full(
            shape,
            movement_disable_all,
            dtype=jnp.bool_,
        ),
        movement_lock_mask=jnp.full(
            shape,
            movement_mask,
            dtype=jnp.int32,
        ),
        cancel_on_item_change=jnp.full(
            shape,
            cancel_on_item_change,
            dtype=jnp.bool_,
        ),
        next_program_id=jnp.full(
            shape,
            next_program_id,
            dtype=jnp.int32,
        ),
        failed_program_id=jnp.full(
            shape,
            failed_program_id,
            dtype=jnp.int32,
        ),
        harvest=jnp.full(shape, harvest, dtype=jnp.bool_),
        remove_item_in_hand=jnp.full(
            shape,
            remove_item,
            dtype=jnp.bool_,
        ),
    )
    return CompiledNativeBlockInteraction(
        program=program,
        implementation_class=node.implementation_class,
        use_latest_target=use_latest_target,
        break_tool_id=break_tool_id,
        break_match_tool=break_match_tool,
        place_block_asset_id=place_asset_id,
        allow_drag_placement=allow_drag,
        native_evidence_version=evidence_version,
        native_contract_sha256=(
            native_item_interaction_v3_contract_sha256()
            if evidence_version >= NATIVE_ITEM_INTERACTION_VERSION_V3
            else native_item_interaction_contract_sha256()
        ),
    )


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _evidence_version(
    value: int | None,
    payload: object,
) -> int:
    if value is None:
        return (
            NATIVE_ITEM_INTERACTION_VERSION_V3
            if isinstance(payload, NativeBreakBlockPayload)
            else NATIVE_ITEM_INTERACTION_VERSION
        )
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("native_evidence_version must be an integer")
    if value not in (
        NATIVE_ITEM_INTERACTION_VERSION,
        NATIVE_ITEM_INTERACTION_VERSION_V3,
    ):
        raise ValueError("native_evidence_version is unsupported")
    return value


def _boolean(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _movement_mask(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("movement_lock_mask must be an integer")
    if not 0 <= value <= INTERACTION_MOVEMENT_ALL_MASK:
        raise ValueError("movement_lock_mask exceeds the seven-bit contract")
    return value


def _mapped_child(interaction_id: str, program_id: int, name: str) -> None:
    if isinstance(program_id, bool) or not isinstance(program_id, int):
        raise TypeError(f"{name}_program_id must be an integer")
    if interaction_id and program_id == NO_INTERACTION_PROGRAM:
        raise ValueError(f"{name} interaction must be mapped to a numeric program ID")
    if not interaction_id and program_id != NO_INTERACTION_PROGRAM:
        raise ValueError(f"{name}_program_id cannot exist without a native child")


__all__ = [
    "CompiledNativeBlockInteraction",
    "block_interaction_program_from_native_node",
]
