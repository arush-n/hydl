"""Typed reset-time capabilities for native World actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Sequence
from typing import Mapping
from ._primitives import (  # noqa: F401  (re-exported: callers unchanged)
    _boolean,
    _bounded_canonical_string,
    _integer,
    _nonempty_string_or_empty,
    _optional_sha256,
    _optional_target,
    _real,
    _sha256,
)


NATIVE_WORLD_ACTION_CAPABILITIES_SCHEMA = (
    "hytalerl_native_world_action_capabilities_v1"
)
NATIVE_WORLD_ACTION_CAPABILITIES_VERSION = 1
NATIVE_WORLD_VERB_LIFECYCLE_SCHEMA = "hytalerl_native_world_verb_lifecycle_v1"
NATIVE_WORLD_VERB_LIFECYCLE_VERSION = 1
NATIVE_WORLD_VERBS = ("place_block", "break_block", "craft_recipe")
NATIVE_WORLD_VERB_TRANSPORT_SCHEMA = (
    "hytalerl_native_world_verb_transport_v4"
)
NATIVE_WORLD_VERB_TRANSPORT_VERSION = 4
NATIVE_TYPED_WORLD_VERBS = (
    "use",
    "place_block",
    "break_block",
    "craft_recipe",
)
NATIVE_BLOCK_USE_EVIDENCE_SCHEMA = "hytalerl_native_block_use_evidence_v1"
NATIVE_BLOCK_USE_EVIDENCE_VERSION = 1
NATIVE_SYNTHETIC_WORLD_INTERACTION_SCHEMA = (
    "hytalerl_synthetic_player_world_interaction_v6"
)
NATIVE_SYNTHETIC_WORLD_INTERACTION_VERSION = 6
NATIVE_SYNTHETIC_WORLD_INTERACTION_VERBS = ("place_block", "break_block")
_SYNTHETIC_SERVER_STATES = frozenset(
    {
        "unrequested",
        "requested",
        "rejected",
        "queued",
        "Finished",
        "Skip",
        "ItemChanged",
        "Failed",
        "NotFinished",
    }
)


@dataclass(frozen=True, slots=True)
class NativeWorldActionCapabilities:
    """Bridge-scoped availability without post-request discovery."""

    bridge_sha256: str
    use_request_available: bool
    place_block_request_available: bool
    break_block_request_available: bool
    craft_recipe_request_available: bool
    live_mutation_acknowledgement_available: bool
    mutable_block_evidence_available: bool
    drop_program_evidence_available: bool
    crafting_catalog_evidence_available: bool
    headless_server_actor_acceptance_available: bool
    public_player_acceptance_certified: bool
    unavailable_reason: str

    @classmethod
    def from_info(
        cls,
        info: Mapping[str, object],
        *,
        expected_bridge_sha256: str | None = None,
    ) -> NativeWorldActionCapabilities:
        version = info.get("world_action_capabilities_version")
        if (
            info.get("world_action_capabilities_schema")
            != NATIVE_WORLD_ACTION_CAPABILITIES_SCHEMA
            or isinstance(version, bool)
            or version != NATIVE_WORLD_ACTION_CAPABILITIES_VERSION
        ):
            raise ValueError("native World-action capability schema changed")
        bridge = _sha256(info.get("bridge_sha256"), "bridge_sha256")
        if (
            expected_bridge_sha256 is not None
            and bridge.lower()
            != _sha256(
                expected_bridge_sha256,
                "expected_bridge_sha256",
            ).lower()
        ):
            raise ValueError("World-action capabilities came from another bridge")
        unavailable_reason = info.get("world_action_unavailable_reason")
        if not isinstance(unavailable_reason, str):
            raise ValueError(
                "world_action_unavailable_reason must be a string"
            )
        result = cls(
            bridge_sha256=bridge,
            use_request_available=_boolean(
                info,
                "world_action_use_request_available",
            ),
            place_block_request_available=_boolean(
                info,
                "world_action_place_block_request_available",
            ),
            break_block_request_available=_boolean(
                info,
                "world_action_break_block_request_available",
            ),
            craft_recipe_request_available=_boolean(
                info,
                "world_action_craft_recipe_request_available",
            ),
            live_mutation_acknowledgement_available=_boolean(
                info,
                "world_action_live_mutation_acknowledgement_available",
            ),
            mutable_block_evidence_available=_boolean(
                info,
                "world_action_mutable_block_evidence_available",
            ),
            drop_program_evidence_available=_boolean(
                info,
                "world_action_drop_program_evidence_available",
            ),
            crafting_catalog_evidence_available=_boolean(
                info,
                "world_action_crafting_catalog_evidence_available",
            ),
            headless_server_actor_acceptance_available=(
                _boolean(
                    info,
                    "world_action_headless_server_actor_acceptance_available",
                )
                if (
                    "world_action_headless_server_actor_acceptance_available"
                    in info
                )
                else False
            ),
            public_player_acceptance_certified=_boolean(
                info,
                "world_action_public_player_acceptance_certified",
            ),
            unavailable_reason=unavailable_reason,
        )
        incomplete = (
            not result.all_requests_available
            or not result.live_mutation_acknowledgement_available
            or not result.mutable_block_evidence_available
            or not result.drop_program_evidence_available
            or not result.crafting_catalog_evidence_available
            or not result.public_player_acceptance_certified
        )
        if incomplete and not result.unavailable_reason:
            raise ValueError("unavailable World actions require a reason")
        return result

    @property
    def all_requests_available(self) -> bool:
        return (
            self.use_request_available
            and self.place_block_request_available
            and self.break_block_request_available
            and self.craft_recipe_request_available
        )

    def request_available(self, verb: str) -> bool:
        try:
            return {
                "use": self.use_request_available,
                "place_block": self.place_block_request_available,
                "break_block": self.break_block_request_available,
                "craft_recipe": self.craft_recipe_request_available,
            }[verb]
        except KeyError as error:
            raise ValueError(f"unknown native World verb: {verb}") from error

    @property
    def server_actor_acceptance_available(self) -> bool:
        """Whether either truthful native actor context can execute verbs."""

        return (
            self.headless_server_actor_acceptance_available
            or self.public_player_acceptance_certified
        )


def native_world_verb_reset_options(
    *,
    hotbar_items: Sequence[tuple[int, str, int]] = (),
) -> dict[str, object]:
    """Opt into native verbs with an optional exact, bounded hotbar loadout."""

    options: dict[str, object] = {"native_world_verbs": True}
    if not hotbar_items:
        return options
    if len(hotbar_items) > 16:
        raise ValueError("hotbar_items must contain at most 16 entries")
    slots: list[int] = []
    item_ids: list[str] = []
    quantities: list[int] = []
    seen_slots: set[int] = set()
    for entry in hotbar_items:
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise TypeError(
                "hotbar_items entries must be (slot, item_id, quantity) tuples"
            )
        slot, item_id, quantity = entry
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot <= 31:
            raise ValueError("hotbar slot must be an integer in [0, 31]")
        if slot in seen_slots:
            raise ValueError(f"duplicate hotbar slot: {slot}")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("hotbar item_id must be a nonblank string")
        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, int)
            or quantity <= 0
        ):
            raise ValueError("hotbar quantity must be a positive integer")
        seen_slots.add(slot)
        slots.append(slot)
        item_ids.append(item_id.strip())
        quantities.append(quantity)
    options.update(
        {
            "native_world_hotbar_slots": slots,
            "native_world_hotbar_item_ids": item_ids,
            "native_world_hotbar_quantities": quantities,
        }
    )
    return options


@dataclass(frozen=True, slots=True)
class NativeWorldVerbLifecycle:
    """One bridge-attributed request outcome; absence is explicit."""

    requested: bool
    accepted: bool
    reject_reason: str


@dataclass(frozen=True, slots=True)
class NativeWorldVerbTelemetry:
    """Typed step-info evidence for the three headless World verbs."""

    bridge_sha256: str
    place_block: NativeWorldVerbLifecycle
    break_block: NativeWorldVerbLifecycle
    craft_recipe: NativeWorldVerbLifecycle

    @classmethod
    def from_info(
        cls,
        info: Mapping[str, object],
        *,
        expected_bridge_sha256: str | None = None,
    ) -> NativeWorldVerbTelemetry:
        schema = info.get("native_world_verb_lifecycle_schema")
        version = info.get("native_world_verb_lifecycle_version")
        if (
            schema != NATIVE_WORLD_VERB_LIFECYCLE_SCHEMA
            or isinstance(version, bool)
            or version != NATIVE_WORLD_VERB_LIFECYCLE_VERSION
        ):
            raise ValueError("native World-verb lifecycle schema changed")
        contract_sha256 = _sha256(
            info.get("native_world_verb_lifecycle_contract_sha256"),
            "native_world_verb_lifecycle_contract_sha256",
        )
        if contract_sha256.lower() != (
            native_world_verb_lifecycle_contract_sha256().lower()
        ):
            raise ValueError("native World-verb lifecycle contract changed")
        bridge = _sha256(info.get("bridge_sha256"), "bridge_sha256")
        if (
            expected_bridge_sha256 is not None
            and bridge.lower()
            != _sha256(
                expected_bridge_sha256,
                "expected_bridge_sha256",
            ).lower()
        ):
            raise ValueError("World-verb telemetry came from another bridge")
        rows = {
            verb: _verb_lifecycle(info, verb)
            for verb in NATIVE_WORLD_VERBS
        }
        return cls(bridge_sha256=bridge, **rows)

    def for_verb(self, verb: str) -> NativeWorldVerbLifecycle:
        if verb not in NATIVE_WORLD_VERBS:
            raise ValueError(f"unknown native World verb: {verb}")
        return getattr(self, verb)


@dataclass(frozen=True, slots=True)
class NativeWorldVerbRequest:
    """One stable-ID, reset-scoped World-verb request."""

    request_id: int
    verb: str
    interaction_type: int
    target_kind: str
    interaction_id: str
    recipe_id: str
    item_id: str
    block_id: str
    target: tuple[int, int, int]
    block_face: int
    rotation: tuple[int, int, int]
    source_container: str
    source_slot: int
    expected_source_quantity: int
    destination_container: str
    expected_block_id: str
    world_epoch: str
    quantity: int
    placement_variant: str
    crafting_context: str = ""
    bench: tuple[int, int, int] = (0, 0, 0)
    expected_bench_block_id: str = ""
    expected_bench_id: str = ""
    expected_bench_type: int = -1
    expected_bench_tier: int = 0
    expected_candidate_generation_sha256: str = ""
    expected_selected_semantic_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_native_world_verb_request(self)

    @property
    def policy_candidate_evidence_bound(self) -> bool:
        """Whether this request carries the mandatory paired policy proof."""

        return bool(
            self.expected_candidate_generation_sha256
            and self.expected_selected_semantic_sha256
        )

    def to_action_fields(self) -> dict[str, object]:
        """Return the nested MessagePack-ready action field."""

        action: dict[str, object] = {
            "native_world_verb_request": {
                "schema": NATIVE_WORLD_VERB_TRANSPORT_SCHEMA,
                "version": NATIVE_WORLD_VERB_TRANSPORT_VERSION,
                **_native_world_verb_request_payload(self),
            }
        }
        if (
            self.source_slot >= 0
            and self.source_container in {"hotbar", "interaction_context"}
        ):
            # AgentAction applies the active hotbar selection before it
            # revalidates a typed verb. Preserve the exact source chosen by
            # the actor-legal evidence row instead of falling back to slot 0.
            action["hotbar_slot"] = self.source_slot
        return action


@dataclass(frozen=True, slots=True)
class NativeWorldVerbTransportSession:
    """Bridge identity and opaque reset epoch required by typed requests."""

    bridge_sha256: str
    world_epoch: str

    @classmethod
    def from_info(
        cls,
        info: Mapping[str, object],
        *,
        expected_bridge_sha256: str | None = None,
    ) -> NativeWorldVerbTransportSession:
        version = info.get("native_world_verb_transport_version")
        if (
            info.get("native_world_verb_transport_schema")
            != NATIVE_WORLD_VERB_TRANSPORT_SCHEMA
            or isinstance(version, bool)
            or version != NATIVE_WORLD_VERB_TRANSPORT_VERSION
        ):
            raise ValueError("native World-verb transport schema changed")
        contract_sha256 = _sha256(
            info.get("native_world_verb_transport_contract_sha256"),
            "native_world_verb_transport_contract_sha256",
        )
        if contract_sha256.lower() != (
            native_world_verb_transport_contract_sha256().lower()
        ):
            raise ValueError("native World-verb transport contract changed")
        bridge = _sha256(info.get("bridge_sha256"), "bridge_sha256")
        if (
            expected_bridge_sha256 is not None
            and bridge.lower()
            != _sha256(
                expected_bridge_sha256,
                "expected_bridge_sha256",
            ).lower()
        ):
            raise ValueError("World-verb transport came from another bridge")
        epoch = _bounded_canonical_string(
            info,
            "native_world_verb_epoch",
            maximum=128,
            allow_empty=False,
        )
        return cls(bridge_sha256=bridge, world_epoch=epoch)


@dataclass(frozen=True, slots=True)
class NativeWorldVerbReceipt:
    """Fail-closed typed request echo plus reserved mutation acknowledgement."""

    bridge_sha256: str
    current_world_epoch: str
    request: NativeWorldVerbRequest | None
    accepted: bool
    started: bool
    active: bool
    finished: bool
    failed: bool
    reject_reason: str
    execution_scope: str
    public_player_certified: bool
    headless_server_actor_certified: bool
    opens_action_mask: bool
    acknowledgement_complete: bool
    mutation_applied: bool
    inventory_applied: bool
    semantic_block_id_before: str
    semantic_block_id_after: str
    runtime_block_id_before: int
    runtime_block_id_after: int
    source_quantity_before: int
    source_quantity_after: int
    destination_quantity_before: int
    destination_quantity_after: int
    geometry_delta_sha256: str
    inventory_delta_sha256: str
    drop_outcome_sha256: str
    requested_world_tick: int
    started_world_tick: int
    finished_world_tick: int

    @classmethod
    def from_info(
        cls,
        info: Mapping[str, object],
        *,
        expected_bridge_sha256: str | None = None,
    ) -> NativeWorldVerbReceipt:
        session = NativeWorldVerbTransportSession.from_info(
            info,
            expected_bridge_sha256=expected_bridge_sha256,
        )
        prefix = "native_world_verb_request_"
        present = _boolean(info, prefix + "present")
        payload = _native_world_verb_request_payload_from_info(info, prefix)
        request = (
            _native_world_verb_request_from_payload(payload)
            if present
            else None
        )
        accepted = _boolean(info, prefix + "accepted")
        started = _boolean(info, prefix + "started")
        active = _boolean(info, prefix + "active")
        finished = _boolean(info, prefix + "finished")
        failed = _boolean(info, prefix + "failed")
        reject_reason = _bounded_canonical_string(
            info,
            prefix + "reject_reason",
            maximum=256,
        )
        execution_scope = _bounded_canonical_string(
            info,
            prefix + "execution_scope",
            maximum=64,
            allow_empty=False,
        )
        public_player_certified = _boolean(
            info,
            prefix + "public_player_certified",
        )
        headless_server_actor_certified = _boolean(
            info,
            prefix + "headless_server_actor_certified",
        )
        opens_action_mask = _boolean(info, prefix + "opens_action_mask")
        acknowledgement_complete = _boolean(
            info,
            prefix + "acknowledgement_complete",
        )
        mutation_applied = _boolean(info, prefix + "mutation_applied")
        inventory_applied = _boolean(info, prefix + "inventory_applied")
        semantic_before = _bounded_canonical_string(
            info,
            prefix + "semantic_block_id_before",
            maximum=256,
        )
        semantic_after = _bounded_canonical_string(
            info,
            prefix + "semantic_block_id_after",
            maximum=256,
        )
        runtime_before = _integer(info, prefix + "runtime_block_id_before")
        runtime_after = _integer(info, prefix + "runtime_block_id_after")
        source_before = _integer(info, prefix + "source_quantity_before")
        source_after = _integer(info, prefix + "source_quantity_after")
        destination_before = _integer(
            info,
            prefix + "destination_quantity_before",
        )
        destination_after = _integer(
            info,
            prefix + "destination_quantity_after",
        )
        geometry_sha = _optional_sha256(
            info,
            prefix + "geometry_delta_sha256",
        )
        inventory_sha = _optional_sha256(
            info,
            prefix + "inventory_delta_sha256",
        )
        drop_sha = _optional_sha256(
            info,
            prefix + "drop_outcome_sha256",
        )
        requested_tick = _integer(
            info,
            prefix + "requested_world_tick",
            minimum=-1,
        )
        started_tick = _integer(
            info,
            prefix + "started_world_tick",
            minimum=-1,
        )
        finished_tick = _integer(
            info,
            prefix + "finished_world_tick",
            minimum=-1,
        )
        result = cls(
            bridge_sha256=session.bridge_sha256,
            current_world_epoch=session.world_epoch,
            request=request,
            accepted=accepted,
            started=started,
            active=active,
            finished=finished,
            failed=failed,
            reject_reason=reject_reason,
            execution_scope=execution_scope,
            public_player_certified=public_player_certified,
            headless_server_actor_certified=(
                headless_server_actor_certified
            ),
            opens_action_mask=opens_action_mask,
            acknowledgement_complete=acknowledgement_complete,
            mutation_applied=mutation_applied,
            inventory_applied=inventory_applied,
            semantic_block_id_before=semantic_before,
            semantic_block_id_after=semantic_after,
            runtime_block_id_before=runtime_before,
            runtime_block_id_after=runtime_after,
            source_quantity_before=source_before,
            source_quantity_after=source_after,
            destination_quantity_before=destination_before,
            destination_quantity_after=destination_after,
            geometry_delta_sha256=geometry_sha,
            inventory_delta_sha256=inventory_sha,
            drop_outcome_sha256=drop_sha,
            requested_world_tick=requested_tick,
            started_world_tick=started_tick,
            finished_world_tick=finished_tick,
        )
        _validate_native_world_verb_receipt(result, payload)
        return result


@dataclass(frozen=True, slots=True)
class NativeBlockUseEvidence:
    """Bridge-attributed headless block-Use availability and lifecycle."""

    bridge_sha256: str
    available: bool
    available_interaction_id: str
    available_block_interaction_id: str
    available_item_id: str
    available_source_container: str
    available_source_slot: int
    available_source_quantity: int
    available_target: tuple[int, int, int] | None
    available_maximum_distance: float
    unavailable_reason: str
    requested: bool
    accepted: bool
    started: bool
    active: bool
    finished: bool
    failed: bool
    reject_reason: str
    interaction_id: str
    block_interaction_id: str
    target: tuple[int, int, int] | None
    maximum_distance: float
    chain_start_world_tick: int
    chain_finish_world_tick: int

    @classmethod
    def from_info(
        cls,
        info: Mapping[str, object],
        *,
        expected_bridge_sha256: str | None = None,
    ) -> NativeBlockUseEvidence:
        bridge = _sha256(info.get("bridge_sha256"), "bridge_sha256")
        if (
            expected_bridge_sha256 is not None
            and bridge.lower()
            != _sha256(
                expected_bridge_sha256,
                "expected_bridge_sha256",
            ).lower()
        ):
            raise ValueError("native block-Use evidence came from another bridge")
        if info.get("native_use_scope") != (
            "authored_block_target_use;entity_target_use_unavailable"
        ) or _boolean(info, "native_use_entity_target_supported"):
            raise ValueError("native block-Use scope changed")

        available = _boolean(info, "native_use_available")
        available_interaction_id = _nonempty_string_or_empty(
            info,
            "native_use_available_interaction_id",
        )
        available_block_interaction_id = _nonempty_string_or_empty(
            info,
            "native_use_available_block_interaction_id",
        )
        available_item_id = _nonempty_string_or_empty(
            info,
            "native_use_available_item_id",
        )
        available_source_container = _nonempty_string_or_empty(
            info,
            "native_use_available_source_container",
        )
        available_source_slot = _integer(
            info,
            "native_use_available_source_slot",
        )
        available_source_quantity = _integer(
            info,
            "native_use_available_source_quantity",
        )
        available_target = _optional_target(
            info,
            "native_use_available_target",
        )
        available_maximum_distance = _real(
            info,
            "native_use_available_maximum_distance",
            minimum=0.0,
        )
        unavailable_reason = _nonempty_string_or_empty(
            info,
            "native_use_unavailable_reason",
        )
        if available != (available_target is not None):
            raise ValueError("native block-Use availability target is inconsistent")
        if available and (
            not available_interaction_id
            or not available_block_interaction_id
            or not available_item_id
            or available_source_container
            not in {"unarmed", "interaction_context"}
            or available_maximum_distance <= 0.0
            or unavailable_reason
        ):
            raise ValueError("available native block-Use row is incomplete")
        if available and (
            (
                available_source_container == "unarmed"
                and (
                    available_item_id != "Empty"
                    or available_source_slot != -1
                    or available_source_quantity != 0
                )
            )
            or (
                available_source_container == "interaction_context"
                and (
                    available_item_id == "Empty"
                    or available_source_slot < 0
                    or available_source_quantity <= 0
                )
            )
        ):
            raise ValueError("native block-Use source binding is inconsistent")
        if not available and (
            available_interaction_id
            or available_block_interaction_id
            or available_item_id
            or available_source_container
            or available_source_slot != -1
            or available_source_quantity != -1
            or available_maximum_distance != 0.0
            or not unavailable_reason
        ):
            raise ValueError("unavailable native block-Use row is inconsistent")

        requested = _boolean(info, "native_use_requested")
        accepted = _boolean(info, "native_use_accepted")
        started = _boolean(info, "native_use_started")
        active = _boolean(info, "native_use_active")
        finished = _boolean(info, "native_use_finished")
        failed = _boolean(info, "native_use_failed")
        reject_reason = _nonempty_string_or_empty(
            info,
            "native_use_reject_reason",
        )
        interaction_id = _nonempty_string_or_empty(
            info,
            "native_use_interaction_id",
        )
        block_interaction_id = _nonempty_string_or_empty(
            info,
            "native_use_block_interaction_id",
        )
        target = _optional_target(
            info,
            "native_use_target_available",
            "native_use_target",
        )
        maximum_distance = _real(
            info,
            "native_use_maximum_distance",
            minimum=0.0,
        )
        start_tick = _integer(info, "native_use_chain_start_world_tick")
        finish_tick = _integer(info, "native_use_chain_finish_world_tick")
        if accepted and (not requested or reject_reason):
            raise ValueError("accepted native block-Use lifecycle is inconsistent")
        if requested and not accepted and not reject_reason:
            raise ValueError("rejected native block-Use request requires a reason")
        if not requested and (accepted or reject_reason):
            raise ValueError("unrequested native block-Use lifecycle is inconsistent")
        if active and (target is None or start_tick < 0 or finish_tick >= 0):
            raise ValueError("active native block-Use lifecycle is inconsistent")
        if finished and (start_tick < 0 or finish_tick < start_tick):
            raise ValueError("finished native block-Use lifecycle is inconsistent")
        if failed and not finished:
            raise ValueError("failed native block-Use lifecycle must be finished")
        has_bound_chain = target is not None
        if has_bound_chain != bool(interaction_id and block_interaction_id):
            raise ValueError("native block-Use chain binding is inconsistent")
        if has_bound_chain != (maximum_distance > 0.0):
            raise ValueError("native block-Use target distance is inconsistent")

        return cls(
            bridge_sha256=bridge,
            available=available,
            available_interaction_id=available_interaction_id,
            available_block_interaction_id=available_block_interaction_id,
            available_item_id=available_item_id,
            available_source_container=available_source_container,
            available_source_slot=available_source_slot,
            available_source_quantity=available_source_quantity,
            available_target=available_target,
            available_maximum_distance=available_maximum_distance,
            unavailable_reason=unavailable_reason,
            requested=requested,
            accepted=accepted,
            started=started,
            active=active,
            finished=finished,
            failed=failed,
            reject_reason=reject_reason,
            interaction_id=interaction_id,
            block_interaction_id=block_interaction_id,
            target=target,
            maximum_distance=maximum_distance,
            chain_start_world_tick=start_tick,
            chain_finish_world_tick=finish_tick,
        )


@dataclass(frozen=True, slots=True)
class NativeSyntheticWorldInteractionRow:
    """One fixture-only native interaction-chain outcome."""

    requested: bool
    accepted: bool
    started: bool
    finished: bool
    failed: bool
    mutation_applied: bool
    reject_reason: str
    interaction_asset_id: str
    held_item_asset_id: str
    server_state: str
    target: tuple[int, int, int]
    runtime_block_id_before: int
    runtime_block_id_after: int
    block_health_before: float
    block_health_after: float
    requested_world_tick: int
    started_world_tick: int
    finished_world_tick: int


@dataclass(frozen=True, slots=True)
class NativeSyntheticWorldInteractionEvidence:
    """Synthetic-Player certificate without public-action promotion."""

    bridge_sha256: str
    fixture_available: bool
    fixture_block_asset_id: str
    fixture_block_runtime_id: int
    fixture_place_item_asset_id: str
    fixture_break_tool_item_asset_id: str
    fixture_place_source_quantity: int
    fixture_break_source_quantity: int
    fixture_place_target: tuple[int, int, int]
    fixture_break_target: tuple[int, int, int]
    public_player_certified: bool
    opens_action_mask: bool
    place_block: NativeSyntheticWorldInteractionRow
    break_block: NativeSyntheticWorldInteractionRow

    @classmethod
    def from_info(
        cls,
        info: Mapping[str, object],
        *,
        expected_bridge_sha256: str | None = None,
    ) -> NativeSyntheticWorldInteractionEvidence:
        version = info.get("synthetic_world_interaction_version")
        if (
            info.get("synthetic_world_interaction_schema")
            != NATIVE_SYNTHETIC_WORLD_INTERACTION_SCHEMA
            or isinstance(version, bool)
            or version != NATIVE_SYNTHETIC_WORLD_INTERACTION_VERSION
        ):
            raise ValueError(
                "native synthetic World-interaction schema changed"
            )
        contract_sha256 = _sha256(
            info.get("synthetic_world_interaction_contract_sha256"),
            "synthetic_world_interaction_contract_sha256",
        )
        if contract_sha256.lower() != (
            native_synthetic_world_interaction_contract_sha256().lower()
        ):
            raise ValueError(
                "native synthetic World-interaction contract changed"
            )
        bridge = _sha256(info.get("bridge_sha256"), "bridge_sha256")
        if (
            expected_bridge_sha256 is not None
            and bridge.lower()
            != _sha256(
                expected_bridge_sha256,
                "expected_bridge_sha256",
            ).lower()
        ):
            raise ValueError(
                "synthetic World evidence came from another bridge"
            )
        fixture_available = _boolean(
            info,
            "synthetic_world_interaction_fixture_available",
        )
        fixture_block_asset_id = _nonempty_string_or_empty(
            info,
            "synthetic_world_interaction_fixture_block_asset_id",
        )
        fixture_block_runtime_id = _integer(
            info,
            "synthetic_world_interaction_fixture_block_runtime_id",
            minimum=0,
        )
        fixture_place_item_asset_id = _nonempty_string_or_empty(
            info,
            "synthetic_world_interaction_fixture_place_item_asset_id",
        )
        fixture_break_tool_item_asset_id = _nonempty_string_or_empty(
            info,
            "synthetic_world_interaction_fixture_break_tool_item_asset_id",
        )
        fixture_place_source_quantity = _integer(
            info,
            "synthetic_world_interaction_fixture_place_source_quantity",
            minimum=0,
        )
        fixture_break_source_quantity = _integer(
            info,
            "synthetic_world_interaction_fixture_break_source_quantity",
            minimum=0,
        )
        fixture_place_target = _synthetic_fixture_target(info, "place")
        fixture_break_target = _synthetic_fixture_target(info, "break")
        assets_bound = bool(
            fixture_block_asset_id
            and fixture_block_runtime_id > 0
            and fixture_place_item_asset_id
            and fixture_break_tool_item_asset_id
            and fixture_place_source_quantity > 0
            and fixture_break_source_quantity > 0
        )
        targets_bound = fixture_place_target != fixture_break_target
        unavailable_targets = (
            fixture_place_target == (0, 0, 0)
            and fixture_break_target == (0, 0, 0)
        )
        if (
            fixture_available
            and (not assets_bound or not targets_bound)
        ) or (
            not fixture_available
            and (
                assets_bound
                or fixture_block_runtime_id != 0
                or fixture_place_source_quantity != 0
                or fixture_break_source_quantity != 0
                or not unavailable_targets
            )
        ):
            raise ValueError(
                "synthetic fixture availability and bound assets disagree"
            )
        public_player_certified = _boolean(
            info,
            "synthetic_world_interaction_public_player_certified",
        )
        opens_action_mask = _boolean(
            info,
            "synthetic_world_interaction_opens_action_mask",
        )
        if public_player_certified or opens_action_mask:
            raise ValueError(
                "synthetic evidence cannot certify a public Player action"
            )
        rows = {
            verb: _synthetic_interaction_row(info, verb)
            for verb in NATIVE_SYNTHETIC_WORLD_INTERACTION_VERBS
        }
        if any(row.accepted for row in rows.values()) and not fixture_available:
            raise ValueError(
                "synthetic interaction was accepted outside its fixture"
            )
        for verb, row in rows.items():
            if not row.accepted:
                continue
            fixture_target = (
                fixture_place_target
                if verb == "place_block"
                else fixture_break_target
            )
            if row.target != fixture_target:
                raise ValueError(
                    f"{verb} target differs from the fixture binding"
                )
            expected_item = (
                fixture_place_item_asset_id
                if verb == "place_block"
                else fixture_break_tool_item_asset_id
            )
            if row.held_item_asset_id != expected_item:
                raise ValueError(
                    f"{verb} held item differs from the fixture binding"
                )
            bound_runtime_id = (
                row.runtime_block_id_after
                if verb == "place_block"
                else row.runtime_block_id_before
            )
            if bound_runtime_id != fixture_block_runtime_id:
                raise ValueError(
                    f"{verb} block differs from the fixture binding"
                )
        return cls(
            bridge_sha256=bridge,
            fixture_available=fixture_available,
            fixture_block_asset_id=fixture_block_asset_id,
            fixture_block_runtime_id=fixture_block_runtime_id,
            fixture_place_item_asset_id=fixture_place_item_asset_id,
            fixture_break_tool_item_asset_id=fixture_break_tool_item_asset_id,
            fixture_place_source_quantity=fixture_place_source_quantity,
            fixture_break_source_quantity=fixture_break_source_quantity,
            fixture_place_target=fixture_place_target,
            fixture_break_target=fixture_break_target,
            public_player_certified=public_player_certified,
            opens_action_mask=opens_action_mask,
            **rows,
        )

    def for_verb(
        self,
        verb: str,
    ) -> NativeSyntheticWorldInteractionRow:
        if verb not in NATIVE_SYNTHETIC_WORLD_INTERACTION_VERBS:
            raise ValueError(
                f"unknown synthetic World interaction: {verb}"
            )
        return getattr(self, verb)


def native_world_action_capabilities_contract() -> dict[str, object]:
    """Return the stable reset-info field contract."""

    return {
        "schema": NATIVE_WORLD_ACTION_CAPABILITIES_SCHEMA,
        "version": NATIVE_WORLD_ACTION_CAPABILITIES_VERSION,
        "verbs": ["use", "place_block", "break_block", "craft_recipe"],
        "publication": "native_reset_info",
        "identity": "runtime_bridge_sha256",
        "evidence_transports": [
            "mutable_block",
            "drop_program",
            "crafting_catalog",
        ],
        "fail_closed": [
            "missing_or_unknown_schema",
            "bridge_identity_mismatch",
            "unavailable_request",
            "missing_live_mutation_acknowledgement",
            "headless_server_actor_not_enabled",
            "uncertified_public_player_acceptance",
        ],
        "actor_contexts": {
            "headless_server_actor": (
                "reset_option_native_world_verbs_true;opt_in_native_"
                "mechanics_not_authenticated_client"
            ),
            "authenticated_public_player": (
                "separate_evidence_bit_remains_false"
            ),
        },
        "capability_is_not_certification": True,
    }


def native_world_action_capabilities_contract_sha256() -> str:
    encoded = json.dumps(
        native_world_action_capabilities_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def native_world_verb_lifecycle_contract() -> dict[str, object]:
    """Return the additive step-info rejection-evidence contract."""

    return {
        "schema": NATIVE_WORLD_VERB_LIFECYCLE_SCHEMA,
        "version": NATIVE_WORLD_VERB_LIFECYCLE_VERSION,
        "verbs": list(NATIVE_WORLD_VERBS),
        "publication": "native_step_info",
        "identity": "runtime_bridge_sha256",
        "fields": ["requested", "accepted", "reject_reason"],
        "invariants": {
            "unrequested": "accepted_false_and_empty_reason",
            "accepted": "requested_true_and_empty_reason",
            "rejected": "requested_true_and_nonempty_reason",
        },
        "current_scope": (
            "headless_rejection_telemetry_not_public_player_acceptance"
        ),
        "opens_action_mask": False,
    }


def native_world_verb_lifecycle_contract_sha256() -> str:
    encoded = json.dumps(
        native_world_verb_lifecycle_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def native_world_verb_transport_contract() -> dict[str, object]:
    """Return the stable request/receipt transport contract."""

    request_fields = [
        "request_id",
        "verb",
        "interaction_type",
        "target_kind",
        "interaction_id",
        "recipe_id",
        "item_id",
        "block_id",
        "target_x",
        "target_y",
        "target_z",
        "block_face",
        "rotation_yaw",
        "rotation_pitch",
        "rotation_roll",
        "source_container",
        "source_slot",
        "expected_source_quantity",
        "destination_container",
        "expected_block_id",
        "world_epoch",
        "quantity",
        "placement_variant",
        "crafting_context",
        "bench_x",
        "bench_y",
        "bench_z",
        "expected_bench_block_id",
        "expected_bench_id",
        "expected_bench_type",
        "expected_bench_tier",
        "expected_candidate_generation_sha256",
        "expected_selected_semantic_sha256",
    ]
    return {
        "schema": NATIVE_WORLD_VERB_TRANSPORT_SCHEMA,
        "version": NATIVE_WORLD_VERB_TRANSPORT_VERSION,
        "verbs": list(NATIVE_TYPED_WORLD_VERBS),
        "reset_options": {
            "native_world_verbs": True,
            "optional_hotbar_loadout": {
                "fields": [
                    "native_world_hotbar_slots",
                    "native_world_hotbar_item_ids",
                    "native_world_hotbar_quantities",
                ],
                "shape": "1..16_aligned_entries",
                "slot_domain": "distinct_integers_0..31_and_runtime_capacity",
                "quantity_domain": "positive_int32",
                "purpose": (
                    "exact_tool_place_item_and_crafting_material_inputs_"
                    "without_overwriting_unspecified_slots"
                ),
            },
        },
        "request": {
            "publication": "action.native_world_verb_request",
            "fields": request_fields,
            "identity": "stable_string_asset_ids",
            "target": "absolute_block_coordinates",
            "epoch": "opaque_reset_scoped",
            "crafting_context": (
                "exact_fieldcraft_or_loaded_bench_block_id_asset_id_"
                "type_and_tier"
            ),
            "policy_candidate_expectation": (
                "paired_atomic_generation_and_selected_semantic_sha256_required"
            ),
        },
        "receipt": {
            "publication": "native_step_info",
            "echo_fields": request_fields,
            "lifecycle_fields": [
                "accepted",
                "started",
                "active",
                "finished",
                "failed",
                "reject_reason",
                "execution_scope",
                "public_player_certified",
                "headless_server_actor_certified",
                "opens_action_mask",
            ],
            "acknowledgement_fields": [
                "acknowledgement_complete",
                "mutation_applied",
                "inventory_applied",
                "semantic_block_id_before",
                "semantic_block_id_after",
                "runtime_block_id_before",
                "runtime_block_id_after",
                "source_quantity_before",
                "source_quantity_after",
                "destination_quantity_before",
                "destination_quantity_after",
                "geometry_delta_sha256",
                "inventory_delta_sha256",
                "drop_outcome_sha256",
                "requested_world_tick",
                "started_world_tick",
                "finished_world_tick",
            ],
        },
        "protocol_enums": {
            "interaction_type": {
                "Primary": 0,
                "Secondary": 1,
                "Use": 5,
                "craft_recipe_sentinel": -1,
            },
            "block_face": "Hytale BlockFace 0..6",
            "rotation": "Hytale Rotation 0..3 for yaw/pitch/roll",
        },
        "source_sentinels": {
            "unarmed_use": {
                "source_container": "unarmed",
                "source_slot": -1,
                "expected_source_quantity": 0,
                "item_id": "Empty",
            },
            "resolved_interaction_context_use": {
                "source_container": "interaction_context",
                "source_slot": "native_InteractionContext.heldItemSlot",
                "expected_source_quantity": (
                    "native_InteractionContext.heldItem.quantity"
                ),
                "item_id": "native_InteractionContext.originalItemType.id",
            },
        },
        "fail_closed": [
            "unknown_schema_or_verb",
            "incomplete_or_mixed_legacy_request",
            "stale_world_epoch",
            "missing_stable_identity",
            "out_of_domain_target",
            "inconsistent_lifecycle_or_acknowledgement",
            "missing_candidate_generation_or_selected_semantic",
        ],
        "current_execution_scope": {
            "use": (
                "headless_native_target_relative_use_chain_with_exact_"
                "geometry_and_inventory_acknowledgement"
            ),
            "place_block": (
                "opt_in_headless_adventure_player_authored_item_root_chain_"
                "with_exact_geometry_block_health_and_inventory_"
                "acknowledgement"
            ),
            "break_block": (
                "opt_in_headless_adventure_player_authored_item_root_chain_"
                "with_exact_geometry_block_health_and_inventory_"
                "acknowledgement"
            ),
            "craft_recipe": (
                "opt_in_headless_adventure_player_native_crafting_manager_"
                "fieldcraft_and_material_inventory_crafting_bench"
            ),
        },
        "fixture_root_bindings": {
            "place_block": {
                "item_id": "Rock_Stone",
                "root_interaction_id": "Block_Secondary",
                "interaction_type": "Secondary",
            },
            "break_block": {
                "item_id": "Tool_Pickaxe_Iron",
                "root_interaction_id": "Pickaxe_Attack",
                "interaction_type": "Primary",
            },
        },
        "fixture_client_semantics": {
            "target_binding": (
                "server_packet_equivalent_TARGET_BLOCK_and_TARGET_BLOCK_RAW_"
                "before_first_tick"
            ),
            "chaining": (
                "fresh_click_selects_first_authored_chain_entry"
            ),
        },
        "authenticated_public_player_certified": False,
        "headless_server_actor_supported": True,
        "specialized_bench_input_state": (
            "diagram_structural_and_processing_fail_closed"
        ),
        "opens_action_mask": (
            "only_after_complete_acknowledgement_when_reset_"
            "native_world_verbs_is_true"
        ),
    }


def native_world_verb_transport_contract_sha256() -> str:
    encoded = json.dumps(
        native_world_verb_transport_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def native_block_use_evidence_contract() -> dict[str, object]:
    """Return the additive native block-Use evidence contract."""

    return {
        "schema": NATIVE_BLOCK_USE_EVIDENCE_SCHEMA,
        "version": NATIVE_BLOCK_USE_EVIDENCE_VERSION,
        "publication": "native_observation_info",
        "identity": "runtime_bridge_sha256",
        "scope": "authored_block_target_use",
        "target_selection": "post_action_camera_block_raycast",
        "fields": [
            "actor_legal_availability",
            "selected_item_and_block_interaction_ids",
            "resolved_source_item_container_slot_and_quantity",
            "exact_block_target",
            "authored_maximum_distance",
            "requested_accepted_started_active_finished_failed",
            "reject_reason",
            "chain_start_and_finish_world_ticks",
        ],
        "does_not_certify": [
            "entity_target_use",
            "authenticated_player_packets",
            "public_player_acceptance",
            "general_block_effect_equivalence",
        ],
        "opens_public_action_mask": False,
    }


def native_block_use_evidence_contract_sha256() -> str:
    encoded = json.dumps(
        native_block_use_evidence_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def native_synthetic_world_interaction_contract() -> dict[str, object]:
    """Return the fixture-only native interaction certificate contract."""

    return {
        "schema": NATIVE_SYNTHETIC_WORLD_INTERACTION_SCHEMA,
        "version": NATIVE_SYNTHETIC_WORLD_INTERACTION_VERSION,
        "verbs": list(NATIVE_SYNTHETIC_WORLD_INTERACTION_VERBS),
        "publication": "native_step_info",
        "identity": "runtime_bridge_sha256",
        "actor": (
            "controlled_NPC_with_fixture_only_Adventure_Player_component_"
            "and_shared_native_inventory"
        ),
        "client_sync": (
            "operation_scoped_InteractionSyncData_with_fresh_click_"
            "chaining_index_zero_only_when_WaitForDataFrom_Client"
        ),
        "target_binding": (
            "server_packet_equivalent_TARGET_BLOCK_and_TARGET_BLOCK_RAW_"
            "before_first_tick"
        ),
        "fixture_root_bindings": {
            "place_block": {
                "item_id": "Rock_Stone",
                "root_interaction_id": "Block_Secondary",
                "interaction_type": "Secondary",
            },
            "break_block": {
                "item_id": "Tool_Pickaxe_Iron",
                "root_interaction_id": "Pickaxe_Attack",
                "interaction_type": "Primary",
            },
        },
        "fields": [
            "fixture_block_asset_and_runtime_id",
            "fixture_place_item_asset_id",
            "fixture_break_tool_item_asset_id",
            "fixture_place_and_break_source_quantities",
            "settled_actor_bound_place_and_break_targets",
            "requested",
            "accepted",
            "started",
            "finished",
            "failed",
            "mutation_applied",
            "reject_reason",
            "interaction_asset_id",
            "held_item_asset_id",
            "server_state",
            "target_xyz",
            "runtime_block_id_before_after",
            "block_health_before_after",
            "requested_started_finished_world_tick",
        ],
        "certifies": [
            "installed_interaction_asset_resolution",
            "authored_item_root_interaction_chain",
            "fresh_click_first_authored_chain_branch",
            "Adventure_Player_permission_and_reach_validation",
            "shared_native_held_item",
            "client_wait_and_runtime_lifecycle",
            "exact_block_or_block_health_mutation_outcome",
        ],
        "does_not_certify": [
            "authenticated_network_Player",
            "public_bridge_action_acceptance",
            "camera_raycast_target_selection",
            "general_tool_or_block_catalog",
            "crafting",
        ],
        "jax_counterpart": {
            "legality": "resolve_block_action_targets",
            "state": "MutableBlockState",
            "acknowledgement": "exact_block_mutation_acknowledgement",
            "comparison": "native_block_interaction_outcome_matches",
        },
        "invariants": {
            "unrequested": "all_lifecycle_flags_false_and_ticks_minus_one",
            "rejected": (
                "requested_and_failed_without_acceptance_or_mutation"
            ),
            "accepted": "requested_with_bound_asset_item_target_and_tick",
            "target": "exactly_matches_the_reset_published_fixture_target",
            "mutation": (
                "place_changes_air_to_bound_block_or_break_decreases_"
                "health_or_removes_block"
            ),
            "tick_order": "requested_less_equal_started_less_equal_finished",
        },
        "public_player_certified": False,
        "opens_action_mask": False,
    }


def native_synthetic_world_interaction_contract_sha256() -> str:
    encoded = json.dumps(
        native_synthetic_world_interaction_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verb_lifecycle(
    info: Mapping[str, object],
    verb: str,
) -> NativeWorldVerbLifecycle:
    prefix = f"native_{verb}"
    requested = _boolean(info, f"{prefix}_requested")
    accepted = _boolean(info, f"{prefix}_accepted")
    reason = info.get(f"{prefix}_reject_reason")
    if not isinstance(reason, str):
        raise ValueError(f"{prefix}_reject_reason must be a string")
    if accepted and (not requested or reason):
        raise ValueError(f"{verb} accepted lifecycle is inconsistent")
    if requested and not accepted and not reason:
        raise ValueError(f"{verb} rejection requires a reason")
    if not requested and (accepted or reason):
        raise ValueError(f"{verb} unrequested lifecycle is inconsistent")
    return NativeWorldVerbLifecycle(requested, accepted, reason)


def _synthetic_interaction_row(
    info: Mapping[str, object],
    verb: str,
) -> NativeSyntheticWorldInteractionRow:
    prefix = f"synthetic_{verb}_"
    requested = _boolean(info, prefix + "requested")
    accepted = _boolean(info, prefix + "accepted")
    started = _boolean(info, prefix + "started")
    finished = _boolean(info, prefix + "finished")
    failed = _boolean(info, prefix + "failed")
    mutation_applied = _boolean(info, prefix + "mutation_applied")
    strings = {}
    for name in (
        "reject_reason",
        "interaction_asset_id",
        "held_item_asset_id",
        "server_state",
    ):
        value = info.get(prefix + name)
        if not isinstance(value, str):
            raise ValueError(f"{prefix}{name} must be a string")
        strings[name] = value
    if strings["server_state"] not in _SYNTHETIC_SERVER_STATES:
        raise ValueError(f"{verb} has an unknown native server state")
    target = tuple(
        _integer(info, prefix + f"target_{axis}")
        for axis in ("x", "y", "z")
    )
    before_block = _integer(
        info,
        prefix + "runtime_block_id_before",
        minimum=0,
    )
    after_block = _integer(
        info,
        prefix + "runtime_block_id_after",
        minimum=0,
    )
    before_health = _real(
        info,
        prefix + "block_health_before",
        minimum=0.0,
    )
    after_health = _real(
        info,
        prefix + "block_health_after",
        minimum=0.0,
    )
    requested_tick = _integer(
        info,
        prefix + "requested_world_tick",
        minimum=-1,
    )
    started_tick = _integer(
        info,
        prefix + "started_world_tick",
        minimum=-1,
    )
    finished_tick = _integer(
        info,
        prefix + "finished_world_tick",
        minimum=-1,
    )
    row = NativeSyntheticWorldInteractionRow(
        requested=requested,
        accepted=accepted,
        started=started,
        finished=finished,
        failed=failed,
        mutation_applied=mutation_applied,
        reject_reason=strings["reject_reason"],
        interaction_asset_id=strings["interaction_asset_id"],
        held_item_asset_id=strings["held_item_asset_id"],
        server_state=strings["server_state"],
        target=target,
        runtime_block_id_before=before_block,
        runtime_block_id_after=after_block,
        block_health_before=before_health,
        block_health_after=after_health,
        requested_world_tick=requested_tick,
        started_world_tick=started_tick,
        finished_world_tick=finished_tick,
    )
    _validate_synthetic_interaction_row(verb, row)
    return row


def _synthetic_fixture_target(
    info: Mapping[str, object],
    verb: str,
) -> tuple[int, int, int]:
    return tuple(
        _integer(
            info,
            f"synthetic_world_interaction_fixture_{verb}_target_{axis}",
        )
        for axis in "xyz"
    )


def _validate_synthetic_interaction_row(
    verb: str,
    row: NativeSyntheticWorldInteractionRow,
) -> None:
    if not row.requested:
        if (
            row.accepted
            or row.started
            or row.finished
            or row.failed
            or row.mutation_applied
            or row.reject_reason
            or row.interaction_asset_id
            or row.held_item_asset_id
            or row.server_state != "unrequested"
            or row.target != (0, 0, 0)
            or row.runtime_block_id_before != 0
            or row.runtime_block_id_after != 0
            or row.block_health_before != 0.0
            or row.block_health_after != 0.0
            or row.requested_world_tick != -1
            or row.started_world_tick != -1
            or row.finished_world_tick != -1
        ):
            raise ValueError(
                f"{verb} unrequested synthetic row is inconsistent"
            )
        return
    if not row.accepted:
        if (
            not row.failed
            or not row.reject_reason
            or row.started
            or row.finished
            or row.mutation_applied
            or row.interaction_asset_id
            or row.held_item_asset_id
            or row.server_state != "rejected"
            or row.requested_world_tick != -1
            or row.started_world_tick != -1
            or row.finished_world_tick != -1
        ):
            raise ValueError(
                f"{verb} rejected synthetic row is inconsistent"
            )
        return
    if (
        row.reject_reason
        or not row.interaction_asset_id
        or not row.held_item_asset_id
        or row.requested_world_tick < 0
    ):
        raise ValueError(
            f"{verb} accepted synthetic row lacks bound evidence"
        )
    if row.started != (row.started_world_tick >= 0):
        raise ValueError(f"{verb} started tick is inconsistent")
    if row.finished != (row.finished_world_tick >= 0):
        raise ValueError(f"{verb} finished tick is inconsistent")
    if row.finished and not row.started:
        raise ValueError(f"{verb} finished before it started")
    if (
        row.started
        and row.started_world_tick < row.requested_world_tick
    ):
        raise ValueError(f"{verb} started before it was requested")
    if (
        row.finished
        and row.finished_world_tick < row.started_world_tick
    ):
        raise ValueError(f"{verb} finished before it started")
    if row.mutation_applied and not row.started:
        raise ValueError(f"{verb} mutated before it started")
    if verb == "place_block" and row.mutation_applied:
        if (
            row.runtime_block_id_before != 0
            or row.runtime_block_id_after == 0
        ):
            raise ValueError("place_block mutation outcome is inconsistent")
    if verb == "break_block" and row.mutation_applied:
        if not (
            row.runtime_block_id_after == 0
            or row.block_health_after < row.block_health_before
        ):
            raise ValueError("break_block mutation outcome is inconsistent")


def _native_world_verb_request_payload(
    request: NativeWorldVerbRequest,
) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "verb": request.verb,
        "interaction_type": request.interaction_type,
        "target_kind": request.target_kind,
        "interaction_id": request.interaction_id,
        "recipe_id": request.recipe_id,
        "item_id": request.item_id,
        "block_id": request.block_id,
        "target_x": request.target[0],
        "target_y": request.target[1],
        "target_z": request.target[2],
        "block_face": request.block_face,
        "rotation_yaw": request.rotation[0],
        "rotation_pitch": request.rotation[1],
        "rotation_roll": request.rotation[2],
        "source_container": request.source_container,
        "source_slot": request.source_slot,
        "expected_source_quantity": request.expected_source_quantity,
        "destination_container": request.destination_container,
        "expected_block_id": request.expected_block_id,
        "world_epoch": request.world_epoch,
        "quantity": request.quantity,
        "placement_variant": request.placement_variant,
        "crafting_context": request.crafting_context,
        "bench_x": request.bench[0],
        "bench_y": request.bench[1],
        "bench_z": request.bench[2],
        "expected_bench_block_id": request.expected_bench_block_id,
        "expected_bench_id": request.expected_bench_id,
        "expected_bench_type": request.expected_bench_type,
        "expected_bench_tier": request.expected_bench_tier,
        "expected_candidate_generation_sha256": (
            request.expected_candidate_generation_sha256
        ),
        "expected_selected_semantic_sha256": (
            request.expected_selected_semantic_sha256
        ),
    }


def _native_world_verb_request_payload_from_info(
    info: Mapping[str, object],
    prefix: str,
) -> dict[str, object]:
    string_limits = {
        "verb": 32,
        "target_kind": 32,
        "interaction_id": 256,
        "recipe_id": 256,
        "item_id": 256,
        "block_id": 256,
        "source_container": 64,
        "destination_container": 64,
        "expected_block_id": 256,
        "world_epoch": 128,
        "placement_variant": 128,
        "crafting_context": 32,
        "expected_bench_block_id": 256,
        "expected_bench_id": 128,
        "expected_candidate_generation_sha256": 64,
        "expected_selected_semantic_sha256": 64,
    }
    integers = (
        "request_id",
        "interaction_type",
        "target_x",
        "target_y",
        "target_z",
        "block_face",
        "rotation_yaw",
        "rotation_pitch",
        "rotation_roll",
        "source_slot",
        "expected_source_quantity",
        "quantity",
        "bench_x",
        "bench_y",
        "bench_z",
        "expected_bench_type",
        "expected_bench_tier",
    )
    payload: dict[str, object] = {
        name: _integer(info, prefix + name)
        for name in integers
    }
    payload.update(
        {
            name: _bounded_canonical_string(
                info,
                prefix + name,
                maximum=maximum,
            )
            for name, maximum in string_limits.items()
        }
    )
    return payload


def _native_world_verb_request_from_payload(
    payload: Mapping[str, object],
) -> NativeWorldVerbRequest:
    return NativeWorldVerbRequest(
        request_id=int(payload["request_id"]),
        verb=str(payload["verb"]),
        interaction_type=int(payload["interaction_type"]),
        target_kind=str(payload["target_kind"]),
        interaction_id=str(payload["interaction_id"]),
        recipe_id=str(payload["recipe_id"]),
        item_id=str(payload["item_id"]),
        block_id=str(payload["block_id"]),
        target=(
            int(payload["target_x"]),
            int(payload["target_y"]),
            int(payload["target_z"]),
        ),
        block_face=int(payload["block_face"]),
        rotation=(
            int(payload["rotation_yaw"]),
            int(payload["rotation_pitch"]),
            int(payload["rotation_roll"]),
        ),
        source_container=str(payload["source_container"]),
        source_slot=int(payload["source_slot"]),
        expected_source_quantity=int(
            payload["expected_source_quantity"]
        ),
        destination_container=str(payload["destination_container"]),
        expected_block_id=str(payload["expected_block_id"]),
        world_epoch=str(payload["world_epoch"]),
        quantity=int(payload["quantity"]),
        placement_variant=str(payload["placement_variant"]),
        crafting_context=str(payload["crafting_context"]),
        bench=(
            int(payload["bench_x"]),
            int(payload["bench_y"]),
            int(payload["bench_z"]),
        ),
        expected_bench_block_id=str(
            payload["expected_bench_block_id"]
        ),
        expected_bench_id=str(payload["expected_bench_id"]),
        expected_bench_type=int(payload["expected_bench_type"]),
        expected_bench_tier=int(payload["expected_bench_tier"]),
        expected_candidate_generation_sha256=str(
            payload["expected_candidate_generation_sha256"]
        ),
        expected_selected_semantic_sha256=str(
            payload["expected_selected_semantic_sha256"]
        ),
    )


def _validate_native_world_verb_request(
    request: NativeWorldVerbRequest,
) -> None:
    if (
        isinstance(request.request_id, bool)
        or not isinstance(request.request_id, int)
        or request.request_id < 0
        or request.request_id > 2**63 - 1
    ):
        raise ValueError("request_id must be a nonnegative int64")
    for name, value, maximum in (
        ("verb", request.verb, 32),
        ("target_kind", request.target_kind, 32),
        ("interaction_id", request.interaction_id, 256),
        ("recipe_id", request.recipe_id, 256),
        ("item_id", request.item_id, 256),
        ("block_id", request.block_id, 256),
        ("source_container", request.source_container, 64),
        ("destination_container", request.destination_container, 64),
        ("expected_block_id", request.expected_block_id, 256),
        ("world_epoch", request.world_epoch, 128),
        ("placement_variant", request.placement_variant, 128),
        ("crafting_context", request.crafting_context, 32),
        (
            "expected_bench_block_id",
            request.expected_bench_block_id,
            256,
        ),
        ("expected_bench_id", request.expected_bench_id, 128),
        (
            "expected_candidate_generation_sha256",
            request.expected_candidate_generation_sha256,
            64,
        ),
        (
            "expected_selected_semantic_sha256",
            request.expected_selected_semantic_sha256,
            64,
        ),
    ):
        if (
            not isinstance(value, str)
            or value != value.strip()
            or "\0" in value
            or len(value) > maximum
        ):
            raise ValueError(f"{name} must be a bounded canonical string")
    if request.verb not in NATIVE_TYPED_WORLD_VERBS:
        raise ValueError(f"unknown native World verb: {request.verb}")
    if not request.world_epoch:
        raise ValueError("world_epoch is required")
    generation = request.expected_candidate_generation_sha256
    semantic = request.expected_selected_semantic_sha256
    if bool(generation) != bool(semantic):
        raise ValueError(
            "candidate generation and selected semantic must be paired"
        )
    for name, value in (
        ("expected_candidate_generation_sha256", generation),
        ("expected_selected_semantic_sha256", semantic),
    ):
        if value and (
            len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise ValueError(f"{name} must be one SHA-256")
    if (
        isinstance(request.quantity, bool)
        or not isinstance(request.quantity, int)
        or request.quantity <= 0
        or request.quantity > 2**31 - 1
    ):
        raise ValueError("quantity must be a positive int32")
    if (
        not isinstance(request.target, tuple)
        or len(request.target) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < -(2**31)
            or value > 2**31 - 1
            for value in request.target
        )
    ):
        raise ValueError("target must contain three int32 coordinates")
    if (
        isinstance(request.block_face, bool)
        or not isinstance(request.block_face, int)
        or request.block_face not in range(7)
    ):
        raise ValueError("block_face must be in [0, 6]")
    if (
        not isinstance(request.rotation, tuple)
        or len(request.rotation) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in range(4)
            for value in request.rotation
        )
    ):
        raise ValueError("rotation must contain three values in [0, 3]")
    if (
        not isinstance(request.bench, tuple)
        or len(request.bench) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < -(2**31)
            or value > 2**31 - 1
            for value in request.bench
        )
    ):
        raise ValueError("bench must contain three int32 coordinates")
    for name, value in (
        ("expected_bench_type", request.expected_bench_type),
        ("expected_bench_tier", request.expected_bench_tier),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < -(2**31)
            or value > 2**31 - 1
        ):
            raise ValueError(f"{name} must be an int32")
    if request.verb == "craft_recipe":
        if (
            request.interaction_type != -1
            or request.target_kind != "recipe"
            or not request.recipe_id
            or request.source_container != "player_inventory"
            or request.source_slot != -1
            or request.expected_source_quantity != -1
            or request.destination_container
            != "player_inventory_or_world_drop"
            or request.interaction_id
            or request.item_id
            or request.block_id
            or request.expected_block_id
            or request.placement_variant
            or request.target != (0, 0, 0)
            or request.block_face != 0
            or request.rotation != (0, 0, 0)
            or request.crafting_context not in ("fieldcraft", "bench")
        ):
            raise ValueError(
                "craft request is incomplete or carries block-only fields"
            )
        if request.crafting_context == "fieldcraft":
            if (
                request.bench != (0, 0, 0)
                or request.expected_bench_block_id
                or request.expected_bench_id != "Fieldcraft"
                or request.expected_bench_type != 0
                or request.expected_bench_tier != 0
            ):
                raise ValueError(
                    "fieldcraft request has an invalid crafting context"
                )
        elif (
            request.bench[1] not in range(320)
            or not request.expected_bench_block_id
            or not request.expected_bench_id
            or request.expected_bench_type not in range(4)
            or request.expected_bench_tier < 1
        ):
            raise ValueError(
                "bench craft request has an invalid crafting context"
            )
        return
    unarmed_use = (
        request.verb == "use"
        and request.source_container == "unarmed"
    )
    context_use = (
        request.verb == "use"
        and request.source_container == "interaction_context"
    )
    if (
        request.target_kind != "block"
        or request.target[1] not in range(320)
        or not request.interaction_id
        or not request.item_id
        or not request.expected_block_id
        or request.recipe_id
        or request.destination_container
        or request.crafting_context
        or request.bench != (0, 0, 0)
        or request.expected_bench_block_id
        or request.expected_bench_id
        or request.expected_bench_type != -1
        or request.expected_bench_tier != 0
    ):
        raise ValueError("block verb request is incomplete")
    if unarmed_use:
        if (
            request.item_id != "Empty"
            or request.source_slot != -1
            or request.expected_source_quantity != 0
        ):
            raise ValueError(
                "unarmed Use requires the explicit Empty sentinel"
            )
    elif context_use:
        if (
            request.item_id == "Empty"
            or request.source_slot < 0
            or request.expected_source_quantity < request.quantity
        ):
            raise ValueError(
                "context Use requires an exact resolved item"
            )
    elif (
        not request.source_container
        or request.source_slot < 0
        or request.source_slot > 2**31 - 1
        or request.expected_source_quantity < request.quantity
        or request.expected_source_quantity > 2**31 - 1
    ):
        raise ValueError("block verb request is missing its source item")
    if (
        request.verb == "use"
        and request.interaction_type != 5
    ) or (
        request.verb != "use"
        and request.interaction_type not in (0, 1)
    ):
        raise ValueError(
            "World verb request has an invalid InteractionType"
        )
    if request.verb == "place_block":
        if not request.block_id or not request.placement_variant:
            raise ValueError(
                "place request requires block_id and placement_variant"
            )
    elif request.block_id or request.placement_variant:
        raise ValueError(
            "only place_block accepts block_id or placement_variant"
        )


def _validate_native_world_verb_receipt(
    receipt: NativeWorldVerbReceipt,
    payload: Mapping[str, object],
) -> None:
    prefix = "native World-verb receipt"
    if receipt.request is None:
        empty = {
            "request_id": -1,
            "verb": "",
            "interaction_type": -1,
            "target_kind": "",
            "interaction_id": "",
            "recipe_id": "",
            "item_id": "",
            "block_id": "",
            "target_x": 0,
            "target_y": 0,
            "target_z": 0,
            "block_face": 0,
            "rotation_yaw": 0,
            "rotation_pitch": 0,
            "rotation_roll": 0,
            "source_container": "",
            "source_slot": -1,
            "expected_source_quantity": -1,
            "destination_container": "",
            "expected_block_id": "",
            "world_epoch": "",
            "quantity": 0,
            "placement_variant": "",
            "crafting_context": "",
            "bench_x": 0,
            "bench_y": 0,
            "bench_z": 0,
            "expected_bench_block_id": "",
            "expected_bench_id": "",
            "expected_bench_type": -1,
            "expected_bench_tier": 0,
            "expected_candidate_generation_sha256": "",
            "expected_selected_semantic_sha256": "",
        }
        if dict(payload) != empty:
            raise ValueError(f"{prefix} unrequested echo is inconsistent")
        if (
            receipt.accepted
            or receipt.started
            or receipt.active
            or receipt.finished
            or receipt.failed
            or receipt.reject_reason
            or receipt.execution_scope != "unrequested"
            or receipt.public_player_certified
            or receipt.headless_server_actor_certified
            or receipt.opens_action_mask
            or receipt.acknowledgement_complete
            or receipt.mutation_applied
            or receipt.inventory_applied
            or receipt.semantic_block_id_before
            or receipt.semantic_block_id_after
            or receipt.runtime_block_id_before != -1
            or receipt.runtime_block_id_after != -1
            or receipt.source_quantity_before != -1
            or receipt.source_quantity_after != -1
            or receipt.destination_quantity_before != -1
            or receipt.destination_quantity_after != -1
            or receipt.geometry_delta_sha256
            or receipt.inventory_delta_sha256
            or receipt.drop_outcome_sha256
            or receipt.requested_world_tick != -1
            or receipt.started_world_tick != -1
            or receipt.finished_world_tick != -1
        ):
            raise ValueError(f"{prefix} unrequested lifecycle is inconsistent")
        return

    request = receipt.request
    if receipt.requested_world_tick < 0:
        raise ValueError(f"{prefix} lacks a request tick")
    if request.world_epoch != receipt.current_world_epoch:
        if (
            receipt.accepted
            or not receipt.failed
            or receipt.reject_reason != "stale_world_epoch"
        ):
            raise ValueError(f"{prefix} accepted a stale epoch")
    if receipt.accepted:
        if receipt.reject_reason:
            raise ValueError(f"{prefix} accepted with a rejection reason")
        expected_scope = {
            "use": {
                "headless_native_target_relative_use_chain",
            },
            "place_block": {
                "fixture_headless_player_native_place_chain",
                "opt_in_headless_player_native_place_chain",
            },
            "break_block": {
                "fixture_headless_player_native_break_chain",
                "opt_in_headless_player_native_break_chain",
            },
            "craft_recipe": {
                "opt_in_headless_player_native_crafting_manager",
            },
        }.get(request.verb, set())
        if receipt.execution_scope not in expected_scope:
            raise ValueError(f"{prefix} has an unknown execution scope")
    elif (
        not receipt.failed
        or not receipt.reject_reason
        or receipt.public_player_certified
        or receipt.headless_server_actor_certified
        or receipt.opens_action_mask
        or receipt.started
        or receipt.active
        or receipt.finished
        or receipt.acknowledgement_complete
        or receipt.mutation_applied
        or receipt.inventory_applied
        or receipt.semantic_block_id_before
        or receipt.semantic_block_id_after
        or receipt.runtime_block_id_before != -1
        or receipt.runtime_block_id_after != -1
        or receipt.source_quantity_before != -1
        or receipt.source_quantity_after != -1
        or receipt.destination_quantity_before != -1
        or receipt.destination_quantity_after != -1
        or receipt.geometry_delta_sha256
        or receipt.inventory_delta_sha256
        or receipt.drop_outcome_sha256
    ):
        raise ValueError(f"{prefix} rejection lifecycle is inconsistent")
    elif receipt.execution_scope != "transport_only_rejected":
        raise ValueError(f"{prefix} has an unknown rejection scope")
    if (
        receipt.public_player_certified
        and receipt.headless_server_actor_certified
    ):
        raise ValueError(f"{prefix} has conflicting actor certification")
    if receipt.started != (receipt.started_world_tick >= 0):
        raise ValueError(f"{prefix} start tick is inconsistent")
    if receipt.finished != (receipt.finished_world_tick >= 0):
        raise ValueError(f"{prefix} finish tick is inconsistent")
    if receipt.active and (not receipt.started or receipt.finished):
        raise ValueError(f"{prefix} active lifecycle is inconsistent")
    if (
        receipt.started
        and receipt.started_world_tick < receipt.requested_world_tick
    ):
        raise ValueError(f"{prefix} started before its request")
    if (
        receipt.finished
        and receipt.finished_world_tick < receipt.started_world_tick
    ):
        raise ValueError(f"{prefix} finished before its start")
    if receipt.acknowledgement_complete:
        if not receipt.accepted or not receipt.finished:
            raise ValueError(f"{prefix} acknowledged an incomplete lifecycle")
        if request.target_kind == "block" and (
            not receipt.semantic_block_id_before
            or not receipt.semantic_block_id_after
            or not receipt.geometry_delta_sha256
        ):
            raise ValueError(f"{prefix} lacks exact geometry evidence")
        if not receipt.inventory_delta_sha256:
            raise ValueError(f"{prefix} lacks exact inventory evidence")
    if receipt.opens_action_mask and (
        not (
            receipt.public_player_certified
            or receipt.headless_server_actor_certified
        )
        or not receipt.acknowledgement_complete
    ):
        raise ValueError(f"{prefix} opens the action mask without certification")


__all__ = [
    "NATIVE_BLOCK_USE_EVIDENCE_SCHEMA",
    "NATIVE_BLOCK_USE_EVIDENCE_VERSION",
    "NATIVE_WORLD_ACTION_CAPABILITIES_SCHEMA",
    "NATIVE_WORLD_ACTION_CAPABILITIES_VERSION",
    "NATIVE_WORLD_VERB_LIFECYCLE_SCHEMA",
    "NATIVE_WORLD_VERB_LIFECYCLE_VERSION",
    "NATIVE_WORLD_VERBS",
    "NATIVE_WORLD_VERB_TRANSPORT_SCHEMA",
    "NATIVE_WORLD_VERB_TRANSPORT_VERSION",
    "NATIVE_TYPED_WORLD_VERBS",
    "NATIVE_SYNTHETIC_WORLD_INTERACTION_SCHEMA",
    "NATIVE_SYNTHETIC_WORLD_INTERACTION_VERSION",
    "NATIVE_SYNTHETIC_WORLD_INTERACTION_VERBS",
    "NativeBlockUseEvidence",
    "NativeSyntheticWorldInteractionEvidence",
    "NativeSyntheticWorldInteractionRow",
    "NativeWorldActionCapabilities",
    "NativeWorldVerbLifecycle",
    "NativeWorldVerbReceipt",
    "NativeWorldVerbRequest",
    "NativeWorldVerbTransportSession",
    "NativeWorldVerbTelemetry",
    "native_block_use_evidence_contract",
    "native_block_use_evidence_contract_sha256",
    "native_world_action_capabilities_contract",
    "native_world_action_capabilities_contract_sha256",
    "native_world_verb_reset_options",
    "native_world_verb_lifecycle_contract",
    "native_world_verb_lifecycle_contract_sha256",
    "native_world_verb_transport_contract",
    "native_world_verb_transport_contract_sha256",
    "native_synthetic_world_interaction_contract",
    "native_synthetic_world_interaction_contract_sha256",
]
