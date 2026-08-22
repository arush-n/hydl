package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence;
import java.io.IOException;
import org.msgpack.core.MessagePacker;

/**
 * Extracted verbatim from {@code ClientHandler}.
 */
public final class ItemInteractionCodec {

    private ItemInteractionCodec() {}

    public static void packItemInteractionEvidence(
        MessagePacker packer,
        NativeItemInteractionEvidence evidence,
        String bridgeSha256
    ) throws IOException {
        if (
            bridgeSha256 == null
                || !bridgeSha256.matches("[0-9A-Fa-f]{64}")
        ) {
            throw new IllegalArgumentException(
                "Runtime bridge identity must be a SHA-256"
            );
        }
        packer.packMapHeader(23);
        packer.packString("type");
        packer.packString("item_interaction_evidence");
        packer.packString("schema");
        packer.packString(NativeItemInteractionEvidence.SCHEMA);
        packer.packString("version");
        packer.packInt(NativeItemInteractionEvidence.VERSION);
        packer.packString("bridge_sha256");
        packer.packString(bridgeSha256);
        packer.packString("server_version");
        packer.packString(evidence.serverVersion());
        packer.packString("world");
        packer.packString(evidence.worldName());
        packer.packString("worldgen_provider");
        packer.packString(evidence.worldgenProvider());
        packer.packString("worldgen_version");
        packer.packString(evidence.worldgenVersion());
        packer.packString("seed");
        packer.packLong(evidence.seed());
        packer.packString("active_game_mode");
        packer.packInt(evidence.activeGameMode());
        packer.packString("equipped_slot");
        packer.packInt(evidence.equippedSlot());
        packer.packString("equipped_slot_capacity");
        packer.packInt(evidence.equippedSlotCapacity());
        packer.packString("equipped_slot_available");
        packer.packBoolean(evidence.equippedSlotAvailable());
        packer.packString("expected_trigger_count");
        packer.packInt(NativeItemInteractionEvidence.TRIGGER_CAPACITY);
        packer.packString("trigger_count");
        packer.packInt(evidence.triggers().size());
        packer.packString("interaction_capacity_per_trigger");
        packer.packInt(NativeItemInteractionEvidence.INTERACTION_CAPACITY);
        packer.packString("edge_capacity_per_trigger");
        packer.packInt(NativeItemInteractionEvidence.EDGE_CAPACITY);
        packer.packString("charge_time_capacity");
        packer.packInt(NativeItemInteractionEvidence.CHARGE_TIME_CAPACITY);
        packer.packString("block_change_capacity_per_node");
        packer.packInt(
            NativeItemInteractionEvidence.BLOCK_CHANGE_CAPACITY
        );
        packer.packString("item_metadata_capacity");
        packer.packInt(
            NativeItemInteractionEvidence.ITEM_METADATA_CAPACITY
        );
        packer.packString("evidence_scope");
        packer.packString(
            "resolved_item_roots_rules_effects_and_reachable_authored_chain"
        );
        packer.packString("public_player_acceptance_certified");
        packer.packBoolean(false);
        packer.packString("triggers");
        packer.packArrayHeader(evidence.triggers().size());
        for (NativeItemInteractionEvidence.Trigger trigger
            : evidence.triggers()) {
            packItemInteractionTrigger(packer, trigger);
        }
    }


    public static void packItemInteractionTrigger(
        MessagePacker packer,
        NativeItemInteractionEvidence.Trigger trigger
    ) throws IOException {
        packer.packMapHeader(11);
        packer.packString("interaction_type");
        packer.packInt(trigger.interactionType());
        packer.packString("interaction_type_name");
        packer.packString(trigger.interactionTypeName());
        packer.packString("item_asset_id");
        packer.packString(trigger.itemAssetId());
        packer.packString("held_item_section_id");
        packer.packInt(trigger.heldItemSectionId());
        packer.packString("held_item_slot");
        packer.packInt(trigger.heldItemSlot());
        packer.packString("root_present");
        packer.packBoolean(trigger.rootPresent());
        packer.packString("root");
        if (trigger.root() == null) {
            packer.packNil();
        } else {
            packItemInteractionRoot(packer, trigger.root());
        }
        packer.packString("node_count");
        packer.packInt(trigger.nodes().size());
        packer.packString("nodes");
        packer.packArrayHeader(trigger.nodes().size());
        for (NativeItemInteractionEvidence.Node node : trigger.nodes()) {
            packItemInteractionNode(packer, node);
        }
        packer.packString("edge_count");
        packer.packInt(trigger.edges().size());
        packer.packString("edges");
        packer.packArrayHeader(trigger.edges().size());
        for (NativeItemInteractionEvidence.Edge edge : trigger.edges()) {
            packer.packMapHeader(3);
            packer.packString("parent_interaction_id");
            packer.packString(edge.parentInteractionId());
            packer.packString("child_interaction_id");
            packer.packString(edge.childInteractionId());
            packer.packString("relation");
            packer.packString(edge.relation());
        }
    }


    public static void packItemInteractionRoot(
        MessagePacker packer,
        NativeItemInteractionEvidence.Root root
    ) throws IOException {
        packer.packMapHeader(6);
        packer.packString("root_id");
        packer.packString(root.rootId());
        packer.packString("click_queuing_timeout");
        packer.packFloat(root.clickQueuingTimeout());
        packer.packString("require_new_click");
        packer.packBoolean(root.requireNewClick());
        packer.packString("rules");
        packItemInteractionRules(packer, root.rules());
        packer.packString("mode_settings");
        packer.packArrayHeader(root.modeSettings().size());
        for (NativeItemInteractionEvidence.ModeSettings settings
            : root.modeSettings()) {
            packer.packMapHeader(8);
            packer.packString("game_mode");
            packer.packInt(settings.gameMode());
            packer.packString("allow_skip_chain_on_click");
            packer.packBoolean(settings.allowSkipChainOnClick());
            packer.packString("cooldown_id");
            packer.packString(settings.cooldownId());
            packer.packString("cooldown_seconds");
            packer.packFloat(settings.cooldownSeconds());
            packer.packString("click_bypass");
            packer.packBoolean(settings.clickBypass());
            packer.packString("skip_cooldown_reset");
            packer.packBoolean(settings.skipCooldownReset());
            packer.packString("interrupt_recharge");
            packer.packBoolean(settings.interruptRecharge());
            packer.packString("charge_times");
            packer.packArrayHeader(settings.chargeTimes().size());
            for (float value : settings.chargeTimes()) {
                packer.packFloat(value);
            }
        }
        packer.packString("initial_interaction_ids");
        packer.packArrayHeader(root.initialInteractionIds().size());
        for (String value : root.initialInteractionIds()) {
            packer.packString(value);
        }
    }


    public static void packItemInteractionRules(
        MessagePacker packer,
        NativeItemInteractionEvidence.Rules rules
    ) throws IOException {
        packer.packMapHeader(11);
        packer.packString("blocked_by_mask");
        packer.packInt(rules.blockedByMask());
        packer.packString("blocked_by_uses_default");
        packer.packBoolean(rules.blockedByUsesDefault());
        packer.packString("blocking_mask");
        packer.packInt(rules.blockingMask());
        packer.packString("interrupted_by_mask");
        packer.packInt(rules.interruptedByMask());
        packer.packString("interrupted_by_unspecified");
        packer.packBoolean(rules.interruptedByUnspecified());
        packer.packString("interrupting_mask");
        packer.packInt(rules.interruptingMask());
        packer.packString("interrupting_unspecified");
        packer.packBoolean(rules.interruptingUnspecified());
        packer.packString("blocked_by_bypass_index");
        packer.packInt(rules.blockedByBypassIndex());
        packer.packString("blocking_bypass_index");
        packer.packInt(rules.blockingBypassIndex());
        packer.packString("interrupted_by_bypass_index");
        packer.packInt(rules.interruptedByBypassIndex());
        packer.packString("interrupting_bypass_index");
        packer.packInt(rules.interruptingBypassIndex());
    }


    public static void packItemInteractionNode(
        MessagePacker packer,
        NativeItemInteractionEvidence.Node node
    ) throws IOException {
        packer.packMapHeader(17);
        packer.packString("interaction_id");
        packer.packString(node.interactionId());
        packer.packString("implementation_class");
        packer.packString(node.implementationClass());
        packer.packString("run_time");
        packer.packFloat(node.runTime());
        packer.packString("cancel_on_item_change");
        packer.packBoolean(node.cancelOnItemChange());
        packer.packString("wait_for_data_from");
        packer.packInt(node.waitForDataFrom());
        packer.packString("next_interaction_id");
        packer.packString(node.nextInteractionId());
        packer.packString("failed_interaction_id");
        packer.packString(node.failedInteractionId());
        packer.packString("use_latest_target");
        packer.packBoolean(node.useLatestTarget());
        packer.packString("harvest");
        packer.packBoolean(node.harvest());
        packer.packString("horizontal_speed_multiplier");
        packer.packFloat(node.horizontalSpeedMultiplier());
        packer.packString("effects_present");
        packer.packBoolean(node.effectsPresent());
        packer.packString("start_delay");
        packer.packFloat(node.startDelay());
        packer.packString("wait_for_animation_to_finish");
        packer.packBoolean(node.waitForAnimationToFinish());
        packer.packString("movement_effects_present");
        packer.packBoolean(node.movementEffectsPresent());
        packer.packString("movement_disable_all");
        packer.packBoolean(node.movementDisableAll());
        packer.packString("movement_lock_mask");
        packer.packInt(node.movementLockMask());
        packer.packString("payload");
        packItemInteractionPayload(packer, node.payload());
    }


    public static void packItemInteractionPayload(
        MessagePacker packer,
        NativeItemInteractionEvidence.Payload payload
    ) throws IOException {
        if (payload instanceof NativeItemInteractionEvidence.EmptyPayload) {
            packer.packMapHeader(1);
            packer.packString("kind");
            packer.packString("");
        } else if (
            payload
                instanceof NativeItemInteractionEvidence.BreakBlockPayload breaking
        ) {
            packer.packMapHeader(3);
            packer.packString("kind");
            packer.packString(breaking.kind());
            packer.packString("tool_id");
            packer.packString(breaking.toolId());
            packer.packString("match_tool");
            packer.packBoolean(breaking.matchTool());
        } else if (
            payload
                instanceof NativeItemInteractionEvidence.PlaceBlockPayload place
        ) {
            packer.packMapHeader(4);
            packer.packString("kind");
            packer.packString(place.kind());
            packer.packString("block_asset_id");
            packer.packString(place.blockAssetId());
            packer.packString("remove_item_in_hand");
            packer.packBoolean(place.removeItemInHand());
            packer.packString("allow_drag_placement");
            packer.packBoolean(place.allowDragPlacement());
        } else if (
            payload
                instanceof NativeItemInteractionEvidence.ChangeBlockPayload change
        ) {
            packer.packMapHeader(4);
            packer.packString("kind");
            packer.packString(change.kind());
            packer.packString("changes");
            packer.packArrayHeader(change.changes().size());
            for (
                NativeItemInteractionEvidence.BlockChange row
                    : change.changes()
            ) {
                packer.packArrayHeader(2);
                packer.packString(row.fromBlockAssetId());
                packer.packString(row.toBlockAssetId());
            }
            packer.packString("world_sound_event_asset_id");
            packer.packString(change.worldSoundEventAssetId());
            packer.packString("require_not_broken");
            packer.packBoolean(change.requireNotBroken());
        } else if (
            payload
                instanceof NativeItemInteractionEvidence.ModifyInventoryPayload modify
        ) {
            packer.packMapHeader(10);
            packer.packString("kind");
            packer.packString(modify.kind());
            packer.packString("required_game_mode");
            packer.packInt(modify.requiredGameMode());
            packer.packString("item_to_remove");
            packItemInteractionItem(packer, modify.itemToRemove());
            packer.packString("adjust_held_item_quantity");
            packer.packInt(modify.adjustHeldItemQuantity());
            packer.packString("item_to_add");
            packItemInteractionItem(packer, modify.itemToAdd());
            packer.packString("broken_item_asset_id");
            packer.packString(modify.brokenItemAssetId());
            packer.packString("adjust_held_item_durability");
            packer.packDouble(modify.adjustHeldItemDurability());
            packer.packString("notify_on_break_specified");
            packer.packBoolean(modify.notifyOnBreakSpecified());
            packer.packString("notify_on_break");
            packer.packBoolean(modify.notifyOnBreak());
            packer.packString("notify_on_break_message");
            packer.packString(modify.notifyOnBreakMessage());
        } else {
            throw new IllegalStateException(
                "Unknown native interaction payload"
            );
        }
    }


    public static void packItemInteractionItem(
        MessagePacker packer,
        NativeItemInteractionEvidence.ItemPayload item
    ) throws IOException {
        if (item == null) {
            packer.packNil();
            return;
        }
        packer.packMapHeader(6);
        packer.packString("item_asset_id");
        packer.packString(item.itemAssetId());
        packer.packString("quantity");
        packer.packInt(item.quantity());
        packer.packString("durability");
        packer.packDouble(item.durability());
        packer.packString("max_durability");
        packer.packDouble(item.maxDurability());
        packer.packString("override_dropped_item_animation");
        packer.packBoolean(item.overrideDroppedItemAnimation());
        packer.packString("metadata_json");
        packer.packString(item.metadataJson());
    }
}
