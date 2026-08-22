package com.hytalerlbridge.worldgen;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.GameMode;
import com.hypixel.hytale.protocol.InteractionCooldown;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.protocol.RootInteractionSettings;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.item.config.Item;
import com.hypixel.hytale.server.core.asset.type.soundevent.config.SoundEvent;
import com.hypixel.hytale.server.core.entity.InteractionContext;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.entity.entities.Player;
import com.hypixel.hytale.server.core.inventory.InventoryComponent;
import com.hypixel.hytale.server.core.modules.interaction.interaction.UnarmedInteractions;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.Interaction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.InteractionRules;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.InteractionTypeUtils;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.data.Collector;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.data.CollectorTag;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.data.StringTag;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.Trigger;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.Root;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.ModeSettings;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.Rules;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.Node;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.Payload;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.EmptyPayload;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.BreakBlockPayload;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.PlaceBlockPayload;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.BlockChange;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.ChangeBlockPayload;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.ItemPayload;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.ModifyInventoryPayload;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence.Edge;

/**
 * Builder helpers for {@link NativeItemInteractionEvidence}.
 *
 * <p>Extracted verbatim from the evidence record. These are pure static
 * projections from decompiled interaction config into the evidence value types;
 * they hold no state and touch no instance field, which is what made the
 * extraction mechanical.</p>
 *
 * <p>They stay in the {@code com.hytalerlbridge.worldgen} package rather than a subpackage
 * so that package-private access to the evidence record is preserved and the
 * public API surface is unchanged.</p>
 */
final class NativeItemInteractionEvidenceBuilders {

    private NativeItemInteractionEvidenceBuilders() {}

    static String selectedItemRootInteractionId(
        Item item,
        InteractionType type
    ) {
        if (item != null) {
            return NativeItemInteractionEvidence.itemRootInteractionId(item.getInteractions(), type);
        }
        UnarmedInteractions unarmed =
            UnarmedInteractions.getAssetMap().getAsset("Empty");
        return unarmed == null
            ? null
            : NativeItemInteractionEvidence.itemRootInteractionId(unarmed.getInteractions(), type);
    }

    static Trigger trigger(
        InteractionType type,
        InteractionContext context,
        Item item,
        RootInteraction root
    ) {
        if (root == null) {
            return new Trigger(
                type.getValue(),
                type.name(),
                item == null ? "" : item.getId(),
                context.getHeldItemSectionId(),
                context.getHeldItemSlot(),
                false,
                null,
                List.of(),
                List.of()
            );
        }
        NativeItemInteractionGraphCollector collector = new NativeItemInteractionGraphCollector();
        InteractionManager.walkChain(
            collector,
            type,
            context,
            root
        );
        return new Trigger(
            type.getValue(),
            type.name(),
            item == null ? "" : item.getId(),
            context.getHeldItemSectionId(),
            context.getHeldItemSlot(),
            true,
            root(type, root),
            collector.nodes(),
            collector.edges()
        );
    }

    static Root root(InteractionType type, RootInteraction root) {
        com.hypixel.hytale.protocol.RootInteraction packet = root.toPacket();
        List<ModeSettings> settings = new ArrayList<>(
            GameMode.VALUES.length
        );
        for (GameMode mode : GameMode.VALUES) {
            settings.add(modeSettings(type, root, mode));
        }
        return new Root(
            root.getId(),
            root.getClickQueuingTimeout(),
            packet.requireNewClick,
            rules(type, root.getRules()),
            settings,
            List.of(root.getInteractionIds())
        );
    }

    static ModeSettings modeSettings(
        InteractionType type,
        RootInteraction root,
        GameMode mode
    ) {
        InteractionCooldown cooldown = root.getCooldown();
        RootInteractionSettings settings = root.getSettings().get(mode);
        if (settings != null && settings.cooldown != null) {
            cooldown = settings.cooldown;
        }
        String cooldownId = root.getId();
        float seconds = InteractionTypeUtils.getDefaultCooldown(type);
        float[] chargeTimes = InteractionManager.DEFAULT_CHARGE_TIMES;
        boolean clickBypass = false;
        boolean skipCooldownReset = false;
        boolean interruptRecharge = false;
        if (cooldown != null) {
            if (cooldown.cooldownId != null) {
                cooldownId = cooldown.cooldownId;
            }
            seconds = cooldown.cooldown;
            if (
                cooldown.chargeTimes != null
                    && cooldown.chargeTimes.length > 0
            ) {
                chargeTimes = cooldown.chargeTimes;
            }
            clickBypass = cooldown.clickBypass;
            skipCooldownReset = cooldown.skipCooldownReset;
            interruptRecharge = cooldown.interruptRecharge;
        }
        if (chargeTimes.length > NativeItemInteractionEvidence.CHARGE_TIME_CAPACITY) {
            throw new IllegalStateException(
                "Native interaction charge-time capacity exceeded"
            );
        }
        List<Float> charges = new ArrayList<>(chargeTimes.length);
        for (float value : chargeTimes) {
            if (!Float.isFinite(value) || value < 0.0f) {
                throw new IllegalStateException(
                    "Native interaction has an invalid charge time"
                );
            }
            charges.add(value);
        }
        return new ModeSettings(
            mode.getValue(),
            settings != null && settings.allowSkipChainOnClick,
            cooldownId,
            seconds,
            clickBypass,
            skipCooldownReset,
            interruptRecharge,
            charges
        );
    }

    static Rules rules(
        InteractionType type,
        InteractionRules rules
    ) {
        com.hypixel.hytale.protocol.InteractionRules packet =
            rules.toPacket();
        boolean defaultBlockedBy = packet.blockedBy == null;
        Set<InteractionType> blockedBy = defaultBlockedBy
            ? InteractionTypeUtils.DEFAULT_INTERACTION_BLOCKED_BY.get(type)
            : Set.of(packet.blockedBy);
        return new Rules(
            interactionMask(blockedBy),
            defaultBlockedBy,
            interactionMask(packet.blocking),
            interactionMask(packet.interruptedBy),
            packet.interruptedBy == null,
            interactionMask(packet.interrupting),
            packet.interrupting == null,
            packet.blockedByBypassIndex,
            packet.blockingBypassIndex,
            packet.interruptedByBypassIndex,
            packet.interruptingBypassIndex
        );
    }

    static int interactionMask(InteractionType[] values) {
        return values == null ? 0 : interactionMask(Set.of(values));
    }

    static int interactionMask(Set<InteractionType> values) {
        int mask = 0;
        if (values == null) return mask;
        for (InteractionType value : values) {
            mask |= 1 << value.getValue();
        }
        return mask;
    }

    static Node node(Interaction interaction) {
        com.hypixel.hytale.protocol.Interaction packet =
            interaction.toPacket();
        String next = "";
        String failed = "";
        boolean useLatestTarget = false;
        boolean harvest = false;
        if (
            packet
                instanceof com.hypixel.hytale.protocol.SimpleInteraction simple
        ) {
            next = interactionId(simple.next);
            failed = interactionId(simple.failed);
        }
        if (
            packet
                instanceof com.hypixel.hytale.protocol.SimpleBlockInteraction block
        ) {
            useLatestTarget = block.useLatestTarget;
        }
        if (
            packet
                instanceof com.hypixel.hytale.protocol.BreakBlockInteraction breaking
        ) {
            harvest = breaking.harvest;
        }
        com.hypixel.hytale.protocol.InteractionEffects effects =
            packet.effects;
        com.hypixel.hytale.protocol.MovementEffects movement =
            effects == null ? null : effects.movementEffects;
        var nativeEffects = interaction.getEffects();
        var nativeMovement = nativeEffects == null
            ? null
            : nativeEffects.getMovementEffects();
        float runTime = interaction.getRunTime();
        float horizontalSpeedMultiplier =
            interaction.getHorizontalSpeedMultiplier();
        float startDelay = effects == null ? 0.0f : effects.startDelay;
        if (
            !Float.isFinite(runTime)
                || runTime < 0.0f
                || !Float.isFinite(horizontalSpeedMultiplier)
                || !Float.isFinite(startDelay)
                || startDelay < 0.0f
        ) {
            throw new IllegalStateException(
                "Native interaction has invalid timing or movement data"
            );
        }
        return new Node(
            interaction.getId(),
            interaction.getClass().getName(),
            runTime,
            interaction.isCancelOnItemChange(),
            interaction.getWaitForDataFrom().getValue(),
            next,
            failed,
            useLatestTarget,
            harvest,
            horizontalSpeedMultiplier,
            effects != null,
            startDelay,
            effects != null && effects.waitForAnimationToFinish,
            movement != null,
            nativeMovement != null && nativeMovement.isDisableAll(),
            movementLockMask(movement),
            payload(interaction, packet)
        );
    }

    static int movementLockMask(
        com.hypixel.hytale.protocol.MovementEffects effects
    ) {
        if (effects == null) return 0;
        return (effects.disableForward ? 1 : 0)
            | (effects.disableBackward ? 1 << 1 : 0)
            | (effects.disableLeft ? 1 << 2 : 0)
            | (effects.disableRight ? 1 << 3 : 0)
            | (effects.disableSprint ? 1 << 4 : 0)
            | (effects.disableJump ? 1 << 5 : 0)
            | (effects.disableCrouch ? 1 << 6 : 0);
    }

    static Payload payload(
        Interaction interaction,
        com.hypixel.hytale.protocol.Interaction packet
    ) {
        if (
            packet
                instanceof com.hypixel.hytale.protocol.BreakBlockInteraction
        ) {
            if (
                !(interaction
                    instanceof com.hypixel.hytale.server.core.modules.interaction.interaction.config.client.BreakBlockInteraction)
            ) {
                throw new IllegalStateException(
                    "Break-block packet came from an unexpected interaction"
                );
            }
            Object toolId = privateField(interaction, "toolId");
            Object matchTool = privateField(interaction, "matchTool");
            if (!(matchTool instanceof Boolean)) {
                throw new IllegalStateException(
                    "Break-block MatchTool field is unavailable"
                );
            }
            return new BreakBlockPayload(
                toolId == null ? "" : toolId.toString(),
                (Boolean) matchTool
            );
        }
        if (
            packet
                instanceof com.hypixel.hytale.protocol.PlaceBlockInteraction place
        ) {
            return new PlaceBlockPayload(
                blockAssetId(place.blockId, true),
                place.removeItemInHand,
                place.allowDragPlacement
            );
        }
        if (
            packet
                instanceof com.hypixel.hytale.protocol.ChangeBlockInteraction change
        ) {
            List<BlockChange> changes = new ArrayList<>();
            if (change.blockChanges != null) {
                for (
                    Map.Entry<Integer, Integer> entry
                        : change.blockChanges.entrySet()
                ) {
                    changes.add(new BlockChange(
                        blockAssetId(entry.getKey(), false),
                        blockAssetId(entry.getValue(), false)
                    ));
                }
                changes.sort(
                    Comparator.comparing(BlockChange::fromBlockAssetId)
                        .thenComparing(BlockChange::toBlockAssetId)
                );
            }
            if (changes.size() > NativeItemInteractionEvidence.BLOCK_CHANGE_CAPACITY) {
                throw new IllegalStateException(
                    "Native block-change payload exceeds capacity"
                );
            }
            return new ChangeBlockPayload(
                changes,
                soundEventAssetId(change.worldSoundEventIndex),
                change.requireNotBroken
            );
        }
        if (
            packet
                instanceof com.hypixel.hytale.protocol.ModifyInventoryInteraction modify
        ) {
            Object requiredMode = privateField(
                interaction,
                "requiredGameMode"
            );
            Object notify = privateField(interaction, "notifyOnBreak");
            Object notifyMessage = privateField(
                interaction,
                "notifyOnBreakMessage"
            );
            return new ModifyInventoryPayload(
                requiredMode instanceof GameMode mode
                    ? mode.getValue()
                    : -1,
                itemPayload(modify.itemToRemove),
                modify.adjustHeldItemQuantity,
                itemPayload(modify.itemToAdd),
                modify.brokenItem == null ? "" : modify.brokenItem,
                modify.adjustHeldItemDurability,
                notify instanceof Boolean,
                Boolean.TRUE.equals(notify),
                notifyMessage == null ? "" : notifyMessage.toString()
            );
        }
        return EmptyPayload.INSTANCE;
    }

    static String blockAssetId(
        int runtimeId,
        boolean allowAbsent
    ) {
        if (allowAbsent && runtimeId < 0) return "";
        BlockType block = BlockType.getAssetMap().getAsset(runtimeId);
        if (block == null) {
            throw new IllegalStateException(
                "Interaction references an unknown block runtime ID"
            );
        }
        return block.getId();
    }

    static String soundEventAssetId(int runtimeId) {
        if (runtimeId == 0) return "";
        SoundEvent sound = SoundEvent.getAssetMap().getAsset(runtimeId);
        if (sound == null) {
            throw new IllegalStateException(
                "Interaction references an unknown sound runtime ID"
            );
        }
        return sound.getId();
    }

    static ItemPayload itemPayload(
        com.hypixel.hytale.protocol.ItemWithAllMetadata item
    ) {
        if (item == null) return null;
        String metadata = item.metadata == null ? "" : item.metadata;
        if (metadata.length() > NativeItemInteractionEvidence.ITEM_METADATA_CAPACITY) {
            throw new IllegalStateException(
                "Native item metadata exceeds interaction evidence capacity"
            );
        }
        return new ItemPayload(
            item.itemId,
            item.quantity,
            item.durability,
            item.maxDurability,
            item.overrideDroppedItemAnimation,
            metadata
        );
    }

    static Object privateField(
        Interaction interaction,
        String name
    ) {
        try {
            var field = interaction.getClass().getDeclaredField(name);
            field.setAccessible(true);
            return field.get(interaction);
        } catch (ReflectiveOperationException error) {
            throw new IllegalStateException(
                "Pinned interaction payload field is unavailable: " + name,
                error
            );
        }
    }

    static String interactionId(int index) {
        Interaction interaction = Interaction.getAssetMap().getAsset(index);
        return interaction == null ? "" : interaction.getId();
    }

    static String tag(CollectorTag value) {
        if (value == CollectorTag.ROOT) return "ROOT";
        if (value instanceof StringTag string) return string.getTag();
        Class<?> type = value.getClass();
        return type.isRecord()
            ? value.toString()
            : type.getSimpleName();
    }
}
