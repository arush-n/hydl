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

/**
 * Runtime-resolved interaction roots and reachable leaves for one controlled
 * entity.
 *
 * <p>This is authored-chain evidence. It does not claim that an authenticated
 * player request passed target, permission, inventory, or lifecycle checks.</p>
 */
public record NativeItemInteractionEvidence(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    int activeGameMode,
    int equippedSlot,
    int equippedSlotCapacity,
    boolean equippedSlotAvailable,
    List<Trigger> triggers
) {

    public static final String SCHEMA =
        "hytalerl_native_item_interaction_evidence_v3";
    public static final int VERSION = 3;
    public static final int TRIGGER_CAPACITY = 25;
    public static final int INTERACTION_CAPACITY = 256;
    public static final int EDGE_CAPACITY = 512;
    public static final int CHARGE_TIME_CAPACITY = 16;
    public static final int BLOCK_CHANGE_CAPACITY = 256;
    public static final int ITEM_METADATA_CAPACITY = 16_384;

    public NativeItemInteractionEvidence {
        if (
            serverVersion == null
                || serverVersion.isBlank()
                || worldName == null
                || worldName.isBlank()
                || worldgenProvider == null
                || worldgenProvider.isBlank()
                || worldgenVersion == null
                || worldgenVersion.isBlank()
                || activeGameMode < 0
                || activeGameMode >= GameMode.VALUES.length
                || equippedSlot < 0
                || equippedSlotCapacity < 0
                || equippedSlotAvailable
                    != (equippedSlot < equippedSlotCapacity)
                || (
                    !equippedSlotAvailable
                        && (
                            equippedSlotCapacity != 0
                                || equippedSlot != 0
                        )
                )
        ) {
            throw new IllegalArgumentException(
                "Native interaction provenance is incomplete"
            );
        }
        triggers = List.copyOf(triggers);
        if (triggers.size() != TRIGGER_CAPACITY) {
            throw new IllegalArgumentException(
                "Native interaction evidence must contain every trigger"
            );
        }
        for (int index = 0; index < triggers.size(); index++) {
            Trigger trigger = triggers.get(index);
            if (
                trigger.interactionType() != index
                    || !trigger.interactionTypeName().equals(
                        InteractionType.VALUES[index].name()
                    )
            ) {
                throw new IllegalArgumentException(
                    "Native interaction triggers must be ordinal-sorted"
                );
            }
        }
    }

    public static NativeItemInteractionEvidence capture(
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed,
        Ref<EntityStore> entity,
        Store<EntityStore> store,
        InteractionManager manager,
        int equippedSlot
    ) {
        if (entity == null || !entity.isValid() || manager == null) {
            throw new IllegalStateException(
                "Native interaction capture requires a controlled entity"
            );
        }
        if (equippedSlot < 0) {
            throw new IllegalArgumentException(
                "Equipped slot must be nonnegative"
            );
        }
        InventoryComponent.Armor armor = store.getComponent(
            entity,
            InventoryComponent.Armor.getComponentType()
        );
        int equippedSlotCapacity = armor == null
            ? 0
            : Short.toUnsignedInt(armor.getInventory().getCapacity());
        if (
            equippedSlotCapacity > 0
                && equippedSlot >= equippedSlotCapacity
        ) {
            throw new IllegalArgumentException(
                "Equipped slot is outside the native armor inventory"
            );
        }
        Player player = store.getComponent(
            entity,
            Player.getComponentType()
        );
        GameMode activeMode = player == null
            ? GameMode.Adventure
            : player.getGameMode();
        List<Trigger> triggers = new ArrayList<>(TRIGGER_CAPACITY);
        for (InteractionType type : InteractionType.VALUES) {
            InteractionContext context = InteractionContext.forInteraction(
                manager,
                entity,
                type,
                type == InteractionType.Equipped ? equippedSlot : 0,
                store
            );
            /*
             * InteractionContext#getRootInteractionId first consults the
             * running entity's Interactions component. NPC roles commonly
             * populate that component, so using it here can pair a held item
             * ID with an NPC role root. This evidence is explicitly about
             * the item branch: forInteraction has already selected the
             * native main/offhand/tool/armor slot, and root resolution must
             * now stay on that selected item (or the authored unarmed
             * fallback).
             */
            String rootId = NativeItemInteractionEvidenceBuilders.selectedItemRootInteractionId(
                context.getOriginalItemType(),
                type
            );
            RootInteraction root = rootId == null
                ? null
                : RootInteraction.getAssetMap().getAsset(rootId);
            Item item = context.getOriginalItemType();
            if (rootId != null && root == null) {
                throw new IllegalStateException(
                    "Resolved interaction root is absent: " + rootId
                );
            }
            triggers.add(NativeItemInteractionEvidenceBuilders.trigger(type, context, item, root));
        }
        return new NativeItemInteractionEvidence(
            serverVersion,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            activeMode.getValue(),
            equippedSlot,
            equippedSlotCapacity,
            equippedSlot < equippedSlotCapacity,
            triggers
        );
    }

    static String itemRootInteractionId(
        Map<InteractionType, String> interactions,
        InteractionType type
    ) {
        return interactions.get(type);
    }

    public record Trigger(
        int interactionType,
        String interactionTypeName,
        String itemAssetId,
        int heldItemSectionId,
        int heldItemSlot,
        boolean rootPresent,
        Root root,
        List<Node> nodes,
        List<Edge> edges
    ) {

        public Trigger {
            if (
                interactionType < 0
                    || interactionType >= TRIGGER_CAPACITY
                    || interactionTypeName == null
                    || interactionTypeName.isBlank()
                    || itemAssetId == null
                    || rootPresent != (root != null)
            ) {
                throw new IllegalArgumentException(
                    "Native interaction trigger is invalid"
                );
            }
            nodes = List.copyOf(nodes);
            edges = List.copyOf(edges);
            if (
                nodes.size() > INTERACTION_CAPACITY
                    || edges.size() > EDGE_CAPACITY
                    || (!rootPresent
                        && (!nodes.isEmpty() || !edges.isEmpty()))
            ) {
                throw new IllegalArgumentException(
                    "Native interaction graph exceeds its capacity"
                );
            }
            Set<String> nodeIds = new HashSet<>();
            String previousNode = "";
            for (Node node : nodes) {
                if (
                    !nodeIds.add(node.interactionId())
                        || node.interactionId().compareTo(previousNode) < 0
                ) {
                    throw new IllegalArgumentException(
                        "Native interaction nodes must be unique and sorted"
                    );
                }
                previousNode = node.interactionId();
            }
            String previousEdge = "";
            for (Edge edge : edges) {
                String key = edge.parentInteractionId()
                    + "\u0000"
                    + edge.childInteractionId()
                    + "\u0000"
                    + edge.relation();
                if (
                    key.compareTo(previousEdge) < 0
                        || !nodeIds.contains(edge.childInteractionId())
                        || (
                            !edge.parentInteractionId().isEmpty()
                                && !nodeIds.contains(
                                    edge.parentInteractionId()
                                )
                        )
                ) {
                    throw new IllegalArgumentException(
                        "Native interaction edges are incomplete or unordered"
                    );
                }
                previousEdge = key;
            }
            for (Node node : nodes) {
                if (
                    (
                        !node.nextInteractionId().isEmpty()
                            && !nodeIds.contains(
                                node.nextInteractionId()
                            )
                    )
                        || (
                            !node.failedInteractionId().isEmpty()
                                && !nodeIds.contains(
                                    node.failedInteractionId()
                                )
                        )
                ) {
                    throw new IllegalArgumentException(
                        "Native interaction node references are incomplete"
                    );
                }
            }
            if (
                root != null
                    && !nodeIds.containsAll(root.initialInteractionIds())
            ) {
                throw new IllegalArgumentException(
                    "Native interaction root references are incomplete"
                );
            }
        }
    }

    public record Root(
        String rootId,
        float clickQueuingTimeout,
        boolean requireNewClick,
        Rules rules,
        List<ModeSettings> modeSettings,
        List<String> initialInteractionIds
    ) {

        public Root {
            modeSettings = List.copyOf(modeSettings);
            initialInteractionIds = List.copyOf(initialInteractionIds);
            if (
                rootId == null
                    || rootId.isBlank()
                    || !Float.isFinite(clickQueuingTimeout)
                    || clickQueuingTimeout < 0.0f
                    || rules == null
                    || modeSettings.size() != GameMode.VALUES.length
                    || initialInteractionIds.isEmpty()
                    || initialInteractionIds.size() > INTERACTION_CAPACITY
            ) {
                throw new IllegalArgumentException(
                    "Native interaction root is invalid"
                );
            }
            for (int index = 0; index < modeSettings.size(); index++) {
                if (modeSettings.get(index).gameMode() != index) {
                    throw new IllegalArgumentException(
                        "Native interaction mode settings must be sorted"
                    );
                }
            }
        }
    }

    public record ModeSettings(
        int gameMode,
        boolean allowSkipChainOnClick,
        String cooldownId,
        float cooldownSeconds,
        boolean clickBypass,
        boolean skipCooldownReset,
        boolean interruptRecharge,
        List<Float> chargeTimes
    ) {

        public ModeSettings {
            chargeTimes = List.copyOf(chargeTimes);
            if (
                gameMode < 0
                    || gameMode >= GameMode.VALUES.length
                    || cooldownId == null
                    || cooldownId.isBlank()
                    || !Float.isFinite(cooldownSeconds)
                    || cooldownSeconds < 0.0f
                    || chargeTimes.isEmpty()
                    || chargeTimes.size() > CHARGE_TIME_CAPACITY
            ) {
                throw new IllegalArgumentException(
                    "Native interaction mode settings are invalid"
                );
            }
            for (float value : chargeTimes) {
                if (!Float.isFinite(value) || value < 0.0f) {
                    throw new IllegalArgumentException(
                        "Native interaction charge time is invalid"
                    );
                }
            }
        }
    }

    public record Rules(
        int blockedByMask,
        boolean blockedByUsesDefault,
        int blockingMask,
        int interruptedByMask,
        boolean interruptedByUnspecified,
        int interruptingMask,
        boolean interruptingUnspecified,
        int blockedByBypassIndex,
        int blockingBypassIndex,
        int interruptedByBypassIndex,
        int interruptingBypassIndex
    ) {}

    public record Node(
        String interactionId,
        String implementationClass,
        float runTime,
        boolean cancelOnItemChange,
        int waitForDataFrom,
        String nextInteractionId,
        String failedInteractionId,
        boolean useLatestTarget,
        boolean harvest,
        float horizontalSpeedMultiplier,
        boolean effectsPresent,
        float startDelay,
        boolean waitForAnimationToFinish,
        boolean movementEffectsPresent,
        boolean movementDisableAll,
        int movementLockMask,
        Payload payload
    ) {

        public Node(
            String interactionId,
            String implementationClass,
            float runTime,
            boolean cancelOnItemChange,
            int waitForDataFrom,
            String nextInteractionId,
            String failedInteractionId,
            boolean useLatestTarget,
            boolean harvest
        ) {
            this(
                interactionId,
                implementationClass,
                runTime,
                cancelOnItemChange,
                waitForDataFrom,
                nextInteractionId,
                failedInteractionId,
                useLatestTarget,
                harvest,
                1.0f,
                false,
                0.0f,
                false,
                false,
                false,
                0,
                EmptyPayload.INSTANCE
            );
        }

        public Node {
            if (
                interactionId == null
                    || interactionId.isBlank()
                    || implementationClass == null
                    || implementationClass.isBlank()
                    || !Float.isFinite(runTime)
                    || runTime < 0.0f
                    || waitForDataFrom < 0
                    || nextInteractionId == null
                    || failedInteractionId == null
                    || !Float.isFinite(horizontalSpeedMultiplier)
                    || !Float.isFinite(startDelay)
                    || startDelay < 0.0f
                    || movementLockMask < 0
                    || movementLockMask > 0x7f
                    || (!movementEffectsPresent && movementLockMask != 0)
                    || (
                        movementDisableAll
                            && (
                                !movementEffectsPresent
                                    || movementLockMask != 0x7f
                            )
                    )
                    || payload == null
            ) {
                throw new IllegalArgumentException(
                    "Native interaction node is invalid"
                );
            }
        }
    }

    public sealed interface Payload permits
        EmptyPayload,
        BreakBlockPayload,
        PlaceBlockPayload,
        ChangeBlockPayload,
        ModifyInventoryPayload {
        String kind();
    }

    public enum EmptyPayload implements Payload {
        INSTANCE;

        @Override
        public String kind() {
            return "";
        }
    }

    /**
     * Server-only BreakBlock configuration omitted from the client packet.
     *
     * <p>{@code toolId} is the authored tool category used by
     * {@code BlockHarvestUtils.performBlockDamage}; it is not an item asset
     * ID. An empty value means the interaction did not override the tool.
     * {@code matchTool} makes a non-empty category an exact legality
     * requirement.</p>
     */
    public record BreakBlockPayload(
        String toolId,
        boolean matchTool
    ) implements Payload {
        @Override
        public String kind() {
            return "break_block";
        }

        public BreakBlockPayload {
            if (
                toolId == null
                    || (matchTool && toolId.isBlank())
            ) {
                throw new IllegalArgumentException(
                    "Break-block payload has invalid tool semantics"
                );
            }
        }
    }

    public record PlaceBlockPayload(
        String blockAssetId,
        boolean removeItemInHand,
        boolean allowDragPlacement
    ) implements Payload {
        @Override
        public String kind() {
            return "place_block";
        }

        public PlaceBlockPayload {
            if (blockAssetId == null) {
                throw new IllegalArgumentException(
                    "Place payload block identity is null"
                );
            }
        }
    }

    public record BlockChange(
        String fromBlockAssetId,
        String toBlockAssetId
    ) {
        public BlockChange {
            if (
                fromBlockAssetId == null
                    || fromBlockAssetId.isBlank()
                    || toBlockAssetId == null
                    || toBlockAssetId.isBlank()
            ) {
                throw new IllegalArgumentException(
                    "Block-change payload lacks stable identities"
                );
            }
        }
    }

    public record ChangeBlockPayload(
        List<BlockChange> changes,
        String worldSoundEventAssetId,
        boolean requireNotBroken
    ) implements Payload {
        @Override
        public String kind() {
            return "change_block";
        }

        public ChangeBlockPayload {
            changes = List.copyOf(changes);
            if (
                changes.size() > BLOCK_CHANGE_CAPACITY
                    || worldSoundEventAssetId == null
            ) {
                throw new IllegalArgumentException(
                    "Change-block payload exceeds its contract"
                );
            }
        }
    }

    public record ItemPayload(
        String itemAssetId,
        int quantity,
        double durability,
        double maxDurability,
        boolean overrideDroppedItemAnimation,
        String metadataJson
    ) {
        public ItemPayload {
            if (
                itemAssetId == null
                    || itemAssetId.isBlank()
                    || quantity <= 0
                    || !Double.isFinite(durability)
                    || !Double.isFinite(maxDurability)
                    || metadataJson == null
                    || metadataJson.length() > ITEM_METADATA_CAPACITY
            ) {
                throw new IllegalArgumentException(
                    "Modify-inventory item payload is invalid"
                );
            }
        }
    }

    public record ModifyInventoryPayload(
        int requiredGameMode,
        ItemPayload itemToRemove,
        int adjustHeldItemQuantity,
        ItemPayload itemToAdd,
        String brokenItemAssetId,
        double adjustHeldItemDurability,
        boolean notifyOnBreakSpecified,
        boolean notifyOnBreak,
        String notifyOnBreakMessage
    ) implements Payload {
        @Override
        public String kind() {
            return "modify_inventory";
        }

        public ModifyInventoryPayload {
            if (
                requiredGameMode < -1
                    || requiredGameMode >= GameMode.VALUES.length
                    || brokenItemAssetId == null
                    || !Double.isFinite(adjustHeldItemDurability)
                    || notifyOnBreakMessage == null
            ) {
                throw new IllegalArgumentException(
                    "Modify-inventory payload is invalid"
                );
            }
        }
    }

    public record Edge(
        String parentInteractionId,
        String childInteractionId,
        String relation
    ) {

        public Edge {
            if (
                parentInteractionId == null
                    || childInteractionId == null
                    || childInteractionId.isBlank()
                    || relation == null
                    || relation.isBlank()
            ) {
                throw new IllegalArgumentException(
                    "Native interaction edge is invalid"
                );
            }
        }
    }

}
