package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.vector.Transform;
import com.hypixel.hytale.protocol.BlockPosition;
import com.hypixel.hytale.protocol.GameMode;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.item.config.Item;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.entity.InteractionContext;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.entity.entities.Player;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.modules.entity.component.ModelComponent;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.interaction.interaction.UnarmedInteractions;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.Interaction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.InteractionConfiguration;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.core.util.TargetUtil;
import java.util.Map;
import org.joml.Vector3d;
import org.joml.Vector3i;

/**
 * Starts the authored Use chain against the first block in the post-action
 * camera ray.
 *
 * <p>This is deliberately narrower than a generic client click. Hytale's
 * headless interaction simulation supplies block sync data, but
 * UseEntityInteraction still expects a client-selected entity id. Entity use
 * therefore stays unavailable rather than guessing a target.
 */
public final class NativeBlockUse {
    // InteractionValidation.getPlayerInteractionDistanceSq adds this buffer
    // after reading the held item's authored Adventure use distance.
    private static final float INTERACTION_DISTANCE_BUFFER = 2.0f;

    private NativeBlockUse() {}

    static StartResult start(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        World world,
        InteractionManager manager,
        float desiredPitch,
        float desiredYaw
    ) {
        return start(
            ref,
            store,
            world,
            manager,
            desiredPitch,
            desiredYaw,
            false
        );
    }

    static StartResult start(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        World world,
        InteractionManager manager,
        float desiredPitch,
        float desiredYaw,
        boolean allowHeadlessPlayerContext
    ) {
        PreparedUse prepared = prepare(
            ref,
            store,
            world,
            manager,
            desiredPitch,
            desiredYaw,
            allowHeadlessPlayerContext
        );
        return startPrepared(manager, prepared);
    }

    /**
     * Starts a target-relative Use request after revalidating every stable
     * identity carried by the typed bridge transport.
     */
    static StartResult startExact(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        World world,
        InteractionManager manager,
        String expectedInteractionId,
        String expectedItemId,
        String expectedSourceContainer,
        int expectedSourceSlot,
        int expectedSourceQuantity,
        BlockPosition target,
        String expectedBlockId
    ) {
        return startExact(
            ref,
            store,
            world,
            manager,
            expectedInteractionId,
            expectedItemId,
            expectedSourceContainer,
            expectedSourceSlot,
            expectedSourceQuantity,
            target,
            expectedBlockId,
            false
        );
    }

    static StartResult startExact(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        World world,
        InteractionManager manager,
        String expectedInteractionId,
        String expectedItemId,
        String expectedSourceContainer,
        int expectedSourceSlot,
        int expectedSourceQuantity,
        BlockPosition target,
        String expectedBlockId,
        boolean allowHeadlessPlayerContext
    ) {
        return startPrepared(
            manager,
            prepareExact(
                ref,
                store,
                world,
                manager,
                expectedInteractionId,
                expectedItemId,
                expectedSourceContainer,
                expectedSourceSlot,
                expectedSourceQuantity,
                target,
                expectedBlockId,
                allowHeadlessPlayerContext
            )
        );
    }

    private static StartResult startPrepared(
        InteractionManager manager,
        PreparedUse prepared
    ) {
        if (!prepared.available()) {
            return StartResult.reject(prepared.rejectReason());
        }

        InteractionChain chain = manager.initChain(
            prepared.type(),
            prepared.context(),
            prepared.root(),
            -1,
            prepared.rawPosition(),
            false
        );
        if (
            !manager.applyRules(
                prepared.context(),
                chain.getChainData(),
                prepared.type(),
                prepared.root()
            )
        ) {
            return StartResult.reject("interaction_rules_rejected");
        }
        manager.queueExecuteChain(chain);
        return StartResult.accept(
            chain,
            prepared.interactionId(),
            prepared.blockInteractionId(),
            prepared.basePosition(),
            prepared.maximumDistance()
        );
    }

    /**
     * Return the actor-legal Use edge for the current held item and camera.
     *
     * <p>The policy mask must not confuse bridge implementation support with
     * authored root/target availability. This probe executes the exact same
     * preflight as {@link #start} but never initializes or queues a chain.
     * {@code start} repeats the preflight at commit time, so a world change
     * between observation and action still fails closed.</p>
     */
    public static Availability inspect(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        World world,
        InteractionManager manager,
        float desiredPitch,
        float desiredYaw
    ) {
        return inspect(
            ref,
            store,
            world,
            manager,
            desiredPitch,
            desiredYaw,
            false
        );
    }

    public static Availability inspect(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        World world,
        InteractionManager manager,
        float desiredPitch,
        float desiredYaw,
        boolean allowHeadlessPlayerContext
    ) {
        PreparedUse prepared = prepare(
            ref,
            store,
            world,
            manager,
            desiredPitch,
            desiredYaw,
            allowHeadlessPlayerContext
        );
        if (!prepared.available()) {
            return Availability.unavailable(prepared.rejectReason());
        }
        InteractionContext context = prepared.context();
        ItemStack source = context == null ? null : context.getHeldItem();
        boolean unarmed = ItemStack.isEmpty(source);
        return new Availability(
            prepared.available(),
            prepared.interactionId(),
            prepared.blockInteractionId(),
            unarmed ? "Empty" : source.getItemId(),
            unarmed ? "unarmed" : "interaction_context",
            unarmed
                ? -1
                : Byte.toUnsignedInt(context.getHeldItemSlot()),
            unarmed ? 0 : source.getQuantity(),
            prepared.actorPosition(),
            prepared.eyePosition(),
            prepared.direction(),
            prepared.rawPosition(),
            prepared.basePosition(),
            prepared.maximumDistance(),
            prepared.rejectReason()
        );
    }

    /**
     * Capture the exact Adventure camera ray used by block interactions.
     *
     * <p>This deliberately does not require an authored Use root. Break and
     * Use share the held item's Adventure interaction distance, while only
     * Use requires a target-block Use root. Keeping the ray in one helper
     * prevents the policy surface and commit preflight from drifting.</p>
     */
    public static CameraTarget inspectCamera(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        World world,
        Item heldItem,
        float desiredPitch,
        float desiredYaw
    ) {
        if (ref == null || !ref.isValid()) {
            return CameraTarget.unavailable("controlled_entity_unavailable");
        }
        if (store == null || world == null) {
            return CameraTarget.unavailable("controlled_world_unavailable");
        }
        TransformComponent transform = store.getComponent(
            ref,
            TransformComponent.getComponentType()
        );
        if (transform == null) {
            return CameraTarget.unavailable("controlled_transform_unavailable");
        }
        double maxDistance = adventureUseDistance(heldItem);
        Vector3d position = transform.getPosition();
        ModelComponent model = store.getComponent(
            ref,
            ModelComponent.getComponentType()
        );
        double eyeHeight = model == null
            ? 0.0
            : model.getModel().getEyeHeight(ref, store);
        Vector3d direction = Transform.getDirection(desiredPitch, desiredYaw);
        Vector3d eyePosition = new Vector3d(
            position.x,
            position.y + eyeHeight,
            position.z
        );
        Vector3i rawTarget = TargetUtil.getTargetBlock(
            world,
            (blockId, _fluidId) -> blockId != 0,
            eyePosition.x,
            eyePosition.y,
            eyePosition.z,
            direction.x,
            direction.y,
            direction.z,
            maxDistance
        );
        if (rawTarget == null) {
            return CameraTarget.unavailable("no_block_in_authored_use_range");
        }
        BlockPosition rawPosition = new BlockPosition(
            rawTarget.x,
            rawTarget.y,
            rawTarget.z
        );
        BlockPosition basePosition = world.getBaseBlock(rawPosition);
        return CameraTarget.available(
            position,
            eyePosition,
            direction,
            rawPosition,
            basePosition,
            maxDistance
        );
    }

    private static PreparedUse prepare(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        World world,
        InteractionManager manager,
        float desiredPitch,
        float desiredYaw,
        boolean allowHeadlessPlayerContext
    ) {
        if (ref == null || !ref.isValid()) {
            return PreparedUse.reject("controlled_entity_unavailable");
        }
        if (manager == null) {
            return PreparedUse.reject("interaction_manager_unavailable");
        }
        if (
            !allowHeadlessPlayerContext
                && store.getComponent(ref, Player.getComponentType()) != null
        ) {
            return PreparedUse.reject("remote_client_context_not_supported");
        }

        InteractionType type = InteractionType.Use;
        InteractionContext context = InteractionContext.forInteraction(
            manager,
            ref,
            type,
            store
        );
        // InteractionContext gives an entity-bound interaction component
        // priority over the held item. NPC roles bind Use to *UseNPC, which
        // is not the player verb being translated here. Resolve the exact
        // held-item/unarmed player branch explicitly instead.
        String interactionId = playerUseInteractionId(
            context.getOriginalItemType()
        );
        RootInteraction root = interactionId == null
            ? null
            : RootInteraction.getAssetMap().getAsset(interactionId);
        if (root == null) {
            return PreparedUse.reject(
                "authored_use_interaction_unavailable"
            );
        }

        CameraTarget camera = inspectCamera(
            ref,
            store,
            world,
            context.getOriginalItemType(),
            desiredPitch,
            desiredYaw
        );
        if (!camera.available()) {
            return PreparedUse.reject(camera.rejectReason());
        }
        BlockPosition rawPosition = camera.rawTarget();
        BlockPosition basePosition = camera.target();
        BlockType blockType = world.getBlockType(
            new Vector3i(basePosition.x, basePosition.y, basePosition.z)
        );
        String blockInteractionId = blockType == null
            ? null
            : blockType.getInteractions().get(type);
        if (
            blockInteractionId == null
                || RootInteraction.getAssetMap().getAsset(blockInteractionId) == null
        ) {
            return PreparedUse.reject("target_block_has_no_authored_use");
        }

        context.getMetaStore().putMetaObject(
            Interaction.TARGET_BLOCK,
            basePosition
        );
        context.getMetaStore().putMetaObject(
            Interaction.TARGET_BLOCK_RAW,
            rawPosition
        );
        return PreparedUse.accept(
            type,
            context,
            root,
            interactionId,
            blockInteractionId,
            camera.actorPosition(),
            camera.eyePosition(),
            camera.direction(),
            rawPosition,
            basePosition,
            camera.maximumDistance()
        );
    }

    private static PreparedUse prepareExact(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        World world,
        InteractionManager manager,
        String expectedInteractionId,
        String expectedItemId,
        String expectedSourceContainer,
        int expectedSourceSlot,
        int expectedSourceQuantity,
        BlockPosition target,
        String expectedBlockId,
        boolean allowHeadlessPlayerContext
    ) {
        if (ref == null || !ref.isValid()) {
            return PreparedUse.reject("controlled_entity_unavailable");
        }
        if (manager == null) {
            return PreparedUse.reject("interaction_manager_unavailable");
        }
        if (
            !allowHeadlessPlayerContext
                && store.getComponent(ref, Player.getComponentType()) != null
        ) {
            return PreparedUse.reject("remote_client_context_not_supported");
        }
        if (target == null || target.y < 0 || target.y >= 320) {
            return PreparedUse.reject("typed_block_target_out_of_domain");
        }

        TransformComponent transform = store.getComponent(
            ref,
            TransformComponent.getComponentType()
        );
        if (transform == null) {
            return PreparedUse.reject("controlled_transform_unavailable");
        }
        InteractionType type = InteractionType.Use;
        InteractionContext context = InteractionContext.forInteraction(
            manager,
            ref,
            type,
            store
        );
        Item heldItem = context.getOriginalItemType();
        ItemStack heldStack = context.getHeldItem();
        boolean unarmed = expectedSourceContainer.equals("unarmed");
        boolean inventoryBound = acceptsExactInventorySource(
            expectedSourceContainer,
            allowHeadlessPlayerContext
        );
        if (
            !expectedItemMatches(heldItem, expectedItemId)
                || (
                    unarmed
                        && (
                            !ItemStack.isEmpty(heldStack)
                                || expectedSourceSlot != -1
                                || expectedSourceQuantity != 0
                        )
                )
                || (
                    !unarmed
                        && (
                            !inventoryBound
                                || ItemStack.isEmpty(heldStack)
                                || Byte.toUnsignedInt(
                                    context.getHeldItemSlot()
                                ) != expectedSourceSlot
                                || heldStack.getQuantity()
                                    != expectedSourceQuantity
                        )
                )
        ) {
            return PreparedUse.reject("typed_held_item_identity_changed");
        }
        String interactionId = playerUseInteractionId(heldItem);
        if (
            interactionId == null
                || !interactionId.equals(expectedInteractionId)
        ) {
            return PreparedUse.reject(
                "typed_item_interaction_identity_changed"
            );
        }
        RootInteraction root = RootInteraction.getAssetMap().getAsset(
            interactionId
        );
        if (root == null) {
            return PreparedUse.reject(
                "authored_use_interaction_unavailable"
            );
        }

        BlockPosition basePosition = world.getBaseBlock(target);
        if (
            basePosition.x != target.x
                || basePosition.y != target.y
                || basePosition.z != target.z
        ) {
            return PreparedUse.reject(
                "typed_non_root_block_target_not_supported"
            );
        }
        BlockType blockType = world.getBlockType(
            new Vector3i(target.x, target.y, target.z)
        );
        if (
            blockType == null
                || !blockType.getId().equals(expectedBlockId)
        ) {
            return PreparedUse.reject("typed_block_identity_changed");
        }
        String blockInteractionId = blockType
            .getInteractions()
            .get(type);
        if (
            blockInteractionId == null
                || RootInteraction.getAssetMap().getAsset(
                    blockInteractionId
                ) == null
        ) {
            return PreparedUse.reject("target_block_has_no_authored_use");
        }

        double maxDistance = adventureUseDistance(heldItem);
        ModelComponent model = store.getComponent(
            ref,
            ModelComponent.getComponentType()
        );
        double eyeHeight = model == null
            ? 0.0
            : model.getModel().getEyeHeight(ref, store);
        Vector3d position = transform.getPosition();
        double dx = target.x + 0.5 - position.x;
        double dy = target.y + 0.5 - (position.y + eyeHeight);
        double dz = target.z + 0.5 - position.z;
        if (dx * dx + dy * dy + dz * dz > maxDistance * maxDistance) {
            return PreparedUse.reject("typed_block_target_out_of_range");
        }
        Vector3d eyePosition = new Vector3d(
            position.x,
            position.y + eyeHeight,
            position.z
        );
        Vector3d direction = new Vector3d(dx, dy, dz).normalize();

        context.getMetaStore().putMetaObject(
            Interaction.TARGET_BLOCK,
            basePosition
        );
        context.getMetaStore().putMetaObject(
            Interaction.TARGET_BLOCK_RAW,
            target
        );
        return PreparedUse.accept(
            type,
            context,
            root,
            interactionId,
            blockInteractionId,
            position,
            eyePosition,
            direction,
            target,
            basePosition,
            maxDistance
        );
    }

    static boolean acceptsExactInventorySource(
        String sourceContainer,
        boolean allowHeadlessPlayerContext
    ) {
        return "interaction_context".equals(sourceContainer)
            || (
                allowHeadlessPlayerContext
                    && "hotbar".equals(sourceContainer)
            );
    }

    static boolean expectedItemMatches(
        Item heldItem,
        String expectedItemId
    ) {
        return heldItem == null
            ? "Empty".equals(expectedItemId)
            : heldItem.getId().equals(expectedItemId);
    }

    public static double adventureUseDistance(Item heldItem) {
        InteractionConfiguration configuration = heldItem == null
            ? InteractionConfiguration.DEFAULT
            : heldItem.getInteractionConfig();
        return configuration.getUseDistance(GameMode.Adventure)
            + INTERACTION_DISTANCE_BUFFER;
    }

    private static String playerUseInteractionId(Item heldItem) {
        if (heldItem != null) {
            return itemUseInteractionId(heldItem.getInteractions());
        }
        UnarmedInteractions unarmed =
            UnarmedInteractions.getAssetMap().getAsset("Empty");
        return unarmed == null
            ? null
            : itemUseInteractionId(unarmed.getInteractions());
    }

    static String itemUseInteractionId(
        Map<InteractionType, String> interactions
    ) {
        return interactions.get(InteractionType.Use);
    }

    record StartResult(
        InteractionChain chain,
        String interactionId,
        String blockInteractionId,
        BlockPosition target,
        double maximumDistance,
        String rejectReason
    ) {
        static StartResult accept(
            InteractionChain chain,
            String interactionId,
            String blockInteractionId,
            BlockPosition target,
            double maximumDistance
        ) {
            return new StartResult(
                chain,
                interactionId,
                blockInteractionId,
                target,
                maximumDistance,
                ""
            );
        }

        static StartResult reject(String reason) {
            return new StartResult(null, "", "", null, 0.0, reason);
        }

        boolean accepted() {
            return chain != null;
        }
    }

    public record Availability(
        boolean available,
        String interactionId,
        String blockInteractionId,
        String itemId,
        String sourceContainer,
        int sourceSlot,
        int sourceQuantity,
        Vector3d actorPosition,
        Vector3d eyePosition,
        Vector3d direction,
        BlockPosition rawTarget,
        BlockPosition target,
        double maximumDistance,
        String rejectReason
    ) {
        static Availability unavailable(String reason) {
            return new Availability(
                false,
                "",
                "",
                "",
                "",
                -1,
                -1,
                null,
                null,
                null,
                null,
                null,
                0.0,
                reason
            );
        }
    }

    /** Authoritative block ray independent of a target's Use affordance. */
    public record CameraTarget(
        boolean available,
        Vector3d actorPosition,
        Vector3d eyePosition,
        Vector3d direction,
        BlockPosition rawTarget,
        BlockPosition target,
        double maximumDistance,
        String rejectReason
    ) {
        static CameraTarget available(
            Vector3d actorPosition,
            Vector3d eyePosition,
            Vector3d direction,
            BlockPosition rawTarget,
            BlockPosition target,
            double maximumDistance
        ) {
            return new CameraTarget(
                true,
                new Vector3d(actorPosition),
                new Vector3d(eyePosition),
                new Vector3d(direction),
                rawTarget,
                target,
                maximumDistance,
                ""
            );
        }

        static CameraTarget unavailable(String reason) {
            return new CameraTarget(
                false,
                null,
                null,
                null,
                null,
                null,
                0.0,
                reason
            );
        }
    }

    private record PreparedUse(
        InteractionType type,
        InteractionContext context,
        RootInteraction root,
        String interactionId,
        String blockInteractionId,
        Vector3d actorPosition,
        Vector3d eyePosition,
        Vector3d direction,
        BlockPosition rawPosition,
        BlockPosition basePosition,
        double maximumDistance,
        String rejectReason
    ) {
        static PreparedUse accept(
            InteractionType type,
            InteractionContext context,
            RootInteraction root,
            String interactionId,
            String blockInteractionId,
            Vector3d actorPosition,
            Vector3d eyePosition,
            Vector3d direction,
            BlockPosition rawPosition,
            BlockPosition basePosition,
            double maximumDistance
        ) {
            return new PreparedUse(
                type,
                context,
                root,
                interactionId,
                blockInteractionId,
                new Vector3d(actorPosition),
                new Vector3d(eyePosition),
                new Vector3d(direction),
                rawPosition,
                basePosition,
                maximumDistance,
                ""
            );
        }

        static PreparedUse reject(String reason) {
            return new PreparedUse(
                null,
                null,
                null,
                "",
                "",
                null,
                null,
                null,
                null,
                null,
                0.0,
                reason
            );
        }

        boolean available() {
            return root != null;
        }
    }
}
