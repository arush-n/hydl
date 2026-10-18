package com.hytalerlbridge.nativebackend;

import static com.hytalerlbridge.nativebackend.support.InteractionSupport.interactionManager;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.BlockPosition;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.inventory.Inventory;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.action.NativeWorldVerbRequest;
import com.hytalerlbridge.nativebackend.model.WorldVerbCell;

/**
 * Stateless commit-time admission for one exact native block-Use request.
 *
 * <p>The caller owns epoch validation, busy state, and lifecycle telemetry.
 * This class owns the engine-facing legality checks and queues the authored
 * chain. Keeping that boundary explicit lets native sessions and autonomous
 * policy actors share one implementation without sharing mutable state.</p>
 */
final class NativeWorldVerbUse {

    private NativeWorldVerbUse() {}

    static StartResult start(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        World world,
        NativeWorldVerbRequest request,
        boolean allowHeadlessPlayerContext
    ) {
        boolean unarmed = request.sourceContainer().equals("unarmed");
        boolean interactionContext = request.sourceContainer().equals(
            "interaction_context"
        );
        if (
            !unarmed
                && !interactionContext
                && !request.sourceContainer().equals("hotbar")
        ) {
            return StartResult.rejected(
                "typed_use_source_container_not_supported"
            );
        }

        Inventory inventory = npc.getInventory();
        ItemContainer hotbar = inventory == null
            ? null
            : inventory.getHotbar();
        if (!unarmed && !interactionContext) {
            if (
                hotbar == null
                    || request.sourceSlot() >= Short.toUnsignedInt(
                        hotbar.getCapacity()
                    )
            ) {
                return StartResult.rejected(
                    "typed_use_source_slot_unavailable"
                );
            }
            ItemStack stack = hotbar.getItemStack(
                (short) request.sourceSlot()
            );
            if (
                ItemStack.isEmpty(stack)
                    || !request.itemId().equals(stack.getItemId())
                    || request.expectedSourceQuantity()
                        != stack.getQuantity()
            ) {
                return StartResult.rejected(
                    "typed_use_source_item_changed"
                );
            }
            inventory.setActiveHotbarSlot(
                ref,
                (byte) request.sourceSlot(),
                store
            );
        }

        WorldVerbCell before = WorldVerbCapture.capture(world, request);
        if (
            before == null
                || !before.semanticBlockId().equals(
                    request.expectedBlockId()
                )
        ) {
            return StartResult.rejected(
                before == null
                    ? "typed_use_target_chunk_unavailable"
                    : "typed_block_identity_changed"
            );
        }
        NativeBlockUse.StartResult result = NativeBlockUse.startExact(
            ref,
            store,
            world,
            interactionManager(ref, store),
            request.interactionId(),
            request.itemId(),
            request.sourceContainer(),
            request.sourceSlot(),
            request.expectedSourceQuantity(),
            new BlockPosition(
                request.targetX(),
                request.targetY(),
                request.targetZ()
            ),
            request.expectedBlockId(),
            allowHeadlessPlayerContext
        );
        if (!result.accepted()) {
            return StartResult.rejected(result.rejectReason());
        }
        ItemStack resolvedSource = result.chain().getContext().getHeldItem();
        return StartResult.accepted(
            unarmed,
            result.chain(),
            before,
            resolvedSource
        );
    }

    record StartResult(
        boolean accepted,
        String rejectReason,
        boolean unarmed,
        InteractionChain chain,
        WorldVerbCell before,
        ItemStack resolvedSource
    ) {
        private static StartResult rejected(String reason) {
            return new StartResult(
                false,
                reason,
                false,
                null,
                null,
                null
            );
        }

        private static StartResult accepted(
            boolean unarmed,
            InteractionChain chain,
            WorldVerbCell before,
            ItemStack resolvedSource
        ) {
            return new StartResult(
                true,
                "",
                unarmed,
                chain,
                before,
                resolvedSource
            );
        }
    }
}
