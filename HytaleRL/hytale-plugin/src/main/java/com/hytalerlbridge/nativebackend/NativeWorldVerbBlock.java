package com.hytalerlbridge.nativebackend;

import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.activeItemId;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.resolvedWorldProbeRoot;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.BlockFace;
import com.hypixel.hytale.protocol.BlockPosition;
import com.hypixel.hytale.protocol.GameMode;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.protocol.Rotation;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.item.config.Item;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.entity.InteractionContext;
import com.hypixel.hytale.server.core.entity.entities.Player;
import com.hypixel.hytale.server.core.inventory.Inventory;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.Interaction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.action.NativeWorldVerbRequest;
import com.hytalerlbridge.nativebackend.model.WorldVerbCell;
import org.joml.Vector3i;

/** Stateless commit-time admission for exact native place/break roots. */
final class NativeWorldVerbBlock {

    private NativeWorldVerbBlock() {}

    static StartResult start(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        World world,
        NativeWorldVerbRequest request,
        boolean headlessWorldVerbsEnabled,
        Plan plan
    ) {
        boolean place = request.verb().equals("place_block");
        if (!headlessWorldVerbsEnabled) {
            return StartResult.rejected(
                "typed_" + request.verb()
                    + "_requires_headless_world_verb_context"
            );
        }
        if (plan.fixture() && !plan.target().equals(plan.expectedTarget())) {
            return StartResult.rejected(
                "target_outside_synthetic_fixture"
            );
        }
        if (
            request.interactionType() != plan.expectedInteractionType()
                || !request.interactionId().equals(
                    plan.expectedInteraction()
                )
                || !request.sourceContainer().equals("hotbar")
                || request.sourceSlot() != plan.expectedSlot()
                || !request.itemId().equals(plan.expectedItem())
                || (
                    plan.fixture()
                        && (
                            request.blockFace() != BlockFace.Up.getValue()
                                || request.rotationYaw()
                                    != Rotation.None.getValue()
                                || request.rotationPitch()
                                    != Rotation.None.getValue()
                                || request.rotationRoll()
                                    != Rotation.None.getValue()
                        )
                )
                || (
                    place
                        && (
                            (plan.fixture()
                                && !request.blockId().equals(
                                    plan.expectedPlaceBlockId()
                                ))
                                || !request.placementVariant().equals(
                                    "default"
                                )
                        )
                )
        ) {
            return StartResult.rejected(
                plan.fixture()
                    ? "typed_request_outside_synthetic_fixture_contract"
                    : "typed_block_request_contract_mismatch"
            );
        }

        Inventory inventory = npc.getInventory();
        ItemContainer hotbar = inventory == null
            ? null
            : inventory.getHotbar();
        if (
            hotbar == null
                || plan.expectedSlot()
                    >= Short.toUnsignedInt(hotbar.getCapacity())
        ) {
            return StartResult.rejected("typed_source_slot_unavailable");
        }
        ItemStack source = hotbar.getItemStack(
            (short) plan.expectedSlot()
        );
        if (
            ItemStack.isEmpty(source)
                || !plan.expectedItem().equals(source.getItemId())
                || request.expectedSourceQuantity() != source.getQuantity()
        ) {
            return StartResult.rejected("typed_source_item_changed");
        }
        inventory.setActiveHotbarSlot(
            ref,
            (byte) plan.expectedSlot(),
            store
        );
        Player player = store.getComponent(ref, Player.getComponentType());
        if (
            player == null
                || player.getGameMode() != GameMode.Adventure
                || !plan.expectedItem().equals(
                    activeItemId(player.getInventory())
                )
        ) {
            return StartResult.rejected(
                "headless_adventure_player_unavailable"
            );
        }
        InteractionType interactionType = InteractionType.fromValue(
            plan.expectedInteractionType()
        );
        Item sourceItem = source.getItem();
        String resolvedInteraction = sourceItem == null
            ? null
            : sourceItem.getInteractions().get(interactionType);
        if (
            resolvedInteraction == null
                || !resolvedInteraction.equals(plan.expectedInteraction())
        ) {
            return StartResult.rejected(
                "typed_item_interaction_identity_changed"
            );
        }
        int placedRuntimeBlockId = 0;
        if (place) {
            String sourceBlockId = source.getBlockKey();
            placedRuntimeBlockId = BlockType.getAssetMap().getIndex(
                request.blockId()
            );
            if (
                sourceBlockId == null
                    || !sourceBlockId.equals(request.blockId())
                    || placedRuntimeBlockId <= 0
            ) {
                return StartResult.rejected(
                    "typed_place_block_identity_changed"
                );
            }
        }

        WorldVerbCell before = WorldVerbCapture.capture(world, request);
        if (
            before == null
                || !before.semanticBlockId().equals(
                    plan.expectedBlockBefore()
                )
                || !request.expectedBlockId().equals(
                    plan.expectedBlockBefore()
                )
        ) {
            return StartResult.rejected(
                before == null
                    ? "typed_target_chunk_unavailable"
                    : "typed_block_identity_changed"
            );
        }

        RootInteraction root = resolvedWorldProbeRoot(
            plan.expectedInteraction()
        );
        InteractionChain chain = NativeInteractionQueue.queue(
            ref,
            store,
            interactionType,
            root,
            0.0
        );
        if (chain == null) {
            return StartResult.rejected(
                "native_interaction_chain_rejected"
            );
        }
        bindTarget(world, chain, plan.target());
        return StartResult.accepted(
            chain,
            source,
            before,
            placedRuntimeBlockId
        );
    }

    private static void bindTarget(
        World world,
        InteractionChain chain,
        Vector3i target
    ) {
        BlockPosition raw = new BlockPosition(target.x, target.y, target.z);
        BlockPosition base = world.getBaseBlock(raw);
        InteractionContext context = chain.getContext();
        context.getMetaStore().putMetaObject(Interaction.TARGET_BLOCK, base);
        context.getMetaStore().putMetaObject(
            Interaction.TARGET_BLOCK_RAW,
            raw
        );
    }

    record Plan(
        boolean fixture,
        Vector3i target,
        Vector3i expectedTarget,
        String expectedInteraction,
        String expectedItem,
        int expectedSlot,
        int expectedInteractionType,
        String expectedBlockBefore,
        String expectedPlaceBlockId
    ) {}

    record StartResult(
        boolean accepted,
        String rejectReason,
        InteractionChain chain,
        ItemStack source,
        WorldVerbCell before,
        int placedRuntimeBlockId
    ) {
        private static StartResult rejected(String reason) {
            return new StartResult(
                false,
                reason,
                null,
                null,
                null,
                0
            );
        }

        private static StartResult accepted(
            InteractionChain chain,
            ItemStack source,
            WorldVerbCell before,
            int placedRuntimeBlockId
        ) {
            return new StartResult(
                true,
                "",
                chain,
                source,
                before,
                placedRuntimeBlockId
            );
        }
    }
}
