package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.builtin.crafting.component.BenchBlock;
import com.hypixel.hytale.builtin.crafting.component.CraftingManager;
import com.hypixel.hytale.builtin.crafting.window.StructuralCraftingWindow;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.item.config.CraftingRecipe;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.MaterialQuantity;
import com.hypixel.hytale.server.core.inventory.container.CombinedItemContainer;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.inventory.transaction.ItemStackTransaction;
import com.hypixel.hytale.server.core.inventory.transaction.ListTransaction;
import com.hypixel.hytale.server.core.inventory.transaction.MaterialSlotTransaction;
import com.hypixel.hytale.server.core.inventory.transaction.MaterialTransaction;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.action.NativeWorldVerbRequest;
import java.util.ArrayList;
import java.util.List;

/**
 * Structural (Builders) bench craft, in the order the native window runs it.
 *
 * <p>0.7.2 evidence: {@code StructuralCraftingWindow.handleAction} for a
 * {@code CraftRecipeAction} calls {@code CraftingManager.queueCraft(ref, store,
 * this, 0, recipe, quantity, inputContainer, InputRemovalType.ORDERED)}. It never
 * calls {@code craftItem}, and the window's {@code inputContainer} is a one-slot
 * container whose filter accepts only items some structural recipe takes. The
 * queued job removes its inputs from that slot at its tick with
 * {@code removeMaterialsOrdered}, then gives the output.</p>
 *
 * <p>The player moves the inputs from the inventory into that slot before the
 * craft; the headless path performs that move as a transport step at the
 * request, then queues the same call. The transport is an inference from the
 * window's option mechanism, not a native event observed in this build.</p>
 */
final class NativeStructuralCrafting {

    private NativeStructuralCrafting() {}

    static NativeHeadlessCrafting.StartResult start(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        World world,
        NativeWorldVerbRequest request,
        CraftingRecipe recipe,
        CombinedItemContainer actorInventory,
        CraftingManager manager,
        BlockType blockType,
        BenchBlock benchBlock,
        int rotationIndex
    ) {
        List<MaterialQuantity> materials = CraftingManager.getInputMaterials(
            recipe,
            request.quantity()
        );
        if (!actorInventory.canRemoveMaterials(materials)) {
            return NativeHeadlessCrafting.StartResult.rejected(
                "structural_input_missing_from_inventory"
            );
        }
        String inventoryBefore = NativeHeadlessCrafting.canonicalInventory(
            reference,
            store
        );
        int outputBefore = NativeHeadlessCrafting.outputQuantity(
            reference,
            store,
            recipe
        );
        StructuralCraftingWindow window = new StructuralCraftingWindow(
            request.benchX(),
            request.benchY(),
            request.benchZ(),
            rotationIndex,
            blockType,
            benchBlock
        );
        ItemContainer benchInput = ((CombinedItemContainer) window
            .getItemContainer()).getContainer(0);
        manager.setBench(
            request.benchX(),
            request.benchY(),
            request.benchZ(),
            blockType
        );

        List<ItemStack> moved = new ArrayList<>();
        collectRemoved(
            actorInventory.removeMaterials(materials, true, true, true),
            moved
        );
        for (ItemStack stack : moved) {
            ItemStackTransaction placed = benchInput.addItemStack(stack);
            if (!placed.succeeded()) {
                rollback(
                    reference,
                    store,
                    manager,
                    actorInventory,
                    benchInput,
                    moved
                );
                return NativeHeadlessCrafting.StartResult.rejected(
                    "structural_input_rejected_by_bench"
                );
            }
        }

        boolean queued = manager.queueCraft(
            reference,
            store,
            window,
            0,
            recipe,
            request.quantity(),
            benchInput,
            CraftingManager.InputRemovalType.ORDERED
        );
        if (!queued) {
            rollback(
                reference,
                store,
                manager,
                actorInventory,
                benchInput,
                moved
            );
            return NativeHeadlessCrafting.StartResult.rejected(
                "native_crafting_manager_rejected"
            );
        }
        return new NativeHeadlessCrafting.StartResult(
            true,
            true,
            manager,
            inventoryBefore,
            NativeHeadlessCrafting.canonicalInventory(reference, store),
            outputBefore,
            NativeHeadlessCrafting.outputQuantity(reference, store, recipe),
            "",
            benchInput
        );
    }

    private static void collectRemoved(
        ListTransaction<MaterialTransaction> removal,
        List<ItemStack> out
    ) {
        for (MaterialTransaction material : removal.getList()) {
            for (MaterialSlotTransaction slot : material.getList()) {
                ItemStack output = slot.getOutput();
                if (!ItemStack.isEmpty(output)) {
                    out.add(output);
                }
            }
        }
    }

    /** Returns every moved stack to the actor and clears the bench slot, so nothing is duplicated. */
    private static void rollback(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        CraftingManager manager,
        CombinedItemContainer actorInventory,
        ItemContainer benchInput,
        List<ItemStack> moved
    ) {
        benchInput.clear();
        actorInventory.addItemStacks(moved);
        manager.clearBench(reference, store);
    }
}
