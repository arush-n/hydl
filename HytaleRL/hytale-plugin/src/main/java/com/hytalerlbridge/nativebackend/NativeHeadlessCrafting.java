package com.hytalerlbridge.nativebackend;

import static com.hytalerlbridge.nativebackend.support.NativeKeys
    .canonicalWorldVerbInventoryState;

import com.hypixel.hytale.builtin.crafting.component.BenchBlock;
import com.hypixel.hytale.builtin.crafting.component.CraftingManager;
import com.hypixel.hytale.builtin.crafting.window.SimpleCraftingWindow;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.protocol.BenchRequirement;
import com.hypixel.hytale.protocol.BenchType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.bench.Bench;
import com.hypixel.hytale.server.core.asset.type.item.config.CraftingRecipe;
import com.hypixel.hytale.server.core.inventory.InventoryComponent;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.CombinedItemContainer;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.entity.entities.player.windows
    .MaterialExtraResourcesSection;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.chunk
    .BlockComponentChunk;
import com.hypixel.hytale.server.core.universe.world.chunk.WorldChunk;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.action.NativeWorldVerbRequest;
import com.hytalerlbridge.nativebackend.worldactions.capture.NativeFieldcraftLegality;
import java.util.HashSet;
import java.util.Set;

/**
 * Native CraftingManager adapter for an opt-in headless Adventure context.
 *
 * <p>This executes server crafting mechanics, not an authenticated player's
 * window protocol. Fieldcraft and ordinary Crafting benches are lossless
 * because their native input removal is inventory/material based. Diagram,
 * Structural and Processing benches are rejected until a request transports
 * their ordered/specialized input state.</p>
 */
final class NativeHeadlessCrafting {

    private NativeHeadlessCrafting() {}

    static StartResult start(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        World world,
        NativeWorldVerbRequest request
    ) {
        CraftingManager manager = store.getComponent(
            reference,
            CraftingManager.getComponentType()
        );
        if (manager == null) {
            return StartResult.rejected(
                "headless_crafting_manager_unavailable"
            );
        }
        if (manager.hasBenchSet() || manager.getRemainingQueueSize() != 0) {
            return StartResult.rejected("headless_crafting_manager_busy");
        }
        CraftingRecipe recipe = CraftingRecipe.getAssetMap().getAsset(
            request.recipeId()
        );
        if (recipe == null) {
            return StartResult.rejected("craft_recipe_identity_changed");
        }
        NativeFieldcraftLegality.ActorContext actorContext =
            NativeFieldcraftLegality.capture(reference, store, world);
        String actorGateReason =
            NativeFieldcraftLegality.authoredActorGateReason(
                recipe,
                actorContext
            );
        if (!actorGateReason.isEmpty()) {
            return StartResult.rejected(actorGateReason);
        }

        CombinedItemContainer actorInventory = InventoryComponent.getCombined(
            store,
            reference,
            InventoryComponent.BACKPACK_STORAGE_HOTBAR
        );
        ItemContainer input = actorInventory;
        SimpleCraftingWindow timedWindow = null;
        boolean benchBound = false;

        if (request.craftingContext().equals("fieldcraft")) {
            // FieldCraftingWindow delegates every recipe directly to
            // CraftingManager.craftItem; authored TimeSeconds is not queued.
            if (
                !matchesBenchRequirements(
                    recipe.getBenchRequirement(),
                    BenchType.Crafting,
                    "Fieldcraft",
                    0
                )
            ) {
                return StartResult.rejected(
                    "craft_recipe_not_available_in_fieldcraft"
                );
            }
        } else {
            BenchResolution bench = resolveBench(world, request);
            if (!bench.accepted()) {
                return StartResult.rejected(bench.rejectReason());
            }
            if (bench.benchAsset().getType() != BenchType.Crafting) {
                return StartResult.rejected(
                    "ordered_or_processing_bench_input_not_transported"
                );
            }
            if (
                !matchesBenchRequirements(
                    recipe.getBenchRequirement(),
                    bench.benchAsset().getType(),
                    bench.benchAsset().getId(),
                    bench.benchBlock().getTierLevel()
                )
            ) {
                return StartResult.rejected(
                    "craft_recipe_not_available_at_bench"
                );
            }

            manager.setBench(
                request.benchX(),
                request.benchY(),
                request.benchZ(),
                bench.blockType()
            );
            benchBound = true;
            MaterialExtraResourcesSection nearbyResources =
                new MaterialExtraResourcesSection();
            CraftingManager.feedExtraResourcesSection(
                world,
                request.benchX(),
                request.benchY(),
                request.benchZ(),
                bench.blockType(),
                bench.rotationIndex(),
                bench.benchAsset(),
                bench.benchBlock().getTierLevel(),
                nearbyResources
            );
            input = new CombinedItemContainer(
                actorInventory,
                nearbyResources.getItemContainer()
            );
            timedWindow = new SimpleCraftingWindow(
                request.benchX(),
                request.benchY(),
                request.benchZ(),
                bench.rotationIndex(),
                bench.blockType(),
                bench.benchBlock()
            );
        }

        String inventoryBefore = canonicalInventory(reference, store);
        int outputBefore = outputQuantity(reference, store, recipe);
        if (
            !input.canRemoveMaterials(
                CraftingManager.getInputMaterials(
                    recipe,
                    request.quantity()
                )
            )
        ) {
            if (benchBound) {
                manager.clearBench(reference, store);
            }
            return StartResult.rejected(
                "native_crafting_materials_changed"
            );
        }
        boolean queued = queuesCraft(
            request.craftingContext(),
            recipe.getTimeSeconds()
        );
        boolean accepted;
        if (queued) {
            if (timedWindow == null) {
                if (benchBound) {
                    manager.clearBench(reference, store);
                }
                return StartResult.rejected(
                    "timed_craft_requires_native_crafting_bench"
                );
            }
            accepted = manager.queueCraft(
                reference,
                store,
                timedWindow,
                0,
                recipe,
                request.quantity(),
                input,
                CraftingManager.InputRemovalType.NORMAL
            );
        } else {
            accepted = manager.craftItem(
                reference,
                store,
                recipe,
                request.quantity(),
                input
            );
        }
        if (!accepted) {
            if (benchBound) {
                manager.clearBench(reference, store);
            }
            return StartResult.rejected(
                "native_crafting_manager_rejected"
            );
        }

        String inventoryAfter = canonicalInventory(reference, store);
        int outputAfter = outputQuantity(reference, store, recipe);
        if (!queued && benchBound) {
            manager.clearBench(reference, store);
        }
        return new StartResult(
            true,
            queued,
            manager,
            inventoryBefore,
            inventoryAfter,
            outputBefore,
            outputAfter,
            ""
        );
    }

    static String canonicalInventory(
        Ref<EntityStore> reference,
        Store<EntityStore> store
    ) {
        ItemContainer inventory = InventoryComponent.getCombined(
            store,
            reference,
            InventoryComponent.EVERYTHING
        );
        StringBuilder result = new StringBuilder();
        int capacity = Short.toUnsignedInt(inventory.getCapacity());
        for (int slot = 0; slot < capacity; slot++) {
            ItemStack stack = inventory.getItemStack((short) slot);
            String value = canonicalWorldVerbInventoryState(stack);
            result.append(slot)
                .append(':')
                .append(value.length())
                .append(':')
                .append(value)
                .append(';');
        }
        return result.toString();
    }

    static int outputQuantity(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        CraftingRecipe recipe
    ) {
        Set<String> outputIds = new HashSet<>();
        for (ItemStack output : CraftingManager.getOutputItemStacks(recipe)) {
            if (!ItemStack.isEmpty(output)) {
                outputIds.add(output.getItemId());
            }
        }
        ItemContainer inventory = InventoryComponent.getCombined(
            store,
            reference,
            InventoryComponent.EVERYTHING
        );
        int total = 0;
        int capacity = Short.toUnsignedInt(inventory.getCapacity());
        for (int slot = 0; slot < capacity; slot++) {
            ItemStack stack = inventory.getItemStack((short) slot);
            if (
                !ItemStack.isEmpty(stack)
                    && outputIds.contains(stack.getItemId())
            ) {
                total = Math.addExact(total, stack.getQuantity());
            }
        }
        return total;
    }

    static CraftingRecipe recipe(String recipeId) {
        return CraftingRecipe.getAssetMap().getAsset(recipeId);
    }

    static boolean queuesCraft(String craftingContext, float timeSeconds) {
        if (
            craftingContext == null
                || (
                    !craftingContext.equals("fieldcraft")
                        && !craftingContext.equals("bench")
                )
                || !Float.isFinite(timeSeconds)
                || timeSeconds < 0.0F
        ) {
            throw new IllegalArgumentException(
                "Invalid native crafting queue decision"
            );
        }
        return craftingContext.equals("bench") && timeSeconds > 0.0F;
    }

    /**
     * Mirrors CraftingManager.isValidBenchForRecipe's resolved bench
     * predicate. The native manager repeats this check during execution; this
     * preflight keeps rejection typed and guarantees that no inventory work
     * begins for a mismatched request context.
     */
    static boolean matchesBenchRequirements(
        BenchRequirement[] requirements,
        BenchType benchType,
        String benchId,
        int benchTier
    ) {
        if (
            requirements == null
                || benchType == null
                || benchId == null
                || benchTier < 0
        ) {
            return false;
        }
        for (BenchRequirement requirement : requirements) {
            if (
                requirement != null
                    && requirement.type == benchType
                    && benchId.equals(requirement.id)
                    && requirement.requiredTierLevel <= benchTier
            ) {
                return true;
            }
        }
        return false;
    }

    private static BenchResolution resolveBench(
        World world,
        NativeWorldVerbRequest request
    ) {
        BlockType blockType = world.getBlockType(
            request.benchX(),
            request.benchY(),
            request.benchZ()
        );
        if (
            blockType == null
                || !blockType.getId().equals(
                    request.expectedBenchBlockId()
                )
        ) {
            return BenchResolution.rejected(
                "craft_bench_block_identity_changed"
            );
        }
        Bench benchAsset = blockType.getBench();
        if (
            benchAsset == null
                || !benchAsset.getId().equals(request.expectedBenchId())
                || benchAsset.getType().ordinal()
                    != request.expectedBenchType()
        ) {
            return BenchResolution.rejected(
                "craft_bench_asset_identity_changed"
            );
        }

        Ref<ChunkStore> chunkReference = world
            .getChunkStore()
            .getChunkReference(
                ChunkUtil.indexChunkFromBlock(
                    request.benchX(),
                    request.benchZ()
                )
            );
        if (chunkReference == null || !chunkReference.isValid()) {
            return BenchResolution.rejected(
                "craft_bench_chunk_unavailable"
            );
        }
        Store<ChunkStore> chunkStore = world.getChunkStore().getStore();
        BlockComponentChunk components = chunkStore.getComponent(
            chunkReference,
            BlockComponentChunk.getComponentType()
        );
        if (components == null) {
            return BenchResolution.rejected(
                "craft_bench_component_chunk_unavailable"
            );
        }
        Ref<ChunkStore> blockReference = components.getEntityReference(
            ChunkUtil.indexBlockInColumn(
                request.benchX(),
                request.benchY(),
                request.benchZ()
            )
        );
        if (blockReference == null || !blockReference.isValid()) {
            return BenchResolution.rejected(
                "craft_bench_entity_unavailable"
            );
        }
        BenchBlock benchBlock = chunkStore.getComponent(
            blockReference,
            BenchBlock.getComponentType()
        );
        if (
            benchBlock == null
                || benchBlock.getTierLevel()
                    != request.expectedBenchTier()
        ) {
            return BenchResolution.rejected(
                "craft_bench_tier_changed"
            );
        }
        WorldChunk worldChunk = world.getChunk(
            ChunkUtil.indexChunkFromBlock(
                request.benchX(),
                request.benchZ()
            )
        );
        if (worldChunk == null) {
            return BenchResolution.rejected(
                "craft_bench_world_chunk_unavailable"
            );
        }
        return new BenchResolution(
            true,
            blockType,
            benchAsset,
            benchBlock,
            worldChunk.getRotationIndex(
                request.benchX(),
                request.benchY(),
                request.benchZ()
            ),
            ""
        );
    }

    record StartResult(
        boolean accepted,
        boolean queued,
        CraftingManager manager,
        String inventoryBefore,
        String inventoryAfter,
        int outputQuantityBefore,
        int outputQuantityAfter,
        String rejectReason
    ) {
        private static StartResult rejected(String reason) {
            return new StartResult(
                false,
                false,
                null,
                "",
                "",
                -1,
                -1,
                reason
            );
        }
    }

    private record BenchResolution(
        boolean accepted,
        BlockType blockType,
        Bench benchAsset,
        BenchBlock benchBlock,
        int rotationIndex,
        String rejectReason
    ) {
        private static BenchResolution rejected(String reason) {
            return new BenchResolution(
                false,
                null,
                null,
                null,
                0,
                reason
            );
        }
    }
}
