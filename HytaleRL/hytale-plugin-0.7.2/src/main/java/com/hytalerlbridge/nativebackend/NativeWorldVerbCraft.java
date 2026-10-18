package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.builtin.crafting.component.CraftingManager;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.asset.type.item.config.CraftingRecipe;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.action.NativeWorldVerbRequest;

/** Stateless commit-time admission for one exact native crafting request. */
final class NativeWorldVerbCraft {

    private NativeWorldVerbCraft() {}

    static StartResult start(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        World world,
        NativeWorldVerbRequest request,
        boolean headlessWorldVerbsEnabled
    ) {
        if (!headlessWorldVerbsEnabled) {
            return StartResult.rejected(
                "typed_craft_recipe_requires_headless_world_verb_context"
            );
        }
        if (
            !request.sourceContainer().equals("player_inventory")
                || !request.destinationContainer().equals(
                    "player_inventory_or_world_drop"
                )
        ) {
            return StartResult.rejected(
                "typed_craft_inventory_contract_mismatch"
            );
        }

        NativeHeadlessCrafting.StartResult result =
            NativeHeadlessCrafting.start(
                ref,
                store,
                world,
                request
            );
        if (!result.accepted()) {
            return StartResult.rejected(result.rejectReason());
        }
        return StartResult.accepted(
            result.queued(),
            result.manager(),
            result.queued()
                ? NativeHeadlessCrafting.recipe(request.recipeId())
                : null,
            result.inventoryBefore(),
            result.inventoryAfter(),
            result.outputQuantityBefore(),
            result.outputQuantityAfter()
        );
    }

    record StartResult(
        boolean accepted,
        boolean queued,
        CraftingManager manager,
        CraftingRecipe recipe,
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
                null,
                "",
                "",
                -1,
                -1,
                reason
            );
        }

        private static StartResult accepted(
            boolean queued,
            CraftingManager manager,
            CraftingRecipe recipe,
            String inventoryBefore,
            String inventoryAfter,
            int outputQuantityBefore,
            int outputQuantityAfter
        ) {
            return new StartResult(
                true,
                queued,
                manager,
                recipe,
                inventoryBefore,
                inventoryAfter,
                outputQuantityBefore,
                outputQuantityAfter,
                ""
            );
        }
    }
}
