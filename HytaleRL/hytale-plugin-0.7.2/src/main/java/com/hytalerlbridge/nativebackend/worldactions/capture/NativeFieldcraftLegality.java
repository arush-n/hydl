package com.hytalerlbridge.nativebackend.worldactions.capture;

import com.hypixel.hytale.builtin.adventure.memories.MemoriesPlugin;
import com.hypixel.hytale.builtin.crafting.component.CraftingManager;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.BenchRequirement;
import com.hypixel.hytale.protocol.BenchType;
import com.hypixel.hytale.server.core.asset.type.item.config.CraftingRecipe;
import com.hypixel.hytale.server.core.entity.entities.Player;
import com.hypixel.hytale.server.core.inventory.InventoryComponent;
import com.hypixel.hytale.server.core.inventory.MaterialQuantity;
import com.hypixel.hytale.server.core.inventory.container.CombinedItemContainer;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import java.util.Set;

/** Exact actor gates shared by Fieldcraft capture and execution. */
public final class NativeFieldcraftLegality {

    private NativeFieldcraftLegality() {}

    public static ActorContext capture(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        World world
    ) {
        if (actor == null || !actor.isValid() || store == null || world == null) {
            return ActorContext.unavailable("controlled_actor_unavailable");
        }
        Player player = store.getComponent(actor, Player.getComponentType());
        if (player == null || player.getPlayerConfigData() == null) {
            return ActorContext.unavailable(
                "crafting_knowledge_context_unavailable"
            );
        }
        MemoriesPlugin memories = MemoriesPlugin.get();
        if (memories == null || world.getGameplayConfig() == null) {
            return ActorContext.unavailable(
                "crafting_memory_context_unavailable"
            );
        }
        CombinedItemContainer inventory = InventoryComponent.getCombined(
            store,
            actor,
            InventoryComponent.BACKPACK_STORAGE_HOTBAR
        );
        if (inventory == null) {
            return ActorContext.unavailable(
                "crafting_input_inventory_unavailable"
            );
        }
        return ActorContext.available(
            player.getPlayerConfigData().getKnownRecipes(),
            memories.getMemoriesLevel(world.getGameplayConfig()),
            inventory
        );
    }

    public static String rejectionReason(
        CraftingRecipe recipe,
        ActorContext context,
        int quantity
    ) {
        if (recipe == null) return "craft_recipe_identity_changed";
        if (quantity < 1) return "craft_quantity_invalid";
        if (!matchesFieldcraft(recipe.getBenchRequirement())) {
            return "craft_recipe_not_available_in_fieldcraft";
        }
        String actorReason = authoredActorGateReason(recipe, context);
        if (!actorReason.isEmpty()) return actorReason;
        if (
            !context.inventory().canRemoveMaterials(
                CraftingManager.getInputMaterials(recipe, quantity)
            )
        ) {
            return "native_crafting_materials_changed";
        }
        return "";
    }

    /** Knowledge and memories checks common to every native bench context. */
    public static String authoredActorGateReason(
        CraftingRecipe recipe,
        ActorContext context
    ) {
        if (recipe == null) return "craft_recipe_identity_changed";
        if (context == null || !context.available()) {
            return context == null
                ? "crafting_actor_context_unavailable"
                : context.unavailableReason();
        }
        String knowledgeKey = knowledgeKey(recipe);
        if (
            recipe.isKnowledgeRequired()
                && (knowledgeKey.isEmpty()
                    || !context.knownRecipes().contains(knowledgeKey))
        ) {
            return "craft_recipe_knowledge_unavailable";
        }
        if (!memoriesSatisfied(
            recipe.getRequiredMemoriesLevel(),
            context.memoriesLevel()
        )) {
            return "craft_recipe_memories_unavailable";
        }
        return "";
    }

    /** Mirrors CraftingManager's authored `required > 1` memories gate. */
    public static boolean memoriesSatisfied(
        int requiredMemoriesLevel,
        int currentMemoriesLevel
    ) {
        if (requiredMemoriesLevel < 0 || currentMemoriesLevel < 0) {
            throw new IllegalArgumentException(
                "memories levels must be nonnegative"
            );
        }
        return requiredMemoriesLevel <= 1
            || currentMemoriesLevel >= requiredMemoriesLevel;
    }

    public static boolean matchesFieldcraft(BenchRequirement[] requirements) {
        if (requirements == null) return false;
        for (BenchRequirement requirement : requirements) {
            if (
                requirement != null
                    && requirement.type == BenchType.Crafting
                    && "Fieldcraft".equals(requirement.id)
                    && requirement.requiredTierLevel <= 0
            ) {
                return true;
            }
        }
        return false;
    }

    public static String knowledgeKey(CraftingRecipe recipe) {
        if (recipe == null || !recipe.isKnowledgeRequired()) return "";
        MaterialQuantity output = recipe.getPrimaryOutput();
        return output == null || output.getItemId() == null
            ? ""
            : output.getItemId();
    }

    public record ActorContext(
        boolean available,
        String unavailableReason,
        Set<String> knownRecipes,
        int memoriesLevel,
        CombinedItemContainer inventory
    ) {
        public ActorContext {
            unavailableReason = unavailableReason == null
                ? ""
                : unavailableReason;
            knownRecipes = knownRecipes == null
                ? Set.of()
                : Set.copyOf(knownRecipes);
            if (available) {
                if (
                    !unavailableReason.isEmpty()
                        || memoriesLevel < 0
                        || inventory == null
                ) {
                    throw new IllegalArgumentException(
                        "Available Fieldcraft actor context is incomplete"
                    );
                }
            } else if (
                unavailableReason.isEmpty()
                    || !knownRecipes.isEmpty()
                    || memoriesLevel != 0
                    || inventory != null
            ) {
                throw new IllegalArgumentException(
                    "Unavailable Fieldcraft actor context must fail closed"
                );
            }
        }

        static ActorContext available(
            Set<String> knownRecipes,
            int memoriesLevel,
            CombinedItemContainer inventory
        ) {
            return new ActorContext(
                true,
                "",
                knownRecipes,
                memoriesLevel,
                inventory
            );
        }

        static ActorContext unavailable(String reason) {
            return new ActorContext(false, reason, Set.of(), 0, null);
        }
    }
}
