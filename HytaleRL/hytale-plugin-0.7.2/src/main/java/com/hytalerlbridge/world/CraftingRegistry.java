package com.hytalerlbridge.world;

import java.util.*;

/**
 * Registry of all crafting recipes available to the NPC agent.
 *
 * Recipes are simplified versions of Hytale's crafting system.
 * In the full game, different recipes require different stations
 * (Workbench, Furnace, Blacksmith Anvil, etc.). For the RL environment,
 * all recipes are available when the agent has the required materials.
 */
public class CraftingRegistry {

    private final Map<Integer, CraftingRecipe> recipes = new LinkedHashMap<>();

    public CraftingRegistry() {
        registerDefaults();
    }

    private void registerDefaults() {
        // LOG -> 4x SOFTWOOD_PLANKS
        register(0, new CraftingRecipe("softwood_planks",
            Map.of(BlockType.LOG.id(), 1),
            BlockType.SOFTWOOD_PLANKS.id(), 4));

        // 3x STONE + 4x LOG -> WORKBENCH
        register(1, new CraftingRecipe("workbench",
            Map.of(BlockType.STONE.id(), 3, BlockType.LOG.id(), 4),
            BlockType.WORKBENCH.id(), 1));

        // 8x QUARTZITE -> FURNACE
        register(2, new CraftingRecipe("furnace",
            Map.of(BlockType.QUARTZITE.id(), 8),
            BlockType.FURNACE.id(), 1));

        // 8x SOFTWOOD_PLANKS -> CHEST
        register(3, new CraftingRecipe("chest",
            Map.of(BlockType.SOFTWOOD_PLANKS.id(), 8),
            BlockType.CHEST.id(), 1));

        // 6x SOFTWOOD_PLANKS -> DOOR
        register(4, new CraftingRecipe("door",
            Map.of(BlockType.SOFTWOOD_PLANKS.id(), 6),
            BlockType.DOOR.id(), 1));

        // 1x RESIN + 1x STICKS -> 4x TORCH
        register(5, new CraftingRecipe("torch",
            Map.of(BlockType.RESIN.id(), 1, BlockType.STICKS.id(), 1),
            BlockType.TORCH.id(), 4));

        // 6x SOFTWOOD_PLANKS -> FENCE
        register(6, new CraftingRecipe("fence",
            Map.of(BlockType.SOFTWOOD_PLANKS.id(), 6),
            BlockType.FENCE.id(), 2));

        // 3x QUARTZITE + 3x SOFTWOOD_PLANKS -> ROOF_BLOCK
        register(7, new CraftingRecipe("roof_block",
            Map.of(BlockType.QUARTZITE.id(), 3, BlockType.SOFTWOOD_PLANKS.id(), 3),
            BlockType.ROOF_BLOCK.id(), 4));

        // 3x PLANT_FIBER + 2x SOFTWOOD_PLANKS -> CRUDE_BEDROLL
        register(8, new CraftingRecipe("crude_bedroll",
            Map.of(BlockType.PLANT_FIBER.id(), 3, BlockType.SOFTWOOD_PLANKS.id(), 2),
            BlockType.CRUDE_BEDROLL.id(), 1));
    }

    public void register(int recipeId, CraftingRecipe recipe) {
        recipes.put(recipeId, recipe);
    }

    public CraftingRecipe get(int recipeId) {
        return recipes.get(recipeId);
    }

    public int getRecipeCount() {
        return recipes.size();
    }

    public Collection<CraftingRecipe> getAll() {
        return recipes.values();
    }
}
