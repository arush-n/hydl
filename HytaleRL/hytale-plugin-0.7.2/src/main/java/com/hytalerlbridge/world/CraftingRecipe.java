package com.hytalerlbridge.world;

import java.util.Map;

/**
 * A crafting recipe mapping input item IDs (block types) to an output.
 */
public record CraftingRecipe(
    String name,
    Map<Integer, Integer> inputs,  // blockType id -> count required
    int outputId,
    int outputCount
) {
    /** Check if an inventory has enough items to craft this recipe. */
    public boolean canCraft(int[] inventory) {
        int consumedSlots = 0;
        for (var entry : inputs.entrySet()) {
            int required = entry.getValue();
            int available = countItem(inventory, entry.getKey());
            if (available < required) return false;
            consumedSlots += required;
        }
        int emptySlots = 0;
        for (int slot : inventory) {
            if (slot == 0) emptySlots++;
        }
        return emptySlots + consumedSlots >= outputCount;
    }

    /** Consume inputs from inventory. */
    public void consumeInputs(int[] inventory) {
        for (var entry : inputs.entrySet()) {
            removeItem(inventory, entry.getKey(), entry.getValue());
        }
    }

    /** Add output to inventory. Returns true if added. */
    public boolean addOutput(int[] inventory) {
        int added = 0;
        for (int i = 0; i < inventory.length && added < outputCount; i++) {
            if (inventory[i] == 0) {
                inventory[i] = outputId;
                added++;
            }
        }
        return added == outputCount;
    }

    private static int countItem(int[] inventory, int itemId) {
        int count = 0;
        for (int slot : inventory) {
            if (slot == itemId) count++;
        }
        return count;
    }

    private static void removeItem(int[] inventory, int itemId, int count) {
        int removed = 0;
        for (int i = 0; i < inventory.length && removed < count; i++) {
            if (inventory[i] == itemId) {
                inventory[i] = 0;
                removed++;
            }
        }
    }
}
