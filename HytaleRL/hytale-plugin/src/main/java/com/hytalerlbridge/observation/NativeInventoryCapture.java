package com.hytalerlbridge.observation;

import com.hypixel.hytale.server.core.inventory.Inventory;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Stateless quantity-aware native inventory capture shared by producers. */
public final class NativeInventoryCapture {

    private NativeInventoryCapture() {}

    public static NativeInventoryFrame capture(
        Inventory inventory,
        Map<String, Integer> itemIds
    ) {
        if (inventory == null) {
            return NativeInventoryFrame.unavailable("inventory_unavailable");
        }
        if (itemIds == null) {
            throw new IllegalArgumentException("itemIds cannot be null");
        }
        ItemContainer storage = inventory.getStorage();
        ItemContainer armor = inventory.getArmor();
        ItemContainer hotbar = inventory.getHotbar();
        ItemContainer utility = inventory.getUtility();
        ItemContainer tools = inventory.getTools();
        ItemContainer backpack = inventory.getBackpack();
        return new NativeInventoryFrame(
            true,
            "",
            hotbar == null ? -1 : inventory.getActiveHotbarSlot(),
            utility == null ? -1 : inventory.getActiveUtilitySlot(),
            tools == null ? -1 : inventory.getActiveToolsSlot(),
            List.of(
                container("storage", -2, storage, itemIds),
                container("armor", -3, armor, itemIds),
                container("hotbar", -1, hotbar, itemIds),
                container("utility", -5, utility, itemIds),
                container("tools", -8, tools, itemIds),
                container("backpack", -9, backpack, itemIds)
            )
        );
    }

    private static NativeInventoryFrame.Container container(
        String name,
        int sectionId,
        ItemContainer container,
        Map<String, Integer> itemIds
    ) {
        if (container == null) {
            return NativeInventoryFrame.Container.unavailable(
                name,
                sectionId,
                "inventory_container_unavailable"
            );
        }
        int capacity = Short.toUnsignedInt(container.getCapacity());
        List<NativeInventoryFrame.Slot> occupied = new ArrayList<>();
        for (int slot = 0; slot < capacity; slot++) {
            ItemStack stack = container.getItemStack((short) slot);
            if (ItemStack.isEmpty(stack)) continue;
            String itemId = stack.getItemId();
            occupied.add(new NativeInventoryFrame.Slot(
                slot,
                itemId,
                itemIds.getOrDefault(itemId, 0),
                stack.getQuantity(),
                stack.getDurability(),
                stack.getMaxDurability(),
                stack.getMetadata() != null
            ));
        }
        return new NativeInventoryFrame.Container(
            name,
            sectionId,
            true,
            "",
            capacity,
            occupied
        );
    }
}
