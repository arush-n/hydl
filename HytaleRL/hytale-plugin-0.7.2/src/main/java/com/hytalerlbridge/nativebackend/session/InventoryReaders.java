package com.hytalerlbridge.nativebackend.session;

import com.hypixel.hytale.server.core.inventory.Inventory;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hytalerlbridge.observation.NativeInventoryFrame;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Inventory readers: item id counts per container and the native inventory frame.
 */
public final class InventoryReaders {

    private static final int INVENTORY_SIZE = 36;

    private InventoryReaders() {}

    public static int[] inventoryIds(
        Map<String, Integer> itemIds,
        Inventory inventory
    ) {
        int[] result = new int[INVENTORY_SIZE];
        if (inventory == null) return result;
        int offset = copyContainer(itemIds, inventory.getHotbar(), result, 0);
        copyContainer(itemIds, inventory.getStorage(), result, offset);
        return result;
    }

    public static NativeInventoryFrame captureNativeInventory(
        Map<String, Integer> itemIds,
        Inventory inventory
    ) {
        if (inventory == null) {
            return NativeInventoryFrame.unavailable("inventory_unavailable");
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
                captureNativeInventoryContainer(itemIds, "storage", -2, storage),
                captureNativeInventoryContainer(itemIds, "armor", -3, armor),
                captureNativeInventoryContainer(itemIds, "hotbar", -1, hotbar),
                captureNativeInventoryContainer(itemIds, "utility", -5, utility),
                captureNativeInventoryContainer(itemIds, "tools", -8, tools),
                captureNativeInventoryContainer(itemIds, "backpack", -9, backpack)
            )
        );
    }

    private static NativeInventoryFrame.Container captureNativeInventoryContainer(
        Map<String, Integer> itemIds,
        String name,
        int sectionId,
        ItemContainer container
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

    private static int copyContainer(
        Map<String, Integer> itemIds,
        ItemContainer container,
        int[] destination,
        int offset
    ) {
        if (container == null || offset >= destination.length) return offset;
        int capacity = Short.toUnsignedInt(container.getCapacity());
        for (int slot = 0; slot < capacity && offset < destination.length; slot++, offset++) {
            ItemStack stack = container.getItemStack((short) slot);
            if (!ItemStack.isEmpty(stack)) {
                destination[offset] = itemIds.getOrDefault(stack.getItemId(), 0);
            }
        }
        return offset;
    }
}
