package com.hytalerlbridge.nativebackend.session;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.asset.type.item.config.Item;
import com.hypixel.hytale.server.core.inventory.Inventory;
import com.hypixel.hytale.server.core.inventory.InventoryComponent;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.environment.EnvironmentOptions;
import com.hytalerlbridge.nativebackend.model.NativeArmorEntry;
import java.util.ArrayList;
import java.util.List;

import static com.hytalerlbridge.nativebackend.support.NativeWorldVerbSupport.requiredHotbarCapacity;

/**
 * Native agent armor and hotbar setup applied when the agent is spawned.
 */
public final class SpawnEquipment {

    private SpawnEquipment() {}

    public static void replaceNativeAgentArmor(
        List<NativeArmorEntry> nativeAgentArmorOverride,
        NPCEntity npc
    ) {
        if (nativeAgentArmorOverride == null) return;
        Inventory inventory = npc.getInventory();
        if (inventory == null || inventory.getArmor() == null) {
            throw new IllegalStateException(
                "Native agent armor override requires an NPC armor inventory"
            );
        }
        ItemContainer armor = inventory.getArmor();
        for (NativeArmorEntry entry : nativeAgentArmorOverride) {
            if (entry.slot() < 0 || entry.slot() >= armor.getCapacity()) {
                throw new IllegalStateException(
                    "Native NPC armor inventory has no slot for "
                        + entry.itemId()
                );
            }
        }
        if (!armor.clear().succeeded()) {
            throw new IllegalStateException(
                "Native NPC rejected clearing its role-authored armor"
            );
        }
        for (NativeArmorEntry entry : nativeAgentArmorOverride) {
            if (!armor.setItemStackForSlot(
                entry.slot(),
                new ItemStack(entry.itemId())
            ).succeeded()) {
                throw new IllegalStateException(
                    "Native NPC rejected armor item: " + entry.itemId()
                );
            }
        }
    }

    public static void equipNativeWorldHotbar(
        Ref<EntityStore> agentRef,
        EnvironmentOptions options,
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        Inventory inventory = npc.getInventory();
        if (inventory == null || inventory.getHotbar() == null) {
            throw new IllegalStateException(
                "Native World hotbar override requires an NPC hotbar"
            );
        }
        InventoryComponent.Hotbar hotbarComponent = store.getComponent(
            agentRef,
            InventoryComponent.Hotbar.getComponentType()
        );
        if (hotbarComponent == null) {
            throw new IllegalStateException(
                "Native World hotbar override requires a hotbar component"
            );
        }
        int capacity = Short.toUnsignedInt(
            hotbarComponent.getInventory().getCapacity()
        );
        int requiredCapacity = requiredHotbarCapacity(
            options.nativeWorldHotbarSlots(),
            capacity
        );
        if (requiredCapacity > capacity) {
            List<ItemStack> remainder = new ArrayList<>();
            hotbarComponent.ensureCapacity(
                (short) requiredCapacity,
                remainder
            );
            if (!remainder.isEmpty()) {
                throw new IllegalStateException(
                    "Native World hotbar expansion displaced existing items"
                );
            }
            capacity = requiredCapacity;
        }
        ItemContainer hotbar = hotbarComponent.getInventory();
        for (int index = 0;
            index < options.nativeWorldHotbarSlots().size();
            index++) {
            int slot = options.nativeWorldHotbarSlots().get(index);
            String itemId = options.nativeWorldHotbarItemIds().get(index);
            int quantity = options.nativeWorldHotbarQuantities().get(index);
            if (slot >= capacity) {
                throw new IllegalStateException(
                    "Native World hotbar slot " + slot
                        + " exceeds NPC hotbar capacity " + capacity
                );
            }
            if (Item.getAssetMap().getAsset(itemId) == null) {
                throw new IllegalStateException(
                    "Native World hotbar item is unknown: " + itemId
                );
            }
            if (!hotbar.setItemStackForSlot(
                (short) slot,
                new ItemStack(itemId, quantity)
            ).succeeded()) {
                throw new IllegalStateException(
                    "Native NPC rejected hotbar item "
                        + itemId + " in slot " + slot
                );
            }
        }
        int activeSlot = options.nativeWorldHotbarSlots().get(0);
        inventory.setActiveHotbarSlot(agentRef, (byte) activeSlot, store);
    }
}
