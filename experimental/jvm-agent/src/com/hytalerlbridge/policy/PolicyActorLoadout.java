package com.hytalerlbridge.policy;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.inventory.InventoryComponent;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.util.InventoryHelper;
import com.hytalerlbridge.policy.perception.acquisition.resources.CatalogBackedActorStats;

/** Installs the exact checkpoint item required by live policy lifecycle reads. */
public final class PolicyActorLoadout {

    private PolicyActorLoadout() {}

    /**
     * Put the policy actor's trained item in hotbar slot zero and settle its
     * catalog-backed stat rows before the actor can be claimed.
     *
     * <p>The native lifecycle contract intentionally rejects an actor whose
     * active item differs from the checkpoint binding. Explicit diagnostic
     * spawns therefore need this same preparation step as the headless combat
     * fixture; otherwise every live tick correctly fails closed as
     * {@code agent_item_mismatch}.</p>
     */
    public static String install(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        String itemId,
        CatalogBackedActorStats.Boundary boundary
    ) {
        if (reference == null || !reference.isValid() || store == null
            || itemId == null || itemId.isBlank() || boundary == null) {
            throw new IllegalArgumentException(
                "policy loadout requires a live actor, store, item, and boundary");
        }
        boolean installed = InventoryHelper.setHotbarItem(
            reference, itemId, (byte) 0, store);
        InventoryHelper.setHotbarSlot(reference, (byte) 0, store);
        ItemStack active = InventoryComponent.getItemInHand(store, reference);
        String actual = ItemStack.isEmpty(active) ? "" : active.getItemId();
        if (!installed || !itemId.equals(actual)) {
            throw new IllegalStateException(
                "policy loadout item mismatch: " + actual + " != " + itemId);
        }
        CatalogBackedActorStats.settle(reference, store, boundary);
        return actual;
    }
}
