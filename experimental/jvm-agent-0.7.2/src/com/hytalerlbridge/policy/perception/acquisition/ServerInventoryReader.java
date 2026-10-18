package com.hytalerlbridge.policy.perception.acquisition;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.inventory.ActiveSlotInventoryComponent;
import com.hypixel.hytale.server.core.inventory.InventoryComponent;
import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.inventory.container.ItemContainer;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.policy.perception.profile.StatusProgramCatalog;
import com.hytalerlbridge.policy.perception.projection.InventoryProjection;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Converts the public six-container inventory into actor-safe policy tokens. */
public final class ServerInventoryReader {

    /** Public-API-independent stack row used by the tested ordering core. */
    record StackRow(
        int slot,
        String itemId,
        int quantity,
        double durability,
        double maximumDurability
    ) {}

    /** One independently available native container. */
    record ContainerRow(
        boolean available,
        int capacity,
        int activeSlot,
        List<StackRow> occupied
    ) {
        ContainerRow {
            occupied = List.copyOf(occupied);
        }
    }

    private ServerInventoryReader() {
    }

    /**
     * Capture occupied stacks in canonical container/local-slot order.
     *
     * <p>A missing individual container remains independently unavailable.
     * Invalid stacks, semantic-ID collisions, or more than 76 occupied stacks
     * invalidate the whole row instead of publishing a truncated inventory.
     */
    public static InventoryProjection.Input capture(
        Ref<EntityStore> ref,
        Store<EntityStore> store
    ) {
        if (ref == null || !ref.isValid() || store == null) {
            return unavailable();
        }
        InventoryComponent.Storage storage = store.getComponent(
            ref, InventoryComponent.Storage.getComponentType());
        InventoryComponent.Armor armor = store.getComponent(
            ref, InventoryComponent.Armor.getComponentType());
        InventoryComponent.Hotbar hotbar = store.getComponent(
            ref, InventoryComponent.Hotbar.getComponentType());
        InventoryComponent.Utility utility = store.getComponent(
            ref, InventoryComponent.Utility.getComponentType());
        InventoryComponent.Tool tools = store.getComponent(
            ref, InventoryComponent.Tool.getComponentType());
        InventoryComponent.Backpack backpack = store.getComponent(
            ref, InventoryComponent.Backpack.getComponentType());
        boolean sourceAvailable = storage != null || armor != null
            || hotbar != null || utility != null || tools != null
            || backpack != null;
        return captureContainers(sourceAvailable, new ItemContainer[] {
            inventory(storage),
            inventory(armor),
            inventory(hotbar),
            inventory(utility),
            inventory(tools),
            inventory(backpack),
        }, new int[] {
            -1,
            -1,
            activeSlot(hotbar),
            activeSlot(utility),
            activeSlot(tools),
            -1,
        });
    }

    static InventoryProjection.Input captureContainers(
        boolean sourceAvailable,
        ItemContainer[] containers,
        int[] activeSlots
    ) {
        if (!sourceAvailable) {
            return unavailable();
        }
        if (containers == null
            || containers.length != InventoryProjection.CONTAINER_COUNT
            || activeSlots == null
            || activeSlots.length != InventoryProjection.CONTAINER_COUNT) {
            throw new IllegalArgumentException("inventory container axis drift");
        }
        ContainerRow[] rows = new ContainerRow[containers.length];
        for (int container = 0; container < containers.length; container++) {
            ItemContainer source = containers[container];
            if (source == null) {
                rows[container] = new ContainerRow(
                    false, 0, -1, List.of());
                continue;
            }
            int capacity = Short.toUnsignedInt(source.getCapacity());
            List<StackRow> occupied = new ArrayList<>();
            for (int slot = 0; slot < capacity; slot++) {
                ItemStack stack = source.getItemStack((short) slot);
                if (!ItemStack.isEmpty(stack)) {
                    occupied.add(new StackRow(
                        slot,
                        stack.getItemId(),
                        stack.getQuantity(),
                        stack.getDurability(),
                        stack.getMaxDurability()
                    ));
                }
            }
            rows[container] = new ContainerRow(
                true, capacity, activeSlots[container], occupied);
        }
        return captureRows(sourceAvailable, rows);
    }

    static InventoryProjection.Input captureRows(
        boolean sourceAvailable,
        ContainerRow[] rows
    ) {
        if (!sourceAvailable) {
            return unavailable();
        }
        if (rows == null
            || rows.length != InventoryProjection.CONTAINER_COUNT) {
            throw new IllegalArgumentException("inventory container axis drift");
        }
        boolean[] containerAvailable = new boolean[
            InventoryProjection.CONTAINER_COUNT];
        int[] containerCapacity = new int[
            InventoryProjection.CONTAINER_COUNT];
        boolean[] tokenMask = new boolean[
            InventoryProjection.TOKEN_CAPACITY];
        int[] containerId = new int[InventoryProjection.TOKEN_CAPACITY];
        int[] containerSlot = new int[InventoryProjection.TOKEN_CAPACITY];
        int[] itemId = new int[InventoryProjection.TOKEN_CAPACITY];
        int[] quantity = new int[InventoryProjection.TOKEN_CAPACITY];
        float[] durability = new float[InventoryProjection.TOKEN_CAPACITY];
        boolean[] active = new boolean[InventoryProjection.TOKEN_CAPACITY];
        Map<Integer, String> semanticIds = new HashMap<>();

        int token = 0;
        for (int container = 0; container < rows.length; container++) {
            ContainerRow source = rows[container];
            if (source == null) {
                throw new IllegalArgumentException("null inventory row");
            }
            if (!source.available()) {
                if (source.capacity() != 0 || source.activeSlot() != -1
                    || !source.occupied().isEmpty()) {
                    return unavailable();
                }
                continue;
            }
            int capacity = source.capacity();
            int activeSlot = source.activeSlot();
            if (capacity < 0 || activeSlot < -1 || activeSlot >= capacity) {
                return unavailable();
            }
            containerAvailable[container] = true;
            containerCapacity[container] = capacity;
            int previousSlot = -1;
            for (StackRow stack : source.occupied()) {
                if (token >= InventoryProjection.TOKEN_CAPACITY
                    || !valid(stack)
                    || stack.slot() <= previousSlot
                    || stack.slot() >= capacity) {
                    return unavailable();
                }
                previousSlot = stack.slot();
                int semanticId;
                try {
                    semanticId = StatusProgramCatalog.semanticId(
                        stack.itemId());
                } catch (IllegalArgumentException invalid) {
                    return unavailable();
                }
                String previous = semanticIds.putIfAbsent(
                    semanticId, stack.itemId());
                if (previous != null && !previous.equals(stack.itemId())) {
                    return unavailable();
                }
                tokenMask[token] = true;
                containerId[token] = container;
                containerSlot[token] = stack.slot();
                itemId[token] = semanticId;
                quantity[token] = stack.quantity();
                durability[token] = stack.maximumDurability() > 0.0
                    ? (float) (stack.durability()
                        / stack.maximumDurability())
                    : 0.0f;
                active[token] = activeSlot >= 0
                    && activeSlot == stack.slot();
                token++;
            }
        }
        return new InventoryProjection.Input(
            true,
            containerAvailable,
            containerCapacity,
            tokenMask,
            containerId,
            containerSlot,
            itemId,
            quantity,
            durability,
            active
        );
    }

    static InventoryProjection.Input unavailable() {
        return new InventoryProjection.Input(
            false,
            new boolean[InventoryProjection.CONTAINER_COUNT],
            new int[InventoryProjection.CONTAINER_COUNT],
            new boolean[InventoryProjection.TOKEN_CAPACITY],
            new int[InventoryProjection.TOKEN_CAPACITY],
            new int[InventoryProjection.TOKEN_CAPACITY],
            new int[InventoryProjection.TOKEN_CAPACITY],
            new int[InventoryProjection.TOKEN_CAPACITY],
            new float[InventoryProjection.TOKEN_CAPACITY],
            new boolean[InventoryProjection.TOKEN_CAPACITY]
        );
    }

    private static ItemContainer inventory(InventoryComponent component) {
        return component == null ? null : component.getInventory();
    }

    private static int activeSlot(ActiveSlotInventoryComponent component) {
        if (component == null) {
            return -1;
        }
        byte value = component.getActiveSlot();
        return value < 0 ? -1 : Byte.toUnsignedInt(value);
    }

    private static boolean valid(StackRow stack) {
        if (stack == null || stack.itemId() == null
            || stack.itemId().isBlank() || stack.quantity() <= 0) {
            return false;
        }
        double value = stack.durability();
        double maximum = stack.maximumDurability();
        return Double.isFinite(value)
            && Double.isFinite(maximum)
            && value >= 0.0
            && maximum >= 0.0
            && (maximum > 0.0 ? value <= maximum : value == 0.0);
    }
}
