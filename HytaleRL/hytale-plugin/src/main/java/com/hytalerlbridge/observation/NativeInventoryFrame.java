package com.hytalerlbridge.observation;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Sparse, quantity-preserving copy of the native six-container inventory.
 *
 * <p>Capacities are read from each native {@code ItemContainer}; only occupied
 * slots cross the wire. Each container carries independent validity so a
 * headless entity's missing armor or backpack container cannot erase valid
 * hotbar/storage evidence or masquerade as an empty container. The legacy
 * {@code int[36]} observation remains unchanged for compatibility.
 */
public record NativeInventoryFrame(
    boolean available,
    String unavailableReason,
    int activeHotbarSlot,
    int activeUtilitySlot,
    int activeToolsSlot,
    List<Container> containers
) {
    public static final String SCHEMA = "hytalerl_native_inventory_v2";
    public static final int VERSION = 2;
    public static final String CONTRACT_SHA256 = "CC393B3EF1AAD2D54D5DA3FE7CB736C8439D9675C1A1D5D0E85E6BAAA82D0D8F";
    public static final List<String> CONTAINER_ORDER = List.of(
        "storage",
        "armor",
        "hotbar",
        "utility",
        "tools",
        "backpack"
    );
    public static final List<Integer> SECTION_IDS = List.of(
        -2,
        -3,
        -1,
        -5,
        -8,
        -9
    );

    public NativeInventoryFrame {
        unavailableReason = unavailableReason == null ? "" : unavailableReason;
        containers = containers == null ? List.of() : List.copyOf(containers);
        if (available) {
            if (!unavailableReason.isEmpty()) {
                throw new IllegalArgumentException(
                    "available inventory cannot carry an unavailable reason"
                );
            }
            if (containers.size() != CONTAINER_ORDER.size()) {
                throw new IllegalArgumentException(
                    "available inventory must carry all six containers"
                );
            }
            for (int index = 0; index < containers.size(); index++) {
                Container container = containers.get(index);
                if (
                    !container.name().equals(CONTAINER_ORDER.get(index))
                        || container.sectionId() != SECTION_IDS.get(index)
                ) {
                    throw new IllegalArgumentException(
                        "native inventory container order or section id drift"
                    );
                }
            }
            requireActiveSlot(
                activeHotbarSlot,
                containers.get(2),
                "hotbar"
            );
            requireActiveSlot(
                activeUtilitySlot,
                containers.get(3),
                "utility"
            );
            requireActiveSlot(
                activeToolsSlot,
                containers.get(4),
                "tools"
            );
        } else {
            if (unavailableReason.isEmpty()) {
                throw new IllegalArgumentException(
                    "unavailable inventory must carry a reason"
                );
            }
            if (!containers.isEmpty()) {
                throw new IllegalArgumentException(
                    "unavailable inventory cannot carry containers"
                );
            }
        }
    }

    public static NativeInventoryFrame unavailable(String reason) {
        return new NativeInventoryFrame(
            false,
            reason,
            -1,
            -1,
            -1,
            List.of()
        );
    }

    private static void requireActiveSlot(
        int slot,
        Container container,
        String name
    ) {
        if (!container.available() && slot != -1) {
            throw new IllegalArgumentException(
                "active " + name + " slot must be -1 when unavailable"
            );
        }
        if (
            container.available()
                && (slot < -1 || slot >= container.capacity())
        ) {
            throw new IllegalArgumentException(
                "active " + name + " slot is outside native capacity"
            );
        }
    }

    public record Container(
        String name,
        int sectionId,
        boolean available,
        String unavailableReason,
        int capacity,
        List<Slot> occupiedSlots
    ) {
        public Container {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException(
                    "container name must be nonempty"
                );
            }
            unavailableReason = unavailableReason == null
                ? ""
                : unavailableReason;
            if (capacity < 0) {
                throw new IllegalArgumentException(
                    "container capacity must be nonnegative"
                );
            }
            occupiedSlots = occupiedSlots == null
                ? List.of()
                : List.copyOf(occupiedSlots);
            if (!available) {
                if (unavailableReason.isEmpty()) {
                    throw new IllegalArgumentException(
                        "unavailable container must carry a reason"
                    );
                }
                if (capacity != 0 || !occupiedSlots.isEmpty()) {
                    throw new IllegalArgumentException(
                        "unavailable container cannot carry contents"
                    );
                }
            } else {
                if (!unavailableReason.isEmpty()) {
                    throw new IllegalArgumentException(
                        "available container cannot carry an unavailable reason"
                    );
                }
                Set<Integer> seen = new HashSet<>();
                int previous = -1;
                for (Slot slot : occupiedSlots) {
                    if (slot.slot() >= capacity) {
                        throw new IllegalArgumentException(
                            "occupied slot is outside native capacity"
                        );
                    }
                    if (!seen.add(slot.slot())) {
                        throw new IllegalArgumentException(
                            "occupied slot is duplicated"
                        );
                    }
                    if (slot.slot() <= previous) {
                        throw new IllegalArgumentException(
                            "occupied slots must be in ascending order"
                        );
                    }
                    previous = slot.slot();
                }
            }
        }

        public static Container unavailable(
            String name,
            int sectionId,
            String reason
        ) {
            return new Container(
                name,
                sectionId,
                false,
                reason,
                0,
                List.of()
            );
        }
    }

    public record Slot(
        int slot,
        String itemId,
        int itemRuntimeIndex,
        int quantity,
        double durability,
        double maxDurability,
        boolean metadataPresent
    ) {
        public Slot {
            if (slot < 0) {
                throw new IllegalArgumentException(
                    "slot must be nonnegative"
                );
            }
            if (itemId == null || itemId.isBlank()) {
                throw new IllegalArgumentException(
                    "item id must be nonempty"
                );
            }
            if (itemRuntimeIndex < 0) {
                throw new IllegalArgumentException(
                    "item runtime index must be nonnegative"
                );
            }
            if (quantity <= 0) {
                throw new IllegalArgumentException(
                    "quantity must be positive"
                );
            }
            if (
                !Double.isFinite(durability)
                    || !Double.isFinite(maxDurability)
                    || durability < 0.0
                    || maxDurability < 0.0
                    || durability > maxDurability
            ) {
                throw new IllegalArgumentException(
                    "durability must be finite and within maximum"
                );
            }
        }
    }
}
