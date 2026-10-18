package com.hytalerlbridge.world;

/**
 * Entity types in the simulated Hytale world.
 *
 * Trorks:     Hostile goblin-like creatures native to Zone 1 (Emerald Wilds).
 *             Variants include Hunters (throw axes), Maulers (stone mauls),
 *             and Sentries (throw spears). Simplified here as a single type.
 * Outlanders: Hostile humanoid raiders found across multiple zones.
 * Scaraks:    Hostile arachnid swarm creatures that lurk in caves.
 * Kweebecs:   Peaceful tree-like forest dwellers. Become hostile if the
 *             player wields an axe near their village (not yet simulated).
 */
public enum EntityType {
    TRORK(0),
    OUTLANDER(1),
    SCARAK(2),
    KWEEBEC(3),
    ITEM_DROP(4);

    private final int id;

    EntityType(int id) {
        this.id = id;
    }

    public int id() {
        return id;
    }

    public boolean isHostile() {
        return this == TRORK || this == OUTLANDER || this == SCARAK;
    }

    public static EntityType fromId(int id) {
        for (EntityType et : values()) {
            if (et.id == id) return et;
        }
        return TRORK;
    }
}
