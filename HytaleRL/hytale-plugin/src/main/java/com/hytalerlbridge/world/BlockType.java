package com.hytalerlbridge.world;

/**
 * Block types in the simulated Hytale world.
 *
 * Hytale has ~427 block types in total. This enum represents the subset
 * relevant to the RL training environment: natural terrain, ores following
 * Hytale's progression (Copper→Iron→Thorium→Cobalt→Adamantite→Mithril),
 * wood from Zone 1 (Emerald Wilds), and buildable structures.
 */
public enum BlockType {
    AIR(0),
    DIRT(1),
    STONE(2),
    WOOD(3),           // generic tree trunk (Zone 1 softwood)
    LEAVES(4),
    SAND(5),
    WATER(6),
    GRASS(7),
    QUARTZITE(8),      // mined from stone (Hytale equivalent of worked stone)
    SOFTWOOD_PLANKS(9),// crafted from logs (Zone 1 primary wood)
    WORKBENCH(10),
    FURNACE(11),
    CHEST(12),
    DOOR(13),
    TORCH(14),
    IRON_ORE(15),
    COPPER_ORE(16),
    ADAMANTITE_ORE(17),
    LOG(18),
    GLASS(19),
    FENCE(20),
    ROOF_BLOCK(21),    // Hytale roof block (supports angled placement)
    CRUDE_BEDROLL(22), // crafted from Plant Fiber + Light Hide
    THORIUM_ORE(23),
    COBALT_ORE(24),
    MITHRIL_ORE(25),
    RESIN(26),         // dropped when felling trees, used for torches
    PLANT_FIBER(27),   // gathered from grass/plants
    STICKS(28);        // dropped from trees, crafting ingredient

    private final int id;

    BlockType(int id) {
        this.id = id;
    }

    public int id() {
        return id;
    }

    public boolean isSolid() {
        return this != AIR && this != WATER && this != TORCH && this != RESIN
            && this != PLANT_FIBER && this != STICKS;
    }

    public static BlockType fromId(int id) {
        for (BlockType bt : values()) {
            if (bt.id == id) return bt;
        }
        return AIR;
    }

    /** Number of block type IDs (for observation space sizing). */
    public static int count() {
        return values().length;
    }
}
