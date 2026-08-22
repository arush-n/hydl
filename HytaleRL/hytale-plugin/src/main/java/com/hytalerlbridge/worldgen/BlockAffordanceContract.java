package com.hytalerlbridge.worldgen;

import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockBreakingDropType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockGathering;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.SoftBlockDropType;
import com.hypixel.hytale.server.core.asset.type.item.config.Item;
import java.util.Arrays;
import java.util.List;

/** Portable policy affordances derived from stable 0.5.7 asset semantics. */
public final class BlockAffordanceContract {

    public static final String SCHEMA =
        "hytalerl_block_affordance_dictionary_v1";
    public static final int VERSION = 1;
    public static final String DICTIONARY_SHA256 =
        "4f2104af54536d75d357c44af972422803e2e51035ec4a19545f3c5089477e24";

    public static final int TAG_BREAKABLE = 1 << 0;
    public static final int TAG_ORE = 1 << 1;
    public static final int TAG_HARVESTABLE = 1 << 2;
    public static final int TAG_SOFT = 1 << 3;
    public static final int TAG_HAS_BREAK_DROP = 1 << 4;
    public static final int TAG_USE_DEFAULT_DROP_WHEN_PLACED = 1 << 5;
    public static final int TAG_DOOR = 1 << 6;
    public static final int TAG_STATEFUL = 1 << 7;
    public static final int KNOWN_TAG_MASK = (1 << 8) - 1;

    public static final List<String> GATHER_TYPES = List.of(
        "none",
        "Benches",
        "Branches",
        "Cloths",
        "DungeonBlocks",
        "OreAdamantite",
        "OreCobalt",
        "OreCopper",
        "OreGold",
        "OreIron",
        "OreMithril",
        "OreSilver",
        "OreThorium",
        "Pickaxe_Tier0",
        "Rocks",
        "SoftBlocks",
        "SoftWoods",
        "Soils",
        "Unbreakable",
        "VolcanicRocks",
        "Woods"
    );

    private BlockAffordanceContract() {}

    public static Value resolve(BlockType blockType) {
        if (blockType == null || blockType == BlockType.EMPTY) {
            return Value.empty();
        }
        BlockGathering gathering = blockType.getGathering();
        BlockBreakingDropType breaking = gathering == null
            ? null
            : gathering.getBreaking();
        SoftBlockDropType soft = gathering == null
            ? null
            : gathering.getSoft();
        String gatherType = breaking == null
                || breaking.getGatherType() == null
            ? GATHER_TYPES.get(0)
            : breaking.getGatherType();
        int gatherTypeIndex = GATHER_TYPES.indexOf(gatherType);
        int requiredQuality = breaking == null ? 0 : breaking.getQuality();
        if (
            gatherTypeIndex < 0
                || requiredQuality < 0
                || requiredQuality > Short.MAX_VALUE
        ) {
            return Value.invalid();
        }

        int tags = 0;
        boolean toolBreakable = breaking != null
            && breaking.getGatherType() != null
            && !"Unbreakable".equals(gatherType);
        boolean softBreakable = soft != null;
        if (toolBreakable || softBreakable) {
            tags |= TAG_BREAKABLE;
        }
        if (isOre(blockType, gatherType)) tags |= TAG_ORE;
        if (gathering != null && gathering.isHarvestable()) {
            tags |= TAG_HARVESTABLE;
        }
        if (softBreakable) tags |= TAG_SOFT;
        boolean toolDrop = toolBreakable
            && breaking.getQuantity() > 0
            && (
                isDrop(breaking.getItemId())
                    || isDrop(breaking.getDropListId())
                    || blockType.getItem() != null
            );
        boolean softDrop = softBreakable
            && (
                isDrop(soft.getItemId())
                    || isDrop(soft.getDropListId())
                    || blockType.getItem() != null
            );
        if (toolDrop || softDrop) {
            tags |= TAG_HAS_BREAK_DROP;
        }
        if (
            gathering != null
                && gathering.shouldUseDefaultDropWhenPlaced()
        ) {
            tags |= TAG_USE_DEFAULT_DROP_WHEN_PLACED;
        }
        if (blockType.isDoor()) tags |= TAG_DOOR;
        if (blockType.isState() || blockType.getState() != null) {
            tags |= TAG_STATEFUL;
        }
        return new Value(true, tags, gatherTypeIndex, requiredQuality);
    }

    private static boolean isOre(BlockType blockType, String gatherType) {
        if (gatherType.startsWith("Ore")) return true;
        Item item = blockType.getItem();
        String[] categories = item == null ? null : item.getCategories();
        return categories != null
            && Arrays.asList(categories).contains("Blocks.Ores");
    }

    private static boolean isDrop(String value) {
        return value != null
            && !value.isBlank()
            && !"Empty".equals(value);
    }

    /** One portable cell value; invalid values are never policy-visible. */
    public record Value(
        boolean valid,
        int tags,
        int gatherTypeIndex,
        int requiredToolQuality
    ) {
        public Value {
            if (
                (tags & ~KNOWN_TAG_MASK) != 0
                    || gatherTypeIndex < 0
                    || gatherTypeIndex >= GATHER_TYPES.size()
                    || requiredToolQuality < 0
                    || requiredToolQuality > Short.MAX_VALUE
            ) {
                throw new IllegalArgumentException(
                    "Block affordance value is outside its dictionary"
                );
            }
            if (!valid && (
                tags != 0
                    || gatherTypeIndex != 0
                    || requiredToolQuality != 0
            )) {
                throw new IllegalArgumentException(
                    "Invalid block affordances must be canonical zero"
                );
            }
        }

        static Value empty() {
            return new Value(false, 0, 0, 0);
        }

        static Value invalid() {
            return empty();
        }
    }
}
