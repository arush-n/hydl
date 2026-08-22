package com.hytalerlbridge.world;

import java.util.*;

/**
 * Sparse voxel world grid for simulated block storage.
 * Uses a HashMap keyed by packed long coordinates for memory efficiency.
 */
public class WorldGrid {

    private final Map<Long, BlockType> blocks = new HashMap<>();
    private final int groundLevel;

    public WorldGrid(int groundLevel) {
        this.groundLevel = groundLevel;
    }

    /** Pack 3D coordinates into a single long key. */
    private static long packKey(int x, int y, int z) {
        return ((long) (x + 30000) << 40) | ((long) (y + 512) << 20) | (long) (z + 30000);
    }

    public BlockType getBlock(int x, int y, int z) {
        // Explicit overrides (including AIR for mined terrain) take precedence.
        BlockType overridden = blocks.get(packKey(x, y, z));
        if (overridden != null) return overridden;

        // Below ground level is dirt/stone by default
        if (y < groundLevel - 3) return BlockType.STONE;
        if (y < groundLevel) return BlockType.DIRT;
        if (y == groundLevel) return BlockType.GRASS;
        return BlockType.AIR;
    }

    public void setBlock(int x, int y, int z, BlockType type) {
        long key = packKey(x, y, z);
        blocks.put(key, type);
    }

    public boolean isSolid(int x, int y, int z) {
        return getBlock(x, y, z).isSolid();
    }

    /** Get all non-air blocks within radius of a position (above ground level). */
    public List<int[]> getBlocksInRadius(int cx, int cy, int cz, int radius) {
        List<int[]> result = new ArrayList<>();
        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dz = -radius; dz <= radius; dz++) {
                    int bx = cx + dx, by = cy + dy, bz = cz + dz;
                    BlockType bt = getBlock(bx, by, bz);
                    if (bt != BlockType.AIR) {
                        result.add(new int[]{dx, dy, dz, bt.id()});
                    }
                }
            }
        }
        return result;
    }

    /** Count placed blocks (non-terrain) above ground. */
    public int getPlacedBlockCount() {
        return (int) blocks.values().stream().filter(type -> type != BlockType.AIR).count();
    }

    /** Get all placed block positions. */
    public Set<Long> getPlacedBlockKeys() {
        Set<Long> keys = new HashSet<>();
        blocks.forEach((key, type) -> {
            if (type != BlockType.AIR) keys.add(key);
        });
        return Collections.unmodifiableSet(keys);
    }

    /** Generate trees at random positions. */
    public void generateTree(Random rng, int tx, int tz) {
        int baseY = groundLevel + 1;
        int height = 4 + rng.nextInt(3);
        // Trunk
        for (int y = baseY; y < baseY + height; y++) {
            setBlock(tx, y, tz, BlockType.LOG);
        }
        // Canopy
        int canopyBase = baseY + height - 2;
        for (int dy = 0; dy < 3; dy++) {
            int radius = dy == 2 ? 1 : 2;
            for (int dx = -radius; dx <= radius; dx++) {
                for (int dz = -radius; dz <= radius; dz++) {
                    if (dx == 0 && dz == 0 && dy < 2) continue; // trunk
                    setBlock(tx + dx, canopyBase + dy, tz + dz, BlockType.LEAVES);
                }
            }
        }
    }

    /** Generate ore deposits below ground. */
    public void generateOre(Random rng, BlockType oreType, int count, int minY, int maxY) {
        for (int i = 0; i < count; i++) {
            int ox = rng.nextInt(60) - 30;
            int oy = minY + rng.nextInt(maxY - minY);
            int oz = rng.nextInt(60) - 30;
            int veinSize = 2 + rng.nextInt(3);
            for (int v = 0; v < veinSize; v++) {
                setBlock(ox + rng.nextInt(2), oy + rng.nextInt(2), oz + rng.nextInt(2), oreType);
            }
        }
    }

    public int getGroundLevel() {
        return groundLevel;
    }

    public void clear() {
        blocks.clear();
    }
}
