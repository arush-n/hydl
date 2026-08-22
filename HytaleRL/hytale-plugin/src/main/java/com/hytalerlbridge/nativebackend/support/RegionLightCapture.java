package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.server.core.universe.world.chunk.WorldChunk;
import com.hypixel.hytale.server.core.universe.world.chunk.section.BlockSection;
import com.hypixel.hytale.server.core.universe.world.chunk.section.ChunkLightData;
import com.hytalerlbridge.worldgen.NativeRegionLightSection;

/** Stateless capture of one native global-light section. */
public final class RegionLightCapture {

    private RegionLightCapture() {}

    public static NativeRegionLightSection capture(
        WorldChunk chunk,
        int chunkX,
        int chunkZ,
        int sectionY
    ) {
        BlockSection section = chunk
            .getBlockChunk()
            .getSectionAtIndex(sectionY);
        int counter = Short.toUnsignedInt(section.getGlobalChangeCounter());
        ChunkLightData globalLight = section.getGlobalLight();
        int lightChangeId = Short.toUnsignedInt(globalLight.getChangeId());
        if (!isAvailable(section)) {
            return unavailable(
                chunkX,
                chunkZ,
                sectionY,
                NativeRegionLightSection.STATUS_NOT_READY,
                counter,
                lightChangeId
            );
        }

        byte[] payload = new byte[NativeRegionLightSection.LIGHT_BYTES];
        for (int localY = 0; localY < ChunkUtil.SIZE; localY++) {
            for (int localZ = 0; localZ < ChunkUtil.SIZE; localZ++) {
                for (int localX = 0; localX < ChunkUtil.SIZE; localX++) {
                    int index = localY * ChunkUtil.SIZE_2
                        + localZ * ChunkUtil.SIZE
                        + localX;
                    int value = Short.toUnsignedInt(
                        globalLight.getLightRaw(localX, localY, localZ)
                    );
                    int offset = 2 * index;
                    payload[offset] = (byte) (value & 0xff);
                    payload[offset + 1] = (byte) ((value >>> 8) & 0xff);
                }
            }
        }

        int finalCounter = Short.toUnsignedInt(section.getGlobalChangeCounter());
        int finalLightChangeId = Short.toUnsignedInt(
            section.getGlobalLight().getChangeId()
        );
        if (
            !isAvailable(section)
                || finalCounter != counter
                || finalLightChangeId != lightChangeId
                || section.getGlobalLight() != globalLight
        ) {
            return unavailable(
                chunkX,
                chunkZ,
                sectionY,
                NativeRegionLightSection.STATUS_CHANGED,
                finalCounter,
                finalLightChangeId
            );
        }
        return new NativeRegionLightSection(
            chunkX,
            chunkZ,
            sectionY,
            NativeRegionLightSection.STATUS_READY,
            counter,
            lightChangeId,
            payload
        );
    }

    /**
     * ``hasGlobalLight`` alone is true for a fresh section because both the
     * default EMPTY light and the initial counter use change id zero.
     */
    public static boolean isAvailable(BlockSection section) {
        return section.hasGlobalLight()
            && section.getGlobalLight() != ChunkLightData.EMPTY;
    }

    /** Make fresh EMPTY placeholders visible to the native lighting worker. */
    public static void invalidateEmptyPlaceholders(BlockSection section) {
        if (section.getLocalLight() == ChunkLightData.EMPTY) {
            section.invalidateLocalLight();
        } else if (section.getGlobalLight() == ChunkLightData.EMPTY) {
            section.invalidateGlobalLight();
        }
    }

    private static NativeRegionLightSection unavailable(
        int chunkX,
        int chunkZ,
        int sectionY,
        String status,
        int counter,
        int lightChangeId
    ) {
        return new NativeRegionLightSection(
            chunkX,
            chunkZ,
            sectionY,
            status,
            counter,
            lightChangeId,
            new byte[0]
        );
    }
}
