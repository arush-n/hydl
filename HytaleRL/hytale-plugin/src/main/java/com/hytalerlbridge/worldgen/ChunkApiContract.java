package com.hytalerlbridge.worldgen;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Runtime-derived fixed-shape contract for Hytale 0.5.7 chunk storage.
 *
 * <p>Reflection is intentional: Java must not inline {@code static final}
 * values from the compile-time server dependency. A changed runtime API fails
 * before any capture can be represented as a valid JAX region.</p>
 */
public final class ChunkApiContract {

    public static final String SCHEMA = "hytalerl_chunk_api_v1";
    public static final int VERSION = 1;
    public static final String API_CLASS =
        "com.hypixel.hytale.math.util.ChunkUtil";

    private final int bits;
    private final int size;
    private final int sizeSquared;
    private final int sectionVolume;
    private final int heightSections;
    private final int height;
    private final int minY;
    private final int minChunkCoordinate;
    private final int maxChunkCoordinate;

    private ChunkApiContract(
        int bits,
        int size,
        int sizeSquared,
        int sectionVolume,
        int heightSections,
        int height,
        int minY,
        int minChunkCoordinate,
        int maxChunkCoordinate
    ) {
        this.bits = bits;
        this.size = size;
        this.sizeSquared = sizeSquared;
        this.sectionVolume = sectionVolume;
        this.heightSections = heightSections;
        this.height = height;
        this.minY = minY;
        this.minChunkCoordinate = minChunkCoordinate;
        this.maxChunkCoordinate = maxChunkCoordinate;
    }

    /** Load, exhaustively verify, and version-gate the runtime ChunkUtil API. */
    public static ChunkApiContract load057() {
        try {
            Class<?> chunkUtil = Class.forName(API_CLASS);
            ChunkApiContract contract = new ChunkApiContract(
                readInt(chunkUtil, "BITS"),
                readInt(chunkUtil, "SIZE"),
                readInt(chunkUtil, "SIZE_2"),
                readInt(chunkUtil, "SIZE_BLOCKS"),
                readInt(chunkUtil, "HEIGHT_SECTIONS"),
                readInt(chunkUtil, "HEIGHT"),
                readInt(chunkUtil, "MIN_Y"),
                readInt(chunkUtil, "MIN_CHUNK_COORD"),
                readInt(chunkUtil, "MAX_CHUNK_COORD")
            );
            contract.requireSupported057();
            contract.verifyRuntimeMethods(chunkUtil);
            return contract;
        } catch (ReflectiveOperationException exception) {
            throw new IllegalStateException(
                "Unable to inspect runtime Hytale ChunkUtil",
                exception
            );
        }
    }

    public Map<String, Object> manifest(String serverVersion) {
        if (!"0.5.7".equals(serverVersion)) {
            throw new IllegalArgumentException(
                "Chunk contract is certified only for Hytale 0.5.7"
            );
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("schema", SCHEMA);
        result.put("version", VERSION);
        result.put("server_version", serverVersion);
        result.put("api_class", API_CLASS);
        result.put("bits", bits);
        result.put("size", size);
        result.put("size_squared", sizeSquared);
        result.put("section_volume", sectionVolume);
        result.put("height_sections", heightSections);
        result.put("height", height);
        result.put("min_y", minY);
        result.put("min_chunk_coordinate", minChunkCoordinate);
        result.put("max_chunk_coordinate", maxChunkCoordinate);
        result.put("local_coordinate", "arithmetic_shift_and_mask");
        result.put("section_index_order", "y_z_x");
        return result;
    }

    public int chunkCoordinate(int blockCoordinate) {
        return Math.floorDiv(blockCoordinate, size);
    }

    public int localCoordinate(int blockCoordinate) {
        return Math.floorMod(blockCoordinate, size);
    }

    public int sectionIndex(int localX, int localY, int localZ) {
        if (
            localX < 0 || localX >= size
                || localY < 0 || localY >= size
                || localZ < 0 || localZ >= size
        ) {
            throw new IllegalArgumentException(
                "Section coordinates must be in [0, " + size + ")"
            );
        }
        return localY * sizeSquared + localZ * size + localX;
    }

    public int size() {
        return size;
    }

    public int height() {
        return height;
    }

    public int heightSections() {
        return heightSections;
    }

    private void requireSupported057() {
        if (
            bits != 5
                || size != 32
                || sizeSquared != 1024
                || sectionVolume != 32768
                || heightSections != 10
                || height != 320
                || minY != 0
                || minChunkCoordinate != -67108864
                || maxChunkCoordinate != 67108863
                || height != size * heightSections
        ) {
            throw new IllegalStateException(
                "Unsupported Hytale ChunkUtil shape: "
                    + bits + "/" + size + "/" + sizeSquared + "/"
                    + sectionVolume + "/" + heightSections + "/"
                    + height + "/" + minY + "/"
                    + minChunkCoordinate + "/" + maxChunkCoordinate
            );
        }
    }

    private void verifyRuntimeMethods(Class<?> chunkUtil)
        throws ReflectiveOperationException {
        Method chunkCoordinate = chunkUtil.getMethod(
            "chunkCoordinate",
            int.class
        );
        Method localCoordinate = chunkUtil.getMethod(
            "localCoordinate",
            long.class
        );
        for (int value = -2 * size - 1; value <= 2 * size + 1; value++) {
            requireEqual(
                Math.floorDiv(value, size),
                invokeInt(chunkCoordinate, value),
                "chunkCoordinate(" + value + ")"
            );
            requireEqual(
                Math.floorMod(value, size),
                invokeInt(localCoordinate, (long) value),
                "localCoordinate(" + value + ")"
            );
        }
        int[] boundaryCoordinates = {
            Integer.MIN_VALUE,
            Integer.MIN_VALUE + size - 1,
            -size,
            -1,
            0,
            size - 1,
            size,
            Integer.MAX_VALUE - size + 1,
            Integer.MAX_VALUE,
        };
        for (int value : boundaryCoordinates) {
            requireEqual(
                Math.floorDiv(value, size),
                invokeInt(chunkCoordinate, value),
                "chunkCoordinate(" + value + ")"
            );
            requireEqual(
                Math.floorMod(value, size),
                invokeInt(localCoordinate, (long) value),
                "localCoordinate(" + value + ")"
            );
        }

        Method indexBlock = chunkUtil.getMethod(
            "indexBlock",
            int.class,
            int.class,
            int.class
        );
        Method xFromIndex = chunkUtil.getMethod("xFromIndex", int.class);
        Method yFromIndex = chunkUtil.getMethod("yFromIndex", int.class);
        Method zFromIndex = chunkUtil.getMethod("zFromIndex", int.class);
        boolean[] visited = new boolean[sectionVolume];
        for (int y = 0; y < size; y++) {
            for (int z = 0; z < size; z++) {
                for (int x = 0; x < size; x++) {
                    int expected = sectionIndex(x, y, z);
                    int actual = invokeInt(indexBlock, x, y, z);
                    requireEqual(expected, actual, "indexBlock");
                    requireEqual(x, invokeInt(xFromIndex, actual), "xFromIndex");
                    requireEqual(y, invokeInt(yFromIndex, actual), "yFromIndex");
                    requireEqual(z, invokeInt(zFromIndex, actual), "zFromIndex");
                    if (visited[actual]) {
                        throw new IllegalStateException(
                            "ChunkUtil indexBlock is not one-to-one"
                        );
                    }
                    visited[actual] = true;
                }
            }
        }
    }

    private static int readInt(Class<?> owner, String name)
        throws ReflectiveOperationException {
        Field field = owner.getField(name);
        return field.getInt(null);
    }

    private static int invokeInt(Method method, Object... arguments)
        throws ReflectiveOperationException {
        return ((Number) method.invoke(null, arguments)).intValue();
    }

    private static void requireEqual(int expected, int actual, String label) {
        if (expected != actual) {
            throw new IllegalStateException(
                label + " mismatch: expected " + expected + ", got " + actual
            );
        }
    }
}
