package com.hytalerlbridge.worldgen;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Bounded exact snapshots of live mutable World cells.
 *
 * <p>An unavailable cell is distinct from exact air. The bridge never maps an
 * unloaded chunk or an out-of-domain Y coordinate to an empty block.</p>
 */
public record NativeMutableBlockCells(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    List<Cell> cells
) {

    public static final String SCHEMA =
        "hytalerl_native_mutable_block_cells_v1";
    public static final int VERSION = 1;
    public static final int MAX_SAMPLES = 64;

    public NativeMutableBlockCells {
        serverVersion = requireText(serverVersion, "serverVersion");
        worldName = requireText(worldName, "worldName");
        worldgenProvider = requireText(worldgenProvider, "worldgenProvider");
        worldgenVersion = requireText(worldgenVersion, "worldgenVersion");
        cells = cells == null ? List.of() : List.copyOf(cells);
        if (cells.isEmpty() || cells.size() > MAX_SAMPLES) {
            throw new IllegalArgumentException(
                "Mutable block cell capture must contain 1.."
                    + MAX_SAMPLES
                    + " cells"
            );
        }
        Set<Position> distinct = new HashSet<>();
        for (Cell cell : cells) {
            int[] position = cell.position();
            if (!distinct.add(new Position(
                position[0],
                position[1],
                position[2]
            ))) {
                throw new IllegalArgumentException(
                    "Mutable block cell positions must be distinct"
                );
            }
        }
    }

    /** One requested cell; {@code row} exists if and only if it was readable. */
    public record Cell(
        int[] position,
        boolean available,
        NativeMutableBlockEvidence.Row row
    ) {

        public Cell {
            position = position == null ? new int[0] : position.clone();
            if (position.length != 3) {
                throw new IllegalArgumentException(
                    "Mutable block cell position must be one XYZ triple"
                );
            }
            if (available != (row != null)) {
                throw new IllegalArgumentException(
                    "Mutable block cell availability must match its row"
                );
            }
            if (row != null && !"snapshot".equals(row.phase())) {
                throw new IllegalArgumentException(
                    "Mutable block cell rows must use the snapshot phase"
                );
            }
        }

        @Override
        public int[] position() {
            return position.clone();
        }
    }

    private record Position(int x, int y, int z) {}

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " cannot be blank");
        }
        return value;
    }
}
