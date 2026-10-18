package com.hytalerlbridge.worldgen;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Complete fail-closed enumeration of requested authored structure markers. */
public record WorldgenStructureMarkerSnapshot(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    WorldgenStructureMarkerQuery query,
    int totalMatching,
    boolean overflow,
    List<WorldgenStructureMarkerRow> rows
) {
    public static final String SCHEMA =
        "hytalerl_worldgen_v2_structure_markers_v1";
    public static final int VERSION = 1;

    public WorldgenStructureMarkerSnapshot {
        requireText(serverVersion, "serverVersion");
        requireText(worldName, "worldName");
        requireText(worldgenProvider, "worldgenProvider");
        requireText(worldgenVersion, "worldgenVersion");
        if (query == null) {
            throw new IllegalArgumentException("query must be present");
        }
        if (totalMatching < 0) {
            throw new IllegalArgumentException(
                "totalMatching must be non-negative"
            );
        }
        rows = rows == null ? List.of() : List.copyOf(rows);
        boolean expectedOverflow = totalMatching > query.capacity();
        if (overflow != expectedOverflow) {
            throw new IllegalArgumentException(
                "overflow must exactly reflect totalMatching > capacity"
            );
        }
        if (overflow) {
            if (!rows.isEmpty()) {
                throw new IllegalArgumentException(
                    "Overflow marker snapshots must not emit partial rows"
                );
            }
        } else {
            if (rows.size() != totalMatching) {
                throw new IllegalArgumentException(
                    "Non-overflow marker snapshots must emit every row"
                );
            }
            WorldgenStructureMarkerRow previous = null;
            Set<Long> nativeGroups = new HashSet<>();
            for (WorldgenStructureMarkerRow row : rows) {
                if (row == null) {
                    throw new IllegalArgumentException(
                        "WorldGen structure marker rows must not be null"
                    );
                }
                if (
                    !query.accepts(row.markerAssetId())
                        || !query.contains(
                            row.positionX(),
                            row.positionY(),
                            row.positionZ()
                        )
                ) {
                    throw new IllegalArgumentException(
                        "WorldGen structure marker row lies outside its query"
                    );
                }
                if (
                    previous != null
                        && WorldgenStructureMarkerRow.UUID_BYTE_ORDER.compare(
                            previous,
                            row
                        ) >= 0
                ) {
                    throw new IllegalArgumentException(
                        "WorldGen structure marker UUIDs must be unique and ordered"
                    );
                }
                long group = Integer.toUnsignedLong(row.nativeWorldgenId()) << 32
                    | Integer.toUnsignedLong(row.nativePrefabInstanceId());
                if (!nativeGroups.add(group)) {
                    throw new IllegalArgumentException(
                        "Exactly one authored marker is required per prefab instance"
                    );
                }
                previous = row;
            }
        }
    }

    public static WorldgenStructureMarkerSnapshot fromMatches(
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed,
        WorldgenStructureMarkerQuery query,
        List<WorldgenStructureMarkerRow> matches
    ) {
        if (matches == null) {
            throw new IllegalArgumentException("matches must be present");
        }
        int total = matches.size();
        if (total > query.capacity()) {
            return new WorldgenStructureMarkerSnapshot(
                serverVersion,
                worldName,
                worldgenProvider,
                worldgenVersion,
                seed,
                query,
                total,
                true,
                List.of()
            );
        }
        List<WorldgenStructureMarkerRow> ordered = new ArrayList<>(matches);
        ordered.sort(WorldgenStructureMarkerRow.UUID_BYTE_ORDER);
        return new WorldgenStructureMarkerSnapshot(
            serverVersion,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            query,
            total,
            false,
            ordered
        );
    }

    public int emittedCount() {
        return rows.size();
    }

    private static void requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
    }
}
