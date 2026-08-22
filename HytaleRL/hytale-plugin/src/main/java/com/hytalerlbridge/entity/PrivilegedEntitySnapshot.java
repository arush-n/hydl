package com.hytalerlbridge.entity;

import java.util.ArrayList;
import java.util.List;

/**
 * Complete bounded native entity enumeration.
 *
 * <p>Overflow is fail-closed: a response carries the exact matching count
 * and no entity rows. Non-overflow rows are strictly ordered by their
 * lossless UUID bytes.</p>
 */
public record PrivilegedEntitySnapshot(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    PrivilegedEntityQuery query,
    int totalMatching,
    boolean overflow,
    List<PrivilegedEntityRow> rows
) {
    public static final String SCHEMA =
        "hytalerl_privileged_entity_snapshot_v1";
    public static final int VERSION = 1;

    public PrivilegedEntitySnapshot {
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
                    "Overflow snapshots must not emit partial rows"
                );
            }
        } else {
            if (rows.size() != totalMatching) {
                throw new IllegalArgumentException(
                    "Non-overflow snapshots must emit every matching row"
                );
            }
            PrivilegedEntityRow previous = null;
            for (PrivilegedEntityRow row : rows) {
                if (row == null) {
                    throw new IllegalArgumentException(
                        "Privileged entity rows must not be null"
                    );
                }
                if (
                    !query.contains(
                        row.positionX(),
                        row.positionY(),
                        row.positionZ()
                    )
                ) {
                    throw new IllegalArgumentException(
                        "Privileged entity row lies outside requested bounds"
                    );
                }
                if (
                    previous != null
                        && PrivilegedEntityRow.UUID_BYTE_ORDER.compare(
                            previous,
                            row
                        ) >= 0
                ) {
                    throw new IllegalArgumentException(
                        "Privileged entity UUIDs must be unique and ordered"
                    );
                }
                previous = row;
            }
        }
    }

    public static PrivilegedEntitySnapshot fromMatches(
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed,
        PrivilegedEntityQuery query,
        List<PrivilegedEntityRow> matches
    ) {
        if (matches == null) {
            throw new IllegalArgumentException("matches must be present");
        }
        int total = matches.size();
        if (total > query.capacity()) {
            return new PrivilegedEntitySnapshot(
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
        List<PrivilegedEntityRow> ordered = new ArrayList<>(matches);
        ordered.sort(PrivilegedEntityRow.UUID_BYTE_ORDER);
        return new PrivilegedEntitySnapshot(
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
