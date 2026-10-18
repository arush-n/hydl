package com.hytalerlbridge.entity;

import java.util.List;

/**
 * Complete native NPC enumeration with palette-independent role asset IDs.
 *
 * <p>This is additive to the spatial-only v1 endpoint. It deliberately
 * carries no inferred behavior or collision certificate.</p>
 */
public record PrivilegedNpcSnapshot(
    PrivilegedEntitySnapshot spatial,
    List<PrivilegedNpcGeometryRow> geometryRows
) {

    public static final String SCHEMA =
        "hytalerl_privileged_npc_snapshot_v2";
    public static final int VERSION = 2;
    public static final String COMPONENT_FILTER =
        "npc_uuid_transform_velocity_role_geometry_v2";

    public PrivilegedNpcSnapshot {
        if (spatial == null) {
            throw new IllegalArgumentException("spatial must be present");
        }
        geometryRows = List.copyOf(geometryRows);
        if (geometryRows.size() != spatial.emittedCount()) {
            throw new IllegalArgumentException(
                "NPC geometry rows must match emitted spatial rows"
            );
        }
        for (PrivilegedEntityRow row : spatial.rows()) {
            if (row.typeAssetId().isBlank()) {
                throw new IllegalArgumentException(
                    "Every NPC row must carry a role asset ID"
                );
            }
        }
        for (int index = 0; index < geometryRows.size(); index++) {
            PrivilegedEntityRow spatialRow = spatial.rows().get(index);
            PrivilegedEntityRow geometrySpatial = geometryRows.get(index).spatial();
            if (
                spatialRow.uuidMostSignificantBits()
                    != geometrySpatial.uuidMostSignificantBits()
                    || spatialRow.uuidLeastSignificantBits()
                        != geometrySpatial.uuidLeastSignificantBits()
            ) {
                throw new IllegalArgumentException(
                    "NPC geometry rows must align with UUID-sorted spatial rows"
                );
            }
        }
    }

    public static PrivilegedNpcSnapshot fromMatches(
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed,
        PrivilegedEntityQuery query,
        List<PrivilegedNpcGeometryRow> matches
    ) {
        List<PrivilegedEntityRow> spatialMatches = matches.stream()
            .map(PrivilegedNpcGeometryRow::spatial)
            .toList();
        PrivilegedEntitySnapshot spatial = PrivilegedEntitySnapshot.fromMatches(
                serverVersion,
                worldName,
                worldgenProvider,
                worldgenVersion,
                seed,
                query,
                spatialMatches
            );
        List<PrivilegedNpcGeometryRow> orderedGeometry = spatial.overflow()
            ? List.of()
            : matches.stream()
                .sorted((left, right) -> PrivilegedEntityRow.UUID_BYTE_ORDER.compare(
                    left.spatial(),
                    right.spatial()
                ))
                .toList();
        return new PrivilegedNpcSnapshot(
            spatial,
            orderedGeometry
        );
    }
}
