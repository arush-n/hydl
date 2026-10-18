package com.hytalerlbridge.worldgen;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Bounded allowlist query for authored WorldGen V2 structure markers. */
public record WorldgenStructureMarkerQuery(
    String componentFilter,
    double[] bounds,
    int capacity,
    List<String> markerAssetIds
) {
    public static final String COMPONENT_FILTER =
        "uuid_transform_worldgen_id_from_prefab_instance_spawn_marker_v1";
    public static final String MARKER_IDENTITY =
        "SpawnMarkerEntity.getSpawnMarkerId";
    public static final int MAX_CAPACITY = 256;
    public static final int MAX_MARKER_ASSET_IDS = 256;

    public WorldgenStructureMarkerQuery {
        if (!COMPONENT_FILTER.equals(componentFilter)) {
            throw new IllegalArgumentException(
                "Unsupported WorldGen structure marker component filter"
            );
        }
        bounds = bounds == null ? new double[0] : bounds.clone();
        if (bounds.length != 6) {
            throw new IllegalArgumentException(
                "WorldGen structure marker bounds must contain min/max XYZ"
            );
        }
        for (double value : bounds) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                    "WorldGen structure marker bounds must be finite"
                );
            }
        }
        if (
            bounds[0] >= bounds[3]
                || bounds[1] >= bounds[4]
                || bounds[2] >= bounds[5]
        ) {
            throw new IllegalArgumentException(
                "WorldGen structure marker bounds must be a positive AABB"
            );
        }
        if (capacity < 1 || capacity > MAX_CAPACITY) {
            throw new IllegalArgumentException(
                "WorldGen structure marker capacity must be 1.."
                    + MAX_CAPACITY
            );
        }
        if (
            markerAssetIds == null
                || markerAssetIds.isEmpty()
                || markerAssetIds.size() > MAX_MARKER_ASSET_IDS
        ) {
            throw new IllegalArgumentException(
                "WorldGen structure marker asset IDs must contain 1.."
                    + MAX_MARKER_ASSET_IDS
                    + " rows"
            );
        }
        Set<String> unique = new HashSet<>();
        List<String> canonical = new ArrayList<>(markerAssetIds.size());
        for (String markerAssetId : markerAssetIds) {
            if (markerAssetId == null || markerAssetId.isBlank()) {
                throw new IllegalArgumentException(
                    "WorldGen structure marker asset IDs must be non-empty"
                );
            }
            if (!unique.add(markerAssetId)) {
                throw new IllegalArgumentException(
                    "WorldGen structure marker asset IDs must be unique"
                );
            }
            canonical.add(markerAssetId);
        }
        canonical.sort(String::compareTo);
        markerAssetIds = List.copyOf(canonical);
    }

    @Override
    public double[] bounds() {
        return bounds.clone();
    }

    public boolean contains(double x, double y, double z) {
        return x >= bounds[0]
            && y >= bounds[1]
            && z >= bounds[2]
            && x < bounds[3]
            && y < bounds[4]
            && z < bounds[5];
    }

    public boolean accepts(String markerAssetId) {
        return markerAssetIds.contains(markerAssetId);
    }
}
