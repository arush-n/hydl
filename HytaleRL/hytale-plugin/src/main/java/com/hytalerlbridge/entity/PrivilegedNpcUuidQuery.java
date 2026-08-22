package com.hytalerlbridge.entity;

import java.util.UUID;

/** Exact NPC identity lookup constrained by one world-space AABB. */
public record PrivilegedNpcUuidQuery(
    String componentFilter,
    UUID npcUuid,
    double[] bounds
) {

    public PrivilegedNpcUuidQuery {
        if (!PrivilegedNpcSnapshot.COMPONENT_FILTER.equals(componentFilter)) {
            throw new IllegalArgumentException(
                "Unsupported privileged NPC component filter"
            );
        }
        if (npcUuid == null) {
            throw new IllegalArgumentException(
                "Privileged NPC UUID must be present"
            );
        }
        bounds = bounds == null ? new double[0] : bounds.clone();
        new PrivilegedEntityQuery(
            PrivilegedEntityQuery.COMPONENT_FILTER,
            bounds,
            1
        );
    }

    @Override
    public double[] bounds() {
        return bounds.clone();
    }

    public PrivilegedEntityQuery spatialQuery() {
        return new PrivilegedEntityQuery(
            PrivilegedEntityQuery.COMPONENT_FILTER,
            bounds,
            1
        );
    }
}
