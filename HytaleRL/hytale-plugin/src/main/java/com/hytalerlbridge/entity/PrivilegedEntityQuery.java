package com.hytalerlbridge.entity;

/**
 * Versioned, bounded request for an authoritative entity snapshot.
 *
 * <p>Bounds are world-space and half-open: {@code [min, max)} on every axis.
 * The filter name is a protocol identifier, never a caller-supplied Java
 * class name.</p>
 */
public record PrivilegedEntityQuery(
    String componentFilter,
    double[] bounds,
    int capacity
) {
    public static final String COMPONENT_FILTER =
        "uuid_transform_velocity_v1";
    public static final int MAX_CAPACITY = 256;

    public PrivilegedEntityQuery {
        if (!COMPONENT_FILTER.equals(componentFilter)) {
            throw new IllegalArgumentException(
                "Unsupported privileged entity component filter"
            );
        }
        bounds = bounds == null ? new double[0] : bounds.clone();
        if (bounds.length != 6) {
            throw new IllegalArgumentException(
                "Privileged entity bounds must contain min/max XYZ"
            );
        }
        for (double value : bounds) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                    "Privileged entity bounds must be finite"
                );
            }
        }
        if (
            bounds[0] >= bounds[3]
                || bounds[1] >= bounds[4]
                || bounds[2] >= bounds[5]
        ) {
            throw new IllegalArgumentException(
                "Privileged entity bounds must be a positive AABB"
            );
        }
        if (capacity < 1 || capacity > MAX_CAPACITY) {
            throw new IllegalArgumentException(
                "Privileged entity capacity must be 1.."
                    + MAX_CAPACITY
            );
        }
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
}
