package com.hytalerlbridge.worldgen;

import com.hytalerlbridge.geometry.GeometryContract;

/** Exact cell-local native collision boxes for one deduplicated shape. */
public record RegionShapePaletteEntry(double[] collisionBoxes) {

    public RegionShapePaletteEntry {
        collisionBoxes = collisionBoxes == null
            ? new double[0]
            : collisionBoxes.clone();
        if (collisionBoxes.length % 6 != 0) {
            throw new IllegalArgumentException(
                "Region shape boxes must be flattened six-value boxes"
            );
        }
        if (
            collisionBoxes.length / 6
                > GeometryContract.MAX_DETAIL_BOXES_0_5_7
        ) {
            throw new IllegalArgumentException(
                "Region shape exceeds the certified nine-box capacity"
            );
        }
        for (int offset = 0; offset < collisionBoxes.length; offset += 6) {
            for (int axis = 0; axis < 6; axis++) {
                if (!Double.isFinite(collisionBoxes[offset + axis])) {
                    throw new IllegalArgumentException(
                        "Region shape contains a non-finite coordinate"
                    );
                }
            }
            for (int axis = 0; axis < 3; axis++) {
                if (
                    collisionBoxes[offset + axis]
                        > collisionBoxes[offset + axis + 3]
                ) {
                    throw new IllegalArgumentException(
                        "Region shape contains an inverted box"
                    );
                }
            }
        }
    }

    @Override
    public double[] collisionBoxes() {
        return collisionBoxes.clone();
    }
}
