package com.hytalerlbridge.geometry;

import java.util.Arrays;

/**
 * One non-empty cell in the canonical local geometry grid.
 *
 * <p>Collision boxes are local to the cell and flattened as
 * {@code [minX,minY,minZ,maxX,maxY,maxZ,...]}.</p>
 */
public record GeometryCell(
    int dx,
    int dy,
    int dz,
    int runtimeBlockId,
    int runtimeFluidId,
    int shapeId,
    int rotationIndex,
    int flags,
    int fluidLevel,
    double fluidFillHeight,
    int supportValue,
    int blockDamage,
    int fluidDamage,
    double friction,
    double drag,
    double horizontalSpeedMultiplier,
    double jumpForceMultiplier,
    double climbUpSpeedMultiplier,
    double climbDownSpeedMultiplier,
    double climbLateralSpeedMultiplier,
    double terminalVelocityModifier,
    double bounceVelocity,
    double fluidSwimUpSpeed,
    double fluidSwimDownSpeed,
    double fluidSinkSpeed,
    double fluidHorizontalSpeedMultiplier,
    double fluidFieldOfViewMultiplier,
    double fluidEntryVelocityMultiplier,
    double[] collisionBoxes
) {
    public GeometryCell {
        GeometryContract.cellIndex(dx, dy, dz);
        collisionBoxes = collisionBoxes == null
            ? new double[0]
            : collisionBoxes.clone();
        if (collisionBoxes.length % 6 != 0) {
            throw new IllegalArgumentException(
                "collisionBoxes length must be divisible by six"
            );
        }
        if (collisionBoxes.length / 6 > GeometryContract.MAX_DETAIL_BOXES_0_5_7) {
            throw new IllegalArgumentException(
                "Hytale 0.5.7 geometry cell exceeds the certified nine-box capacity"
            );
        }
        if (!Double.isFinite(fluidFillHeight)
            || fluidFillHeight < 0.0
            || fluidFillHeight > 1.0) {
            throw new IllegalArgumentException(
                "fluidFillHeight must be finite and in [0, 1]"
            );
        }
    }

    @Override
    public double[] collisionBoxes() {
        return collisionBoxes.clone();
    }

    public int collisionBoxCount() {
        return collisionBoxes.length / 6;
    }

    @Override
    public String toString() {
        return "GeometryCell[dx=" + dx
            + ", dy=" + dy
            + ", dz=" + dz
            + ", runtimeBlockId=" + runtimeBlockId
            + ", shapeId=" + shapeId
            + ", rotationIndex=" + rotationIndex
            + ", flags=" + flags
            + ", collisionBoxes=" + Arrays.toString(collisionBoxes)
            + "]";
    }
}
