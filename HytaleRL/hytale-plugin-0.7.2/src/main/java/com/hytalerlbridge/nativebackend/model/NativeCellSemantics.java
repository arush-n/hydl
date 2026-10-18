package com.hytalerlbridge.nativebackend.model;

import com.hytalerlbridge.geometry.GeometryCell;
import com.hytalerlbridge.worldgen.RegionCellPaletteEntry;

/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record NativeCellSemantics(
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
    double[] movement,
    double[] fluidMovement,
    double[] collisionBoxes
) {
    public NativeCellSemantics {
        movement = movement.clone();
        fluidMovement = fluidMovement.clone();
        collisionBoxes = collisionBoxes.clone();
    }

    @Override
    public double[] movement() {
        return movement.clone();
    }

    @Override
    public double[] fluidMovement() {
        return fluidMovement.clone();
    }

    @Override
    public double[] collisionBoxes() {
        return collisionBoxes.clone();
    }

    public GeometryCell geometryCell(int dx, int dy, int dz) {
        return new GeometryCell(
            dx,
            dy,
            dz,
            runtimeBlockId,
            runtimeFluidId,
            shapeId,
            rotationIndex,
            flags,
            fluidLevel,
            fluidFillHeight,
            supportValue,
            blockDamage,
            fluidDamage,
            movement[0],
            movement[1],
            movement[2],
            movement[3],
            movement[4],
            movement[5],
            movement[6],
            movement[7],
            movement[8],
            fluidMovement[0],
            fluidMovement[1],
            fluidMovement[2],
            fluidMovement[3],
            fluidMovement[4],
            fluidMovement[5],
            collisionBoxes
        );
    }

    public RegionCellPaletteEntry regionEntry(int regionShapeIndex) {
        return new RegionCellPaletteEntry(
            flags,
            regionShapeIndex,
            fluidLevel,
            fluidFillHeight,
            supportValue,
            blockDamage,
            fluidDamage,
            movement,
            fluidMovement
        );
    }
}
