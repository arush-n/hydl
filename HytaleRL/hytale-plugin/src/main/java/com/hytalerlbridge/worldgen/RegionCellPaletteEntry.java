package com.hytalerlbridge.worldgen;

/**
 * One portable immutable-cell entry in a native region section palette.
 *
 * <p>Runtime block, fluid, and hitbox asset IDs are intentionally absent.
 * Explicit semantic flags, settings, and collision-shape references are the
 * portable contract.</p>
 */
public record RegionCellPaletteEntry(
    int flags,
    int shapeIndex,
    int fluidLevel,
    double fluidFillHeight,
    int supportValue,
    int blockDamage,
    int fluidDamage,
    double[] movement,
    double[] fluidMovement
) {

    public static final int MOVEMENT_VALUES = 9;
    public static final int FLUID_MOVEMENT_VALUES = 6;

    public RegionCellPaletteEntry {
        movement = movement == null ? new double[MOVEMENT_VALUES] : movement.clone();
        fluidMovement = fluidMovement == null
            ? new double[FLUID_MOVEMENT_VALUES]
            : fluidMovement.clone();
        if (movement.length != MOVEMENT_VALUES) {
            throw new IllegalArgumentException(
                "Region cell movement must contain nine values"
            );
        }
        if (fluidMovement.length != FLUID_MOVEMENT_VALUES) {
            throw new IllegalArgumentException(
                "Region cell FluidFX movement must contain six values"
            );
        }
        if (flags < 0 || flags > 0xffff) {
            throw new IllegalArgumentException(
                "Region cell flags must fit uint16"
            );
        }
        if (shapeIndex < 0 || shapeIndex >= NativeRegionSection.MAX_SHAPES) {
            throw new IllegalArgumentException(
                "Region cell shape index exceeds the fixed palette"
            );
        }
        if (fluidLevel < 0 || fluidLevel > 255) {
            throw new IllegalArgumentException("Fluid level must fit uint8");
        }
        if (!Double.isFinite(fluidFillHeight)
            || fluidFillHeight < 0.0
            || fluidFillHeight > 1.0) {
            throw new IllegalArgumentException(
                "Fluid fill height must be finite and in [0, 1]"
            );
        }
        requireFinite(movement, "movement");
        requireFinite(fluidMovement, "FluidFX movement");
    }

    @Override
    public double[] movement() {
        return movement.clone();
    }

    @Override
    public double[] fluidMovement() {
        return fluidMovement.clone();
    }

    public boolean isCanonicalAir() {
        return flags == 0
            && shapeIndex == 0
            && fluidLevel == 0
            && fluidFillHeight == 0.0
            && supportValue == 0
            && blockDamage == 0
            && fluidDamage == 0
            && allZero(movement)
            && allZero(fluidMovement);
    }

    private static void requireFinite(double[] values, String label) {
        for (double value : values) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                    "Region cell " + label + " contains a non-finite value"
                );
            }
        }
    }

    private static boolean allZero(double[] values) {
        for (double value : values) {
            if (value != 0.0) return false;
        }
        return true;
    }
}
