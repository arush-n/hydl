package com.hytalerlbridge.nativebackend.model;

import com.hytalerlbridge.worldgen.RegionCellPaletteEntry;

/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record RegionCellKey(
    int flags,
    int shapeIndex,
    int fluidLevel,
    double fluidFillHeight,
    int supportValue,
    int blockDamage,
    int fluidDamage,
    DoubleArrayKey movement,
    DoubleArrayKey fluidMovement
) {
    public static RegionCellKey from(RegionCellPaletteEntry entry) {
        return new RegionCellKey(
            entry.flags(),
            entry.shapeIndex(),
            entry.fluidLevel(),
            entry.fluidFillHeight(),
            entry.supportValue(),
            entry.blockDamage(),
            entry.fluidDamage(),
            new DoubleArrayKey(entry.movement()),
            new DoubleArrayKey(entry.fluidMovement())
        );
    }
}
