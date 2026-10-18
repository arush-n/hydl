package com.hytalerlbridge.worldgen;

import java.util.List;

/**
 * One palette-compressed 32x32x32 native section.
 *
 * <p>Codes are little-endian uint16 in Hytale's runtime-verified y-z-x
 * section order. Palette entry zero is canonical known air.</p>
 */
public record NativeRegionSection(
    int chunkX,
    int chunkZ,
    int sectionY,
    byte[] cellCodes,
    byte[] fillerRootOffsets,
    List<RegionCellPaletteEntry> cellPalette,
    List<RegionShapePaletteEntry> shapePalette
) {

    public static final String SCHEMA = "hytalerl_native_region_section_v3";
    public static final int VERSION = 3;
    public static final String CODE_ENCODING = "uint16_le_y_z_x";
    public static final String FILLER_ROOT_ENCODING =
        "uint16_le_signed_5bit_xyz_y_z_x";
    public static final int MAX_CELLS = 1 << 16;
    public static final int MAX_SHAPES = 1 << 14;
    public static final int CODE_BYTES = 2 * 32768;
    public static final int FILLER_ROOT_BYTES = 2 * 32768;

    public NativeRegionSection {
        cellCodes = cellCodes == null ? new byte[0] : cellCodes.clone();
        fillerRootOffsets = fillerRootOffsets == null
            ? new byte[0]
            : fillerRootOffsets.clone();
        cellPalette = cellPalette == null ? List.of() : List.copyOf(cellPalette);
        shapePalette = shapePalette == null
            ? List.of()
            : List.copyOf(shapePalette);
        if (sectionY < 0 || sectionY >= 10) {
            throw new IllegalArgumentException("Section Y must be in [0, 10)");
        }
        if (cellCodes.length != CODE_BYTES) {
            throw new IllegalArgumentException(
                "Region section code payload must contain 32768 uint16 values"
            );
        }
        if (fillerRootOffsets.length != FILLER_ROOT_BYTES) {
            throw new IllegalArgumentException(
                "Region filler-root payload must contain 32768 uint16 values"
            );
        }
        if (cellPalette.isEmpty() || cellPalette.size() > MAX_CELLS) {
            throw new IllegalArgumentException(
                "Region cell palette exceeds uint16 capacity"
            );
        }
        if (shapePalette.isEmpty() || shapePalette.size() > MAX_SHAPES) {
            throw new IllegalArgumentException(
                "Region shape palette exceeds uint14 capacity"
            );
        }
        if (!cellPalette.getFirst().isCanonicalAir()) {
            throw new IllegalArgumentException(
                "Region cell palette entry zero must be canonical air"
            );
        }
        if (shapePalette.getFirst().collisionBoxes().length != 0) {
            throw new IllegalArgumentException(
                "Region shape palette entry zero must be empty"
            );
        }
        for (RegionCellPaletteEntry entry : cellPalette) {
            if (entry.shapeIndex() >= shapePalette.size()) {
                throw new IllegalArgumentException(
                    "Region cell references an absent shape"
                );
            }
        }
        for (int offset = 0; offset < cellCodes.length; offset += 2) {
            int code = Byte.toUnsignedInt(cellCodes[offset])
                | (Byte.toUnsignedInt(cellCodes[offset + 1]) << 8);
            if (code >= cellPalette.size()) {
                throw new IllegalArgumentException(
                    "Region section code exceeds its cell palette"
                );
            }
            int filler = Byte.toUnsignedInt(fillerRootOffsets[offset])
                | (Byte.toUnsignedInt(fillerRootOffsets[offset + 1]) << 8);
            if ((filler & 0x8000) != 0) {
                throw new IllegalArgumentException(
                    "Region filler-root value exceeds packed 15-bit capacity"
                );
            }
            if (code == 0 && filler != 0) {
                throw new IllegalArgumentException(
                    "Canonical air cannot carry a filler-root offset"
                );
            }
        }
    }

    @Override
    public byte[] cellCodes() {
        return cellCodes.clone();
    }

    @Override
    public byte[] fillerRootOffsets() {
        return fillerRootOffsets.clone();
    }
}
