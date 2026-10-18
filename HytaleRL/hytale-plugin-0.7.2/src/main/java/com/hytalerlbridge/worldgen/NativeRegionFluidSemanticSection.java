package com.hytalerlbridge.worldgen;

import java.util.List;

/** One palette-compressed 32-cubed stable native fluid-identity sidecar. */
public record NativeRegionFluidSemanticSection(
    int chunkX,
    int chunkZ,
    int sectionY,
    byte[] cellCodes,
    List<Entry> palette
) {

    public static final String SCHEMA =
        "hytalerl_native_region_fluid_semantics_v1";
    public static final int VERSION = 1;
    public static final String CODE_ENCODING = "uint16_le_y_z_x";
    public static final int CELL_COUNT = 32 * 32 * 32;
    public static final int CODE_BYTES = CELL_COUNT * 2;
    public static final int MAX_ENTRIES = 1 << 16;

    public NativeRegionFluidSemanticSection {
        cellCodes = cellCodes == null ? new byte[0] : cellCodes.clone();
        palette = palette == null ? List.of() : List.copyOf(palette);
        if (sectionY < 0 || sectionY >= 10) {
            throw new IllegalArgumentException(
                "Region fluid-semantic section Y must be in [0, 10)"
            );
        }
        if (cellCodes.length != CODE_BYTES) {
            throw new IllegalArgumentException(
                "Region fluid-semantic codes have the wrong length"
            );
        }
        if (palette.isEmpty() || palette.size() > MAX_ENTRIES) {
            throw new IllegalArgumentException(
                "Region fluid-semantic palette exceeds uint16 capacity"
            );
        }
        if (!palette.getFirst().assetId().isEmpty()) {
            throw new IllegalArgumentException(
                "Region fluid-semantic palette entry zero must be empty"
            );
        }
        for (int index = 1; index < palette.size(); index++) {
            if (palette.get(index).assetId().isEmpty()) {
                throw new IllegalArgumentException(
                    "Only Region fluid-semantic entry zero may be empty"
                );
            }
        }
        for (int offset = 0; offset < cellCodes.length; offset += 2) {
            int code = Byte.toUnsignedInt(cellCodes[offset])
                | (Byte.toUnsignedInt(cellCodes[offset + 1]) << 8);
            if (code >= palette.size()) {
                throw new IllegalArgumentException(
                    "Region fluid-semantic code exceeds its palette"
                );
            }
        }
    }

    @Override
    public byte[] cellCodes() {
        return cellCodes.clone();
    }

    public record Entry(String assetId) {

        public Entry {
            assetId = assetId == null ? "" : assetId;
            if (!assetId.equals(assetId.strip())) {
                throw new IllegalArgumentException(
                    "Region fluid asset ID cannot have outer whitespace"
                );
            }
        }

        public static Entry empty() {
            return new Entry("");
        }
    }
}
