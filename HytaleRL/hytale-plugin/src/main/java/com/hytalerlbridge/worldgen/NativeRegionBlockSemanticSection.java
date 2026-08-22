package com.hytalerlbridge.worldgen;

import java.util.List;

/** One palette-compressed 32-cubed block-identity/affordance sidecar. */
public record NativeRegionBlockSemanticSection(
    int chunkX,
    int chunkZ,
    int sectionY,
    byte[] cellCodes,
    List<Entry> palette
) {

    public static final String SCHEMA =
        "hytalerl_native_region_block_semantics_v1";
    public static final int VERSION = 1;
    public static final String CODE_ENCODING = "uint16_le_y_z_x";
    public static final int CELL_COUNT = 32 * 32 * 32;
    public static final int CODE_BYTES = CELL_COUNT * 2;
    public static final int MAX_ENTRIES = 1 << 16;

    public NativeRegionBlockSemanticSection {
        cellCodes = cellCodes == null ? new byte[0] : cellCodes.clone();
        palette = palette == null ? List.of() : List.copyOf(palette);
        if (sectionY < 0 || sectionY >= 10) {
            throw new IllegalArgumentException(
                "Region block-semantic section Y must be in [0, 10)"
            );
        }
        if (cellCodes.length != CODE_BYTES) {
            throw new IllegalArgumentException(
                "Region block-semantic codes have the wrong length"
            );
        }
        if (palette.isEmpty() || palette.size() > MAX_ENTRIES) {
            throw new IllegalArgumentException(
                "Region block-semantic palette exceeds uint16 capacity"
            );
        }
        Entry first = palette.getFirst();
        if (
            first.valid()
                || first.semanticKeySha256().length != 0
                || first.assetKeySha256().length != 0
                || first.affordanceTags() != 0
                || first.gatherTypeIndex() != 0
                || first.requiredToolQuality() != 0
                || first.rotationIndex() != 0
        ) {
            throw new IllegalArgumentException(
                "Region block-semantic palette entry zero must be air"
            );
        }
        for (int offset = 0; offset < cellCodes.length; offset += 2) {
            int code = Byte.toUnsignedInt(cellCodes[offset])
                | (Byte.toUnsignedInt(cellCodes[offset + 1]) << 8);
            if (code >= palette.size()) {
                throw new IllegalArgumentException(
                    "Region block-semantic code exceeds its palette"
                );
            }
        }
    }

    @Override
    public byte[] cellCodes() {
        return cellCodes.clone();
    }

    public record Entry(
        byte[] semanticKeySha256,
        byte[] assetKeySha256,
        boolean valid,
        int affordanceTags,
        int gatherTypeIndex,
        int requiredToolQuality,
        int rotationIndex
    ) {

        public Entry {
            semanticKeySha256 = semanticKeySha256 == null
                ? new byte[0]
                : semanticKeySha256.clone();
            assetKeySha256 = assetKeySha256 == null
                ? new byte[0]
                : assetKeySha256.clone();
            if (
                semanticKeySha256.length != (valid ? 32 : 0)
                    || assetKeySha256.length != (valid ? 32 : 0)
            ) {
                throw new IllegalArgumentException(
                    "Region block identity must be two SHA-256 digests"
                );
            }
            if (
                (affordanceTags & ~BlockAffordanceContract.KNOWN_TAG_MASK)
                    != 0
                    || gatherTypeIndex < 0
                    || gatherTypeIndex
                        >= BlockAffordanceContract.GATHER_TYPES.size()
                    || requiredToolQuality < 0
                    || requiredToolQuality > Short.MAX_VALUE
                    || rotationIndex < 0
                    || (!valid && (
                        affordanceTags != 0
                            || gatherTypeIndex != 0
                            || requiredToolQuality != 0
                            || rotationIndex != 0
                    ))
            ) {
                throw new IllegalArgumentException(
                    "Region block affordance is outside its dictionary"
                );
            }
        }

        @Override
        public byte[] semanticKeySha256() {
            return semanticKeySha256.clone();
        }

        @Override
        public byte[] assetKeySha256() {
            return assetKeySha256.clone();
        }

        public static Entry air() {
            return new Entry(
                new byte[0],
                new byte[0],
                false,
                0,
                0,
                0,
                0
            );
        }
    }
}
