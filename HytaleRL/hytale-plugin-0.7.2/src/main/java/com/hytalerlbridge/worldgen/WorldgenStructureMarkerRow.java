package com.hytalerlbridge.worldgen;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Comparator;
import java.util.UUID;

/** One authored marker retaining its exact native prefab grouping and pose. */
public record WorldgenStructureMarkerRow(
    long uuidMostSignificantBits,
    long uuidLeastSignificantBits,
    String markerAssetId,
    double positionX,
    double positionY,
    double positionZ,
    double rotationYaw,
    double rotationPitch,
    double rotationRoll,
    int nativeWorldgenId,
    int nativePrefabInstanceId
) {
    public static final Comparator<WorldgenStructureMarkerRow> UUID_BYTE_ORDER =
        (left, right) -> {
            int most = Long.compareUnsigned(
                left.uuidMostSignificantBits,
                right.uuidMostSignificantBits
            );
            return most != 0
                ? most
                : Long.compareUnsigned(
                    left.uuidLeastSignificantBits,
                    right.uuidLeastSignificantBits
                );
        };

    public WorldgenStructureMarkerRow {
        if (markerAssetId == null || markerAssetId.isBlank()) {
            throw new IllegalArgumentException(
                "WorldGen structure marker asset ID must be non-empty"
            );
        }
        requireFinite(positionX, "positionX");
        requireFinite(positionY, "positionY");
        requireFinite(positionZ, "positionZ");
        requireFinite(rotationYaw, "rotationYaw");
        requireFinite(rotationPitch, "rotationPitch");
        requireFinite(rotationRoll, "rotationRoll");
        if (nativeWorldgenId < 0 || nativePrefabInstanceId < 0) {
            throw new IllegalArgumentException(
                "WorldGen structure marker native IDs must be non-negative"
            );
        }
    }

    public static WorldgenStructureMarkerRow of(
        UUID uuid,
        String markerAssetId,
        double positionX,
        double positionY,
        double positionZ,
        double rotationYaw,
        double rotationPitch,
        double rotationRoll,
        int nativeWorldgenId,
        int nativePrefabInstanceId
    ) {
        if (uuid == null) {
            throw new IllegalArgumentException(
                "WorldGen structure marker UUID must be present"
            );
        }
        return new WorldgenStructureMarkerRow(
            uuid.getMostSignificantBits(),
            uuid.getLeastSignificantBits(),
            markerAssetId,
            positionX,
            positionY,
            positionZ,
            rotationYaw,
            rotationPitch,
            rotationRoll,
            nativeWorldgenId,
            nativePrefabInstanceId
        );
    }

    public byte[] uuidBytes() {
        return ByteBuffer.allocate(16)
            .order(ByteOrder.BIG_ENDIAN)
            .putLong(uuidMostSignificantBits)
            .putLong(uuidLeastSignificantBits)
            .array();
    }

    private static void requireFinite(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
    }
}
