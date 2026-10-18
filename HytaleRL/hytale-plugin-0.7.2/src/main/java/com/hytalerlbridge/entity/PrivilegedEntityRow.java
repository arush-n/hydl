package com.hytalerlbridge.entity;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Comparator;
import java.util.UUID;

/** Lossless identity plus authoritative pose and velocity for one entity. */
public record PrivilegedEntityRow(
    long uuidMostSignificantBits,
    long uuidLeastSignificantBits,
    String typeAssetId,
    double positionX,
    double positionY,
    double positionZ,
    double rotationYaw,
    double rotationPitch,
    double rotationRoll,
    double velocityX,
    double velocityY,
    double velocityZ
) {
    public static final Comparator<PrivilegedEntityRow> UUID_BYTE_ORDER =
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

    public PrivilegedEntityRow {
        if (typeAssetId == null) {
            throw new IllegalArgumentException("typeAssetId must not be null");
        }
        requireFinite(positionX, "positionX");
        requireFinite(positionY, "positionY");
        requireFinite(positionZ, "positionZ");
        requireFinite(rotationYaw, "rotationYaw");
        requireFinite(rotationPitch, "rotationPitch");
        requireFinite(rotationRoll, "rotationRoll");
        requireFinite(velocityX, "velocityX");
        requireFinite(velocityY, "velocityY");
        requireFinite(velocityZ, "velocityZ");
    }

    public static PrivilegedEntityRow of(
        UUID uuid,
        double positionX,
        double positionY,
        double positionZ,
        double rotationYaw,
        double rotationPitch,
        double rotationRoll,
        double velocityX,
        double velocityY,
        double velocityZ
    ) {
        if (uuid == null) {
            throw new IllegalArgumentException("Entity UUID must be present");
        }
        return new PrivilegedEntityRow(
            uuid.getMostSignificantBits(),
            uuid.getLeastSignificantBits(),
            "",
            positionX,
            positionY,
            positionZ,
            rotationYaw,
            rotationPitch,
            rotationRoll,
            velocityX,
            velocityY,
            velocityZ
        );
    }

    public static PrivilegedEntityRow ofNpc(
        UUID uuid,
        String typeAssetId,
        double positionX,
        double positionY,
        double positionZ,
        double rotationYaw,
        double rotationPitch,
        double rotationRoll,
        double velocityX,
        double velocityY,
        double velocityZ
    ) {
        if (uuid == null) {
            throw new IllegalArgumentException("Entity UUID must be present");
        }
        if (typeAssetId == null || typeAssetId.isBlank()) {
            throw new IllegalArgumentException(
                "NPC type asset ID must be present"
            );
        }
        return new PrivilegedEntityRow(
            uuid.getMostSignificantBits(),
            uuid.getLeastSignificantBits(),
            typeAssetId,
            positionX,
            positionY,
            positionZ,
            rotationYaw,
            rotationPitch,
            rotationRoll,
            velocityX,
            velocityY,
            velocityZ
        );
    }

    /** RFC-4122/network byte order; never a lossy UUID projection. */
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
