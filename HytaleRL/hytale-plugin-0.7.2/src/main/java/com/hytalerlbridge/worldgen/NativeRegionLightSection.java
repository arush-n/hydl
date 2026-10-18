package com.hytalerlbridge.worldgen;

/**
 * One native 32x32x32 global-light section with explicit readiness.
 *
 * <p>The packed uint16 value uses the server's native nibble order:
 * red, green, blue, sky. An unavailable section carries no payload, so a
 * lighting-queue miss cannot be confused with a genuinely dark section.</p>
 */
public record NativeRegionLightSection(
    int chunkX,
    int chunkZ,
    int sectionY,
    String status,
    int globalChangeCounter,
    int globalLightChangeId,
    byte[] lightData
) {

    public static final String SCHEMA = "hytalerl_native_region_light_section_v1";
    public static final int VERSION = 1;
    public static final String LIGHT_ENCODING = "uint16_le_y_z_x_rgbs_nibbles";
    public static final String STATUS_READY = "ready";
    public static final String STATUS_NOT_READY = "global_light_not_ready";
    public static final String STATUS_CHANGED = "changed_during_capture";
    public static final int LIGHT_BYTES = 2 * 32768;

    public NativeRegionLightSection {
        lightData = lightData == null ? new byte[0] : lightData.clone();
        if (sectionY < 0 || sectionY >= 10) {
            throw new IllegalArgumentException("Section Y must be in [0, 10)");
        }
        if (
            !STATUS_READY.equals(status)
                && !STATUS_NOT_READY.equals(status)
                && !STATUS_CHANGED.equals(status)
        ) {
            throw new IllegalArgumentException("Unknown Region light status");
        }
        if (
            globalChangeCounter < 0
                || globalChangeCounter > 0xffff
                || globalLightChangeId < 0
                || globalLightChangeId > 0xffff
        ) {
            throw new IllegalArgumentException(
                "Region light counters must be unsigned 16-bit values"
            );
        }
        if (STATUS_READY.equals(status)) {
            if (lightData.length != LIGHT_BYTES) {
                throw new IllegalArgumentException(
                    "Ready Region light must contain 32768 uint16 values"
                );
            }
            if (globalChangeCounter != globalLightChangeId) {
                throw new IllegalArgumentException(
                    "Ready Region light counters must agree"
                );
            }
        } else if (lightData.length != 0) {
            throw new IllegalArgumentException(
                "Unavailable Region light must not publish a payload"
            );
        }
    }

    public boolean available() {
        return STATUS_READY.equals(status);
    }

    @Override
    public byte[] lightData() {
        return lightData.clone();
    }
}
