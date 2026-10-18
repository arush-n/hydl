package com.hytalerlbridge.worldgen;

/**
 * Fixed-capacity native BlockChunk channel samples.
 *
 * <p>Positions are absolute block coordinates. Invalid channels are zero and
 * must be distinguished through {@link #channelValidity()}.</p>
 */
public record NativePerceptionChannels(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    int[] positions,
    byte[] available,
    byte[] channelValidity,
    short[] heightmap,
    byte[] skyLight,
    byte[] blockLightRgb,
    int[] environment,
    int[] tintArgb
) {

    public static final String SCHEMA = "hytalerl_native_perception_channels_v1";
    public static final int VERSION = 1;
    public static final int MAX_SAMPLES = 4096;
    public static final int WORLD_HEIGHT = 320;

    public static final int VALID_HEIGHTMAP = 1;
    public static final int VALID_SKY_LIGHT = 1 << 1;
    public static final int VALID_BLOCK_LIGHT_RGB = 1 << 2;
    public static final int VALID_ENVIRONMENT = 1 << 3;
    public static final int VALID_TINT_RGB = 1 << 4;
    public static final int VALID_ALL = (1 << 5) - 1;

    public NativePerceptionChannels {
        serverVersion = requireText(serverVersion, "serverVersion");
        worldName = requireText(worldName, "worldName");
        worldgenProvider = requireText(worldgenProvider, "worldgenProvider");
        worldgenVersion = requireText(worldgenVersion, "worldgenVersion");
        positions = copy(positions);
        available = copy(available);
        channelValidity = copy(channelValidity);
        heightmap = copy(heightmap);
        skyLight = copy(skyLight);
        blockLightRgb = copy(blockLightRgb);
        environment = copy(environment);
        tintArgb = copy(tintArgb);

        if (positions.length == 0 || positions.length % 3 != 0) {
            throw new IllegalArgumentException(
                "Perception positions must contain absolute XYZ triples"
            );
        }
        int samples = positions.length / 3;
        if (samples > MAX_SAMPLES) {
            throw new IllegalArgumentException(
                "Perception channel capture exceeds fixed sample capacity"
            );
        }
        requireLength(available.length, samples, "availability");
        requireLength(channelValidity.length, samples, "channel validity");
        requireLength(heightmap.length, samples, "heightmap");
        requireLength(skyLight.length, samples, "sky light");
        requireLength(blockLightRgb.length, samples * 3, "RGB block light");
        requireLength(environment.length, samples, "environment");
        requireLength(tintArgb.length, samples, "tint");

        for (int sample = 0; sample < samples; sample++) {
            int availableValue = Byte.toUnsignedInt(available[sample]);
            if (availableValue > 1) {
                throw new IllegalArgumentException(
                    "Perception availability must be encoded as uint8 0 or 1"
                );
            }
            int validity = Byte.toUnsignedInt(channelValidity[sample]);
            if ((validity & ~VALID_ALL) != 0) {
                throw new IllegalArgumentException(
                    "Perception channel validity contains unknown bits"
                );
            }
            int y = positions[sample * 3 + 1];
            boolean insideY = y >= 0 && y < WORLD_HEIGHT;
            if (availableValue == 0) {
                if (
                    validity != 0
                        || !valuesAreZero(
                            sample,
                            heightmap,
                            skyLight,
                            blockLightRgb,
                            environment,
                            tintArgb
                        )
                ) {
                    throw new IllegalArgumentException(
                        "Unavailable perception samples must be zero-masked"
                    );
                }
                continue;
            }
            if ((validity & VALID_TINT_RGB) == 0) {
                throw new IllegalArgumentException(
                    "A covered native column must publish tint"
                );
            }
            boolean skyValid = (validity & VALID_SKY_LIGHT) != 0;
            boolean blockLightValid = (validity & VALID_BLOCK_LIGHT_RGB) != 0;
            boolean environmentValid = (validity & VALID_ENVIRONMENT) != 0;
            if (
                environmentValid != insideY
                    || skyValid != blockLightValid
                    || (skyValid && !insideY)
            ) {
                throw new IllegalArgumentException(
                    "Y-indexed channel validity disagrees with domain or light readiness"
                );
            }
            if ((validity & VALID_HEIGHTMAP) == 0 && heightmap[sample] != 0) {
                throw new IllegalArgumentException(
                    "Invalid native heightmap values must be zero"
                );
            }
            int rgb = sample * 3;
            if (!skyValid && skyLight[sample] != 0) {
                throw new IllegalArgumentException(
                    "Invalid native sky light values must be zero"
                );
            }
            if (
                !blockLightValid
                    && (
                        blockLightRgb[rgb] != 0
                            || blockLightRgb[rgb + 1] != 0
                            || blockLightRgb[rgb + 2] != 0
                    )
            ) {
                throw new IllegalArgumentException(
                    "Invalid native block light values must be zero"
                );
            }
            if (!environmentValid && environment[sample] != 0) {
                throw new IllegalArgumentException(
                    "Invalid native environment values must be zero"
                );
            }
        }
    }

    public int sampleCount() {
        return available.length;
    }

    @Override
    public int[] positions() {
        return positions.clone();
    }

    @Override
    public byte[] available() {
        return available.clone();
    }

    @Override
    public byte[] channelValidity() {
        return channelValidity.clone();
    }

    @Override
    public short[] heightmap() {
        return heightmap.clone();
    }

    @Override
    public byte[] skyLight() {
        return skyLight.clone();
    }

    @Override
    public byte[] blockLightRgb() {
        return blockLightRgb.clone();
    }

    @Override
    public int[] environment() {
        return environment.clone();
    }

    @Override
    public int[] tintArgb() {
        return tintArgb.clone();
    }

    private static boolean valuesAreZero(
        int sample,
        short[] heightmap,
        byte[] skyLight,
        byte[] blockLightRgb,
        int[] environment,
        int[] tintArgb
    ) {
        return heightmap[sample] == 0
            && yValuesAreZero(sample, skyLight, blockLightRgb, environment)
            && tintArgb[sample] == 0;
    }

    private static boolean yValuesAreZero(
        int sample,
        byte[] skyLight,
        byte[] blockLightRgb,
        int[] environment
    ) {
        int rgb = sample * 3;
        return skyLight[sample] == 0
            && blockLightRgb[rgb] == 0
            && blockLightRgb[rgb + 1] == 0
            && blockLightRgb[rgb + 2] == 0
            && environment[sample] == 0;
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " cannot be blank");
        }
        return value;
    }

    private static void requireLength(int actual, int expected, String name) {
        if (actual != expected) {
            throw new IllegalArgumentException(
                "Perception " + name + " payload has the wrong length"
            );
        }
    }

    private static int[] copy(int[] values) {
        return values == null ? new int[0] : values.clone();
    }

    private static short[] copy(short[] values) {
        return values == null ? new short[0] : values.clone();
    }

    private static byte[] copy(byte[] values) {
        return values == null ? new byte[0] : values.clone();
    }
}
