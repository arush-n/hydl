package com.hytalerlbridge.worldgen;

import com.hytalerlbridge.geometry.GeometryContract;
import java.util.List;

/** Controlled native block-health and exact-geometry transition evidence. */
public record NativeMutableBlockEvidence(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    int[] position,
    List<Row> rows
) {

    public static final String SCHEMA = "hytalerl_native_mutable_block_evidence_v1";
    public static final int VERSION = 1;
    public static final List<String> PHASES = List.of(
        "initial",
        "damaged",
        "repaired",
        "removed",
        "placed"
    );

    public NativeMutableBlockEvidence {
        serverVersion = requireText(serverVersion, "serverVersion");
        worldName = requireText(worldName, "worldName");
        worldgenProvider = requireText(worldgenProvider, "worldgenProvider");
        worldgenVersion = requireText(worldgenVersion, "worldgenVersion");
        position = position == null ? new int[0] : position.clone();
        rows = rows == null ? List.of() : List.copyOf(rows);
        if (position.length != 3) {
            throw new IllegalArgumentException(
                "Mutable block evidence position must be one XYZ triple"
            );
        }
        if (
            rows.size() != PHASES.size()
                || !rows.stream().map(Row::phase).toList().equals(PHASES)
        ) {
            throw new IllegalArgumentException(
                "Mutable block evidence must contain the canonical transition phases"
            );
        }
    }

    @Override
    public int[] position() {
        return position.clone();
    }

    /** One exact post-transition value, never a Boolean legality result. */
    public record Row(
        String phase,
        boolean blockPresent,
        String blockAssetId,
        int runtimeBlockId,
        byte[] semanticKeySha256,
        boolean semanticKeyValid,
        boolean affordanceValid,
        int affordanceTags,
        int gatherTypeIndex,
        int requiredToolQuality,
        int rotationIndex,
        int flags,
        int fluidLevel,
        double fluidFillHeight,
        int supportValue,
        int blockDamage,
        int fluidDamage,
        double[] movement,
        double[] fluidMovement,
        double[] collisionBoxes,
        float blockHealth,
        boolean blockHealthValid,
        double secondsSinceDamage,
        boolean damageAgeValid,
        short localChangeCounter,
        short globalChangeCounter
    ) {

        public Row {
            phase = requireText(phase, "phase");
            blockAssetId = blockAssetId == null ? "" : blockAssetId;
            semanticKeySha256 = semanticKeySha256 == null
                ? new byte[0]
                : semanticKeySha256.clone();
            movement = movement == null ? new double[0] : movement.clone();
            fluidMovement = fluidMovement == null
                ? new double[0]
                : fluidMovement.clone();
            collisionBoxes = collisionBoxes == null
                ? new double[0]
                : collisionBoxes.clone();

            if (semanticKeyValid != blockPresent) {
                throw new IllegalArgumentException(
                    "Mutable block semantic identity must match block presence"
                );
            }
            if (affordanceValid != blockPresent) {
                throw new IllegalArgumentException(
                    "Mutable block affordances must match block presence"
                );
            }
            if (
                (affordanceTags & ~BlockAffordanceContract.KNOWN_TAG_MASK) != 0
                    || gatherTypeIndex < 0
                    || gatherTypeIndex
                        >= BlockAffordanceContract.GATHER_TYPES.size()
                    || requiredToolQuality < 0
                    || requiredToolQuality > Short.MAX_VALUE
                    || (!affordanceValid && (
                        affordanceTags != 0
                            || gatherTypeIndex != 0
                            || requiredToolQuality != 0
                    ))
            ) {
                throw new IllegalArgumentException(
                    "Mutable block affordance value is outside its dictionary"
                );
            }
            if (
                semanticKeySha256.length
                    != (semanticKeyValid ? 32 : 0)
            ) {
                throw new IllegalArgumentException(
                    "Mutable block semantic key must be one SHA-256 digest"
                );
            }
            if (movement.length != 9 || fluidMovement.length != 6) {
                throw new IllegalArgumentException(
                    "Mutable block movement payload has the wrong shape"
                );
            }
            if (
                collisionBoxes.length % 6 != 0
                    || collisionBoxes.length / 6
                        > GeometryContract.MAX_DETAIL_BOXES_0_5_7
            ) {
                throw new IllegalArgumentException(
                    "Mutable block collision payload exceeds Hytale 0.5.7 capacity"
                );
            }
            if (
                !Double.isFinite(fluidFillHeight)
                    || fluidFillHeight < 0.0
                    || fluidFillHeight > 1.0
                    || !Double.isFinite(secondsSinceDamage)
                    || secondsSinceDamage < 0.0
            ) {
                throw new IllegalArgumentException(
                    "Mutable block scalar payload is outside its domain"
                );
            }
            if (
                blockHealthValid != blockPresent
                    || !Float.isFinite(blockHealth)
                    || blockHealth < 0.0f
                    || blockHealth > 1.0f
                    || (!blockPresent && blockHealth != 0.0f)
            ) {
                throw new IllegalArgumentException(
                    "Mutable block health must be normalized and presence-gated"
                );
            }
            if (!damageAgeValid && secondsSinceDamage != 0.0) {
                throw new IllegalArgumentException(
                    "Invalid mutable block damage age must be zero"
                );
            }
        }

        @Override
        public byte[] semanticKeySha256() {
            return semanticKeySha256.clone();
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
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " cannot be blank");
        }
        return value;
    }
}
