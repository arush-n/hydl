package com.hytalerlbridge.worldgen;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Bounded empirical evidence for one native block-drop route. */
public record NativeDropProgramEvidence(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    String blockAssetId,
    String route,
    int authoredQuantity,
    String resolvedItemId,
    String resolvedDropListId,
    int sampleCount,
    List<Outcome> outcomes
) {

    public static final String SCHEMA =
        "hytalerl_native_drop_program_evidence_v1";
    public static final int VERSION = 1;
    public static final int MAX_SAMPLES = 65_536;
    public static final int MAX_OUTCOMES = 75;
    public static final int MAX_STACKS_PER_OUTCOME = 10;
    public static final Set<String> ROUTES = Set.of(
        "breaking",
        "soft",
        "harvest"
    );
    public static final String RNG =
        "java.util.concurrent.ThreadLocalRandom";

    public NativeDropProgramEvidence {
        serverVersion = requireText(serverVersion, "serverVersion");
        worldName = requireText(worldName, "worldName");
        worldgenProvider = requireText(
            worldgenProvider,
            "worldgenProvider"
        );
        worldgenVersion = requireText(worldgenVersion, "worldgenVersion");
        blockAssetId = requireText(blockAssetId, "blockAssetId");
        route = requireText(route, "route");
        resolvedItemId = nullToEmpty(resolvedItemId);
        resolvedDropListId = nullToEmpty(resolvedDropListId);
        outcomes = outcomes == null ? List.of() : List.copyOf(outcomes);
        if (!ROUTES.contains(route)) {
            throw new IllegalArgumentException(
                "Unsupported native drop route: " + route
            );
        }
        if (authoredQuantity < 0) {
            throw new IllegalArgumentException(
                "Native drop quantity cannot be negative"
            );
        }
        if (sampleCount < 1 || sampleCount > MAX_SAMPLES) {
            throw new IllegalArgumentException(
                "Native drop sample count exceeds capacity"
            );
        }
        if (outcomes.isEmpty() || outcomes.size() > MAX_OUTCOMES) {
            throw new IllegalArgumentException(
                "Native drop outcome count exceeds capacity"
            );
        }
        int total = outcomes.stream().mapToInt(Outcome::count).sum();
        if (total != sampleCount) {
            throw new IllegalArgumentException(
                "Native drop outcome counts do not match sample count"
            );
        }
        Set<List<Stack>> unique = new HashSet<>();
        for (Outcome outcome : outcomes) {
            if (!unique.add(outcome.stacks())) {
                throw new IllegalArgumentException(
                    "Native drop outcomes must be unique"
                );
            }
        }
    }

    /** One canonical native output bundle and its empirical count. */
    public record Outcome(int count, List<Stack> stacks) {

        public Outcome {
            stacks = stacks == null ? List.of() : List.copyOf(stacks);
            if (
                count < 1
                    || stacks.size() > MAX_STACKS_PER_OUTCOME
            ) {
                throw new IllegalArgumentException(
                    "Native drop outcome is outside capacity"
                );
            }
        }
    }

    /** One non-empty stack in native list order. */
    public record Stack(
        String itemAssetId,
        int quantity,
        double durability,
        double maxDurability,
        String metadataJsonSha256
    ) {

        public Stack {
            itemAssetId = requireText(itemAssetId, "itemAssetId");
            metadataJsonSha256 = nullToEmpty(metadataJsonSha256);
            if (
                quantity < 1
                    || !Double.isFinite(durability)
                    || !Double.isFinite(maxDurability)
                    || durability < 0.0
                    || maxDurability < 0.0
                    || (
                        !metadataJsonSha256.isEmpty()
                            && !metadataJsonSha256.matches(
                                "[0-9a-f]{64}"
                            )
                    )
            ) {
                throw new IllegalArgumentException(
                    "Native drop stack is outside its domain"
                );
            }
        }
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " cannot be blank");
        }
        return value;
    }

    private static String nullToEmpty(String value) {
        return value == null ? "" : value;
    }
}
