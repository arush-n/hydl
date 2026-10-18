package com.hytalerlbridge.worldgen;

import java.util.Arrays;

/** Bounded native evidence for the two Hytale 0.5.7 LOS consumers. */
public record NativeLineOfSightEvidence(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    double[] start,
    double[] loadedEnd,
    double[] unloadedEnd,
    int loadedEndChunkX,
    int loadedEndChunkZ,
    int unloadedEndChunkX,
    int unloadedEndChunkZ,
    boolean loadedEndChunkPresent,
    boolean unloadedEndChunkPresentBefore,
    boolean unloadedEndChunkPresentAfter,
    boolean perceptionLoadedClear,
    boolean perceptionUnloadedClear,
    boolean selectorLoadedClear,
    boolean selectorUnloadedClear,
    boolean forwardCachedVisibilityAfterMove,
    boolean inverseUncachedVisibilityAfterMove,
    double cacheStepSeconds,
    int[] cacheExpirySteps
) {
    public static final String SCHEMA = "hytalerl_native_los_evidence_v1";
    public static final int VERSION = 1;
    public static final int MAX_CACHE_TRIALS = 32;
    public static final int MAX_CACHE_STEPS = 128;
    public static final String PERCEPTION_CALL_SITE =
        "EntityFilterLineOfSight.matchesEntity->PositionCache.hasLineOfSight";
    public static final String HIT_CONFIRMATION_CALL_SITE =
        "HorizontalSelector.RuntimeSelector.tick->HitDetectionExecutor.LineOfSightProvider";

    public NativeLineOfSightEvidence {
        requireText(serverVersion, "serverVersion");
        requireText(worldName, "worldName");
        requireText(worldgenProvider, "worldgenProvider");
        requireText(worldgenVersion, "worldgenVersion");
        start = point(start, "start");
        loadedEnd = point(loadedEnd, "loadedEnd");
        unloadedEnd = point(unloadedEnd, "unloadedEnd");
        if (!Double.isFinite(cacheStepSeconds) || cacheStepSeconds <= 0.0) {
            throw new IllegalArgumentException("cache step seconds must be finite and positive");
        }
        if (
            cacheExpirySteps == null
                || cacheExpirySteps.length == 0
                || cacheExpirySteps.length > MAX_CACHE_TRIALS
        ) {
            throw new IllegalArgumentException(
                "cache expiry steps must contain 1.." + MAX_CACHE_TRIALS + " trials"
            );
        }
        cacheExpirySteps = cacheExpirySteps.clone();
        for (int steps : cacheExpirySteps) {
            if (steps < 1 || steps > MAX_CACHE_STEPS) {
                throw new IllegalArgumentException(
                    "cache expiry step lies outside the bounded probe"
                );
            }
        }
    }

    @Override
    public double[] start() {
        return start.clone();
    }

    @Override
    public double[] loadedEnd() {
        return loadedEnd.clone();
    }

    @Override
    public double[] unloadedEnd() {
        return unloadedEnd.clone();
    }

    @Override
    public int[] cacheExpirySteps() {
        return cacheExpirySteps.clone();
    }

    private static double[] point(double[] value, String name) {
        if (
            value == null
                || value.length != 3
                || Arrays.stream(value).anyMatch(component -> !Double.isFinite(component))
        ) {
            throw new IllegalArgumentException(name + " must be one finite XYZ point");
        }
        return value.clone();
    }

    private static void requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " cannot be blank");
        }
    }
}
