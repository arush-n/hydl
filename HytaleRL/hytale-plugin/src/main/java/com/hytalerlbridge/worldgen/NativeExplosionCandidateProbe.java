package com.hytalerlbridge.worldgen;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;

/** Complete bounded output of Hytale's entity-only explosion admission pass. */
public record NativeExplosionCandidateProbe(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    double[] origin,
    int blockDamageRadius,
    float entityDamageRadius,
    boolean ignoredControlledActor,
    int capacity,
    int totalMatching,
    boolean overflow,
    List<Candidate> candidates
) {
    public static final String SCHEMA =
        "hytalerl_native_explosion_candidate_probe_v1";
    public static final int VERSION = 1;
    public static final int MAX_CAPACITY = 64;
    public static final int MAX_RADIUS = 5;

    public NativeExplosionCandidateProbe {
        requireText(serverVersion, "serverVersion");
        requireText(worldName, "worldName");
        requireText(worldgenProvider, "worldgenProvider");
        requireText(worldgenVersion, "worldgenVersion");
        origin = origin == null ? null : origin.clone();
        if (
            origin == null
                || origin.length != 3
                || !Double.isFinite(origin[0])
                || !Double.isFinite(origin[1])
                || !Double.isFinite(origin[2])
        ) {
            throw new IllegalArgumentException("origin must be one finite XYZ row");
        }
        if (blockDamageRadius < 1 || blockDamageRadius > MAX_RADIUS) {
            throw new IllegalArgumentException("blockDamageRadius must be 1..4");
        }
        if (
            !Float.isFinite(entityDamageRadius)
                || entityDamageRadius <= 0.0f
                || entityDamageRadius > MAX_RADIUS
        ) {
            throw new IllegalArgumentException("entityDamageRadius must be in (0,4]");
        }
        if (capacity < 1 || capacity > MAX_CAPACITY) {
            throw new IllegalArgumentException("capacity must be 1..64");
        }
        if (totalMatching < 0) {
            throw new IllegalArgumentException("totalMatching must be non-negative");
        }
        candidates = candidates == null ? List.of() : List.copyOf(candidates);
        boolean expectedOverflow = totalMatching > capacity;
        if (overflow != expectedOverflow) {
            throw new IllegalArgumentException(
                "overflow must exactly reflect totalMatching > capacity"
            );
        }
        if (overflow && !candidates.isEmpty()) {
            throw new IllegalArgumentException("overflow must not emit partial rows");
        }
        if (!overflow && candidates.size() != totalMatching) {
            throw new IllegalArgumentException(
                "non-overflow probes must emit every candidate"
            );
        }
        UUID previous = null;
        for (Candidate candidate : candidates) {
            if (candidate == null) {
                throw new IllegalArgumentException("candidate must not be null");
            }
            if (
                previous != null
                    && UUID_TEXT_ORDER.compare(previous, candidate.uuid()) >= 0
            ) {
                throw new IllegalArgumentException(
                    "candidate UUIDs must be unique and ordered"
                );
            }
            previous = candidate.uuid();
        }
    }

    public static NativeExplosionCandidateProbe fromMatches(
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed,
        double[] origin,
        int blockDamageRadius,
        float entityDamageRadius,
        boolean ignoredControlledActor,
        int capacity,
        List<Candidate> matches
    ) {
        if (matches == null) {
            throw new IllegalArgumentException("matches must be present");
        }
        int total = matches.size();
        List<Candidate> ordered = new ArrayList<>(matches);
        ordered.sort(Comparator.comparing(Candidate::uuid, UUID_TEXT_ORDER));
        return new NativeExplosionCandidateProbe(
            serverVersion,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            origin,
            blockDamageRadius,
            entityDamageRadius,
            ignoredControlledActor,
            capacity,
            total,
            total > capacity,
            total > capacity ? List.of() : ordered
        );
    }

    public int emittedCount() {
        return candidates.size();
    }

    @Override
    public double[] origin() {
        return origin.clone();
    }

    public record Candidate(UUID uuid, double x, double y, double z) {
        public Candidate {
            if (uuid == null) {
                throw new IllegalArgumentException("uuid must be present");
            }
            if (!Double.isFinite(x) || !Double.isFinite(y) || !Double.isFinite(z)) {
                throw new IllegalArgumentException("candidate position must be finite");
            }
        }
    }

    private static final Comparator<UUID> UUID_TEXT_ORDER =
        Comparator.comparing(UUID::toString);

    private static void requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
    }
}
