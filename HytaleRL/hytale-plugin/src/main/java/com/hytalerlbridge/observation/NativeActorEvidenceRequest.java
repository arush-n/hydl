package com.hytalerlbridge.observation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/**
 * Reset-time negotiation for the native learner-v3 actor-evidence adapter.
 *
 * <p>The bridge and host must agree on the complete evidence contract before
 * any policy-facing row is emitted. A missing request keeps the legacy wire
 * compatible and emits an explicitly unavailable frame.</p>
 */
public record NativeActorEvidenceRequest(
    String contractSha256,
    int entityCount,
    int statusCapacity,
    int abilityCapacity,
    List<String> resourceStatIds
) {
    private static final Pattern SHA256 = Pattern.compile("[0-9A-F]{64}");
    private static final Pattern STAT_ID = Pattern.compile(
        "[A-Za-z][A-Za-z0-9_]{0,127}"
    );
    public static final List<String> DEFAULT_RESOURCE_STAT_IDS = List.of(
        "Stamina",
        "Mana",
        "MagicCharges",
        "SignatureEnergy",
        "SignatureCharges",
        "Ammo",
        "Oxygen"
    );

    /** Compatibility constructor for the original fixed resource mapping. */
    public NativeActorEvidenceRequest(
        String contractSha256,
        int entityCount,
        int statusCapacity,
        int abilityCapacity
    ) {
        this(
            contractSha256,
            entityCount,
            statusCapacity,
            abilityCapacity,
            DEFAULT_RESOURCE_STAT_IDS
        );
    }

    public NativeActorEvidenceRequest {
        contractSha256 = contractSha256 == null
            ? ""
            : contractSha256.trim().toUpperCase(Locale.ROOT);
        if (!SHA256.matcher(contractSha256).matches()) {
            throw new IllegalArgumentException(
                "learner-v3 actor-evidence contract SHA-256 must contain 64 hex digits"
            );
        }
        if (entityCount < 1) {
            throw new IllegalArgumentException(
                "learner-v3 actor-evidence entity count must be positive"
            );
        }
        if (statusCapacity < 1 || abilityCapacity < 1) {
            throw new IllegalArgumentException(
                "learner-v3 actor-evidence capacities must be positive"
            );
        }
        List<String> source = resourceStatIds == null
            ? DEFAULT_RESOURCE_STAT_IDS
            : resourceStatIds;
        if (source.size() != NativeActorEvidenceFrame.RESOURCE_COUNT) {
            throw new IllegalArgumentException(
                "learner-v3 actor-evidence resource stat count must equal "
                    + NativeActorEvidenceFrame.RESOURCE_COUNT
            );
        }
        ArrayList<String> canonical = new ArrayList<>(source.size());
        for (String value : source) {
            String statId = value == null ? "" : value.trim();
            if (!STAT_ID.matcher(statId).matches()) {
                throw new IllegalArgumentException(
                    "learner-v3 actor-evidence resource stat IDs must be "
                        + "non-empty asset identifiers"
                );
            }
            canonical.add(statId);
        }
        if (new HashSet<>(canonical).size() != canonical.size()) {
            throw new IllegalArgumentException(
                "learner-v3 actor-evidence resource stat IDs must be unique"
            );
        }
        resourceStatIds = List.copyOf(canonical);
    }

    /** Reject a host whose compiled actor-evidence ABI differs from this jar. */
    public void requireSupported() {
        if (!contractSha256.equals(NativeActorEvidenceFrame.CONTRACT_SHA256)) {
            throw new IllegalArgumentException(
                "learner-v3 actor-evidence contract mismatch: "
                    + contractSha256
                    + " != "
                    + NativeActorEvidenceFrame.CONTRACT_SHA256
            );
        }
        if (entityCount != NativeActorEvidenceFrame.ENTITY_COUNT) {
            throw new IllegalArgumentException(
                "learner-v3 actor-evidence entity count mismatch: "
                    + entityCount
                    + " != "
                    + NativeActorEvidenceFrame.ENTITY_COUNT
            );
        }
        if (statusCapacity != NativeActorEvidenceFrame.STATUS_CAPACITY) {
            throw new IllegalArgumentException(
                "learner-v3 actor-evidence status capacity mismatch: "
                    + statusCapacity
                    + " != "
                    + NativeActorEvidenceFrame.STATUS_CAPACITY
            );
        }
        if (abilityCapacity != NativeActorEvidenceFrame.ABILITY_CAPACITY) {
            throw new IllegalArgumentException(
                "learner-v3 actor-evidence ability capacity mismatch: "
                    + abilityCapacity
                    + " != "
                    + NativeActorEvidenceFrame.ABILITY_CAPACITY
            );
        }
    }
}
