package com.hytalerlbridge.worldgen;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Fixed native evidence for vertical single- and double-door transitions. */
public record NativeDoorTransitionEvidence(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    String doorAssetId,
    String rootInteractionId,
    List<Row> rows
) {
    public static final String SCHEMA =
        "hytalerl_native_door_transition_evidence_v1";
    public static final int VERSION = 1;
    public static final int YAW_COUNT = 4;
    public static final String EXECUTION =
        "native_vertical_door_interaction_direct_on_world_thread";

    public NativeDoorTransitionEvidence {
        requireText(serverVersion, "serverVersion");
        requireText(worldName, "worldName");
        requireText(worldgenProvider, "worldgenProvider");
        requireText(worldgenVersion, "worldgenVersion");
        requireText(doorAssetId, "doorAssetId");
        requireText(rootInteractionId, "rootInteractionId");
        if (rows == null || rows.size() != YAW_COUNT) {
            throw new IllegalArgumentException(
                "Native door evidence requires exactly four yaw rows"
            );
        }
        rows = List.copyOf(rows);
        Set<Integer> yaws = new HashSet<>();
        for (Row row : rows) {
            if (row == null || !yaws.add(row.yawDegrees())) {
                throw new IllegalArgumentException(
                    "Native door evidence yaw rows must be unique"
                );
            }
        }
        if (!yaws.equals(Set.of(0, 90, 180, 270))) {
            throw new IllegalArgumentException(
                "Native door evidence must cover 0/90/180/270 degrees"
            );
        }
    }

    /** One yaw's observed physical states and collision identities. */
    public record Row(
        int yawDegrees,
        int[] partnerOffset,
        String singleFrontOpenState,
        String singleFrontOpenHitboxType,
        boolean singleClosedAfterFront,
        String singleBackOpenState,
        String singleBackOpenHitboxType,
        boolean singleClosedAfterBack,
        String doubleRootOpenState,
        String doubleRootOpenHitboxType,
        String doublePartnerOpenState,
        String doublePartnerOpenHitboxType,
        boolean doubleRootClosed,
        boolean doublePartnerClosed
    ) {
        public Row {
            if (yawDegrees % 90 != 0 || yawDegrees < 0 || yawDegrees > 270) {
                throw new IllegalArgumentException("Door yaw is invalid");
            }
            if (partnerOffset == null || partnerOffset.length != 3) {
                throw new IllegalArgumentException(
                    "Door partner offset must contain XYZ"
                );
            }
            partnerOffset = partnerOffset.clone();
            requireText(singleFrontOpenState, "singleFrontOpenState");
            requireText(singleFrontOpenHitboxType, "singleFrontOpenHitboxType");
            requireText(singleBackOpenState, "singleBackOpenState");
            requireText(singleBackOpenHitboxType, "singleBackOpenHitboxType");
            requireText(doubleRootOpenState, "doubleRootOpenState");
            requireText(doubleRootOpenHitboxType, "doubleRootOpenHitboxType");
            requireText(doublePartnerOpenState, "doublePartnerOpenState");
            requireText(
                doublePartnerOpenHitboxType,
                "doublePartnerOpenHitboxType"
            );
        }

        @Override
        public int[] partnerOffset() {
            return partnerOffset.clone();
        }
    }

    private static void requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must be nonblank");
        }
    }
}
