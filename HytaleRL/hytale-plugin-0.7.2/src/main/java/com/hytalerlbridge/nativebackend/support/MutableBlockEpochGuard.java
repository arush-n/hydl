package com.hytalerlbridge.nativebackend.support;

import java.util.Objects;
import java.util.UUID;

/** Optional reservation guard for targeted mutable-block reads. */
public final class MutableBlockEpochGuard {

    private MutableBlockEpochGuard() {}

    public static void require(
        UUID reservationId,
        String expectedWorldEpoch,
        String phase
    ) {
        Objects.requireNonNull(reservationId, "reservationId");
        Objects.requireNonNull(phase, "phase");
        if (expectedWorldEpoch == null) return;
        if (
            expectedWorldEpoch.isBlank()
                || expectedWorldEpoch.length() > 128
        ) {
            throw new IllegalArgumentException(
                "expected_world_epoch must contain 1..128 characters"
            );
        }
        if (!reservationId.toString().equals(expectedWorldEpoch)) {
            throw new IllegalStateException(
                "Mutable block expected_world_epoch is stale " + phase
            );
        }
    }
}
