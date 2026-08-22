package com.hytalerlbridge.nativebackend.worldactions.capture.provider;

import java.util.UUID;

/** Strict actor-slot/UUID routing shared by observe and commit capture. */
public final class PolicyActorCaptureRoute {

    private PolicyActorCaptureRoute() {}

    /**
     * Validate that a requested slot names the same live actor negotiated by
     * the host.  Slot selection and identity matching are deliberately
     * separate: neither an in-range slot nor a matching UUID alone is enough.
     */
    public static Route require(
        int actorSlot,
        String expectedActorIdentity,
        int actorCapacity,
        String liveActorIdentity
    ) {
        if (actorCapacity < 0) {
            throw new IllegalArgumentException(
                "Policy actor capacity must be nonnegative"
            );
        }
        if (actorSlot < 0 || actorSlot >= actorCapacity) {
            throw new IllegalArgumentException(
                "Policy actor slot is outside negotiated capacity"
            );
        }
        String expected = canonicalUuid(
            expectedActorIdentity,
            "Expected policy actor identity"
        );
        String live = canonicalUuid(
            liveActorIdentity,
            "Live policy actor identity"
        );
        if (!expected.equals(live)) {
            throw new IllegalStateException(
                "Policy actor identity does not match the requested slot"
            );
        }
        return new Route(actorSlot, live);
    }

    private static String canonicalUuid(String value, String name) {
        if (value == null || value.isBlank() || !value.equals(value.trim())) {
            throw new IllegalArgumentException(name + " must be a canonical UUID");
        }
        try {
            String canonical = UUID.fromString(value).toString();
            if (!canonical.equals(value)) {
                throw new IllegalArgumentException(
                    name + " must be a canonical UUID"
                );
            }
            return canonical;
        } catch (IllegalArgumentException invalid) {
            throw new IllegalArgumentException(
                name + " must be a canonical UUID",
                invalid
            );
        }
    }

    public record Route(int actorSlot, String actorIdentity) {}
}
