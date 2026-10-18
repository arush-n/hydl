package com.hytalerlbridge.worldgen.policyactions.capture;

import com.hytalerlbridge.worldgen.policyactions.capture.model.PolicyWorldActionSelection;
import java.util.List;
import java.util.UUID;

/** Validated observe/commit request before any World-thread work begins. */
public record PolicyWorldActionCaptureRequest(
    String phase,
    int actorSlot,
    String expectedActorIdentity,
    String expectedCandidateGenerationSha256,
    PolicyWorldActionSelection selection
) {
    public PolicyWorldActionCaptureRequest {
        if (phase == null || !List.of("observe", "commit").contains(phase)) {
            throw new IllegalArgumentException(
                "World-action capture phase must be observe or commit"
            );
        }
        if (actorSlot < 0) {
            throw new IllegalArgumentException(
                "World-action capture actor slot must be nonnegative"
            );
        }
        expectedActorIdentity = canonicalUuid(expectedActorIdentity);
        expectedCandidateGenerationSha256 =
            expectedCandidateGenerationSha256 == null
                ? ""
                : expectedCandidateGenerationSha256.toUpperCase();
        if (
            !expectedCandidateGenerationSha256.isEmpty()
                && !expectedCandidateGenerationSha256.matches(
                    "[0-9A-F]{64}"
                )
        ) {
            throw new IllegalArgumentException(
                "Expected candidate generation must be SHA-256"
            );
        }
        if (phase.equals("observe")) {
            if (
                !expectedCandidateGenerationSha256.isEmpty()
                    || selection != null
            ) {
                throw new IllegalArgumentException(
                    "Observe cannot carry commit fields"
                );
            }
        } else if (
            expectedCandidateGenerationSha256.isEmpty()
                || selection == null
        ) {
            throw new IllegalArgumentException(
                "Commit requires generation and one bounded selection"
            );
        }
    }

    public static PolicyWorldActionCaptureRequest observe(
        int actorSlot,
        String expectedActorIdentity
    ) {
        return new PolicyWorldActionCaptureRequest(
            "observe",
            actorSlot,
            expectedActorIdentity,
            "",
            null
        );
    }

    private static String canonicalUuid(String value) {
        if (value == null || value.isBlank() || !value.equals(value.trim())) {
            throw new IllegalArgumentException(
                "Expected policy actor identity must be a canonical UUID"
            );
        }
        try {
            String canonical = UUID.fromString(value).toString();
            if (!canonical.equals(value)) {
                throw new IllegalArgumentException(
                    "Expected policy actor identity must be a canonical UUID"
                );
            }
            return canonical;
        } catch (IllegalArgumentException invalid) {
            throw new IllegalArgumentException(
                "Expected policy actor identity must be a canonical UUID",
                invalid
            );
        }
    }
}
