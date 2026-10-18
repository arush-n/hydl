package com.hytalerlbridge.worldgen.policyactions.capture.model;

import com.hytalerlbridge.worldgen.NativeMutableBlockEvidence;
import java.util.List;

/** Complete fixed-capacity block candidate surface. */
public record PolicyWorldActionBlockSurface(
    boolean available,
    String unavailableReason,
    int sourceCount,
    int emittedCount,
    boolean capacityExceeded,
    String primaryUnavailableReason,
    String secondaryUnavailableReason,
    List<Candidate> candidates
) {
    public PolicyWorldActionBlockSurface {
        unavailableReason = CaptureValidation.optionalText(unavailableReason);
        primaryUnavailableReason = CaptureValidation.optionalText(
            primaryUnavailableReason
        );
        secondaryUnavailableReason = CaptureValidation.optionalText(
            secondaryUnavailableReason
        );
        candidates = candidates == null ? List.of() : List.copyOf(candidates);
        if (
            sourceCount < 0
                || emittedCount < 0
                || emittedCount
                    > NativePolicyWorldActionCapture.BLOCK_CANDIDATE_CAPACITY
                || emittedCount != candidates.size()
                || capacityExceeded != (
                    sourceCount
                        > NativePolicyWorldActionCapture.BLOCK_CANDIDATE_CAPACITY
                )
        ) {
            throw new IllegalArgumentException(
                "Block candidate counts are inconsistent"
            );
        }
        if (available) {
            if (
                !unavailableReason.isEmpty()
                    || capacityExceeded
                    || sourceCount != emittedCount
                    || sourceCount == 0
            ) {
                throw new IllegalArgumentException(
                    "Available block surface is incomplete"
                );
            }
        } else if (
            unavailableReason.isEmpty()
                || emittedCount != 0
                || !candidates.isEmpty()
        ) {
            throw new IllegalArgumentException(
                "Unavailable block surface must fail the complete row closed"
            );
        }
        for (int index = 0; index < candidates.size(); index++) {
            if (candidates.get(index).slot() != index) {
                throw new IllegalArgumentException(
                    "Block candidates must be contiguous and slot ordered"
                );
            }
        }
        boolean primary = candidates.stream().anyMatch(
            candidate -> candidate.primary() != null
        );
        boolean secondary = candidates.stream().anyMatch(
            candidate -> candidate.secondary() != null
        );
        if (
            available
                && ((primary != primaryUnavailableReason.isEmpty())
                    || (secondary != secondaryUnavailableReason.isEmpty()))
        ) {
            throw new IllegalArgumentException(
                "Block trigger reasons do not match their bindings"
            );
        }
    }

    public static PolicyWorldActionBlockSurface unavailable(
        String reason,
        int sourceCount
    ) {
        return new PolicyWorldActionBlockSurface(
            false,
            CaptureValidation.text(reason, "block unavailable reason"),
            sourceCount,
            0,
            sourceCount
                > NativePolicyWorldActionCapture.BLOCK_CANDIDATE_CAPACITY,
            "block_primary_unavailable",
            "rotation_zero_destination_geometry_unavailable",
            List.of()
        );
    }

    public record Candidate(
        int slot,
        int[] visibleCell,
        int[] actionCell,
        double[] visibleRelativePosition,
        NativeMutableBlockEvidence.Row semantics,
        PolicyWorldActionBinding primary,
        PolicyWorldActionBinding secondary
    ) {
        public Candidate {
            visibleCell = CaptureValidation.ints(visibleCell);
            actionCell = CaptureValidation.ints(actionCell);
            visibleRelativePosition = CaptureValidation.doubles(
                visibleRelativePosition
            );
            if (
                slot < 0
                    || slot
                        >= NativePolicyWorldActionCapture.BLOCK_CANDIDATE_CAPACITY
                    || visibleCell.length != 3
                    || actionCell.length != 3
                    || visibleRelativePosition.length != 3
                    || !CaptureValidation.finite(visibleRelativePosition)
                    || semantics == null
                    || !semantics.blockPresent()
                    || !semantics.semanticKeyValid()
                    || !semantics.affordanceValid()
                    || (primary == null && secondary == null)
            ) {
                throw new IllegalArgumentException(
                    "Block candidate is incomplete"
                );
            }
            if (primary != null && !primary.verb().equals("break_block")) {
                throw new IllegalArgumentException(
                    "Primary binding must be break_block"
                );
            }
            if (secondary != null) {
                throw new IllegalArgumentException(
                    "Place remains fail-closed until the rotation-zero "
                        + "destination and placed geometry are exact"
                );
            }
        }

        @Override public int[] visibleCell() { return visibleCell.clone(); }
        @Override public int[] actionCell() { return actionCell.clone(); }
        @Override public double[] visibleRelativePosition() {
            return visibleRelativePosition.clone();
        }
    }
}
