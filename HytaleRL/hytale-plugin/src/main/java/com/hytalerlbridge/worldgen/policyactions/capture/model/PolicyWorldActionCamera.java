package com.hytalerlbridge.worldgen.policyactions.capture.model;

/** Exact server ray used to derive a policy World-action surface. */
public record PolicyWorldActionCamera(
    boolean available,
    String unavailableReason,
    double[] actorPosition,
    double[] eyePosition,
    double[] direction,
    double maximumDistance,
    int[] rawHitCell,
    int[] canonicalActionCell,
    int blockFace
) {
    public PolicyWorldActionCamera {
        unavailableReason = CaptureValidation.optionalText(unavailableReason);
        actorPosition = CaptureValidation.doubles(actorPosition);
        eyePosition = CaptureValidation.doubles(eyePosition);
        direction = CaptureValidation.doubles(direction);
        rawHitCell = CaptureValidation.ints(rawHitCell);
        canonicalActionCell = CaptureValidation.ints(canonicalActionCell);
        if (available) {
            double norm = direction.length == 3
                ? Math.sqrt(
                    direction[0] * direction[0]
                        + direction[1] * direction[1]
                        + direction[2] * direction[2]
                )
                : 0.0;
            if (
                !unavailableReason.isEmpty()
                    || actorPosition.length != 3
                    || eyePosition.length != 3
                    || direction.length != 3
                    || rawHitCell.length != 3
                    || canonicalActionCell.length != 3
                    || !CaptureValidation.finite(actorPosition)
                    || !CaptureValidation.finite(eyePosition)
                    || !CaptureValidation.finite(direction)
                    || !Double.isFinite(maximumDistance)
                    || maximumDistance <= 0.0
                    || Math.abs(norm - 1.0) > 1.0e-8
                    || blockFace < 1
                    || blockFace > 6
            ) {
                throw new IllegalArgumentException(
                    "Available policy camera row is incomplete"
                );
            }
        } else if (
            unavailableReason.isEmpty()
                || actorPosition.length != 0
                || eyePosition.length != 0
                || direction.length != 0
                || rawHitCell.length != 0
                || canonicalActionCell.length != 0
                || maximumDistance != 0.0
                || blockFace != 0
        ) {
            throw new IllegalArgumentException(
                "Unavailable policy camera row must be structurally empty"
            );
        }
    }

    @Override public double[] actorPosition() { return actorPosition.clone(); }
    @Override public double[] eyePosition() { return eyePosition.clone(); }
    @Override public double[] direction() { return direction.clone(); }
    @Override public int[] rawHitCell() { return rawHitCell.clone(); }
    @Override public int[] canonicalActionCell() {
        return canonicalActionCell.clone();
    }

    public static PolicyWorldActionCamera unavailable(String reason) {
        return new PolicyWorldActionCamera(
            false,
            CaptureValidation.text(reason, "camera unavailable reason"),
            new double[0],
            new double[0],
            new double[0],
            0.0,
            new int[0],
            new int[0],
            0
        );
    }
}
