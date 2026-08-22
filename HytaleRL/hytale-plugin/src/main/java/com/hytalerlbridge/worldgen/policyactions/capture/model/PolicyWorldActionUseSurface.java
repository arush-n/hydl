package com.hytalerlbridge.worldgen.policyactions.capture.model;

/** Exact current Use edge, or one typed unavailable reason. */
public record PolicyWorldActionUseSurface(
    boolean available,
    String unavailableReason,
    PolicyWorldActionBinding binding
) {
    public PolicyWorldActionUseSurface {
        unavailableReason = CaptureValidation.optionalText(unavailableReason);
        if (available) {
            if (
                !unavailableReason.isEmpty()
                    || binding == null
                    || !binding.verb().equals("use")
            ) {
                throw new IllegalArgumentException(
                    "Available Use surface is incomplete"
                );
            }
        } else if (unavailableReason.isEmpty() || binding != null) {
            throw new IllegalArgumentException(
                "Unavailable Use surface must be structurally empty"
            );
        }
    }

    public static PolicyWorldActionUseSurface unavailable(String reason) {
        return new PolicyWorldActionUseSurface(
            false,
            CaptureValidation.text(reason, "Use unavailable reason"),
            null
        );
    }
}
