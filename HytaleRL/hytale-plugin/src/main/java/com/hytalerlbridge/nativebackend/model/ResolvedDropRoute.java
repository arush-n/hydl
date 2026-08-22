package com.hytalerlbridge.nativebackend.model;


/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record ResolvedDropRoute(
    int quantity,
    String itemId,
    String dropListId
) {

    public ResolvedDropRoute {
        if (quantity < 0) {
            throw new IllegalArgumentException(
                "Native drop quantity cannot be negative"
            );
        }
    }
}
