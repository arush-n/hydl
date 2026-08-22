package com.hytalerlbridge.nativebackend.model;


/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record CapturedRoleOpacity(
    boolean available,
    boolean[] cellMask
) {
    public CapturedRoleOpacity {
        cellMask = cellMask.clone();
    }

    @Override
    public boolean[] cellMask() {
        return cellMask.clone();
    }
}
