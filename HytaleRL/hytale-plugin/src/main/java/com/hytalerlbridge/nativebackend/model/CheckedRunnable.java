package com.hytalerlbridge.nativebackend.model;


/** Extracted verbatim from {@code NativeEnvironmentSession}. */
@FunctionalInterface
public interface CheckedRunnable {
    public void run() throws Exception;
}
