package com.hytalerlbridge.nativebackend.model;


/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record EntityReading(int[] encoded, double distanceSquared, int entityIndex) {}
