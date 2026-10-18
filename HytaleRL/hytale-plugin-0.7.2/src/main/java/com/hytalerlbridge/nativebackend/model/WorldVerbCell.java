package com.hytalerlbridge.nativebackend.model;


/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record WorldVerbCell(
    String semanticBlockId,
    int runtimeBlockId,
    double blockHealth,
    String canonical
) {}
