package com.hytalerlbridge.nativebackend.model;


/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record RegionBlockSemanticKey(
    String assetId,
    int rotationIndex,
    int affordanceTags,
    int gatherTypeIndex,
    int requiredToolQuality
) {}
