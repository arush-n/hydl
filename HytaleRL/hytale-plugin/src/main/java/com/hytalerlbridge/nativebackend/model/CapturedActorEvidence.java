package com.hytalerlbridge.nativebackend.model;

import com.hytalerlbridge.observation.NativeActorEvidenceFrame;

/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record CapturedActorEvidence(
    NativeActorEvidenceFrame.Actor actor,
    boolean statusOverflow,
    boolean statusInvalid
) {}
