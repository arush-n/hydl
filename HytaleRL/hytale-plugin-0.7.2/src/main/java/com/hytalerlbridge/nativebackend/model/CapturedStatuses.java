package com.hytalerlbridge.nativebackend.model;

import com.hytalerlbridge.observation.NativeActorEvidenceFrame;
import java.util.List;

/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record CapturedStatuses(
    List<NativeActorEvidenceFrame.Status> statuses,
    boolean overflow,
    boolean invalid
) {}
