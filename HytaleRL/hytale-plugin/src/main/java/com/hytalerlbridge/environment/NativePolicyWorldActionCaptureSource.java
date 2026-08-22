package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.policyactions.capture.PolicyWorldActionCaptureRequest;
import com.hytalerlbridge.worldgen.policyactions.capture.model.NativePolicyWorldActionCapture;

/** Optional native capability for atomic policy World-action capture. */
public interface NativePolicyWorldActionCaptureSource {

    NativePolicyWorldActionCapture capturePolicyWorldActions(
        PolicyWorldActionCaptureRequest request
    );
}
