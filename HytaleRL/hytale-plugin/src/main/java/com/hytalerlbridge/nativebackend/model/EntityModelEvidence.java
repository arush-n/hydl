package com.hytalerlbridge.nativebackend.model;

import java.util.Map;

/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record EntityModelEvidence(
    boolean modelPresent,
    String modelAssetId,
    double modelScale,
    double modelEyeHeight,
    boolean boundingBoxPresent,
    boolean entityScalePresent,
    double entityScale
) {
    public static EntityModelEvidence absent() {
        return new EntityModelEvidence(
            false,
            "",
            0.0,
            0.0,
            false,
            false,
            0.0
        );
    }

    public void putInto(Map<String, Object> info, String subject) {
        info.put(subject + "_model_present", modelPresent);
        info.put(subject + "_model_asset_id", modelAssetId);
        info.put(subject + "_model_scale", modelScale);
        info.put(subject + "_model_eye_height", modelEyeHeight);
        info.put(subject + "_bounding_box_present", boundingBoxPresent);
        info.put(subject + "_entity_scale_present", entityScalePresent);
        info.put(subject + "_entity_scale", entityScale);
    }
}
