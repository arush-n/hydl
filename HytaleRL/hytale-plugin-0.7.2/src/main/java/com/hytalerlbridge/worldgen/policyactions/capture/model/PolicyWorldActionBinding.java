package com.hytalerlbridge.worldgen.policyactions.capture.model;

import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence;
import java.util.List;

/** Privileged source/root/target identity for one candidate trigger. */
public record PolicyWorldActionBinding(
    String verb,
    int interactionType,
    String interactionId,
    String blockInteractionId,
    int[] target,
    int blockFace,
    boolean rotationApplicable,
    int[] rotation,
    String sourceContainer,
    String resolvedSourceContainer,
    int sourceSlot,
    int sourceQuantity,
    String sourceItemId,
    String sourceBlockId,
    String expectedInteractionToolId,
    String expectedBlockId,
    byte[] expectedSemanticKeySha256
) {
    public PolicyWorldActionBinding {
        verb = CaptureValidation.choice(
            verb,
            "verb",
            List.of("use", "break_block", "place_block")
        );
        interactionId = CaptureValidation.text(interactionId, "interactionId");
        blockInteractionId = CaptureValidation.optionalText(blockInteractionId);
        target = CaptureValidation.ints(target);
        rotation = CaptureValidation.ints(rotation);
        sourceContainer = CaptureValidation.text(
            sourceContainer,
            "sourceContainer"
        );
        resolvedSourceContainer = CaptureValidation.text(
            resolvedSourceContainer,
            "resolvedSourceContainer"
        );
        sourceItemId = CaptureValidation.text(sourceItemId, "sourceItemId");
        sourceBlockId = CaptureValidation.optionalText(sourceBlockId);
        expectedInteractionToolId = CaptureValidation.optionalText(
            expectedInteractionToolId
        );
        expectedBlockId = CaptureValidation.text(
            expectedBlockId,
            "expectedBlockId"
        );
        expectedSemanticKeySha256 = CaptureValidation.bytes(
            expectedSemanticKeySha256
        );
        if (
            interactionType < 0
                || interactionType >= NativeItemInteractionEvidence.TRIGGER_CAPACITY
                || target.length != 3
                || blockFace < 1
                || blockFace > 6
                || rotation.length != 3
                || CaptureValidation.anyOutside(rotation, 0, 3)
                || (!rotationApplicable
                    && CaptureValidation.anyNonzero(rotation))
                || sourceSlot < -1
                || sourceQuantity < 0
                || expectedSemanticKeySha256.length != 32
        ) {
            throw new IllegalArgumentException(
                "World-action binding is outside its domain"
            );
        }
        if (verb.equals("place_block")) {
            // Mirrors the Python decoder's rotation-zero Place subset
            // (world_action_capture.py, rotation_zero_place): an explicit
            // (0, 0, 0) rotation into an exact empty destination.
            if (
                !rotationApplicable
                    || CaptureValidation.anyNonzero(rotation)
                    || !expectedBlockId.equals("Empty")
                    || sourceSlot < 0
            ) {
                throw new IllegalArgumentException(
                    "Place is admitted only for the rotation-zero subset "
                        + "into an exact empty cell"
                );
            }
        }
        if (
            verb.equals("break_block")
                && (!sourceContainer.equals("hotbar")
                    || !resolvedSourceContainer.equals("hotbar")
                    || sourceSlot < 0
                    || sourceQuantity < 1
                    || expectedInteractionToolId.isEmpty())
        ) {
            throw new IllegalArgumentException(
                "Break requires an exact hotbar source and authored tool"
            );
        }
        if (verb.equals("use")) {
            boolean unarmed = sourceContainer.equals("unarmed");
            boolean held = sourceContainer.equals("interaction_context");
            if (
                interactionType != 5
                    || (!unarmed && !held)
                    || (!resolvedSourceContainer.equals("hotbar")
                        && !resolvedSourceContainer.equals("tools"))
                    || (unarmed && (!sourceItemId.equals("Empty")
                        || sourceSlot != -1
                        || sourceQuantity != 0
                        ))
                    || (held && (sourceItemId.equals("Empty")
                        || sourceSlot < 0
                        || sourceQuantity < 1
                        || (!resolvedSourceContainer.equals("hotbar")
                            && !resolvedSourceContainer.equals("tools"))))
            ) {
                throw new IllegalArgumentException(
                    "Use requires one exact native held-item source"
                );
            }
        }
    }

    @Override public int[] target() { return target.clone(); }
    @Override public int[] rotation() { return rotation.clone(); }
    @Override public byte[] expectedSemanticKeySha256() {
        return expectedSemanticKeySha256.clone();
    }
}
