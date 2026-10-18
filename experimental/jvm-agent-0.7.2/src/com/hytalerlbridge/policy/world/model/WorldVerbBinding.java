package com.hytalerlbridge.policy.world.model;

import java.util.Set;

/**
 * Commit-ready, actor-privileged identity behind one visible World-action edge.
 *
 * <p>The learner never sees this value. It is captured beside the policy row,
 * retained while a decision is held, and compared with a fresh capture before
 * the reflective bridge request is constructed. Request ids and world epochs
 * remain commit-time values and are deliberately absent here.</p>
 */
public record WorldVerbBinding(
    boolean available,
    String verb,
    int interactionType,
    String interactionId,
    String targetBlockInteractionId,
    String itemId,
    String blockId,
    int targetX,
    int targetY,
    int targetZ,
    int blockFace,
    int rotationYaw,
    int rotationPitch,
    int rotationRoll,
    String sourceContainer,
    int sourceSlot,
    int expectedSourceQuantity,
    String expectedBlockId,
    String placementVariant
) {
    private static final Set<String> VERBS = Set.of(
        "use", "break_block", "place_block");

    public WorldVerbBinding {
        verb = canonical(verb);
        interactionId = canonical(interactionId);
        targetBlockInteractionId = canonical(targetBlockInteractionId);
        itemId = canonical(itemId);
        blockId = canonical(blockId);
        sourceContainer = canonical(sourceContainer);
        expectedBlockId = canonical(expectedBlockId);
        placementVariant = canonical(placementVariant);
        if (!available) {
            if (!verb.isEmpty() || interactionType != -1
                || !interactionId.isEmpty()
                || !targetBlockInteractionId.isEmpty()
                || !itemId.isEmpty() || !blockId.isEmpty()
                || targetX != 0 || targetY != 0 || targetZ != 0
                || blockFace != 0 || rotationYaw != 0
                || rotationPitch != 0 || rotationRoll != 0
                || !sourceContainer.isEmpty() || sourceSlot != -1
                || expectedSourceQuantity != -1
                || !expectedBlockId.isEmpty()
                || !placementVariant.isEmpty()) {
                throw new IllegalArgumentException(
                    "unavailable World-verb binding must be canonical empty");
            }
        } else {
            if (!VERBS.contains(verb) || interactionId.isEmpty()
                || itemId.isEmpty() || expectedBlockId.isEmpty()
                || targetY < 0 || targetY >= 320
                || blockFace < 0 || blockFace > 6
                || rotationYaw < 0 || rotationYaw > 3
                || rotationPitch < 0 || rotationPitch > 3
                || rotationRoll < 0 || rotationRoll > 3) {
                throw new IllegalArgumentException("invalid World-verb binding");
            }
            if (verb.equals("use")) {
                boolean unarmed = sourceContainer.equals("unarmed");
                if (interactionType != 5 || targetBlockInteractionId.isEmpty()
                    || !blockId.isEmpty() || !placementVariant.isEmpty()
                    || blockFace != 0 || rotationYaw != 0
                    || rotationPitch != 0 || rotationRoll != 0
                    || (unarmed && (!itemId.equals("Empty") || sourceSlot != -1
                        || expectedSourceQuantity != 0))
                    || (!unarmed
                        && (!sourceContainer.equals("interaction_context")
                            || sourceSlot < 0
                            || expectedSourceQuantity <= 0))) {
                    throw new IllegalArgumentException("invalid Use binding");
                }
            } else {
                if (!targetBlockInteractionId.isEmpty()
                    || !sourceContainer.equals("hotbar") || sourceSlot < 0
                    || expectedSourceQuantity <= 0
                    || (interactionType != 0 && interactionType != 1)) {
                    throw new IllegalArgumentException(
                        "invalid block-action binding");
                }
                if (verb.equals("place_block")) {
                    if (blockId.isEmpty()
                        || !placementVariant.equals("default")
                        || blockFace == 0
                        || !expectedBlockId.equals("Empty")) {
                        throw new IllegalArgumentException(
                            "invalid Place binding");
                    }
                } else if (!blockId.isEmpty()
                    || !placementVariant.isEmpty()) {
                    throw new IllegalArgumentException("invalid Break binding");
                }
            }
        }
    }

    public static WorldVerbBinding empty() {
        return new WorldVerbBinding(
            false, "", -1, "", "", "", "",
            0, 0, 0, 0, 0, 0, 0, "", -1, -1, "", "");
    }

    private static String canonical(String value) {
        if (value == null) return "";
        if (!value.equals(value.trim()) || value.indexOf('\0') >= 0) {
            throw new IllegalArgumentException(
                "World-verb identities must be canonical strings");
        }
        return value;
    }
}
