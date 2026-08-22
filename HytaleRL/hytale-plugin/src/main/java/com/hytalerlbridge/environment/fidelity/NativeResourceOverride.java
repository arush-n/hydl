package com.hytalerlbridge.environment.fidelity;

import com.hytalerlbridge.observation.NativeActorEvidenceFrame;
import java.util.Locale;
import java.util.Set;

/**
 * Explicit reset state for a resource-gated native fidelity discriminator.
 *
 * <p>This is fixture setup, not a policy action. The allow-list is the same
 * fixed resource vocabulary carried by native actor evidence, so callers
 * cannot address arbitrary engine stats or item-specific fields.</p>
 */
public record NativeResourceOverride(
    int entityId,
    String resourceId,
    float value
) {
    public static final String SCHEMA =
        "hytalerl_native_fidelity_resource_override_v1";
    public static final int VERSION = 1;

    private static final Set<String> RESOURCE_IDS = Set.of(
        "Stamina",
        "Mana",
        "MagicCharges",
        "SignatureEnergy",
        "SignatureCharges",
        "Ammo",
        "Oxygen"
    );

    public NativeResourceOverride {
        if (entityId < 0 || entityId >= NativeActorEvidenceFrame.ENTITY_COUNT) {
            throw new IllegalArgumentException(
                "native resource override entity_id is outside actor capacity"
            );
        }
        if (resourceId == null || resourceId.isBlank()) {
            throw new IllegalArgumentException(
                "native resource override resource_id must be nonblank"
            );
        }
        resourceId = canonicalResourceId(resourceId.trim());
        if (!Float.isFinite(value) || value < 0.0f) {
            throw new IllegalArgumentException(
                "native resource override value must be finite and non-negative"
            );
        }
    }

    private static String canonicalResourceId(String value) {
        for (String resourceId : RESOURCE_IDS) {
            if (resourceId.toLowerCase(Locale.ROOT).equals(
                value.toLowerCase(Locale.ROOT)
            )) {
                return resourceId;
            }
        }
        throw new IllegalArgumentException(
            "unknown native resource override resource_id: " + value
        );
    }
}
