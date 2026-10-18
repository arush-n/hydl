package com.hytalerlbridge.worldgen;

import java.util.Comparator;
import java.util.List;
import java.util.UUID;

/** Bounded native entity-damage fixture for Hytale explosion dynamics. */
public record NativeExplosionDynamicsProbe(
    String serverVersion,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    String worldEpoch,
    double originX,
    double originY,
    double originZ,
    float entityDamageRadius,
    float entityDamage,
    float entityDamageFalloff,
    int capacity,
    int totalMatching,
    boolean overflow,
    boolean complete,
    boolean sessionResetRequired,
    String failureReason,
    List<EntityEffect> effects
) {
    public static final String SCHEMA =
        "hytalerl_native_explosion_dynamics_probe_v1";
    public static final int VERSION = 1;
    public static final int MAX_CAPACITY = 8;
    public static final float ENTITY_DAMAGE_RADIUS = 5.0f;
    public static final float ENTITY_DAMAGE = 8.0f;
    public static final float ENTITY_DAMAGE_FALLOFF = 2.0f;
    public static final String FIXTURE_KIND = "open_two_entity";
    public static final String PHASE_ORDER =
        "processTargetBlocks_then_processTargetEntities_then_effects";
    public static final String ROW_ORDER =
        "UUID_text_ascending_serialization_not_native_execution_order";

    public NativeExplosionDynamicsProbe {
        requireText(serverVersion, "serverVersion");
        requireText(worldName, "worldName");
        requireText(worldgenProvider, "worldgenProvider");
        requireText(worldgenVersion, "worldgenVersion");
        requireText(worldEpoch, "worldEpoch");
        if (
            !Double.isFinite(originX)
                || !Double.isFinite(originY)
                || !Double.isFinite(originZ)
        ) {
            throw new IllegalArgumentException("origin must be finite");
        }
        if (
            entityDamageRadius != ENTITY_DAMAGE_RADIUS
                || entityDamage != ENTITY_DAMAGE
                || entityDamageFalloff != ENTITY_DAMAGE_FALLOFF
        ) {
            throw new IllegalArgumentException(
                "v1 explosion dynamics configuration is server-owned"
            );
        }
        if (capacity < 1 || capacity > MAX_CAPACITY) {
            throw new IllegalArgumentException("capacity must be 1..8");
        }
        if (totalMatching < 0) {
            throw new IllegalArgumentException(
                "totalMatching must be non-negative"
            );
        }
        if (overflow != (totalMatching > capacity)) {
            throw new IllegalArgumentException(
                "overflow must exactly reflect totalMatching > capacity"
            );
        }
        failureReason = failureReason == null ? "" : failureReason;
        effects = effects == null ? List.of() : List.copyOf(effects);
        if (
            complete
                != (!overflow
                    && failureReason.isEmpty()
                    && effects.size() == totalMatching)
        ) {
            throw new IllegalArgumentException(
                "complete must exactly describe a full bounded frame"
            );
        }
        if (!complete && !effects.isEmpty()) {
            throw new IllegalArgumentException(
                "incomplete frames must not publish partial effects"
            );
        }
        if (!sessionResetRequired) {
            throw new IllegalArgumentException(
                "native damage fixture always requires a reset"
            );
        }
        UUID previous = null;
        for (int index = 0; index < effects.size(); index++) {
            EntityEffect effect = effects.get(index);
            if (effect == null || effect.ordinal() != index) {
                throw new IllegalArgumentException(
                    "effects must have consecutive ordinals"
                );
            }
            if (
                previous != null
                    && UUID_TEXT_ORDER.compare(previous, effect.uuid()) >= 0
            ) {
                throw new IllegalArgumentException(
                    "effects must have unique UUID order"
                );
            }
            previous = effect.uuid();
        }
    }

    public record EntityEffect(
        int ordinal,
        UUID uuid,
        double x,
        double y,
        double z,
        double distance,
        float healthBefore,
        float healthAfter,
        float expectedRawDamage
    ) {
        public EntityEffect {
            if (ordinal < 0 || uuid == null) {
                throw new IllegalArgumentException(
                    "effect identity must be present"
                );
            }
            if (
                !Double.isFinite(x)
                    || !Double.isFinite(y)
                    || !Double.isFinite(z)
                    || !Double.isFinite(distance)
                    || distance < 0.0
                    || !Float.isFinite(healthBefore)
                    || !Float.isFinite(healthAfter)
                    || !Float.isFinite(expectedRawDamage)
                    || expectedRawDamage < 0.0f
            ) {
                throw new IllegalArgumentException(
                    "effect values must be finite and non-negative"
                );
            }
        }
    }

    private static final Comparator<UUID> UUID_TEXT_ORDER =
        Comparator.comparing(UUID::toString);

    private static void requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
    }
}
