package com.hytalerlbridge.imitation;

import java.util.UUID;

/** One post-filter server damage event, viewed from the traced NPC. */
public record NpcDamageEventSnapshot(
    String sourceType,
    String environmentType,
    int damageCauseIndex,
    String damageCauseId,
    float initialAmount,
    float finalAmount,
    boolean cancelled,
    boolean blocked,
    boolean actorSource,
    boolean actorTarget,
    int sourceEntityIndex,
    UUID sourceUuid,
    int targetEntityIndex,
    UUID targetUuid,
    int projectileEntityIndex,
    UUID projectileUuid,
    boolean hitLocationAvailable,
    double[] hitLocation,
    boolean healthAvailable,
    float targetHealthAfter,
    float targetMaxHealth,
    boolean lethal
) {
    public static final int WIDTH = 22;
    public static final int CAPACITY = 32;
    public static final String LAYOUT =
        "source_type:utf8,environment_type:utf8,damage_cause_index:i32,"
            + "damage_cause_id:utf8,initial_amount:f32,final_amount:f32,"
            + "cancelled:bool,blocked:bool,actor_source:bool,actor_target:bool,"
            + "source_entity_index:i32,source_uuid:uuid?,target_entity_index:i32,"
            + "target_uuid:uuid?,projectile_entity_index:i32,projectile_uuid:uuid?,"
            + "hit_location_available:bool,hit_location:f64[4],health_available:bool,"
            + "target_health_after:f32,target_max_health:f32,lethal:bool";

    public NpcDamageEventSnapshot {
        sourceType = text(sourceType);
        environmentType = text(environmentType);
        damageCauseId = text(damageCauseId);
        if (!actorSource && !actorTarget) {
            throw new IllegalArgumentException("damage event must involve the traced NPC");
        }
        if (
            !Float.isFinite(initialAmount)
                || initialAmount < 0.0f
                || !Float.isFinite(finalAmount)
                || finalAmount < 0.0f
        ) {
            throw new IllegalArgumentException("damage amounts must be finite and nonnegative");
        }
        hitLocation = finiteRow(hitLocation, 4, "hitLocation");
        if (!hitLocationAvailable && anyNonzero(hitLocation)) {
            throw new IllegalArgumentException("unavailable hit location must be zero");
        }
        if (
            !Float.isFinite(targetHealthAfter)
                || !Float.isFinite(targetMaxHealth)
                || targetHealthAfter < 0.0f
                || targetMaxHealth < 0.0f
                || (!healthAvailable && (targetHealthAfter != 0.0f || targetMaxHealth != 0.0f))
                || (healthAvailable && targetHealthAfter > targetMaxHealth)
                || (lethal && (!healthAvailable || targetHealthAfter > 0.0f))
        ) {
            throw new IllegalArgumentException("damage health outcome is inconsistent");
        }
    }

    private static double[] finiteRow(double[] value, int width, String name) {
        if (value == null || value.length != width) {
            throw new IllegalArgumentException(name + " must have width " + width);
        }
        double[] copy = value.clone();
        for (double scalar : copy) {
            if (!Double.isFinite(scalar)) {
                throw new IllegalArgumentException(name + " must be finite");
            }
        }
        return copy;
    }

    private static boolean anyNonzero(double[] values) {
        for (double value : values) if (value != 0.0) return true;
        return false;
    }

    private static String text(String value) {
        return value == null ? "" : value;
    }
}
