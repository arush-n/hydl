package com.hytalerlbridge.imitation;

/** Exact lifecycle fields for one selected native ActionAttack. */
public record NpcAttackActionSnapshot(
    String label,
    boolean active,
    boolean triggered,
    boolean ready,
    float aimingSecondsRemaining,
    float chargeSeconds,
    String interactionType,
    String interactionId,
    String path
) {
    public static final String LAYOUT =
        "label:utf8,active:bool,triggered:bool,ready:bool,"
            + "aiming_seconds_remaining:f32,charge_seconds:f32,"
            + "interaction_type:utf8,interaction_id:utf8,path:utf8";
    public static final int WIDTH = 9;

    public NpcAttackActionSnapshot(
        String label,
        boolean active,
        boolean triggered,
        boolean ready,
        float aimingSecondsRemaining,
        float chargeSeconds,
        String interactionType,
        String interactionId
    ) {
        this(
            label,
            active,
            triggered,
            ready,
            aimingSecondsRemaining,
            chargeSeconds,
            interactionType,
            interactionId,
            ""
        );
    }

    public NpcAttackActionSnapshot {
        label = text(label);
        interactionType = text(interactionType);
        interactionId = text(interactionId);
        path = text(path);
        if (
            !Float.isFinite(aimingSecondsRemaining)
                || !Float.isFinite(chargeSeconds)
                || chargeSeconds < 0.0f
        ) {
            throw new IllegalArgumentException("attack action times must be finite");
        }
    }

    private static String text(String value) {
        return value == null ? "" : value;
    }
}
