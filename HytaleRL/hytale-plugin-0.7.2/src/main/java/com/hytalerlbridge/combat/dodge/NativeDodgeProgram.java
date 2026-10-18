package com.hytalerlbridge.combat.dodge;

/**
 * Hytale 0.5.7's authored dodge-direction surface.
 *
 * <p>The public action uses {@code 0=none, 1=forward, 2=back, 3=left,
 * 4=right}. The shipped {@code Dodge} movement-condition asset routes only
 * Left and Right to payload interactions; the other authored branches are
 * {@code Simple} no-ops. Each lateral payload applies
 * {@code Dodge_Invulnerability} before lateral force, so it is both evasion
 * and displacement. Keeping that asset mapping here prevents the native
 * session from growing a second policy contract.</p>
 */
public final class NativeDodgeProgram {

    public static final int DIRECTION_COUNT = 4;
    public static final int LEFT_DIRECTION = 3;
    public static final int RIGHT_DIRECTION = 4;
    public static final String LEFT_INTERACTION_ID = "Dodge_Left";
    public static final String RIGHT_INTERACTION_ID = "Dodge_Right";
    /** The public player root owns one cooldown shared by both directions. */
    public static final String COOLDOWN_ID = "Dodge";
    /** Hytale 0.5.7 InteractionTypeUtils default for InteractionType.Dodge. */
    public static final float COOLDOWN_SECONDS = 0.35F;
    public static final String INVULNERABILITY_EFFECT_ID =
        "Dodge_Invulnerability";

    private NativeDodgeProgram() {}

    public static boolean isAuthored(int direction) {
        return direction == LEFT_DIRECTION || direction == RIGHT_DIRECTION;
    }

    public static boolean isAvailable(
        int direction,
        boolean leftAvailable,
        boolean rightAvailable
    ) {
        return switch (direction) {
            case LEFT_DIRECTION -> leftAvailable;
            case RIGHT_DIRECTION -> rightAvailable;
            default -> false;
        };
    }

    public static String interactionId(int direction) {
        return switch (direction) {
            case LEFT_DIRECTION -> LEFT_INTERACTION_ID;
            case RIGHT_DIRECTION -> RIGHT_INTERACTION_ID;
            default -> "";
        };
    }

    /**
     * Return policy-mask order [forward, back, left, right].
     */
    public static boolean[] actionMask(
        boolean leftAvailable,
        boolean rightAvailable
    ) {
        return new boolean[] {
            false,
            false,
            leftAvailable,
            rightAvailable
        };
    }
}
