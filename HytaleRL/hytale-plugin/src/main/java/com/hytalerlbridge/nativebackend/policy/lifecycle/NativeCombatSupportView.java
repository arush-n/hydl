package com.hytalerlbridge.nativebackend.policy.lifecycle;

import com.hypixel.hytale.server.npc.role.support.CombatSupport;
import java.lang.reflect.Field;

/** Read-only access to the engine-owned NPC attack-pause clock. */
public final class NativeCombatSupportView {
    private static final Field ATTACK_PAUSE = field(
        CombatSupport.class,
        "attackPause"
    );

    private NativeCombatSupportView() {}

    public record State(
        boolean available,
        String reason,
        float attackPauseSeconds
    ) {
        public State {
            reason = reason == null ? "" : reason;
            if (available != reason.isBlank()) {
                throw new IllegalArgumentException(
                    "exactly one of available and reason is required"
                );
            }
            if (
                !Float.isFinite(attackPauseSeconds)
                    || attackPauseSeconds < 0.0f
            ) {
                throw new IllegalArgumentException(
                    "attackPauseSeconds must be finite and nonnegative"
                );
            }
        }

        private static State unavailable(String reason) {
            return new State(false, reason, 0.0f);
        }
    }

    /** Source-layout gate used before a bridge candidate is deployed. */
    public static boolean layoutAvailable() {
        return ATTACK_PAUSE != null;
    }

    /** Capture the current clock without advancing or resetting it. */
    public static State capture(CombatSupport support) {
        if (support == null) {
            return new State(true, "", 0.0f);
        }
        if (!layoutAvailable()) {
            return State.unavailable("combat_support_layout_unavailable");
        }
        try {
            double raw = ATTACK_PAUSE.getDouble(support);
            if (!Double.isFinite(raw)) {
                return State.unavailable("combat_support_clock_invalid");
            }
            float seconds = (float) Math.max(0.0, raw);
            return Float.isFinite(seconds)
                ? new State(true, "", seconds)
                : State.unavailable("combat_support_clock_invalid");
        } catch (IllegalAccessException | RuntimeException unavailable) {
            return State.unavailable("combat_support_read_failed");
        }
    }

    private static Field field(Class<?> owner, String name) {
        try {
            Field value = owner.getDeclaredField(name);
            return value.trySetAccessible() ? value : null;
        } catch (NoSuchFieldException | RuntimeException unavailable) {
            return null;
        }
    }
}

