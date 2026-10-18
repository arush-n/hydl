package com.hytalerlbridge.combat.dodge;

import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.modules.interaction.interaction.CooldownHandler;
import java.lang.reflect.Field;

/** Read-only, fail-closed view of the engine-owned shared Dodge cooldown. */
public final class NativeDodgeCooldownView {
    private static final Field HANDLER_FIELD = field(
        InteractionManager.class,
        "cooldownHandler"
    );

    private NativeDodgeCooldownView() {}

    public record State(boolean available, boolean onCooldown) {
        static State unavailable() {
            return new State(false, true);
        }
    }

    public static boolean layoutAvailable() {
        return HANDLER_FIELD != null;
    }

    public static State capture(InteractionManager manager) {
        if (manager == null || HANDLER_FIELD == null) return State.unavailable();
        try {
            Object value = HANDLER_FIELD.get(manager);
            return value instanceof CooldownHandler handler
                ? capture(handler)
                : State.unavailable();
        } catch (IllegalAccessException | RuntimeException unavailable) {
            return State.unavailable();
        }
    }

    /** Package-visible server-free gate; getCooldown(String) does not create state. */
    static State capture(CooldownHandler handler) {
        if (handler == null) return State.unavailable();
        CooldownHandler.Cooldown cooldown = handler.getCooldown(
            NativeDodgeProgram.COOLDOWN_ID
        );
        return new State(
            true,
            cooldown != null && cooldown.hasCooldown(false)
        );
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
