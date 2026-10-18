package com.hytalerlbridge.nativebackend.policy.lifecycle;

import com.hypixel.hytale.protocol.GameMode;
import com.hypixel.hytale.protocol.InteractionCooldown;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.protocol.RootInteractionSettings;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.modules.interaction.interaction.CooldownHandler;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import java.lang.reflect.Field;

/**
 * Version-gated, read-only access to the engine-owned cooldown clock.
 *
 * <p>Hytale 0.5.7 exposes cooldown lookup and availability publicly, but keeps
 * the manager's handler and the current remaining seconds private. The bridge
 * already uses the same guarded-reflection pattern for motion-force channels.
 * This class caches only immutable {@link Field} metadata; it retains no
 * gameplay state and never calls a deducting cooldown method.</p>
 */
public final class NativePolicyCooldownView {
    private static final Field HANDLER_FIELD = field(
        InteractionManager.class,
        "cooldownHandler"
    );
    private static final Field REMAINING_FIELD = field(
        CooldownHandler.Cooldown.class,
        "remainingCooldown"
    );

    private NativePolicyCooldownView() {}

    public record State(
        boolean available,
        float remainingSeconds,
        boolean onCooldown
    ) {
        public State {
            if (
                !Float.isFinite(remainingSeconds)
                    || remainingSeconds < 0.0f
            ) {
                throw new IllegalArgumentException(
                    "remainingSeconds must be finite and nonnegative"
                );
            }
        }

        public static State unavailable() {
            return new State(false, 0.0f, true);
        }
    }

    /** Source-layout gate used before a bridge candidate is deployed. */
    public static boolean layoutAvailable() {
        return HANDLER_FIELD != null && REMAINING_FIELD != null;
    }

    public static State capture(
        InteractionManager manager,
        RootInteraction root,
        InteractionType type
    ) {
        if (manager == null || root == null || type == null || !layoutAvailable()) {
            return State.unavailable();
        }
        try {
            Object rawHandler = HANDLER_FIELD.get(manager);
            if (!(rawHandler instanceof CooldownHandler handler)) {
                return State.unavailable();
            }
            String id = cooldownId(root, GameMode.Adventure);
            CooldownHandler.Cooldown cooldown = handler.getCooldown(id);
            return cooldown == null
                ? new State(true, 0.0f, false)
                : capture(cooldown);
        } catch (IllegalAccessException | RuntimeException unavailable) {
            return State.unavailable();
        }
    }

    /** Package-visible pure read used by the server-free layout test. */
    static State capture(CooldownHandler.Cooldown cooldown) {
        if (cooldown == null || REMAINING_FIELD == null) {
            return State.unavailable();
        }
        try {
            float remaining = Math.max(0.0f, REMAINING_FIELD.getFloat(cooldown));
            if (!Float.isFinite(remaining)) return State.unavailable();
            // false is the documented non-deducting branch.
            return new State(true, remaining, cooldown.hasCooldown(false));
        } catch (IllegalAccessException | RuntimeException unavailable) {
            return State.unavailable();
        }
    }

    private static String cooldownId(RootInteraction root, GameMode mode) {
        InteractionCooldown base = root.getCooldown();
        RootInteractionSettings settings = root.getSettings().get(mode);
        InteractionCooldown override = settings == null
            ? null
            : settings.cooldown;
        return cooldownId(
            root.getId(),
            base == null ? null : base.cooldownId,
            override == null ? null : override.cooldownId
        );
    }

    /**
     * Package-visible precedence gate mirroring
     * {@code InteractionManager.isOnCooldown} exactly.
     */
    static String cooldownId(
        String rootId,
        String rootOverride,
        String gameModeOverride
    ) {
        String result = rootId;
        if (rootOverride != null) result = rootOverride;
        if (gameModeOverride != null) result = gameModeOverride;
        return result;
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

