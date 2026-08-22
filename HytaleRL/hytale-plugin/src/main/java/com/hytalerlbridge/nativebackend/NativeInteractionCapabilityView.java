package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.protocol.GameMode;
import com.hypixel.hytale.protocol.InteractionCooldown;
import com.hypixel.hytale.protocol.RootInteractionSettings;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.modules.interaction.interaction.CooldownHandler;
import com.hytalerlbridge.nativebackend.model.NativeInteractionBinding;
import java.lang.reflect.Field;

/** Read-only view of native rule, cooldown, and charge admission. */
public final class NativeInteractionCapabilityView {
    private static final Field HANDLER_FIELD = field(
        InteractionManager.class,
        "cooldownHandler"
    );

    private NativeInteractionCapabilityView() {}

    public record State(boolean available, boolean legal) {
        static State unavailable() {
            return new State(false, false);
        }
    }

    public static State capture(
        InteractionManager manager,
        NativeInteractionBinding binding
    ) {
        if (manager == null || binding == null || HANDLER_FIELD == null) {
            return State.unavailable();
        }
        try {
            Object value = HANDLER_FIELD.get(manager);
            if (!(value instanceof CooldownHandler handler)) {
                return State.unavailable();
            }
            return capture(
                handler,
                binding,
                manager.canRun(binding.type(), binding.root())
            );
        } catch (IllegalAccessException | RuntimeException unavailable) {
            return State.unavailable();
        }
    }

    /** Package-visible server-free gate used by the contract tests. */
    static State capture(
        CooldownHandler handler,
        NativeInteractionBinding binding,
        boolean rulesLegal
    ) {
        if (handler == null || binding == null) return State.unavailable();
        if (!rulesLegal) return new State(true, false);

        InteractionCooldown configured = binding.root().getCooldown();
        var settings = binding.root().getSettings();
        RootInteractionSettings adventure = settings == null
            ? null
            : settings.get(GameMode.Adventure);
        if (adventure != null && adventure.cooldown != null) {
            configured = adventure.cooldown;
        }
        String cooldownId = binding.root().getId();
        float seconds = com.hypixel.hytale.server.core.modules.interaction
            .interaction.config.InteractionTypeUtils.getDefaultCooldown(
                binding.type()
            );
        if (configured != null) {
            if (configured.cooldownId != null) {
                cooldownId = configured.cooldownId;
            }
            seconds = configured.cooldown;
        }
        return capture(handler, cooldownId, seconds, true);
    }

    static State capture(
        CooldownHandler handler,
        String cooldownId,
        float seconds,
        boolean rulesLegal
    ) {
        if (handler == null || cooldownId == null) return State.unavailable();
        if (!rulesLegal) return new State(true, false);
        if (seconds <= 0.0f) return new State(true, true);

        CooldownHandler.Cooldown cooldown = handler.getCooldown(cooldownId);
        return new State(
            true,
            cooldown == null || !cooldown.hasCooldown(false)
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
