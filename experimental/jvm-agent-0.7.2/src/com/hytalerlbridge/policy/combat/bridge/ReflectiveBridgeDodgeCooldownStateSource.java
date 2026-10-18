package com.hytalerlbridge.policy.combat.bridge;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

/**
 * Reflection-only adapter over the bridge's existing read-only Dodge view.
 *
 * <p>No bridge class is linked into the standalone policy artifact. The
 * adapter retains method handles only; all cooldown state remains owned by the
 * engine's {@code CooldownHandler}.</p>
 */
public final class ReflectiveBridgeDodgeCooldownStateSource
    implements BridgeDodgeCooldownStateSource {

    private static final String SUPPORT_CLASS =
        "com.hytalerlbridge.nativebackend.support.InteractionSupport";
    private static final String VIEW_CLASS =
        "com.hytalerlbridge.combat.dodge.NativeDodgeCooldownView";

    private final Method interactionManager;
    private final Method layoutAvailable;
    private final Method capture;
    private final Method stateAvailable;
    private final Method stateOnCooldown;

    private ReflectiveBridgeDodgeCooldownStateSource(
        Method interactionManager,
        Method layoutAvailable,
        Method capture,
        Method stateAvailable,
        Method stateOnCooldown
    ) {
        this.interactionManager = interactionManager;
        this.layoutAvailable = layoutAvailable;
        this.capture = capture;
        this.stateAvailable = stateAvailable;
        this.stateOnCooldown = stateOnCooldown;
    }

    public static ReflectiveBridgeDodgeCooldownStateSource load()
        throws ReflectiveOperationException {
        ClassLoader loader =
            ReflectiveBridgeDodgeCooldownStateSource.class.getClassLoader();
        return load(
            Class.forName(SUPPORT_CLASS, true, loader),
            Class.forName(VIEW_CLASS, true, loader)
        );
    }

    /** Server-free construction seam for the structural adapter gate. */
    static ReflectiveBridgeDodgeCooldownStateSource load(
        Class<?> support,
        Class<?> view
    ) throws ReflectiveOperationException {
        Method manager = publicStatic(support, "interactionManager", 2);
        Class<?>[] managerParameters = manager.getParameterTypes();
        if (managerParameters[0] != Ref.class
                || managerParameters[1] != Store.class) {
            throw new NoSuchMethodException(
                "InteractionSupport.interactionManager(Ref, Store)");
        }
        Method layout = publicStatic(view, "layoutAvailable", 0);
        if (layout.getReturnType() != boolean.class) {
            throw new NoSuchMethodException(
                "NativeDodgeCooldownView.layoutAvailable(): boolean");
        }
        Method capture = view.getMethod("capture", manager.getReturnType());
        if (!Modifier.isStatic(capture.getModifiers())) {
            throw new NoSuchMethodException(
                "NativeDodgeCooldownView.capture must be static");
        }
        Class<?> state = capture.getReturnType();
        Method available = state.getMethod("available");
        Method onCooldown = state.getMethod("onCooldown");
        if (available.getReturnType() != boolean.class
                || onCooldown.getReturnType() != boolean.class) {
            throw new NoSuchMethodException(
                "native Dodge state accessors must return boolean");
        }
        return new ReflectiveBridgeDodgeCooldownStateSource(
            manager, layout, capture, available, onCooldown);
    }

    @Override
    public State capture(Ref<EntityStore> actor, Store<EntityStore> store) {
        if (actor == null || store == null) {
            return State.unavailable("dodge_cooldown_actor_or_store_unavailable");
        }
        try {
            if (!layoutAvailable()) {
                return State.unavailable(
                    "native_dodge_cooldown_layout_unavailable");
            }
            Object manager = interactionManager.invoke(null, actor, store);
            Object state = capture.invoke(null, manager);
            if (state == null
                    || !Boolean.TRUE.equals(stateAvailable.invoke(state))) {
                return State.unavailable(
                    "native_dodge_cooldown_state_unavailable");
            }
            return State.available(
                Boolean.TRUE.equals(stateOnCooldown.invoke(state)));
        } catch (IllegalAccessException | InvocationTargetException
                | RuntimeException unavailable) {
            return State.unavailable(
                "native_dodge_cooldown_capture_failed_"
                    + unavailable.getClass().getSimpleName());
        }
    }

    /** Whether the current Server layout exposes its existing cooldown map. */
    public boolean layoutAvailable() {
        try {
            return Boolean.TRUE.equals(layoutAvailable.invoke(null));
        } catch (IllegalAccessException | InvocationTargetException
                | RuntimeException unavailable) {
            return false;
        }
    }

    private static Method publicStatic(
        Class<?> owner,
        String name,
        int parameterCount
    ) throws NoSuchMethodException {
        for (Method method : owner.getMethods()) {
            if (method.getName().equals(name)
                    && method.getParameterCount() == parameterCount
                    && Modifier.isStatic(method.getModifiers())) {
                return method;
            }
        }
        throw new NoSuchMethodException(
            owner.getName() + "." + name + "/" + parameterCount);
    }
}
