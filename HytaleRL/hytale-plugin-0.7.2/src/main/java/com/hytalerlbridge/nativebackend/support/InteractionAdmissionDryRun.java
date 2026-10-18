package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.protocol.InteractionChainData;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.util.UUIDUtil;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Side-effect-free replay of the engine's admission for a root that
 * {@link com.hytalerlbridge.nativebackend.NativeInteractionQueue} would queue.
 *
 * <p>{@code InteractionManager.canRun} passes no cancel list, so the engine's
 * private static {@code applyRules} recurses into predicted, unfinished forks
 * and can refuse a root that admission accepts once an authored interrupt
 * matches. FIDELITY-0.7.2.md section (p) records the crossbow case: the reload's
 * Repeat window admits Primary while canRun is false.
 *
 * <p>This replay calls that same private method with a fresh cancel list and
 * never calls {@code cancelChains}. The static method reads the chain map and
 * the data object and only appends to the list it is given; the instance method,
 * which cancels the chains it lists, is not called.
 */
public final class InteractionAdmissionDryRun {

    private static final System.Logger LOGGER = System.getLogger(
        InteractionAdmissionDryRun.class.getName()
    );
    private static final Method ENGINE_RULES = resolveEngineRules();
    private static final AtomicBoolean FALLBACK_LOGGED = new AtomicBoolean();

    private InteractionAdmissionDryRun() {}

    /** Whether the engine would admit queueing {@code root} right now. */
    public static boolean admits(
        InteractionManager manager,
        InteractionType type,
        RootInteraction root
    ) {
        if (manager == null || type == null || root == null) return false;
        Boolean admitted = replay(manager.getChains(), type, root);
        if (admitted != null) return admitted;
        if (FALLBACK_LOGGED.compareAndSet(false, true)) {
            LOGGER.log(
                System.Logger.Level.WARNING,
                "engine admission replay unavailable; interaction legality falls back to InteractionManager.canRun"
            );
        }
        return manager.canRun(type, root);
    }

    /** Null when the engine's private method is unavailable or throws. */
    static Boolean replay(
        Map<?, InteractionChain> chains,
        InteractionType type,
        RootInteraction root
    ) {
        if (ENGINE_RULES == null) return null;
        try {
            return (Boolean) ENGINE_RULES.invoke(
                null,
                new InteractionChainData(
                    -1,
                    UUIDUtil.EMPTY_UUID,
                    null,
                    null,
                    null,
                    Integer.MIN_VALUE,
                    null
                ),
                type,
                -1,
                root,
                chains,
                new ObjectArrayList<InteractionChain>()
            );
        } catch (
            IllegalAccessException
                | InvocationTargetException
                | RuntimeException unavailable
        ) {
            return null;
        }
    }

    private static Method resolveEngineRules() {
        try {
            Method method = InteractionManager.class.getDeclaredMethod(
                "applyRules",
                InteractionChainData.class,
                InteractionType.class,
                int.class,
                RootInteraction.class,
                Map.class,
                List.class
            );
            boolean usable =
                Modifier.isStatic(method.getModifiers()) &&
                method.getReturnType() == boolean.class &&
                method.trySetAccessible();
            return usable ? method : null;
        } catch (NoSuchMethodException | RuntimeException unavailable) {
            return null;
        }
    }
}
