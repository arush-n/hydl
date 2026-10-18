package com.hytalerlbridge.policy.combat.bridge;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.policy.perception.profile.PerceptionProfile;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

/** Reflection-only adapter to the independently deployed combat facade. */
public final class ReflectiveBridgeCombatFacade
    implements BridgeCombatFacade {

    public static final String EXPECTED_SCHEMA =
        "hytalerl_native_policy_combat_facade_v1";
    public static final int EXPECTED_VERSION = 1;

    private static final String FACADE_CLASS =
        "com.hytalerlbridge.nativebackend.policy.combat."
            + "NativePolicyCombatFacade";
    private static final int ABILITY_CAPACITY = 16;

    private final Methods methods;
    private final String itemId;
    private final int[] abilitySlots;
    private final String[] abilityIds;
    private final String[] abilityTypes;
    private final double[] requestedChargeSeconds;
    private final String guardId;
    private final String guardType;

    private ReflectiveBridgeCombatFacade(
        PerceptionProfile profile,
        Methods methods
    ) {
        if (profile == null || methods == null) {
            throw new IllegalArgumentException(
                "combat facade requires profile and reflection metadata");
        }
        this.methods = methods;
        this.itemId = profile.agentItemId();
        PerceptionProfile.AbilityBinding[] bindings =
            profile.agentAbilityBindings();
        abilitySlots = new int[bindings.length];
        abilityIds = new String[bindings.length];
        abilityTypes = new String[bindings.length];
        requestedChargeSeconds = new double[ABILITY_CAPACITY];
        for (int index = 0; index < bindings.length; index++) {
            PerceptionProfile.AbilityBinding binding = bindings[index];
            abilitySlots[index] = binding.slot();
            abilityIds[index] = binding.interactionId();
            abilityTypes[index] = binding.interactionType();
            requestedChargeSeconds[binding.slot()] =
                binding.requestedChargeSeconds();
        }
        PerceptionProfile.GuardBinding guard = profile.agentGuardBinding();
        guardId = guard == null ? "" : guard.interactionId();
        guardType = guard == null ? "" : guard.interactionType();
    }

    public static ReflectiveBridgeCombatFacade load(PerceptionProfile profile)
        throws ReflectiveOperationException {
        ClassLoader loader =
            ReflectiveBridgeCombatFacade.class.getClassLoader();
        return load(profile, Class.forName(FACADE_CLASS, true, loader));
    }

    /** Server-free ABI seam. */
    static ReflectiveBridgeCombatFacade load(
        PerceptionProfile profile,
        Class<?> facade
    ) throws ReflectiveOperationException {
        return new ReflectiveBridgeCombatFacade(
            profile, Methods.resolve(facade));
    }

    @Override
    public Binding bind() {
        try {
            Object raw = methods.bind().invoke(
                null,
                itemId,
                abilitySlots.clone(),
                abilityIds.clone(),
                abilityTypes.clone(),
                guardId,
                guardType
            );
            if (raw == null || !methods.bindingType().isInstance(raw)) {
                return Binding.rejected("bridge_combat_binding_invalid");
            }
            boolean accepted = bool(methods.binding("accepted"), raw);
            String reason = string(methods.binding("rejectReason"), raw);
            Object handle = methods.binding("handle").invoke(raw);
            return accepted && handle != null
                ? new Binding(true, reason, new Execution(handle))
                : Binding.rejected(reason.isEmpty()
                    ? "bridge_combat_binding_rejected"
                    : reason);
        } catch (ReflectiveOperationException | RuntimeException failure) {
            return Binding.rejected("bridge_combat_binding_failed");
        }
    }

    @Override
    public Receipt initialize(
        Execution execution,
        Ref<EntityStore> actor,
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        if (execution == null) {
            return Receipt.unavailable("native_combat_handle_invalid");
        }
        return receiptCall(
            methods.initialize(), execution.nativeHandle(), actor, npc, store);
    }

    @Override
    public Receipt prepare(
        Execution execution,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        float deltaTime,
        boolean externalRemoteClientActive
    ) {
        if (execution == null) {
            return Receipt.unavailable("native_combat_handle_invalid");
        }
        return receiptCall(
            methods.prepare(),
            execution.nativeHandle(),
            actor,
            store,
            deltaTime,
            externalRemoteClientActive
        );
    }

    @Override
    public Receipt apply(
        Execution execution,
        Ref<EntityStore> actor,
        NPCEntity npc,
        Store<EntityStore> store,
        float deltaTime,
        boolean firstControlTick,
        boolean attackRequested,
        int abilitySlot,
        boolean guardHeld,
        int dodgeDirection,
        double requestedCharge,
        boolean externalRootActive,
        boolean externalRemoteClientActive
    ) {
        if (execution == null) {
            return Receipt.unavailable("native_combat_handle_invalid");
        }
        return receiptCall(
            methods.apply(),
            execution.nativeHandle(),
            actor,
            npc,
            store,
            deltaTime,
            firstControlTick,
            attackRequested,
            abilitySlot,
            guardHeld,
            dodgeDirection,
            requestedCharge,
            externalRootActive,
            externalRemoteClientActive
        );
    }

    @Override
    public void close(
        Execution execution,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        boolean externalRemoteClientActive
    ) {
        if (execution == null) return;
        try {
            methods.close().invoke(
                null,
                execution.nativeHandle(),
                actor,
                store,
                externalRemoteClientActive
            );
        } catch (ReflectiveOperationException | RuntimeException ignored) {
            // Shutdown is best effort; never take the server down while
            // unloading an independently deployed policy plugin.
        }
    }

    @Override
    public double requestedChargeSeconds(int abilitySlot) {
        return abilitySlot < 0 || abilitySlot >= requestedChargeSeconds.length
            ? 0.0
            : requestedChargeSeconds[abilitySlot];
    }

    private Receipt receiptCall(Method method, Object... arguments) {
        try {
            return adaptReceipt(method.invoke(null, arguments));
        } catch (ReflectiveOperationException | RuntimeException failure) {
            return Receipt.unavailable("bridge_combat_call_failed");
        }
    }

    Receipt adaptReceipt(Object raw)
        throws IllegalAccessException, InvocationTargetException {
        if (raw == null || !methods.receiptType().isInstance(raw)) {
            return Receipt.unavailable("bridge_combat_receipt_invalid");
        }
        return new Receipt(
            bool(methods.receipt("available"), raw),
            string(methods.receipt("unavailableReason"), raw),
            bool(methods.receipt("attackRequested"), raw),
            bool(methods.receipt("attackAccepted"), raw),
            string(methods.receipt("attackInteractionId"), raw),
            string(methods.receipt("attackRejectReason"), raw),
            bool(methods.receipt("abilityRequested"), raw),
            bool(methods.receipt("abilityAccepted"), raw),
            string(methods.receipt("abilityInteractionId"), raw),
            string(methods.receipt("abilityRejectReason"), raw),
            bool(methods.receipt("dodgeRequested"), raw),
            bool(methods.receipt("dodgeAccepted"), raw),
            string(methods.receipt("dodgeInteractionId"), raw),
            string(methods.receipt("dodgeRejectReason"), raw),
            bool(methods.receipt("guardRequested"), raw),
            bool(methods.receipt("guardAccepted"), raw),
            string(methods.receipt("guardInteractionId"), raw),
            string(methods.receipt("guardRejectReason"), raw),
            bool(methods.receipt("abilityStarted"), raw),
            bool(methods.receipt("abilityFinished"), raw),
            bool(methods.receipt("abilityFailed"), raw),
            bool(methods.receipt("abilityRejectedBeforeStart"), raw),
            number(methods.receipt("abilitySlot"), raw).intValue(),
            bool(methods.receipt("dodgeStarted"), raw),
            bool(methods.receipt("dodgeFinished"), raw),
            bool(methods.receipt("dodgeFailed"), raw),
            bool(methods.receipt("dodgeRejectedBeforeStart"), raw),
            number(methods.receipt("dodgeDirection"), raw).intValue(),
            bool(methods.receipt("guardStarted"), raw),
            bool(methods.receipt("guardFinished"), raw),
            bool(methods.receipt("abilityActive"), raw),
            bool(methods.receipt("dodgeActive"), raw),
            bool(methods.receipt("guardChainActive"), raw),
            bool(methods.receipt("guardWieldingActive"), raw),
            number(methods.receipt("syntheticClientChains"), raw).intValue()
        );
    }

    private static boolean bool(Method method, Object target)
        throws IllegalAccessException, InvocationTargetException {
        return (Boolean) method.invoke(target);
    }

    private static Number number(Method method, Object target)
        throws IllegalAccessException, InvocationTargetException {
        return (Number) method.invoke(target);
    }

    private static String string(Method method, Object target)
        throws IllegalAccessException, InvocationTargetException {
        Object value = method.invoke(target);
        return value == null ? "" : (String) value;
    }

    private record Methods(
        Method bind,
        Method initialize,
        Method prepare,
        Method apply,
        Method close,
        Class<?> bindingType,
        Map<String, Method> bindingAccessors,
        Class<?> receiptType,
        Map<String, Method> receiptAccessors
    ) {
        Method binding(String name) {
            return required(bindingAccessors, name);
        }

        Method receipt(String name) {
            return required(receiptAccessors, name);
        }

        static Methods resolve(Class<?> facade)
            throws ReflectiveOperationException {
            if (facade == null) {
                throw new ClassNotFoundException("bridge combat facade");
            }
            String schema = (String) facade.getMethod("schema").invoke(null);
            int version = (Integer) facade.getMethod("version").invoke(null);
            if (!EXPECTED_SCHEMA.equals(schema) || version != EXPECTED_VERSION) {
                throw new NoSuchMethodException(
                    "unsupported bridge combat contract: "
                        + schema + " v" + version);
            }
            Method bind = facade.getMethod(
                "bind",
                String.class,
                int[].class,
                String[].class,
                String[].class,
                String.class,
                String.class
            );
            Method initialize = facade.getMethod(
                "initialize", Object.class, Ref.class, NPCEntity.class,
                Store.class);
            Method prepare = facade.getMethod(
                "prepare", Object.class, Ref.class, Store.class, float.class,
                boolean.class);
            Method apply = facade.getMethod(
                "apply", Object.class, Ref.class, NPCEntity.class, Store.class,
                float.class, boolean.class, boolean.class, int.class,
                boolean.class, int.class, double.class, boolean.class,
                boolean.class);
            Method close = facade.getMethod(
                "close", Object.class, Ref.class, Store.class, boolean.class);
            Class<?> bindingType = bind.getReturnType();
            Class<?> receiptType = initialize.getReturnType();
            if (prepare.getReturnType() != receiptType
                || apply.getReturnType() != receiptType) {
                throw new NoSuchMethodException(
                    "bridge combat receipt type differs across calls");
            }
            return new Methods(
                bind,
                initialize,
                prepare,
                apply,
                close,
                bindingType,
                accessors(bindingType,
                    "accepted", "rejectReason", "handle"),
                receiptType,
                accessors(receiptType,
                    "available", "unavailableReason",
                    "attackRequested", "attackAccepted",
                    "attackInteractionId", "attackRejectReason",
                    "abilityRequested", "abilityAccepted",
                    "abilityInteractionId", "abilityRejectReason",
                    "dodgeRequested", "dodgeAccepted",
                    "dodgeInteractionId", "dodgeRejectReason",
                    "guardRequested", "guardAccepted",
                    "guardInteractionId", "guardRejectReason",
                    "abilityStarted", "abilityFinished", "abilityFailed",
                    "abilityRejectedBeforeStart", "abilitySlot",
                    "dodgeStarted", "dodgeFinished", "dodgeFailed",
                    "dodgeRejectedBeforeStart", "dodgeDirection",
                    "guardStarted", "guardFinished", "abilityActive",
                    "dodgeActive", "guardChainActive",
                    "guardWieldingActive", "syntheticClientChains")
            );
        }

        private static Map<String, Method> accessors(
            Class<?> type,
            String... names
        ) throws NoSuchMethodException {
            Map<String, Method> result = new HashMap<>();
            for (String name : names) {
                result.put(name, type.getMethod(name));
            }
            return Map.copyOf(result);
        }

        private static Method required(Map<String, Method> values, String name) {
            Method method = values.get(name);
            if (method == null) {
                throw new IllegalStateException(
                    "missing reflected combat accessor " + name);
            }
            return method;
        }
    }
}
