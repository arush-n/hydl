package com.hytalerlbridge.policy.world.bridge;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.PlayerRef;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.policy.world.model.WorldVerbBinding;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * Reflection-only adapter to the independently deployed bridge plugin.
 *
 * <p>The policy mod keeps its standalone Server-jar classpath. Both facade and
 * request identities are checked before any action can be advertised; a
 * missing or older bridge therefore closes craft rather than producing a
 * silent no-op.</p>
 */
public final class ReflectiveBridgeWorldVerbFacade
    implements BridgeWorldVerbFacade {

    public static final String EXPECTED_FACADE_SCHEMA =
        "hytalerl_native_policy_world_verb_facade_v3";
    public static final int EXPECTED_FACADE_VERSION = 3;
    public static final String EXPECTED_REQUEST_SCHEMA =
        "hytalerl_native_world_verb_transport_v4";
    public static final int EXPECTED_REQUEST_VERSION = 4;
    public static final String EXPECTED_REQUEST_CONTRACT_SHA256 =
        "3869A467733B057DD2C629243F3A6F0D5E7A116624E1413B48AFDEF664A8900C";
    public static final String POLICY_EVIDENCE_REQUIRED =
        "native_policy_world_action_evidence_required";

    private static final String FACADE_CLASS =
        "com.hytalerlbridge.nativebackend.NativePolicyWorldVerbFacade";
    private static final String REQUEST_CLASS =
        "com.hytalerlbridge.action.NativeWorldVerbRequest";

    private final Methods methods;

    private ReflectiveBridgeWorldVerbFacade(Methods methods) {
        this.methods = methods;
    }

    public static ReflectiveBridgeWorldVerbFacade load()
        throws ReflectiveOperationException {
        ClassLoader loader =
            ReflectiveBridgeWorldVerbFacade.class.getClassLoader();
        return load(
            Class.forName(FACADE_CLASS, true, loader),
            Class.forName(REQUEST_CLASS, true, loader)
        );
    }

    /** Server-free ABI seam. */
    static ReflectiveBridgeWorldVerbFacade load(
        Class<?> facade,
        Class<?> request
    ) throws ReflectiveOperationException {
        return new ReflectiveBridgeWorldVerbFacade(
            Methods.resolve(facade, request));
    }

    /**
     * The current transport requires candidate-generation and selected-
     * semantic hashes from one bridge-owned atomic capture. The standalone
     * public-server capture
     * does not produce those privileged identities, so it must not advertise
     * executable World verbs merely because the reflection ABI resolves.
     */
    @Override
    public boolean policyEvidenceSupported() {
        return false;
    }

    @Override
    public ContextResult installContext(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        PlayerRef packetAnchor
    ) {
        return contextCall(methods.installContext(), actor, store, packetAnchor);
    }

    @Override
    public ContextResult uninstallContext(
        Ref<EntityStore> actor,
        Store<EntityStore> store
    ) {
        return contextCall(methods.uninstallContext(), actor, store);
    }

    @Override
    public Admission startFieldcraft(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        long requestId,
        String recipeId,
        String worldEpoch
    ) {
        if (!policyEvidenceSupported()) {
            return rejectedAdmission(POLICY_EVIDENCE_REQUIRED);
        }
        if (requestId < 0L || recipeId == null || recipeId.isBlank()
            || worldEpoch == null || worldEpoch.isBlank()) {
            return rejectedAdmission("invalid_fieldcraft_binding");
        }
        try {
            Object request = methods.requestConstructor().newInstance(
                requestId,
                "craft_recipe",
                -1,
                "recipe",
                "",
                recipeId,
                "",
                "",
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                "player_inventory",
                -1,
                -1,
                "player_inventory_or_world_drop",
                "",
                worldEpoch,
                1,
                "",
                "fieldcraft",
                0,
                0,
                0,
                "",
                "Fieldcraft",
                0,
                0,
                "",
                ""
            );
            Object raw = methods.start().invoke(
                null, actor, store, request, worldEpoch);
            return adaptAdmission(raw);
        } catch (ReflectiveOperationException | RuntimeException failure) {
            return rejectedAdmission("bridge_world_verb_start_failed");
        }
    }

    @Override
    public Admission startBoundWorldVerb(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        long requestId,
        WorldVerbBinding binding,
        String worldEpoch
    ) {
        if (!policyEvidenceSupported()) {
            return rejectedAdmission(POLICY_EVIDENCE_REQUIRED);
        }
        if (requestId < 0L || binding == null || !binding.available()
            || worldEpoch == null || worldEpoch.isBlank()) {
            return rejectedAdmission("invalid_block_action_binding");
        }
        try {
            Object request = methods.requestConstructor().newInstance(
                requestId,
                binding.verb(),
                binding.interactionType(),
                "block",
                binding.interactionId(),
                "",
                binding.itemId(),
                binding.blockId(),
                binding.targetX(),
                binding.targetY(),
                binding.targetZ(),
                binding.blockFace(),
                binding.rotationYaw(),
                binding.rotationPitch(),
                binding.rotationRoll(),
                binding.sourceContainer(),
                binding.sourceSlot(),
                binding.expectedSourceQuantity(),
                "",
                binding.expectedBlockId(),
                worldEpoch,
                1,
                binding.placementVariant(),
                "",
                0,
                0,
                0,
                "",
                "",
                -1,
                0,
                "",
                ""
            );
            Object raw = methods.start().invoke(
                null, actor, store, request, worldEpoch);
            return adaptAdmission(raw);
        } catch (ReflectiveOperationException | RuntimeException failure) {
            return rejectedAdmission("bridge_world_verb_start_failed");
        }
    }

    @Override
    public Lifecycle prepare(
        Execution execution,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        float deltaTime
    ) {
        if (execution == null) {
            return Lifecycle.rejected("native_world_verb_handle_invalid");
        }
        try {
            return adaptLifecycle(methods.prepare().invoke(
                null, execution.nativeHandle(), actor, store, deltaTime));
        } catch (ReflectiveOperationException | RuntimeException failure) {
            return Lifecycle.rejected("bridge_world_verb_prepare_failed");
        }
    }

    @Override
    public Lifecycle poll(
        Execution execution,
        Ref<EntityStore> actor,
        Store<EntityStore> store
    ) {
        if (execution == null) {
            return Lifecycle.rejected("native_world_verb_handle_invalid");
        }
        try {
            return adaptLifecycle(methods.poll().invoke(
                null, execution.nativeHandle(), actor, store));
        } catch (ReflectiveOperationException | RuntimeException failure) {
            return Lifecycle.rejected("bridge_world_verb_poll_failed");
        }
    }

    private ContextResult contextCall(Method method, Object... arguments) {
        try {
            Object raw = method.invoke(null, arguments);
            return new ContextResult(
                (Boolean) methods.contextAccepted().invoke(raw),
                (String) methods.contextRejectReason().invoke(raw)
            );
        } catch (ReflectiveOperationException | RuntimeException failure) {
            return new ContextResult(false, "bridge_world_verb_context_failed");
        }
    }

    private Admission adaptAdmission(Object raw)
        throws IllegalAccessException, InvocationTargetException {
        if (raw == null || !methods.admissionType().isInstance(raw)) {
            return rejectedAdmission("bridge_world_verb_admission_invalid");
        }
        boolean accepted = (Boolean) methods.admissionAccepted().invoke(raw);
        String reason = (String) methods.admissionRejectReason().invoke(raw);
        Object handle = methods.admissionHandle().invoke(raw);
        Lifecycle lifecycle = adaptLifecycle(
            methods.admissionLifecycle().invoke(raw));
        if (!accepted || handle == null) {
            return new Admission(false, reason, null, lifecycle);
        }
        return new Admission(
            true, reason, new Execution(handle), lifecycle);
    }

    private Lifecycle adaptLifecycle(Object raw)
        throws IllegalAccessException, InvocationTargetException {
        if (raw == null || !methods.lifecycleType().isInstance(raw)) {
            return Lifecycle.rejected("bridge_world_verb_lifecycle_invalid");
        }
        return new Lifecycle(
            bool(methods.lifecycleAccepted(), raw),
            bool(methods.lifecyclePending(), raw),
            bool(methods.lifecycleStarted(), raw),
            bool(methods.lifecycleFinished(), raw),
            bool(methods.lifecycleFailed(), raw),
            number(methods.lifecycleRequestedTick(), raw).longValue(),
            number(methods.lifecycleStartedTick(), raw).longValue(),
            number(methods.lifecycleFinishedTick(), raw).longValue(),
            bool(methods.lifecycleGeometryChanged(), raw),
            bool(methods.lifecycleInventoryChanged(), raw),
            (String) methods.lifecycleBlockIdBefore().invoke(raw),
            (String) methods.lifecycleBlockIdAfter().invoke(raw),
            number(methods.lifecycleSourceQuantityBefore(), raw).intValue(),
            number(methods.lifecycleSourceQuantityAfter(), raw).intValue(),
            (String) methods.lifecycleRejectReason().invoke(raw)
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

    private static Admission rejectedAdmission(String reason) {
        return new Admission(
            false, reason, null, Lifecycle.rejected(reason));
    }

    private record Methods(
        Constructor<?> requestConstructor,
        Method installContext,
        Method uninstallContext,
        Method start,
        Method prepare,
        Method poll,
        Class<?> contextType,
        Method contextAccepted,
        Method contextRejectReason,
        Class<?> admissionType,
        Method admissionAccepted,
        Method admissionRejectReason,
        Method admissionHandle,
        Method admissionLifecycle,
        Class<?> lifecycleType,
        Method lifecycleAccepted,
        Method lifecyclePending,
        Method lifecycleStarted,
        Method lifecycleFinished,
        Method lifecycleFailed,
        Method lifecycleRequestedTick,
        Method lifecycleStartedTick,
        Method lifecycleFinishedTick,
        Method lifecycleGeometryChanged,
        Method lifecycleInventoryChanged,
        Method lifecycleBlockIdBefore,
        Method lifecycleBlockIdAfter,
        Method lifecycleSourceQuantityBefore,
        Method lifecycleSourceQuantityAfter,
        Method lifecycleRejectReason
    ) {
        static Methods resolve(Class<?> facade, Class<?> request)
            throws ReflectiveOperationException {
            if (facade == null || request == null) {
                throw new ClassNotFoundException("bridge World-verb ABI");
            }
            requireIdentity(
                facade,
                EXPECTED_FACADE_SCHEMA,
                EXPECTED_FACADE_VERSION,
                null
            );
            requireIdentity(
                request,
                EXPECTED_REQUEST_SCHEMA,
                EXPECTED_REQUEST_VERSION,
                EXPECTED_REQUEST_CONTRACT_SHA256
            );
            Constructor<?> constructor = request.getConstructor(
                long.class, String.class, int.class, String.class,
                String.class, String.class, String.class, String.class,
                int.class, int.class, int.class, int.class, int.class,
                int.class, int.class, String.class, int.class, int.class,
                String.class, String.class, String.class, int.class,
                String.class, String.class, int.class, int.class, int.class,
                String.class, String.class, int.class, int.class
                , String.class, String.class
            );
            Method install = facade.getMethod(
                "installContext", Ref.class, Store.class, PlayerRef.class);
            Method uninstall = facade.getMethod(
                "uninstallContext", Ref.class, Store.class);
            Method start = facade.getMethod(
                "start", Ref.class, Store.class, request, String.class);
            Method prepare = facade.getMethod(
                "prepare", Object.class, Ref.class, Store.class, float.class);
            Method poll = facade.getMethod(
                "poll", Object.class, Ref.class, Store.class);
            Class<?> context = install.getReturnType();
            Class<?> admission = start.getReturnType();
            Method admissionLifecycle = admission.getMethod("lifecycle");
            Class<?> lifecycle = admissionLifecycle.getReturnType();
            if (uninstall.getReturnType() != context
                || prepare.getReturnType() != lifecycle
                || poll.getReturnType() != lifecycle) {
                throw new NoSuchMethodException(
                    "bridge World-verb return types drifted");
            }
            return new Methods(
                constructor, install, uninstall, start, prepare, poll,
                context,
                context.getMethod("accepted"),
                context.getMethod("rejectReason"),
                admission,
                admission.getMethod("accepted"),
                admission.getMethod("rejectReason"),
                admission.getMethod("handle"),
                admissionLifecycle,
                lifecycle,
                lifecycle.getMethod("accepted"),
                lifecycle.getMethod("pending"),
                lifecycle.getMethod("started"),
                lifecycle.getMethod("finished"),
                lifecycle.getMethod("failed"),
                lifecycle.getMethod("requestedTick"),
                lifecycle.getMethod("startedTick"),
                lifecycle.getMethod("finishedTick"),
                lifecycle.getMethod("geometryChanged"),
                lifecycle.getMethod("inventoryChanged"),
                lifecycle.getMethod("blockIdBefore"),
                lifecycle.getMethod("blockIdAfter"),
                lifecycle.getMethod("sourceQuantityBefore"),
                lifecycle.getMethod("sourceQuantityAfter"),
                lifecycle.getMethod("rejectReason")
            );
        }

        private static void requireIdentity(
            Class<?> type,
            String expectedSchema,
            int expectedVersion,
            String expectedContract
        ) throws ReflectiveOperationException {
            String schema = (String) type.getField("SCHEMA").get(null);
            int version = type.getField("VERSION").getInt(null);
            String contract = expectedContract == null
                ? null
                : (String) type.getField("CONTRACT_SHA256").get(null);
            if (!expectedSchema.equals(schema) || version != expectedVersion
                || (expectedContract != null
                    && !expectedContract.equals(contract))) {
                throw new NoSuchMethodException(
                    "unsupported bridge World-verb contract: "
                        + schema + " v" + version);
            }
        }
    }
}
