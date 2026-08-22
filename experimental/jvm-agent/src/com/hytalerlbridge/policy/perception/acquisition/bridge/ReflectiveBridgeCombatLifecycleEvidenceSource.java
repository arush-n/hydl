package com.hytalerlbridge.policy.perception.acquisition.bridge;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.policy.perception.acquisition.CombatLifecycleEvidenceSource;
import com.hytalerlbridge.policy.perception.profile.PerceptionProfile;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * Standalone, reflection-only adapter to the deployed bridge lifecycle facade.
 *
 * <p>The policy mod compiles against {@code Server-0.5.7.jar} alone. At runtime
 * Hytale's {@code PluginClassLoader} searches the other loaded plugin class
 * loaders, so asking this mod's loader for the facade resolves the deployed
 * bridge implementation without making the bridge jar a compile dependency.
 * Schema and version are checked before the source can be constructed.</p>
 *
 * <p>Only immutable reflection metadata and checkpoint-owned ability bindings
 * are retained. Every gameplay value is read through the bridge facade on the
 * current call; this class has no per-entity clocks, caches, or previous-tick
 * state.</p>
 */
public final class ReflectiveBridgeCombatLifecycleEvidenceSource
    implements CombatLifecycleEvidenceSource {

    public static final String EXPECTED_SCHEMA =
        "hytalerl_native_policy_lifecycle_read_through_v2";
    public static final int EXPECTED_VERSION = 2;

    private static final String FACADE_CLASS =
        "com.hytalerlbridge.nativebackend.policy.lifecycle."
            + "NativePolicyLifecycleFacade";

    private final FacadeMethods facade;
    private final String agentItemId;
    private final int[] agentSlots;
    private final String[] agentInteractionIds;
    private final String[] agentInteractionTypes;
    private final String targetItemId;
    private final int[] targetSlots;
    private final String[] targetInteractionIds;
    private final String[] targetInteractionTypes;
    private final float agentForceDeadzone;

    private ReflectiveBridgeCombatLifecycleEvidenceSource(
        PerceptionProfile profile,
        FacadeMethods facade
    ) {
        if (profile == null || facade == null) {
            throw new IllegalArgumentException(
                "bridge lifecycle source requires profile and facade");
        }
        this.facade = facade;
        agentItemId = profile.agentItemId();
        AbilityArrays agent = AbilityArrays.from(
            profile.agentAbilityBindings());
        agentSlots = agent.slots();
        agentInteractionIds = agent.interactionIds();
        agentInteractionTypes = agent.interactionTypes();
        targetItemId = profile.targetItemId();
        AbilityArrays target = AbilityArrays.from(
            profile.targetAbilityBindings());
        targetSlots = target.slots();
        targetInteractionIds = target.interactionIds();
        targetInteractionTypes = target.interactionTypes();
        agentForceDeadzone = profile.agentForcePerAxisDeadzone();
    }

    /** Resolve and validate the lifecycle facade from the deployed bridge. */
    public static ReflectiveBridgeCombatLifecycleEvidenceSource load(
        PerceptionProfile profile
    ) throws ReflectiveOperationException {
        ClassLoader loader =
            ReflectiveBridgeCombatLifecycleEvidenceSource.class
                .getClassLoader();
        Class<?> facadeClass = Class.forName(FACADE_CLASS, true, loader);
        return load(profile, facadeClass);
    }

    /** Package-private seam used by the server-free reflection contract test. */
    static ReflectiveBridgeCombatLifecycleEvidenceSource load(
        PerceptionProfile profile,
        Class<?> facadeClass
    ) throws ReflectiveOperationException {
        return new ReflectiveBridgeCombatLifecycleEvidenceSource(
            profile,
            FacadeMethods.resolve(facadeClass)
        );
    }

    @Override
    public Evidence capture(
        Ref<EntityStore> agent,
        Ref<EntityStore> target,
        Store<EntityStore> store,
        int policySlot,
        float deltaTime
    ) {
        if (agent == null || !agent.isValid() || store == null
            || !Float.isFinite(deltaTime) || deltaTime <= 0.0f) {
            return null;
        }
        try {
            Object snapshot = facade.capture().invoke(
                null,
                agent,
                target,
                store,
                agentItemId,
                agentSlots.clone(),
                agentInteractionIds.clone(),
                agentInteractionTypes.clone(),
                targetItemId,
                targetSlots.clone(),
                targetInteractionIds.clone(),
                targetInteractionTypes.clone(),
                agentForceDeadzone
            );
            return adaptSnapshot(snapshot);
        } catch (IllegalAccessException | InvocationTargetException
            | ClassCastException | IllegalArgumentException failure) {
            // Missing, malformed, or failed native evidence is not equivalent
            // to an all-zero row. The public source skips this policy tick.
            return null;
        }
    }

    /** Convert one facade return value; package-private for the offline ABI test. */
    Evidence adaptSnapshot(Object snapshot)
        throws IllegalAccessException, InvocationTargetException {
        if (snapshot == null
            || !facade.snapshotType().isInstance(snapshot)
            || !((Boolean) facade.available().invoke(snapshot))) {
            return null;
        }
        return new Evidence(
            (Boolean) facade.agentAttackExecuting().invoke(snapshot),
            (Float) facade.agentAttackCooldownSeconds().invoke(snapshot),
            (Boolean) facade.agentKnockbackControlLock().invoke(snapshot),
            (float[]) facade.appliedVelocity().invoke(snapshot),
            (float[]) facade.dodgeInvulnerabilityRemaining().invoke(snapshot),
            (Integer) facade.targetAttackPhase().invoke(snapshot),
            (Float) facade.targetAttackProgress().invoke(snapshot),
            (Integer) facade.targetReportedAbilitySlot().invoke(snapshot),
            (String) facade.targetAttackEvidenceSource().invoke(snapshot),
            (Integer) facade.targetAttackSequenceCursor().invoke(snapshot),
            (Integer) facade.targetAttackSequenceCount().invoke(snapshot),
            (Float) facade.targetAttackPauseSeconds().invoke(snapshot),
            (float[]) facade.abilityCooldownSeconds().invoke(snapshot),
            (int[]) facade.activeAbilitySlot().invoke(snapshot),
            (float[]) facade.abilityElapsedSeconds().invoke(snapshot),
            (boolean[]) facade.abilityLegal().invoke(snapshot)
        );
    }

    private record AbilityArrays(
        int[] slots,
        String[] interactionIds,
        String[] interactionTypes
    ) {
        static AbilityArrays from(PerceptionProfile.AbilityBinding[] bindings) {
            int[] slots = new int[bindings.length];
            String[] ids = new String[bindings.length];
            String[] types = new String[bindings.length];
            for (int index = 0; index < bindings.length; index++) {
                PerceptionProfile.AbilityBinding binding = bindings[index];
                slots[index] = binding.slot();
                ids[index] = binding.interactionId();
                types[index] = binding.interactionType();
            }
            return new AbilityArrays(slots, ids, types);
        }
    }

    private record FacadeMethods(
        Class<?> snapshotType,
        Method capture,
        Method available,
        Method agentAttackExecuting,
        Method agentAttackCooldownSeconds,
        Method agentKnockbackControlLock,
        Method appliedVelocity,
        Method dodgeInvulnerabilityRemaining,
        Method targetAttackPhase,
        Method targetAttackProgress,
        Method targetReportedAbilitySlot,
        Method targetAttackEvidenceSource,
        Method targetAttackSequenceCursor,
        Method targetAttackSequenceCount,
        Method targetAttackPauseSeconds,
        Method abilityCooldownSeconds,
        Method activeAbilitySlot,
        Method abilityElapsedSeconds,
        Method abilityLegal
    ) {
        static FacadeMethods resolve(Class<?> facade)
            throws ReflectiveOperationException {
            if (facade == null) {
                throw new ClassNotFoundException("bridge lifecycle facade");
            }
            String schema = (String) facade.getMethod("schema").invoke(null);
            int version = (Integer) facade.getMethod("version").invoke(null);
            if (!EXPECTED_SCHEMA.equals(schema)
                || version != EXPECTED_VERSION) {
                throw new NoSuchMethodException(
                    "unsupported bridge lifecycle contract: "
                        + schema + " v" + version);
            }
            Method capture = facade.getMethod(
                "capture",
                Ref.class,
                Ref.class,
                Store.class,
                String.class,
                int[].class,
                String[].class,
                String[].class,
                String.class,
                int[].class,
                String[].class,
                String[].class,
                float.class
            );
            Class<?> snapshot = capture.getReturnType();
            return new FacadeMethods(
                snapshot,
                capture,
                snapshot.getMethod("available"),
                snapshot.getMethod("agentAttackExecuting"),
                snapshot.getMethod("agentAttackCooldownSeconds"),
                snapshot.getMethod("agentKnockbackControlLock"),
                snapshot.getMethod("appliedVelocity"),
                snapshot.getMethod("dodgeInvulnerabilityRemaining"),
                snapshot.getMethod("targetAttackPhase"),
                snapshot.getMethod("targetAttackProgress"),
                snapshot.getMethod("targetReportedAbilitySlot"),
                snapshot.getMethod("targetAttackEvidenceSource"),
                snapshot.getMethod("targetAttackSequenceCursor"),
                snapshot.getMethod("targetAttackSequenceCount"),
                snapshot.getMethod("targetAttackPauseSeconds"),
                snapshot.getMethod("abilityCooldownSeconds"),
                snapshot.getMethod("activeAbilitySlot"),
                snapshot.getMethod("abilityElapsedSeconds"),
                snapshot.getMethod("abilityLegal")
            );
        }
    }
}
