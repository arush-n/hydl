package com.hytalerlbridge.nativebackend.policy.lifecycle;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.InteractionState;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.entity.knockback.KnockbackComponent;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.movement.controllers.MotionController;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.role.support.CombatSupport;
import com.hytalerlbridge.combat.CombatPhase;
import com.hytalerlbridge.combat.HytaleCombatAssets;
import com.hytalerlbridge.combat.MeleeAttackProfile;
import com.hytalerlbridge.combat.dodge.NativeDodgeProgram;
import com.hytalerlbridge.nativebackend.model.CapturedStatuses;
import com.hytalerlbridge.nativebackend.model.NativeInteractionBinding;
import com.hytalerlbridge.observation.NativeActorEvidenceFrame;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.captureStatuses;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.activeItemId;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.interactionManager;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.resolveNativeAbilityBinding;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.NATIVE_COMBAT_PROBE_ROOT_PREFIX;

/**
 * Stateless native lifecycle read-through for arbitrary role-claimed NPCs.
 *
 * <p>This facade deliberately does not consult {@code NativeAgentMarker}, a
 * native environment session, or any previous capture. Every dynamic value is
 * read from the actor's engine-owned components and interaction manager on the
 * current call. Checkpoint-owned ability bindings and the JAX force deadzone
 * are explicit arguments.</p>
 */
public final class NativePolicyLifecycleFacade {
    public static final String SCHEMA =
        "hytalerl_native_policy_lifecycle_read_through_v2";
    public static final int VERSION = 2;
    public static final int ABILITY_CAPACITY =
        NativePolicyLifecycleSnapshot.ABILITY_CAPACITY;

    private NativePolicyLifecycleFacade() {}

    public static String schema() {
        return SCHEMA;
    }

    public static int version() {
        return VERSION;
    }

    /**
     * Capture one same-tick agent/target lifecycle row.
     *
     * <p>Binding arrays are sparse and parallel: each row names its authored
     * learner slot, exact native interaction, and interaction type. The item ID
     * lets the bridge preserve item-root cooldown/rule ownership when the root
     * is unambiguous.</p>
     */
    public static NativePolicyLifecycleSnapshot capture(
        Ref<EntityStore> agent,
        Ref<EntityStore> target,
        Store<EntityStore> store,
        String agentItemId,
        int[] agentSlots,
        String[] agentInteractionIds,
        String[] agentInteractionTypes,
        String targetItemId,
        int[] targetSlots,
        String[] targetInteractionIds,
        String[] targetInteractionTypes,
        float agentForceDeadzone
    ) {
        if (
            agent == null
                || !agent.isValid()
                || store == null
                || !Float.isFinite(agentForceDeadzone)
                || agentForceDeadzone < 0.0f
        ) {
            return NativePolicyLifecycleSnapshot.unavailable("invalid_request");
        }
        try {
            List<ResolvedAbility> agentBindings = resolve(
                agentItemId,
                agentSlots,
                agentInteractionIds,
                agentInteractionTypes
            );
            List<ResolvedAbility> targetBindings = resolve(
                targetItemId,
                targetSlots,
                targetInteractionIds,
                targetInteractionTypes
            );
            ActorLifecycle agentLife = actor(
                agent,
                store,
                agentBindings,
                agentItemId
            );
            if (!agentLife.available()) {
                return NativePolicyLifecycleSnapshot.unavailable(
                    "agent_" + agentLife.reason()
                );
            }

            boolean targetPresent = target != null && target.isValid();
            ActorLifecycle targetLife = targetPresent
                ? actor(
                    target,
                    store,
                    targetBindings,
                    targetItemId
                )
                : ActorLifecycle.absent();
            if (!targetLife.available()) {
                return NativePolicyLifecycleSnapshot.unavailable(
                    "target_" + targetLife.reason()
                );
            }

            TargetAttack targetAttack = targetPresent
                ? targetAttack(
                    targetLife.manager(),
                    targetLife.combat(),
                    targetLife.role()
                )
                : TargetAttack.idle();
            if (!targetAttack.available()) {
                return NativePolicyLifecycleSnapshot.unavailable(
                    targetAttack.reason()
                );
            }

            float[] velocity = new float[6];
            System.arraycopy(agentLife.projectedVelocity(), 0, velocity, 0, 3);
            if (targetPresent) {
                System.arraycopy(
                    targetLife.projectedVelocity(),
                    0,
                    velocity,
                    3,
                    3
                );
            }
            float[] dodge = {
                agentLife.dodgeRemaining(),
                targetPresent ? targetLife.dodgeRemaining() : 0.0f,
            };
            float[] cooldown = new float[2 * ABILITY_CAPACITY];
            boolean[] legal = new boolean[2 * ABILITY_CAPACITY];
            System.arraycopy(
                agentLife.cooldownSeconds(),
                0,
                cooldown,
                0,
                ABILITY_CAPACITY
            );
            System.arraycopy(
                agentLife.legal(),
                0,
                legal,
                0,
                ABILITY_CAPACITY
            );
            if (targetPresent) {
                System.arraycopy(
                    targetLife.cooldownSeconds(),
                    0,
                    cooldown,
                    ABILITY_CAPACITY,
                    ABILITY_CAPACITY
                );
                System.arraycopy(
                    targetLife.legal(),
                    0,
                    legal,
                    ABILITY_CAPACITY,
                    ABILITY_CAPACITY
                );
            }
            int[] active = {
                agentLife.activeSlot(),
                targetPresent ? targetLife.activeSlot() : -1,
            };
            float[] elapsed = {
                agentLife.activeElapsedSeconds(),
                targetPresent ? targetLife.activeElapsedSeconds() : 0.0f,
            };
            float forceSpeed = norm(agentLife.projectedVelocity());
            boolean attackExecuting = agentLife.combat() != null
                && agentLife.combat().isExecutingAttack();
            NativeCombatSupportView.State attackPause =
                NativeCombatSupportView.capture(agentLife.combat());
            if (!attackPause.available()) {
                return NativePolicyLifecycleSnapshot.unavailable(
                    "agent_" + attackPause.reason()
                );
            }
            return new NativePolicyLifecycleSnapshot(
                true,
                "",
                attackExecuting,
                attackPause.attackPauseSeconds(),
                forceSpeed > agentForceDeadzone,
                velocity,
                dodge,
                targetAttack.phase().code(),
                targetAttack.progress(),
                targetAttack.reportedSlot(),
                targetAttack.evidenceSource(),
                targetAttack.sequenceCursor(),
                targetAttack.sequenceCount(),
                targetAttack.pauseSeconds(),
                cooldown,
                active,
                elapsed,
                legal
            );
        } catch (IllegalArgumentException invalidBinding) {
            return NativePolicyLifecycleSnapshot.unavailable("binding_invalid");
        } catch (RuntimeException unavailable) {
            return NativePolicyLifecycleSnapshot.unavailable("engine_read_failed");
        }
    }

    private static ActorLifecycle actor(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        List<ResolvedAbility> bindings,
        String expectedItemId
    ) {
        NPCEntity npc = store.getComponent(reference, NPCEntity.getComponentType());
        Role role = npc == null ? null : npc.getRole();
        if (npc == null || role == null) {
            return ActorLifecycle.unavailable("role_unavailable");
        }
        if (
            !bindings.isEmpty()
                && !expectedItemId.equals(activeItemId(npc.getInventory()))
        ) {
            return ActorLifecycle.unavailable("item_mismatch");
        }
        InteractionManager manager = interactionManager(reference, store);
        // NPCInteractionSystems.AddSimulationManagerSystem installs this on
        // every NPC. Its absence is unreadable evidence, not proof that an
        // actor is idle; accepting it would create a false-negative attack row.
        if (manager == null) {
            return ActorLifecycle.unavailable("interaction_manager_unavailable");
        }

        KnockbackComponent knockback = store.getComponent(
            reference,
            KnockbackComponent.getComponentType()
        );
        MotionController controller = role.getActiveMotionController();
        NativeActorEvidenceFrame.MotionForce force =
            com.hytalerlbridge.nativebackend.support.NativeMotionForceCapture
                .capture(controller, knockback);
        if (!force.projectedAvailable()) {
            return ActorLifecycle.unavailable("motion_force_unavailable");
        }
        float[] projected = floats(force.projectedVelocity());
        if (projected == null) {
            return ActorLifecycle.unavailable("motion_force_invalid");
        }

        CapturedStatuses statuses = captureStatuses(store, reference);
        if (statuses.overflow() || statuses.invalid()) {
            return ActorLifecycle.unavailable("status_evidence_invalid");
        }
        float dodgeRemaining = 0.0f;
        for (NativeActorEvidenceFrame.Status status : statuses.statuses()) {
            if (NativeDodgeProgram.INVULNERABILITY_EFFECT_ID.equals(status.effectId())) {
                dodgeRemaining = Math.max(
                    dodgeRemaining,
                    (float) status.remainingDurationSeconds()
                );
            }
        }

        float[] cooldowns = new float[ABILITY_CAPACITY];
        boolean[] legal = new boolean[ABILITY_CAPACITY];
        int activeSlot = -1;
        float activeElapsed = 0.0f;
        if (manager != null) {
            ActiveAbility active = activeAbility(manager, bindings);
            if (!active.available()) {
                return ActorLifecycle.unavailable(active.reason());
            }
            activeSlot = active.slot();
            activeElapsed = active.elapsedSeconds();
            for (ResolvedAbility binding : bindings) {
                NativePolicyCooldownView.State cooldown =
                    NativePolicyCooldownView.capture(
                        manager,
                        binding.binding().root(),
                        binding.binding().type()
                    );
                if (!cooldown.available()) {
                    return ActorLifecycle.unavailable("cooldown_layout_unavailable");
                }
                cooldowns[binding.slot()] = cooldown.remainingSeconds();
                legal[binding.slot()] = !cooldown.onCooldown()
                    && manager.canRun(
                        binding.binding().type(),
                        binding.binding().root()
                    );
            }
        }
        return new ActorLifecycle(
            true,
            "",
            manager,
            role,
            role.getCombatSupport(),
            projected,
            dodgeRemaining,
            cooldowns,
            legal,
            activeSlot,
            activeElapsed
        );
    }

    private static ActiveAbility activeAbility(
        InteractionManager manager,
        List<ResolvedAbility> bindings
    ) {
        int slot = -1;
        float elapsed = 0.0f;
        for (InteractionChain chain : manager.getChains().values()) {
            if (
                chain == null
                    || chain.getServerState() != InteractionState.NotFinished
                    || chain.getInitialRootInteraction() == null
            ) {
                continue;
            }
            String rootId = chain.getInitialRootInteraction().getId();
            for (ResolvedAbility candidate : bindings) {
                if (!matches(rootId, candidate.binding())) continue;
                if (slot >= 0) {
                    return ActiveAbility.unavailable("active_ability_ambiguous");
                }
                float value = chain.getTimeInSeconds();
                if (!Float.isFinite(value) || value < 0.0f) {
                    return ActiveAbility.unavailable("active_ability_clock_invalid");
                }
                slot = candidate.slot();
                elapsed = value;
            }
        }
        return new ActiveAbility(true, "", slot, elapsed);
    }

    private static TargetAttack targetAttack(
        InteractionManager manager,
        CombatSupport combat,
        Role role
    ) {
        NativeCombatSupportView.State pause =
            NativeCombatSupportView.capture(combat);
        if (!pause.available()) {
            return TargetAttack.unavailable("target_" + pause.reason());
        }
        if (manager == null) {
            return TargetAttack.idle(pause.attackPauseSeconds());
        }
        int attackIndex = -1;
        float elapsed = 0.0f;
        for (InteractionChain chain : manager.getChains().values()) {
            if (
                chain == null
                    || chain.getServerState() != InteractionState.NotFinished
                    || chain.getInitialRootInteraction() == null
            ) {
                continue;
            }
            String id = chain.getInitialRootInteraction().getId();
            int index = HytaleCombatAssets.brawlerAttackIndex(id);
            if (index < 0) continue;
            if (attackIndex >= 0) {
                return TargetAttack.unavailable("target_attack_ambiguous");
            }
            float value = chain.getTimeInSeconds();
            if (!Float.isFinite(value) || value < 0.0f) {
                return TargetAttack.unavailable("target_attack_clock_invalid");
            }
            attackIndex = index;
            elapsed = value;
        }
        if (attackIndex < 0) {
            if (combat == null || !combat.isExecutingAttack()) {
                return TargetAttack.idle(pause.attackPauseSeconds());
            }
            NativeNpcAttackSequenceView.State sequence =
                NativeNpcAttackSequenceView.capture(role);
            if (!sequence.available()) {
                return TargetAttack.unavailable(sequence.reason());
            }
            return new TargetAttack(
                true,
                "",
                CombatPhase.COOLDOWN,
                1.0f,
                sequence.attackIndex(),
                NativePolicyLifecycleSnapshot
                    .TARGET_ATTACK_SOURCE_ACTION_LIST_CURSOR,
                sequence.cursor(),
                sequence.actionCount(),
                pause.attackPauseSeconds()
            );
        }
        MeleeAttackProfile profile =
            HytaleCombatAssets.TRORK_BRAWLER_ATTACKS.get(attackIndex);
        int ticks = Math.max(
            1,
            (int) Math.ceil(
                elapsed * HytaleCombatAssets.TICKS_PER_SECOND
            )
        );
        return new TargetAttack(
            true,
            "",
            profile.phaseAt(ticks),
            (float) profile.phaseProgressAt(ticks),
            attackIndex,
            NativePolicyLifecycleSnapshot.TARGET_ATTACK_SOURCE_ACTIVE_CHAIN,
            -1,
            0,
            pause.attackPauseSeconds()
        );
    }

    private static List<ResolvedAbility> resolve(
        String itemId,
        int[] slots,
        String[] interactionIds,
        String[] interactionTypes
    ) {
        if (
            slots == null
                || interactionIds == null
                || interactionTypes == null
                || slots.length != interactionIds.length
                || slots.length != interactionTypes.length
                || slots.length > ABILITY_CAPACITY
        ) {
            throw new IllegalArgumentException("ability binding width drift");
        }
        if (slots.length > 0 && (itemId == null || itemId.isBlank())) {
            throw new IllegalArgumentException("ability item ID is required");
        }
        boolean[] seen = new boolean[ABILITY_CAPACITY];
        List<ResolvedAbility> result = new ArrayList<>(slots.length);
        for (int index = 0; index < slots.length; index++) {
            int slot = slots[index];
            String id = interactionIds[index];
            String type = interactionTypes[index];
            if (
                slot < 0
                    || slot >= ABILITY_CAPACITY
                    || seen[slot]
                    || id == null
                    || id.isBlank()
                    || type == null
                    || type.isBlank()
            ) {
                throw new IllegalArgumentException("invalid ability binding");
            }
            seen[slot] = true;
            result.add(new ResolvedAbility(
                slot,
                resolveNativeAbilityBinding(itemId, id, type)
            ));
        }
        return List.copyOf(result);
    }

    static boolean matches(
        String activeRootId,
        NativeInteractionBinding binding
    ) {
        if (activeRootId == null || binding == null || binding.root() == null) {
            return false;
        }
        return matches(activeRootId, binding.root().getId(), binding.id());
    }

    /** Package-visible identity-only form for a server-free collision gate. */
    static boolean matches(
        String activeRootId,
        String resolvedRootId,
        String interactionId
    ) {
        if (
            activeRootId == null
                || resolvedRootId == null
                || interactionId == null
        ) {
            return false;
        }
        return activeRootId.equals(resolvedRootId)
            || activeRootId.equals(interactionId)
            || activeRootId.equals(
                NATIVE_COMBAT_PROBE_ROOT_PREFIX + interactionId
            );
    }

    private static float[] floats(double[] values) {
        if (values == null || values.length != 3) return null;
        float[] result = new float[3];
        for (int index = 0; index < result.length; index++) {
            if (!Double.isFinite(values[index])) return null;
            result[index] = (float) values[index];
            if (!Float.isFinite(result[index])) return null;
        }
        return result;
    }

    private static float norm(float[] value) {
        return (float) Math.sqrt(
            value[0] * value[0]
                + value[1] * value[1]
                + value[2] * value[2]
        );
    }

    private record ResolvedAbility(
        int slot,
        NativeInteractionBinding binding
    ) {}

    private record ActiveAbility(
        boolean available,
        String reason,
        int slot,
        float elapsedSeconds
    ) {
        private static ActiveAbility unavailable(String reason) {
            return new ActiveAbility(false, reason, -1, 0.0f);
        }
    }

    private record ActorLifecycle(
        boolean available,
        String reason,
        InteractionManager manager,
        Role role,
        CombatSupport combat,
        float[] projectedVelocity,
        float dodgeRemaining,
        float[] cooldownSeconds,
        boolean[] legal,
        int activeSlot,
        float activeElapsedSeconds
    ) {
        private static ActorLifecycle unavailable(String reason) {
            return new ActorLifecycle(
                false,
                reason,
                null,
                null,
                null,
                new float[3],
                0.0f,
                new float[ABILITY_CAPACITY],
                new boolean[ABILITY_CAPACITY],
                -1,
                0.0f
            );
        }

        private static ActorLifecycle absent() {
            return new ActorLifecycle(
                true,
                "",
                null,
                null,
                null,
                new float[3],
                0.0f,
                new float[ABILITY_CAPACITY],
                new boolean[ABILITY_CAPACITY],
                -1,
                0.0f
            );
        }
    }

    private record TargetAttack(
        boolean available,
        String reason,
        CombatPhase phase,
        float progress,
        int reportedSlot,
        String evidenceSource,
        int sequenceCursor,
        int sequenceCount,
        float pauseSeconds
    ) {
        private static TargetAttack idle() {
            return idle(0.0f);
        }

        private static TargetAttack idle(float pauseSeconds) {
            return new TargetAttack(
                true,
                "",
                CombatPhase.IDLE,
                0.0f,
                -1,
                NativePolicyLifecycleSnapshot.TARGET_ATTACK_SOURCE_IDLE,
                -1,
                0,
                pauseSeconds
            );
        }

        private static TargetAttack unavailable(String reason) {
            return new TargetAttack(
                false,
                reason,
                CombatPhase.IDLE,
                0.0f,
                -1,
                NativePolicyLifecycleSnapshot.TARGET_ATTACK_SOURCE_IDLE,
                -1,
                0,
                0.0f
            );
        }
    }
}

