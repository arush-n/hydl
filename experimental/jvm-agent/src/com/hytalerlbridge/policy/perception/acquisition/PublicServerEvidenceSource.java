package com.hytalerlbridge.policy.perception.acquisition;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.logger.HytaleLogger;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.policy.Projection;
import com.hytalerlbridge.policy.perception.acquisition.trace.LifecycleTraceSink;
import com.hytalerlbridge.policy.perception.acquisition.crafting.ServerRecipeCandidateReader;
import com.hytalerlbridge.policy.perception.acquisition.blocks.ServerBlockCandidateReader;
import com.hytalerlbridge.policy.perception.acquisition.motion.ServerDodgeCorridorReader;
import com.hytalerlbridge.policy.perception.acquisition.motion.DodgeReadinessDiagnostic;
import com.hytalerlbridge.policy.perception.model.PerceptionFrame;
import com.hytalerlbridge.policy.perception.profile.PerceptionProfile;
import com.hytalerlbridge.policy.perception.projection.InventoryProjection;
import com.hytalerlbridge.policy.perception.projection.MechanicsProjection;
import com.hytalerlbridge.policy.perception.projection.MovementProjection;
import com.hytalerlbridge.policy.perception.projection.WorldProjection;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Production acquisition from public Server 0.5.7 components.
 *
 * <p>World geometry/light/block candidates remain explicitly unavailable.
 * Fieldcraft candidates are native-catalog and inventory derived; their action
 * head opens only when the matching bridge executor is installed. Bridge-owned
 * combat lifecycle facts are required as one coherent row; absence skips the
 * policy tick rather than fabricating zeros.
 */
public final class PublicServerEvidenceSource
    implements PerceptionEvidenceSource {

    private static final float RESOURCE_MAXIMUM_TOLERANCE = 1.0e-5f;
    private static final long DODGE_DIAGNOSTIC_INTERVAL_NANOS =
        java.util.concurrent.TimeUnit.SECONDS.toNanos(5L);
    private static final HytaleLogger LOGGER = HytaleLogger.forEnclosingClass();

    private final PerceptionProfile profile;
    private final CombatLifecycleEvidenceSource lifecycle;
    private final LifecycleTraceSink lifecycleTrace;
    private final ServerRecipeCandidateReader recipeCandidates;
    private final boolean worldVerbAvailable;
    private final ConcurrentMap<Integer, DamageClock> damageClocks =
        new ConcurrentHashMap<>();
    /** Diagnostics-only rate state; never read by perception or policy. */
    private final ConcurrentMap<Integer, Long> lastDodgeDiagnosticNanos =
        new ConcurrentHashMap<>();

    public PublicServerEvidenceSource(
        PerceptionProfile profile,
        CombatLifecycleEvidenceSource lifecycle
    ) {
        this(profile, lifecycle, LifecycleTraceSink.disabled());
    }

    public PublicServerEvidenceSource(
        PerceptionProfile profile,
        CombatLifecycleEvidenceSource lifecycle,
        LifecycleTraceSink lifecycleTrace
    ) {
        this(
            profile,
            lifecycle,
            lifecycleTrace,
            ServerRecipeCandidateReader.load(),
            false
        );
    }

    /**
     * Full production constructor. Candidate evidence may be live while the
     * executor remains unavailable; {@code worldVerbAvailable} controls only
     * action masks, never whether observations report real candidates.
     */
    public PublicServerEvidenceSource(
        PerceptionProfile profile,
        CombatLifecycleEvidenceSource lifecycle,
        LifecycleTraceSink lifecycleTrace,
        ServerRecipeCandidateReader recipeCandidates,
        boolean worldVerbAvailable
    ) {
        if (profile == null || lifecycle == null || lifecycleTrace == null) {
            throw new IllegalArgumentException(
                "public acquisition requires profile, lifecycle, and trace");
        }
        if (recipeCandidates == null) {
            throw new IllegalArgumentException(
                "public acquisition requires recipe candidates");
        }
        this.profile = profile;
        this.lifecycle = lifecycle;
        this.lifecycleTrace = lifecycleTrace;
        this.recipeCandidates = recipeCandidates;
        this.worldVerbAvailable = worldVerbAvailable;
    }

    @Override
    public PerceptionFrame capture(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        int slot,
        float deltaTime
    ) {
        if (ref == null || !ref.isValid() || npc == null || store == null
            || !Float.isFinite(deltaTime) || deltaTime <= 0.0f) {
            return null;
        }
        ServerActorReader.Actor agent = ServerActorReader.capture(ref, store);
        if (agent == null || agent.health() <= 0.0f) {
            return null;
        }
        Ref<EntityStore> targetRef = ServerActorReader.selectTarget(
            ref, npc, store);
        ServerActorReader.Actor rawTarget = ServerActorReader.capture(
            targetRef, store);
        boolean targetPerceptible = ServerActorReader.perceptible(
            agent, rawTarget, store);

        CombatLifecycleEvidenceSource.Evidence life = lifecycle.capture(
            ref, targetRef, store, slot, deltaTime);
        if (life == null) {
            return null;
        }
        lifecycleTrace.observe(slot, life);

        float agentHealth = scaleHealth(
            agent, (float) profile.params().agentMaxHealth());
        float targetHealth = targetPerceptible
            ? scaleHealth(
                rawTarget, (float) profile.params().targetMaxHealth())
            : 0.0f;
        float[] health = {agentHealth, targetHealth};
        float[] resources = profile.resourceMinimum();
        boolean[] resourceAvailable = new boolean[
            MechanicsProjection.RESOURCE_VALUES];
        copyResources(agent, 0, resources, resourceAvailable);
        if (targetPerceptible) {
            copyResources(rawTarget, 1, resources, resourceAvailable);
        }

        PerceptionFrame.Status status = ServerStatusReader.capture(
            store,
            ref,
            targetPerceptible ? targetRef : null,
            profile.statusPrograms(),
            new float[] {
                (float) profile.params().agentMaxHealth(),
                (float) profile.params().targetMaxHealth(),
            }
        );
        if (status == null) {
            return null;
        }

        float[] appliedVelocity = life.appliedVelocity();
        float[] dodgeRemaining = life.dodgeInvulnerabilityRemaining();
        boolean[] guardActive = {
            agent.guardActive(),
            targetPerceptible && rawTarget.guardActive(),
        };
        boolean[] staminaBroken = {
            agent.staminaBroken(),
            targetPerceptible && rawTarget.staminaBroken(),
        };
        float[] staminaRegenDelay = {
            agent.staminaRegenDelay(),
            targetPerceptible ? rawTarget.staminaRegenDelay() : 0.0f,
        };
        float[] controlImmunity = {
            agent.controlImmunity(),
            targetPerceptible ? rawTarget.controlImmunity() : 0.0f,
        };
        if (!targetPerceptible) {
            appliedVelocity[3] = 0.0f;
            appliedVelocity[4] = 0.0f;
            appliedVelocity[5] = 0.0f;
            dodgeRemaining[1] = 0.0f;
        }
        PerceptionFrame.Defense defense = new PerceptionFrame.Defense(
            guardActive,
            staminaBroken,
            dodgeRemaining,
            appliedVelocity,
            staminaRegenDelay,
            controlImmunity,
            health
        );

        float[] cooldown = life.abilityCooldownSeconds();
        int[] activeSlot = life.activeAbilitySlot();
        float[] elapsed = life.abilityElapsedSeconds();
        boolean[] legal = publishAbilityLegality(life);
        if (!targetPerceptible) {
            java.util.Arrays.fill(cooldown, 16, 32, 0.0f);
            activeSlot[1] = -1;
            elapsed[1] = 0.0f;
        }
        PerceptionFrame.Ability ability = new PerceptionFrame.Ability(
            cooldown, activeSlot, elapsed, legal);
        float stamina = resourceAvailable[0]
            ? resources[0]
            : profile.resourceMinimum()[0];
        ServerDodgeCorridorReader.Capture dodgeCorridor =
            ServerDodgeCorridorReader.captureDetailed(
            ref,
            store,
            agent,
            profile.dodgeMotion()
        );
        boolean[] dodgeCorridorClear = dodgeCorridor.clear();
        boolean movementEnabled = true;
        boolean alive = health[0] > 0.0f;
        boolean[] finalDodgeMask = Projection.dodgeActionMask(profile.dodge(
            dodgeCorridorClear,
            movementEnabled,
            alive,
            stamina
        ));
        DodgeReadinessDiagnostic.Snapshot dodgeDiagnostic =
            DodgeReadinessDiagnostic.classify(
                dodgeCorridor,
                movementEnabled,
                alive,
                resourceAvailable[0],
                stamina,
                profile.dodgeCost(),
                finalDodgeMask
            );
        logDodgeDiagnostic(slot, dodgeDiagnostic);
        PerceptionFrame.Mechanics mechanics = new PerceptionFrame.Mechanics(
            resources,
            resourceAvailable,
            defense,
            status,
            ability,
            dodgeCorridorClear,
            movementEnabled,
            stamina
        );

        Projection.TargetRaw targetRaw = targetRaw(
            agent, rawTarget, targetPerceptible, life, targetHealth);
        Projection.TargetEvidence targetEvidence = Projection.targetEvidence(
            targetRaw, profile.params());
        Projection.TargetState target = targetState(
            agent,
            rawTarget,
            targetPerceptible,
            targetHealth,
            targetEvidence
        );
        double agentHealthFraction = agentHealth
            / profile.params().agentMaxHealth();
        Projection.SelfState self = new Projection.SelfState(
            agent.velocityX(),
            agent.velocityY(),
            agent.velocityZ(),
            agent.yawDegrees(),
            agent.pitchDegrees(),
            agentHealth,
            agent.movement(15),
            life.agentAttackExecuting() ? 1.0 : 0.0,
            life.agentAttackCooldownSeconds(),
            life.agentKnockbackControlLock(),
            ticksSinceDamage(slot, ref, agentHealth),
            agent.movement(2) ? agent.velocityY() : 0.0,
            Math.min(0.0, agent.velocityY()),
            deltaTime
        );
        Projection.CombatSelf combatSelf = new Projection.CombatSelf(
            agent.velocityX(),
            agent.velocityY(),
            agent.velocityZ(),
            agent.yawDegrees(),
            agentHealthFraction,
            life.agentAttackExecuting()
        );

        InventoryProjection.Input inventory = ServerInventoryReader.capture(
            ref, store);
        boolean[] skill = profile.skillMask();
        for (int index = 6; index < skill.length; index++) {
            skill[index] = false;
        }
        ServerRecipeCandidateReader.Capture recipes =
            recipeCandidates.capture(ref, store);
        ServerBlockCandidateReader.Capture blocks =
            ServerBlockCandidateReader.capture(ref, store);
        PerceptionFrame.Actions actions = new PerceptionFrame.Actions(
            skill,
            new boolean[3],
            false,
            profile.jumpConfigured(),
            true,
            true,
            worldVerbAvailable && blocks.useAvailable(),
            worldVerbAvailable,
            new boolean[] {
                worldVerbAvailable && blocks.primaryAvailable(),
                worldVerbAvailable && blocks.secondaryAvailable(),
            }
        );
        PerceptionFrame.WorldGroups world =
            PerceptionFrame.WorldGroups.unavailable();
        world = new PerceptionFrame.WorldGroups(
            world.geometry(),
            world.light(),
            blocks.candidateFeatures(),
            blocks.candidateMask(),
            blocks.available(),
            blocks.bindings(),
            recipes.policy(),
            recipes.recipeIds()
        );
        return new PerceptionFrame(
            true,
            targetPerceptible,
            self,
            target,
            combatSelf,
            targetRaw,
            agent.movementBits(),
            MovementProjection.ALL_BITS,
            new WorldProjection.ActorWorldInput(
                true,
                agent.movement(12),
                false,
                false,
                false,
                false,
                false,
                0.0f,
                profile.maximumDropHeight()
            ),
            mechanics,
            inventory,
            world,
            actions
        );
    }

    private void logDodgeDiagnostic(
        int slot,
        DodgeReadinessDiagnostic.Snapshot diagnostic
    ) {
        long now = System.nanoTime();
        Long previous = lastDodgeDiagnosticNanos.putIfAbsent(slot, now);
        if (previous != null) {
            if (now - previous < DODGE_DIAGNOSTIC_INTERVAL_NANOS
                || !lastDodgeDiagnosticNanos.replace(slot, previous, now)) {
                return;
            }
        }
        LOGGER.at(java.util.logging.Level.INFO).log(
            "PolicyAgent Dodge readiness: slot=%d %s",
            slot,
            diagnostic.receipt()
        );
    }

    /**
     * Publish the bridge's commit-time interaction legality unchanged.
     *
     * <p>The combat sink now executes abilities through
     * {@code NativePolicyCombatFacade}; clearing this row was a legacy guard
     * from the movement-only sink and made every ability permanently
     * unreachable. The lifecycle record owns native cooldown, active-chain,
     * guard-fork, and interaction-rule arbitration, so duplicating or
     * weakening it here would create a second legality model.
     */
    static boolean[] publishAbilityLegality(
        CombatLifecycleEvidenceSource.Evidence lifecycle
    ) {
        if (lifecycle == null) {
            throw new IllegalArgumentException(
                "combat lifecycle evidence is required");
        }
        return lifecycle.abilityLegal();
    }

    private void copyResources(
        ServerActorReader.Actor actor,
        int entity,
        float[] destination,
        boolean[] available
    ) {
        float[] minimum = profile.resourceMinimum();
        float[] maximum = profile.resourceMaximum();
        float[] nativeValues = actor.resourceValues();
        float[] nativeMaximums = actor.resourceMaximums();
        boolean[] nativeAvailable = actor.resourceAvailable();
        int offset = entity * ServerActorReader.RESOURCE_STAT_NAMES.length;
        for (int resource = 0;
            resource < ServerActorReader.RESOURCE_STAT_NAMES.length;
            resource++) {
            int index = offset + resource;
            boolean usable = nativeAvailable[resource]
                && maximum[index] > minimum[index]
                && Math.abs(nativeMaximums[resource] - maximum[index])
                    <= RESOURCE_MAXIMUM_TOLERANCE;
            available[index] = usable;
            destination[index] = usable
                ? Math.max(minimum[index], Math.min(
                    maximum[index], nativeValues[resource]))
                : minimum[index];
        }
    }

    private static float scaleHealth(
        ServerActorReader.Actor actor,
        float expectedMaximum
    ) {
        float fraction = Math.max(0.0f, Math.min(
            1.0f, actor.health() / actor.maximumHealth()));
        return fraction * expectedMaximum;
    }

    private Projection.TargetRaw targetRaw(
        ServerActorReader.Actor agent,
        ServerActorReader.Actor target,
        boolean perceptible,
        CombatLifecycleEvidenceSource.Evidence life,
        float targetHealth
    ) {
        if (target == null) {
            return new Projection.TargetRaw(
                agent.x(), agent.y(), agent.z(),
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0,
                0.0, 0.0, 0.0,
                false, false, 0, 0.0, -1
            );
        }
        return new Projection.TargetRaw(
            agent.x(), agent.y(), agent.z(),
            target.x(), target.y(), target.z(),
            target.velocityX(), target.velocityY(), target.velocityZ(),
            targetHealth,
            target.yawDegrees(),
            target.headYawDegrees(),
            target.headPitchDegrees(),
            perceptible,
            true,
            life.targetAttackPhase(),
            life.targetAttackProgress(),
            life.targetReportedAbilitySlot()
        );
    }

    private Projection.TargetState targetState(
        ServerActorReader.Actor agent,
        ServerActorReader.Actor target,
        boolean perceptible,
        float targetHealth,
        Projection.TargetEvidence evidence
    ) {
        double targetX = target == null ? 0.0 : target.x();
        double targetY = target == null ? 0.0 : target.y();
        double targetZ = target == null ? 0.0 : target.z();
        double velocityX = target == null ? 0.0 : target.velocityX();
        double velocityY = target == null ? 0.0 : target.velocityY();
        double velocityZ = target == null ? 0.0 : target.velocityZ();
        return new Projection.TargetState(
            agent.x(), agent.y(), agent.z(),
            agent.velocityX(), agent.velocityY(), agent.velocityZ(),
            agent.yawDegrees(),
            targetX, targetY, targetZ,
            velocityX, velocityY, velocityZ,
            targetHealth,
            perceptible ? 1.0 : 0.0,
            evidence.facingErrorDegrees()
                / profile.params().facingErrorDegreesScale(),
            evidence.headFacingErrorDegrees()
                / profile.params().facingErrorDegreesScale(),
            evidence.headPitchDegrees()
                / profile.params().headPitchDegreesScale(),
            evidence.attackProgress()
        );
    }

    private float ticksSinceDamage(
        int slot,
        Ref<EntityStore> ref,
        float health
    ) {
        DamageClock previous = damageClocks.get(slot);
        int ticks = 0;
        if (previous != null && previous.reference() == ref) {
            ticks = health < previous.health()
                ? 0
                : Math.min(Integer.MAX_VALUE, previous.ticks() + 1);
        }
        damageClocks.put(slot, new DamageClock(ref, health, ticks));
        return ticks;
    }

    private record DamageClock(
        Ref<EntityStore> reference,
        float health,
        int ticks
    ) {
    }
}
