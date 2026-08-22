package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.builtin.deployables.component.DeployableComponent;
import com.hypixel.hytale.component.ComponentType;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.query.Query;
import com.hypixel.hytale.math.shape.Box;
import com.hypixel.hytale.math.vector.Rotation3f;
import com.hypixel.hytale.server.core.entity.Frozen;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.entity.UUIDComponent;
import com.hypixel.hytale.server.core.entity.entities.ProjectileComponent;
import com.hypixel.hytale.server.core.event.events.ecs.InventoryChangeEvent;
import com.hypixel.hytale.server.core.modules.entity.component.BoundingBox;
import com.hypixel.hytale.server.core.modules.entity.component.HeadRotation;
import com.hypixel.hytale.server.core.modules.entity.component.ModelComponent;
import com.hypixel.hytale.server.core.modules.entity.component.NewSpawnComponent;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.entity.damage.Damage;
import com.hypixel.hytale.server.core.modules.entity.damage.DamageCause;
import com.hypixel.hytale.server.core.modules.entity.DespawnComponent;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatMap;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatValue;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatsModule;
import com.hypixel.hytale.server.core.modules.physics.component.Velocity;
import com.hypixel.hytale.server.core.modules.physics.SimplePhysicsProvider;
import com.hypixel.hytale.server.core.modules.projectile.component.PredictedProjectile;
import com.hypixel.hytale.server.core.modules.projectile.config.StandardPhysicsProvider;
import com.hypixel.hytale.server.core.modules.time.TimeResource;
import com.hypixel.hytale.server.core.modules.time.WorldTimeResource;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.components.StepComponent;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.instructions.Action;
import com.hypixel.hytale.server.npc.instructions.Instruction;
import com.hypixel.hytale.server.npc.corecomponents.combat.ActionAttack;
import com.hypixel.hytale.server.npc.movement.Steering;
import com.hypixel.hytale.server.npc.movement.controllers.MotionController;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.role.support.MarkedEntitySupport;
import com.hypixel.hytale.server.npc.util.IAnnotatedComponent;
import com.hytalerlbridge.imitation.NpcTraceBatch;
import com.hytalerlbridge.imitation.NpcTraceFrame;
import com.hytalerlbridge.imitation.NpcDamageEventSnapshot;
import com.hytalerlbridge.imitation.NpcLifecycleEventSnapshot;
import com.hytalerlbridge.imitation.NpcAttackActionSnapshot;
import com.hytalerlbridge.imitation.NpcAttackExecutionCause;
import com.hytalerlbridge.imitation.NpcInteractionSnapshot;
import com.hytalerlbridge.imitation.NpcInternalSnapshot;
import com.hytalerlbridge.imitation.NpcObservationSnapshot;
import com.hytalerlbridge.imitation.NpcWorldSnapshot;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.IdentityHashMap;
import java.time.Instant;
import java.util.UUID;
import org.joml.Vector3d;
import org.joml.Vector4d;

import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.stat;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.discoverAttackActions;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.interactionManager;

/** World-thread-only recorder at the native behavior-to-steering boundary. */
final class NativeNpcTraceRecorder {

    static final int MAX_ACTIVE_TRACES = 8;

    private final ComponentType<EntityStore, NativeNpcTraceMarker> markerType;
    private final ObservationCapture observationCapture;
    private final boolean suppressMotion;
    private final Map<UUID, Active> traces = new LinkedHashMap<>();
    private final Map<UUID, Active> tracesByNpc = new HashMap<>();
    private long preWorldviewTick = Long.MIN_VALUE;
    private WorldCapture preWorldview;

    NativeNpcTraceRecorder(
        ComponentType<EntityStore, NativeNpcTraceMarker> markerType,
        ObservationCapture observationCapture,
        boolean suppressMotion
    ) {
        this.markerType = markerType;
        this.observationCapture = observationCapture;
        this.suppressMotion = suppressMotion;
    }

    NpcTraceBatch start(
        World world,
        String serverVersion,
        String worldgenProvider,
        String worldgenVersion,
        long seed,
        Map<String, String> environmentParameters,
        UUID npcUuid,
        String expectedRole,
        int capacity
    ) {
        if (npcUuid == null) throw new IllegalArgumentException("npc UUID is required");
        if (tracesByNpc.containsKey(npcUuid)) {
            throw new IllegalStateException("The selected NPC is already being traced");
        }
        if (traces.size() == MAX_ACTIVE_TRACES) {
            throw new IllegalStateException("Too many NPC traces are active");
        }
        if (capacity < 1 || capacity > NpcTraceBatch.MAX_CAPACITY) {
            throw new IllegalArgumentException("trace capacity is out of range");
        }
        Store<EntityStore> store = world.getEntityStore().getStore();
        Ref<EntityStore> ref = world.getEntityStore().getRefFromUUID(npcUuid);
        NPCEntity npc = ref == null || !ref.isValid()
            ? null
            : store.getComponent(ref, NPCEntity.getComponentType());
        UUIDComponent identity = ref == null || !ref.isValid()
            ? null
            : store.getComponent(ref, UUIDComponent.getComponentType());
        if (npc == null || identity == null || !npcUuid.equals(identity.getUuid())) {
            throw new IllegalArgumentException("No live NPC has UUID " + npcUuid);
        }
        String role = npc.getRoleName();
        if (role == null || role.isBlank()) {
            throw new IllegalStateException("Selected NPC has no role asset ID");
        }
        if (expectedRole != null && !expectedRole.isBlank() && !role.equals(expectedRole)) {
            throw new IllegalArgumentException(
                "NPC role changed: expected " + expectedRole + ", got " + role
            );
        }
        Active active = new Active(
            serverVersion,
            world.getName(),
            worldgenProvider,
            worldgenVersion,
            seed,
            environmentParameters,
            UUID.randomUUID(),
            npcUuid,
            role,
            capacity,
            ref,
            discoverAttackActions(npc.getRole())
        );
        store.addComponent(
            ref,
            markerType,
            new NativeNpcTraceMarker(this, npcUuid)
        );
        traces.put(active.traceUuid, active);
        tracesByNpc.put(active.npcUuid, active);
        return active.batch(0);
    }

    NpcTraceBatch drain(Store<EntityStore> store, UUID traceUuid, int maxFrames, boolean stop) {
        Active trace = requireActive(traceUuid);
        if (maxFrames < 1 || maxFrames > NpcTraceBatch.MAX_DRAIN) {
            throw new IllegalArgumentException("maxFrames is out of range");
        }
        if (stop && trace.status.equals("recording")) trace.status = "stopped";
        NpcTraceBatch batch = trace.batch(maxFrames);
        if (stop) {
            removeMarker(store, trace);
            traces.remove(trace.traceUuid);
            tracesByNpc.remove(trace.npcUuid);
        }
        return batch;
    }

    void capturePre(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        float systemDelta
    ) {
        Active trace = matching(ref, npc, store);
        if (trace == null || !trace.status.equals("recording")) return;
        World world = store.getExternalData().getWorld();
        if (store.getComponent(ref, NewSpawnComponent.getComponentType()) != null) {
            trace.pending = null;
            return;
        }
        boolean frozen = store.getComponent(ref, Frozen.getComponentType()) != null
            || world.getWorldConfig().isAllNPCFrozen();
        StepComponent step = store.getComponent(ref, StepComponent.getComponentType());
        if (frozen && step == null) {
            trace.pending = null;
            return;
        }
        float delta = frozen ? step.getTickLength() : systemDelta;
        Role role = npc.getRole();
        if (preWorldviewTick != world.getTick()) {
            preWorldviewTick = world.getTick();
            preWorldview = captureWorld(store);
        }
        Ref<EntityStore> target = role == null ? null : lockedTarget(role);
        NpcWorldSnapshot worldview = preWorldview.snapshot();
        NativeNpcLifecycleCapture.Boundary lifecycle =
            NativeNpcLifecycleCapture.boundary(ref, store);
        NativeNpcLifecycleCapture.Events interRowLifecycle =
            trace.previousPostLifecycle == null
                ? NativeNpcLifecycleCapture.emptyEvents()
                : NativeNpcLifecycleCapture.betweenRows(
                    trace.previousPostLifecycle,
                    lifecycle,
                    trace.previousPostWorldview,
                    worldview
                );
        trace.pending = new Pending(
            world.getTick(),
            delta,
            state(ref, npc, store),
            attackActionActive(role),
            internal(ref, role, store),
            worldview,
            uuid(target, store),
            lifecycle,
            interRowLifecycle,
            observationCapture.capture(
                ref,
                store,
                trace.attackCandidates,
                preWorldview.perceptionCandidates()
            )
        );
    }

    void captureIntent(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        Active trace = matching(ref, npc, store);
        Pending pending = trace == null ? null : trace.pending;
        if (
            pending == null
                || pending.tick != store.getExternalData().getWorld().getTick()
        ) {
            return;
        }
        Role role = npc.getRole();
        if (role == null) return;
        Instruction body = role.getEntitySupport().getNextBodyMotionStep();
        Instruction head = role.getEntitySupport().getNextHeadMotionStep();
        List<NpcAttackActionSnapshot> attackActions = attackActions(body, head);
        boolean attackActionActive = attackActions.stream().anyMatch(
            NpcAttackActionSnapshot::active
        );
        ActorState actor = actorState(
            ref,
            npc,
            store,
            body,
            head,
            trace.attackCandidates,
            false,
            List.of()
        );
        pending.intent = new Intent(
            control(role.getBodySteering(), role.getHeadSteering()),
            controlMask(role.getBodySteering(), role.getHeadSteering()),
            label(body),
            path(body),
            label(head),
            path(head),
            activeActions(body, head),
            activeActionPaths(body, head),
            attackActionActive,
            attackActionActive && !pending.attackActionActive,
            actor
        );
        if (suppressMotion) {
            Steering steering = role.getBodySteering();
            if (steering != null) steering.clearTranslation();
            Velocity velocity = store.getComponent(
                ref,
                Velocity.getComponentType()
            );
            if (velocity != null) velocity.setZero();
        }
    }

    void markBridgeOverride(Ref<EntityStore> ref) {
        for (Active trace : traces.values()) {
            if (trace.ref.equals(ref) && trace.pending != null) {
                trace.pending.nativeControl = false;
                return;
            }
        }
    }

    void captureDamage(
        Ref<EntityStore> target,
        Store<EntityStore> store,
        Damage damage
    ) {
        if (traces.isEmpty() || damage == null) return;
        Ref<EntityStore> source = damage.getSource() instanceof Damage.EntitySource value
            ? value.getRef()
            : null;
        Ref<EntityStore> projectile = damage.getSource() instanceof Damage.ProjectileSource value
            ? value.getProjectile()
            : null;
        String sourceType = damage.getSource() instanceof Damage.ProjectileSource
            ? "projectile"
            : damage.getSource() instanceof Damage.EntitySource
                ? "entity"
                : damage.getSource() instanceof Damage.EnvironmentSource
                    ? "environment"
                    : "other";
        String environmentType = damage.getSource() instanceof Damage.EnvironmentSource value
            ? value.getType()
            : "";
        DamageCause cause = DamageCause.getAssetMap().getAsset(
            damage.getDamageCauseIndex()
        );
        Vector4d hit = damage.getIfPresentMetaObject(Damage.HIT_LOCATION);
        EntityStatMap stats = target == null || !target.isValid()
            ? null
            : store.getComponent(
                target,
                EntityStatsModule.get().getEntityStatMapComponentType()
            );
        EntityStatValue health = stat(stats, "Health");
        float currentHealth = health == null ? 0.0f : (float) health.get();
        float maximumHealth = health == null ? 0.0f : (float) health.getMax();
        UUID sourceUuid = uuid(source, store);
        UUID targetUuid = uuid(target, store);
        UUID projectileUuid = uuid(projectile, store);
        long tick = store.getExternalData().getWorld().getTick();
        for (Active trace : traces.values()) {
            boolean actorSource = trace.ref.equals(source);
            boolean actorTarget = trace.ref.equals(target);
            if (!actorSource && !actorTarget) continue;
            trace.addDamageEvent(
                tick,
                new NpcDamageEventSnapshot(
                    sourceType,
                    environmentType,
                    damage.getDamageCauseIndex(),
                    cause == null ? "" : cause.getId(),
                    damage.getInitialAmount(),
                    damage.getAmount(),
                    damage.isCancelled(),
                    Boolean.TRUE.equals(
                        damage.getMetaStore().getMetaObject(Damage.BLOCKED)
                    ),
                    actorSource,
                    actorTarget,
                    entityIndex(source),
                    sourceUuid,
                    entityIndex(target),
                    targetUuid,
                    entityIndex(projectile),
                    projectileUuid,
                    hit != null,
                    hit == null
                        ? new double[4]
                        : new double[] {hit.x, hit.y, hit.z, hit.w},
                    health != null,
                    currentHealth,
                    maximumHealth,
                    health != null && currentHealth <= 0.0f
                )
            );
        }
    }

    void captureInventory(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        InventoryChangeEvent event
    ) {
        Active trace = traces.values().stream()
            .filter(value -> value.ref.equals(ref))
            .findFirst()
            .orElse(null);
        if (trace == null || !trace.status.equals("recording")) return;
        NativeNpcLifecycleCapture.InventoryCapture captured =
            NativeNpcLifecycleCapture.inventory(ref, store, event);
        trace.addInventoryEvents(
            store.getExternalData().getWorld().getTick(),
            captured.events(),
            captured.count(),
            captured.partial()
        );
    }

    void afterTick(Store<EntityStore> store) {
        if (traces.isEmpty()) return;
        WorldCapture nextWorldview = captureWorld(store);
        for (Active trace : traces.values()) {
            afterTick(store, trace, nextWorldview);
        }
    }

    private void afterTick(
        Store<EntityStore> store,
        Active trace,
        WorldCapture nextWorldview
    ) {
        if (trace == null || !trace.status.equals("recording")) return;
        Ref<EntityStore> ref = trace.ref;
        if (ref == null || !ref.isValid()) {
            trace.status = "target_missing";
            trace.pending = null;
            return;
        }
        NPCEntity npc = store.getComponent(ref, NPCEntity.getComponentType());
        UUIDComponent identity = store.getComponent(ref, UUIDComponent.getComponentType());
        if (
            npc == null
                || identity == null
                || !trace.npcUuid.equals(identity.getUuid())
        ) {
            trace.status = "target_missing";
            trace.pending = null;
            return;
        }
        if (!trace.role.equals(npc.getRoleName())) {
            trace.status = "role_changed";
            trace.pending = null;
            return;
        }
        Pending pending = trace.pending;
        trace.pending = null;
        if (pending == null || pending.intent == null) return;
        if (trace.frames.size() == trace.capacity) {
            trace.status = "overflow";
            return;
        }
        Intent intent = pending.intent;
        Role role = npc.getRole();
        Instruction nextBody = role == null
            ? null
            : role.getEntitySupport().getNextBodyMotionStep();
        Instruction nextHead = role == null
            ? null
            : role.getEntitySupport().getNextHeadMotionStep();
        ActorState nextActor = actorState(
            ref,
            npc,
            store,
            nextBody,
            nextHead,
            trace.attackCandidates,
            true,
            nextWorldview.perceptionCandidates()
        );
        DamageEvents damageEvents = trace.takeDamageEvents(pending.tick);
        NativeNpcLifecycleCapture.InventoryCapture inventoryEvents =
            trace.takeInventoryEvents(pending.tick);
        NativeNpcLifecycleCapture.Boundary nextLifecycle =
            NativeNpcLifecycleCapture.boundary(ref, store);
        NpcWorldSnapshot nextWorld = nextWorldview.snapshot();
        NativeNpcLifecycleCapture.Events lifecycleEvents =
            NativeNpcLifecycleCapture.combine(
                pending.interRowLifecycle,
                NativeNpcLifecycleCapture.derive(
                    pending.lifecycle,
                    nextLifecycle,
                    pending.worldview,
                    nextWorld,
                    inventoryEvents.events(),
                    inventoryEvents.count(),
                    inventoryEvents.partial()
                )
            );
        trace.frames.addLast(new NpcTraceFrame(
            pending.tick,
            pending.deltaSeconds,
            pending.state,
            intent.control,
            intent.controlMask,
            state(ref, npc, store),
            pending.nativeControl,
            intent.attackActionActive,
            intent.attackActivation,
            attackExecutionCause(
                pending.observation,
                nextActor.observation,
                intent.actor.interactions
            ),
            intent.actor.combatAttack,
            nextActor.combatAttack,
            intent.actor.attackPauseSeconds,
            nextActor.attackPauseSeconds,
            pending.targetUuid,
            nextActor.targetUuid,
            intent.actor.targetUuid,
            pending.internal.stateName(),
            nextActor.internal.stateName(),
            pending.internal,
            intent.actor.internal,
            nextActor.internal,
            intent.bodyInstruction,
            intent.bodyDecisionPath,
            intent.headInstruction,
            intent.headDecisionPath,
            intent.activeActions,
            intent.activeActionPaths,
            intent.actor.attackActions,
            nextActor.attackActions,
            intent.actor.interactions,
            nextActor.interactions,
            damageEvents.events(),
            damageEvents.count(),
            damageEvents.overflow(),
            lifecycleEvents.events(),
            lifecycleEvents.count(),
            lifecycleEvents.overflow(),
            lifecycleEvents.availableBits(),
            lifecycleEvents.partialBits(),
            pending.observation,
            nextActor.observation,
            pending.worldview,
            nextWorld
        ));
        trace.previousPostLifecycle = nextLifecycle;
        trace.previousPostWorldview = nextWorld;
        trace.totalCaptured++;
    }

    private Active matching(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        if (npc == null) return null;
        UUIDComponent identity = store.getComponent(ref, UUIDComponent.getComponentType());
        if (identity == null) return null;
        Active trace = tracesByNpc.get(identity.getUuid());
        return trace != null && trace.ref.equals(ref) ? trace : null;
    }

    private Active requireActive(UUID traceUuid) {
        if (traces.isEmpty()) throw new IllegalStateException("No NPC trace is active");
        Active active = traceUuid == null ? null : traces.get(traceUuid);
        if (active == null) {
            throw new IllegalArgumentException("NPC trace UUID does not match");
        }
        return active;
    }

    private void removeMarker(Store<EntityStore> store, Active trace) {
        if (trace.ref != null && trace.ref.isValid()) {
            store.removeComponentIfExists(trace.ref, markerType);
        }
    }

    private static double[] state(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        TransformComponent transform = store.getComponent(
            ref,
            TransformComponent.getComponentType()
        );
        if (transform == null) {
            throw new IllegalStateException("Traced NPC lost TransformComponent");
        }
        Velocity velocity = store.getComponent(ref, Velocity.getComponentType());
        HeadRotation head = store.getComponent(ref, HeadRotation.getComponentType());
        Role role = npc.getRole();
        EntityStatMap stats = store.getComponent(
            ref,
            EntityStatsModule.get().getEntityStatMapComponentType()
        );
        EntityStatValue health = stat(stats, "Health");
        double currentHealth = health == null
            ? (role == null ? 0.0 : role.getInitialMaxHealth())
            : health.get();
        double maxHealth = health == null
            ? currentHealth
            : health.getMax();
        Vector3d position = transform.getPosition();
        var rotation = transform.getRotation();
        var headRotation = head == null ? rotation : head.getRotation();
        return new double[] {
            position.x,
            position.y,
            position.z,
            velocity == null ? 0.0 : velocity.getX(),
            velocity == null ? 0.0 : velocity.getY(),
            velocity == null ? 0.0 : velocity.getZ(),
            rotation.yaw(),
            rotation.pitch(),
            rotation.roll(),
            headRotation.yaw(),
            headRotation.pitch(),
            currentHealth,
            maxHealth,
            role != null && role.isOnGround() ? 1.0 : 0.0
        };
    }

    private static double[] control(Steering body, Steering head) {
        return new double[] {
            body == null ? 0.0 : body.getX(),
            body == null ? 0.0 : body.getY(),
            body == null ? 0.0 : body.getZ(),
            body == null || !body.hasYawOrDirection() ? 0.0 : body.getYawOrDirection(),
            body == null || !body.hasPitchOrDirection() ? 0.0 : body.getPitchOrDirection(),
            head == null || !head.hasYawOrDirection() ? 0.0 : head.getYawOrDirection(),
            head == null || !head.hasPitchOrDirection() ? 0.0 : head.getPitchOrDirection(),
            body == null ? 1.0 : body.getRelativeTurnSpeed(),
            head == null ? 1.0 : head.getRelativeTurnSpeed()
        };
    }

    private static int controlMask(Steering body, Steering head) {
        int mask = body == null ? 0 : 1 << 5;
        if (head != null) mask |= 1 << 6;
        if (body != null && body.hasTranslation()) mask |= 1;
        if (body != null && body.hasYawOrDirection()) mask |= 1 << 1;
        if (body != null && body.hasPitchOrDirection()) mask |= 1 << 2;
        if (head != null && head.hasYawOrDirection()) mask |= 1 << 3;
        if (head != null && head.hasPitchOrDirection()) mask |= 1 << 4;
        return mask;
    }

    private static String label(Instruction instruction) {
        return instruction == null ? "" : instruction.getLabel();
    }

    private static String path(Instruction instruction) {
        return instruction == null ? "" : instruction.getBreadCrumbs();
    }

    private static List<String> activeActions(Instruction... instructions) {
        LinkedHashSet<String> labels = new LinkedHashSet<>();
        for (Instruction instruction : instructions) {
            if (instruction == null) continue;
            for (int index = 0; index < instruction.componentCount(); index++) {
                IAnnotatedComponent component = instruction.getComponent(index);
                if (component instanceof Action action && action.isActivated()) {
                    labels.add(component.getLabel());
                }
            }
        }
        return List.copyOf(labels);
    }

    private static List<String> activeActionPaths(Instruction... instructions) {
        LinkedHashSet<String> paths = new LinkedHashSet<>();
        for (Instruction instruction : instructions) {
            if (instruction == null) continue;
            for (int index = 0; index < instruction.componentCount(); index++) {
                IAnnotatedComponent component = instruction.getComponent(index);
                if (component instanceof Action action && action.isActivated()) {
                    paths.add(component.getBreadCrumbs());
                }
            }
        }
        return List.copyOf(paths);
    }

    private static List<NpcAttackActionSnapshot> attackActions(
        Instruction... instructions
    ) {
        var seen = java.util.Collections.newSetFromMap(
            new IdentityHashMap<ActionAttack, Boolean>()
        );
        List<NpcAttackActionSnapshot> result = new ArrayList<>();
        for (Instruction instruction : instructions) {
            if (instruction == null) continue;
            for (int index = 0; index < instruction.componentCount(); index++) {
                IAnnotatedComponent component = instruction.getComponent(index);
                if (component instanceof ActionAttack attack && seen.add(attack)) {
                    result.add(NativeNpcCombatIntrospection.attack(attack));
                }
            }
        }
        return List.copyOf(result);
    }

    private static boolean attackActionActive(Role role) {
        if (role == null) return false;
        return attackActions(
            role.getEntitySupport().getNextBodyMotionStep(),
            role.getEntitySupport().getNextHeadMotionStep()
        ).stream().anyMatch(NpcAttackActionSnapshot::active);
    }

    private static Ref<EntityStore> lockedTarget(Role role) {
        return role.getMarkedEntitySupport().getMarkedEntityRef(
            MarkedEntitySupport.DEFAULT_TARGET_SLOT
        );
    }

    private static UUID uuid(Ref<EntityStore> ref, Store<EntityStore> store) {
        UUIDComponent identity = ref == null || !ref.isValid()
            ? null
            : store.getComponent(ref, UUIDComponent.getComponentType());
        return identity == null ? null : identity.getUuid();
    }

    private static int entityIndex(Ref<EntityStore> ref) {
        return ref == null || !ref.isValid() ? -1 : ref.getIndex();
    }

    private static UUID lockedTargetUuid(Role role, Store<EntityStore> store) {
        Ref<EntityStore> target = lockedTarget(role);
        UUIDComponent identity = target == null || !target.isValid()
            ? null
            : store.getComponent(target, UUIDComponent.getComponentType());
        return identity == null ? null : identity.getUuid();
    }

    private static NpcInternalSnapshot internal(
        Ref<EntityStore> ref,
        Role role,
        Store<EntityStore> store
    ) {
        if (role == null) {
            return new NpcInternalSnapshot(
                "", -1, -1, false, false, false, false, false, "",
                false, false, false, 0.0, 0.0,
                new double[3], new double[3], List.of()
            );
        }
        var state = role.getStateSupport();
        MotionController motion = role.getActiveMotionController();
        List<NpcInternalSnapshot.MarkedTarget> targets = new ArrayList<>();
        MarkedEntitySupport marked = role.getMarkedEntitySupport();
        for (int slot = 0; slot < marked.getMarkedEntitySlotCount(); slot++) {
            Ref<EntityStore> target = marked.getMarkedEntityRef(slot);
            UUID targetUuid = uuid(target, store);
            if (targetUuid != null) {
                targets.add(new NpcInternalSnapshot.MarkedTarget(
                    slot,
                    marked.getSlotName(slot),
                    targetUuid
                ));
            }
        }
        Vector3d avoidance = role.getLastAvoidanceSteering();
        Vector3d separation = role.getLastSeparationSteering();
        return new NpcInternalSnapshot(
            state == null ? "" : state.getStateName(),
            state == null ? -1 : state.getStateIndex(),
            state == null ? -1 : state.getSubStateIndex(),
            state != null && state.isInBusyState(),
            state != null && state.isRunningTransitionActions(),
            role.isRoleChangeRequested(),
            role.hasReachedTerminalAction(),
            role.isBackingAway(),
            role.getSteeringMotionName(),
            motion != null,
            motion != null && motion.isInProgress(),
            motion != null && motion.isObstructed(),
            motion == null ? 0.0 : motion.getCurrentSpeed(),
            motion == null ? 0.0 : motion.getMaximumSpeed(),
            vector(avoidance),
            vector(separation),
            targets
        );
    }

    private static double[] vector(Vector3d value) {
        return value == null
            ? new double[3]
            : new double[] {value.x, value.y, value.z};
    }

    private ActorState actorState(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        Instruction body,
        Instruction head,
        List<ActionAttack> attackCandidates,
        boolean captureObservation,
        List<Ref<EntityStore>> perceptionCandidates
    ) {
        Role role = npc.getRole();
        var combat = role == null ? null : role.getCombatSupport();
        Ref<EntityStore> target = role == null ? null : lockedTarget(role);
        return new ActorState(
            combat != null && combat.isExecutingAttack(),
            NativeNpcCombatIntrospection.attackPauseSeconds(combat),
            uuid(target, store),
            attackActions(body, head),
            interactions(ref, store, NativeNpcCombatIntrospection.activeAttack(combat)),
            captureObservation
                ? observationCapture.capture(
                    ref,
                    store,
                    attackCandidates,
                    perceptionCandidates
                )
                : null,
            internal(ref, role, store)
        );
    }

    private static WorldCapture captureWorld(Store<EntityStore> store) {
        World world = store.getExternalData().getWorld();
        WorldTimeResource time = store.getResource(WorldTimeResource.getResourceType());
        TimeResource simulationTime = store.getResource(TimeResource.getResourceType());
        List<NpcWorldSnapshot.Actor> actors = new ArrayList<>();
        List<NpcWorldSnapshot.Entity> entities = new ArrayList<>();
        List<Ref<EntityStore>> perceptionCandidates = new ArrayList<>();
        Query<EntityStore> query = Query.and(TransformComponent.getComponentType());
        store.forEachChunk(query, (chunk, commandBuffer) -> {
            for (int index = 0; index < chunk.size(); index++) {
                Ref<EntityStore> ref = chunk.getReferenceTo(index);
                TransformComponent transform = chunk.getComponent(
                    index,
                    TransformComponent.getComponentType()
                );
                if (transform == null) continue;
                NpcWorldSnapshot.Entity entity = worldEntity(ref, transform, store);
                entities.add(entity);
                if ((entity.flags() & (
                    NpcWorldSnapshot.ENTITY_NPC
                        | NpcWorldSnapshot.ENTITY_LEGACY_PROJECTILE
                        | NpcWorldSnapshot.ENTITY_STANDARD_PROJECTILE
                        | NpcWorldSnapshot.ENTITY_PREDICTED_PROJECTILE
                )) != 0) perceptionCandidates.add(ref);
                UUIDComponent identity = chunk.getComponent(
                    index,
                    UUIDComponent.getComponentType()
                );
                NPCEntity npc = chunk.getComponent(index, NPCEntity.getComponentType());
                if (identity == null || npc == null) continue;
                Role role = npc.getRole();
                var combat = role == null ? null : role.getCombatSupport();
                actors.add(new NpcWorldSnapshot.Actor(
                    identity.getUuid(),
                    npc.getRoleName(),
                    state(ref, npc, store),
                    role == null || role.getStateSupport() == null
                        ? ""
                        : role.getStateSupport().getStateName(),
                    role == null ? null : lockedTargetUuid(role, store),
                    combat != null && combat.isExecutingAttack(),
                    NativeNpcCombatIntrospection.attackPauseSeconds(combat)
                ));
            }
        });
        actors.sort((left, right) -> {
            int most = Long.compareUnsigned(
                left.uuid().getMostSignificantBits(),
                right.uuid().getMostSignificantBits()
            );
            return most != 0
                ? most
                : Long.compareUnsigned(
                    left.uuid().getLeastSignificantBits(),
                    right.uuid().getLeastSignificantBits()
                );
        });
        int count = actors.size();
        List<NpcWorldSnapshot.Actor> emitted = count <= NpcWorldSnapshot.ACTOR_CAPACITY
            ? actors
            : actors.subList(0, NpcWorldSnapshot.ACTOR_CAPACITY);
        entities.sort(java.util.Comparator.comparingInt(
            NpcWorldSnapshot.Entity::entityIndex
        ));
        int entityCount = entities.size();
        List<NpcWorldSnapshot.Entity> emittedEntities =
            entityCount <= NpcWorldSnapshot.ENTITY_CAPACITY
                ? entities
                : entities.subList(0, NpcWorldSnapshot.ENTITY_CAPACITY);
        var gameTime = time == null ? null : time.getGameTime();
        return new WorldCapture(
            new NpcWorldSnapshot(
                world.getTick(),
                gameTime == null ? 0L : gameTime.getEpochSecond(),
                gameTime == null ? 0 : gameTime.getNano(),
                time == null ? 0.0f : time.getDayProgress(),
                time == null ? 0.0f : (float) time.getSunlightFactor(),
                time == null ? 0 : time.getMoonPhase(),
                count,
                count > emitted.size(),
                emitted,
                entityCount,
                entityCount > emittedEntities.size(),
                emittedEntities,
                simulationTime.getNow().getEpochSecond(),
                simulationTime.getNow().getNano()
            ),
            List.copyOf(perceptionCandidates)
        );
    }

    private record WorldCapture(
        NpcWorldSnapshot snapshot,
        List<Ref<EntityStore>> perceptionCandidates
    ) {}

    private static NpcWorldSnapshot.Entity worldEntity(
        Ref<EntityStore> ref,
        TransformComponent transform,
        Store<EntityStore> store
    ) {
        UUIDComponent identity = store.getComponent(
            ref,
            UUIDComponent.getComponentType()
        );
        NPCEntity npc = store.getComponent(ref, NPCEntity.getComponentType());
        ProjectileComponent legacy = store.getComponent(
            ref,
            ProjectileComponent.getComponentType()
        );
        var standardMarker = store.getComponent(
            ref,
            com.hypixel.hytale.server.core.modules.projectile.component.Projectile
                .getComponentType()
        );
        StandardPhysicsProvider standard = store.getComponent(
            ref,
            StandardPhysicsProvider.getComponentType()
        );
        PredictedProjectile predicted = store.getComponent(
            ref,
            PredictedProjectile.getComponentType()
        );
        DeployableComponent deployable = store.getComponent(
            ref,
            DeployableComponent.getComponentType()
        );
        DespawnComponent despawn = store.getComponent(
            ref,
            DespawnComponent.getComponentType()
        );
        Velocity velocity = store.getComponent(ref, Velocity.getComponentType());
        BoundingBox bounding = store.getComponent(
            ref,
            BoundingBox.getComponentType()
        );
        ModelComponent model = store.getComponent(
            ref,
            ModelComponent.getComponentType()
        );

        int flags = npc == null ? 0 : NpcWorldSnapshot.ENTITY_NPC;
        if (legacy != null) flags |= NpcWorldSnapshot.ENTITY_LEGACY_PROJECTILE;
        if (standardMarker != null || standard != null) {
            flags |= NpcWorldSnapshot.ENTITY_STANDARD_PROJECTILE;
        }
        if (predicted != null) {
            flags |= NpcWorldSnapshot.ENTITY_PREDICTED_PROJECTILE;
        }
        if (deployable != null) {
            flags |= NpcWorldSnapshot.ENTITY_DEPLOYABLE;
        }
        SimplePhysicsProvider legacyPhysics = legacy == null
            ? null
            : legacy.getSimplePhysicsProvider();
        if (
            (legacyPhysics != null && legacyPhysics.isOnGround())
                || (standard != null && standard.isOnGround())
        ) flags |= NpcWorldSnapshot.ENTITY_ON_GROUND;
        if (
            (legacyPhysics != null && legacyPhysics.isSwimming())
                || (standard != null && standard.isInFluid())
        ) flags |= NpcWorldSnapshot.ENTITY_IN_FLUID;
        if (
            (legacyPhysics != null && legacyPhysics.isImpacted())
                || (standard != null && standard.isBounced())
        ) flags |= NpcWorldSnapshot.ENTITY_IMPACTED_OR_BOUNCED;
        if (
            (legacyPhysics != null && legacyPhysics.isResting())
                || (standard != null && standard.isSliding())
        ) flags |= NpcWorldSnapshot.ENTITY_RESTING_OR_SLIDING;

        Vector3d position = transform.getPosition();
        Rotation3f rotation = transform.getRotation();
        Vector3d velocityValue = velocity == null
            ? (standard == null ? null : standard.getVelocity())
            : new Vector3d(velocity.getX(), velocity.getY(), velocity.getZ());
        Box box = bounding == null ? null : bounding.getBoundingBox();
        String modelId = model == null || model.getModel() == null
            ? ""
            : model.getModel().getModelAssetId();
        String assetId = npc != null
            ? npc.getRoleName()
            : (legacy != null
                ? legacy.getProjectileAssetName()
                : (deployable == null || deployable.getConfig() == null
                    ? ""
                    : deployable.getConfig().getId()));
        UUID owner = legacy != null
            ? legacy.getCreatorUuid()
            : (standard != null
                ? standard.getCreatorUuid()
                : (deployable == null ? null : deployable.getOwnerUUID()));
        String physicsState = standard == null || standard.getState() == null
            ? ""
            : standard.getState().name();
        Instant lifecycleStart = deployable == null
            ? null
            : deployable.getSpawnInstant();
        Instant lifecycleEnd = despawn == null ? null : despawn.getDespawn();
        boolean lifecycleTimingAvailable = lifecycleStart != null
            && lifecycleEnd != null;
        return new NpcWorldSnapshot.Entity(
            ref.getIndex(),
            identity == null
                ? (predicted == null ? null : predicted.getUuid())
                : identity.getUuid(),
            flags,
            assetId,
            modelId,
            new double[] {position.x, position.y, position.z},
            new double[] {rotation.yaw(), rotation.pitch(), rotation.roll()},
            velocityValue != null,
            velocityValue == null
                ? new double[3]
                : new double[] {velocityValue.x, velocityValue.y, velocityValue.z},
            box != null,
            box == null
                ? new double[6]
                : new double[] {
                    box.min.x, box.min.y, box.min.z,
                    box.max.x, box.max.y, box.max.z
            },
            owner,
            physicsState,
            lifecycleTimingAvailable,
            lifecycleTimingAvailable ? lifecycleStart.getEpochSecond() : 0L,
            lifecycleTimingAvailable ? lifecycleStart.getNano() : 0,
            lifecycleTimingAvailable ? lifecycleEnd.getEpochSecond() : 0L,
            lifecycleTimingAvailable ? lifecycleEnd.getNano() : 0
        );
    }

    private static List<NpcInteractionSnapshot> interactions(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        InteractionChain activeAttack
    ) {
        InteractionManager manager = interactionManager(ref, store);
        Map<InteractionChain, String> chains = new IdentityHashMap<>();
        if (manager != null) {
            for (InteractionChain chain : manager.getChains().values()) {
                if (chain != null) chains.put(chain, "interaction_manager");
            }
        }
        if (activeAttack != null) {
            chains.merge(activeAttack, "combat_support", (left, right) -> left + "+" + right);
        }
        if (chains.isEmpty()) return List.of();
        List<NpcInteractionSnapshot> result = new ArrayList<>(chains.size());
        for (var entry : chains.entrySet()) {
            InteractionChain chain = entry.getKey();
            if (chain == null) continue;
            var context = chain.getContext();
            Ref<EntityStore> target = context == null ? null : context.getTargetEntity();
            UUIDComponent targetIdentity = target == null || !target.isValid()
                ? null
                : store.getComponent(target, UUIDComponent.getComponentType());
            result.add(new NpcInteractionSnapshot(
                entry.getValue(),
                name(chain.getType()),
                name(chain.getBaseType()),
                chain.getChainId(),
                chain.getInitialRootInteraction() == null
                    ? ""
                    : chain.getInitialRootInteraction().getId(),
                chain.getRootInteraction() == null ? "" : chain.getRootInteraction().getId(),
                name(chain.getServerState()),
                name(chain.getClientState()),
                name(chain.getFinalState()),
                chain.getTimeInSeconds(),
                chain.getTimeShift(),
                chain.getOperationCounter(),
                chain.getSimulatedOperationCounter(),
                chain.getOperationIndex(),
                chain.getClientOperationIndex(),
                chain.getCallDepth(),
                chain.getSimulatedCallDepth(),
                chain.isPredicted(),
                chain.requiresClient(),
                chain.isFirstRun(),
                chain.wasPreTicked(),
                chain.isDesynced(),
                targetIdentity == null ? null : targetIdentity.getUuid()
            ));
        }
        result.sort(java.util.Comparator.comparingInt(NpcInteractionSnapshot::chainId));
        return List.copyOf(result);
    }

    private static NpcAttackExecutionCause attackExecutionCause(
        NpcObservationSnapshot current,
        NpcObservationSnapshot next,
        List<NpcInteractionSnapshot> interactions
    ) {
        if (
            current.attackCandidateOverflow()
                || next.attackCandidateOverflow()
                || current.attackCandidateCount() != next.attackCandidateCount()
        ) {
            return NpcAttackExecutionCause.unavailable();
        }
        List<NpcAttackActionSnapshot> before = current.attackCandidates();
        List<NpcAttackActionSnapshot> after = next.attackCandidates();
        for (int index = 0; index < before.size(); index++) {
            if (
                !before.get(index).path().equals(after.get(index).path())
                    || !before.get(index).label().equals(after.get(index).label())
            ) {
                return NpcAttackExecutionCause.unavailable();
            }
        }
        LinkedHashSet<String> roots = new LinkedHashSet<>();
        String interactionType = "";
        for (NpcInteractionSnapshot interaction : interactions) {
            if (
                interaction.source().equals("combat_support")
                    && interaction.firstRun()
                    && !interaction.initialRootId().isEmpty()
            ) {
                roots.add(interaction.initialRootId());
                interactionType = interaction.type();
            }
        }
        if (roots.isEmpty()) return NpcAttackExecutionCause.none();
        if (roots.size() != 1) return NpcAttackExecutionCause.unavailable();
        String root = roots.getFirst();
        int match = -1;
        for (int index = 0; index < after.size(); index++) {
            if (!root.equals(after.get(index).interactionId())) continue;
            if (match >= 0) return NpcAttackExecutionCause.unavailable();
            match = index;
        }
        if (match < 0) return NpcAttackExecutionCause.unavailable();
        NpcAttackActionSnapshot candidate = after.get(match);
        if (!interactionType.equals(candidate.interactionType())) {
            return NpcAttackExecutionCause.unavailable();
        }
        return new NpcAttackExecutionCause(
            true,
            true,
            match,
            interactionType,
            root,
            candidate.path()
        );
    }

    private static String name(Enum<?> value) {
        return value == null ? "" : value.name();
    }

    private static final class Pending {
        private final long tick;
        private final float deltaSeconds;
        private final double[] state;
        private final boolean attackActionActive;
        private final NpcInternalSnapshot internal;
        private final NpcWorldSnapshot worldview;
        private final UUID targetUuid;
        private final NativeNpcLifecycleCapture.Boundary lifecycle;
        private final NativeNpcLifecycleCapture.Events interRowLifecycle;
        private final NpcObservationSnapshot observation;
        private Intent intent;
        private boolean nativeControl = true;

        private Pending(
            long tick,
            float deltaSeconds,
            double[] state,
            boolean attackActionActive,
            NpcInternalSnapshot internal,
            NpcWorldSnapshot worldview,
            UUID targetUuid,
            NativeNpcLifecycleCapture.Boundary lifecycle,
            NativeNpcLifecycleCapture.Events interRowLifecycle,
            NpcObservationSnapshot observation
        ) {
            this.tick = tick;
            this.deltaSeconds = deltaSeconds;
            this.state = state;
            this.attackActionActive = attackActionActive;
            this.internal = internal;
            this.worldview = worldview;
            this.targetUuid = targetUuid;
            this.lifecycle = lifecycle;
            this.interRowLifecycle = interRowLifecycle;
            this.observation = observation;
        }
    }

    private record Intent(
        double[] control,
        int controlMask,
        String bodyInstruction,
        String bodyDecisionPath,
        String headInstruction,
        String headDecisionPath,
        List<String> activeActions,
        List<String> activeActionPaths,
        boolean attackActionActive,
        boolean attackActivation,
        ActorState actor
    ) {}

    private record ActorState(
        boolean combatAttack,
        float attackPauseSeconds,
        UUID targetUuid,
        List<NpcAttackActionSnapshot> attackActions,
        List<NpcInteractionSnapshot> interactions,
        NpcObservationSnapshot observation,
        NpcInternalSnapshot internal
    ) {}

    private record DamageEvents(
        List<NpcDamageEventSnapshot> events,
        int count,
        boolean overflow
    ) {
        private static final DamageEvents EMPTY = new DamageEvents(
            List.of(), 0, false
        );
    }

    @FunctionalInterface
    interface ObservationCapture {
        NpcObservationSnapshot capture(
            Ref<EntityStore> ref,
            Store<EntityStore> store,
            List<ActionAttack> attackCandidates,
            List<Ref<EntityStore>> perceptionCandidates
        );
    }

    private static final class Active {
        private final String serverVersion;
        private final String world;
        private final String worldgenProvider;
        private final String worldgenVersion;
        private final long seed;
        private final Map<String, String> environmentParameters;
        private final UUID traceUuid;
        private final UUID npcUuid;
        private final String role;
        private final int capacity;
        private final Ref<EntityStore> ref;
        private final List<ActionAttack> attackCandidates;
        private final ArrayDeque<NpcTraceFrame> frames = new ArrayDeque<>();
        private final List<NpcDamageEventSnapshot> damageEvents = new ArrayList<>();
        private final List<NpcLifecycleEventSnapshot> inventoryEvents = new ArrayList<>();
        private String status = "recording";
        private long totalCaptured;
        private Pending pending;
        private long damageEventTick = Long.MIN_VALUE;
        private int damageEventCount;
        private long inventoryEventTick = Long.MIN_VALUE;
        private int inventoryEventCount;
        private boolean inventoryPartial;
        private NativeNpcLifecycleCapture.Boundary previousPostLifecycle;
        private NpcWorldSnapshot previousPostWorldview;

        private Active(
            String serverVersion,
            String world,
            String worldgenProvider,
            String worldgenVersion,
            long seed,
            Map<String, String> environmentParameters,
            UUID traceUuid,
            UUID npcUuid,
            String role,
            int capacity,
            Ref<EntityStore> ref,
            List<ActionAttack> attackCandidates
        ) {
            this.serverVersion = serverVersion;
            this.world = world;
            this.worldgenProvider = worldgenProvider;
            this.worldgenVersion = worldgenVersion;
            this.seed = seed;
            this.environmentParameters = Map.copyOf(environmentParameters);
            this.traceUuid = traceUuid;
            this.npcUuid = npcUuid;
            this.role = role;
            this.capacity = capacity;
            this.ref = ref;
            this.attackCandidates = List.copyOf(attackCandidates);
        }

        private void addDamageEvent(long tick, NpcDamageEventSnapshot event) {
            if (damageEventTick != tick) {
                damageEvents.clear();
                damageEventCount = 0;
                damageEventTick = tick;
            }
            damageEventCount++;
            if (damageEvents.size() < NpcDamageEventSnapshot.CAPACITY) {
                damageEvents.add(event);
            }
        }

        private DamageEvents takeDamageEvents(long tick) {
            if (damageEventTick != tick) return DamageEvents.EMPTY;
            DamageEvents result = new DamageEvents(
                List.copyOf(damageEvents),
                damageEventCount,
                damageEventCount > damageEvents.size()
            );
            damageEvents.clear();
            damageEventCount = 0;
            damageEventTick = Long.MIN_VALUE;
            return result;
        }

        private void addInventoryEvents(
            long tick,
            List<NpcLifecycleEventSnapshot> events,
            int count,
            boolean partial
        ) {
            if (inventoryEventTick != tick) {
                inventoryEvents.clear();
                inventoryEventCount = 0;
                inventoryPartial = false;
                inventoryEventTick = tick;
            }
            inventoryPartial |= partial;
            inventoryEventCount += count;
            int remaining = NpcLifecycleEventSnapshot.CAPACITY - inventoryEvents.size();
            inventoryEvents.addAll(events.subList(0, Math.min(remaining, events.size())));
        }

        private NativeNpcLifecycleCapture.InventoryCapture takeInventoryEvents(
            long tick
        ) {
            if (inventoryEventTick != tick) {
                return new NativeNpcLifecycleCapture.InventoryCapture(
                    List.of(), 0, false
                );
            }
            NativeNpcLifecycleCapture.InventoryCapture result =
                new NativeNpcLifecycleCapture.InventoryCapture(
                    List.copyOf(inventoryEvents),
                    inventoryEventCount,
                    inventoryPartial || inventoryEventCount > inventoryEvents.size()
                );
            inventoryEvents.clear();
            inventoryEventCount = 0;
            inventoryPartial = false;
            inventoryEventTick = Long.MIN_VALUE;
            return result;
        }

        private NpcTraceBatch batch(int maxFrames) {
            List<NpcTraceFrame> emitted = new ArrayList<>(
                Math.min(maxFrames, frames.size())
            );
            while (emitted.size() < maxFrames && !frames.isEmpty()) {
                emitted.add(frames.removeFirst());
            }
            return new NpcTraceBatch(
                serverVersion,
                world,
                worldgenProvider,
                worldgenVersion,
                seed,
                environmentParameters,
                traceUuid,
                npcUuid,
                role,
                status,
                capacity,
                totalCaptured,
                emitted
            );
        }
    }
}
