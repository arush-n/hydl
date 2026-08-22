package com.hytalerlbridge.nativebackend.policy.combat;

import static com.hytalerlbridge.nativebackend.support.InteractionSupport.interactionManager;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.sameEntity;

import com.hypixel.hytale.component.CommandBuffer;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.spatial.SpatialResource;
import com.hypixel.hytale.component.spatial.SpatialStructure;
import com.hypixel.hytale.protocol.AppliedForce;
import com.hypixel.hytale.protocol.ApplyForceState;
import com.hypixel.hytale.protocol.ChangeVelocityType;
import com.hypixel.hytale.protocol.Direction;
import com.hypixel.hytale.protocol.FloatRange;
import com.hypixel.hytale.protocol.InteractionState;
import com.hypixel.hytale.protocol.InteractionSyncData;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.protocol.MovementStates;
import com.hypixel.hytale.protocol.Position;
import com.hypixel.hytale.protocol.SelectedHitEntity;
import com.hypixel.hytale.protocol.WaitForDataFrom;
import com.hypixel.hytale.builtin.deployables.interaction.SpawnDeployableFromRaycastInteraction;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.entity.InteractionEntry;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.entity.movement.MovementStatesComponent;
import com.hypixel.hytale.server.core.modules.entity.EntityModule;
import com.hypixel.hytale.server.core.modules.entity.component.HeadRotation;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.entity.tracker.NetworkId;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.Interaction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.client.ApplyForceInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.client.ChainingInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.client.ChargingInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.none.SelectInteraction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.selector.Selector;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.selector.SelectorType;
import com.hypixel.hytale.server.core.modules.interaction.interaction.operation.Operation;
import com.hypixel.hytale.server.core.modules.physics.component.Velocity;
import com.hypixel.hytale.server.core.modules.projectile.interaction.ProjectileInteraction;
import com.hypixel.hytale.server.core.modules.splitvelocity.VelocityConfig;
import com.hypixel.hytale.server.core.util.PositionUtil;
import com.hypixel.hytale.server.core.util.TargetUtil;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.nativebackend.NativeChargeSelection;
import com.hytalerlbridge.nativebackend.policy.combat.raycast.NativePointRaycast;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.joml.Vector3d;
import org.joml.Vector3f;
import org.joml.Vector4d;

/**
 * Caller-owned synthetic client state for explicitly controlled combat roots.
 *
 * <p>This is the single implementation used by native sessions and autonomous
 * policy actors. It supplies client-wait rows before Hytale's interaction tick;
 * it does not retain actors globally and it never decides which root to run.
 * The caller owns one instance per actor and the engine-owned
 * {@link InteractionChain} remains the lifecycle authority.</p>
 */
public final class NativeSyntheticCombatClient {

    private static final SelectedHitEntity[] NO_SELECTED_HITS =
        new SelectedHitEntity[0];
    private static final Field SELECT_INTERACTION_SELECTOR =
        selectInteractionSelectorField();

    private final Map<InteractionChain, Cursor> cursors = new HashMap<>();
    private long bindCount;
    private long forkBindCount;
    private long syncCount;
    private long npcControllerForceCount;
    private long queuedForceCount;
    private long selectionAttemptCount;
    private long selectionHitCount;
    private long selectionEmptyCount;
    private long selectionUnavailableCount;
    private long selectionPrematureServerTerminalCount;
    private float maximumSelectionRunTimeSeconds;
    private float maximumSelectionElapsedSeconds;
    private String lastSelectionUnavailableReason = "";
    private String lastSelectionClientState = "";
    private String lastSelectionServerState = "";
    private float lastSelectionAttemptChainTimeShiftSeconds = -1.0f;
    private int lastSelectionAttemptOperationCounter = -1;
    private int lastSelectionAttemptOperationIndex = -1;
    private String lastSelectionAttemptRootId = "";
    private String lastSelectionAttemptEntryServerState = "";
    private float lastSelectionAttemptEntryServerProgressSeconds = -1.0f;
    private String lastSelectionAttemptPublishedClientState = "";
    private float lastSelectionAttemptPublishedClientProgressSeconds = -1.0f;
    private int lastSelectionTerminalOperationCounter = -1;
    private int lastSelectionTerminalOperationIndex = -1;
    private String lastSelectionTerminalRootId = "";
    private String lastSelectionTerminalChainClientState = "";
    private String lastSelectionTerminalEntryServerState = "";
    private float lastSelectionTerminalEntryServerProgressSeconds = -1.0f;
    private String lastSelectionTerminalEntryClientState = "";
    private float lastSelectionTerminalEntryClientProgressSeconds = -1.0f;
    private String lastSelectionTerminalTrackedChains = "";

    /** Lease remote-client simulation for one explicitly controlled chain. */
    public void bind(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        InteractionChain chain,
        double requestedChargeSeconds
    ) {
        bind(actor, store, chain, requestedChargeSeconds, null);
    }

    /**
     * Lease remote-client simulation with the caller's selected combat target.
     *
     * <p>The target is interaction input, not bridge-owned target memory. It is
     * retained only with this engine-owned chain so a Player-backed
     * {@link SelectInteraction} can receive the same client hit row a native
     * client would publish. No weapon or selector family is interpreted here.
     * </p>
     */
    public void bind(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        InteractionChain chain,
        double requestedChargeSeconds,
        Ref<EntityStore> selectedTarget
    ) {
        if (chain == null) return;
        InteractionManager manager = interactionManager(actor, store);
        if (manager == null) {
            throw new IllegalStateException(
                "Synthetic combat client has no InteractionManager"
            );
        }
        manager.setHasRemoteClient(true);
        Cursor previous = cursors.put(
            chain,
            new Cursor(requestedChargeSeconds, selectedTarget)
        );
        if (previous == null) bindCount++;
    }

    /** Deliver the release edge for a continuation-held interaction. */
    public boolean release(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        InteractionChain chain
    ) {
        if (chain == null) return false;
        InteractionManager manager = interactionManager(actor, store);
        if (manager == null || !NativeChargeSelection.apply(
            manager.getInteractionSimulationHandler(),
            0.0
        )) {
            return false;
        }
        Cursor cursor = cursors.get(chain);
        if (cursor == null) return false;
        cursor.release();
        return true;
    }

    /** Request one authored input fork without replacing the held parent. */
    public boolean requestFork(
        InteractionChain parent,
        InteractionType forkType
    ) {
        if (parent == null || forkType == null) return false;
        Cursor cursor = cursors.get(parent);
        if (cursor == null) return false;
        cursor.requestFork(forkType);
        return true;
    }

    public boolean contains(InteractionChain chain) {
        return chain != null && cursors.containsKey(chain);
    }

    /** Selected interaction input retained with one engine-owned chain. */
    Ref<EntityStore> selectedTarget(InteractionChain chain) {
        Cursor cursor = chain == null ? null : cursors.get(chain);
        return cursor == null ? null : cursor.selectedTarget;
    }

    public boolean isEmpty() {
        return cursors.isEmpty();
    }

    public int activeCount() {
        return cursors.size();
    }

    public long bindCount() {
        return bindCount;
    }

    public long forkBindCount() {
        return forkBindCount;
    }

    public long syncCount() {
        return syncCount;
    }

    public long npcControllerForceCount() {
        return npcControllerForceCount;
    }

    public long queuedForceCount() {
        return queuedForceCount;
    }

    public long selectionAttemptCount() {
        return selectionAttemptCount;
    }

    public long selectionHitCount() {
        return selectionHitCount;
    }

    public long selectionEmptyCount() {
        return selectionEmptyCount;
    }

    public long selectionUnavailableCount() {
        return selectionUnavailableCount;
    }

    public long selectionPrematureServerTerminalCount() {
        return selectionPrematureServerTerminalCount;
    }

    public float maximumSelectionRunTimeSeconds() {
        return maximumSelectionRunTimeSeconds;
    }

    public float maximumSelectionElapsedSeconds() {
        return maximumSelectionElapsedSeconds;
    }

    public String lastSelectionUnavailableReason() {
        return lastSelectionUnavailableReason;
    }

    public String lastSelectionClientState() {
        return lastSelectionClientState;
    }

    public String lastSelectionServerState() {
        return lastSelectionServerState;
    }

    public float lastSelectionAttemptChainTimeShiftSeconds() {
        return lastSelectionAttemptChainTimeShiftSeconds;
    }

    public int lastSelectionAttemptOperationCounter() {
        return lastSelectionAttemptOperationCounter;
    }

    public int lastSelectionAttemptOperationIndex() {
        return lastSelectionAttemptOperationIndex;
    }

    public String lastSelectionAttemptRootId() {
        return lastSelectionAttemptRootId;
    }

    public String lastSelectionAttemptEntryServerState() {
        return lastSelectionAttemptEntryServerState;
    }

    public float lastSelectionAttemptEntryServerProgressSeconds() {
        return lastSelectionAttemptEntryServerProgressSeconds;
    }

    public String lastSelectionAttemptPublishedClientState() {
        return lastSelectionAttemptPublishedClientState;
    }

    public float lastSelectionAttemptPublishedClientProgressSeconds() {
        return lastSelectionAttemptPublishedClientProgressSeconds;
    }

    public int lastSelectionTerminalOperationCounter() {
        return lastSelectionTerminalOperationCounter;
    }

    public int lastSelectionTerminalOperationIndex() {
        return lastSelectionTerminalOperationIndex;
    }

    public String lastSelectionTerminalRootId() {
        return lastSelectionTerminalRootId;
    }

    public String lastSelectionTerminalChainClientState() {
        return lastSelectionTerminalChainClientState;
    }

    public String lastSelectionTerminalEntryServerState() {
        return lastSelectionTerminalEntryServerState;
    }

    public float lastSelectionTerminalEntryServerProgressSeconds() {
        return lastSelectionTerminalEntryServerProgressSeconds;
    }

    public String lastSelectionTerminalEntryClientState() {
        return lastSelectionTerminalEntryClientState;
    }

    public float lastSelectionTerminalEntryClientProgressSeconds() {
        return lastSelectionTerminalEntryClientProgressSeconds;
    }

    public String lastSelectionTerminalTrackedChains() {
        return lastSelectionTerminalTrackedChains;
    }

    /**
     * Validate a caller-owned charge duration.
     *
     * <p>Ordinary explicit charge requests must fit the float wire row. The
     * positive-infinity sentinel is also valid: Hytale's NPC interaction
     * handler uses it for an indefinitely held Wielding interaction, and the
     * synthetic cursor keeps that interaction unfinished until
     * {@link #release(Ref, Store, InteractionChain)} supplies the release edge.
     * Wire publication already clamps the sentinel to {@link Float#MAX_VALUE}.
     * </p>
     */
    static boolean validRequestedChargeSeconds(double value) {
        return value == Double.POSITIVE_INFINITY
            || (Double.isFinite(value)
                && value >= 0.0
                && value <= Float.MAX_VALUE);
    }

    /** Clear caller-owned state at an episode/plugin lifecycle boundary. */
    public void reset() {
        cursors.clear();
        bindCount = 0L;
        forkBindCount = 0L;
        syncCount = 0L;
        npcControllerForceCount = 0L;
        queuedForceCount = 0L;
        selectionAttemptCount = 0L;
        selectionHitCount = 0L;
        selectionEmptyCount = 0L;
        selectionUnavailableCount = 0L;
        selectionPrematureServerTerminalCount = 0L;
        maximumSelectionRunTimeSeconds = 0.0f;
        maximumSelectionElapsedSeconds = 0.0f;
        lastSelectionUnavailableReason = "";
        lastSelectionClientState = "";
        lastSelectionServerState = "";
        lastSelectionAttemptChainTimeShiftSeconds = -1.0f;
        lastSelectionAttemptOperationCounter = -1;
        lastSelectionAttemptOperationIndex = -1;
        lastSelectionAttemptRootId = "";
        lastSelectionAttemptEntryServerState = "";
        lastSelectionAttemptEntryServerProgressSeconds = -1.0f;
        lastSelectionAttemptPublishedClientState = "";
        lastSelectionAttemptPublishedClientProgressSeconds = -1.0f;
        lastSelectionTerminalOperationCounter = -1;
        lastSelectionTerminalOperationIndex = -1;
        lastSelectionTerminalRootId = "";
        lastSelectionTerminalChainClientState = "";
        lastSelectionTerminalEntryServerState = "";
        lastSelectionTerminalEntryServerProgressSeconds = -1.0f;
        lastSelectionTerminalEntryClientState = "";
        lastSelectionTerminalEntryClientProgressSeconds = -1.0f;
        lastSelectionTerminalTrackedChains = "";
    }

    /**
     * Attach client state to newly-created engine fork descendants.
     *
     * <p>Parallel and selector interactions create child
     * {@link InteractionChain}s after the admitted root is already bound.
     * Remote-client mode applies to the actor's whole interaction manager, so
     * every descendant that reaches a client-wait operation needs an
     * independent operation cursor. A child inherits only caller-owned input
     * (requested charge and selected target); its elapsed time, operation
     * index, selector instance, and requested fork counts start empty.</p>
     */
    private void discoverForkCursors() {
        Map<InteractionChain, Cursor> discovered = new IdentityHashMap<>();
        Set<InteractionChain> visited = java.util.Collections.newSetFromMap(
            new IdentityHashMap<>()
        );
        for (Map.Entry<InteractionChain, Cursor> tracked : List.copyOf(
            cursors.entrySet()
        )) {
            discoverForkCursors(
                tracked.getKey(),
                tracked.getValue(),
                visited,
                discovered
            );
        }
        for (Map.Entry<InteractionChain, Cursor> child : discovered.entrySet()) {
            if (cursors.putIfAbsent(child.getKey(), child.getValue()) == null) {
                forkBindCount++;
            }
        }
    }

    private void discoverForkCursors(
        InteractionChain parent,
        Cursor inherited,
        Set<InteractionChain> visited,
        Map<InteractionChain, Cursor> discovered
    ) {
        if (parent == null || !visited.add(parent)) return;
        for (InteractionChain child : parent.getForkedChains().values()) {
            Cursor cursor = cursors.get(child);
            if (cursor == null) {
                cursor = discovered.computeIfAbsent(
                    child,
                    ignored -> inherited.forkChild()
                );
            }
            discoverForkCursors(child, cursor, visited, discovered);
        }
    }

    /**
     * Supply client state before {@code TickInteractionManagerSystem}.
     *
     * @param externalRemoteClientActive true when another exact bridge-owned
     *     context (currently the headless World-verb Player) still owns remote
     *     simulation after the last combat cursor finishes
     */
    public void prepare(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        float deltaTime,
        boolean externalRemoteClientActive
    ) {
        if (actor == null || !actor.isValid() || store == null
            || cursors.isEmpty()) {
            return;
        }
        discoverForkCursors();
        double interactionTickSeconds = interactionTickSeconds(
            store.getExternalData().getWorld().getTickStepNanos()
        );
        boolean removed = false;
        InteractionManager manager = interactionManager(actor, store);
        if (manager == null) {
            throw new IllegalStateException(
                "Synthetic combat client has no InteractionManager"
            );
        }
        var iterator = cursors.entrySet().iterator();
        while (iterator.hasNext()) {
            var tracked = iterator.next();
            InteractionChain chain = tracked.getKey();
            Cursor cursor = tracked.getValue();
            InteractionState serverState = chain.getServerState();
            if (serverState != InteractionState.NotFinished) {
                if (cursor.selectionAttempted
                    && !cursor.selectionServerTerminalObserved) {
                    cursor.selectionServerTerminalObserved = true;
                    lastSelectionServerState = serverState.name();
                    recordSelectionTerminal(chain);
                    if (cursor.lastSelectionClientState
                        == InteractionState.NotFinished) {
                        selectionPrematureServerTerminalCount++;
                    }
                }
                /*
                 * Remote-client chains are not removed by InteractionManager
                 * until both sides publish a terminal state.  This synthetic
                 * client used to drop its cursor as soon as the server
                 * finished, leaving clientState=NotFinished forever.  The
                 * manager then waited for a client acknowledgement that the
                 * headless actor could no longer send, so lifecycle telemetry
                 * never observed completion and CombatSupport stayed busy.
                 */
                if (!terminalCursorMayReleaseRemoteClient(
                    serverState,
                    registeredWithManager(manager, chain)
                )) {
                    chain.setClientState(clientTerminalState(serverState));
                    syncCount++;
                    continue;
                }
                iterator.remove();
                removed = true;
                continue;
            }
            RootInteraction root = chain.getRootInteraction();
            int operationCounter = chain.getOperationCounter();
            if (root == null || operationCounter < 0
                || operationCounter >= root.getOperationMax()) {
                continue;
            }
            Operation operation = root.getOperation(operationCounter);
            if (operation == null
                || operation.getWaitForDataFrom() != WaitForDataFrom.Client) {
                continue;
            }
            int operationIndex = chain.getOperationIndex();
            boolean operationChanged = operationIndex != cursor.operationIndex;
            if (operationChanged) {
                cursor.operationIndex = operationIndex;
                cursor.elapsedSeconds = 0.0;
            }
            Operation inner = operation.getInnerOperation();
            SelectInteraction select = inner instanceof SelectInteraction value
                ? value
                : null;
            boolean charging = inner instanceof ChargingInteraction;
            com.hypixel.hytale.protocol.ApplyForceInteraction applyForce =
                inner instanceof ApplyForceInteraction interaction
                    ? (com.hypixel.hytale.protocol.ApplyForceInteraction)
                        interaction.toPacket()
                    : null;
            com.hypixel.hytale.protocol.SpawnDeployableFromRaycastInteraction
                deployable =
                    inner instanceof SpawnDeployableFromRaycastInteraction interaction
                        ? (com.hypixel.hytale.protocol.SpawnDeployableFromRaycastInteraction)
                            interaction.toPacket()
                        : null;
            float runTime = inner instanceof Interaction interaction
                ? Math.max(0.0f, interaction.getRunTime())
                : 0.0f;
            double finishSeconds = charging
                ? cursor.requestedChargeSeconds
                : applyForce == null
                    ? runTime
                    : applyForceFinishSeconds(runTime, applyForce.duration);
            float wireElapsedSeconds = (float) Math.min(
                Float.MAX_VALUE,
                cursor.elapsedSeconds
            );
            boolean temporalFinished = wireElapsedSeconds >= (float) Math.min(
                Float.MAX_VALUE,
                finishSeconds
            );
            ApplyForceState applyForceState = applyForce == null
                ? null
                : applyForceCompletionState(
                    operationChanged,
                    wireElapsedSeconds,
                    runTime,
                    applyForce,
                    actor,
                    store
                );
            boolean finished = applyForceState == null
                ? temporalFinished
                : applyForceState != ApplyForceState.Waiting;
            if (applyForce != null && applyForceRunsThisTick(
                operationChanged,
                cursor.elapsedSeconds,
                applyForce.duration
            )) {
                applyClientForces(actor, store, applyForce);
            }
            NativePointRaycast.Result deployableRay = deployable == null
                ? null
                : NativePointRaycast.traceActorLook(
                    actor,
                    store,
                    deployable.maxDistance
                );

            InteractionSyncData data = new InteractionSyncData();
            data.state = deployableRay != null && !deployableRay.available()
                ? InteractionState.Failed
                : finished
                    ? InteractionState.Finished
                    : InteractionState.NotFinished;
            data.progress = wireElapsedSeconds;
            data.operationCounter = operationCounter;
            data.rootInteraction =
                RootInteraction.getRootInteractionIdOrUnknown(root.getId());
            data.forkCounts = cursor.forkCounts();
            data.totalForks = cursor.totalForks();
            if (publishesActorLook(
                inner == null ? null : inner.getClass(),
                operationChanged
            )) {
                publishActorLook(data, actor, store);
            }
            if (select != null) {
                selectionAttemptCount++;
                maximumSelectionRunTimeSeconds = Math.max(
                    maximumSelectionRunTimeSeconds,
                    runTime
                );
                maximumSelectionElapsedSeconds = Math.max(
                    maximumSelectionElapsedSeconds,
                    wireElapsedSeconds
                );
                ClientSelection selection = cursor.selectTarget(
                    select,
                    actor,
                    store,
                    wireElapsedSeconds,
                    runTime,
                    operationChanged
                );
                if (!selection.available()) {
                    selectionUnavailableCount++;
                    lastSelectionUnavailableReason =
                        selection.unavailableReason();
                    data.state = InteractionState.Failed;
                } else {
                    if (selection.hits().length == 0) {
                        selectionEmptyCount++;
                    } else {
                        selectionHitCount += selection.hits().length;
                    }
                    data.hitEntities = selection.hits();
                    data.attackerPos = selection.attackerPosition();
                    data.attackerRot = selection.attackerRotation();
                }
                cursor.selectionAttempted = true;
                cursor.lastSelectionClientState = data.state;
                lastSelectionClientState = data.state.name();
                recordSelectionAttempt(chain, data);
            }
            if (charging) {
                data.chargeValue = finished
                    ? (float) Math.min(
                        Float.MAX_VALUE,
                        cursor.requestedChargeSeconds
                    )
                    : -1.0f;
            }
            if (applyForceState != null) {
                data.applyForceState = applyForceState;
            }
            if (deployableRay != null) {
                HeadRotation headRotation = store.getComponent(
                    actor,
                    HeadRotation.getComponentType()
                );
                if (headRotation == null) {
                    data.state = InteractionState.Failed;
                } else {
                    var rotation = headRotation.getRotation();
                    data.attackerRot = new Direction(
                        rotation.yaw(),
                        rotation.pitch(),
                        rotation.roll()
                    );
                }
                data.raycastDistance = deployableRay.distance();
                if (deployableRay.available() && deployableRay.hit()) {
                    data.raycastHit = new Position(
                        deployableRay.hitX(),
                        deployableRay.hitY(),
                        deployableRay.hitZ()
                    );
                    data.raycastNormal = new Vector3f(
                        deployableRay.normalX(),
                        deployableRay.normalY(),
                        deployableRay.normalZ()
                    );
                }
            }
            if (inner instanceof ChainingInteraction) data.chainingIndex = 0;

            InteractionEntry entry = chain.getInteraction(operationIndex);
            if (entry == null) {
                chain.putInteractionSyncData(operationIndex, data);
            } else if (!entry.setClientState(data)) {
                throw new IllegalStateException(
                    "Synthetic combat client interaction sync desynchronized"
                );
            }
            syncCount++;
            if (data.state == InteractionState.Finished
                && chain.getCallDepth() == 0
                && operationCounter + 1 >= root.getOperationMax()) {
                chain.setClientState(InteractionState.Finished);
            }
            cursor.elapsedSeconds += interactionTickSeconds;
        }
        if (removed) restoreNpcSimulationIfIdle(
            actor,
            store,
            externalRemoteClientActive
        );
    }

    /**
     * Convert the server's authoritative interaction tick step to seconds.
     *
     * <p>{@code InteractionManager} advances entry timestamps by
     * {@code World.getTickStepNanos()}, not by the measured {@code deltaTime}
     * supplied to entity systems.  The synthetic client must use the same
     * clock or a fixed requested charge releases on different ticks as server
     * load changes.</p>
     */
    static double interactionTickSeconds(long tickStepNanos) {
        if (tickStepNanos <= 0L) {
            throw new IllegalArgumentException(
                "tickStepNanos must be positive"
            );
        }
        return tickStepNanos / 1_000_000_000.0;
    }

    /**
     * Publish the same eye-origin pose as a native client-side interaction.
     *
     * <p>This is selected by the compiled operation type, not an item, root,
     * or projectile-config ID. Dynamic replacement roots and future ranged
     * weapon families therefore receive the same lifecycle behavior without
     * adding weapon branches.</p>
     */
    private static void publishActorLook(
        InteractionSyncData data,
        Ref<EntityStore> actor,
        Store<EntityStore> store
    ) {
        var look = TargetUtil.getLook(actor, store);
        data.attackerPos = PositionUtil.toPositionPacket(look.getPosition());
        data.attackerRot = PositionUtil.toDirectionPacket(look.getRotation());
    }

    /** Select pose publication by lifecycle operation, never weapon identity. */
    static boolean publishesActorLook(
        Class<?> operationType,
        boolean operationChanged
    ) {
        return operationChanged
            && operationType != null
            && ProjectileInteraction.class.isAssignableFrom(operationType);
    }

    /** Timer completion waits for both authored run time and force duration. */
    static double applyForceFinishSeconds(float runTime, float duration) {
        return Math.max(Math.max(0.0f, runTime), Math.max(0.0f, duration));
    }

    /** Reproduce the public server simulation's ApplyForce completion probes. */
    private static ApplyForceState applyForceCompletionState(
        boolean firstRun,
        float elapsedSeconds,
        float runTime,
        com.hypixel.hytale.protocol.ApplyForceInteraction interaction,
        Ref<EntityStore> actor,
        Store<EntityStore> store
    ) {
        boolean durationFinished = !firstRun
            && (!(interaction.duration > 0.0f)
                || !(elapsedSeconds < interaction.duration));
        boolean groundEligible = durationFinished
            && interaction.waitForGround
            && elapsedSeconds >= interaction.groundCheckDelay;
        boolean collisionEligible = durationFinished
            && interaction.waitForCollision
            && elapsedSeconds >= interaction.collisionCheckDelay;
        boolean grounded = groundEligible && applyForceGrounded(actor, store);
        boolean collided = collisionEligible && applyForceCollided(actor, store);
        return applyForceCompletionState(
            firstRun,
            elapsedSeconds,
            runTime,
            interaction.duration,
            interaction.waitForGround,
            interaction.waitForCollision,
            interaction.groundCheckDelay,
            interaction.collisionCheckDelay,
            grounded,
            collided
        );
    }

    /**
     * Pure form of {@code ApplyForceInteraction.simulateTick0}'s branch order.
     *
     * <p>Ground wins over Collision, which wins over Timer. A conditional row
     * with no positive run time remains Waiting until its condition is
     * observed; elapsed time alone never fabricates the acknowledgement.</p>
     */
    static ApplyForceState applyForceCompletionState(
        boolean firstRun,
        float elapsedSeconds,
        float runTime,
        float duration,
        boolean waitForGround,
        boolean waitForCollision,
        float groundCheckDelay,
        float collisionCheckDelay,
        boolean grounded,
        boolean collided
    ) {
        if (firstRun || (duration > 0.0f && elapsedSeconds < duration)) {
            return ApplyForceState.Waiting;
        }
        if (waitForGround
            && elapsedSeconds >= groundCheckDelay
            && grounded) {
            return ApplyForceState.Ground;
        }
        if (waitForCollision
            && elapsedSeconds >= collisionCheckDelay
            && collided) {
            return ApplyForceState.Collision;
        }
        boolean instantlyComplete = runTime <= 0.0f
            && !waitForGround
            && !waitForCollision;
        boolean timerFinished = instantlyComplete
            || (runTime > 0.0f && elapsedSeconds >= runTime);
        return timerFinished
            ? ApplyForceState.Timer
            : ApplyForceState.Waiting;
    }

    /** Exact ground predicate used by native ApplyForce simulation. */
    private static boolean applyForceGrounded(
        Ref<EntityStore> actor,
        Store<EntityStore> store
    ) {
        MovementStatesComponent component = store.getComponent(
            actor,
            MovementStatesComponent.getComponentType()
        );
        if (component == null) {
            throw new IllegalStateException(
                "Synthetic ApplyForce requires MovementStatesComponent"
            );
        }
        MovementStates states = component.getMovementStates();
        return states.onGround || states.inFluid || states.climbing;
    }

    /** Exact 1.5-block entity-overlap predicate used by native simulation. */
    private static boolean applyForceCollided(
        Ref<EntityStore> actor,
        Store<EntityStore> store
    ) {
        TransformComponent transform = store.getComponent(
            actor,
            TransformComponent.getComponentType()
        );
        if (transform == null) {
            throw new IllegalStateException(
                "Synthetic ApplyForce requires TransformComponent"
            );
        }
        SpatialResource<Ref<EntityStore>, EntityStore> resource =
            store.getResource(
                EntityModule.get().getNetworkSendableSpatialResourceType()
            );
        SpatialStructure<Ref<EntityStore>> spatial =
            resource.getSpatialStructure();
        List<Ref<EntityStore>> entities = SpatialResource
            .getThreadLocalReferenceList();
        spatial.collect(transform.getPosition(), 1.5, entities);
        return entities.size() > 1;
    }

    /** Apply an instantaneous force once, or a duration force on every tick. */
    static boolean applyForceRunsThisTick(
        boolean operationChanged,
        double elapsedSeconds,
        float duration
    ) {
        return operationChanged
            || (duration > 0.0f && (float) elapsedSeconds < duration);
    }

    /** Mirror ApplyForceInteraction.simulateTick0 on the headless actor. */
    private void applyClientForces(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        com.hypixel.hytale.protocol.ApplyForceInteraction interaction
    ) {
        HeadRotation head = store.getComponent(
            actor,
            HeadRotation.getComponentType()
        );
        Velocity velocity = store.getComponent(
            actor,
            Velocity.getComponentType()
        );
        if (head == null || velocity == null) {
            throw new IllegalStateException(
                "Synthetic ApplyForce requires HeadRotation and Velocity"
            );
        }
        AppliedForce[] forces = interaction.forces;
        if (forces == null || forces.length == 0) return;

        float headPitch = head.getRotation().x;
        float headYaw = head.getRotation().y;
        ChangeVelocityType velocityType = interaction.changeVelocityType;
        VelocityConfig velocityConfig = velocityConfig(interaction.velocityConfig);
        NPCEntity npc = store.getComponent(actor, NPCEntity.getComponentType());
        var role = npc == null ? null : npc.getRole();
        for (AppliedForce force : forces) {
            Vector3d direction = clientForceVector(
                force,
                headPitch,
                headYaw,
                interaction.verticalClamp
            );
            if (role != null) {
                /*
                 * The bridge's opt-in Adventure context is deliberately a
                 * hybrid: the controlled entity retains NPCEntity while a
                 * PlayerRef is attached for player-only interaction rules.
                 * Both Hytale velocity-instruction systems match that
                 * archetype, and PlayerVelocityInstructionSystem consumes
                 * the shared queue into the headless packet sink before the
                 * NPC controller can apply it. Route the synthetic client's
                 * local force through the same Role methods used by
                 * NPCVelocityInstructionSystem instead. This is selected by
                 * actor capability, never by item, profile, or force scalar.
                 */
                if (velocityType == ChangeVelocityType.Set) {
                    role.processSetVelocityInstruction(direction, velocityConfig);
                } else {
                    role.processAddVelocityInstruction(direction, velocityConfig);
                }
                npcControllerForceCount++;
            } else {
                velocity.addInstruction(direction, velocityConfig, velocityType);
                queuedForceCount++;
            }
            velocityType = ChangeVelocityType.Add;
        }
    }

    /** Exact packet-space force transform used by the Hytale client path. */
    static Vector3d clientForceVector(
        AppliedForce force,
        float headPitch,
        float headYaw,
        FloatRange verticalClamp
    ) {
        Vector3d direction = new Vector3d(
            force.direction.x(),
            force.direction.y(),
            force.direction.z()
        );
        if (force.adjustVertical) {
            float pitch = headPitch;
            if (verticalClamp != null) {
                pitch = Math.max(
                    verticalClamp.inclusiveMin,
                    Math.min(verticalClamp.inclusiveMax, pitch)
                );
            }
            direction.rotateX(pitch);
        }
        return direction.mul(force.force).rotateY(headYaw);
    }

    private static VelocityConfig velocityConfig(
        com.hypixel.hytale.protocol.VelocityConfig packet
    ) {
        if (packet == null) return null;
        VelocityConfig value = new VelocityConfig();
        value.setGroundResistance(packet.groundResistance);
        value.setGroundResistanceMax(packet.groundResistanceMax);
        value.setAirResistance(packet.airResistance);
        value.setAirResistanceMax(packet.airResistanceMax);
        value.setThreshold(packet.threshold);
        value.setStyle(packet.style);
        return value;
    }

    /** Build the exact packet row consumed by ClientSourcedSelector. */
    static SelectedHitEntity selectedHitEntity(
        int networkId,
        Vector4d hit,
        Position targetPosition,
        Direction targetBodyRotation
    ) {
        if (hit == null || targetPosition == null || targetBodyRotation == null) {
            throw new IllegalArgumentException(
                "Synthetic SelectInteraction snapshots must be complete"
            );
        }
        return new SelectedHitEntity(
            networkId,
            new Vector3f((float) hit.x, (float) hit.y, (float) hit.z),
            targetPosition,
            targetBodyRotation
        );
    }

    private static ClientSelection clientSelection(
        Selector selector,
        Ref<EntityStore> actor,
        Ref<EntityStore> selectedTarget,
        Store<EntityStore> store,
        float elapsedSeconds,
        float runTime
    ) {
        if (selector == null) return ClientSelection.unavailable("selector_missing");
        if (actor == null) return ClientSelection.unavailable("actor_missing");
        if (!actor.isValid()) return ClientSelection.unavailable("actor_invalid");
        if (selectedTarget == null) {
            return ClientSelection.unavailable("selected_target_missing");
        }
        if (!selectedTarget.isValid()) {
            return ClientSelection.unavailable("selected_target_invalid");
        }
        if (!sameEntityStore(actor, selectedTarget)) {
            return ClientSelection.unavailable("selected_target_cross_store");
        }
        if (store == null) return ClientSelection.unavailable("store_missing");
        TransformComponent attackerTransform = store.getComponent(
            actor,
            TransformComponent.getComponentType()
        );
        HeadRotation attackerHead = store.getComponent(
            actor,
            HeadRotation.getComponentType()
        );
        TransformComponent targetTransform = store.getComponent(
            selectedTarget,
            TransformComponent.getComponentType()
        );
        NetworkId targetNetworkId = store.getComponent(
            selectedTarget,
            NetworkId.getComponentType()
        );
        if (attackerTransform == null) {
            return ClientSelection.unavailable("attacker_transform_missing");
        }
        if (attackerHead == null) {
            return ClientSelection.unavailable("attacker_head_rotation_missing");
        }
        if (targetTransform == null) {
            return ClientSelection.unavailable("selected_target_transform_missing");
        }
        if (targetNetworkId == null) {
            return ClientSelection.unavailable("selected_target_network_id_missing");
        }

        ReadOnlyCommandBuffer commandBuffer = new ReadOnlyCommandBuffer(store);
        float selectorTime = Math.min(
            Math.max(0.0f, elapsedSeconds),
            Math.max(0.0f, runTime)
        );
        selector.tick(commandBuffer, actor, selectorTime, runTime);
        Vector4d[] selectedHit = new Vector4d[1];
        selector.selectTargetEntities(
            commandBuffer,
            actor,
            (candidate, hit) -> {
                if (selectedHit[0] == null
                    && sameEntity(candidate, selectedTarget)) {
                    selectedHit[0] = new Vector4d(hit);
                }
            },
            candidate -> sameEntity(candidate, selectedTarget)
        );

        SelectedHitEntity[] hits = NO_SELECTED_HITS;
        if (selectedHit[0] != null) {
            hits = new SelectedHitEntity[] { selectedHitEntity(
                targetNetworkId.getId(),
                selectedHit[0],
                PositionUtil.toPositionPacket(targetTransform.getPosition()),
                PositionUtil.toDirectionPacket(targetTransform.getRotation())
            ) };
        }
        return new ClientSelection(
            true,
            hits,
            PositionUtil.toPositionPacket(attackerTransform.getPosition()),
            PositionUtil.toDirectionPacket(attackerHead.getRotation()),
            ""
        );
    }

    private static boolean sameEntityStore(
        Ref<EntityStore> left,
        Ref<EntityStore> right
    ) {
        return left.getStore() == right.getStore();
    }

    private static Selector newServerSelector(SelectInteraction interaction) {
        try {
            SelectorType type = (SelectorType) SELECT_INTERACTION_SELECTOR.get(
                interaction
            );
            if (type == null) {
                throw new IllegalStateException(
                    "SelectInteraction has no authored selector"
                );
            }
            return type.newSelector();
        } catch (IllegalAccessException failure) {
            throw new IllegalStateException(
                "Cannot read SelectInteraction selector for synthetic client",
                failure
            );
        }
    }

    private static Field selectInteractionSelectorField() {
        try {
            Field field = SelectInteraction.class.getDeclaredField("selector");
            field.setAccessible(true);
            return field;
        } catch (ReflectiveOperationException | RuntimeException failure) {
            throw new ExceptionInInitializerError(failure);
        }
    }

    /** Restore ordinary NPC simulation only after every lease is gone. */
    public void restoreNpcSimulationIfIdle(
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        boolean externalRemoteClientActive
    ) {
        if (externalRemoteClientActive || !cursors.isEmpty()) return;
        InteractionManager manager = interactionManager(actor, store);
        if (manager != null) manager.setHasRemoteClient(false);
    }

    /** Capture one read-only client publication without changing the chain. */
    private void recordSelectionAttempt(
        InteractionChain chain,
        InteractionSyncData publishedClientState
    ) {
        lastSelectionAttemptChainTimeShiftSeconds = chain.getTimeShift();
        lastSelectionAttemptOperationCounter = chain.getOperationCounter();
        lastSelectionAttemptOperationIndex = chain.getOperationIndex();
        lastSelectionAttemptRootId = rootId(chain);
        InteractionEntry entry = currentEntryOrNull(chain);
        InteractionSyncData server = entry == null
            ? null
            : entry.getServerState();
        lastSelectionAttemptEntryServerState = syncStateName(server);
        lastSelectionAttemptEntryServerProgressSeconds = syncProgress(server);
        lastSelectionAttemptPublishedClientState = syncStateName(
            publishedClientState
        );
        lastSelectionAttemptPublishedClientProgressSeconds = syncProgress(
            publishedClientState
        );
    }

    /** Capture why the engine declared a client-owned selector terminal. */
    private void recordSelectionTerminal(InteractionChain chain) {
        lastSelectionTerminalOperationCounter = chain.getOperationCounter();
        lastSelectionTerminalOperationIndex = chain.getOperationIndex();
        lastSelectionTerminalRootId = rootId(chain);
        InteractionState chainClientState = chain.getClientState();
        lastSelectionTerminalChainClientState = chainClientState == null
            ? ""
            : chainClientState.name();
        InteractionEntry entry = currentEntryOrNull(chain);
        InteractionSyncData server = entry == null
            ? null
            : entry.getServerState();
        InteractionSyncData client = entry == null
            ? null
            : entry.getClientState();
        lastSelectionTerminalEntryServerState = syncStateName(server);
        lastSelectionTerminalEntryServerProgressSeconds = syncProgress(server);
        lastSelectionTerminalEntryClientState = syncStateName(client);
        lastSelectionTerminalEntryClientProgressSeconds = syncProgress(client);
        lastSelectionTerminalTrackedChains = trackedChainSnapshot();
    }

    /** Deterministic read-only snapshot of the whole tracked fork forest. */
    private String trackedChainSnapshot() {
        List<String> rows = new ArrayList<>(cursors.size());
        for (InteractionChain tracked : cursors.keySet()) {
            rows.add(chainSnapshot(tracked));
        }
        rows.sort(String::compareTo);
        return String.join(";", rows);
    }

    private static String chainSnapshot(InteractionChain chain) {
        InteractionEntry entry = currentEntryOrNull(chain);
        InteractionSyncData server = entry == null
            ? null
            : entry.getServerState();
        InteractionSyncData client = entry == null
            ? null
            : entry.getClientState();
        RootInteraction initialRoot = chain.getInitialRootInteraction();
        String initialRootId = initialRoot == null || initialRoot.getId() == null
            ? ""
            : initialRoot.getId();
        return "root=" + rootId(chain)
            + "|initial_root=" + initialRootId
            + "|fork=" + String.valueOf(chain.getForkedChainId())
            + "|base_fork=" + String.valueOf(chain.getBaseForkedChainId())
            + "|server=" + String.valueOf(chain.getServerState())
            + "|client=" + String.valueOf(chain.getClientState())
            + "|operation_counter=" + chain.getOperationCounter()
            + "|operation_index=" + chain.getOperationIndex()
            + "|entry_server=" + syncStateName(server)
            + "|entry_server_progress=" + syncProgress(server)
            + "|entry_client=" + syncStateName(client)
            + "|entry_client_progress=" + syncProgress(client)
            + "|fork_children=" + chain.getForkedChains().size();
    }

    private static InteractionEntry currentEntryOrNull(
        InteractionChain chain
    ) {
        int operationIndex = chain.getOperationIndex();
        if (operationIndex < 0) return null;
        try {
            return chain.getInteraction(operationIndex);
        } catch (IllegalArgumentException ignored) {
            return null;
        }
    }

    private static String rootId(InteractionChain chain) {
        RootInteraction root = chain.getRootInteraction();
        return root == null || root.getId() == null ? "" : root.getId();
    }

    private static String syncStateName(InteractionSyncData data) {
        return data == null || data.state == null ? "" : data.state.name();
    }

    private static float syncProgress(InteractionSyncData data) {
        return data == null ? -1.0f : data.progress;
    }

    private static final class Cursor {
        private double requestedChargeSeconds;
        private final Ref<EntityStore> selectedTarget;
        private int operationIndex = -1;
        private double elapsedSeconds;
        private Selector selector;
        private boolean selectionAttempted;
        private boolean selectionServerTerminalObserved;
        private InteractionState lastSelectionClientState;
        private final Map<InteractionType, Integer> requestedForkCounts =
            new HashMap<>();

        private Cursor(
            double requestedChargeSeconds,
            Ref<EntityStore> selectedTarget
        ) {
            if (!validRequestedChargeSeconds(requestedChargeSeconds)) {
                throw new IllegalArgumentException(
                    "Synthetic client charge duration must be a non-negative "
                        + "float or the positive-infinity hold sentinel"
                );
            }
            this.requestedChargeSeconds = requestedChargeSeconds;
            this.selectedTarget = selectedTarget;
        }

        private ClientSelection selectTarget(
            SelectInteraction interaction,
            Ref<EntityStore> actor,
            Store<EntityStore> store,
            float elapsedSeconds,
            float runTime,
            boolean operationChanged
        ) {
            if (operationChanged || selector == null) {
                selector = newServerSelector(interaction);
            }
            return clientSelection(
                selector,
                actor,
                selectedTarget,
                store,
                elapsedSeconds,
                runTime
            );
        }

        private void release() {
            requestedChargeSeconds = 0.0;
        }

        private Cursor forkChild() {
            return new Cursor(requestedChargeSeconds, selectedTarget);
        }

        private void requestFork(InteractionType forkType) {
            requestedForkCounts.merge(forkType, 1, Integer::sum);
        }

        private Map<InteractionType, Integer> forkCounts() {
            return requestedForkCounts.isEmpty()
                ? null
                : Map.copyOf(requestedForkCounts);
        }

        private int totalForks() {
            int total = 0;
            for (int count : requestedForkCounts.values()) total += count;
            return total;
        }
    }

    private record ClientSelection(
        boolean available,
        SelectedHitEntity[] hits,
        Position attackerPosition,
        Direction attackerRotation,
        String unavailableReason
    ) {
        private static ClientSelection unavailable(String reason) {
            return new ClientSelection(false, null, null, null, reason);
        }
    }

    /** Read-only view required by Hytale's authored selector implementations. */
    private static final class ReadOnlyCommandBuffer
        extends CommandBuffer<EntityStore> {

        private ReadOnlyCommandBuffer(Store<EntityStore> store) {
            super(store);
        }
    }

    /** Mirror the authoritative server terminal for the synthetic peer. */
    static InteractionState clientTerminalState(InteractionState serverState) {
        if (serverState == null || serverState == InteractionState.NotFinished) {
            throw new IllegalArgumentException(
                "synthetic client acknowledgement requires a terminal state"
            );
        }
        return serverState;
    }

    /**
     * A terminal acknowledgement is not permission to leave remote-client
     * mode until Hytale has removed the acknowledged chain from its manager.
     */
    static boolean terminalCursorMayReleaseRemoteClient(
        InteractionState serverState,
        boolean registeredWithManager
    ) {
        return serverState != null
            && serverState != InteractionState.NotFinished
            && !registeredWithManager;
    }

    /** Include engine-owned fork descendants, not only top-level chains. */
    static boolean registeredWithManager(
        InteractionManager manager,
        InteractionChain target
    ) {
        if (manager == null || target == null) return false;
        Set<InteractionChain> visited = java.util.Collections.newSetFromMap(
            new IdentityHashMap<>()
        );
        for (InteractionChain root : manager.getChains().values()) {
            if (containsChain(root, target, visited)) return true;
        }
        return false;
    }

    private static boolean containsChain(
        InteractionChain candidate,
        InteractionChain target,
        Set<InteractionChain> visited
    ) {
        if (candidate == null || !visited.add(candidate)) return false;
        if (candidate == target) return true;
        for (InteractionChain child : candidate.getForkedChains().values()) {
            if (containsChain(child, target, visited)) return true;
        }
        return false;
    }
}
