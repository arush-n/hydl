package com.hytalerlbridge.policy.world.runtime;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.PolicyActionSink;
import com.hytalerlbridge.policy.PolicyAgentMarker;
import com.hytalerlbridge.policy.SteeringActionSink;
import com.hytalerlbridge.policy.perception.acquisition.blocks.ServerBlockCandidateReader;
import com.hytalerlbridge.policy.perception.acquisition.crafting.ServerRecipeCandidateReader;
import com.hytalerlbridge.policy.world.bridge.BridgeWorldVerbFacade;
import com.hytalerlbridge.policy.world.model.BlockCandidateBinding;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;
import com.hytalerlbridge.policy.world.model.WorldVerbBinding;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Composes continuous public steering with bridge-owned discrete World verbs.
 *
 * <p>Movement/look are independently concurrent in the published JAX action
 * contract, so they are applied even when native admission rejects a verb.
 * Discrete verbs run only on the first control tick of a decision. Recipe IDs
 * are actor-privileged and are re-read from current inventory immediately
 * before the bridge request is constructed.</p>
 */
public final class WorldVerbActionSink implements PolicyActionSink {

    private final SteeringActionSink steering;
    private final BridgeWorldVerbFacade facade;
    private final ServerRecipeCandidateReader recipes;
    private final PolicyWorldAnchorRegistry anchors;
    private final AtomicLong nextRequestId = new AtomicLong();

    /** Resource-cleanup index only; gameplay lifecycle remains on the marker. */
    private final Map<PolicyAgentMarker, ActiveActor> activeActors =
        new IdentityHashMap<>();

    private long requested;
    private long accepted;
    private long finished;
    private long failed;
    private long rejectedBinding;
    private long rejectedBusy;
    private long unsupported;
    private long heldEdgeSuppressed;
    private long contextFailures;

    public WorldVerbActionSink(
        SteeringActionSink steering,
        BridgeWorldVerbFacade facade,
        ServerRecipeCandidateReader recipes,
        PolicyWorldAnchorRegistry anchors
    ) {
        if (steering == null || facade == null || recipes == null
            || anchors == null) {
            throw new IllegalArgumentException(
                "World-verb sink requires steering, facade, recipes, and anchors");
        }
        this.steering = steering;
        this.facade = facade;
        this.recipes = recipes;
        this.anchors = anchors;
    }

    @Override
    public boolean supportsWorldVerb(ActionDecoder.Decoded action) {
        if (action == null || !action.actionLegal()) return false;
        if (action.recipeCandidateIndex() >= 0) {
            return !action.useRequested()
                && action.blockCandidateIndex() < 0;
        }
        if (action.useRequested()) {
            return action.blockCandidateIndex() >= 0;
        }
        return action.blockCandidateIndex() >= 0
            && (action.blockInteractionTrigger() == 1
                || action.blockInteractionTrigger() == 2);
    }

    /** Called by the pre-interaction ECS system. */
    public void prepare(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        float deltaTime,
        PolicyAgentMarker marker
    ) {
        PolicyWorldVerbState state = marker.worldVerb();
        BridgeWorldVerbFacade.Execution execution = state.execution();
        if (execution == null) return;
        state.observe(facade.prepare(execution, ref, store, deltaTime));
    }

    @Override
    public void apply(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        float deltaTime,
        PolicyAgentMarker marker,
        ActionDecoder.Decoded action,
        WorldActionEvidence evidence,
        boolean firstControlTick
    ) {
        pollActive(ref, store, marker);

        // The action contract makes movement/look independent of main-root
        // admission. The old PolicyControlSystem return dropped these too.
        steering.apply(
            ref, npc, store, deltaTime, marker, action, evidence,
            firstControlTick);

        if (!action.requestsWorldVerb()) return;
        requested++;
        if (!firstControlTick) {
            heldEdgeSuppressed++;
            return;
        }
        if (marker.worldVerb().active()) {
            rejectedBusy++;
            return;
        }
        if (action.recipeCandidateIndex() >= 0) {
            startFieldcraft(
                ref,
                store,
                marker,
                action.recipeCandidateIndex(),
                evidence
            );
        } else if (action.blockCandidateIndex() >= 0) {
            startBlockAction(ref, store, marker, action, evidence);
        } else {
            unsupported++;
        }
    }

    private void startBlockAction(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        PolicyAgentMarker marker,
        ActionDecoder.Decoded action,
        WorldActionEvidence observed
    ) {
        int candidateIndex = action.blockCandidateIndex();
        BlockCandidateBinding observedCandidate =
            observed.blockBinding(candidateIndex);
        WorldVerbBinding observedBinding = observedCandidate.forAction(
            action.useRequested(), action.blockInteractionTrigger());
        ServerBlockCandidateReader.Capture commit =
            ServerBlockCandidateReader.capture(ref, store);
        boolean[] commitMask = commit.candidateMask();
        BlockCandidateBinding[] commitBindings = commit.bindings();
        BlockCandidateBinding commitCandidate =
            candidateIndex < 0 || candidateIndex >= commitBindings.length
                ? BlockCandidateBinding.empty()
                : commitBindings[candidateIndex];
        WorldVerbBinding commitBinding = commitCandidate.forAction(
            action.useRequested(), action.blockInteractionTrigger());
        if (!commit.available()
            || candidateIndex < 0
            || candidateIndex >= commitMask.length
            || !commitMask[candidateIndex]
            || !observedCandidate.available()
            || !observedCandidate.equals(commitCandidate)
            || !observedBinding.available()
            || !observedBinding.equals(commitBinding)) {
            rejectedBinding++;
            return;
        }

        World world = store.getExternalData().getWorld();
        if (world == null) {
            contextFailures++;
            return;
        }
        PolicyWorldAnchorRegistry.Context context = anchors.context(world);
        BridgeWorldVerbFacade.ContextResult installed = facade.installContext(
            ref, store, context.packetAnchor());
        if (!installed.accepted()) {
            contextFailures++;
            return;
        }
        long requestId = nextRequestId.getAndIncrement();
        if (requestId < 0L) {
            facade.uninstallContext(ref, store);
            rejectedBinding++;
            return;
        }
        BridgeWorldVerbFacade.Admission admission =
            facade.startBoundWorldVerb(
                ref, store, requestId, commitBinding, context.worldEpoch());
        if (!admission.accepted()) {
            facade.uninstallContext(ref, store);
            failed++;
            return;
        }
        begin(marker, ref, store, commitBinding.verb(), admission);
    }

    private void startFieldcraft(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        PolicyAgentMarker marker,
        int candidateIndex,
        WorldActionEvidence observed
    ) {
        String observedId = observed.recipeId(candidateIndex);
        ServerRecipeCandidateReader.Capture commit = recipes.capture(ref, store);
        boolean[] commitMask = commit.policy().candidateMask();
        String[] commitIds = commit.recipeIds();
        if (!commit.policy().available()
            || candidateIndex < 0
            || candidateIndex >= commitMask.length
            || !commitMask[candidateIndex]
            || observedId.isEmpty()
            || !observedId.equals(commitIds[candidateIndex])) {
            rejectedBinding++;
            return;
        }

        World world = store.getExternalData().getWorld();
        if (world == null) {
            contextFailures++;
            return;
        }
        PolicyWorldAnchorRegistry.Context context = anchors.context(world);
        BridgeWorldVerbFacade.ContextResult installed = facade.installContext(
            ref, store, context.packetAnchor());
        if (!installed.accepted()) {
            contextFailures++;
            return;
        }

        long requestId = nextRequestId.getAndIncrement();
        if (requestId < 0L) {
            facade.uninstallContext(ref, store);
            rejectedBinding++;
            return;
        }
        BridgeWorldVerbFacade.Admission admission = facade.startFieldcraft(
            ref, store, requestId, observedId, context.worldEpoch());
        if (!admission.accepted()) {
            facade.uninstallContext(ref, store);
            failed++;
            return;
        }
        begin(marker, ref, store, "craft_recipe", admission);
    }

    private void begin(
        PolicyAgentMarker marker,
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        String verb,
        BridgeWorldVerbFacade.Admission admission
    ) {
        accepted++;
        if (admission.lifecycle().finished()) {
            if (admission.lifecycle().failed()) failed++;
            else finished++;
            facade.uninstallContext(ref, store);
            marker.worldVerb().observe(admission.lifecycle());
            return;
        }
        marker.worldVerb().begin(
            verb, admission.execution(), admission.lifecycle());
        synchronized (activeActors) {
            activeActors.put(marker, new ActiveActor(ref, store));
        }
    }

    private void pollActive(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        PolicyAgentMarker marker
    ) {
        PolicyWorldVerbState state = marker.worldVerb();
        BridgeWorldVerbFacade.Execution execution = state.execution();
        if (execution == null) return;
        BridgeWorldVerbFacade.Lifecycle lifecycle = facade.poll(
            execution, ref, store);
        state.observe(lifecycle);
        if (!lifecycle.finished()) return;
        if (lifecycle.failed()) failed++;
        else finished++;
        facade.uninstallContext(ref, store);
        state.finish(lifecycle);
        synchronized (activeActors) {
            activeActors.remove(marker);
        }
    }

    /** Remove active synthetic Player contexts before the policy plugin unloads. */
    public void closeActive() {
        Map<PolicyAgentMarker, ActiveActor> copy;
        synchronized (activeActors) {
            copy = new IdentityHashMap<>(activeActors);
            activeActors.clear();
        }
        for (Map.Entry<PolicyAgentMarker, ActiveActor> entry : copy.entrySet()) {
            ActiveActor actor = entry.getValue();
            BridgeWorldVerbFacade.Lifecycle lifecycle =
                BridgeWorldVerbFacade.Lifecycle.rejected(
                    "policy_plugin_shutdown");
            facade.uninstallContext(actor.ref(), actor.store());
            entry.getKey().worldVerb().finish(lifecycle);
        }
    }

    public long[] counters() {
        return new long[] {
            requested, accepted, finished, failed, rejectedBinding,
            rejectedBusy, unsupported, heldEdgeSuppressed, contextFailures,
        };
    }

    private record ActiveActor(
        Ref<EntityStore> ref,
        Store<EntityStore> store
    ) {}
}
