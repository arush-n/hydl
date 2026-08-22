package com.hytalerlbridge.policy.combat.runtime;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.PolicyActionSink;
import com.hytalerlbridge.policy.PolicyAgentMarker;
import com.hytalerlbridge.policy.combat.bridge.BridgeCombatFacade;
import com.hytalerlbridge.policy.combat.bridge.BridgeDodgeCooldownStateSource;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Composes an existing locomotion/World sink with bridge-owned combat roots.
 *
 * <p>Attack, ability, and dodge are first-control-tick edges. Guard is a held
 * level and is sent every controlled tick so its release edge cannot be lost.
 * The bridge remains the only implementation of interaction admission and
 * lifecycle state.</p>
 */
public final class CombatActionSink implements PolicyActionSink {

    private final PolicyActionSink base;
    private final BridgeCombatFacade facade;
    private final CombatReceiptObserver receiptObserver;
    private final BridgeDodgeCooldownStateSource dodgeCooldownStateSource;
    private final Map<PolicyAgentMarker, ActiveActor> activeActors =
        new IdentityHashMap<>();

    private long bindingsAccepted;
    private long bindingsRejected;
    private long callsUnavailable;
    private long attackAccepted;
    private long abilityAccepted;
    private long dodgeAccepted;
    private long guardStarted;
    private long abilityFinished;
    private long abilityFailed;
    private long dodgeFinished;
    private long dodgeFailed;

    public CombatActionSink(
        PolicyActionSink base,
        BridgeCombatFacade facade
    ) {
        this(
            base,
            facade,
            CombatReceiptObserver.NONE,
            BridgeDodgeCooldownStateSource.NONE
        );
    }

    public CombatActionSink(
        PolicyActionSink base,
        BridgeCombatFacade facade,
        CombatReceiptObserver receiptObserver
    ) {
        this(
            base,
            facade,
            receiptObserver,
            BridgeDodgeCooldownStateSource.NONE
        );
    }

    public CombatActionSink(
        PolicyActionSink base,
        BridgeCombatFacade facade,
        CombatReceiptObserver receiptObserver,
        BridgeDodgeCooldownStateSource dodgeCooldownStateSource
    ) {
        if (base == null || facade == null) {
            throw new IllegalArgumentException(
                "combat sink requires a base sink and bridge facade");
        }
        this.base = base;
        this.facade = facade;
        this.receiptObserver = receiptObserver == null
            ? CombatReceiptObserver.NONE : receiptObserver;
        this.dodgeCooldownStateSource = dodgeCooldownStateSource == null
            ? BridgeDodgeCooldownStateSource.NONE : dodgeCooldownStateSource;
    }

    /**
     * Resolve every bridge-private combat asset off the world thread.
     *
     * <p>The profile warm-up covers authored ability and Guard roots, but the
     * combat facade also owns Dodge probe roots.  Its first {@code bind()}
     * may therefore register assets and take the asset-store write lock.  A
     * world tick already holds the read lock, so discovering one there parks
     * the world permanently.  This disposable binding makes the later
     * per-actor binding hit only memoized assets; it captures no actor state.
     */
    public WarmupResult warmupBindingAssets() {
        BridgeCombatFacade.Binding binding = facade.bind();
        if (!binding.accepted() || binding.execution() == null) {
            String reason = binding.rejectReason();
            return new WarmupResult(
                false,
                reason == null || reason.isBlank()
                    ? "bridge_combat_binding_rejected"
                    : reason
            );
        }
        return new WarmupResult(true, "facade binding assets resolved");
    }

    @Override
    public boolean supportsWorldVerb(ActionDecoder.Decoded action) {
        return base.supportsWorldVerb(action);
    }

    /** Called by the pre-interaction ECS system. */
    public void prepare(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        float deltaTime,
        PolicyAgentMarker marker
    ) {
        BridgeCombatFacade.Execution execution = marker.combat().execution();
        if (execution == null) return;
        BridgeDodgeCooldownStateSource.State cooldownBefore =
            dodgeCooldownStateSource.capture(ref, store);
        BridgeCombatFacade.Receipt receipt = facade.prepare(
            execution,
            ref,
            store,
            deltaTime,
            marker.worldVerb().active()
        );
        marker.combat().observe(receipt);
        receiptObserver.observeCombatReceipt(
            CombatReceiptObserver.Phase.PREPARE,
            worldTick(store),
            marker.slot(),
            null,
            false,
            receipt,
            cooldownBefore,
            dodgeCooldownStateSource.capture(ref, store)
        );
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
        // World admission runs first. This lets the combat controller see an
        // active external root in the same control tick and suppress only the
        // conflicting interaction, never movement/look.
        base.apply(
            ref,
            npc,
            store,
            deltaTime,
            marker,
            action,
            evidence,
            firstControlTick
        );

        BridgeCombatFacade.Execution execution = ensureBound(
            ref, npc, store, marker);
        if (execution == null) return;
        int abilitySlot = action.abilitySlot();
        double requestedCharge =
            facade.requestedChargeSeconds(abilitySlot);
        boolean externalRootActive = marker.worldVerb().active();
        BridgeDodgeCooldownStateSource.State cooldownBefore =
            dodgeCooldownStateSource.capture(ref, store);
        BridgeCombatFacade.Receipt receipt = facade.apply(
            execution,
            ref,
            npc,
            store,
            deltaTime,
            firstControlTick,
            action.attack(),
            abilitySlot,
            action.guardHeld(),
            action.dodgeDirection(),
            requestedCharge,
            externalRootActive,
            externalRootActive
        );
        marker.combat().observe(receipt);
        receiptObserver.observeCombatReceipt(
            CombatReceiptObserver.Phase.APPLY,
            worldTick(store),
            marker.slot(),
            action,
            firstControlTick,
            receipt,
            cooldownBefore,
            dodgeCooldownStateSource.capture(ref, store)
        );
        count(receipt);
    }

    private BridgeCombatFacade.Execution ensureBound(
        Ref<EntityStore> ref,
        NPCEntity npc,
        Store<EntityStore> store,
        PolicyAgentMarker marker
    ) {
        BridgeCombatFacade.Execution current = marker.combat().execution();
        if (current != null) return current;
        BridgeCombatFacade.Binding binding = facade.bind();
        if (!binding.accepted() || binding.execution() == null) {
            bindingsRejected++;
            return null;
        }
        BridgeCombatFacade.Receipt initialized = facade.initialize(
            binding.execution(), ref, npc, store);
        if (!initialized.available()) {
            facade.close(
                binding.execution(), ref, store, marker.worldVerb().active());
            bindingsRejected++;
            return null;
        }
        bindingsAccepted++;
        marker.combat().begin(binding.execution(), initialized);
        synchronized (activeActors) {
            activeActors.put(marker, new ActiveActor(ref, store));
        }
        return binding.execution();
    }

    private void count(BridgeCombatFacade.Receipt receipt) {
        if (!receipt.available()) {
            callsUnavailable++;
            return;
        }
        if (receipt.attackAccepted()) attackAccepted++;
        if (receipt.abilityAccepted()) abilityAccepted++;
        if (receipt.dodgeAccepted()) dodgeAccepted++;
        if (receipt.guardStarted()) guardStarted++;
        if (receipt.abilityFinished()) abilityFinished++;
        if (receipt.abilityFailed()
            || receipt.abilityRejectedBeforeStart()) abilityFailed++;
        if (receipt.dodgeFinished()) dodgeFinished++;
        if (receipt.dodgeFailed()
            || receipt.dodgeRejectedBeforeStart()) dodgeFailed++;
    }

    /** Release only bridge-owned chains before this plugin unloads. */
    public void closeActive() {
        Map<PolicyAgentMarker, ActiveActor> copy;
        synchronized (activeActors) {
            copy = new IdentityHashMap<>(activeActors);
            activeActors.clear();
        }
        for (Map.Entry<PolicyAgentMarker, ActiveActor> entry : copy.entrySet()) {
            PolicyAgentMarker marker = entry.getKey();
            ActiveActor actor = entry.getValue();
            facade.close(
                marker.combat().execution(),
                actor.ref(),
                actor.store(),
                marker.worldVerb().active()
            );
            marker.combat().resetLocal();
        }
    }

    public long[] counters() {
        return new long[] {
            bindingsAccepted,
            bindingsRejected,
            callsUnavailable,
            attackAccepted,
            abilityAccepted,
            dodgeAccepted,
            guardStarted,
            abilityFinished,
            abilityFailed,
            dodgeFinished,
            dodgeFailed,
        };
    }

    private static long worldTick(Store<EntityStore> store) {
        if (store == null || store.getExternalData() == null
                || store.getExternalData().getWorld() == null) {
            return -1L;
        }
        return store.getExternalData().getWorld().getTick();
    }

    private record ActiveActor(
        Ref<EntityStore> ref,
        Store<EntityStore> store
    ) {}

    public record WarmupResult(boolean warmed, String detail) {
        public WarmupResult {
            detail = detail == null ? "" : detail;
        }
    }
}
