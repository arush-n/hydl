package com.hytalerlbridge.nativebackend.policy.combat;

import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.activeItemId;
import static com.hytalerlbridge.nativebackend.support.EvidenceCapture.sameEntity;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.clearMarkedTargets;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.clearNativeCombat;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.discoverAttackActions;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.hasUnfinishedInteractionTree;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.interactionManager;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.InteractionState;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.server.core.entity.InteractionChain;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.entity.damage.DamageDataComponent;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.corecomponents.combat.ActionAttack;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.role.support.CombatSupport;
import com.hypixel.hytale.server.npc.role.support.MarkedEntitySupport;
import com.hytalerlbridge.combat.dodge.NativeDodgeProgram;
import com.hytalerlbridge.nativebackend.NativeInteractionQueue;
import com.hytalerlbridge.nativebackend.NativeQueuedInteractionLifecycle;
import com.hytalerlbridge.nativebackend.model.NativeInteractionBinding;
import java.util.List;

/**
 * One actor's bridge-owned combat execution state.
 *
 * <p>The autonomous policy facade owns one instance per actor. The socket
 * environment shares this controller's ref-based queue, charge, lifecycle,
 * and synthetic-client primitives; migrating its legacy telemetry fields onto
 * this high-level holder is intentionally separate from the production action
 * path. The controller retains only caller input edges/cursors and references
 * to engine-owned interaction chains. Damage, cooldowns, active interactions,
 * wielding, and attack pauses remain authoritative Hytale state.</p>
 */
public final class NativeCombatActorController {

    public static final int ABILITY_CAPACITY = 16;
    public static final double INDEFINITE_HOLD_SECONDS = Float.MAX_VALUE;

    private final Bindings bindings;
    private final NativeSyntheticCombatClient syntheticClient =
        new NativeSyntheticCombatClient();

    private Ref<EntityStore> actor;
    private World world;
    private List<ActionAttack> roleAttacks = List.of();
    private int roleAttackIndex;
    private boolean manualAttackWindow;
    private boolean guardInputHeld;
    private boolean guardActive;
    private InteractionChain guardChain;
    private Tracked ability = Tracked.none();
    private NativeGuardFork.Request pendingAbilityFork;
    private Tracked dodge = Tracked.none();

    public NativeCombatActorController(Bindings bindings) {
        if (bindings == null) {
            throw new IllegalArgumentException("combat bindings are required");
        }
        this.bindings = bindings;
    }

    /** Bind this caller-owned state to exactly one live actor and world. */
    public synchronized Result initialize(
        Ref<EntityStore> reference,
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        if (reference == null || !reference.isValid() || npc == null
            || store == null || store.getExternalData().getWorld() == null) {
            return Result.rejected("controlled_entity_unavailable");
        }
        Role role = npc.getRole();
        if (role == null) return Result.rejected("native_role_unavailable");
        if (!itemMatches(npc)) {
            return Result.rejected("native_combat_item_mismatch");
        }
        this.actor = reference;
        this.world = store.getExternalData().getWorld();
        this.roleAttacks = discoverAttackActions(role);
        this.roleAttackIndex = 0;
        this.manualAttackWindow = false;
        this.guardInputHeld = false;
        this.guardActive = hasActiveWielding(reference, store);
        this.guardChain = null;
        this.ability = Tracked.none();
        this.pendingAbilityFork = null;
        this.dodge = Tracked.none();
        this.syntheticClient.reset();
        return Result.accepted(snapshot());
    }

    /** Supply synthetic client rows before Hytale drains interactions. */
    public synchronized Result prepare(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        float deltaTime,
        boolean externalRemoteClientActive
    ) {
        String invalid = validateActor(reference, store);
        if (!invalid.isEmpty()) return Result.rejected(invalid);
        if (!Float.isFinite(deltaTime) || deltaTime < 0.0f) {
            return Result.rejected("native_combat_delta_time_invalid");
        }
        syntheticClient.prepare(
            reference,
            store,
            deltaTime,
            externalRemoteClientActive
        );
        return Result.accepted(snapshot());
    }

    /**
     * Poll completed chains, suppress authored autonomous attacks, update the
     * guard level, and start first-control-tick edges.
     */
    public synchronized Result apply(
        Ref<EntityStore> reference,
        NPCEntity npc,
        Store<EntityStore> store,
        float deltaTime,
        boolean firstControlTick,
        boolean attackRequested,
        int abilitySlot,
        boolean guardHeld,
        int dodgeDirection,
        double requestedChargeSeconds,
        boolean externalRootActive,
        boolean externalRemoteClientActive
    ) {
        String invalid = validateActor(reference, store);
        if (!invalid.isEmpty()) return Result.rejected(invalid);
        if (npc == null || npc.getRole() == null || !itemMatches(npc)) {
            return Result.rejected("native_combat_item_or_role_changed");
        }
        if (!Float.isFinite(deltaTime) || deltaTime < 0.0f
            || !Double.isFinite(requestedChargeSeconds)
            || requestedChargeSeconds < 0.0
            || requestedChargeSeconds > Float.MAX_VALUE) {
            return Result.rejected("native_combat_timing_invalid");
        }

        Role role = npc.getRole();
        Ref<EntityStore> selectedTarget = selectedCombatTarget(
            reference,
            role,
            store
        );
        Events events = poll(reference, store, selectedTarget);
        suppressAutonomousCombat(
            reference,
            role,
            store,
            externalRootActive
        );
        NativeInteractionBinding requestedAbility = abilitySlot >= 0
            && abilitySlot < ABILITY_CAPACITY
            ? bindings.ability(abilitySlot)
            : null;
        InteractionType requestedGuardFork = firstControlTick
            ? NativeGuardFork.resolveType(bindings.guard(), requestedAbility)
            : null;
        boolean retainGuardForFork = requestedGuardFork != null
            && guardInputHeld
            && guardActive;
        Admission guardState = updateGuard(
            reference,
            store,
            guardHeld || retainGuardForFork,
            externalRemoteClientActive,
            selectedTarget
        );
        Admission guard = retainGuardForFork
            && !guardHeld
            && guardState.accepted()
            ? Admission.unrequested()
            : guardState;

        Admission attack = Admission.unrequested();
        Admission abilityAdmission = Admission.unrequested();
        Admission dodgeAdmission = Admission.unrequested();
        if (firstControlTick) {
            if (attackRequested) {
                attack = startAttack(
                    reference,
                    role,
                    store,
                    deltaTime,
                    requestedChargeSeconds,
                    selectedTarget
                );
            }
            if (abilitySlot >= 0) {
                abilityAdmission = startAbility(
                    reference,
                    role,
                    store,
                    abilitySlot,
                    requestedChargeSeconds,
                    selectedTarget
                );
            }
            if (dodgeDirection > 0) {
                dodgeAdmission = startDodge(
                    reference,
                    store,
                    dodgeDirection,
                    selectedTarget
                );
            }
        }
        return Result.accepted(new Snapshot(
            attack,
            abilityAdmission,
            dodgeAdmission,
            guard,
            events,
            activeSnapshot()
        ));
    }

    /** Cancel only chains this controller owns and forget its actor binding. */
    public synchronized void close(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        boolean externalRemoteClientActive
    ) {
        if (reference != null && reference.isValid() && store != null) {
            InteractionManager manager = interactionManager(reference, store);
            if (manager != null) {
                cancel(manager, guardChain);
                cancel(manager, ability.chain());
                cancel(manager, dodge.chain());
            }
            syntheticClient.reset();
            syntheticClient.restoreNpcSimulationIfIdle(
                reference,
                store,
                externalRemoteClientActive
            );
        } else {
            syntheticClient.reset();
        }
        actor = null;
        world = null;
        roleAttacks = List.of();
        roleAttackIndex = 0;
        manualAttackWindow = false;
        guardInputHeld = false;
        guardActive = false;
        guardChain = null;
        ability = Tracked.none();
        pendingAbilityFork = null;
        dodge = Tracked.none();
    }

    public Bindings bindings() {
        return bindings;
    }

    public synchronized Snapshot snapshot() {
        return new Snapshot(
            Admission.unrequested(),
            Admission.unrequested(),
            Admission.unrequested(),
            Admission.unrequested(),
            Events.none(),
            activeSnapshot()
        );
    }

    private Admission startAttack(
        Ref<EntityStore> reference,
        Role role,
        Store<EntityStore> store,
        double deltaTime,
        double charge,
        Ref<EntityStore> selectedTarget
    ) {
        CombatSupport combat = role.getCombatSupport();
        if (combat == null) {
            return Admission.rejected("native_combat_busy_or_unavailable");
        }
        NativeInteractionBinding binding = bindings.attack();
        if (binding != null) {
            InteractionChain chain = NativeInteractionQueue.queue(
                reference,
                store,
                binding.type(),
                binding.root(),
                charge
            );
            if (chain == null) {
                return Admission.rejected(
                    "native_interaction_rejected_charge_or_rules"
                );
            }
            syntheticClient.bind(
                reference,
                store,
                chain,
                charge,
                selectedTarget
            );
            combat.setExecutingAttack(chain, false, 0.0);
            manualAttackWindow = true;
            return Admission.accepted(binding.id());
        }
        if (combat.isExecutingAttack()) {
            return Admission.rejected("native_combat_busy_or_unavailable");
        }
        if (charge != 0.0) {
            return Admission.rejected(
                "role_attack_does_not_accept_explicit_charge"
            );
        }
        if (roleAttacks.isEmpty()) {
            return Admission.rejected("role_attack_unavailable");
        }
        int index = Math.floorMod(roleAttackIndex, roleAttacks.size());
        ActionAttack attack = roleAttacks.get(index);
        roleAttackIndex = (index + 1) % roleAttacks.size();
        boolean accepted = attack.execute(
            reference,
            role,
            null,
            deltaTime,
            store
        );
        if (accepted) manualAttackWindow = true;
        return accepted
            ? Admission.accepted("role_attack:" + index)
            : Admission.rejected("role_attack_not_ready");
    }

    private Admission startAbility(
        Ref<EntityStore> reference,
        Role role,
        Store<EntityStore> store,
        int slot,
        double charge,
        Ref<EntityStore> selectedTarget
    ) {
        if (slot < 0 || slot >= ABILITY_CAPACITY) {
            return Admission.rejected("ability_slot_out_of_range");
        }
        NativeInteractionBinding binding = bindings.ability(slot);
        if (binding == null) {
            return Admission.rejected("ability_slot_not_bound");
        }
        if (ability.chain() != null || pendingAbilityFork != null) {
            return Admission.rejected("native_combat_busy");
        }
        InteractionType guardForkType = NativeGuardFork.resolveType(
            bindings.guard(),
            binding
        );
        if (guardForkType != null) {
            if (!guardInputHeld || !guardActive || guardChain == null
                || guardChain.getServerState()
                    != InteractionState.NotFinished) {
                return Admission.rejected(
                    "native_guard_fork_requires_held_parent"
                );
            }
            NativeGuardFork.Request request = NativeGuardFork.capture(
                guardChain,
                binding,
                slot,
                guardForkType,
                charge
            );
            if (!syntheticClient.requestFork(guardChain, guardForkType)) {
                return Admission.rejected(
                    "native_guard_fork_client_unavailable"
                );
            }
            pendingAbilityFork = request;
            manualAttackWindow = true;
            return Admission.accepted(binding.id());
        }
        CombatSupport combat = role.getCombatSupport();
        if (combat == null) {
            return Admission.rejected("native_combat_busy");
        }
        InteractionChain chain = NativeInteractionQueue.queue(
            reference,
            store,
            binding.type(),
            binding.root(),
            charge
        );
        if (chain == null) {
            return Admission.rejected(
                "native_interaction_rejected_charge_or_rules"
            );
        }
        syntheticClient.bind(
            reference,
            store,
            chain,
            charge,
            selectedTarget
        );
        combat.setExecutingAttack(chain, false, 0.0);
        manualAttackWindow = true;
        ability = Tracked.requested(
            chain,
            slot,
            binding.id(),
            binding.type().name()
        );
        return Admission.accepted(binding.id());
    }

    private Admission startDodge(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        int direction,
        Ref<EntityStore> selectedTarget
    ) {
        NativeInteractionBinding binding = switch (direction) {
            case NativeDodgeProgram.LEFT_DIRECTION -> bindings.dodgeLeft();
            case NativeDodgeProgram.RIGHT_DIRECTION -> bindings.dodgeRight();
            default -> null;
        };
        if (binding == null) {
            return Admission.rejected(
                "dodge_direction_has_no_authored_payload_hytale_0_5_7"
            );
        }
        InteractionChain chain = NativeInteractionQueue.queue(
            reference,
            store,
            binding.type(),
            binding.root(),
            0.0
        );
        if (chain == null) {
            return Admission.rejected("native_dodge_chain_rejected");
        }
        syntheticClient.bind(
            reference,
            store,
            chain,
            0.0,
            selectedTarget
        );
        dodge = Tracked.requested(
            chain,
            direction,
            binding.id(),
            binding.type().name()
        );
        return Admission.accepted(binding.id());
    }

    private Admission updateGuard(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        boolean held,
        boolean externalRemoteClientActive,
        Ref<EntityStore> selectedTarget
    ) {
        InteractionManager manager = interactionManager(reference, store);
        boolean rising = held && !guardInputHeld;
        boolean falling = !held && guardInputHeld;
        guardInputHeld = held;

        if (!held) {
            if (falling && manager != null && guardChain != null
                && !syntheticClient.release(reference, store, guardChain)) {
                return Admission.rejected("native_guard_release_rejected");
            }
            if (terminalGuard(reference, store)) {
                guardChain = null;
                syntheticClient.restoreNpcSimulationIfIdle(
                    reference,
                    store,
                    externalRemoteClientActive
                );
            }
            return falling
                ? Admission.accepted("guard_release")
                : Admission.unrequested();
        }
        NativeInteractionBinding binding = bindings.guard();
        if (binding == null) {
            return Admission.rejected("guard_interaction_not_bound");
        }
        if (manager == null) {
            return Admission.rejected("interaction_manager_unavailable");
        }
        if (guardChain != null) {
            if (terminalGuard(reference, store)) {
                guardChain = null;
                syntheticClient.restoreNpcSimulationIfIdle(
                    reference,
                    store,
                    externalRemoteClientActive
                );
            } else {
                return rising
                    ? Admission.rejected("native_guard_release_pending")
                    : Admission.accepted(binding.id());
            }
        }
        if (!rising) {
            return Admission.rejected("native_guard_requires_new_press");
        }
        guardChain = NativeInteractionQueue.queue(
            reference,
            store,
            binding.type(),
            binding.root(),
            INDEFINITE_HOLD_SECONDS
        );
        if (guardChain == null) {
            return Admission.rejected("native_guard_chain_rejected");
        }
        // Guard is continuation-held even for an NPC and therefore always
        // needs the explicit release row supplied by the synthetic cursor.
        syntheticClient.bind(
            reference,
            store,
            guardChain,
            INDEFINITE_HOLD_SECONDS,
            selectedTarget
        );
        return Admission.accepted(binding.id());
    }

    private Events poll(
        Ref<EntityStore> reference,
        Store<EntityStore> store,
        Ref<EntityStore> selectedTarget
    ) {
        InteractionManager manager = interactionManager(reference, store);
        Poll pendingPoll = Poll.none();
        if (pendingAbilityFork != null) {
            InteractionChain fork = NativeGuardFork.findSpawned(
                pendingAbilityFork
            );
            if (fork != null) {
                Ref<EntityStore> forkTarget = selectedTarget != null
                    ? selectedTarget
                    : syntheticClient.selectedTarget(
                        pendingAbilityFork.parent()
                    );
                syntheticClient.bind(
                    reference,
                    store,
                    fork,
                    pendingAbilityFork.requestedChargeSeconds(),
                    forkTarget
                );
                ability = Tracked.requested(
                    fork,
                    pendingAbilityFork.parent(),
                    pendingAbilityFork.abilitySlot(),
                    pendingAbilityFork.ability().id(),
                    pendingAbilityFork.forkType().name()
                );
                pendingAbilityFork = null;
            } else if (
                pendingAbilityFork.parent().getServerState()
                    != InteractionState.NotFinished
            ) {
                pendingPoll = Poll.rejectedBeforeStart(
                    pendingAbilityFork.abilitySlot()
                );
                pendingAbilityFork = null;
            }
        }
        Poll abilityPoll = pendingPoll.rejectedBeforeStart()
            ? pendingPoll
            : pollTracked(manager, ability);
        if (abilityPoll.clear()) {
            ability = Tracked.none();
        } else {
            ability = abilityPoll.tracked();
        }
        Poll dodgePoll = pollTracked(manager, dodge);
        if (dodgePoll.clear()) {
            dodge = Tracked.none();
        } else {
            dodge = dodgePoll.tracked();
        }
        boolean wielding = hasActiveWielding(reference, store);
        boolean guardStarted = wielding && !guardActive;
        boolean guardFinished = !wielding && guardActive;
        guardActive = wielding;
        return new Events(
            abilityPoll.started(),
            abilityPoll.finished(),
            abilityPoll.failed(),
            abilityPoll.rejectedBeforeStart(),
            abilityPoll.slot(),
            dodgePoll.started(),
            dodgePoll.finished(),
            dodgePoll.failed(),
            dodgePoll.rejectedBeforeStart(),
            dodgePoll.slot(),
            guardStarted,
            guardFinished
        );
    }

    private static Poll pollTracked(
        InteractionManager manager,
        Tracked tracked
    ) {
        if (tracked.chain() == null) return Poll.none();
        InteractionChain chain = tracked.chain();
        boolean registered = manager != null
            && (
                manager.getChains().values().contains(chain)
                || NativeGuardFork.containsChild(
                    tracked.parent(),
                    chain
                )
            );
        NativeQueuedInteractionLifecycle.Observation observation =
            NativeQueuedInteractionLifecycle.classify(
                registered,
                chain.getChainId(),
                chain.getServerState()
            );
        boolean newStart = observation.admitted() && !tracked.started();
        Tracked updated = newStart ? tracked.startedNow() : tracked;
        boolean terminal = observation.terminal();
        boolean rejected = observation.rejectedBeforeStart();
        return new Poll(
            updated,
            newStart,
            terminal,
            terminal && chain.getServerState() == InteractionState.Failed,
            rejected,
            terminal || rejected,
            tracked.slot()
        );
    }

    private void suppressAutonomousCombat(
        Ref<EntityStore> reference,
        Role role,
        Store<EntityStore> store,
        boolean externalRootActive
    ) {
        CombatSupport combat = role.getCombatSupport();
        InteractionManager manager = interactionManager(reference, store);
        if (combat == null || manager == null) return;
        clearMarkedTargets(role);
        if (manualAttackWindow
            && (combat.isExecutingAttack()
                || hasUnfinishedInteractionTree(manager))) return;
        if (guardChain != null
            || active(manager, ability)
            || pendingAbilityFork != null
            || active(manager, dodge.chain())
            || externalRootActive) {
            manualAttackWindow = false;
            combat.setExecutingAttack(null, false, 0.0);
            return;
        }
        manualAttackWindow = false;
        clearNativeCombat(manager, combat);
    }

    private boolean terminalGuard(
        Ref<EntityStore> reference,
        Store<EntityStore> store
    ) {
        return guardChain != null
            && guardChain.getServerState() != InteractionState.NotFinished
            && !hasActiveWielding(reference, store);
    }

    private static Ref<EntityStore> selectedCombatTarget(
        Ref<EntityStore> reference,
        Role role,
        Store<EntityStore> store
    ) {
        MarkedEntitySupport marked = role == null
            ? null
            : role.getMarkedEntitySupport();
        Ref<EntityStore> target = marked == null
            ? null
            : marked.getMarkedEntityRef(
                MarkedEntitySupport.DEFAULT_TARGET_SLOT
            );
        if (target == null || !target.isValid() || target == reference
            || target.getStore() != store) {
            return null;
        }
        return target;
    }

    private boolean itemMatches(NPCEntity npc) {
        String expected = bindings.expectedItemId();
        return expected.isBlank()
            || expected.equals(activeItemId(npc.getInventory()));
    }

    private String validateActor(
        Ref<EntityStore> reference,
        Store<EntityStore> store
    ) {
        if (actor == null || world == null) {
            return "native_combat_actor_not_initialized";
        }
        if (reference == null || !reference.isValid() || store == null
            || reference.getStore() != store
            || !sameActor(reference, actor)
            || store.getExternalData().getWorld() != world) {
            return "native_combat_actor_or_world_changed";
        }
        return "";
    }

    /** ECS may issue a new {@link Ref} wrapper for the same store/index. */
    static boolean sameActor(
        Ref<EntityStore> left,
        Ref<EntityStore> right
    ) {
        return sameEntity(left, right);
    }

    private Active activeSnapshot() {
        NativeGuardFork.Request pendingFork = pendingAbilityFork;
        boolean abilityActive = ability.chain() != null
            || pendingFork != null;
        int abilitySlot = pendingFork == null
            ? ability.slot()
            : pendingFork.abilitySlot();
        String abilityInteractionId = pendingFork == null
            ? ability.interactionId()
            : pendingFork.ability().id();
        return new Active(
            abilityActive,
            abilitySlot,
            abilityInteractionId,
            dodge.chain() != null,
            dodge.slot(),
            dodge.interactionId(),
            guardChain != null,
            guardActive,
            syntheticClient.activeCount(),
            syntheticClient.bindCount(),
            syntheticClient.syncCount()
        );
    }

    private static boolean active(
        InteractionManager manager,
        InteractionChain chain
    ) {
        return chain != null
            && manager.getChains().values().contains(chain);
    }

    private static boolean active(
        InteractionManager manager,
        Tracked tracked
    ) {
        return tracked != null && tracked.chain() != null
            && (
                active(manager, tracked.chain())
                || NativeGuardFork.containsChild(
                    tracked.parent(),
                    tracked.chain()
                )
            );
    }

    private static void cancel(
        InteractionManager manager,
        InteractionChain chain
    ) {
        if (chain != null && manager.getChains().values().contains(chain)) {
            manager.cancelChains(chain);
        }
    }

    private static boolean hasActiveWielding(
        Ref<EntityStore> reference,
        Store<EntityStore> store
    ) {
        DamageDataComponent damage = store.getComponent(
            reference,
            DamageDataComponent.getComponentType()
        );
        return damage != null && damage.getCurrentWielding() != null;
    }

    /** Immutable resolved roots; array accessors never expose mutable storage. */
    public record Bindings(
        String expectedItemId,
        NativeInteractionBinding attack,
        NativeInteractionBinding[] abilities,
        NativeInteractionBinding guard,
        NativeInteractionBinding dodgeLeft,
        NativeInteractionBinding dodgeRight
    ) {
        public Bindings {
            expectedItemId = expectedItemId == null ? "" : expectedItemId;
            if (abilities == null || abilities.length != ABILITY_CAPACITY) {
                throw new IllegalArgumentException(
                    "combat ability binding width must be " + ABILITY_CAPACITY
                );
            }
            abilities = abilities.clone();
        }

        @Override
        public NativeInteractionBinding[] abilities() {
            return abilities.clone();
        }

        /** Allocation-free immutable slot lookup for the per-tick path. */
        public NativeInteractionBinding ability(int slot) {
            return slot < 0 || slot >= abilities.length
                ? null
                : abilities[slot];
        }
    }

    public record Admission(
        boolean requested,
        boolean accepted,
        String interactionId,
        String rejectReason
    ) {
        public Admission {
            interactionId = interactionId == null ? "" : interactionId;
            rejectReason = rejectReason == null ? "" : rejectReason;
        }

        public static Admission unrequested() {
            return new Admission(false, false, "", "");
        }

        public static Admission accepted(String id) {
            return new Admission(true, true, id, "");
        }

        public static Admission rejected(String reason) {
            return new Admission(true, false, "", reason);
        }
    }

    public record Events(
        boolean abilityStarted,
        boolean abilityFinished,
        boolean abilityFailed,
        boolean abilityRejectedBeforeStart,
        int abilitySlot,
        boolean dodgeStarted,
        boolean dodgeFinished,
        boolean dodgeFailed,
        boolean dodgeRejectedBeforeStart,
        int dodgeDirection,
        boolean guardStarted,
        boolean guardFinished
    ) {
        public static Events none() {
            return new Events(
                false, false, false, false, -1,
                false, false, false, false, 0,
                false, false
            );
        }
    }

    public record Active(
        boolean abilityActive,
        int abilitySlot,
        String abilityInteractionId,
        boolean dodgeActive,
        int dodgeDirection,
        String dodgeInteractionId,
        boolean guardChainActive,
        boolean guardWieldingActive,
        int syntheticClientChains,
        long syntheticClientBinds,
        long syntheticClientSyncs
    ) {}

    public record Snapshot(
        Admission attack,
        Admission ability,
        Admission dodge,
        Admission guard,
        Events events,
        Active active
    ) {}

    public record Result(
        boolean available,
        String unavailableReason,
        Snapshot snapshot
    ) {
        public static Result accepted(Snapshot snapshot) {
            return new Result(true, "", snapshot);
        }

        public static Result rejected(String reason) {
            return new Result(false, reason, null);
        }
    }

    private record Tracked(
        InteractionChain chain,
        InteractionChain parent,
        int slot,
        String interactionId,
        String interactionType,
        boolean started
    ) {
        static Tracked none() {
            return new Tracked(null, null, -1, "", "", false);
        }

        static Tracked requested(
            InteractionChain chain,
            int slot,
            String id,
            String type
        ) {
            return requested(chain, null, slot, id, type);
        }

        static Tracked requested(
            InteractionChain chain,
            InteractionChain parent,
            int slot,
            String id,
            String type
        ) {
            return new Tracked(chain, parent, slot, id, type, false);
        }

        Tracked startedNow() {
            return new Tracked(
                chain,
                parent,
                slot,
                interactionId,
                interactionType,
                true
            );
        }
    }

    private record Poll(
        Tracked tracked,
        boolean started,
        boolean finished,
        boolean failed,
        boolean rejectedBeforeStart,
        boolean clear,
        int slot
    ) {
        static Poll none() {
            return new Poll(
                Tracked.none(),
                false,
                false,
                false,
                false,
                false,
                -1
            );
        }

        static Poll rejectedBeforeStart(int slot) {
            return new Poll(
                Tracked.none(),
                false,
                false,
                false,
                true,
                true,
                slot
            );
        }
    }
}
