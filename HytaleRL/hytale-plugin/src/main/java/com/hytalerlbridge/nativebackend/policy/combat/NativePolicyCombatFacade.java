package com.hytalerlbridge.nativebackend.policy.combat;

import static com.hytalerlbridge.nativebackend.support.InteractionSupport.resolveNativeAbilityBindings;
import static com.hytalerlbridge.nativebackend.support.InteractionSupport.resolveNativeInteractionBinding;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.combat.dodge.NativeDodgeBindingResolver;
import com.hytalerlbridge.combat.dodge.NativeDodgeProgram;
import com.hytalerlbridge.nativebackend.model.NativeInteractionBinding;
import java.util.HashSet;
import java.util.Set;

/**
 * Reflection-safe combat admission for autonomous policy-controlled NPCs.
 *
 * <p>The facade owns no actor registry. The caller retains one opaque handle
 * per actor and calls {@link #prepare} before Hytale's interaction tick and
 * {@link #apply} after it. The same ref-based controller is available to the
 * socket environment, so this class does not reproduce interaction rules in
 * the standalone policy mod.</p>
 */
public final class NativePolicyCombatFacade {

    public static final String SCHEMA =
        "hytalerl_native_policy_combat_facade_v1";
    public static final int VERSION = 1;

    private NativePolicyCombatFacade() {}

    public static String schema() {
        return SCHEMA;
    }

    public static int version() {
        return VERSION;
    }

    /** Resolve checkpoint-pinned roots once; no gameplay state is captured. */
    public static Binding bind(
        String expectedItemId,
        int[] abilitySlots,
        String[] abilityInteractionIds,
        String[] abilityInteractionTypes,
        String guardInteractionId,
        String guardInteractionType
    ) {
        if (expectedItemId == null || abilitySlots == null
            || abilityInteractionIds == null
            || abilityInteractionTypes == null) {
            return Binding.rejected("native_combat_binding_invalid");
        }
        if (abilitySlots.length != abilityInteractionIds.length
            || abilitySlots.length != abilityInteractionTypes.length) {
            return Binding.rejected("native_combat_binding_width_mismatch");
        }
        try {
            NativeInteractionBinding[] abilities =
                new NativeInteractionBinding[
                    NativeCombatActorController.ABILITY_CAPACITY];
            Set<Integer> seen = new HashSet<>();
            for (int index = 0; index < abilitySlots.length; index++) {
                int slot = abilitySlots[index];
                if (slot < 0 || slot >= abilities.length || !seen.add(slot)) {
                    return Binding.rejected(
                        "native_combat_ability_slot_invalid");
                }
            }
            NativeInteractionBinding[] compactAbilities =
                resolveNativeAbilityBindings(
                    expectedItemId,
                    abilityInteractionIds,
                    abilityInteractionTypes
                );
            for (int index = 0; index < abilitySlots.length; index++) {
                abilities[abilitySlots[index]] = compactAbilities[index];
            }

            NativeInteractionBinding guard = null;
            boolean guardIdPresent = guardInteractionId != null
                && !guardInteractionId.isBlank();
            boolean guardTypePresent = guardInteractionType != null
                && !guardInteractionType.isBlank();
            if (guardIdPresent != guardTypePresent) {
                return Binding.rejected("native_guard_binding_incomplete");
            }
            if (guardIdPresent) {
                guard = resolveNativeInteractionBinding(
                    guardInteractionId,
                    guardInteractionType
                );
            }

            NativeInteractionBinding dodgeLeft =
                NativeDodgeBindingResolver.resolve(
                    NativeDodgeProgram.LEFT_DIRECTION
                );
            NativeInteractionBinding dodgeRight =
                NativeDodgeBindingResolver.resolve(
                    NativeDodgeProgram.RIGHT_DIRECTION
                );
            NativeCombatActorController controller =
                new NativeCombatActorController(
                    new NativeCombatActorController.Bindings(
                        expectedItemId,
                        null,
                        abilities,
                        guard,
                        dodgeLeft,
                        dodgeRight
                    )
                );
            return Binding.accepted(controller);
        } catch (IllegalArgumentException | IllegalStateException unavailable) {
            return Binding.rejected("native_combat_binding_unavailable");
        }
    }

    public static Receipt initialize(
        Object opaqueHandle,
        Ref<EntityStore> actor,
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        NativeCombatActorController controller = handle(opaqueHandle);
        return controller == null
            ? Receipt.unavailable("native_combat_handle_invalid")
            : Receipt.from(controller.initialize(actor, npc, store));
    }

    /** Supply client-owned interaction rows before InteractionManager ticks. */
    public static Receipt prepare(
        Object opaqueHandle,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        float deltaTime,
        boolean externalRemoteClientActive
    ) {
        NativeCombatActorController controller = handle(opaqueHandle);
        return controller == null
            ? Receipt.unavailable("native_combat_handle_invalid")
            : Receipt.from(controller.prepare(
                actor, store, deltaTime, externalRemoteClientActive));
    }

    /** Apply held guard plus first-control-tick combat edges. */
    public static Receipt apply(
        Object opaqueHandle,
        Ref<EntityStore> actor,
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
        NativeCombatActorController controller = handle(opaqueHandle);
        return controller == null
            ? Receipt.unavailable("native_combat_handle_invalid")
            : Receipt.from(controller.apply(
                actor,
                npc,
                store,
                deltaTime,
                firstControlTick,
                attackRequested,
                abilitySlot,
                guardHeld,
                dodgeDirection,
                requestedChargeSeconds,
                externalRootActive,
                externalRemoteClientActive
            ));
    }

    public static void close(
        Object opaqueHandle,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        boolean externalRemoteClientActive
    ) {
        NativeCombatActorController controller = handle(opaqueHandle);
        if (controller != null) {
            controller.close(actor, store, externalRemoteClientActive);
        }
    }

    private static NativeCombatActorController handle(Object value) {
        return value instanceof NativeCombatActorController controller
            ? controller
            : null;
    }

    public record Binding(
        boolean accepted,
        String rejectReason,
        Object handle
    ) {
        private static Binding accepted(NativeCombatActorController value) {
            return new Binding(true, "", value);
        }

        private static Binding rejected(String reason) {
            return new Binding(false, reason, null);
        }
    }

    /** Immutable admission, lifecycle, and active-state receipt. */
    public record Receipt(
        boolean available,
        String unavailableReason,
        boolean attackRequested,
        boolean attackAccepted,
        String attackInteractionId,
        String attackRejectReason,
        boolean abilityRequested,
        boolean abilityAccepted,
        String abilityInteractionId,
        String abilityRejectReason,
        boolean dodgeRequested,
        boolean dodgeAccepted,
        String dodgeInteractionId,
        String dodgeRejectReason,
        boolean guardRequested,
        boolean guardAccepted,
        String guardInteractionId,
        String guardRejectReason,
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
        boolean guardFinished,
        boolean abilityActive,
        int activeAbilitySlot,
        boolean dodgeActive,
        int activeDodgeDirection,
        boolean guardChainActive,
        boolean guardWieldingActive,
        int syntheticClientChains
    ) {
        private static Receipt from(
            NativeCombatActorController.Result result
        ) {
            if (result == null || !result.available()
                || result.snapshot() == null) {
                return unavailable(result == null
                    ? "native_combat_receipt_unavailable"
                    : result.unavailableReason());
            }
            NativeCombatActorController.Snapshot snapshot = result.snapshot();
            NativeCombatActorController.Admission attack = snapshot.attack();
            NativeCombatActorController.Admission ability = snapshot.ability();
            NativeCombatActorController.Admission dodge = snapshot.dodge();
            NativeCombatActorController.Admission guard = snapshot.guard();
            NativeCombatActorController.Events events = snapshot.events();
            NativeCombatActorController.Active active = snapshot.active();
            return new Receipt(
                true,
                "",
                attack.requested(),
                attack.accepted(),
                attack.interactionId(),
                attack.rejectReason(),
                ability.requested(),
                ability.accepted(),
                ability.interactionId(),
                ability.rejectReason(),
                dodge.requested(),
                dodge.accepted(),
                dodge.interactionId(),
                dodge.rejectReason(),
                guard.requested(),
                guard.accepted(),
                guard.interactionId(),
                guard.rejectReason(),
                events.abilityStarted(),
                events.abilityFinished(),
                events.abilityFailed(),
                events.abilityRejectedBeforeStart(),
                events.abilitySlot(),
                events.dodgeStarted(),
                events.dodgeFinished(),
                events.dodgeFailed(),
                events.dodgeRejectedBeforeStart(),
                events.dodgeDirection(),
                events.guardStarted(),
                events.guardFinished(),
                active.abilityActive(),
                active.abilitySlot(),
                active.dodgeActive(),
                active.dodgeDirection(),
                active.guardChainActive(),
                active.guardWieldingActive(),
                active.syntheticClientChains()
            );
        }

        private static Receipt unavailable(String reason) {
            return new Receipt(
                false,
                reason == null ? "" : reason,
                false, false, "", "",
                false, false, "", "",
                false, false, "", "",
                false, false, "", "",
                false, false, false, false, -1,
                false, false, false, false, 0,
                false, false,
                false, -1, false, 0, false, false, 0
            );
        }
    }
}
