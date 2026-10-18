package com.hytalerlbridge.nativebackend.session;

import com.hytalerlbridge.action.group.NativePolicyCombatBindingSpec;
import com.hytalerlbridge.environment.EnvironmentOptions;
import com.hytalerlbridge.nativebackend.model.NativeInteractionBinding;
import java.util.List;

/**
 * Per-actor policy combat values for the info map (comma-separated, one entry per actor slot).
 */
public final class PolicyCombatKeys {

    private PolicyCombatKeys() {}

    public static String nativePolicyCombatBoundByActor(
        List<NativeInteractionBinding> nativeCombatAbilities,
        NativeInteractionBinding nativeDodgeLeftInteraction,
        NativeInteractionBinding nativeDodgeRightInteraction,
        NativeInteractionBinding nativeGuardInteraction,
        Object[] nativePolicyCombatHandles
    ) {
        boolean[] values = new boolean[nativePolicyCombatHandles.length];
        values[0] = nativeGuardInteraction != null
            || !nativeCombatAbilities.isEmpty()
            || nativeDodgeLeftInteraction != null
            || nativeDodgeRightInteraction != null;
        for (int actorId = 1; actorId < values.length; actorId++) {
            values[actorId] = nativePolicyCombatHandles[actorId] != null;
        }
        return joinActorBooleans(values);
    }

    public static String nativePolicyCombatAbilityMasksByActor(
        List<NativeInteractionBinding> nativeCombatAbilities,
        Object[] nativePolicyCombatHandles,
        NativePolicyCombatBindingSpec[] nativePolicyCombatSpecs
    ) {
        int[] values = new int[nativePolicyCombatHandles.length];
        values[0] = nativeCombatAbilities.isEmpty()
            ? 0
            : (1 << nativeCombatAbilities.size()) - 1;
        for (int actorId = 1; actorId < values.length; actorId++) {
            NativePolicyCombatBindingSpec spec =
                nativePolicyCombatSpecs[actorId];
            values[actorId] = spec == null ? 0 : spec.abilityMaskBits();
        }
        return joinActorInts(values);
    }

    public static String nativePolicyCombatGuardByActor(
        NativeInteractionBinding nativeGuardInteraction,
        Object[] nativePolicyCombatHandles,
        NativePolicyCombatBindingSpec[] nativePolicyCombatSpecs
    ) {
        boolean[] values = new boolean[nativePolicyCombatHandles.length];
        values[0] = nativeGuardInteraction != null;
        for (int actorId = 1; actorId < values.length; actorId++) {
            NativePolicyCombatBindingSpec spec =
                nativePolicyCombatSpecs[actorId];
            values[actorId] = spec != null && spec.hasGuard();
        }
        return joinActorBooleans(values);
    }

    public static String nativePolicyCombatDodgeByActor(
        NativeInteractionBinding nativeDodgeLeftInteraction,
        NativeInteractionBinding nativeDodgeRightInteraction,
        Object[] nativePolicyCombatHandles
    ) {
        boolean[] values = new boolean[nativePolicyCombatHandles.length];
        values[0] = nativeDodgeLeftInteraction != null
            && nativeDodgeRightInteraction != null;
        for (int actorId = 1; actorId < values.length; actorId++) {
            values[actorId] = nativePolicyCombatHandles[actorId] != null;
        }
        return joinActorBooleans(values);
    }

    public static String nativePolicyCombatItemIdsByActor(
        Object[] nativePolicyCombatHandles,
        NativePolicyCombatBindingSpec[] nativePolicyCombatSpecs,
        EnvironmentOptions options
    ) {
        String[] values = new String[nativePolicyCombatHandles.length];
        values[0] = options.hasNativeCombatItem()
            ? options.nativeCombatItemId()
            : "";
        for (int actorId = 1; actorId < values.length; actorId++) {
            NativePolicyCombatBindingSpec spec =
                nativePolicyCombatSpecs[actorId];
            values[actorId] = spec == null ? "" : spec.itemId();
        }
        return String.join(",", values);
    }

    private static String joinActorBooleans(
        boolean[] values
    ) {
        StringBuilder result = new StringBuilder();
        for (int index = 0; index < values.length; index++) {
            if (index > 0) result.append(',');
            result.append(values[index] ? '1' : '0');
        }
        return result.toString();
    }

    private static String joinActorInts(
        int[] values
    ) {
        StringBuilder result = new StringBuilder();
        for (int index = 0; index < values.length; index++) {
            if (index > 0) result.append(',');
            result.append(values[index]);
        }
        return result.toString();
    }
}
