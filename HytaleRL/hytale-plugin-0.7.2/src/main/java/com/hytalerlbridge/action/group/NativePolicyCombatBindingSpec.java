package com.hytalerlbridge.action.group;

import com.hytalerlbridge.observation.NativeActorEvidenceFrame;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Reset-pinned item-root bindings for one policy-controlled native actor.
 *
 * <p>The slot numbers are transport-local compact slots, not authored JAX
 * ability indices.  The host keeps the authored-to-native mapping and sends
 * only these slots on the wire.  No weapon family or interaction ID is
 * interpreted here.</p>
 */
public record NativePolicyCombatBindingSpec(
    int entityId,
    String itemId,
    List<Integer> abilitySlots,
    List<String> abilityInteractionIds,
    List<String> abilityInteractionTypes,
    String guardInteractionId,
    String guardInteractionType
) {

    public static final String SCHEMA =
        "hytalerl_native_policy_combat_binding_v1";
    public static final int VERSION = 1;
    public static final int ABILITY_CAPACITY = 16;
    private static final Set<String> INTERACTION_TYPES = Set.of(
        "Primary",
        "Secondary",
        "Ability1",
        "Ability2",
        "Ability3"
    );

    public NativePolicyCombatBindingSpec {
        if (entityId < 0 || entityId >= NativeActorEvidenceFrame.ENTITY_COUNT) {
            throw new IllegalArgumentException(
                "native policy-combat entity_id is outside actor capacity"
            );
        }
        itemId = required(itemId, "item_id");
        abilitySlots = copyIntegers(abilitySlots, "ability_slots");
        abilityInteractionIds = copyStrings(
            abilityInteractionIds,
            "ability_interaction_ids"
        );
        abilityInteractionTypes = copyStrings(
            abilityInteractionTypes,
            "ability_interaction_types"
        );
        if (
            abilitySlots.size() != abilityInteractionIds.size()
                || abilitySlots.size() != abilityInteractionTypes.size()
        ) {
            throw new IllegalArgumentException(
                "native policy-combat ability rows must have equal widths"
            );
        }
        if (abilitySlots.size() > ABILITY_CAPACITY) {
            throw new IllegalArgumentException(
                "native policy-combat ability rows exceed capacity"
            );
        }
        HashSet<Integer> seen = new HashSet<>();
        for (int index = 0; index < abilitySlots.size(); index++) {
            int slot = abilitySlots.get(index);
            if (slot < 0 || slot >= ABILITY_CAPACITY || !seen.add(slot)) {
                throw new IllegalArgumentException(
                    "native policy-combat ability slots must be unique and in range"
                );
            }
            requireInteractionType(
                abilityInteractionTypes.get(index),
                "ability_interaction_types"
            );
        }

        guardInteractionId = optional(guardInteractionId);
        guardInteractionType = optional(guardInteractionType);
        if ((guardInteractionId == null) != (guardInteractionType == null)) {
            throw new IllegalArgumentException(
                "native policy-combat guard id and type must be supplied together"
            );
        }
        if (guardInteractionType != null) {
            requireInteractionType(
                guardInteractionType,
                "guard_interaction_type"
            );
        }
        if (abilitySlots.isEmpty() && guardInteractionId == null) {
            throw new IllegalArgumentException(
                "native policy-combat binding must expose an ability or guard"
            );
        }
    }

    public boolean hasGuard() {
        return guardInteractionId != null;
    }

    public int abilityMaskBits() {
        int bits = 0;
        for (int slot : abilitySlots) bits |= 1 << slot;
        return bits;
    }

    private static List<Integer> copyIntegers(
        List<Integer> values,
        String field
    ) {
        if (values == null) return List.of();
        ArrayList<Integer> copy = new ArrayList<>(values.size());
        for (Integer value : values) {
            if (value == null) {
                throw new IllegalArgumentException(
                    "native policy-combat " + field + " contains null"
                );
            }
            copy.add(value);
        }
        return List.copyOf(copy);
    }

    private static List<String> copyStrings(
        List<String> values,
        String field
    ) {
        if (values == null) return List.of();
        ArrayList<String> copy = new ArrayList<>(values.size());
        for (String value : values) copy.add(required(value, field));
        return List.copyOf(copy);
    }

    private static String required(String value, String field) {
        String normalized = optional(value);
        if (normalized == null) {
            throw new IllegalArgumentException(
                "native policy-combat " + field + " must be nonblank"
            );
        }
        return normalized;
    }

    private static String optional(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }

    private static void requireInteractionType(String value, String field) {
        if (!INTERACTION_TYPES.contains(value)) {
            throw new IllegalArgumentException(
                "unknown native policy-combat " + field + ": " + value
            );
        }
    }
}
