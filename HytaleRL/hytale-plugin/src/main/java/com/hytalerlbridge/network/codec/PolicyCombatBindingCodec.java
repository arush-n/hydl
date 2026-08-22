package com.hytalerlbridge.network.codec;

import static com.hytalerlbridge.network.wire.MessageFields.checkedInt;
import static com.hytalerlbridge.network.wire.MessageFields.getField;
import static com.hytalerlbridge.network.wire.MessageFields.getLongField;
import static com.hytalerlbridge.network.wire.MessageFields.getOptionalIntListField;
import static com.hytalerlbridge.network.wire.MessageFields.getOptionalStringField;
import static com.hytalerlbridge.network.wire.MessageFields.getOptionalStringListField;

import com.hytalerlbridge.action.group.NativePolicyCombatBindingSpec;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.msgpack.value.Value;

/** Decode actor-major combat bindings from one reset request. */
public final class PolicyCombatBindingCodec {

    public static final String FIELD =
        "native_group_policy_combat_bindings";

    private PolicyCombatBindingCodec() {}

    public static List<NativePolicyCombatBindingSpec> decode(
        Map<Value, Value> options
    ) {
        Value value = getField(options, FIELD);
        if (value == null || value.isNilValue()) return null;
        if (!value.isArrayValue()) {
            throw new IllegalArgumentException(FIELD + " must be an array");
        }
        ArrayList<NativePolicyCombatBindingSpec> rows = new ArrayList<>(
            value.asArrayValue().size()
        );
        for (Value raw : value.asArrayValue()) {
            if (!raw.isMapValue()) {
                throw new IllegalArgumentException(
                    FIELD + " entries must be maps"
                );
            }
            Map<Value, Value> row = raw.asMapValue().map();
            List<Integer> slots = getOptionalIntListField(
                row,
                "ability_slots"
            );
            List<String> ids = getOptionalStringListField(
                row,
                "ability_interaction_ids"
            );
            List<String> types = getOptionalStringListField(
                row,
                "ability_interaction_types"
            );
            rows.add(new NativePolicyCombatBindingSpec(
                checkedInt(
                    getLongField(row, "entity_id", -1L),
                    "entity_id"
                ),
                getOptionalStringField(row, "item_id"),
                slots == null ? List.of() : slots,
                ids == null ? List.of() : ids,
                types == null ? List.of() : types,
                getOptionalStringField(row, "guard_interaction_id"),
                getOptionalStringField(row, "guard_interaction_type")
            ));
        }
        return List.copyOf(rows);
    }
}
