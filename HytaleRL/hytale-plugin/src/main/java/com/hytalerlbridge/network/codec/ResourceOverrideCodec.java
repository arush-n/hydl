package com.hytalerlbridge.network.codec;

import static com.hytalerlbridge.network.wire.MessageFields.checkedInt;
import static com.hytalerlbridge.network.wire.MessageFields.getField;
import static com.hytalerlbridge.network.wire.MessageFields.getLongField;
import static com.hytalerlbridge.network.wire.MessageFields.getOptionalDoubleField;
import static com.hytalerlbridge.network.wire.MessageFields.getOptionalStringField;

import com.hytalerlbridge.environment.fidelity.NativeResourceOverride;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.msgpack.value.Value;

/** Decode fixture-only actor-resource setup from one reset request. */
public final class ResourceOverrideCodec {

    public static final String FIELD =
        "native_fidelity_resource_overrides";

    private ResourceOverrideCodec() {}

    public static List<NativeResourceOverride> decode(
        Map<Value, Value> options
    ) {
        Value value = getField(options, FIELD);
        if (value == null || value.isNilValue()) return null;
        if (!value.isArrayValue()) {
            throw new IllegalArgumentException(FIELD + " must be an array");
        }
        ArrayList<NativeResourceOverride> rows = new ArrayList<>(
            value.asArrayValue().size()
        );
        for (Value raw : value.asArrayValue()) {
            if (!raw.isMapValue()) {
                throw new IllegalArgumentException(
                    FIELD + " entries must be maps"
                );
            }
            Map<Value, Value> row = raw.asMapValue().map();
            Double resourceValue = getOptionalDoubleField(row, "value");
            if (resourceValue == null) {
                throw new IllegalArgumentException(
                    FIELD + " entries require value"
                );
            }
            rows.add(new NativeResourceOverride(
                checkedInt(
                    getLongField(row, "entity_id", -1L),
                    "entity_id"
                ),
                getOptionalStringField(row, "resource_id"),
                resourceValue.floatValue()
            ));
        }
        return List.copyOf(rows);
    }
}
