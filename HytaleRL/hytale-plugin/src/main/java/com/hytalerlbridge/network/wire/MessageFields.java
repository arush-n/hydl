package com.hytalerlbridge.network.wire;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.msgpack.value.Value;

/**
 * Typed accessors for MessagePack request fields.
 *
 * <p>Extracted verbatim from {@code ClientHandler}. Every accessor resolves
 * through {@link #getField}, and none of them reach back into the network
 * package, so callers may depend on this without a cycle.
 */
public final class MessageFields {

    private MessageFields() {}

    public static Value getField(Map<Value, Value> map, String key) {
        for (Map.Entry<Value, Value> entry : map.entrySet()) {
            if (entry.getKey().isStringValue()
                && entry.getKey().asStringValue().asString().equals(key)) {
                return entry.getValue();
            }
        }
        return null;
    }

    public static String getStringField(Map<Value, Value> map, String key) {
        Value value = getField(map, key);
        return value != null && value.isStringValue()
            ? value.asStringValue().asString()
            : "";
    }

    public static String getOptionalStringField(
        Map<Value, Value> map,
        String key
    ) {
        Value value = getField(map, key);
        if (value == null || value.isNilValue()) return null;
        if (!value.isStringValue()) {
            throw new IllegalArgumentException(key + " must be a string");
        }
        return value.asStringValue().asString();
    }

    public static long getLongField(
        Map<Value, Value> map,
        String key,
        long defaultValue
    ) {
        Value value = getField(map, key);
        return value != null && value.isIntegerValue()
            ? value.asIntegerValue().asLong()
            : defaultValue;
    }

    public static Double getOptionalDoubleField(Map<Value, Value> map, String key) {
        Value value = getField(map, key);
        if (value == null || value.isNilValue()) return null;
        if (value.isFloatValue()) return value.asFloatValue().toDouble();
        if (value.isIntegerValue()) return (double) value.asIntegerValue().asLong();
        throw new IllegalArgumentException(key + " must be numeric");
    }

    public static boolean getBooleanField(
        Map<Value, Value> map,
        String key,
        boolean defaultValue
    ) {
        Value value = getField(map, key);
        if (value == null || value.isNilValue()) return defaultValue;
        if (value.isBooleanValue()) return value.asBooleanValue().getBoolean();
        if (value.isIntegerValue()) return value.asIntegerValue().asLong() != 0;
        throw new IllegalArgumentException(key + " must be boolean");
    }

    public static Map<Value, Value> getMapField(Map<Value, Value> map, String key) {
        Value value = getField(map, key);
        return value != null && value.isMapValue()
            ? value.asMapValue().map()
            : Map.of();
    }

    public static List<String> getOptionalStringListField(
        Map<Value, Value> map,
        String key
    ) {
        Value value = getField(map, key);
        if (value == null || value.isNilValue()) return null;
        if (!value.isArrayValue()) {
            throw new IllegalArgumentException(key + " must be an array of strings");
        }
        List<String> values = new ArrayList<>(
            value.asArrayValue().size()
        );
        for (Value element : value.asArrayValue()) {
            if (!element.isStringValue()) {
                throw new IllegalArgumentException(
                    key + " must be an array of strings"
                );
            }
            values.add(element.asStringValue().asString());
        }
        return List.copyOf(values);
    }

    public static List<Integer> getOptionalIntListField(
        Map<Value, Value> map,
        String key
    ) {
        Value value = getField(map, key);
        if (value == null || value.isNilValue()) return null;
        if (!value.isArrayValue()) {
            throw new IllegalArgumentException(key + " must be an array of integers");
        }
        List<Integer> values = new ArrayList<>(
            value.asArrayValue().size()
        );
        for (Value element : value.asArrayValue()) {
            if (!element.isIntegerValue()) {
                throw new IllegalArgumentException(
                    key + " must be an array of integers"
                );
            }
            values.add(checkedInt(element.asIntegerValue().asLong(), key));
        }
        return List.copyOf(values);
    }

    public static byte[] getBinaryField(Map<Value, Value> map, String key) {
        Value value = getField(map, key);
        if (value == null || !value.isBinaryValue()) {
            throw new IllegalArgumentException(key + " must be binary");
        }
        return value.asBinaryValue().asByteArray();
    }

    public static int checkedInt(long value, String name) {
        if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(name + " is outside integer range");
        }
        return (int) value;
    }
}
