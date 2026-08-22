package com.hytalerlbridge.worldgen.policyactions.capture.semantic;

import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/** Minimal RFC-8259 writer matching Python's compact sorted-key JSON. */
public final class CanonicalJson {

    private CanonicalJson() {}

    public static String sha256(Object value) {
        try {
            return HexFormat.of().withUpperCase().formatHex(
                MessageDigest.getInstance("SHA-256").digest(
                    encode(value).getBytes(StandardCharsets.UTF_8)
                )
            );
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
    }

    public static String encode(Object value) {
        StringBuilder result = new StringBuilder();
        append(result, value);
        return result.toString();
    }

    private static void append(StringBuilder result, Object value) {
        if (value == null) {
            result.append("null");
        } else if (value instanceof String string) {
            appendString(result, string);
        } else if (value instanceof Boolean bool) {
            result.append(bool ? "true" : "false");
        } else if (
            value instanceof Byte
                || value instanceof Short
                || value instanceof Integer
                || value instanceof Long
        ) {
            result.append(value);
        } else if (value instanceof Float number) {
            appendFloating(result, (double) number);
        } else if (value instanceof Double number) {
            appendFloating(result, number);
        } else if (value instanceof Map<?, ?> map) {
            TreeMap<String, Object> sorted = new TreeMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (!(entry.getKey() instanceof String key)) {
                    throw new IllegalArgumentException(
                        "Canonical JSON map keys must be strings"
                    );
                }
                sorted.put(key, entry.getValue());
            }
            result.append('{');
            boolean first = true;
            for (Map.Entry<String, Object> entry : sorted.entrySet()) {
                if (!first) result.append(',');
                first = false;
                appendString(result, entry.getKey());
                result.append(':');
                append(result, entry.getValue());
            }
            result.append('}');
        } else if (value instanceof Iterable<?> iterable) {
            result.append('[');
            boolean first = true;
            for (Object item : iterable) {
                if (!first) result.append(',');
                first = false;
                append(result, item);
            }
            result.append(']');
        } else if (value.getClass().isArray()) {
            result.append('[');
            for (int index = 0; index < Array.getLength(value); index++) {
                if (index > 0) result.append(',');
                append(result, Array.get(value, index));
            }
            result.append(']');
        } else {
            throw new IllegalArgumentException(
                "Unsupported canonical JSON value: "
                    + value.getClass().getName()
            );
        }
    }

    private static void appendString(StringBuilder result, String value) {
        result.append('"');
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            switch (character) {
                case '"' -> result.append("\\\"");
                case '\\' -> result.append("\\\\");
                case '\b' -> result.append("\\b");
                case '\f' -> result.append("\\f");
                case '\n' -> result.append("\\n");
                case '\r' -> result.append("\\r");
                case '\t' -> result.append("\\t");
                default -> {
                    if (character < 0x20 || character > 0x7e) {
                        result.append("\\u");
                        String hex = Integer.toHexString(character);
                        result.append("0".repeat(4 - hex.length()));
                        result.append(hex);
                    } else {
                        result.append(character);
                    }
                }
            }
        }
        result.append('"');
    }

    private static void appendFloating(StringBuilder result, double value) {
        result.append(pythonFloating(value));
    }

    static String pythonFloating(double value) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                "Canonical JSON numbers must be finite"
            );
        }
        String java = Double.toString(value);
        int marker = java.indexOf('E');
        if (marker < 0) return java;
        String mantissa = java.substring(0, marker);
        int exponent = Integer.parseInt(java.substring(marker + 1));
        if (exponent >= -4 && exponent < 16) {
            String fixed = BigDecimal.valueOf(value).toPlainString();
            return fixed.indexOf('.') < 0 ? fixed + ".0" : fixed;
        }
        if (mantissa.endsWith(".0")) {
            mantissa = mantissa.substring(0, mantissa.length() - 2);
        }
        String sign = exponent >= 0 ? "+" : "-";
        String digits = Integer.toString(Math.abs(exponent));
        if (digits.length() < 2) digits = "0" + digits;
        return mantissa + "e" + sign + digits;
    }
}
