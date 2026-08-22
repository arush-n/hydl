package com.hytalerlbridge.worldgen.policyactions.capture.model;

import java.util.List;

final class CaptureValidation {

    private CaptureValidation() {}

    static String text(String value, String name) {
        if (value == null || value.isBlank() || !value.equals(value.trim())) {
            throw new IllegalArgumentException(name + " cannot be blank");
        }
        return value;
    }

    static String optionalText(String value) {
        if (value == null) return "";
        if (!value.equals(value.trim())) {
            throw new IllegalArgumentException("optional text cannot have edge whitespace");
        }
        return value;
    }

    static String sha256(String value, String name) {
        if (value == null || !value.matches("[0-9A-Fa-f]{64}")) {
            throw new IllegalArgumentException(name + " must be SHA-256");
        }
        return value.toUpperCase();
    }

    static String optionalSha256(String value, String name) {
        if (value == null || value.isEmpty()) return "";
        return sha256(value, name);
    }

    static String choice(String value, String name, List<String> choices) {
        value = text(value, name);
        if (!choices.contains(value)) {
            throw new IllegalArgumentException(name + " is unsupported");
        }
        return value;
    }

    static int[] ints(int[] value) {
        return value == null ? new int[0] : value.clone();
    }

    static double[] doubles(double[] value) {
        return value == null ? new double[0] : value.clone();
    }

    static byte[] bytes(byte[] value) {
        return value == null ? new byte[0] : value.clone();
    }

    static boolean finite(double[] values) {
        for (double value : values) {
            if (!Double.isFinite(value)) return false;
        }
        return true;
    }

    static boolean anyNonzero(int[] values) {
        for (int value : values) {
            if (value != 0) return true;
        }
        return false;
    }

    static boolean anyOutside(int[] values, int minimum, int maximum) {
        for (int value : values) {
            if (value < minimum || value > maximum) return true;
        }
        return false;
    }
}
