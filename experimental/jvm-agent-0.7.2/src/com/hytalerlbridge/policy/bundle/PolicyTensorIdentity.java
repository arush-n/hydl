package com.hytalerlbridge.policy.bundle;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;

/**
 * Content identity of the exact raw policy tensors consumed by Java.
 *
 * <p>The checkpoint digest names the source container, not the eleven
 * {@code .bin} files installed beside the mod.  This verifier hashes those raw
 * files themselves before the plugin arms.  Its byte format is implemented
 * independently by {@code adk.deploy.tensor_identity}; neither side trusts a
 * digest copied from the other.
 *
 * <p>The canonical stream is the ASCII domain (including its trailing NUL),
 * unsigned big-endian format version and tensor count, followed by each tensor
 * in the order below: UTF-8 name length, name, unsigned big-endian byte length,
 * then exact file bytes.  Length prefixes make names and contents
 * unambiguous.  Filesystem enumeration order is never part of the contract.
 */
public final class PolicyTensorIdentity {

    public static final String SCHEMA = "hytalerl_policy_tensor_content_v1";

    /** The model-layout order read by {@code Policy}; not a loadout catalogue. */
    public static final List<String> TENSOR_NAMES = List.of(
        "encoder_input_kernel",
        "encoder_input_bias",
        "encoder_hidden_kernel",
        "encoder_hidden_bias",
        "gru_input_kernel",
        "gru_recurrent_kernel",
        "gru_bias",
        "actor_kernel",
        "actor_bias",
        "critic_kernel",
        "critic_bias"
    );

    private static final byte[] DOMAIN =
        "HYTALERL_POLICY_TENSOR_CONTENT\0".getBytes(StandardCharsets.US_ASCII);
    private static final int FORMAT_VERSION = 1;

    private PolicyTensorIdentity() {
    }

    /** Hash every required tensor from {@code directory}. */
    public static String sha256(Path directory) throws IOException {
        MessageDigest digest = newDigest();
        begin(digest, TENSOR_NAMES);
        byte[] buffer = new byte[1 << 16];
        for (String name : TENSOR_NAMES) {
            byte[] encoded = name.getBytes(StandardCharsets.UTF_8);
            updateUnsignedInt(digest, encoded.length);
            digest.update(encoded);

            Path path = directory.resolve(name + ".bin");
            if (!Files.isRegularFile(path)) {
                throw new IOException("missing policy tensor " + path);
            }
            updateLong(digest, Files.size(path));
            try (InputStream input = Files.newInputStream(path)) {
                int count;
                while ((count = input.read(buffer)) != -1) {
                    if (count > 0) {
                        digest.update(buffer, 0, count);
                    }
                }
            }
        }
        return HexFormat.of().withUpperCase().formatHex(digest.digest());
    }

    /**
     * Canonical in-memory form used by the dependency-free conformance gate.
     * Package-private so production callers cannot substitute an arbitrary
     * tensor list for the exact installed model layout.
     */
    static String sha256(List<String> names, List<byte[]> contents) {
        if (names.size() != contents.size()) {
            throw new IllegalArgumentException("tensor names/content count differs");
        }
        MessageDigest digest = newDigest();
        begin(digest, names);
        for (int index = 0; index < names.size(); index++) {
            byte[] encoded = names.get(index).getBytes(StandardCharsets.UTF_8);
            byte[] content = contents.get(index);
            if (content == null) {
                throw new IllegalArgumentException("tensor content cannot be null");
            }
            updateUnsignedInt(digest, encoded.length);
            digest.update(encoded);
            updateLong(digest, content.length);
            digest.update(content);
        }
        return HexFormat.of().withUpperCase().formatHex(digest.digest());
    }

    private static void begin(MessageDigest digest, List<String> names) {
        if (names.isEmpty()) {
            throw new IllegalArgumentException("policy tensor list cannot be empty");
        }
        Set<String> unique = new HashSet<>();
        for (String name : names) {
            if (name == null || name.isEmpty()) {
                throw new IllegalArgumentException(
                    "policy tensor names must be nonempty");
            }
            if (!unique.add(name)) {
                throw new IllegalArgumentException(
                    "duplicate policy tensor name " + name);
            }
        }
        digest.update(DOMAIN);
        updateUnsignedInt(digest, FORMAT_VERSION);
        updateUnsignedInt(digest, names.size());
    }

    private static MessageDigest newDigest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("JVM has no SHA-256", impossible);
        }
    }

    private static void updateUnsignedInt(MessageDigest digest, int value) {
        if (value < 0) {
            throw new IllegalArgumentException("negative canonical length");
        }
        digest.update((byte) (value >>> 24));
        digest.update((byte) (value >>> 16));
        digest.update((byte) (value >>> 8));
        digest.update((byte) value);
    }

    private static void updateLong(MessageDigest digest, long value) {
        if (value < 0) {
            throw new IllegalArgumentException("negative canonical length");
        }
        for (int shift = 56; shift >= 0; shift -= 8) {
            digest.update((byte) (value >>> shift));
        }
    }
}
