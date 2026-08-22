package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.server.core.inventory.ItemStack;
import com.hypixel.hytale.server.core.universe.world.World;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * Extracted verbatim from {@code NativeEnvironmentSession}.
 *
 * <p>Every method here touches no session instance field, so the move
 * required no state surgery.
 */
public final class NativeKeys {

    private NativeKeys() {}

    public static String sha256Hex(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(
                digest.digest(value.getBytes(StandardCharsets.UTF_8))
            );
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException(
                "SHA-256 is unavailable",
                exception
            );
        }
    }

    public static byte[] mutableBlockSemanticKey(
        String blockAssetId,
        int rotationIndex
    ) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(blockAssetId.getBytes(StandardCharsets.UTF_8));
            digest.update((byte) 0);
            digest.update(Integer.toString(rotationIndex).getBytes(
                StandardCharsets.US_ASCII
            ));
            return digest.digest();
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
    }

    public static byte[] mutableBlockAssetKey(String blockAssetId) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(blockAssetId.getBytes(StandardCharsets.UTF_8));
            return digest.digest();
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
    }

    public static String canonicalWorldVerbInventoryState(
        ItemStack stack
    ) {
        if (ItemStack.isEmpty(stack)) return "empty";
        var metadata = stack.getMetadata();
        String metadataSha256 = metadata == null || metadata.isEmpty()
            ? ""
            : sha256Hex(metadata.toJson());
        return String.join(
            "|",
            stack.getItemId(),
            Integer.toString(stack.getQuantity()),
            Double.toHexString(stack.getDurability()),
            Double.toHexString(stack.getMaxDurability()),
            metadataSha256
        );
    }

    public static String implementationVersion() {
        Package serverPackage = World.class.getPackage();
        String version = serverPackage == null
            ? null
            : serverPackage.getImplementationVersion();
        return version == null || version.isBlank() ? "unknown" : version;
    }
}
