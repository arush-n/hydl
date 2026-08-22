package com.hytalerlbridge.geometry.trigger;

import com.hytalerlbridge.geometry.GeometryContract;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;

/** Fixed capacities and identities for native collision-trigger bindings. */
public final class TriggerContract {

    public static final String SCHEMA = "hytale_trigger_transport_v1";
    public static final int VERSION = 1;
    public static final int NO_PROGRAM = -1;
    public static final int MAX_CELLS = GeometryContract.CELL_COUNT;

    /** 47 authored 0.5.7 environmental phase roots fit this JAX-safe bound. */
    public static final int MAX_PROGRAMS_0_5_7 = 64;

    public static final int MAX_BOXES_0_5_7 =
        GeometryContract.MAX_DETAIL_BOXES_0_5_7;

    private TriggerContract() {}

    static String catalogSha256(List<String> programIds) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            for (String id : programIds) {
                byte[] encoded = id.getBytes(StandardCharsets.UTF_8);
                digest.update((byte) (encoded.length >>> 24));
                digest.update((byte) (encoded.length >>> 16));
                digest.update((byte) (encoded.length >>> 8));
                digest.update((byte) encoded.length);
                digest.update(encoded);
            }
            return HexFormat.of().withUpperCase().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
    }
}
