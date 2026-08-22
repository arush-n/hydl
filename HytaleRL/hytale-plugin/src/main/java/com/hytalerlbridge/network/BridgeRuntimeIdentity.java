package com.hytalerlbridge.network;

import java.io.IOException;
import java.io.InputStream;
import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.net.URISyntaxException;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Optional;

/** Process identity reported by the bridge that actually accepted the client. */
public final class BridgeRuntimeIdentity {

    private static final RuntimeMXBean RUNTIME = ManagementFactory.getRuntimeMXBean();

    private BridgeRuntimeIdentity() {
    }

    public static Optional<String> runtimeJarSha256(Class<?> anchor) {
        try {
            var source = anchor.getProtectionDomain().getCodeSource();
            if (source == null) {
                return Optional.empty();
            }
            Path path = Path.of(source.getLocation().toURI());
            if (!Files.isRegularFile(path)) {
                return Optional.empty();
            }
            return Optional.of(fileSha256(path));
        } catch (
            IOException
                | URISyntaxException
                | FileSystemNotFoundException
                | IllegalArgumentException
                | SecurityException exception
        ) {
            return Optional.empty();
        }
    }

    public static String fileSha256(Path path) throws IOException {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is unavailable", exception);
        }
        try (InputStream input = Files.newInputStream(path)) {
            byte[] buffer = new byte[64 * 1024];
            int count;
            while ((count = input.read(buffer)) >= 0) {
                digest.update(buffer, 0, count);
            }
        }
        return HexFormat.of().withUpperCase().formatHex(digest.digest());
    }

    public static double processUptimeSeconds() {
        return RUNTIME.getUptime() / 1000.0;
    }
}
