package com.hytalerlbridge.network.wire;

import com.hytalerlbridge.network.BridgeRuntimeIdentity;
import java.util.Optional;

/**
 * The deployed jar's identity, resolved once.
 *
 * <p>Lifted out of {@code ClientHandler} because both the handler and the
 * extracted codecs stamp it into outgoing frames. Resolving it here keeps the
 * codecs from depending back on {@code ClientHandler}.
 */
public final class BridgeIdentity {

    /** SHA-256 of the running bridge jar, empty when it cannot be resolved. */
    public static final Optional<String> BRIDGE_SHA256 =
        BridgeRuntimeIdentity.runtimeJarSha256(BridgeIdentity.class);

    private BridgeIdentity() {}
}
