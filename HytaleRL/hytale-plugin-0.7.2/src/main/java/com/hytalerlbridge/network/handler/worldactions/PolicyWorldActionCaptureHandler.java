package com.hytalerlbridge.network.handler.worldactions;

import static com.hytalerlbridge.network.codec.worldactions.PolicyWorldActionCaptureRequestCodec.decode;
import static com.hytalerlbridge.network.codec.worldactions.PolicyWorldActionCaptureResponseCodec.send;

import com.hytalerlbridge.environment.EnvironmentManager;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;
import org.msgpack.value.Value;

/** Thin socket dispatch for the World-thread capture capability. */
public final class PolicyWorldActionCaptureHandler {

    private PolicyWorldActionCaptureHandler() {}

    public static void handle(
        Map<Value, Value> message,
        DataOutputStream output,
        EnvironmentManager environments,
        String environmentId
    ) throws IOException {
        if (environmentId == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
        send(
            output,
            environments.capturePolicyWorldActions(
                environmentId,
                decode(message)
            )
        );
    }
}
