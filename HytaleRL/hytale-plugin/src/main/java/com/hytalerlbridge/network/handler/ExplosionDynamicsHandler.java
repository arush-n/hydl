package com.hytalerlbridge.network.handler;

import com.hytalerlbridge.environment.EnvironmentManager;
import com.hytalerlbridge.worldgen.NativeExplosionDynamicsProbe;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;
import org.msgpack.value.Value;
import static com.hytalerlbridge.network.codec.ExplosionDynamicsCodec.sendExplosionDynamicsProbe;
import static com.hytalerlbridge.network.wire.MessageFields.checkedInt;
import static com.hytalerlbridge.network.wire.MessageFields.getLongField;
import static com.hytalerlbridge.network.wire.MessageFields.getStringField;

/** Parses the bounded terminal explosion-dynamics fixture request. */
public final class ExplosionDynamicsHandler {

    private ExplosionDynamicsHandler() {}

    public static void handleExplosionDynamics(
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
        if (
            !NativeExplosionDynamicsProbe.SCHEMA.equals(
                getStringField(message, "schema")
            )
        ) {
            throw new IllegalArgumentException(
                "Unsupported explosion dynamics schema"
            );
        }
        int version = checkedInt(
            getLongField(message, "version", Long.MIN_VALUE),
            "version"
        );
        if (version != NativeExplosionDynamicsProbe.VERSION) {
            throw new IllegalArgumentException(
                "Unsupported explosion dynamics version: " + version
            );
        }
        sendExplosionDynamicsProbe(
            output,
            environments.captureExplosionDynamics(
                environmentId,
                getStringField(message, "world_epoch"),
                getStringField(message, "fixture_kind"),
                checkedInt(
                    getLongField(message, "capacity", Long.MIN_VALUE),
                    "capacity"
                )
            )
        );
    }
}
