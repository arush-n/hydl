package com.hytalerlbridge.network.handler;

import com.hytalerlbridge.environment.EnvironmentManager;
import com.hytalerlbridge.worldgen.NativeExplosionMutationProbe;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;
import org.msgpack.value.Value;
import static com.hytalerlbridge.network.codec.ExplosionMutationCodec.sendExplosionMutationProbe;
import static com.hytalerlbridge.network.wire.MessageFields.checkedInt;
import static com.hytalerlbridge.network.wire.MessageFields.getLongField;
import static com.hytalerlbridge.network.wire.MessageFields.getStringField;

/** Parses the bounded, server-owned explosion-mutation fixture request. */
public final class ExplosionMutationHandler {

    private ExplosionMutationHandler() {}

    public static void handleExplosionMutation(
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
        String schema = getStringField(message, "schema");
        if (!NativeExplosionMutationProbe.SCHEMA.equals(schema)) {
            throw new IllegalArgumentException(
                "Unsupported explosion mutation schema: " + schema
            );
        }
        int version = checkedInt(
            getLongField(message, "version", Long.MIN_VALUE),
            "version"
        );
        if (version != NativeExplosionMutationProbe.VERSION) {
            throw new IllegalArgumentException(
                "Unsupported explosion mutation version: " + version
            );
        }
        sendExplosionMutationProbe(
            output,
            environments.captureExplosionMutation(
                environmentId,
                getStringField(message, "world_epoch"),
                getStringField(message, "fixture_kind"),
                checkedInt(
                    getLongField(
                        message,
                        "cell_capacity",
                        Long.MIN_VALUE
                    ),
                    "cell_capacity"
                ),
                checkedInt(
                    getLongField(
                        message,
                        "drop_capacity",
                        Long.MIN_VALUE
                    ),
                    "drop_capacity"
                )
            )
        );
    }
}
