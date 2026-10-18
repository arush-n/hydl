package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.worldgen.NativeExplosionDynamicsProbe;
import java.io.DataOutputStream;
import java.io.IOException;
import org.msgpack.core.MessagePacker;
import static com.hytalerlbridge.network.wire.BridgeIdentity.BRIDGE_SHA256;
import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;

/** Wire encoding for the terminal native explosion-damage fixture. */
public final class ExplosionDynamicsCodec {

    private ExplosionDynamicsCodec() {}

    public static void sendExplosionDynamicsProbe(
        DataOutputStream output,
        NativeExplosionDynamicsProbe probe
    ) throws IOException {
        String bridgeSha256 = BRIDGE_SHA256.orElseThrow(
            () -> new IllegalStateException(
                "Runtime bridge identity is unavailable"
            )
        );
        sendFrame(
            output,
            packer -> packExplosionDynamicsProbe(
                packer,
                probe,
                bridgeSha256
            )
        );
    }

    public static void packExplosionDynamicsProbe(
        MessagePacker packer,
        NativeExplosionDynamicsProbe probe,
        String bridgeSha256
    ) throws IOException {
        if (
            bridgeSha256 == null
                || !bridgeSha256.matches("[0-9A-Fa-f]{64}")
        ) {
            throw new IllegalArgumentException(
                "Runtime bridge identity must be a SHA-256"
            );
        }
        packer.packMapHeader(28);
        packer.packString("type");
        packer.packString("explosion_dynamics_probe");
        packer.packString("schema");
        packer.packString(NativeExplosionDynamicsProbe.SCHEMA);
        packer.packString("version");
        packer.packInt(NativeExplosionDynamicsProbe.VERSION);
        packer.packString("bridge_sha256");
        packer.packString(bridgeSha256.toUpperCase());
        packer.packString("server_version");
        packer.packString(probe.serverVersion());
        packer.packString("world");
        packer.packString(probe.worldName());
        packer.packString("worldgen_provider");
        packer.packString(probe.worldgenProvider());
        packer.packString("worldgen_version");
        packer.packString(probe.worldgenVersion());
        packer.packString("seed");
        packer.packLong(probe.seed());
        packer.packString("world_epoch");
        packer.packString(probe.worldEpoch());
        packer.packString("fixture_kind");
        packer.packString(NativeExplosionDynamicsProbe.FIXTURE_KIND);
        packer.packString("origin_f64_xyz");
        packer.packArrayHeader(3);
        packer.packDouble(probe.originX());
        packer.packDouble(probe.originY());
        packer.packDouble(probe.originZ());
        packer.packString("damage_blocks");
        packer.packBoolean(false);
        packer.packString("damage_entities");
        packer.packBoolean(true);
        packer.packString("entity_damage_radius");
        packer.packFloat(probe.entityDamageRadius());
        packer.packString("entity_damage");
        packer.packFloat(probe.entityDamage());
        packer.packString("entity_damage_falloff");
        packer.packFloat(probe.entityDamageFalloff());
        packer.packString("knockback");
        packer.packBoolean(false);
        packer.packString("capacity");
        packer.packInt(probe.capacity());
        packer.packString("total_matching");
        packer.packInt(probe.totalMatching());
        packer.packString("overflow");
        packer.packBoolean(probe.overflow());
        packer.packString("complete");
        packer.packBoolean(probe.complete());
        packer.packString("session_reset_required");
        packer.packBoolean(probe.sessionResetRequired());
        packer.packString("failure_reason");
        packer.packString(probe.failureReason());
        packer.packString("phase_order");
        packer.packString(NativeExplosionDynamicsProbe.PHASE_ORDER);
        packer.packString("row_order");
        packer.packString(NativeExplosionDynamicsProbe.ROW_ORDER);
        packer.packString("native_method");
        packer.packString("ExplosionUtils.performExplosion");
        packer.packString("effects");
        packer.packArrayHeader(probe.effects().size());
        for (NativeExplosionDynamicsProbe.EntityEffect effect
            : probe.effects()) {
            packer.packMapHeader(9);
            packer.packString("ordinal");
            packer.packInt(effect.ordinal());
            packer.packString("uuid");
            packer.packString(effect.uuid().toString());
            packer.packString("position_f64_xyz");
            packer.packArrayHeader(3);
            packer.packDouble(effect.x());
            packer.packDouble(effect.y());
            packer.packDouble(effect.z());
            packer.packString("distance");
            packer.packDouble(effect.distance());
            packer.packString("health_before");
            packer.packFloat(effect.healthBefore());
            packer.packString("health_after");
            packer.packFloat(effect.healthAfter());
            packer.packString("expected_raw_damage");
            packer.packFloat(effect.expectedRawDamage());
            packer.packString("applied_health_delta");
            packer.packFloat(effect.healthBefore() - effect.healthAfter());
            packer.packString("damage_cause");
            packer.packString("ENVIRONMENT");
        }
    }
}
