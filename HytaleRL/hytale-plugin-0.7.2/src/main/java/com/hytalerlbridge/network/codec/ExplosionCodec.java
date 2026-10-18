package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.worldgen.NativeExplosionCandidateProbe;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.UUID;
import org.msgpack.core.MessagePacker;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeFloat64LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.packBinary;
import static com.hytalerlbridge.network.wire.BridgeIdentity.BRIDGE_SHA256;
import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;

/** Wire encoding for the native entity-only explosion admission probe. */
public final class ExplosionCodec {

    private ExplosionCodec() {}

    public static void sendExplosionCandidateProbe(
        DataOutputStream output,
        NativeExplosionCandidateProbe probe
    ) throws IOException {
        String bridgeSha256 = BRIDGE_SHA256.orElseThrow(
            () -> new IllegalStateException("Runtime bridge identity is unavailable")
        );
        sendFrame(
            output,
            packer -> packExplosionCandidateProbe(
                packer,
                probe,
                bridgeSha256
            )
        );
    }

    public static void packExplosionCandidateProbe(
        MessagePacker packer,
        NativeExplosionCandidateProbe probe,
        String bridgeSha256
    ) throws IOException {
        if (bridgeSha256 == null || !bridgeSha256.matches("[0-9A-Fa-f]{64}")) {
            throw new IllegalArgumentException(
                "Runtime bridge identity must be a SHA-256"
            );
        }
        int count = probe.emittedCount();
        byte[] identities = new byte[count * 16];
        double[] positions = new double[count * 3];
        for (int index = 0; index < count; index++) {
            NativeExplosionCandidateProbe.Candidate candidate =
                probe.candidates().get(index);
            byte[] uuid = uuidBytes(candidate.uuid());
            System.arraycopy(uuid, 0, identities, index * 16, 16);
            int offset = index * 3;
            positions[offset] = candidate.x();
            positions[offset + 1] = candidate.y();
            positions[offset + 2] = candidate.z();
        }

        packer.packMapHeader(22);
        packer.packString("type");
        packer.packString("explosion_candidate_probe");
        packer.packString("schema");
        packer.packString(NativeExplosionCandidateProbe.SCHEMA);
        packer.packString("version");
        packer.packInt(NativeExplosionCandidateProbe.VERSION);
        packer.packString("bridge_sha256");
        packer.packString(bridgeSha256);
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
        packBinary(
            packer,
            "origin_f64_le_xyz",
            encodeFloat64LittleEndian(probe.origin())
        );
        packer.packString("block_damage_radius");
        packer.packInt(probe.blockDamageRadius());
        packer.packString("entity_damage_radius");
        packer.packFloat(probe.entityDamageRadius());
        packer.packString("damage_blocks");
        packer.packBoolean(false);
        packer.packString("ignore_controlled_actor");
        packer.packBoolean(probe.ignoredControlledActor());
        packer.packString("capacity");
        packer.packInt(probe.capacity());
        packer.packString("total_matching");
        packer.packInt(probe.totalMatching());
        packer.packString("emitted_count");
        packer.packInt(count);
        packer.packString("overflow");
        packer.packBoolean(probe.overflow());
        packer.packString("uuid_encoding");
        packer.packString("rfc4122_network_order_16_bytes_per_row");
        packBinary(packer, "uuid_bytes", identities);
        packBinary(
            packer,
            "positions_f64_le_xyz",
            encodeFloat64LittleEndian(positions)
        );
        packer.packString("native_method");
        packer.packString("ExplosionUtils.processTargetBlocks");
    }

    private static byte[] uuidBytes(UUID uuid) {
        return ByteBuffer.allocate(16)
            .order(ByteOrder.BIG_ENDIAN)
            .putLong(uuid.getMostSignificantBits())
            .putLong(uuid.getLeastSignificantBits())
            .array();
    }
}
