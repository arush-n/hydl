package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.worldgen.WorldgenStructureMarkerQuery;
import com.hytalerlbridge.worldgen.WorldgenStructureMarkerRow;
import com.hytalerlbridge.worldgen.WorldgenStructureMarkerSnapshot;
import java.io.DataOutputStream;
import java.io.IOException;
import org.msgpack.core.MessagePacker;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeFloat64LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeInt32LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.packBinary;
import static com.hytalerlbridge.network.wire.BridgeIdentity.BRIDGE_SHA256;
import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;

/** Columnar wire codec for authored WorldGen V2 structure markers. */
public final class WorldgenStructureMarkerCodec {

    private WorldgenStructureMarkerCodec() {}

    public static void sendWorldgenStructureMarkers(
        DataOutputStream output,
        WorldgenStructureMarkerSnapshot snapshot
    ) throws IOException {
        String bridgeSha256 = BRIDGE_SHA256.orElseThrow(
            () -> new IllegalStateException(
                "Runtime bridge identity is unavailable"
            )
        );
        sendFrame(
            output,
            packer -> packWorldgenStructureMarkers(
                packer,
                snapshot,
                bridgeSha256
            )
        );
    }

    public static void packWorldgenStructureMarkers(
        MessagePacker packer,
        WorldgenStructureMarkerSnapshot snapshot,
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
        int count = snapshot.emittedCount();
        byte[] identities = new byte[count * 16];
        double[] positions = new double[count * 3];
        double[] rotations = new double[count * 3];
        int[] worldgenIds = new int[count];
        int[] prefabInstanceIds = new int[count];
        for (int index = 0; index < count; index++) {
            WorldgenStructureMarkerRow row = snapshot.rows().get(index);
            System.arraycopy(
                row.uuidBytes(),
                0,
                identities,
                index * 16,
                16
            );
            int offset = index * 3;
            positions[offset] = row.positionX();
            positions[offset + 1] = row.positionY();
            positions[offset + 2] = row.positionZ();
            rotations[offset] = row.rotationYaw();
            rotations[offset + 1] = row.rotationPitch();
            rotations[offset + 2] = row.rotationRoll();
            worldgenIds[index] = row.nativeWorldgenId();
            prefabInstanceIds[index] = row.nativePrefabInstanceId();
        }

        WorldgenStructureMarkerQuery query = snapshot.query();
        packer.packMapHeader(26);
        packer.packString("type");
        packer.packString("worldgen_structure_markers");
        packer.packString("schema");
        packer.packString(WorldgenStructureMarkerSnapshot.SCHEMA);
        packer.packString("version");
        packer.packInt(WorldgenStructureMarkerSnapshot.VERSION);
        packer.packString("bridge_sha256");
        packer.packString(bridgeSha256);
        packer.packString("server_version");
        packer.packString(snapshot.serverVersion());
        packer.packString("world");
        packer.packString(snapshot.worldName());
        packer.packString("worldgen_provider");
        packer.packString(snapshot.worldgenProvider());
        packer.packString("worldgen_version");
        packer.packString(snapshot.worldgenVersion());
        packer.packString("seed");
        packer.packLong(snapshot.seed());
        packer.packString("component_filter");
        packer.packString(WorldgenStructureMarkerQuery.COMPONENT_FILTER);
        packer.packString("marker_identity");
        packer.packString(WorldgenStructureMarkerQuery.MARKER_IDENTITY);
        packer.packString("bounds_semantics");
        packer.packString("half_open_min_inclusive_max_exclusive");
        packBinary(
            packer,
            "bounds_f64_le_min_max_xyz",
            encodeFloat64LittleEndian(query.bounds())
        );
        packer.packString("capacity");
        packer.packInt(query.capacity());
        packer.packString("requested_marker_asset_ids");
        packer.packArrayHeader(query.markerAssetIds().size());
        for (String markerAssetId : query.markerAssetIds()) {
            packer.packString(markerAssetId);
        }
        packer.packString("total_matching");
        packer.packInt(snapshot.totalMatching());
        packer.packString("emitted_count");
        packer.packInt(count);
        packer.packString("overflow");
        packer.packBoolean(snapshot.overflow());
        packer.packString("uuid_encoding");
        packer.packString("rfc4122_network_order_16_bytes_per_row");
        packBinary(packer, "uuid_bytes", identities);
        packBinary(
            packer,
            "positions_f64_le_xyz",
            encodeFloat64LittleEndian(positions)
        );
        packer.packString("rotation_units");
        packer.packString("radians_yaw_pitch_roll");
        packBinary(
            packer,
            "rotations_f64_le_yaw_pitch_roll",
            encodeFloat64LittleEndian(rotations)
        );
        packer.packString("marker_asset_ids");
        packer.packArrayHeader(count);
        for (WorldgenStructureMarkerRow row : snapshot.rows()) {
            packer.packString(row.markerAssetId());
        }
        packBinary(
            packer,
            "native_worldgen_ids_i32_le",
            encodeInt32LittleEndian(worldgenIds)
        );
        packBinary(
            packer,
            "native_prefab_instance_ids_i32_le",
            encodeInt32LittleEndian(prefabInstanceIds)
        );
    }
}
