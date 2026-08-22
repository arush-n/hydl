package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.entity.PrivilegedEntityQuery;
import com.hytalerlbridge.entity.PrivilegedEntityRow;
import com.hytalerlbridge.entity.PrivilegedEntitySnapshot;
import com.hytalerlbridge.entity.PrivilegedNpcGeometryRow;
import com.hytalerlbridge.entity.PrivilegedNpcSnapshot;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.List;
import org.msgpack.core.MessagePacker;
import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;
import static com.hytalerlbridge.network.wire.BridgeIdentity.BRIDGE_SHA256;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeFloat64LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.packBinary;

/**
 * Extracted verbatim from {@code ClientHandler}.
 */
public final class SnapshotCodec {

    private SnapshotCodec() {}

    public static void sendPrivilegedEntitySnapshot(
        DataOutputStream output,
        PrivilegedEntitySnapshot snapshot
    ) throws IOException {
        String bridgeSha256 = BRIDGE_SHA256.orElseThrow(
            () -> new IllegalStateException(
                "Runtime bridge identity is unavailable"
            )
        );
        sendFrame(
            output,
            packer -> packPrivilegedEntitySnapshot(
                packer,
                snapshot,
                bridgeSha256
            )
        );
    }


    public static void packPrivilegedEntitySnapshot(
        MessagePacker packer,
        PrivilegedEntitySnapshot snapshot,
        String bridgeSha256
    ) throws IOException {
        packPrivilegedSnapshot(
            packer,
            snapshot,
            bridgeSha256,
            "privileged_entity_snapshot",
            PrivilegedEntitySnapshot.SCHEMA,
            PrivilegedEntitySnapshot.VERSION,
            PrivilegedEntityQuery.COMPONENT_FILTER,
            null,
            null
        );
    }


    public static void sendPrivilegedNpcSnapshot(
        DataOutputStream output,
        PrivilegedNpcSnapshot snapshot
    ) throws IOException {
        String bridgeSha256 = BRIDGE_SHA256.orElseThrow(
            () -> new IllegalStateException(
                "Runtime bridge identity is unavailable"
            )
        );
        sendFrame(
            output,
            packer -> packPrivilegedNpcSnapshot(
                packer,
                snapshot,
                bridgeSha256
            )
        );
    }


    public static void packPrivilegedNpcSnapshot(
        MessagePacker packer,
        PrivilegedNpcSnapshot snapshot,
        String bridgeSha256
    ) throws IOException {
        PrivilegedEntitySnapshot spatial = snapshot.spatial();
        packPrivilegedSnapshot(
            packer,
            spatial,
            bridgeSha256,
            "privileged_npc_snapshot",
            PrivilegedNpcSnapshot.SCHEMA,
            PrivilegedNpcSnapshot.VERSION,
            PrivilegedNpcSnapshot.COMPONENT_FILTER,
            spatial.rows().stream()
                .map(PrivilegedEntityRow::typeAssetId)
                .toList(),
            snapshot.geometryRows()
        );
    }


    public static void packPrivilegedSnapshot(
        MessagePacker packer,
        PrivilegedEntitySnapshot snapshot,
        String bridgeSha256,
        String messageType,
        String schema,
        int version,
        String componentFilter,
        List<String> typeAssetIds,
        List<PrivilegedNpcGeometryRow> geometryRows
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
        double[] velocities = new double[count * 3];
        for (int index = 0; index < count; index++) {
            PrivilegedEntityRow row = snapshot.rows().get(index);
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
            velocities[offset] = row.velocityX();
            velocities[offset + 1] = row.velocityY();
            velocities[offset + 2] = row.velocityZ();
        }
        int extraNpcFields = geometryRows == null ? 0 : 13;
        packer.packMapHeader((typeAssetIds == null ? 22 : 23) + extraNpcFields);
        packer.packString("type");
        packer.packString(messageType);
        packer.packString("schema");
        packer.packString(schema);
        packer.packString("version");
        packer.packInt(version);
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
        packer.packString(componentFilter);
        packer.packString("bounds_semantics");
        packer.packString("half_open_min_inclusive_max_exclusive");
        packBinary(
            packer,
            "bounds_f64_le_min_max_xyz",
            encodeFloat64LittleEndian(snapshot.query().bounds())
        );
        packer.packString("capacity");
        packer.packInt(snapshot.query().capacity());
        packer.packString("total_matching");
        packer.packInt(snapshot.totalMatching());
        packer.packString("emitted_count");
        packer.packInt(snapshot.emittedCount());
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
        packBinary(
            packer,
            "velocities_f64_le_xyz",
            encodeFloat64LittleEndian(velocities)
        );
        if (typeAssetIds != null) {
            if (typeAssetIds.size() != count) {
                throw new IllegalArgumentException(
                    "NPC type asset IDs must match emitted rows"
                );
            }
            packer.packString("type_asset_ids");
            packer.packArrayHeader(count);
            for (String typeAssetId : typeAssetIds) {
                if (typeAssetId == null || typeAssetId.isBlank()) {
                    throw new IllegalArgumentException(
                        "NPC type asset IDs must be non-empty"
                    );
                }
                packer.packString(typeAssetId);
            }
        }
        if (geometryRows != null) {
            packNpcGeometry(packer, geometryRows, count);
        }
    }


    private static void packNpcGeometry(
        MessagePacker packer,
        List<PrivilegedNpcGeometryRow> rows,
        int count
    ) throws IOException {
        if (rows.size() != count) {
            throw new IllegalArgumentException(
                "NPC geometry rows must match emitted rows"
            );
        }
        byte[] supported = new byte[count];
        double[] bounds = new double[count * 6];
        byte[] collidable = new byte[count];
        byte[] blocksLos = new byte[count];
        byte[] offsetSupported = new byte[count];
        double[] offsets = new double[count * 3];
        byte[] modelPresent = new byte[count];
        double[] modelScales = new double[count];
        double[] modelEyeHeights = new double[count];
        byte[] entityScalePresent = new byte[count];
        double[] entityScales = new double[count];
        for (int index = 0; index < count; index++) {
            PrivilegedNpcGeometryRow row = rows.get(index);
            supported[index] = (byte) (row.geometrySupported() ? 1 : 0);
            collidable[index] = (byte) (row.collidable() ? 1 : 0);
            blocksLos[index] = (byte) (row.blocksLineOfSight() ? 1 : 0);
            offsetSupported[index] = (byte) (
                row.lineOfSightOffsetSupported() ? 1 : 0
            );
            modelPresent[index] = (byte) (row.modelPresent() ? 1 : 0);
            entityScalePresent[index] = (byte) (
                row.entityScalePresent() ? 1 : 0
            );
            int boundsOffset = index * 6;
            bounds[boundsOffset] = row.minimumX();
            bounds[boundsOffset + 1] = row.minimumY();
            bounds[boundsOffset + 2] = row.minimumZ();
            bounds[boundsOffset + 3] = row.maximumX();
            bounds[boundsOffset + 4] = row.maximumY();
            bounds[boundsOffset + 5] = row.maximumZ();
            int offset = index * 3;
            offsets[offset] = row.lineOfSightOffsetX();
            offsets[offset + 1] = row.lineOfSightOffsetY();
            offsets[offset + 2] = row.lineOfSightOffsetZ();
            modelScales[index] = row.modelScale();
            modelEyeHeights[index] = row.modelEyeHeight();
            entityScales[index] = row.entityScale();
        }
        packer.packString("geometry_semantics");
        packer.packString(
            "runtime_rotation_adjusted_local_aabb_position_cache_los_v1"
        );
        packBinary(packer, "geometry_supported_u8", supported);
        packBinary(
            packer,
            "local_bounds_f64_le_min_max_xyz",
            encodeFloat64LittleEndian(bounds)
        );
        packBinary(packer, "collidable_u8", collidable);
        packBinary(packer, "blocks_los_u8", blocksLos);
        packBinary(packer, "los_offset_supported_u8", offsetSupported);
        packBinary(
            packer,
            "los_offsets_f64_le_xyz",
            encodeFloat64LittleEndian(offsets)
        );
        packBinary(packer, "model_present_u8", modelPresent);
        packer.packString("model_asset_ids");
        packer.packArrayHeader(count);
        for (PrivilegedNpcGeometryRow row : rows) {
            packer.packString(row.modelAssetId());
        }
        packBinary(
            packer,
            "model_scales_f64_le",
            encodeFloat64LittleEndian(modelScales)
        );
        packBinary(
            packer,
            "model_eye_heights_f64_le",
            encodeFloat64LittleEndian(modelEyeHeights)
        );
        packBinary(packer, "entity_scale_present_u8", entityScalePresent);
        packBinary(
            packer,
            "entity_scales_f64_le",
            encodeFloat64LittleEndian(entityScales)
        );
    }


    public static void packPoint(
        MessagePacker packer,
        String name,
        double[] point
    ) throws IOException {
        packer.packString(name);
        packer.packArrayHeader(point.length);
        for (double value : point) packer.packDouble(value);
    }
}
