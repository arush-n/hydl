package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.worldgen.NativePerceptionChannels;
import com.hytalerlbridge.worldgen.NativeDoorTransitionEvidence;
import com.hytalerlbridge.worldgen.NativeCraftingCatalogEvidence;
import com.hytalerlbridge.worldgen.NativeDropProgramEvidence;
import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence;
import com.hytalerlbridge.worldgen.NativeLineOfSightEvidence;
import com.hytalerlbridge.worldgen.NativeMutableBlockCells;
import com.hytalerlbridge.worldgen.NativeMutableBlockEvidence;
import com.hytalerlbridge.worldgen.BlockAffordanceContract;
import java.io.DataOutputStream;
import java.io.IOException;
import org.msgpack.core.MessagePacker;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeInt16LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeInt32LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeFloat64LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.packBinary;
import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;
import static com.hytalerlbridge.network.wire.BridgeIdentity.BRIDGE_SHA256;
import static com.hytalerlbridge.network.codec.SnapshotCodec.packPoint;
import static com.hytalerlbridge.network.codec.CraftingCodec.packCraftingCatalogEvidence;
import static com.hytalerlbridge.network.codec.ItemInteractionCodec.packItemInteractionEvidence;

/**
 * Extracted verbatim from {@code ClientHandler}.
 */
public final class EvidenceCodec {

    private EvidenceCodec() {}

    public static void sendPerceptionChannels(
        DataOutputStream output,
        NativePerceptionChannels channels
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(17);
            packer.packString("type");
            packer.packString("perception_channels");
            packer.packString("schema");
            packer.packString(NativePerceptionChannels.SCHEMA);
            packer.packString("version");
            packer.packInt(NativePerceptionChannels.VERSION);
            packer.packString("server_version");
            packer.packString(channels.serverVersion());
            packer.packString("world");
            packer.packString(channels.worldName());
            packer.packString("worldgen_provider");
            packer.packString(channels.worldgenProvider());
            packer.packString("worldgen_version");
            packer.packString(channels.worldgenVersion());
            packer.packString("seed");
            packer.packLong(channels.seed());
            packer.packString("sample_count");
            packer.packInt(channels.sampleCount());
            packBinary(
                packer,
                "positions_i32_le_xyz",
                encodeInt32LittleEndian(channels.positions())
            );
            packBinary(packer, "available_u8", channels.available());
            packBinary(
                packer,
                "channel_validity_u8_bits",
                channels.channelValidity()
            );
            packBinary(
                packer,
                "heightmap_i16_le",
                encodeInt16LittleEndian(channels.heightmap())
            );
            packBinary(packer, "sky_light_u8", channels.skyLight());
            packBinary(
                packer,
                "block_light_rgb_u8",
                channels.blockLightRgb()
            );
            packBinary(
                packer,
                "environment_i32_le",
                encodeInt32LittleEndian(channels.environment())
            );
            packBinary(
                packer,
                "tint_argb_i32_le",
                encodeInt32LittleEndian(channels.tintArgb())
            );
        });
    }


    public static void sendLineOfSightEvidence(
        DataOutputStream output,
        NativeLineOfSightEvidence evidence
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(28);
            packer.packString("type");
            packer.packString("los_evidence");
            packer.packString("schema");
            packer.packString(NativeLineOfSightEvidence.SCHEMA);
            packer.packString("version");
            packer.packInt(NativeLineOfSightEvidence.VERSION);
            packer.packString("server_version");
            packer.packString(evidence.serverVersion());
            packer.packString("world");
            packer.packString(evidence.worldName());
            packer.packString("worldgen_provider");
            packer.packString(evidence.worldgenProvider());
            packer.packString("worldgen_version");
            packer.packString(evidence.worldgenVersion());
            packer.packString("seed");
            packer.packLong(evidence.seed());
            packer.packString("perception_call_site");
            packer.packString(NativeLineOfSightEvidence.PERCEPTION_CALL_SITE);
            packer.packString("hit_confirmation_call_site");
            packer.packString(NativeLineOfSightEvidence.HIT_CONFIRMATION_CALL_SITE);
            packPoint(packer, "start", evidence.start());
            packPoint(packer, "loaded_end", evidence.loadedEnd());
            packPoint(packer, "unloaded_end", evidence.unloadedEnd());
            packer.packString("loaded_end_chunk_x");
            packer.packInt(evidence.loadedEndChunkX());
            packer.packString("loaded_end_chunk_z");
            packer.packInt(evidence.loadedEndChunkZ());
            packer.packString("unloaded_end_chunk_x");
            packer.packInt(evidence.unloadedEndChunkX());
            packer.packString("unloaded_end_chunk_z");
            packer.packInt(evidence.unloadedEndChunkZ());
            packer.packString("loaded_end_chunk_present");
            packer.packBoolean(evidence.loadedEndChunkPresent());
            packer.packString("unloaded_end_chunk_present_before");
            packer.packBoolean(evidence.unloadedEndChunkPresentBefore());
            packer.packString("unloaded_end_chunk_present_after");
            packer.packBoolean(evidence.unloadedEndChunkPresentAfter());
            packer.packString("perception_loaded_clear");
            packer.packBoolean(evidence.perceptionLoadedClear());
            packer.packString("perception_unloaded_clear");
            packer.packBoolean(evidence.perceptionUnloadedClear());
            packer.packString("selector_loaded_clear");
            packer.packBoolean(evidence.selectorLoadedClear());
            packer.packString("selector_unloaded_clear");
            packer.packBoolean(evidence.selectorUnloadedClear());
            packer.packString("forward_cached_visibility_after_move");
            packer.packBoolean(evidence.forwardCachedVisibilityAfterMove());
            packer.packString("inverse_uncached_visibility_after_move");
            packer.packBoolean(evidence.inverseUncachedVisibilityAfterMove());
            packer.packString("cache_step_seconds");
            packer.packDouble(evidence.cacheStepSeconds());
            packer.packString("cache_expiry_steps");
            int[] expirySteps = evidence.cacheExpirySteps();
            packer.packArrayHeader(expirySteps.length);
            for (int steps : expirySteps) packer.packInt(steps);
        });
    }


    public static void sendMutableBlockEvidence(
        DataOutputStream output,
        NativeMutableBlockEvidence evidence
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(11);
            packer.packString("type");
            packer.packString("mutable_block_evidence");
            packer.packString("schema");
            packer.packString(NativeMutableBlockEvidence.SCHEMA);
            packer.packString("version");
            packer.packInt(NativeMutableBlockEvidence.VERSION);
            packer.packString("server_version");
            packer.packString(evidence.serverVersion());
            packer.packString("world");
            packer.packString(evidence.worldName());
            packer.packString("worldgen_provider");
            packer.packString(evidence.worldgenProvider());
            packer.packString("worldgen_version");
            packer.packString(evidence.worldgenVersion());
            packer.packString("seed");
            packer.packLong(evidence.seed());
            packer.packString("affordance_dictionary_sha256");
            packer.packString(BlockAffordanceContract.DICTIONARY_SHA256);
            packer.packString("position_i32_xyz");
            int[] position = evidence.position();
            packer.packArrayHeader(position.length);
            for (int value : position) packer.packInt(value);
            packer.packString("rows");
            packer.packArrayHeader(evidence.rows().size());
            for (NativeMutableBlockEvidence.Row row : evidence.rows()) {
                packMutableBlockRow(packer, row);
            }
        });
    }

    public static void sendMutableBlockCells(
        DataOutputStream output,
        NativeMutableBlockCells capture
    ) throws IOException {
        String bridgeSha256 = BRIDGE_SHA256.orElseThrow(
            () -> new IllegalStateException(
                "Runtime bridge identity is unavailable"
            )
        );
        sendFrame(
            output,
            packer -> packMutableBlockCells(
                packer,
                capture,
                bridgeSha256
            )
        );
    }

    public static void packMutableBlockCells(
        MessagePacker packer,
        NativeMutableBlockCells capture,
        String bridgeSha256
    ) throws IOException {
        if (
            bridgeSha256 == null
                || !bridgeSha256.matches("[0-9A-Fa-f]{64}")
        ) {
            throw new IllegalArgumentException(
                "Mutable block cells require one bridge SHA-256"
            );
        }
        packer.packMapHeader(11);
        packer.packString("type");
        packer.packString("mutable_block_cells");
        packer.packString("schema");
        packer.packString(NativeMutableBlockCells.SCHEMA);
        packer.packString("version");
        packer.packInt(NativeMutableBlockCells.VERSION);
        packer.packString("server_version");
        packer.packString(capture.serverVersion());
        packer.packString("world");
        packer.packString(capture.worldName());
        packer.packString("worldgen_provider");
        packer.packString(capture.worldgenProvider());
        packer.packString("worldgen_version");
        packer.packString(capture.worldgenVersion());
        packer.packString("seed");
        packer.packLong(capture.seed());
        packer.packString("bridge_sha256");
        packer.packString(bridgeSha256.toUpperCase());
        packer.packString("affordance_dictionary_sha256");
        packer.packString(BlockAffordanceContract.DICTIONARY_SHA256);
        packer.packString("cells");
        packer.packArrayHeader(capture.cells().size());
        for (NativeMutableBlockCells.Cell cell : capture.cells()) {
            packer.packMapHeader(3);
            packer.packString("position_i32_xyz");
            int[] position = cell.position();
            packer.packArrayHeader(position.length);
            for (int value : position) packer.packInt(value);
            packer.packString("available");
            packer.packBoolean(cell.available());
            packer.packString("row");
            if (cell.available()) {
                packMutableBlockRow(packer, cell.row());
            } else {
                packer.packNil();
            }
        }
    }


    public static void sendDropProgramEvidence(
        DataOutputStream output,
        NativeDropProgramEvidence evidence
    ) throws IOException {
        String bridgeSha256 = BRIDGE_SHA256.orElseThrow(
            () -> new IllegalStateException(
                "Runtime bridge identity is unavailable"
            )
        );
        sendFrame(
            output,
            packer -> packDropProgramEvidence(
                packer,
                evidence,
                bridgeSha256
            )
        );
    }


    public static void sendCraftingCatalogEvidence(
        DataOutputStream output,
        NativeCraftingCatalogEvidence evidence
    ) throws IOException {
        String bridgeSha256 = BRIDGE_SHA256.orElseThrow(
            () -> new IllegalStateException(
                "Runtime bridge identity is unavailable"
            )
        );
        sendFrame(
            output,
            packer -> packCraftingCatalogEvidence(
                packer,
                evidence,
                bridgeSha256
            )
        );
    }


    public static void sendItemInteractionEvidence(
        DataOutputStream output,
        NativeItemInteractionEvidence evidence
    ) throws IOException {
        String bridgeSha256 = BRIDGE_SHA256.orElseThrow(
            () -> new IllegalStateException(
                "Runtime bridge identity is unavailable"
            )
        );
        sendFrame(
            output,
            packer -> packItemInteractionEvidence(
                packer,
                evidence,
                bridgeSha256
            )
        );
    }


    public static void sendDoorTransitionEvidence(
        DataOutputStream output,
        NativeDoorTransitionEvidence evidence
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(13);
            packer.packString("type");
            packer.packString("door_evidence");
            packer.packString("schema");
            packer.packString(NativeDoorTransitionEvidence.SCHEMA);
            packer.packString("version");
            packer.packInt(NativeDoorTransitionEvidence.VERSION);
            packer.packString("server_version");
            packer.packString(evidence.serverVersion());
            packer.packString("world");
            packer.packString(evidence.worldName());
            packer.packString("worldgen_provider");
            packer.packString(evidence.worldgenProvider());
            packer.packString("worldgen_version");
            packer.packString(evidence.worldgenVersion());
            packer.packString("seed");
            packer.packLong(evidence.seed());
            packer.packString("door_asset_id");
            packer.packString(evidence.doorAssetId());
            packer.packString("root_interaction_id");
            packer.packString(evidence.rootInteractionId());
            packer.packString("transition_execution");
            packer.packString(NativeDoorTransitionEvidence.EXECUTION);
            packer.packString("timing_certified");
            packer.packBoolean(false);
            packer.packString("rows");
            packer.packArrayHeader(evidence.rows().size());
            for (NativeDoorTransitionEvidence.Row row : evidence.rows()) {
                packer.packMapHeader(14);
                packer.packString("yaw_degrees");
                packer.packInt(row.yawDegrees());
                packer.packString("partner_offset");
                int[] offset = row.partnerOffset();
                packer.packArrayHeader(offset.length);
                for (int value : offset) packer.packInt(value);
                packer.packString("single_front_open_state");
                packer.packString(row.singleFrontOpenState());
                packer.packString("single_front_open_hitbox_type");
                packer.packString(row.singleFrontOpenHitboxType());
                packer.packString("single_closed_after_front");
                packer.packBoolean(row.singleClosedAfterFront());
                packer.packString("single_back_open_state");
                packer.packString(row.singleBackOpenState());
                packer.packString("single_back_open_hitbox_type");
                packer.packString(row.singleBackOpenHitboxType());
                packer.packString("single_closed_after_back");
                packer.packBoolean(row.singleClosedAfterBack());
                packer.packString("double_root_open_state");
                packer.packString(row.doubleRootOpenState());
                packer.packString("double_root_open_hitbox_type");
                packer.packString(row.doubleRootOpenHitboxType());
                packer.packString("double_partner_open_state");
                packer.packString(row.doublePartnerOpenState());
                packer.packString("double_partner_open_hitbox_type");
                packer.packString(row.doublePartnerOpenHitboxType());
                packer.packString("double_root_closed");
                packer.packBoolean(row.doubleRootClosed());
                packer.packString("double_partner_closed");
                packer.packBoolean(row.doublePartnerClosed());
            }
        });
    }


    public static void packDropProgramEvidence(
        MessagePacker packer,
        NativeDropProgramEvidence evidence,
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
        packer.packMapHeader(19);
        packer.packString("type");
        packer.packString("drop_program_evidence");
        packer.packString("schema");
        packer.packString(NativeDropProgramEvidence.SCHEMA);
        packer.packString("version");
        packer.packInt(NativeDropProgramEvidence.VERSION);
        packer.packString("bridge_sha256");
        packer.packString(bridgeSha256);
        packer.packString("server_version");
        packer.packString(evidence.serverVersion());
        packer.packString("world");
        packer.packString(evidence.worldName());
        packer.packString("worldgen_provider");
        packer.packString(evidence.worldgenProvider());
        packer.packString("worldgen_version");
        packer.packString(evidence.worldgenVersion());
        packer.packString("seed");
        packer.packLong(evidence.seed());
        packer.packString("block_asset_id");
        packer.packString(evidence.blockAssetId());
        packer.packString("route");
        packer.packString(evidence.route());
        packer.packString("authored_quantity");
        packer.packInt(evidence.authoredQuantity());
        packer.packString("resolved_item_id");
        packer.packString(evidence.resolvedItemId());
        packer.packString("resolved_drop_list_id");
        packer.packString(evidence.resolvedDropListId());
        packer.packString("sample_count");
        packer.packInt(evidence.sampleCount());
        packer.packString("rng");
        packer.packString(NativeDropProgramEvidence.RNG);
        packer.packString("same_seed_replayable");
        packer.packBoolean(false);
        packer.packString("empty_stack_semantics");
        packer.packString("elided_semantic_no_drop");
        packer.packString("outcomes");
        packer.packArrayHeader(evidence.outcomes().size());
        for (NativeDropProgramEvidence.Outcome outcome
            : evidence.outcomes()) {
            packer.packMapHeader(2);
            packer.packString("count");
            packer.packInt(outcome.count());
            packer.packString("stacks");
            packer.packArrayHeader(outcome.stacks().size());
            for (NativeDropProgramEvidence.Stack stack
                : outcome.stacks()) {
                packer.packMapHeader(5);
                packer.packString("item_asset_id");
                packer.packString(stack.itemAssetId());
                packer.packString("quantity");
                packer.packInt(stack.quantity());
                packer.packString("durability");
                packer.packDouble(stack.durability());
                packer.packString("max_durability");
                packer.packDouble(stack.maxDurability());
                packer.packString("metadata_json_sha256");
                packer.packString(stack.metadataJsonSha256());
            }
        }
    }


    public static void packMutableBlockRow(
        MessagePacker packer,
        NativeMutableBlockEvidence.Row row
    ) throws IOException {
        packer.packMapHeader(27);
        packer.packString("phase");
        packer.packString(row.phase());
        packer.packString("block_present");
        packer.packBoolean(row.blockPresent());
        packer.packString("block_asset_id");
        packer.packString(row.blockAssetId());
        packer.packString("runtime_block_id");
        packer.packInt(row.runtimeBlockId());
        packBinary(
            packer,
            "semantic_key_sha256",
            row.semanticKeySha256()
        );
        packer.packString("semantic_key_valid");
        packer.packBoolean(row.semanticKeyValid());
        packer.packString("affordance_valid");
        packer.packBoolean(row.affordanceValid());
        packer.packString("affordance_tags");
        packer.packInt(row.affordanceTags());
        packer.packString("gather_type_index");
        packer.packInt(row.gatherTypeIndex());
        packer.packString("required_tool_quality");
        packer.packInt(row.requiredToolQuality());
        packer.packString("rotation_index");
        packer.packInt(row.rotationIndex());
        packer.packString("flags");
        packer.packInt(row.flags());
        packer.packString("fluid_level");
        packer.packInt(row.fluidLevel());
        packer.packString("fluid_fill_height");
        packer.packDouble(row.fluidFillHeight());
        packer.packString("support");
        packer.packInt(row.supportValue());
        packer.packString("block_damage");
        packer.packInt(row.blockDamage());
        packer.packString("fluid_damage");
        packer.packInt(row.fluidDamage());
        packBinary(
            packer,
            "movement_f64_le",
            encodeFloat64LittleEndian(row.movement())
        );
        packBinary(
            packer,
            "fluid_movement_f64_le",
            encodeFloat64LittleEndian(row.fluidMovement())
        );
        double[] boxes = row.collisionBoxes();
        packBinary(
            packer,
            "collision_boxes_f64_le",
            encodeFloat64LittleEndian(boxes)
        );
        packer.packString("collision_box_count");
        packer.packInt(boxes.length / 6);
        packer.packString("block_health");
        packer.packFloat(row.blockHealth());
        packer.packString("block_health_valid");
        packer.packBoolean(row.blockHealthValid());
        packer.packString("seconds_since_damage");
        packer.packDouble(row.secondsSinceDamage());
        packer.packString("damage_age_valid");
        packer.packBoolean(row.damageAgeValid());
        packer.packString("local_change_counter");
        packer.packShort(row.localChangeCounter());
        packer.packString("global_change_counter");
        packer.packShort(row.globalChangeCounter());
    }
}
