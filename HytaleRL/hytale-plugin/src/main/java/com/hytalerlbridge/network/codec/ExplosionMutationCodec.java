package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.worldgen.BlockAffordanceContract;
import com.hytalerlbridge.worldgen.NativeExplosionMutationProbe;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HexFormat;
import org.msgpack.core.MessagePacker;
import static com.hytalerlbridge.network.wire.BinaryCodec.packBinary;
import static com.hytalerlbridge.network.wire.BridgeIdentity.BRIDGE_SHA256;
import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;

/** Wire encoding for complete-or-empty native explosion mutation evidence. */
public final class ExplosionMutationCodec {

    private ExplosionMutationCodec() {}

    public static void sendExplosionMutationProbe(
        DataOutputStream output,
        NativeExplosionMutationProbe probe
    ) throws IOException {
        String bridgeSha256 = BRIDGE_SHA256.orElseThrow(
            () -> new IllegalStateException(
                "Runtime bridge identity is unavailable"
            )
        );
        sendFrame(
            output,
            packer -> packExplosionMutationProbe(
                packer,
                probe,
                bridgeSha256
            )
        );
    }

    public static void packExplosionMutationProbe(
        MessagePacker packer,
        NativeExplosionMutationProbe probe,
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

        packer.packMapHeader(54);
        packer.packString("type");
        packer.packString("explosion_mutation_probe");
        packer.packString("schema");
        packer.packString(NativeExplosionMutationProbe.SCHEMA);
        packer.packString("version");
        packer.packInt(NativeExplosionMutationProbe.VERSION);
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
        packer.packString("explosion_config_source_id");
        packer.packString(probe.explosionConfigSourceId());
        packer.packString("fixture_kind");
        packer.packString(probe.fixtureKind());
        packer.packString("fixture_block_asset_id");
        packer.packString(probe.fixtureBlockAssetId());
        packer.packString("explosion_config_semantic_sha256");
        packer.packString(probe.explosionConfigSemanticSha256());
        packer.packString("origin_x");
        packer.packDouble(probe.originX());
        packer.packString("origin_y");
        packer.packDouble(probe.originY());
        packer.packString("origin_z");
        packer.packDouble(probe.originZ());
        packer.packString("damage_blocks");
        packer.packBoolean(true);
        packer.packString("damage_entities");
        packer.packBoolean(probe.damageEntities());
        packer.packString("block_damage_radius");
        packer.packInt(probe.blockDamageRadius());
        packer.packString("block_damage_falloff");
        packer.packFloat(probe.blockDamageFalloff());
        packer.packString("block_drop_chance");
        packer.packFloat(probe.blockDropChance());
        packer.packString("entity_damage_radius");
        packer.packFloat(probe.entityDamageRadius());
        packer.packString("entity_damage");
        packer.packFloat(probe.entityDamage());
        packer.packString("entity_damage_falloff");
        packer.packFloat(probe.entityDamageFalloff());
        packer.packString("ignore_controlled_actor");
        packer.packBoolean(probe.ignoredControlledActor());
        packer.packString("scope");
        packer.packString(NativeExplosionMutationProbe.CONTROLLED_SCOPE);
        packer.packString("dry_controlled_scope");
        packer.packBoolean(probe.controlledDryScope());
        packer.packString("unsupported_fluid_mutation");
        packer.packBoolean(probe.unsupportedFluidMutation());
        packer.packString("unsupported_filler_mutation");
        packer.packBoolean(probe.unsupportedFillerMutation());
        packer.packString("unsupported_support_cascade");
        packer.packBoolean(probe.unsupportedSupportCascade());
        packer.packString("entity_capacity");
        packer.packInt(probe.entityCapacity());
        packer.packString("total_entity_admissions");
        packer.packInt(probe.totalEntityAdmissions());
        packer.packString("entity_overflow");
        packer.packBoolean(probe.entityOverflow());
        packer.packString("cell_capacity");
        packer.packInt(probe.cellCapacity());
        packer.packString("total_changed_cells");
        packer.packInt(probe.totalChangedCells());
        packer.packString("cell_overflow");
        packer.packBoolean(probe.cellOverflow());
        packer.packString("drop_capacity");
        packer.packInt(probe.dropCapacity());
        packer.packString("total_resolved_drops");
        packer.packInt(probe.totalResolvedDrops());
        packer.packString("drop_overflow");
        packer.packBoolean(probe.dropOverflow());
        packer.packString("execution_started");
        packer.packBoolean(probe.executionStarted());
        packer.packString("complete");
        packer.packBoolean(probe.complete());
        packer.packString("resync_required");
        packer.packBoolean(probe.resyncRequired());
        packer.packString("failure_reason");
        packer.packString(probe.failureReason());
        packer.packString("entity_admissions");
        packer.packArrayHeader(probe.entityAdmissions().size());
        for (NativeExplosionMutationProbe.EntityAdmission admission
            : probe.entityAdmissions()) {
            packEntityAdmission(packer, admission);
        }
        packer.packString("changed_cells");
        packer.packArrayHeader(probe.changedCells().size());
        for (NativeExplosionMutationProbe.ChangedCell cell
            : probe.changedCells()) {
            packChangedCell(packer, cell);
        }
        packer.packString("resolved_drops");
        packer.packArrayHeader(probe.resolvedDrops().size());
        for (NativeExplosionMutationProbe.ResolvedDrop drop
            : probe.resolvedDrops()) {
            packResolvedDrop(packer, drop);
        }
        packer.packString("affordance_dictionary_sha256");
        packer.packString(BlockAffordanceContract.DICTIONARY_SHA256);
        packer.packString("cell_scalar_compatibility");
        packer.packString(
            NativeExplosionMutationProbe.CELL_SCALAR_COMPATIBILITY
        );
        packer.packString("counter_semantics");
        packer.packString("signed_short_inequality_only");
        packer.packString("geometry_refresh");
        packer.packString("targeted_requery_via_mutable_block_cells");
        packer.packString("changed_cell_ordering");
        packer.packString(NativeExplosionMutationProbe.CHANGED_CELL_ORDERING);
        packer.packString("resolved_drop_ordering");
        packer.packString(NativeExplosionMutationProbe.RESOLVED_DROP_ORDERING);
        packer.packString("native_method");
        packer.packString(
            "ExplosionUtils.performExplosion/processTargetBlocks"
        );
    }

    private static void packEntityAdmission(
        MessagePacker packer,
        NativeExplosionMutationProbe.EntityAdmission admission
    ) throws IOException {
        packer.packMapHeader(5);
        packer.packString("ordinal");
        packer.packInt(admission.ordinal());
        packer.packString("uuid");
        packer.packString(admission.uuid().toString());
        packer.packString("position_f64_xyz");
        packer.packArrayHeader(3);
        packer.packDouble(admission.x());
        packer.packDouble(admission.y());
        packer.packDouble(admission.z());
        packer.packString("distance");
        packer.packDouble(admission.distance());
        packer.packString("damage");
        packer.packFloat(admission.damage());
    }

    private static void packChangedCell(
        MessagePacker packer,
        NativeExplosionMutationProbe.ChangedCell cell
    ) throws IOException {
        packer.packMapHeader(5);
        packer.packString("ordinal");
        packer.packInt(cell.ordinal());
        packer.packString("position_i32_xyz");
        packer.packArrayHeader(3);
        packer.packInt(cell.x());
        packer.packInt(cell.y());
        packer.packInt(cell.z());
        packer.packString("before");
        packCellState(packer, cell.before());
        packer.packString("after");
        packCellState(packer, cell.after());
        packer.packString("partial_block_health");
        packer.packBoolean(cell.partialBlockHealth());
    }

    private static void packCellState(
        MessagePacker packer,
        NativeExplosionMutationProbe.CellState state
    ) throws IOException {
        packer.packMapHeader(22);
        packer.packString("block_present");
        packer.packBoolean(state.blockPresent());
        packer.packString("block_asset_id");
        packer.packString(state.blockAssetId());
        packer.packString("runtime_block_id");
        packer.packInt(state.runtimeBlockId());
        packBinary(
            packer,
            "semantic_key_sha256",
            state.semanticKeyValid()
                ? HexFormat.of().parseHex(state.semanticKeySha256())
                : new byte[0]
        );
        packer.packString("semantic_key_valid");
        packer.packBoolean(state.semanticKeyValid());
        packer.packString("affordance_valid");
        packer.packBoolean(state.affordanceValid());
        packer.packString("affordance_tags");
        packer.packInt(state.affordanceTags());
        packer.packString("gather_type_index");
        packer.packInt(state.gatherTypeIndex());
        packer.packString("required_tool_quality");
        packer.packInt(state.requiredToolQuality());
        packer.packString("rotation_index");
        packer.packInt(state.rotationIndex());
        packer.packString("flags");
        packer.packInt(state.flags());
        packer.packString("fluid_level");
        packer.packInt(state.fluidLevel());
        packer.packString("fluid_fill_height");
        packer.packDouble(state.fluidFillHeight());
        packer.packString("support");
        packer.packInt(state.supportValue());
        packer.packString("block_damage");
        packer.packInt(state.blockDamage());
        packer.packString("fluid_damage");
        packer.packInt(state.fluidDamage());
        packer.packString("block_health");
        packer.packFloat(state.blockHealth());
        packer.packString("block_health_valid");
        packer.packBoolean(state.blockHealthValid());
        packer.packString("seconds_since_damage");
        packer.packDouble(state.secondsSinceDamage());
        packer.packString("damage_age_valid");
        packer.packBoolean(state.damageAgeValid());
        packer.packString("local_change_counter");
        packer.packShort(state.localChangeCounter());
        packer.packString("global_change_counter");
        packer.packShort(state.globalChangeCounter());
    }

    private static void packResolvedDrop(
        MessagePacker packer,
        NativeExplosionMutationProbe.ResolvedDrop drop
    ) throws IOException {
        packer.packMapHeader(8);
        packer.packString("ordinal");
        packer.packInt(drop.ordinal());
        packer.packString("source_position_i32_xyz");
        packer.packArrayHeader(3);
        packer.packInt(drop.sourceX());
        packer.packInt(drop.sourceY());
        packer.packInt(drop.sourceZ());
        packer.packString("item_asset_id");
        packer.packString(drop.itemAssetId());
        packer.packString("quantity");
        packer.packInt(drop.quantity());
        packer.packString("durability");
        packer.packDouble(drop.durability());
        packer.packString("max_durability");
        packer.packDouble(drop.maxDurability());
        packer.packString("metadata_json_sha256");
        packer.packString(drop.metadataJsonSha256());
        packer.packString("spawn_position_f64_xyz");
        packer.packArrayHeader(3);
        packer.packDouble(drop.spawnX());
        packer.packDouble(drop.spawnY());
        packer.packDouble(drop.spawnZ());
    }
}
