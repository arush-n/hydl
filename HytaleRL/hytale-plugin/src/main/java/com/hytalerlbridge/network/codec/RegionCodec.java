package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.worldgen.NativeRegionManifest;
import com.hytalerlbridge.worldgen.NativeRegionBlockSemanticSection;
import com.hytalerlbridge.worldgen.NativeRegionFluidSemanticSection;
import com.hytalerlbridge.worldgen.NativeRegionLightSection;
import com.hytalerlbridge.worldgen.NativeRegionSection;
import com.hytalerlbridge.worldgen.BlockAffordanceContract;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;
import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;
import static com.hytalerlbridge.network.wire.FrameWriter.packInfoValue;

/**
 * Extracted verbatim from {@code ClientHandler}.
 */
public final class RegionCodec {

    private RegionCodec() {}

    public static void sendRegionManifest(
        DataOutputStream output,
        NativeRegionManifest manifest
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(18);
            packer.packString("type");
            packer.packString("region_manifest");
            packer.packString("schema");
            packer.packString(NativeRegionManifest.SCHEMA);
            packer.packString("version");
            packer.packInt(NativeRegionManifest.VERSION);
            packer.packString("server_version");
            packer.packString(manifest.serverVersion());
            packer.packString("world");
            packer.packString(manifest.worldName());
            packer.packString("worldgen_provider");
            packer.packString(manifest.worldgenProvider());
            packer.packString("worldgen_version");
            packer.packString(manifest.worldgenVersion());
            packer.packString("seed");
            packer.packLong(manifest.seed());
            packer.packString("capture_mode");
            packer.packString(NativeRegionManifest.CAPTURE_MODE);
            packer.packString("exact_collision_shapes");
            packer.packBoolean(true);
            packer.packString("dynamic_state");
            packer.packString(NativeRegionManifest.DYNAMIC_STATE);
            packer.packString("core_min_chunk_x");
            packer.packInt(manifest.coreMinChunkX());
            packer.packString("core_min_chunk_z");
            packer.packInt(manifest.coreMinChunkZ());
            packer.packString("capture_min_chunk_x");
            packer.packInt(manifest.captureMinChunkX());
            packer.packString("capture_min_chunk_z");
            packer.packInt(manifest.captureMinChunkZ());
            packer.packString("capture_chunks_per_axis");
            packer.packInt(NativeRegionManifest.CAPTURE_CHUNKS_PER_AXIS);
            packer.packString("capture_section_count");
            packer.packInt(NativeRegionManifest.CAPTURE_SECTION_COUNT);
            packer.packString("chunk_api");
            packer.packMapHeader(manifest.chunkApi().size());
            for (
                Map.Entry<String, Object> entry
                    : manifest.chunkApi().entrySet()
            ) {
                packer.packString(entry.getKey());
                packInfoValue(packer, entry.getValue());
            }
        });
    }


    public static void sendRegionSection(
        DataOutputStream output,
        NativeRegionSection section
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(12);
            packer.packString("type");
            packer.packString("region_section");
            packer.packString("schema");
            packer.packString(NativeRegionSection.SCHEMA);
            packer.packString("version");
            packer.packInt(NativeRegionSection.VERSION);
            packer.packString("chunk_x");
            packer.packInt(section.chunkX());
            packer.packString("chunk_z");
            packer.packInt(section.chunkZ());
            packer.packString("section_y");
            packer.packInt(section.sectionY());
            packer.packString("code_encoding");
            packer.packString(NativeRegionSection.CODE_ENCODING);
            packer.packString("cell_codes");
            byte[] codes = section.cellCodes();
            packer.packBinaryHeader(codes.length);
            packer.writePayload(codes);
            packer.packString("filler_root_encoding");
            packer.packString(NativeRegionSection.FILLER_ROOT_ENCODING);
            packer.packString("filler_root_offsets");
            byte[] fillerRootOffsets = section.fillerRootOffsets();
            packer.packBinaryHeader(fillerRootOffsets.length);
            packer.writePayload(fillerRootOffsets);
            packer.packString("cell_palette");
            packer.packArrayHeader(section.cellPalette().size());
            for (var cell : section.cellPalette()) {
                packer.packArrayHeader(9);
                packer.packInt(cell.flags());
                packer.packInt(cell.shapeIndex());
                packer.packInt(cell.fluidLevel());
                packer.packDouble(cell.fluidFillHeight());
                packer.packInt(cell.supportValue());
                packer.packInt(cell.blockDamage());
                packer.packInt(cell.fluidDamage());
                double[] movement = cell.movement();
                packer.packArrayHeader(movement.length);
                for (double value : movement) packer.packDouble(value);
                double[] fluidMovement = cell.fluidMovement();
                packer.packArrayHeader(fluidMovement.length);
                for (double value : fluidMovement) {
                    packer.packDouble(value);
                }
            }
            packer.packString("shape_palette");
            packer.packArrayHeader(section.shapePalette().size());
            for (var shape : section.shapePalette()) {
                double[] boxes = shape.collisionBoxes();
                packer.packArrayHeader(boxes.length);
                for (double value : boxes) packer.packDouble(value);
            }
        });
    }

    public static void sendRegionLightSection(
        DataOutputStream output,
        NativeRegionLightSection section
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(12);
            packer.packString("type");
            packer.packString("region_light_section");
            packer.packString("schema");
            packer.packString(NativeRegionLightSection.SCHEMA);
            packer.packString("version");
            packer.packInt(NativeRegionLightSection.VERSION);
            packer.packString("chunk_x");
            packer.packInt(section.chunkX());
            packer.packString("chunk_z");
            packer.packInt(section.chunkZ());
            packer.packString("section_y");
            packer.packInt(section.sectionY());
            packer.packString("status");
            packer.packString(section.status());
            packer.packString("available");
            packer.packBoolean(section.available());
            packer.packString("global_change_counter");
            packer.packInt(section.globalChangeCounter());
            packer.packString("global_light_change_id");
            packer.packInt(section.globalLightChangeId());
            packer.packString("light_encoding");
            packer.packString(NativeRegionLightSection.LIGHT_ENCODING);
            packer.packString("light_data");
            byte[] payload = section.lightData();
            packer.packBinaryHeader(payload.length);
            packer.writePayload(payload);
        });
    }


    public static void sendRegionBlockSemantics(
        DataOutputStream output,
        NativeRegionBlockSemanticSection section
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(10);
            packer.packString("type");
            packer.packString("region_block_semantics");
            packer.packString("schema");
            packer.packString(NativeRegionBlockSemanticSection.SCHEMA);
            packer.packString("version");
            packer.packInt(NativeRegionBlockSemanticSection.VERSION);
            packer.packString("chunk_x");
            packer.packInt(section.chunkX());
            packer.packString("chunk_z");
            packer.packInt(section.chunkZ());
            packer.packString("section_y");
            packer.packInt(section.sectionY());
            packer.packString("code_encoding");
            packer.packString(
                NativeRegionBlockSemanticSection.CODE_ENCODING
            );
            packer.packString("cell_codes");
            byte[] codes = section.cellCodes();
            packer.packBinaryHeader(codes.length);
            packer.writePayload(codes);
            packer.packString("affordance_dictionary_sha256");
            packer.packString(
                BlockAffordanceContract.DICTIONARY_SHA256
            );
            packer.packString("palette");
            packer.packArrayHeader(section.palette().size());
            for (var entry : section.palette()) {
                packer.packArrayHeader(7);
                byte[] semanticKey = entry.semanticKeySha256();
                packer.packBinaryHeader(semanticKey.length);
                packer.writePayload(semanticKey);
                byte[] assetKey = entry.assetKeySha256();
                packer.packBinaryHeader(assetKey.length);
                packer.writePayload(assetKey);
                packer.packBoolean(entry.valid());
                packer.packInt(entry.affordanceTags());
                packer.packInt(entry.gatherTypeIndex());
                packer.packInt(entry.requiredToolQuality());
                packer.packInt(entry.rotationIndex());
            }
        });
    }

    public static void sendRegionFluidSemantics(
        DataOutputStream output,
        NativeRegionFluidSemanticSection section
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(9);
            packer.packString("type");
            packer.packString("region_fluid_semantics");
            packer.packString("schema");
            packer.packString(NativeRegionFluidSemanticSection.SCHEMA);
            packer.packString("version");
            packer.packInt(NativeRegionFluidSemanticSection.VERSION);
            packer.packString("chunk_x");
            packer.packInt(section.chunkX());
            packer.packString("chunk_z");
            packer.packInt(section.chunkZ());
            packer.packString("section_y");
            packer.packInt(section.sectionY());
            packer.packString("code_encoding");
            packer.packString(NativeRegionFluidSemanticSection.CODE_ENCODING);
            packer.packString("cell_codes");
            byte[] codes = section.cellCodes();
            packer.packBinaryHeader(codes.length);
            packer.writePayload(codes);
            packer.packString("palette");
            packer.packArrayHeader(section.palette().size());
            for (var entry : section.palette()) {
                packer.packString(entry.assetId());
            }
        });
    }
}
