package com.hytalerlbridge.network.codec;

import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;

import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.msgpack.core.MessagePacker;

/**
 * Block-type table: runtime block id to asset id, for the observation window.
 *
 * <p>Runtime ids are the {@code BlockType} asset-map indices, the same ids
 * {@code nearby_blocks} rows carry (column.getBlock stores the index that
 * {@code getIndex(assetId)} returns). The table is global to the process.
 */
public final class BlockTypeTableCodec {

    public static final String SCHEMA = "hytalerl_block_type_table_v1";
    public static final int VERSION = 1;

    private BlockTypeTableCodec() {}

    /** One table row: a runtime id and the asset id it names. */
    public record Entry(int runtimeId, String assetId) {}

    /** The live table, sorted by runtime id. Cheap: a few hundred map reads. */
    public static List<Entry> liveEntries() {
        var assets = BlockType.getAssetMap();
        List<Entry> rows = new ArrayList<>();
        for (String assetId : assets.getAssetMap().keySet()) {
            rows.add(new Entry(assets.getIndex(assetId), assetId));
        }
        rows.sort(Comparator.comparingInt(Entry::runtimeId));
        return rows;
    }

    public static void sendBlockTypeTable(
        DataOutputStream output,
        List<Entry> entries
    ) throws IOException {
        sendFrame(output, packer -> pack(packer, entries));
    }

    public static void pack(MessagePacker packer, List<Entry> entries)
        throws IOException {
        packer.packMapHeader(5);
        packer.packString("type");
        packer.packString("block_type_table");
        packer.packString("schema");
        packer.packString(SCHEMA);
        packer.packString("version");
        packer.packInt(VERSION);
        packer.packString("count");
        packer.packInt(entries.size());
        packer.packString("entries");
        packer.packArrayHeader(entries.size());
        for (Entry entry : entries) {
            packer.packArrayHeader(2);
            packer.packInt(entry.runtimeId());
            packer.packString(entry.assetId());
        }
    }
}
