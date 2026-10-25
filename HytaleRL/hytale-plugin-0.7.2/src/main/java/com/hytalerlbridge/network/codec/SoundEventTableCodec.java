package com.hytalerlbridge.network.codec;

import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;

import com.hypixel.hytale.server.core.asset.type.soundevent.config.SoundEvent;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.msgpack.core.MessagePacker;

/**
 * Sound-event table: runtime sound-event index to asset id, for the live 3-D sound probe.
 *
 * <p>Runtime ids are the {@code SoundEvent} asset-map indices, the same ids a
 * {@code PlaySoundEvent3D} call carries. The table is global to the process, like
 * {@link BlockTypeTableCodec}.
 */
public final class SoundEventTableCodec {

    public static final String SCHEMA = "hytalerl_sound_event_table_v1";
    public static final int VERSION = 1;

    private SoundEventTableCodec() {}

    /** One table row: a runtime index and the asset id it names. */
    public record Entry(int index, String assetId) {}

    /** The live table, sorted by index. */
    public static List<Entry> liveEntries() {
        var assets = SoundEvent.getAssetMap();
        List<Entry> rows = new ArrayList<>();
        for (String assetId : assets.getAssetMap().keySet()) {
            rows.add(new Entry(assets.getIndex(assetId), assetId));
        }
        rows.sort(Comparator.comparingInt(Entry::index));
        return rows;
    }

    public static void sendSoundEventTable(
        DataOutputStream output,
        List<Entry> entries
    ) throws IOException {
        sendFrame(output, packer -> pack(packer, entries));
    }

    public static void pack(MessagePacker packer, List<Entry> entries)
        throws IOException {
        packer.packMapHeader(5);
        packer.packString("type");
        packer.packString("sound_event_table");
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
            packer.packInt(entry.index());
            packer.packString(entry.assetId());
        }
    }
}
