package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.geometry.trigger.TriggerContract;
import com.hytalerlbridge.geometry.trigger.TriggerFrame;
import java.io.IOException;
import org.msgpack.core.MessagePacker;

/** MessagePack encoding for the optional native trigger sidecar. */
public final class TriggerCodec {

    private TriggerCodec() {}

    public static void pack(MessagePacker packer, TriggerFrame frame)
        throws IOException {
        packer.packMapHeader(11);
        packer.packString("schema");
        packer.packString(TriggerContract.SCHEMA);
        packer.packString("version");
        packer.packInt(TriggerContract.VERSION);
        packer.packString("available");
        packer.packBoolean(frame.available());
        packer.packString("data_unavailable");
        packer.packBoolean(frame.dataUnavailable());
        packer.packString("capacity_exceeded");
        packer.packBoolean(frame.capacityExceeded());
        packer.packString("program_capacity");
        packer.packInt(TriggerContract.MAX_PROGRAMS_0_5_7);
        packer.packString("cell_capacity");
        packer.packInt(TriggerContract.MAX_CELLS);
        packer.packString("box_capacity_per_cell");
        packer.packInt(TriggerContract.MAX_BOXES_0_5_7);
        packer.packString("program_catalog_sha256");
        packer.packString(frame.programCatalogSha256());
        packer.packString("program_ids");
        packer.packArrayHeader(frame.programIds().size());
        for (String programId : frame.programIds()) {
            packer.packString(programId);
        }
        packer.packString("cells");
        packer.packArrayHeader(frame.cells().size());
        for (TriggerFrame.Cell cell : frame.cells()) packCell(packer, cell);
    }

    private static void packCell(MessagePacker packer, TriggerFrame.Cell cell)
        throws IOException {
        packer.packArrayHeader(10);
        packer.packInt(cell.dx());
        packer.packInt(cell.dy());
        packer.packInt(cell.dz());
        packer.packInt(cell.baseOffsetX());
        packer.packInt(cell.baseOffsetY());
        packer.packInt(cell.baseOffsetZ());
        packer.packInt(cell.enterProgram());
        packer.packInt(cell.collisionProgram());
        packer.packInt(cell.leaveProgram());
        double[] boxes = cell.boxes();
        packer.packArrayHeader(boxes.length);
        for (double value : boxes) packer.packDouble(value);
    }
}
