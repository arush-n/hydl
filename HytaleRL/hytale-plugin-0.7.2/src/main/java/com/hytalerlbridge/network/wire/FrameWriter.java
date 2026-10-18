package com.hytalerlbridge.network.wire;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import org.msgpack.core.MessageBufferPacker;
import org.msgpack.core.MessagePack;
import org.msgpack.core.MessagePacker;
import org.msgpack.core.MessageUnpacker;
import org.msgpack.value.Value;

/**
 * Length-prefixed MessagePack frame IO.
 *
 * <p>Extracted verbatim from {@code ClientHandler}, along with the
 * {@code MAX_MESSAGE_BYTES} bound and the {@code PackerWriter} callback, both of
 * which were used only by these methods.
 */
public final class FrameWriter {

    public static final int MAX_MESSAGE_BYTES = 16 * 1024 * 1024;

    private FrameWriter() {}

    /** Writes the body of one outgoing frame. */
    @FunctionalInterface
    public interface PackerWriter {
        void write(MessagePacker packer) throws IOException;
    }

    public static Value readFrame(DataInputStream input) throws IOException {
        int length = input.readInt();
        if (length <= 0 || length > MAX_MESSAGE_BYTES) {
            throw new IllegalArgumentException(
                "Invalid MessagePack frame length: " + length
            );
        }
        byte[] payload = input.readNBytes(length);
        if (payload.length != length) {
            throw new EOFException("Connection closed during MessagePack frame");
        }

        try (MessageUnpacker unpacker = MessagePack.newDefaultUnpacker(payload)) {
            Value value = unpacker.unpackValue();
            if (unpacker.hasNext()) {
                throw new IllegalArgumentException("MessagePack frame contains trailing values");
            }
            return value;
        }
    }

    public static void sendFrame(DataOutputStream output, PackerWriter writer)
        throws IOException {
        byte[] payload;
        try (MessageBufferPacker packer = MessagePack.newDefaultBufferPacker()) {
            writer.write(packer);
            packer.flush();
            payload = packer.toByteArray();
        }
        if (payload.length > MAX_MESSAGE_BYTES) {
            throw new IllegalArgumentException(
                "Outgoing MessagePack frame exceeds "
                    + MAX_MESSAGE_BYTES
                    + " bytes"
            );
        }
        output.writeInt(payload.length);
        output.write(payload);
        output.flush();
    }

    public static void sendError(DataOutputStream output, String message) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(2);
            packer.packString("type");
            packer.packString("error");
            packer.packString("message");
            packer.packString(message == null ? "Unknown bridge error" : message);
        });
    }

    public static void sendAck(DataOutputStream output) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(1);
            packer.packString("type");
            packer.packString("ack");
        });
    }

    public static void packInfoValue(MessagePacker packer, Object value) throws IOException {
        if (value == null) {
            packer.packNil();
        } else if (value instanceof Boolean bool) {
            packer.packBoolean(bool);
        } else if (value instanceof Byte || value instanceof Short || value instanceof Integer) {
            packer.packInt(((Number) value).intValue());
        } else if (value instanceof Long number) {
            packer.packLong(number);
        } else if (value instanceof Number number) {
            packer.packDouble(number.doubleValue());
        } else {
            packer.packString(value.toString());
        }
    }
}
