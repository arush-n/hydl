package com.hytalerlbridge.network.wire;

import com.hytalerlbridge.worldgen.NativePerceptionChannels;
import com.hytalerlbridge.worldgen.NativeTraversalProbe;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.msgpack.core.MessagePacker;

/**
 * Little-endian binary payload encoding and decoding for the wire protocol.
 *
 * <p>Extracted verbatim from {@code ClientHandler}. These are leaves: they call
 * nothing else in the network package, so every other codec may depend on them
 * without a cycle.
 */
public final class BinaryCodec {

    private BinaryCodec() {}

    public static int[] decodeInt32LittleEndianTriples(byte[] payload) {
        if (
            payload.length == 0
                || payload.length % (3 * Integer.BYTES) != 0
                || payload.length / (3 * Integer.BYTES)
                    > NativePerceptionChannels.MAX_SAMPLES
        ) {
            throw new IllegalArgumentException(
                "positions_i32_le_xyz must contain 1.."
                    + NativePerceptionChannels.MAX_SAMPLES
                    + " XYZ triples"
            );
        }
        int[] values = new int[payload.length / Integer.BYTES];
        ByteBuffer buffer = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        for (int index = 0; index < values.length; index++) {
            values[index] = buffer.getInt();
        }
        return values;
    }

    public static double[] decodeFloat64LittleEndian(
        byte[] payload,
        int valuesPerSample,
        String name
    ) {
        int sampleBytes = valuesPerSample * Double.BYTES;
        if (
            payload.length == 0
                || payload.length % sampleBytes != 0
                || payload.length / sampleBytes > NativeTraversalProbe.MAX_SAMPLES
        ) {
            throw new IllegalArgumentException(
                name
                    + " must contain 1.."
                    + NativeTraversalProbe.MAX_SAMPLES
                    + " samples"
            );
        }
        double[] values = new double[payload.length / Double.BYTES];
        ByteBuffer buffer = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        for (int index = 0; index < values.length; index++) {
            values[index] = buffer.getDouble();
            if (!Double.isFinite(values[index])) {
                throw new IllegalArgumentException(name + " must be finite");
            }
        }
        return values;
    }

    public static byte[] encodeInt16LittleEndian(short[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Short.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (short value : values) buffer.putShort(value);
        return buffer.array();
    }

    public static byte[] encodeInt32LittleEndian(int[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Integer.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (int value : values) buffer.putInt(value);
        return buffer.array();
    }

    public static byte[] encodeInt64LittleEndian(long[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Long.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (long value : values) buffer.putLong(value);
        return buffer.array();
    }

    public static byte[] encodeFloat32LittleEndian(float[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) buffer.putFloat(value);
        return buffer.array();
    }

    public static byte[] encodeFloat64LittleEndian(double[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Double.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (double value : values) buffer.putDouble(value);
        return buffer.array();
    }

    public static void packBinary(
        MessagePacker packer,
        String name,
        byte[] payload
    ) throws IOException {
        packer.packString(name);
        packer.packBinaryHeader(payload.length);
        packer.writePayload(payload);
    }
}
