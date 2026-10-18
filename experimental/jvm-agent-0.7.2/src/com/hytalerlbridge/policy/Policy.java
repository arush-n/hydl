package com.hytalerlbridge.policy;

import com.hytalerlbridge.policy.action.StandardRootSelector;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * The JAX recurrent policy, forward pass only, in plain Java.
 *
 * <p>Deliberately dependency-free: no ONNX runtime, no tensor library, no JSON
 * parser. The network is an MLP-GRU actor-critic -- two dense layers, one GRU
 * cell, a linear actor head and a linear critic head -- so the whole of
 * inference is four matrix-vector products and some element-wise math. Pulling
 * in an inference runtime to execute that would add a numerics surface that
 * would then itself need certifying against JAX.
 *
 * <p>Shapes are derived from file lengths rather than read from a manifest.
 * Encoder and recurrent widths are independent and every tensor is checked
 * against both derived dimensions before inference.
 *
 * <p>Mirrors {@code hytalegym/jax/training/policy.py:apply_policy}. Note the
 * GRU output convention is {@code update * h + (1 - update) * candidate},
 * which is the opposite of the usual Keras ordering -- getting this backwards
 * still produces plausible-looking logits, so it is asserted numerically
 * against JAX rather than assumed.
 */
public final class Policy {

    /**
     * Pinned action-surface contract: 12 heads summing to 124 logits.
     *
     * <p>Still twelve heads, but NOT the same twelve. Against the previous
     * {@code {12,17,2,5,2,9,9,5,2,2,17,17}} = 99 layout:
     *
     * <ul>
     *   <li>the separate {@code dodge} (5) and {@code world_move} (9) heads
     *       merged into one 37-wide {@code locomotion_gait_compass}, which also
     *       carries gait -- they occupy the same physical slot, so a single
     *       categorical is what makes them mutually exclusive by construction;
     *   <li>{@code body_yaw_delta_bins} (9) is new, and is NOT a duplicate of
     *       {@code yaw_delta_bins}: head yaw aims, body yaw steers, and they
     *       are independent controls;
     *   <li>{@code hotbar_none_plus_slots} (10) is new;
     *   <li>the {@code recipe} head (17) is no longer published for a combat
     *       agent.
     * </ul>
     *
     * <p>Order matters as much as width -- these are concatenated logits, so a
     * head in the wrong position silently reads another head's values.
     * {@code ActionDecoder} holds the index constants; this is the only place
     * the widths are written.
     */
    public static final int[] HEAD_SIZES = {12, 17, 2, 2, 37, 9, 9, 5, 10, 2, 2, 17};

    private final float[] encoderInputKernel;   // [observation][encoder]
    private final float[] encoderInputBias;     // [encoder]
    private final float[] encoderHiddenKernel;  // [encoder][encoder]
    private final float[] encoderHiddenBias;    // [encoder]
    private final float[] gruInputKernel;       // [encoder][3 * recurrent]
    private final float[] gruRecurrentKernel;   // [recurrent][3 * recurrent]
    private final float[] gruBias;              // [3 * recurrent]
    private final float[] actorKernel;          // [recurrent][actions]
    private final float[] actorBias;            // [actions]
    private final float[] criticKernel;         // [recurrent][1]
    private final float[] criticBias;           // [1]

    public final int observationSize;
    public final int encoderSize;
    public final int recurrentSize;
    /** @deprecated use {@link #recurrentSize}; retained for fixture callers. */
    @Deprecated
    public final int hiddenSize;
    public final int actionSize;

    private Policy(Path directory) throws IOException {
        encoderInputBias = read(directory, "encoder_input_bias");
        actorBias = read(directory, "actor_bias");
        encoderSize = encoderInputBias.length;
        actionSize = actorBias.length;

        encoderInputKernel = read(directory, "encoder_input_kernel");
        expect(encoderSize > 0, "encoder width");
        expect(actionSize > 0, "action width");
        observationSize = encoderInputKernel.length / encoderSize;

        encoderHiddenKernel = read(directory, "encoder_hidden_kernel");
        encoderHiddenBias = read(directory, "encoder_hidden_bias");
        gruInputKernel = read(directory, "gru_input_kernel");
        gruRecurrentKernel = read(directory, "gru_recurrent_kernel");
        gruBias = read(directory, "gru_bias");
        actorKernel = read(directory, "actor_kernel");
        criticKernel = read(directory, "critic_kernel");
        criticBias = read(directory, "critic_bias");

        expect(gruBias.length > 0 && gruBias.length % 3 == 0, "gru_bias");
        recurrentSize = gruBias.length / 3;
        // Historical callers used hiddenSize for the GRU carry. Keep that API
        // meaning while allowing the preceding encoder to have its own width.
        hiddenSize = recurrentSize;

        expect(encoderInputKernel.length == observationSize * encoderSize, "encoder_input");
        expect(encoderHiddenBias.length == encoderSize, "encoder_hidden_bias");
        expect(encoderHiddenKernel.length == encoderSize * encoderSize, "encoder_hidden");
        expect(gruInputKernel.length == encoderSize * 3 * recurrentSize, "gru_input");
        expect(gruRecurrentKernel.length == recurrentSize * 3 * recurrentSize, "gru_recurrent");
        expect(actorKernel.length == recurrentSize * actionSize, "actor");
        expect(criticKernel.length == recurrentSize, "critic");
        expect(criticBias.length == 1, "critic_bias");
        int total = 0;
        for (int size : HEAD_SIZES) {
            total += size;
        }
        expect(total == actionSize, "action head sizes must sum to the actor width");
    }

    public static Policy load(Path directory) throws IOException {
        return new Policy(directory);
    }

    /** One recurrent step. Returns logits; {@code carry} is updated in place. */
    public float[] step(float[] observation, float[] carry, boolean[] legal) {
        expect(observation.length == observationSize, "observation width");
        expect(carry.length == recurrentSize, "recurrent width");
        expect(legal.length == actionSize, "action mask width");

        float[] encoded = tanh(affine(observation, encoderInputKernel,
                encoderInputBias, observationSize, encoderSize));
        encoded = tanh(affine(encoded, encoderHiddenKernel, encoderHiddenBias,
                encoderSize, encoderSize));

        int gates = 3 * recurrentSize;
        float[] fromInput = affine(encoded, gruInputKernel, gruBias, encoderSize, gates);
        float[] fromCarry = affine(carry, gruRecurrentKernel, null, recurrentSize, gates);

        for (int i = 0; i < recurrentSize; i++) {
            float reset = sigmoid(fromInput[i] + fromCarry[i]);
            float update = sigmoid(
                    fromInput[recurrentSize + i] + fromCarry[recurrentSize + i]);
            float candidate = (float) Math.tanh(
                    fromInput[2 * recurrentSize + i]
                            + reset * fromCarry[2 * recurrentSize + i]);
            carry[i] = update * carry[i] + (1.0f - update) * candidate;
        }

        float[] logits = affine(
                carry, actorKernel, actorBias, recurrentSize, actionSize);
        for (int i = 0; i < actionSize; i++) {
            if (!legal[i]) {
                logits[i] = -1.0e9f;
            }
        }
        return logits;
    }

    public float value(float[] carry) {
        expect(carry.length == recurrentSize, "recurrent width");
        return affine(carry, criticKernel, criticBias, recurrentSize, 1)[0];
    }

    /** Greedy decode under the one-standard-root constraint. */
    public static int[] factors(float[] logits) {
        return StandardRootSelector.select(logits, HEAD_SIZES);
    }

    // -- plumbing -----------------------------------------------------------

    /** {@code inputs @ kernel + bias} for a row-major [in][out] kernel. */
    private static float[] affine(float[] inputs, float[] kernel, float[] bias,
                                  int inputSize, int outputSize) {
        float[] out = new float[outputSize];
        if (bias != null) {
            System.arraycopy(bias, 0, out, 0, outputSize);
        }
        for (int i = 0; i < inputSize; i++) {
            float value = inputs[i];
            if (value == 0.0f) {
                continue;  // the observation is ~99% zeros; skip the row
            }
            int row = i * outputSize;
            for (int j = 0; j < outputSize; j++) {
                out[j] += value * kernel[row + j];
            }
        }
        return out;
    }

    private static float[] tanh(float[] values) {
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) Math.tanh(values[i]);
        }
        return values;
    }

    private static float sigmoid(float value) {
        return (float) (1.0 / (1.0 + Math.exp(-value)));
    }

    /** Load a little-endian float32 array written by the exporters. */
    public static float[] read(Path directory, String name) throws IOException {
        byte[] bytes = Files.readAllBytes(directory.resolve(name + ".bin"));
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[bytes.length / Float.BYTES];
        buffer.asFloatBuffer().get(values);
        return values;
    }

    /** Load a byte-per-entry boolean array written by the exporters. */
    public static boolean[] readMask(Path directory, String name) throws IOException {
        byte[] bytes = Files.readAllBytes(directory.resolve(name + ".bin"));
        boolean[] values = new boolean[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            values[i] = bytes[i] != 0;
        }
        return values;
    }

    private static void expect(boolean condition, String what) {
        if (!condition) {
            throw new IllegalStateException("policy weights inconsistent: " + what);
        }
    }
}
