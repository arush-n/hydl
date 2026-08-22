package com.hytalerlbridge.policy.perception.projection;

import com.hytalerlbridge.policy.Policy;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Actor-safe structured recipe rows to the learner-v3 32-value embedding.
 *
 * <p>This is a literal float32 port of {@code encode_recipe_candidates}. The
 * native recipe string, global recipe index, execution hash, diagnostics and
 * source counts are intentionally absent: those values are privileged lookup
 * state and must never enter a policy observation.
 */
public final class RecipeCandidateProjection {

    public static final int CANDIDATE_CAPACITY = 16;
    public static final int INGREDIENT_CAPACITY = 28;
    public static final int OUTPUT_CAPACITY = 4;
    public static final int REQUIREMENT_CAPACITY = 3;
    public static final int METADATA_HASH_WORDS = 2;
    public static final int IDENTITY_HASH_WORDS = 8;
    public static final int EMBEDDING_SIZE = 32;

    public static final int INPUT_FEATURE_SIZE = 130;
    public static final int OUTPUT_FEATURE_SIZE = 97;
    public static final int REQUIREMENT_FEATURE_SIZE = 289;
    public static final int CANDIDATE_SCALAR_SIZE = 6;
    public static final int CANDIDATE_FEATURE_SIZE =
        3 * EMBEDDING_SIZE + CANDIDATE_SCALAR_SIZE;
    public static final int EMBEDDING_VALUE_COUNT =
        CANDIDATE_CAPACITY * EMBEDDING_SIZE;

    public static final String ENCODING_CONTRACT_SHA256 =
        "E74F568D196B46A30E2850341C40C8BA51E7CEE5EA70D7768A8EA81DCF77DE65";

    private static final String CONTRACT_FILE = "recipe_encoder_contract.tsv";

    /** One dense projection in row-major {@code [input, output]} order. */
    public record Dense(float[] kernel, float[] bias, int inputSize) {
        public Dense {
            require(inputSize > 0, "dense input size must be positive");
            require(kernel.length == inputSize * EMBEDDING_SIZE,
                "dense kernel width drift");
            require(bias.length == EMBEDDING_SIZE,
                "dense bias width drift");
            requireFinite(kernel, "dense kernel");
            requireFinite(bias, "dense bias");
            kernel = kernel.clone();
            bias = bias.clone();
        }

        @Override public float[] kernel() { return kernel.clone(); }
        @Override public float[] bias() { return bias.clone(); }
    }

    /** Exact checkpoint-pinned encoder weights. */
    public record Parameters(
        Dense inputProjection,
        Dense outputProjection,
        Dense requirementProjection,
        Dense candidateProjection
    ) {
        public Parameters {
            require(inputProjection.inputSize() == INPUT_FEATURE_SIZE,
                "input projection feature width drift");
            require(outputProjection.inputSize() == OUTPUT_FEATURE_SIZE,
                "output projection feature width drift");
            require(requirementProjection.inputSize() == REQUIREMENT_FEATURE_SIZE,
                "requirement projection feature width drift");
            require(candidateProjection.inputSize() == CANDIDATE_FEATURE_SIZE,
                "candidate projection feature width drift");
        }

        /** Load exact little-endian float32 weights exported beside a profile. */
        public static Parameters load(Path root) throws IOException {
            verifyContract(root.resolve(CONTRACT_FILE));
            return new Parameters(
                loadDense(root, "recipe_encoder_input", INPUT_FEATURE_SIZE),
                loadDense(root, "recipe_encoder_output", OUTPUT_FEATURE_SIZE),
                loadDense(
                    root, "recipe_encoder_requirement", REQUIREMENT_FEATURE_SIZE),
                loadDense(
                    root, "recipe_encoder_candidate", CANDIDATE_FEATURE_SIZE)
            );
        }
    }

    /** One actor's bounded, policy-visible recipe candidates. */
    public record Input(
        boolean available,
        boolean[] candidateMask,
        boolean[] inputMask,
        int[] inputItemId,
        int[] inputResourceTypeId,
        int[] inputQuantity,
        boolean[] inputMetadataRequired,
        int[] inputMetadataHash,
        boolean[] outputMask,
        int[] outputItemId,
        int[] outputQuantity,
        int[] outputMetadataHash,
        boolean[] requirementMask,
        int[] requirementBenchType,
        int[] requirementBenchIdHash,
        int[] requirementTierLevel,
        boolean[] knowledgeRequired,
        int[] requiredMemoriesLevel,
        float[] timeSeconds
    ) {
        public Input {
            int inputs = CANDIDATE_CAPACITY * INGREDIENT_CAPACITY;
            int outputs = CANDIDATE_CAPACITY * OUTPUT_CAPACITY;
            int requirements = CANDIDATE_CAPACITY * REQUIREMENT_CAPACITY;
            requireWidth(candidateMask, CANDIDATE_CAPACITY, "candidate mask");
            requireWidth(inputMask, inputs, "input mask");
            requireWidth(inputItemId, inputs, "input item id");
            requireWidth(inputResourceTypeId, inputs, "input resource id");
            requireWidth(inputQuantity, inputs, "input quantity");
            requireWidth(inputMetadataRequired, inputs,
                "input metadata required");
            requireWidth(inputMetadataHash, inputs * METADATA_HASH_WORDS,
                "input metadata hash");
            requireWidth(outputMask, outputs, "output mask");
            requireWidth(outputItemId, outputs, "output item id");
            requireWidth(outputQuantity, outputs, "output quantity");
            requireWidth(outputMetadataHash, outputs * METADATA_HASH_WORDS,
                "output metadata hash");
            requireWidth(requirementMask, requirements, "requirement mask");
            requireWidth(requirementBenchType, requirements,
                "requirement bench type");
            requireWidth(requirementBenchIdHash,
                requirements * IDENTITY_HASH_WORDS,
                "requirement bench id hash");
            requireWidth(requirementTierLevel, requirements,
                "requirement tier");
            requireWidth(knowledgeRequired, CANDIDATE_CAPACITY,
                "knowledge required");
            requireWidth(requiredMemoriesLevel, CANDIDATE_CAPACITY,
                "required memories");
            requireWidth(timeSeconds, CANDIDATE_CAPACITY, "recipe time");
            requireFinite(timeSeconds, "recipe time");

            candidateMask = candidateMask.clone();
            inputMask = inputMask.clone();
            inputItemId = inputItemId.clone();
            inputResourceTypeId = inputResourceTypeId.clone();
            inputQuantity = inputQuantity.clone();
            inputMetadataRequired = inputMetadataRequired.clone();
            inputMetadataHash = inputMetadataHash.clone();
            outputMask = outputMask.clone();
            outputItemId = outputItemId.clone();
            outputQuantity = outputQuantity.clone();
            outputMetadataHash = outputMetadataHash.clone();
            requirementMask = requirementMask.clone();
            requirementBenchType = requirementBenchType.clone();
            requirementBenchIdHash = requirementBenchIdHash.clone();
            requirementTierLevel = requirementTierLevel.clone();
            knowledgeRequired = knowledgeRequired.clone();
            requiredMemoriesLevel = requiredMemoriesLevel.clone();
            timeSeconds = timeSeconds.clone();
        }

        public static Input unavailable() {
            return new Input(
                false,
                new boolean[CANDIDATE_CAPACITY],
                new boolean[CANDIDATE_CAPACITY * INGREDIENT_CAPACITY],
                new int[CANDIDATE_CAPACITY * INGREDIENT_CAPACITY],
                new int[CANDIDATE_CAPACITY * INGREDIENT_CAPACITY],
                new int[CANDIDATE_CAPACITY * INGREDIENT_CAPACITY],
                new boolean[CANDIDATE_CAPACITY * INGREDIENT_CAPACITY],
                new int[CANDIDATE_CAPACITY * INGREDIENT_CAPACITY
                    * METADATA_HASH_WORDS],
                new boolean[CANDIDATE_CAPACITY * OUTPUT_CAPACITY],
                new int[CANDIDATE_CAPACITY * OUTPUT_CAPACITY],
                new int[CANDIDATE_CAPACITY * OUTPUT_CAPACITY],
                new int[CANDIDATE_CAPACITY * OUTPUT_CAPACITY
                    * METADATA_HASH_WORDS],
                new boolean[CANDIDATE_CAPACITY * REQUIREMENT_CAPACITY],
                new int[CANDIDATE_CAPACITY * REQUIREMENT_CAPACITY],
                new int[CANDIDATE_CAPACITY * REQUIREMENT_CAPACITY
                    * IDENTITY_HASH_WORDS],
                new int[CANDIDATE_CAPACITY * REQUIREMENT_CAPACITY],
                new boolean[CANDIDATE_CAPACITY],
                new int[CANDIDATE_CAPACITY],
                new float[CANDIDATE_CAPACITY]
            );
        }
    }

    public record Result(
        boolean available,
        boolean[] candidateMask,
        float[] candidateEmbedding
    ) {
        public Result {
            requireWidth(candidateMask, CANDIDATE_CAPACITY, "candidate mask");
            requireWidth(candidateEmbedding, EMBEDDING_VALUE_COUNT,
                "candidate embedding");
            candidateMask = candidateMask.clone();
            candidateEmbedding = candidateEmbedding.clone();
        }

        @Override public boolean[] candidateMask() {
            return candidateMask.clone();
        }

        @Override public float[] candidateEmbedding() {
            return candidateEmbedding.clone();
        }
    }

    private RecipeCandidateProjection() {
    }

    /** Match JAX's hierarchical encoder without exposing native recipe IDs. */
    public static Result project(Parameters parameters, Input input) {
        boolean[] candidateMask = input.candidateMask.clone();
        float[] embedding = new float[EMBEDDING_VALUE_COUNT];
        for (int candidate = 0; candidate < CANDIDATE_CAPACITY; candidate++) {
            candidateMask[candidate] &= input.available;
            if (!candidateMask[candidate]) {
                continue;
            }

            float[] inputMean = inputMean(parameters.inputProjection, input,
                candidate);
            float[] outputMean = outputMean(parameters.outputProjection, input,
                candidate);
            float[] requirementMean = requirementMean(
                parameters.requirementProjection, input, candidate);
            float[] features = new float[CANDIDATE_FEATURE_SIZE];
            System.arraycopy(inputMean, 0, features, 0, EMBEDDING_SIZE);
            System.arraycopy(outputMean, 0, features, EMBEDDING_SIZE,
                EMBEDDING_SIZE);
            System.arraycopy(requirementMean, 0, features, 2 * EMBEDDING_SIZE,
                EMBEDDING_SIZE);
            int scalar = 3 * EMBEDDING_SIZE;
            features[scalar] = input.knowledgeRequired[candidate] ? 1.0f : 0.0f;
            features[scalar + 1] = bounded(
                input.requiredMemoriesLevel[candidate]);
            features[scalar + 2] = bounded(input.timeSeconds[candidate]);
            features[scalar + 3] = (float) countInputs(input, candidate)
                / (float) INGREDIENT_CAPACITY;
            features[scalar + 4] = (float) countOutputs(input, candidate)
                / (float) OUTPUT_CAPACITY;
            features[scalar + 5] = (float) countRequirements(input, candidate)
                / (float) REQUIREMENT_CAPACITY;

            float[] projected = linearTanh(
                parameters.candidateProjection, features);
            System.arraycopy(projected, 0, embedding,
                candidate * EMBEDDING_SIZE, EMBEDDING_SIZE);
        }
        return new Result(input.available, candidateMask, embedding);
    }

    private static float[] inputMean(Dense dense, Input input, int candidate) {
        float[] sum = new float[EMBEDDING_SIZE];
        int count = 0;
        boolean[] mask = input.inputMask;
        int[] item = input.inputItemId;
        int[] resource = input.inputResourceTypeId;
        int[] quantity = input.inputQuantity;
        boolean[] metadataRequired = input.inputMetadataRequired;
        int[] metadataHash = input.inputMetadataHash;
        int base = candidate * INGREDIENT_CAPACITY;
        for (int local = 0; local < INGREDIENT_CAPACITY; local++) {
            int row = base + local;
            if (!mask[row]) {
                continue;
            }
            float[] features = new float[INPUT_FEATURE_SIZE];
            writeBits(features, 0, item[row]);
            writeBits(features, 32, resource[row]);
            features[64] = bounded(quantity[row]);
            features[65] = metadataRequired[row] ? 1.0f : 0.0f;
            writeHashBits(features, 66, metadataHash,
                row * METADATA_HASH_WORDS, METADATA_HASH_WORDS);
            add(sum, linearTanh(dense, features));
            count++;
        }
        divide(sum, count);
        return sum;
    }

    private static float[] outputMean(Dense dense, Input input, int candidate) {
        float[] sum = new float[EMBEDDING_SIZE];
        int count = 0;
        boolean[] mask = input.outputMask;
        int[] item = input.outputItemId;
        int[] quantity = input.outputQuantity;
        int[] metadataHash = input.outputMetadataHash;
        int base = candidate * OUTPUT_CAPACITY;
        for (int local = 0; local < OUTPUT_CAPACITY; local++) {
            int row = base + local;
            if (!mask[row]) {
                continue;
            }
            float[] features = new float[OUTPUT_FEATURE_SIZE];
            writeBits(features, 0, item[row]);
            features[32] = bounded(quantity[row]);
            writeHashBits(features, 33, metadataHash,
                row * METADATA_HASH_WORDS, METADATA_HASH_WORDS);
            add(sum, linearTanh(dense, features));
            count++;
        }
        divide(sum, count);
        return sum;
    }

    private static float[] requirementMean(
        Dense dense, Input input, int candidate
    ) {
        float[] sum = new float[EMBEDDING_SIZE];
        int count = 0;
        boolean[] mask = input.requirementMask;
        int[] bench = input.requirementBenchType;
        int[] identityHash = input.requirementBenchIdHash;
        int[] tier = input.requirementTierLevel;
        int base = candidate * REQUIREMENT_CAPACITY;
        for (int local = 0; local < REQUIREMENT_CAPACITY; local++) {
            int row = base + local;
            if (!mask[row]) {
                continue;
            }
            float[] features = new float[REQUIREMENT_FEATURE_SIZE];
            writeBits(features, 0, bench[row]);
            writeHashBits(features, 32, identityHash,
                row * IDENTITY_HASH_WORDS, IDENTITY_HASH_WORDS);
            features[288] = bounded(tier[row]);
            add(sum, linearTanh(dense, features));
            count++;
        }
        divide(sum, count);
        return sum;
    }

    private static float[] linearTanh(Dense dense, float[] features) {
        float[] kernel = dense.kernel;
        float[] bias = dense.bias;
        float[] result = new float[EMBEDDING_SIZE];
        for (int output = 0; output < EMBEDDING_SIZE; output++) {
            float value = 0.0f;
            for (int input = 0; input < features.length; input++) {
                value += features[input]
                    * kernel[input * EMBEDDING_SIZE + output];
            }
            value += bias[output];
            result[output] = (float) Math.tanh(value);
        }
        return result;
    }

    private static void writeBits(float[] out, int offset, int value) {
        for (int bit = 0; bit < Integer.SIZE; bit++) {
            out[offset + bit] = (float) ((value >>> bit) & 1);
        }
    }

    private static void writeHashBits(
        float[] out, int outputOffset, int[] words, int inputOffset, int count
    ) {
        for (int word = 0; word < count; word++) {
            writeBits(out, outputOffset + word * Integer.SIZE,
                words[inputOffset + word]);
        }
    }

    private static float bounded(int value) {
        return bounded((float) value);
    }

    private static float bounded(float value) {
        float numeric = Math.max(value, 0.0f);
        return numeric / (1.0f + numeric);
    }

    private static int countInputs(Input input, int candidate) {
        return count(input.inputMask, candidate * INGREDIENT_CAPACITY,
            INGREDIENT_CAPACITY);
    }

    private static int countOutputs(Input input, int candidate) {
        return count(input.outputMask, candidate * OUTPUT_CAPACITY,
            OUTPUT_CAPACITY);
    }

    private static int countRequirements(Input input, int candidate) {
        return count(input.requirementMask, candidate * REQUIREMENT_CAPACITY,
            REQUIREMENT_CAPACITY);
    }

    private static int count(boolean[] mask, int offset, int width) {
        int result = 0;
        for (int index = 0; index < width; index++) {
            if (mask[offset + index]) {
                result++;
            }
        }
        return result;
    }

    private static void add(float[] target, float[] value) {
        for (int index = 0; index < target.length; index++) {
            target[index] += value[index];
        }
    }

    private static void divide(float[] value, int count) {
        float divisor = (float) Math.max(count, 1);
        for (int index = 0; index < value.length; index++) {
            value[index] /= divisor;
        }
    }

    private static Dense loadDense(Path root, String prefix, int inputSize)
        throws IOException {
        return new Dense(
            Policy.read(root, prefix + "_kernel"),
            Policy.read(root, prefix + "_bias"),
            inputSize
        );
    }

    private static void verifyContract(Path path) throws IOException {
        Map<String, String> values = new HashMap<>();
        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] parts = line.split("\\t", -1);
            if (parts.length != 2 || parts[0].isBlank()
                || values.put(parts[0], parts[1]) != null) {
                throw new IOException("invalid recipe encoder contract row: " + line);
            }
        }
        requireContract(values, "schema",
            "hytalerl_recipe_candidate_encoder_profile_v1");
        requireContract(values, "encoding_contract_sha256",
            ENCODING_CONTRACT_SHA256);
        requireContract(values, "parameter_source", "contract_seed");
        requireContract(values, "parameter_seed_hex",
            ENCODING_CONTRACT_SHA256.substring(0, 8));
        requireContract(values, "candidate_capacity",
            Integer.toString(CANDIDATE_CAPACITY));
        requireContract(values, "embedding_size",
            Integer.toString(EMBEDDING_SIZE));
        requireContract(values, "input_feature_size",
            Integer.toString(INPUT_FEATURE_SIZE));
        requireContract(values, "output_feature_size",
            Integer.toString(OUTPUT_FEATURE_SIZE));
        requireContract(values, "requirement_feature_size",
            Integer.toString(REQUIREMENT_FEATURE_SIZE));
        requireContract(values, "candidate_feature_size",
            Integer.toString(CANDIDATE_FEATURE_SIZE));
    }

    private static void requireContract(
        Map<String, String> values, String name, String expected
    ) throws IOException {
        String actual = values.get(name);
        if (!expected.equals(actual)) {
            throw new IOException("recipe encoder " + name + " drift: "
                + actual + " != " + expected);
        }
    }

    private static void requireWidth(boolean[] value, int width, String name) {
        require(value.length == width,
            name + " width drift: " + value.length + " != " + width);
    }

    private static void requireWidth(int[] value, int width, String name) {
        require(value.length == width,
            name + " width drift: " + value.length + " != " + width);
    }

    private static void requireWidth(float[] value, int width, String name) {
        require(value.length == width,
            name + " width drift: " + value.length + " != " + width);
    }

    private static void requireFinite(float[] value, String name) {
        for (float item : value) {
            require(Float.isFinite(item), name + " contains a non-finite value");
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }
}
