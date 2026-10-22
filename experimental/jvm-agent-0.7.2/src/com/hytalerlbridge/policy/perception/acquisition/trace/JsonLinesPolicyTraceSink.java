package com.hytalerlbridge.policy.perception.acquisition.trace;

import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.Policy;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

/**
 * Bounded JSONL trace of the policy's native observation/action transitions.
 *
 * <p>Each transition row contains the observation and legality mask sampled at
 * tick {@code t}, the raw policy factors and selected semantic action applied
 * at {@code t}, and the next observation/mask when the next sample for that
 * slot is the immediately following world tick. A gap is recorded explicitly
 * rather than silently pairing unrelated samples. Arrays use base64 encoded
 * little-endian float32/uint8 so the 15,656-column learner row remains exact
 * without making JSON parsing the numerical contract.</p>
 *
 * <p>The recorder is opt-in, bounded, and failure-isolated. The configured
 * limit applies to nonterminal rows; one terminal row may follow that budget
 * so the trace can still explain why control stopped. A full disk or malformed
 * callback disables evidence collection and never suppresses a policy tick.</p>
 */
public final class JsonLinesPolicyTraceSink implements PolicyTraceSink {

    public static final String SCHEMA = "hytale_policy_native_policy_trace_v1";
    public static final int DEFAULT_MAXIMUM_TRANSITIONS = 512;

    private static final System.Logger LOGGER =
        System.getLogger(JsonLinesPolicyTraceSink.class.getName());

    private final Path path;
    private final int maximumTransitions;
    private final int observationSize;
    private final int actionSize;
    private final Map<Integer, Pending> pending = new HashMap<>();
    private int transitionsWritten;
    private int nonterminalTransitionsWritten;
    private int terminalRowsWritten;
    private boolean disabled;
    private boolean closed;

    public JsonLinesPolicyTraceSink(
        Path directory,
        int maximumTransitions,
        int observationSize,
        int actionSize,
        String role,
        String profileSha256
    ) throws IOException {
        if (directory == null || maximumTransitions < 1
                || observationSize < 1 || actionSize < 1) {
            throw new IllegalArgumentException(
                "policy trace requires directory, positive limit, and widths");
        }
        Files.createDirectories(directory);
        path = directory.resolve(
            "policy-trace-" + System.currentTimeMillis() + ".jsonl");
        this.maximumTransitions = maximumTransitions;
        this.observationSize = observationSize;
        this.actionSize = actionSize;
        writeLine(header(role, profileSha256));
    }

    /** Read the optional positive row limit stored in the policy-trace marker. */
    public static int readMaximumTransitions(Path marker) throws IOException {
        String configured = Files.readString(marker, StandardCharsets.UTF_8).trim();
        if (configured.isEmpty()) {
            return DEFAULT_MAXIMUM_TRANSITIONS;
        }
        if (!configured.matches("[0-9]+")) {
            throw new IllegalArgumentException(
                "policy-trace maximum transitions must be a positive integer");
        }
        final int maximum;
        try {
            maximum = Integer.parseInt(configured);
        } catch (NumberFormatException invalid) {
            throw new IllegalArgumentException(
                "policy-trace maximum transitions is outside the integer range",
                invalid);
        }
        if (maximum < 1) {
            throw new IllegalArgumentException(
                "policy-trace maximum transitions must be positive");
        }
        return maximum;
    }

    /** Unique trace file for this process start. */
    public Path path() {
        return path;
    }

    @Override
    public synchronized long transitionsWritten() {
        return transitionsWritten;
    }

    @Override
    public synchronized void observe(
        int policySlot,
        long worldTick,
        float deltaTime,
        float[] observation,
        boolean[] actionMask,
        int[] inferredFactors,
        ActionDecoder.Decoded action,
        boolean fresh,
        boolean constrained,
        String route
    ) {
        if (closed || disabled || terminalRowsWritten > 0
                || observation == null || actionMask == null
                || inferredFactors == null || action == null || route == null) {
            return;
        }
        if (observation.length != observationSize
                || actionMask.length != actionSize
                || inferredFactors.length != Policy.HEAD_SIZES.length) {
            disable("callback width mismatch");
            return;
        }
        Pending current = new Pending(
            policySlot,
            worldTick,
            deltaTime,
            observation,
            actionMask,
            inferredFactors,
            action,
            fresh,
            constrained,
            route
        );
        Pending previous = pending.put(policySlot, current);
        if (previous == null) {
            return;
        }
        if (nonterminalTransitionsWritten >= maximumTransitions) {
            return;
        }
        if (worldTick == previous.worldTick + 1L) {
            writeTransition(previous, current, null, false);
        } else {
            writeTransition(previous, null, "world_tick_discontinuity", false);
        }
        if (disabled) {
            pending.clear();
        }
    }

    @Override
    public synchronized void skip(
        int policySlot,
        long worldTick,
        String reason,
        boolean terminal
    ) {
        if (closed || disabled || terminalRowsWritten > 0) {
            return;
        }
        Pending previous = pending.remove(policySlot);
        if (previous != null) {
            if (!terminal
                    && nonterminalTransitionsWritten >= maximumTransitions) {
                return;
            }
            writeTransition(
                previous,
                null,
                terminal ? "terminal:" + safe(reason) : safe(reason),
                terminal
            );
        }
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        for (Pending previous : pending.values()) {
            writeTransition(previous, null, "trace_end", false);
            if (disabled
                    || nonterminalTransitionsWritten >= maximumTransitions) {
                break;
            }
        }
        pending.clear();
        closed = true;
    }

    private void writeTransition(
        Pending current,
        Pending next,
        String gapReason,
        boolean terminal
    ) {
        if (disabled
                || terminalRowsWritten >= 1
                || (!terminal
                    && nonterminalTransitionsWritten >= maximumTransitions)) {
            return;
        }
        StringBuilder line = new StringBuilder(1024);
        line.append('{')
            .append("\"schema\":").append(quote(SCHEMA))
            .append(",\"event\":\"transition\"")
            .append(",\"sequence\":").append(transitionsWritten + 1)
            .append(",\"policy_slot\":").append(current.policySlot)
            .append(",\"world_tick\":").append(current.worldTick)
            .append(",\"delta_seconds\":").append(number(current.deltaTime))
            .append(",\"fresh_decision\":").append(current.fresh)
            .append(",\"control_constrained\":")
            .append(current.constrained)
            .append(",\"control_route\":").append(quote(current.route))
            .append(",\"inferred_factors\":")
            .append(factors(current.inferredFactors))
            .append(",\"observation_f32_le_b64\":")
            .append(quote(encodeFloats(current.observation)))
            .append(",\"action_mask_u8_b64\":")
            .append(quote(encodeMask(current.actionMask)))
            .append(",\"action\":");
        appendAction(line, current.action);
        line.append(",\"next_observation_available\":")
            .append(next != null);
        if (next != null) {
            line.append(",\"next_world_tick\":").append(next.worldTick)
                .append(",\"next_delta_seconds\":")
                .append(number(next.deltaTime))
                .append(",\"next_observation_f32_le_b64\":")
                .append(quote(encodeFloats(next.observation)))
                .append(",\"next_action_mask_u8_b64\":")
                .append(quote(encodeMask(next.actionMask)));
        } else {
            line.append(",\"terminal\":").append(terminal)
                .append(",\"gap_reason\":").append(quote(safe(gapReason)));
        }
        line.append('}');
        try {
            writeLine(line.toString());
            transitionsWritten++;
            if (terminal) {
                terminalRowsWritten++;
            } else {
                nonterminalTransitionsWritten++;
            }
        } catch (IOException failure) {
            disable("write failure: " + failure);
        }
    }

    private void writeLine(String line) throws IOException {
        Files.writeString(
            path,
            line + "\n",
            StandardCharsets.UTF_8,
            StandardOpenOption.CREATE,
            StandardOpenOption.APPEND
        );
    }

    private String header(String role, String profileSha256) {
        StringBuilder line = new StringBuilder(256);
        line.append('{')
            .append("\"schema\":").append(quote(SCHEMA))
            .append(",\"event\":\"header\"")
            .append(",\"observation_size\":").append(observationSize)
            .append(",\"action_size\":").append(actionSize)
            .append(",\"maximum_transitions\":")
            .append(maximumTransitions)
            .append(",\"head_sizes\":").append(factors(Policy.HEAD_SIZES))
            .append(",\"float_encoding\":\"float32-le\"")
            .append(",\"mask_encoding\":\"uint8\"")
            .append(",\"role\":").append(quote(role == null ? "*" : role))
            .append(",\"profile_sha256\":")
            .append(quote(profileSha256 == null ? "" : profileSha256))
            .append('}');
        return line.toString();
    }

    private static void appendAction(
        StringBuilder line,
        ActionDecoder.Decoded action
    ) {
        line.append('{')
            .append("\"action_legal\":").append(action.actionLegal())
            .append(",\"skill_id\":").append(action.skillId())
            .append(",\"ability_slot\":").append(action.abilitySlot())
            .append(",\"guard_held\":").append(action.guardHeld())
            .append(",\"dodge_direction\":").append(action.dodgeDirection())
            .append(",\"jump_held\":").append(action.jumpHeld())
            .append(",\"world_move_direction\":")
            .append(action.worldMoveDirection())
            .append(",\"gait\":").append(action.gait())
            .append(",\"yaw_delta_degrees\":")
            .append(number(action.yawDeltaDegrees()))
            .append(",\"body_yaw_delta_degrees\":")
            .append(number(action.bodyYawDeltaDegrees()))
            .append(",\"pitch_delta_degrees\":")
            .append(number(action.pitchDeltaDegrees()))
            .append(",\"hotbar_slot\":").append(action.hotbarSlot())
            .append(",\"use_requested\":").append(action.useRequested())
            .append(",\"block_interaction_trigger\":")
            .append(action.blockInteractionTrigger())
            .append(",\"recipe_candidate_index\":")
            .append(action.recipeCandidateIndex())
            .append(",\"block_candidate_index\":")
            .append(action.blockCandidateIndex())
            .append(",\"requests_world_verb\":")
            .append(action.requestsWorldVerb())
            .append(",\"forward\":").append(action.forward())
            .append(",\"back\":").append(action.back())
            .append(",\"left\":").append(action.left())
            .append(",\"right\":").append(action.right())
            .append(",\"attack\":").append(action.attack())
            .append('}');
    }

    private static String encodeFloats(float[] values) {
        ByteBuffer bytes = ByteBuffer
            .allocate(values.length * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) {
            bytes.putFloat(value);
        }
        return Base64.getEncoder().encodeToString(bytes.array());
    }

    private static String encodeMask(boolean[] values) {
        byte[] bytes = new byte[values.length];
        for (int index = 0; index < values.length; index++) {
            bytes[index] = (byte) (values[index] ? 1 : 0);
        }
        return Base64.getEncoder().encodeToString(bytes);
    }

    private static String factors(int[] values) {
        StringBuilder result = new StringBuilder(values.length * 3 + 2)
            .append('[');
        for (int index = 0; index < values.length; index++) {
            if (index > 0) result.append(',');
            result.append(values[index]);
        }
        return result.append(']').toString();
    }

    private static String number(double value) {
        return Double.isFinite(value) ? Double.toString(value) : "0.0";
    }

    private static String number(float value) {
        return Float.isFinite(value) ? Float.toString(value) : "0.0";
    }

    private static String safe(String value) {
        return value == null || value.isBlank() ? "unknown" : value;
    }

    private static String quote(String value) {
        StringBuilder result = new StringBuilder(value.length() + 2)
            .append('"');
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            switch (character) {
                case '\\' -> result.append("\\\\");
                case '"' -> result.append("\\\"");
                case '\n' -> result.append("\\n");
                case '\r' -> result.append("\\r");
                case '\t' -> result.append("\\t");
                default -> result.append(character);
            }
        }
        return result.append('"').toString();
    }

    private void disable(String reason) {
        disabled = true;
        pending.clear();
        LOGGER.log(
            System.Logger.Level.WARNING,
            "policy trace disabled: " + reason
        );
    }

    private record Pending(
        int policySlot,
        long worldTick,
        float deltaTime,
        float[] observation,
        boolean[] actionMask,
        int[] inferredFactors,
        ActionDecoder.Decoded action,
        boolean fresh,
        boolean constrained,
        String route
    ) {
        private Pending {
            observation = observation.clone();
            actionMask = actionMask.clone();
            inferredFactors = inferredFactors.clone();
        }
    }
}
