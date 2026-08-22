package com.hytalerlbridge.policy;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Publishes the hook's counters as a JSON file for the console to read.
 *
 * <p>A file rather than a socket, deliberately. The console is a read-only
 * viewer -- it already refuses to open a `NativeBridgeSession` because that
 * would take the native evidence lease from whichever lane holds it -- so
 * giving it another live port to poll would be the wrong shape. A file also
 * survives the server going down, which is exactly when you most want to see
 * the last known state, and it needs no dependency inside the JVM.
 *
 * <p>Writes are atomic: a temporary file then an {@code ATOMIC_MOVE}. Without
 * that the console will eventually read a half-written object and show a parse
 * error that looks like a fault in the agent.
 *
 * <p>The JSON is hand-rolled because the mod has no dependencies and the shape
 * is one flat object of numbers and short strings. Values are escaped on the
 * way out even so -- a role name is externally supplied.
 */
final class PolicyMetrics {

    static final String FILE_NAME = "metrics.json";
    static final int SCHEMA_VERSION = 3;
    static final double MINIMUM_HORIZONTAL_PATH = 0.05;

    private PolicyMetrics() {
    }

    /** Build the payload; separated from writing so it can be asserted on. */
    static Map<String, Object> snapshot(
        String role,
        PolicyRuntime runtime,
        PolicyAttachmentSystem attachment,
        PolicyControlSystem control,
        SteeringActionSink sink,
        long[] worldVerbCounters,
        long worldTick,
        boolean worldTicking,
        double worldTps,
        double policyTps,
        int captureNonZero,
        int captureLegal
    ) {
        long[] counters = control.counters();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("schema", "hytale_policy_agent_metrics");
        out.put("schema_version", SCHEMA_VERSION);
        out.put("timestamp_millis", System.currentTimeMillis());

        out.put("role", role == null ? "*" : role);
        out.put("observation_size", runtime.observationSize());
        out.put("action_size", runtime.actionSize());
        out.put("encoder_size", runtime.encoderSize());
        out.put("recurrent_size", runtime.recurrentSize());

        out.put("claimed", attachment.claimed());
        out.put("policy_ticks", counters[0]);
        out.put("skipped_no_observation", counters[1]);
        out.put("rejected_illegal", counters[2]);
        out.put("dropped_world_verb", counters[3]);
        out.put("world_verb_requests", control.worldVerbRequests());
        out.put("locomotion_requests", control.locomotionRequests());
        appendWorldVerbCounters(out, worldVerbCounters);
        // Held ticks are ones where the network ran but its action was
        // discarded in favour of the previous decision. `policy_ticks` still
        // counts them, so held/ticks should approach (period-1)/period.
        out.put("decision_period", control.decisionPeriod());
        out.put("held_decisions", counters[4]);
        out.put("skipped_interactions", sink == null ? 0L : sink.skippedInteractions());

        out.put("world_tick", worldTick);
        out.put("world_ticking", worldTicking);
        out.put("world_tps", round(worldTps));
        out.put("policy_ticks_per_second", round(policyTps));

        // Frozen captures have static counts. Live perception reports -1 here:
        // its values change per actor/tick and must be diagnosed from sampled
        // rows rather than one setup-time number.
        out.put("capture_non_zero_columns", captureNonZero);
        out.put("capture_legal_actions", captureLegal);

        // Motion evidence for one pinned NPC. `moved` is the claim the whole
        // action-application path rests on: the sink is called every tick and
        // throws nothing either way, so only displacement distinguishes
        // "steering applied" from "steering silently discarded".
        PolicyAgentMarker witness = control.witness();
        if (witness != null && witness.havePosition()) {
            double[] delta = witness.displacement();
            double[] here = witness.position();
            out.put("motion_samples", witness.positionSamples());
            out.put("motion_path_length", round(witness.pathLength()));
            out.put("motion_horizontal_path", round(witness.horizontalPathLength()));
            out.put("motion_horizontal_displacement",
                round(witness.horizontalDisplacement()));
            out.put("motion_dx", round(delta[0]));
            out.put("motion_dy", round(delta[1]));
            out.put("motion_dz", round(delta[2]));
            out.put("motion_x", round(here[0]));
            out.put("motion_y", round(here[1]));
            out.put("motion_z", round(here[2]));
            // Horizontal only, and paired with a decoded locomotion request:
            // falling or external knockback alone is not policy movement.
            out.put("moved", meaningfulMotion(
                witness.horizontalPathLength(), control.locomotionRequests()));
        } else {
            out.put("motion_samples", 0);
            out.put("moved", false);
        }

        ActionDecoder.Decoded last = control.lastDecision();
        if (last != null) {
            out.put("last_skill", last.skillId());
            out.put("last_world_move", last.worldMoveDirection());
            out.put("last_yaw_delta_degrees", round(last.yawDeltaDegrees()));
            out.put("last_pitch_delta_degrees", round(last.pitchDeltaDegrees()));
            out.put("last_jump", last.jumpHeld());
            out.put("last_attack", last.attack());
            out.put("last_guard", last.guardHeld());
            // Published because skipped_interactions counts attack, guard,
            // ability, dodge and use together. Without knowing which one the
            // policy keeps asking for, a skip count equal to the tick count is
            // an unexplainable number rather than a finding.
            out.put("last_dodge", last.dodgeDirection());
            out.put("last_use", last.useRequested());
            out.put("last_ability_slot", last.abilitySlot());
            out.put("last_action_legal", last.actionLegal());
            out.put("last_value", round(control.lastValue()));
        }
        return out;
    }

    static boolean meaningfulMotion(double horizontalPath, long requests) {
        return requests > 0 && horizontalPath > MINIMUM_HORIZONTAL_PATH;
    }

    /** Publish the executor result counters, not only decoder-side requests. */
    static void appendWorldVerbCounters(
        Map<String, Object> out,
        long[] counters
    ) {
        if (out == null) {
            throw new IllegalArgumentException("metrics map is required");
        }
        boolean available = counters != null;
        if (available && counters.length != 9) {
            throw new IllegalArgumentException(
                "World-verb counters must contain exactly nine values");
        }
        long[] value = available ? counters : new long[9];
        out.put("world_verb_executor_available", available);
        out.put("world_verb_requested_edges", value[0]);
        out.put("world_verb_accepted", value[1]);
        out.put("world_verb_finished", value[2]);
        out.put("world_verb_failed", value[3]);
        out.put("world_verb_binding_rejected", value[4]);
        out.put("world_verb_busy_rejected", value[5]);
        out.put("world_verb_unsupported", value[6]);
        out.put("world_verb_held_edges", value[7]);
        out.put("world_verb_context_failures", value[8]);
    }

    static void write(Path directory, Map<String, Object> payload) throws IOException {
        Path target = directory.resolve(FILE_NAME);
        Path temporary = directory.resolve(FILE_NAME + ".tmp");
        Files.writeString(temporary, toJson(payload), StandardCharsets.UTF_8);
        try {
            Files.move(temporary, target,
                StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (java.nio.file.AtomicMoveNotSupportedException fallback) {
            Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static String toJson(Map<String, Object> payload) {
        StringBuilder json = new StringBuilder(1024).append("{\n");
        int remaining = payload.size();
        for (Map.Entry<String, Object> entry : payload.entrySet()) {
            json.append("  \"").append(escape(entry.getKey())).append("\": ");
            Object value = entry.getValue();
            if (value instanceof Number || value instanceof Boolean) {
                json.append(value);
            } else {
                json.append('"').append(escape(String.valueOf(value))).append('"');
            }
            json.append(--remaining > 0 ? ",\n" : "\n");
        }
        return json.append("}\n").toString();
    }

    private static double round(double value) {
        if (!Double.isFinite(value)) {
            return 0.0;
        }
        return Math.round(value * 10000.0) / 10000.0;
    }

    private static String escape(String value) {
        StringBuilder out = new StringBuilder(value.length() + 8);
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        return out.toString();
    }
}
