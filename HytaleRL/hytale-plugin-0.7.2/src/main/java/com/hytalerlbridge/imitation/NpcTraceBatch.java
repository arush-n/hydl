package com.hytalerlbridge.imitation;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/** Bounded, drainable wire batch for one UUID-pinned native NPC trace. */
public record NpcTraceBatch(
    String serverVersion,
    String world,
    String worldgenProvider,
    String worldgenVersion,
    long seed,
    Map<String, String> environmentParameters,
    UUID traceUuid,
    UUID npcUuid,
    String role,
    String status,
    int capacity,
    long totalCaptured,
    List<NpcTraceFrame> frames
) {
    public static final String SCHEMA = "hytalerl_native_npc_transition_trace_v13";
    public static final int VERSION = 13;
    public static final String ACTION_CONTRACT_SCHEMA =
        "hytalerl_native_npc_action_contract_v2";
    public static final int ACTION_CONTRACT_VERSION = 2;
    public static final String ACTION_CONTRACT_SHA256 =
        sha256(actionContractJson());
    public static final String EVENT_CONTRACT_SCHEMA =
        "hytalerl_native_npc_event_contract_v1";
    public static final int EVENT_CONTRACT_VERSION = 1;
    public static final String EVENT_CONTRACT_SHA256 =
        sha256(eventContractJson());
    public static final String LIFECYCLE_CONTRACT_SCHEMA =
        "hytalerl_native_npc_lifecycle_contract_v1";
    public static final int LIFECYCLE_CONTRACT_VERSION = 1;
    public static final String LIFECYCLE_CONTRACT_SHA256 =
        sha256(lifecycleContractJson());
    public static final int DEFAULT_CAPACITY = 4096;
    public static final int MAX_CAPACITY = 65_536;
    // Full semantic rows include two 9x9x9 geometry frames. Keep one RPC well
    // below the bridge's 16 MiB frame ceiling; long traces are poll/concat.
    public static final int MAX_DRAIN = 32;
    public static final Set<String> STATUSES = Set.of(
        "recording",
        "stopped",
        "target_missing",
        "role_changed",
        "overflow"
    );

    public NpcTraceBatch {
        if (
            serverVersion == null
                || world == null
                || worldgenProvider == null
                || worldgenVersion == null
                || traceUuid == null
                || npcUuid == null
                || role == null
                || role.isBlank()
        ) {
            throw new IllegalArgumentException("trace identity must be complete");
        }
        if (!STATUSES.contains(status)) {
            throw new IllegalArgumentException("unknown trace status: " + status);
        }
        if (capacity < 1 || capacity > MAX_CAPACITY) {
            throw new IllegalArgumentException("trace capacity is out of range");
        }
        if (totalCaptured < 0) {
            throw new IllegalArgumentException("totalCaptured must be nonnegative");
        }
        frames = frames == null ? List.of() : List.copyOf(frames);
        environmentParameters = environmentParameters == null
            ? Map.of()
            : Map.copyOf(environmentParameters);
        if (environmentParameters.entrySet().stream().anyMatch(
            entry -> entry.getKey() == null
                || entry.getKey().isBlank()
                || entry.getValue() == null
        )) {
            throw new IllegalArgumentException("trace environment parameters are invalid");
        }
        if (frames.size() > MAX_DRAIN || totalCaptured < frames.size()) {
            throw new IllegalArgumentException("invalid drained trace frame count");
        }
    }

    private static String actionContractJson() {
        return "{\"attack\":{\"activation\":\"selected_attack_action_active_and_pre_behavior_attack_action_inactive\","
            + "\"execution\":\"combat_support_execution_at_decision_and_next_boundaries\","
            + "\"execution_cause\":{\"availability\":\"false_on_candidate_overflow_or_non_unique_join\","
            + "\"candidate_join\":\"same_path_next_candidate_interaction_id_equals_chain_initial_root_id\","
            + "\"interaction_start\":\"decision_boundary_combat_support_first_run\",\"layout\":"
            + quote(NpcAttackExecutionCause.LAYOUT) + "},\"layout\":"
            + quote(NpcAttackActionSnapshot.LAYOUT)
            + "},\"boundary\":{\"decision\":\"after_behavior_and_avoidance_before_bridge_and_steering\","
            + "\"next_observation\":\"after_native_tick\",\"observation\":\"after_pre_behavior_support_before_behavior\"},"
            + "\"control\":{\"layout\":" + quote(NpcTraceFrame.CONTROL_LAYOUT)
            + ",\"mask_layout\":" + quote(NpcTraceFrame.CONTROL_MASK_LAYOUT)
            + ",\"native_control\":\"true_if_no_bridge_override_was_marked_for_transition\"},"
            + "\"identity\":{\"labels\":\"raw_java_strings\",\"paths\":\"behavior_tree_paths\","
            + "\"symbols\":\"capture_specific_table_sha256_stamped_separately\"},"
            + "\"interaction\":{\"layout\":" + quote(NpcInteractionSnapshot.LAYOUT) + "},"
            + "\"internal\":{\"layout\":" + quote(NpcInternalSnapshot.LAYOUT) + "},"
            + "\"schema\":" + quote(ACTION_CONTRACT_SCHEMA)
            + ",\"version\":" + ACTION_CONTRACT_VERSION + "}";
    }

    private static String eventContractJson() {
        return "{\"amounts\":{\"final\":\"post_filter_server_amount_rounded_when_applied\"," 
            + "\"initial\":\"pre_filter_server_damage_amount\"},"
            + "\"boundary\":\"inspect_damage_group_after_apply_damage\","
            + "\"capacity\":" + NpcDamageEventSnapshot.CAPACITY + ","
            + "\"layout\":" + quote(NpcDamageEventSnapshot.LAYOUT) + ","
            + "\"overflow\":\"count_gt_emitted_events\","
            + "\"perspective\":\"actor_source_and_actor_target_are_trace_relative\","
            + "\"schema\":" + quote(EVENT_CONTRACT_SCHEMA) + ","
            + "\"version\":" + EVENT_CONTRACT_VERSION + "}";
    }

    private static String lifecycleContractJson() {
        return "{\"boundary\":\"inter_row_context_and_pre_behavior_to_after_native_tick\"," 
            + "\"capacity\":" + NpcLifecycleEventSnapshot.CAPACITY + ","
            + "\"event_order\":\"inter_row_deltas_then_server_inventory_arrival_then_transition_deltas\"," 
            + "\"layout\":" + quote(NpcLifecycleEventSnapshot.LAYOUT) + ","
            + "\"missingness\":\"available_union_partial_if_any_observed_subinterval_unavailable_or_overflow\"," 
            + "\"origins\":\"server_inventory_event_or_server_boundary_state_delta_or_server_inter_row_state_delta_or_server_death_component\"," 
            + "\"schema\":" + quote(LIFECYCLE_CONTRACT_SCHEMA) + ","
            + "\"source_layout\":"
            + quote(NpcLifecycleEventSnapshot.SOURCE_LAYOUT) + ","
            + "\"version\":" + LIFECYCLE_CONTRACT_VERSION + "}";
    }

    private static String quote(String value) {
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    private static String sha256(String value) {
        try {
            return HexFormat.of().withUpperCase().formatHex(
                MessageDigest.getInstance("SHA-256").digest(
                    value.getBytes(StandardCharsets.UTF_8)
                )
            );
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable", error);
        }
    }
}
