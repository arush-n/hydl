package com.hytalerlbridge.nativebackend.session.info;

import java.util.Map;
import com.hytalerlbridge.environment.EnvironmentOptions;
import com.hytalerlbridge.nativebackend.policy.combat.NativeSyntheticCombatClient;

/**
 * Synthetic combat client counters and the agent armor override.
 *
 * <p>One section of the info map built each step by
 * {@code NativeEnvironmentSession.buildInfo}. See the package README for
 * why these sections are split out and what the ordering contract is.
 */
public final class SyntheticClientInfo {

    private SyntheticClientInfo() {}

    public static void contribute(
        Map<String, Object> info,
        NativeSyntheticCombatClient syntheticCombatClient,
        EnvironmentOptions options
    ) {
        info.put(
            "native_synthetic_combat_client_bind_count",
            syntheticCombatClient.bindCount()
        );
        info.put(
            "native_synthetic_combat_client_fork_bind_count",
            syntheticCombatClient.forkBindCount()
        );
        info.put(
            "native_synthetic_combat_client_sync_count",
            syntheticCombatClient.syncCount()
        );
        info.put(
            "native_synthetic_combat_client_npc_controller_force_count",
            syntheticCombatClient.npcControllerForceCount()
        );
        info.put(
            "native_synthetic_combat_client_queued_force_count",
            syntheticCombatClient.queuedForceCount()
        );
        info.put(
            "native_synthetic_combat_client_active_count",
            syntheticCombatClient.activeCount()
        );
        info.put(
            "native_synthetic_combat_client_selection_attempt_count",
            syntheticCombatClient.selectionAttemptCount()
        );
        info.put(
            "native_synthetic_combat_client_selection_hit_count",
            syntheticCombatClient.selectionHitCount()
        );
        info.put(
            "native_synthetic_combat_client_selection_empty_count",
            syntheticCombatClient.selectionEmptyCount()
        );
        info.put(
            "native_synthetic_combat_client_selection_unavailable_count",
            syntheticCombatClient.selectionUnavailableCount()
        );
        info.put(
            "native_synthetic_combat_client_selection_"
                + "premature_server_terminal_count",
            syntheticCombatClient.selectionPrematureServerTerminalCount()
        );
        info.put(
            "native_synthetic_combat_client_selection_max_run_time_seconds",
            syntheticCombatClient.maximumSelectionRunTimeSeconds()
        );
        info.put(
            "native_synthetic_combat_client_selection_max_elapsed_seconds",
            syntheticCombatClient.maximumSelectionElapsedSeconds()
        );
        info.put(
            "native_synthetic_combat_client_selection_unavailable_reason",
            syntheticCombatClient.lastSelectionUnavailableReason()
        );
        info.put(
            "native_synthetic_combat_client_selection_client_state",
            syntheticCombatClient.lastSelectionClientState()
        );
        info.put(
            "native_synthetic_combat_client_selection_server_state",
            syntheticCombatClient.lastSelectionServerState()
        );
        info.put(
            "native_synthetic_combat_client_selection_attempt_"
                + "chain_time_shift_seconds",
            syntheticCombatClient.lastSelectionAttemptChainTimeShiftSeconds()
        );
        info.put(
            "native_synthetic_combat_client_selection_attempt_"
                + "operation_counter",
            syntheticCombatClient.lastSelectionAttemptOperationCounter()
        );
        info.put(
            "native_synthetic_combat_client_selection_attempt_"
                + "operation_index",
            syntheticCombatClient.lastSelectionAttemptOperationIndex()
        );
        info.put(
            "native_synthetic_combat_client_selection_attempt_root_id",
            syntheticCombatClient.lastSelectionAttemptRootId()
        );
        info.put(
            "native_synthetic_combat_client_selection_attempt_"
                + "entry_server_state",
            syntheticCombatClient.lastSelectionAttemptEntryServerState()
        );
        info.put(
            "native_synthetic_combat_client_selection_attempt_"
                + "entry_server_progress_seconds",
            syntheticCombatClient
                .lastSelectionAttemptEntryServerProgressSeconds()
        );
        info.put(
            "native_synthetic_combat_client_selection_attempt_"
                + "published_client_state",
            syntheticCombatClient.lastSelectionAttemptPublishedClientState()
        );
        info.put(
            "native_synthetic_combat_client_selection_attempt_"
                + "published_client_progress_seconds",
            syntheticCombatClient
                .lastSelectionAttemptPublishedClientProgressSeconds()
        );
        info.put(
            "native_synthetic_combat_client_selection_terminal_"
                + "operation_counter",
            syntheticCombatClient.lastSelectionTerminalOperationCounter()
        );
        info.put(
            "native_synthetic_combat_client_selection_terminal_"
                + "operation_index",
            syntheticCombatClient.lastSelectionTerminalOperationIndex()
        );
        info.put(
            "native_synthetic_combat_client_selection_terminal_root_id",
            syntheticCombatClient.lastSelectionTerminalRootId()
        );
        info.put(
            "native_synthetic_combat_client_selection_terminal_"
                + "chain_client_state",
            syntheticCombatClient.lastSelectionTerminalChainClientState()
        );
        info.put(
            "native_synthetic_combat_client_selection_terminal_"
                + "entry_server_state",
            syntheticCombatClient.lastSelectionTerminalEntryServerState()
        );
        info.put(
            "native_synthetic_combat_client_selection_terminal_"
                + "entry_server_progress_seconds",
            syntheticCombatClient
                .lastSelectionTerminalEntryServerProgressSeconds()
        );
        info.put(
            "native_synthetic_combat_client_selection_terminal_"
                + "entry_client_state",
            syntheticCombatClient.lastSelectionTerminalEntryClientState()
        );
        info.put(
            "native_synthetic_combat_client_selection_terminal_"
                + "entry_client_progress_seconds",
            syntheticCombatClient
                .lastSelectionTerminalEntryClientProgressSeconds()
        );
        info.put(
            "native_synthetic_combat_client_selection_terminal_"
                + "tracked_chains",
            syntheticCombatClient.lastSelectionTerminalTrackedChains()
        );
        info.put(
            "native_agent_armor_override",
            options.hasNativeAgentArmorOverride()
        );
        info.put(
            "native_agent_armor_item_ids",
            options.hasNativeAgentArmorOverride()
                ? String.join(",", options.nativeAgentArmorItemIds())
                : ""
        );
    }
}
