package com.hytalerlbridge.nativebackend;

import com.hytalerlbridge.action.AgentAction;
import java.util.Map;
import org.joml.Vector3i;

/**
 * Fixture-only Player/interaction-chain evidence.
 *
 * <p>This proves native reach, permission, held-item, timing, and mutation
 * execution for a controlled actor. It deliberately does not certify an
 * authenticated network Player or open the public action mask.</p>
 */
final class SyntheticWorldInteractionEvidence {
    static final String SCHEMA =
        "hytalerl_synthetic_player_world_interaction_v6";
    static final int VERSION = 6;
    static final String CONTRACT_SHA256 =
        "F41C13862D6643030F3BA850E1C9DC8F33CF49995CFD806A881CE6F462131E6C";

    private final boolean fixtureAvailable;
    private final String fixtureBlockAssetId;
    private final int fixtureBlockRuntimeId;
    private final String fixturePlaceItemAssetId;
    private final String fixtureBreakToolItemAssetId;
    private final int fixturePlaceSourceQuantity;
    private final int fixtureBreakSourceQuantity;
    private final Vector3i fixturePlaceTarget;
    private final Vector3i fixtureBreakTarget;
    private final Row placeBlock;
    private final Row breakBlock;

    SyntheticWorldInteractionEvidence(
        boolean fixtureAvailable,
        AgentAction action,
        String fixtureBlockAssetId,
        int fixtureBlockRuntimeId,
        String fixturePlaceItemAssetId,
        String fixtureBreakToolItemAssetId,
        int fixturePlaceSourceQuantity,
        int fixtureBreakSourceQuantity,
        Vector3i fixturePlaceTarget,
        Vector3i fixtureBreakTarget
    ) {
        if (
            fixtureAvailable
                && (
                    fixtureBlockAssetId == null
                        || fixtureBlockAssetId.isEmpty()
                        || fixtureBlockRuntimeId <= 0
                        || fixturePlaceItemAssetId == null
                        || fixturePlaceItemAssetId.isEmpty()
                        || fixtureBreakToolItemAssetId == null
                        || fixtureBreakToolItemAssetId.isEmpty()
                        || fixturePlaceSourceQuantity <= 0
                        || fixtureBreakSourceQuantity <= 0
                        || fixturePlaceTarget == null
                        || fixtureBreakTarget == null
                        || fixturePlaceTarget.equals(fixtureBreakTarget)
                )
        ) {
            throw new IllegalArgumentException(
                "Available synthetic World evidence requires bound assets"
            );
        }
        this.fixtureAvailable = fixtureAvailable;
        this.fixtureBlockAssetId = fixtureBlockAssetId == null
            ? ""
            : fixtureBlockAssetId;
        this.fixtureBlockRuntimeId = fixtureBlockRuntimeId;
        this.fixturePlaceItemAssetId = fixturePlaceItemAssetId == null
            ? ""
            : fixturePlaceItemAssetId;
        this.fixtureBreakToolItemAssetId =
            fixtureBreakToolItemAssetId == null
                ? ""
                : fixtureBreakToolItemAssetId;
        this.fixturePlaceSourceQuantity = fixturePlaceSourceQuantity;
        this.fixtureBreakSourceQuantity = fixtureBreakSourceQuantity;
        this.fixturePlaceTarget = fixturePlaceTarget == null
            ? new Vector3i()
            : new Vector3i(fixturePlaceTarget);
        this.fixtureBreakTarget = fixtureBreakTarget == null
            ? new Vector3i()
            : new Vector3i(fixtureBreakTarget);
        this.placeBlock = new Row(action.hasPlacement());
        this.breakBlock = new Row(action.hasBreak());
    }

    Row placeBlock() {
        return placeBlock;
    }

    Row breakBlock() {
        return breakBlock;
    }

    void putInto(Map<String, Object> info) {
        info.put("synthetic_world_interaction_schema", SCHEMA);
        info.put("synthetic_world_interaction_version", VERSION);
        info.put(
            "synthetic_world_interaction_contract_sha256",
            CONTRACT_SHA256
        );
        info.put(
            "synthetic_world_interaction_fixture_available",
            fixtureAvailable
        );
        info.put(
            "synthetic_world_interaction_fixture_block_asset_id",
            fixtureBlockAssetId
        );
        info.put(
            "synthetic_world_interaction_fixture_block_runtime_id",
            fixtureBlockRuntimeId
        );
        info.put(
            "synthetic_world_interaction_fixture_place_item_asset_id",
            fixturePlaceItemAssetId
        );
        info.put(
            "synthetic_world_interaction_fixture_break_tool_item_asset_id",
            fixtureBreakToolItemAssetId
        );
        info.put(
            "synthetic_world_interaction_fixture_place_source_quantity",
            fixturePlaceSourceQuantity
        );
        info.put(
            "synthetic_world_interaction_fixture_break_source_quantity",
            fixtureBreakSourceQuantity
        );
        putTarget(
            info,
            "synthetic_world_interaction_fixture_place_target_",
            fixturePlaceTarget
        );
        putTarget(
            info,
            "synthetic_world_interaction_fixture_break_target_",
            fixtureBreakTarget
        );
        info.put(
            "synthetic_world_interaction_public_player_certified",
            false
        );
        info.put(
            "synthetic_world_interaction_opens_action_mask",
            false
        );
        placeBlock.putInto(info, "synthetic_place_block_");
        breakBlock.putInto(info, "synthetic_break_block_");
    }

    private static void putTarget(
        Map<String, Object> info,
        String prefix,
        Vector3i target
    ) {
        info.put(prefix + "x", target.x);
        info.put(prefix + "y", target.y);
        info.put(prefix + "z", target.z);
    }

    static final class Row {
        private final boolean requested;
        private boolean accepted;
        private boolean started;
        private boolean finished;
        private boolean failed;
        private boolean mutationApplied;
        private String rejection = "";
        private String interactionAssetId = "";
        private String heldItemAssetId = "";
        private String serverState = "unrequested";
        private int targetX;
        private int targetY;
        private int targetZ;
        private int blockBefore;
        private int blockAfter;
        private double healthBefore;
        private double healthAfter;
        private long requestedTick = -1L;
        private long startedTick = -1L;
        private long finishedTick = -1L;

        private Row(boolean requested) {
            this.requested = requested;
            if (requested) serverState = "requested";
        }

        void reject(String reason) {
            if (!requested || accepted) return;
            rejection = reason;
            failed = true;
            serverState = "rejected";
        }

        void accept(
            String assetId,
            String heldItem,
            Vector3i target,
            int beforeBlock,
            double beforeHealth,
            long worldTick
        ) {
            if (!requested) return;
            accepted = true;
            interactionAssetId = assetId;
            heldItemAssetId = heldItem;
            targetX = target.x;
            targetY = target.y;
            targetZ = target.z;
            blockBefore = beforeBlock;
            blockAfter = beforeBlock;
            healthBefore = beforeHealth;
            healthAfter = beforeHealth;
            requestedTick = worldTick;
            serverState = "queued";
        }

        void observe(
            String state,
            boolean registered,
            int afterBlock,
            double afterHealth,
            boolean expectedMutation,
            long worldTick
        ) {
            if (!accepted) return;
            blockAfter = afterBlock;
            healthAfter = afterHealth;
            mutationApplied |= expectedMutation;
            serverState = state;
            if (!started && (registered || !"NotFinished".equals(state))) {
                started = true;
                startedTick = worldTick;
            }
            if (
                started
                    && (!registered || "Finished".equals(state)
                        || "Failed".equals(state))
            ) {
                finished = true;
                finishedTick = worldTick;
            }
            failed |= "Failed".equals(state);
        }

        void putInto(Map<String, Object> info, String prefix) {
            info.put(prefix + "requested", requested);
            info.put(prefix + "accepted", accepted);
            info.put(prefix + "started", started);
            info.put(prefix + "finished", finished);
            info.put(prefix + "failed", failed);
            info.put(prefix + "mutation_applied", mutationApplied);
            info.put(prefix + "reject_reason", rejection);
            info.put(prefix + "interaction_asset_id", interactionAssetId);
            info.put(prefix + "held_item_asset_id", heldItemAssetId);
            info.put(prefix + "server_state", serverState);
            info.put(prefix + "target_x", targetX);
            info.put(prefix + "target_y", targetY);
            info.put(prefix + "target_z", targetZ);
            info.put(prefix + "runtime_block_id_before", blockBefore);
            info.put(prefix + "runtime_block_id_after", blockAfter);
            info.put(prefix + "block_health_before", healthBefore);
            info.put(prefix + "block_health_after", healthAfter);
            info.put(prefix + "requested_world_tick", requestedTick);
            info.put(prefix + "started_world_tick", startedTick);
            info.put(prefix + "finished_world_tick", finishedTick);
        }
    }
}
