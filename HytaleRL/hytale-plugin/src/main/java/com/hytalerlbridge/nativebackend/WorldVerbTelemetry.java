package com.hytalerlbridge.nativebackend;

import com.hytalerlbridge.action.AgentAction;
import com.hytalerlbridge.action.NativeWorldVerbRequest;
import java.util.Map;

/** Fail-closed per-step lifecycle and acknowledgement evidence for World verbs. */
final class WorldVerbTelemetry {
    static final String SCHEMA = "hytalerl_native_world_verb_lifecycle_v1";
    static final int VERSION = 1;
    static final String CONTRACT_SHA256 =
        "99E83A86312F331D3C70803DB178D91A0D16F630830D24CA4D781CA4894601E5";

    private WorldVerbTelemetry() {}

    /** Backward-compatible test helper for legacy rejection rows. */
    static void putUnsupported(Map<String, Object> info, AgentAction action) {
        putUnsupported(info, action, "test-world-epoch", -1L);
    }

    static void putUnsupported(
        Map<String, Object> info,
        AgentAction action,
        String worldEpoch,
        long worldTick
    ) {
        put(
            info,
            action,
            worldEpoch,
            Execution.transportOnly(
                action.nativeWorldVerbRequest(),
                worldEpoch,
                worldTick
            )
        );
    }

    static void put(
        Map<String, Object> info,
        AgentAction action,
        String worldEpoch,
        Execution execution
    ) {
        info.put("native_world_verb_lifecycle_schema", SCHEMA);
        info.put("native_world_verb_lifecycle_version", VERSION);
        info.put(
            "native_world_verb_lifecycle_contract_sha256",
            CONTRACT_SHA256
        );
        put(
            info,
            "place_block",
            action.hasPlacement(),
            "native_place_block_requires_player_client_sync"
        );
        put(
            info,
            "break_block",
            action.hasBreak(),
            "native_break_block_interaction_requires_player"
        );
        put(
            info,
            "craft_recipe",
            action.hasCraft(),
            "native_craft_recipe_requires_player"
        );
        putTypedTransport(
            info,
            execution,
            worldEpoch,
            execution.requestedWorldTick()
        );
    }

    private static void putTypedTransport(
        Map<String, Object> info,
        Execution execution,
        String currentEpoch,
        long worldTick
    ) {
        NativeWorldVerbRequest request = execution.request();
        boolean present = request.present();

        info.put(
            "native_world_verb_transport_schema",
            NativeWorldVerbRequest.SCHEMA
        );
        info.put(
            "native_world_verb_transport_version",
            NativeWorldVerbRequest.VERSION
        );
        info.put(
            "native_world_verb_transport_contract_sha256",
            NativeWorldVerbRequest.CONTRACT_SHA256
        );
        info.put("native_world_verb_epoch", currentEpoch);
        info.put("native_world_verb_request_present", present);
        info.put(
            "native_world_verb_request_request_id",
            request.requestId()
        );
        info.put("native_world_verb_request_verb", request.verb());
        info.put(
            "native_world_verb_request_interaction_type",
            request.interactionType()
        );
        info.put(
            "native_world_verb_request_target_kind",
            request.targetKind()
        );
        info.put(
            "native_world_verb_request_interaction_id",
            request.interactionId()
        );
        info.put(
            "native_world_verb_request_recipe_id",
            request.recipeId()
        );
        info.put("native_world_verb_request_item_id", request.itemId());
        info.put("native_world_verb_request_block_id", request.blockId());
        info.put("native_world_verb_request_target_x", request.targetX());
        info.put("native_world_verb_request_target_y", request.targetY());
        info.put("native_world_verb_request_target_z", request.targetZ());
        info.put(
            "native_world_verb_request_block_face",
            request.blockFace()
        );
        info.put(
            "native_world_verb_request_rotation_yaw",
            request.rotationYaw()
        );
        info.put(
            "native_world_verb_request_rotation_pitch",
            request.rotationPitch()
        );
        info.put(
            "native_world_verb_request_rotation_roll",
            request.rotationRoll()
        );
        info.put(
            "native_world_verb_request_source_container",
            request.sourceContainer()
        );
        info.put(
            "native_world_verb_request_source_slot",
            request.sourceSlot()
        );
        info.put(
            "native_world_verb_request_expected_source_quantity",
            request.expectedSourceQuantity()
        );
        info.put(
            "native_world_verb_request_destination_container",
            request.destinationContainer()
        );
        info.put(
            "native_world_verb_request_expected_block_id",
            request.expectedBlockId()
        );
        info.put(
            "native_world_verb_request_world_epoch",
            request.worldEpoch()
        );
        info.put(
            "native_world_verb_request_quantity",
            request.quantity()
        );
        info.put(
            "native_world_verb_request_placement_variant",
            request.placementVariant()
        );
        info.put(
            "native_world_verb_request_crafting_context",
            request.craftingContext()
        );
        info.put("native_world_verb_request_bench_x", request.benchX());
        info.put("native_world_verb_request_bench_y", request.benchY());
        info.put("native_world_verb_request_bench_z", request.benchZ());
        info.put(
            "native_world_verb_request_expected_bench_block_id",
            request.expectedBenchBlockId()
        );
        info.put(
            "native_world_verb_request_expected_bench_id",
            request.expectedBenchId()
        );
        info.put(
            "native_world_verb_request_expected_bench_type",
            request.expectedBenchType()
        );
        info.put(
            "native_world_verb_request_expected_bench_tier",
            request.expectedBenchTier()
        );
        info.put(
            "native_world_verb_request_expected_candidate_generation_sha256",
            request.expectedCandidateGenerationSha256()
        );
        info.put(
            "native_world_verb_request_expected_selected_semantic_sha256",
            request.expectedSelectedSemanticSha256()
        );
        info.put(
            "native_world_verb_request_accepted",
            execution.accepted()
        );
        info.put(
            "native_world_verb_request_started",
            execution.started()
        );
        info.put(
            "native_world_verb_request_active",
            execution.active()
        );
        info.put(
            "native_world_verb_request_finished",
            execution.finished()
        );
        info.put(
            "native_world_verb_request_failed",
            execution.failed()
        );
        info.put(
            "native_world_verb_request_reject_reason",
            execution.rejectReason()
        );
        info.put(
            "native_world_verb_request_execution_scope",
            execution.executionScope()
        );
        info.put(
            "native_world_verb_request_public_player_certified",
            execution.publicPlayerCertified()
        );
        info.put(
            "native_world_verb_request_headless_server_actor_certified",
            execution.headlessServerActorCertified()
        );
        info.put(
            "native_world_verb_request_opens_action_mask",
            execution.opensActionMask()
        );
        info.put(
            "native_world_verb_request_acknowledgement_complete",
            execution.acknowledgementComplete()
        );
        info.put(
            "native_world_verb_request_mutation_applied",
            execution.mutationApplied()
        );
        info.put(
            "native_world_verb_request_inventory_applied",
            execution.inventoryApplied()
        );
        info.put(
            "native_world_verb_request_semantic_block_id_before",
            execution.semanticBlockIdBefore()
        );
        info.put(
            "native_world_verb_request_semantic_block_id_after",
            execution.semanticBlockIdAfter()
        );
        info.put(
            "native_world_verb_request_runtime_block_id_before",
            execution.runtimeBlockIdBefore()
        );
        info.put(
            "native_world_verb_request_runtime_block_id_after",
            execution.runtimeBlockIdAfter()
        );
        info.put(
            "native_world_verb_request_source_quantity_before",
            execution.sourceQuantityBefore()
        );
        info.put(
            "native_world_verb_request_source_quantity_after",
            execution.sourceQuantityAfter()
        );
        info.put(
            "native_world_verb_request_destination_quantity_before",
            execution.destinationQuantityBefore()
        );
        info.put(
            "native_world_verb_request_destination_quantity_after",
            execution.destinationQuantityAfter()
        );
        info.put(
            "native_world_verb_request_geometry_delta_sha256",
            execution.geometryDeltaSha256()
        );
        info.put(
            "native_world_verb_request_inventory_delta_sha256",
            execution.inventoryDeltaSha256()
        );
        info.put(
            "native_world_verb_request_drop_outcome_sha256",
            execution.dropOutcomeSha256()
        );
        info.put(
            "native_world_verb_request_requested_world_tick",
            worldTick
        );
        info.put(
            "native_world_verb_request_started_world_tick",
            execution.startedWorldTick()
        );
        info.put(
            "native_world_verb_request_finished_world_tick",
            execution.finishedWorldTick()
        );
    }

    private static void put(
        Map<String, Object> info,
        String verb,
        boolean requested,
        String rejection
    ) {
        String prefix = "native_" + verb;
        info.put(prefix + "_requested", requested);
        info.put(prefix + "_accepted", false);
        info.put(prefix + "_reject_reason", requested ? rejection : "");
    }

    static final class Execution {
        private final NativeWorldVerbRequest request;
        private final boolean accepted;
        private final String rejectReason;
        private final String executionScope;
        private final boolean publicPlayerCertified;
        private final boolean headlessServerActorCertified;
        private final boolean opensActionMask;
        private final long requestedWorldTick;
        private final String semanticBlockIdBefore;
        private final int runtimeBlockIdBefore;
        private final int sourceQuantityBefore;
        private final int destinationQuantityBefore;
        private boolean started;
        private boolean finished;
        private boolean failed;
        private boolean acknowledgementComplete;
        private boolean mutationApplied;
        private boolean inventoryApplied;
        private String semanticBlockIdAfter = "";
        private int runtimeBlockIdAfter = -1;
        private int sourceQuantityAfter = -1;
        private int destinationQuantityAfter = -1;
        private String geometryDeltaSha256 = "";
        private String inventoryDeltaSha256 = "";
        private String dropOutcomeSha256 = "";
        private long startedWorldTick = -1L;
        private long finishedWorldTick = -1L;

        private Execution(
            NativeWorldVerbRequest request,
            boolean accepted,
            boolean failed,
            String rejectReason,
            String executionScope,
            boolean publicPlayerCertified,
            boolean headlessServerActorCertified,
            boolean opensActionMask,
            long requestedWorldTick,
            String semanticBlockIdBefore,
            int runtimeBlockIdBefore,
            int sourceQuantityBefore,
            int destinationQuantityBefore
        ) {
            this.request = request;
            this.accepted = accepted;
            this.failed = failed;
            this.rejectReason = rejectReason;
            this.executionScope = executionScope;
            this.publicPlayerCertified = publicPlayerCertified;
            this.headlessServerActorCertified =
                headlessServerActorCertified;
            this.opensActionMask = opensActionMask;
            this.requestedWorldTick = requestedWorldTick;
            this.semanticBlockIdBefore = semanticBlockIdBefore;
            this.runtimeBlockIdBefore = runtimeBlockIdBefore;
            this.sourceQuantityBefore = sourceQuantityBefore;
            this.destinationQuantityBefore = destinationQuantityBefore;
        }

        static Execution unrequested() {
            return new Execution(
                NativeWorldVerbRequest.none(),
                false,
                false,
                "",
                "unrequested",
                false,
                false,
                false,
                -1L,
                "",
                -1,
                -1,
                -1
            );
        }

        static Execution transportOnly(
            NativeWorldVerbRequest request,
            String currentEpoch,
            long worldTick
        ) {
            if (!request.present()) return unrequested();
            String reason = request.worldEpoch().equals(currentEpoch)
                ? "typed_" + request.verb() + "_execution_not_implemented"
                : "stale_world_epoch";
            return rejected(request, reason, worldTick);
        }

        static Execution rejected(
            NativeWorldVerbRequest request,
            String reason,
            long worldTick
        ) {
            if (!request.present() || reason == null || reason.isBlank()) {
                throw new IllegalArgumentException(
                    "Rejected World verb requires a request and reason"
                );
            }
            return new Execution(
                request,
                false,
                true,
                reason,
                "transport_only_rejected",
                false,
                false,
                false,
                worldTick,
                "",
                -1,
                -1,
                -1
            );
        }

        /** Actor-zero compatibility projection from the v5 actor-major path. */
        static Execution actorMajorProjection(
            NativeWorldVerbRequest request,
            NativePolicyWorldVerbFacade.Lifecycle lifecycle
        ) {
            if (request == null || !request.present() || lifecycle == null) {
                return unrequested();
            }
            Execution result = new Execution(
                request,
                lifecycle.accepted(),
                lifecycle.failed(),
                lifecycle.rejectReason(),
                "actor_major_native_policy_world_verb",
                false,
                lifecycle.accepted(),
                false,
                lifecycle.requestedTick(),
                lifecycle.blockIdBefore().isEmpty()
                    ? "unavailable"
                    : lifecycle.blockIdBefore(),
                -1,
                lifecycle.sourceQuantityBefore(),
                -1
            );
            result.started = lifecycle.started();
            result.finished = lifecycle.finished();
            result.failed = lifecycle.failed();
            result.mutationApplied = lifecycle.geometryChanged();
            result.inventoryApplied = lifecycle.inventoryChanged();
            result.semanticBlockIdAfter = lifecycle.blockIdAfter();
            result.sourceQuantityAfter = lifecycle.sourceQuantityAfter();
            result.startedWorldTick = lifecycle.startedTick();
            result.finishedWorldTick = lifecycle.finishedTick();
            // The actor-major v4 receipt deliberately does not synthesize the
            // v1 runtime IDs/delta hashes, so acknowledgementComplete stays
            // false. Consumers must use the v5 actor-major receipt for exact
            // completion evidence.
            return result;
        }

        static Execution accepted(
            NativeWorldVerbRequest request,
            String scope,
            long worldTick,
            String semanticBlockIdBefore,
            int runtimeBlockIdBefore,
            int sourceQuantityBefore,
            int destinationQuantityBefore
        ) {
            return accepted(
                request,
                scope,
                worldTick,
                semanticBlockIdBefore,
                runtimeBlockIdBefore,
                sourceQuantityBefore,
                destinationQuantityBefore,
                false
            );
        }

        static Execution accepted(
            NativeWorldVerbRequest request,
            String scope,
            long worldTick,
            String semanticBlockIdBefore,
            int runtimeBlockIdBefore,
            int sourceQuantityBefore,
            int destinationQuantityBefore,
            boolean opensActionMask
        ) {
            if (
                !request.present()
                    || scope == null
                    || scope.isBlank()
                    || semanticBlockIdBefore == null
                    || semanticBlockIdBefore.isBlank()
                    || runtimeBlockIdBefore < 0
                    || sourceQuantityBefore < 0
            ) {
                throw new IllegalArgumentException(
                    "Accepted World verb lacks preflight evidence"
                );
            }
            return new Execution(
                request,
                true,
                false,
                "",
                scope,
                false,
                opensActionMask,
                opensActionMask,
                worldTick,
                semanticBlockIdBefore,
                runtimeBlockIdBefore,
                sourceQuantityBefore,
                destinationQuantityBefore
            );
        }

        void observeStarted(long worldTick) {
            if (!accepted || finished || worldTick < requestedWorldTick) {
                throw new IllegalStateException(
                    "World verb start lifecycle is inconsistent"
                );
            }
            if (!started) {
                started = true;
                startedWorldTick = worldTick;
            }
        }

        void observeFinished(
            boolean nativeFailed,
            long worldTick,
            String semanticBlockIdAfter,
            int runtimeBlockIdAfter,
            int sourceQuantityAfter,
            int destinationQuantityAfter,
            boolean mutationApplied,
            boolean inventoryApplied,
            String geometryDeltaSha256,
            String inventoryDeltaSha256,
            String dropOutcomeSha256
        ) {
            if (
                !accepted
                    || !started
                    || worldTick < startedWorldTick
                    || semanticBlockIdAfter == null
                    || semanticBlockIdAfter.isBlank()
                    || runtimeBlockIdAfter < 0
                    || sourceQuantityAfter < 0
                    || geometryDeltaSha256 == null
                    || !geometryDeltaSha256.matches("[0-9a-f]{64}")
                    || inventoryDeltaSha256 == null
                    || !inventoryDeltaSha256.matches("[0-9a-f]{64}")
                    || dropOutcomeSha256 == null
            ) {
                throw new IllegalStateException(
                    "World verb completion evidence is inconsistent"
                );
            }
            finished = true;
            failed = nativeFailed;
            acknowledgementComplete = true;
            this.mutationApplied = mutationApplied;
            this.inventoryApplied = inventoryApplied;
            this.semanticBlockIdAfter = semanticBlockIdAfter;
            this.runtimeBlockIdAfter = runtimeBlockIdAfter;
            this.sourceQuantityAfter = sourceQuantityAfter;
            this.destinationQuantityAfter = destinationQuantityAfter;
            this.geometryDeltaSha256 = geometryDeltaSha256;
            this.inventoryDeltaSha256 = inventoryDeltaSha256;
            this.dropOutcomeSha256 = dropOutcomeSha256;
            finishedWorldTick = worldTick;
        }

        void observeFinishedWithoutAcknowledgement(
            boolean nativeFailed,
            long worldTick
        ) {
            if (
                !accepted
                    || !started
                    || worldTick < startedWorldTick
            ) {
                throw new IllegalStateException(
                    "World verb completion lifecycle is inconsistent"
                );
            }
            finished = true;
            failed = nativeFailed;
            finishedWorldTick = worldTick;
        }

        NativeWorldVerbRequest request() {
            return request;
        }

        boolean accepted() {
            return accepted;
        }

        boolean started() {
            return started;
        }

        boolean active() {
            return accepted && started && !finished;
        }

        boolean finished() {
            return finished;
        }

        boolean failed() {
            return failed;
        }

        boolean pending() {
            return accepted && !finished;
        }

        String rejectReason() {
            return rejectReason;
        }

        String executionScope() {
            return executionScope;
        }

        boolean publicPlayerCertified() {
            return publicPlayerCertified;
        }

        boolean headlessServerActorCertified() {
            return headlessServerActorCertified;
        }

        boolean opensActionMask() {
            return opensActionMask && acknowledgementComplete;
        }

        boolean acknowledgementComplete() {
            return acknowledgementComplete;
        }

        boolean mutationApplied() {
            return mutationApplied;
        }

        boolean inventoryApplied() {
            return inventoryApplied;
        }

        String semanticBlockIdBefore() {
            return semanticBlockIdBefore;
        }

        String semanticBlockIdAfter() {
            return semanticBlockIdAfter;
        }

        int runtimeBlockIdBefore() {
            return runtimeBlockIdBefore;
        }

        int runtimeBlockIdAfter() {
            return runtimeBlockIdAfter;
        }

        int sourceQuantityBefore() {
            return sourceQuantityBefore;
        }

        int sourceQuantityAfter() {
            return sourceQuantityAfter;
        }

        int destinationQuantityBefore() {
            return destinationQuantityBefore;
        }

        int destinationQuantityAfter() {
            return destinationQuantityAfter;
        }

        String geometryDeltaSha256() {
            return geometryDeltaSha256;
        }

        String inventoryDeltaSha256() {
            return inventoryDeltaSha256;
        }

        String dropOutcomeSha256() {
            return dropOutcomeSha256;
        }

        long requestedWorldTick() {
            return requestedWorldTick;
        }

        long startedWorldTick() {
            return startedWorldTick;
        }

        long finishedWorldTick() {
            return finishedWorldTick;
        }
    }
}
