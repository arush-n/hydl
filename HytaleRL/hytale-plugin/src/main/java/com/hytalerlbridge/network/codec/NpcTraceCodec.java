package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.imitation.NpcTraceBatch;
import com.hytalerlbridge.imitation.NpcTraceFrame;
import com.hytalerlbridge.imitation.NpcAttackActionSnapshot;
import com.hytalerlbridge.imitation.NpcAttackExecutionCause;
import com.hytalerlbridge.imitation.NpcDamageEventSnapshot;
import com.hytalerlbridge.imitation.NpcLifecycleEventSnapshot;
import com.hytalerlbridge.imitation.NpcInteractionSnapshot;
import com.hytalerlbridge.imitation.NpcInternalSnapshot;
import com.hytalerlbridge.imitation.NpcObservationSnapshot;
import com.hytalerlbridge.imitation.NpcWorldSnapshot;
import com.hytalerlbridge.observation.NativeActorEvidenceFrame;
import com.hytalerlbridge.worldgen.NativePerceptionChannels;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.UUID;
import org.msgpack.core.MessagePacker;

import static com.hytalerlbridge.network.wire.BinaryCodec.encodeFloat32LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeFloat64LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeInt32LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeInt64LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeInt16LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.packBinary;
import static com.hytalerlbridge.network.wire.BridgeIdentity.BRIDGE_SHA256;
import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;

/** Columnar MessagePack transport for bounded native NPC trace drains. */
public final class NpcTraceCodec {

    private static final boolean LEGACY_TRACE_GEOMETRY = Boolean.getBoolean(
        "hytalerl.trace.geometry.legacy"
    );

    private NpcTraceCodec() {}

    public static void sendNpcTrace(
        DataOutputStream output,
        NpcTraceBatch batch
    ) throws IOException {
        String bridgeSha256 = BRIDGE_SHA256.orElseThrow(
            () -> new IllegalStateException("Runtime bridge identity is unavailable")
        );
        sendFrame(output, packer -> packNpcTrace(packer, batch, bridgeSha256));
    }

    public static void packNpcTrace(
        MessagePacker packer,
        NpcTraceBatch batch,
        String bridgeSha256
    ) throws IOException {
        if (bridgeSha256 == null || !bridgeSha256.matches("[0-9A-Fa-f]{64}")) {
            throw new IllegalArgumentException("Runtime bridge identity must be a SHA-256");
        }
        List<NpcTraceFrame> frames = batch.frames();
        int count = frames.size();
        long[] ticks = new long[count];
        float[] deltas = new float[count];
        double[] states = new double[count * NpcTraceFrame.STATE_WIDTH];
        double[] controls = new double[count * NpcTraceFrame.CONTROL_WIDTH];
        int[] controlMasks = new int[count];
        double[] nextStates = new double[count * NpcTraceFrame.STATE_WIDTH];
        byte[] nativeControl = new byte[count];
        byte[] attackActionActive = new byte[count];
        byte[] attackActivations = new byte[count];
        byte[] attackCauseAvailable = new byte[count];
        byte[] attackCauseExecuted = new byte[count];
        int[] attackCauseCandidate = new int[count];
        byte[] attacks = new byte[count];
        byte[] nextAttacks = new byte[count];
        float[] attackPause = new float[count];
        float[] nextAttackPause = new float[count];
        byte[] targetPresent = new byte[count];
        byte[] targetUuids = new byte[count * 16];
        byte[] nextTargetPresent = new byte[count];
        byte[] nextTargetUuids = new byte[count * 16];
        byte[] decisionTargetPresent = new byte[count];
        byte[] decisionTargetUuids = new byte[count * 16];
        int[] damageEventCounts = new int[count];
        byte[] damageEventOverflow = new byte[count];
        int[] lifecycleEventCounts = new int[count];
        byte[] lifecycleEventOverflow = new byte[count];
        int[] lifecycleSourceAvailable = new int[count];
        int[] lifecycleSourcePartial = new int[count];
        for (int index = 0; index < count; index++) {
            NpcTraceFrame frame = frames.get(index);
            ticks[index] = frame.tick();
            deltas[index] = frame.deltaSeconds();
            System.arraycopy(
                frame.state(),
                0,
                states,
                index * NpcTraceFrame.STATE_WIDTH,
                NpcTraceFrame.STATE_WIDTH
            );
            System.arraycopy(
                frame.control(),
                0,
                controls,
                index * NpcTraceFrame.CONTROL_WIDTH,
                NpcTraceFrame.CONTROL_WIDTH
            );
            controlMasks[index] = frame.controlMask();
            System.arraycopy(
                frame.nextState(),
                0,
                nextStates,
                index * NpcTraceFrame.STATE_WIDTH,
                NpcTraceFrame.STATE_WIDTH
            );
            nativeControl[index] = (byte) (frame.nativeControl() ? 1 : 0);
            attackActionActive[index] = (byte) (frame.attackActionActive() ? 1 : 0);
            attackActivations[index] = (byte) (frame.attackActivation() ? 1 : 0);
            NpcAttackExecutionCause cause = frame.attackExecutionCause();
            attackCauseAvailable[index] = (byte) (cause.available() ? 1 : 0);
            attackCauseExecuted[index] = (byte) (cause.executed() ? 1 : 0);
            attackCauseCandidate[index] = cause.candidateIndex();
            attacks[index] = (byte) (frame.combatAttack() ? 1 : 0);
            nextAttacks[index] = (byte) (frame.nextCombatAttack() ? 1 : 0);
            attackPause[index] = frame.attackPauseSeconds();
            nextAttackPause[index] = frame.nextAttackPauseSeconds();
            if (frame.targetUuid() != null) {
                targetPresent[index] = 1;
                System.arraycopy(
                    uuidBytes(frame.targetUuid()),
                    0,
                    targetUuids,
                    index * 16,
                    16
                );
            }
            if (frame.nextTargetUuid() != null) {
                nextTargetPresent[index] = 1;
                System.arraycopy(
                    uuidBytes(frame.nextTargetUuid()),
                    0,
                    nextTargetUuids,
                    index * 16,
                    16
                );
            }
            if (frame.decisionTargetUuid() != null) {
                decisionTargetPresent[index] = 1;
                System.arraycopy(
                    uuidBytes(frame.decisionTargetUuid()),
                    0,
                    decisionTargetUuids,
                    index * 16,
                    16
                );
            }
            damageEventCounts[index] = frame.damageEventCount();
            damageEventOverflow[index] = (byte) (
                frame.damageEventOverflow() ? 1 : 0
            );
            lifecycleEventCounts[index] = frame.lifecycleEventCount();
            lifecycleEventOverflow[index] = (byte) (
                frame.lifecycleEventOverflow() ? 1 : 0
            );
            lifecycleSourceAvailable[index] =
                frame.lifecycleSourceAvailableBits();
            lifecycleSourcePartial[index] = frame.lifecycleSourcePartialBits();
        }

        packer.packMapHeader(111);
        string(packer, "type", "npc_transition_trace");
        string(packer, "schema", NpcTraceBatch.SCHEMA);
        integer(packer, "version", NpcTraceBatch.VERSION);
        string(packer, "bridge_sha256", bridgeSha256);
        string(packer, "server_version", batch.serverVersion());
        string(packer, "world", batch.world());
        string(packer, "worldgen_provider", batch.worldgenProvider());
        string(packer, "worldgen_version", batch.worldgenVersion());
        packer.packString("seed");
        packer.packLong(batch.seed());
        stringMap(packer, "environment_parameters", batch.environmentParameters());
        packBinary(packer, "trace_uuid_bytes", uuidBytes(batch.traceUuid()));
        packBinary(packer, "npc_uuid_bytes", uuidBytes(batch.npcUuid()));
        string(packer, "role", batch.role());
        string(packer, "status", batch.status());
        integer(packer, "capacity", batch.capacity());
        packer.packString("total_captured");
        packer.packLong(batch.totalCaptured());
        integer(packer, "emitted_count", count);
        string(packer, "state_layout", NpcTraceFrame.STATE_LAYOUT);
        string(packer, "control_layout", NpcTraceFrame.CONTROL_LAYOUT);
        string(packer, "control_mask_layout", NpcTraceFrame.CONTROL_MASK_LAYOUT);
        string(
            packer,
            "action_contract_schema",
            NpcTraceBatch.ACTION_CONTRACT_SCHEMA
        );
        integer(
            packer,
            "action_contract_version",
            NpcTraceBatch.ACTION_CONTRACT_VERSION
        );
        string(
            packer,
            "action_contract_sha256",
            NpcTraceBatch.ACTION_CONTRACT_SHA256
        );
        string(packer, "event_contract_schema", NpcTraceBatch.EVENT_CONTRACT_SCHEMA);
        integer(packer, "event_contract_version", NpcTraceBatch.EVENT_CONTRACT_VERSION);
        string(
            packer,
            "event_contract_sha256",
            NpcTraceBatch.EVENT_CONTRACT_SHA256
        );
        string(
            packer,
            "lifecycle_contract_schema",
            NpcTraceBatch.LIFECYCLE_CONTRACT_SCHEMA
        );
        integer(
            packer,
            "lifecycle_contract_version",
            NpcTraceBatch.LIFECYCLE_CONTRACT_VERSION
        );
        string(
            packer,
            "lifecycle_contract_sha256",
            NpcTraceBatch.LIFECYCLE_CONTRACT_SHA256
        );
        string(packer, "angle_units", "radians");
        packBinary(packer, "ticks_i64_le", encodeInt64LittleEndian(ticks));
        packBinary(packer, "delta_seconds_f32_le", encodeFloat32LittleEndian(deltas));
        packBinary(packer, "state_f64_le_rows", encodeFloat64LittleEndian(states));
        packBinary(packer, "control_f64_le_rows", encodeFloat64LittleEndian(controls));
        packBinary(packer, "control_mask_i32_le", encodeInt32LittleEndian(controlMasks));
        packBinary(packer, "next_state_f64_le_rows", encodeFloat64LittleEndian(nextStates));
        packBinary(packer, "native_control_u8", nativeControl);
        packBinary(packer, "attack_action_active_u8", attackActionActive);
        packBinary(packer, "attack_activation_u8", attackActivations);
        string(
            packer,
            "attack_execution_cause_layout",
            NpcAttackExecutionCause.LAYOUT
        );
        packBinary(
            packer,
            "attack_execution_cause_available_u8",
            attackCauseAvailable
        );
        packBinary(
            packer,
            "attack_execution_cause_executed_u8",
            attackCauseExecuted
        );
        packBinary(
            packer,
            "attack_execution_cause_candidate_i32_le",
            encodeInt32LittleEndian(attackCauseCandidate)
        );
        strings(
            packer,
            "attack_execution_cause_interaction_types",
            frames.stream().map(frame ->
                frame.attackExecutionCause().interactionType()
            ).toList()
        );
        strings(
            packer,
            "attack_execution_cause_interaction_ids",
            frames.stream().map(frame ->
                frame.attackExecutionCause().interactionId()
            ).toList()
        );
        strings(
            packer,
            "attack_execution_cause_paths",
            frames.stream().map(frame -> frame.attackExecutionCause().path()).toList()
        );
        packBinary(packer, "combat_attack_u8", attacks);
        packBinary(packer, "next_combat_attack_u8", nextAttacks);
        packBinary(
            packer,
            "attack_pause_seconds_f32_le",
            encodeFloat32LittleEndian(attackPause)
        );
        packBinary(
            packer,
            "next_attack_pause_seconds_f32_le",
            encodeFloat32LittleEndian(nextAttackPause)
        );
        packBinary(packer, "target_present_u8", targetPresent);
        packBinary(packer, "target_uuid_bytes", targetUuids);
        packBinary(packer, "next_target_present_u8", nextTargetPresent);
        packBinary(packer, "next_target_uuid_bytes", nextTargetUuids);
        packBinary(packer, "decision_target_present_u8", decisionTargetPresent);
        packBinary(packer, "decision_target_uuid_bytes", decisionTargetUuids);
        strings(packer, "state_names", frames.stream().map(NpcTraceFrame::stateName).toList());
        strings(
            packer,
            "next_state_names",
            frames.stream().map(NpcTraceFrame::nextStateName).toList()
        );
        string(packer, "internal_layout", NpcInternalSnapshot.LAYOUT);
        internalStates(packer, "internal_states", frames, 0);
        internalStates(packer, "decision_internal_states", frames, 1);
        internalStates(packer, "next_internal_states", frames, 2);
        strings(packer, "body_instructions", frames.stream().map(NpcTraceFrame::bodyInstruction).toList());
        strings(
            packer,
            "body_decision_paths",
            frames.stream().map(NpcTraceFrame::bodyDecisionPath).toList()
        );
        strings(packer, "head_instructions", frames.stream().map(NpcTraceFrame::headInstruction).toList());
        strings(
            packer,
            "head_decision_paths",
            frames.stream().map(NpcTraceFrame::headDecisionPath).toList()
        );
        packer.packString("active_actions");
        packer.packArrayHeader(count);
        for (NpcTraceFrame frame : frames) {
            packer.packArrayHeader(frame.activeActions().size());
            for (String action : frame.activeActions()) packer.packString(action);
        }
        stringLists(
            packer,
            "active_action_paths",
            frames.stream().map(NpcTraceFrame::activeActionPaths).toList()
        );
        string(packer, "attack_action_layout", NpcAttackActionSnapshot.LAYOUT);
        attackActions(packer, "attack_actions", frames, false);
        attackActions(packer, "next_attack_actions", frames, true);
        string(packer, "interaction_layout", NpcInteractionSnapshot.LAYOUT);
        interactions(packer, "interactions", frames, false);
        interactions(packer, "next_interactions", frames, true);
        string(packer, "damage_event_layout", NpcDamageEventSnapshot.LAYOUT);
        integer(
            packer,
            "damage_event_capacity",
            NpcDamageEventSnapshot.CAPACITY
        );
        packBinary(
            packer,
            "damage_event_counts_i32_le",
            encodeInt32LittleEndian(damageEventCounts)
        );
        packBinary(packer, "damage_event_overflow_u8", damageEventOverflow);
        damageEvents(packer, frames);
        string(packer, "lifecycle_event_layout", NpcLifecycleEventSnapshot.LAYOUT);
        integer(
            packer,
            "lifecycle_event_capacity",
            NpcLifecycleEventSnapshot.CAPACITY
        );
        string(
            packer,
            "lifecycle_source_layout",
            NpcLifecycleEventSnapshot.SOURCE_LAYOUT
        );
        packBinary(
            packer,
            "lifecycle_event_counts_i32_le",
            encodeInt32LittleEndian(lifecycleEventCounts)
        );
        packBinary(
            packer,
            "lifecycle_event_overflow_u8",
            lifecycleEventOverflow
        );
        packBinary(
            packer,
            "lifecycle_source_available_i32_le",
            encodeInt32LittleEndian(lifecycleSourceAvailable)
        );
        packBinary(
            packer,
            "lifecycle_source_partial_i32_le",
            encodeInt32LittleEndian(lifecycleSourcePartial)
        );
        lifecycleEvents(packer, frames);
        string(packer, "actor_evidence_schema", NativeActorEvidenceFrame.SCHEMA);
        integer(packer, "actor_evidence_version", NativeActorEvidenceFrame.VERSION);
        string(
            packer,
            "actor_evidence_contract_sha256",
            NativeActorEvidenceFrame.CONTRACT_SHA256
        );
        actorEvidence(packer, "actor_evidence", frames, false);
        actorEvidence(packer, "next_actor_evidence", frames, true);
        targetActorEvidence(packer, "target_actor_evidence", frames, false);
        targetActorEvidence(packer, "next_target_actor_evidence", frames, true);
        string(packer, "observation_schema", NpcObservationSnapshot.SCHEMA);
        integer(packer, "observation_version", NpcObservationSnapshot.VERSION);
        string(
            packer,
            "observation_contract_sha256",
            NpcObservationSnapshot.CONTRACT_SHA256
        );
        string(
            packer,
            "observation_group_layout",
            NpcObservationSnapshot.GROUP_LAYOUT
        );
        integer(
            packer,
            "observation_attack_candidate_capacity",
            NpcObservationSnapshot.ATTACK_CANDIDATE_CAPACITY
        );
        integer(
            packer,
            "observation_perceptible_npc_capacity",
            NpcObservationSnapshot.PERCEPTIBLE_NPC_CAPACITY
        );
        integer(
            packer,
            "observation_perceptible_entity_capacity",
            NpcObservationSnapshot.PERCEPTIBLE_ENTITY_CAPACITY
        );
        observations(packer, "observations", frames, false);
        observations(packer, "next_observations", frames, true);
        string(packer, "worldview_schema", NpcWorldSnapshot.SCHEMA);
        integer(packer, "worldview_version", NpcWorldSnapshot.VERSION);
        integer(packer, "worldview_actor_capacity", NpcWorldSnapshot.ACTOR_CAPACITY);
        string(packer, "worldview_actor_layout", NpcWorldSnapshot.ACTOR_LAYOUT);
        integer(packer, "worldview_entity_capacity", NpcWorldSnapshot.ENTITY_CAPACITY);
        string(packer, "worldview_entity_layout", NpcWorldSnapshot.ENTITY_LAYOUT);
        worldviews(packer, "worldviews", frames, false);
        worldviews(packer, "next_worldviews", frames, true);
    }

    private static void observations(
        MessagePacker packer,
        String key,
        List<NpcTraceFrame> frames,
        boolean next
    ) throws IOException {
        packer.packString(key);
        packer.packArrayHeader(frames.size());
        for (NpcTraceFrame frame : frames) {
            observation(
                packer,
                next ? frame.nextObservation() : frame.observation()
            );
        }
    }

    private static void observation(
        MessagePacker packer,
        NpcObservationSnapshot value
    ) throws IOException {
        packer.packMapHeader(30);
        packer.packString("world_tick");
        packer.packLong(value.worldTick());
        packer.packString("action_capability_evidence");
        ObservationCodec.packNativeActorEvidence(
            packer,
            value.actionCapabilityEvidence()
        );
        packer.packString("complete_group_bits");
        packer.packLong(value.completeGroupBits());
        packer.packString("partial_group_bits");
        packer.packLong(value.partialGroupBits());
        packer.packString("unavailable_group_bits");
        packer.packLong(value.unavailableGroupBits());
        packer.packString("geometry");
        if (LEGACY_TRACE_GEOMETRY) {
            ObservationCodec.packGeometry(packer, value.geometry());
        } else {
            ObservationCodec.packTraceGeometry(packer, value.geometry());
        }
        packer.packString("inventory");
        ObservationCodec.packNativeInventory(packer, value.inventory());
        packer.packString("role_opaque_cell_mask_available");
        packer.packBoolean(value.roleOpaqueCellMaskAvailable());
        packBinary(
            packer,
            "role_opaque_cell_mask_u8",
            booleans(value.roleOpaqueCellMask())
        );
        packer.packString("local_perception");
        localPerception(packer, value.localPerception());
        packer.packString("combat_lifecycle_available");
        packer.packBoolean(value.combatLifecycleAvailable());
        packer.packString("combat_attack_executing");
        packer.packBoolean(value.combatAttackExecuting());
        packer.packString("attack_pause_seconds");
        packer.packFloat(value.attackPauseSeconds());
        packer.packString("attack_candidate_count");
        packer.packInt(value.attackCandidateCount());
        packer.packString("attack_candidate_overflow");
        packer.packBoolean(value.attackCandidateOverflow());
        packer.packString("attack_candidates");
        attackActionList(packer, value.attackCandidates());
        packer.packString("target_attack_candidate_count");
        packer.packInt(value.targetAttackCandidateCount());
        packer.packString("target_attack_candidate_overflow");
        packer.packBoolean(value.targetAttackCandidateOverflow());
        packer.packString("target_attack_candidates");
        attackActionList(packer, value.targetAttackCandidates());
        packer.packString("perception_available");
        packer.packBoolean(value.perceptionAvailable());
        packer.packString("target_present");
        packer.packBoolean(value.targetPresent());
        packer.packString("target_perceptible");
        packer.packBoolean(value.targetPerceptible());
        packer.packString("target_distance");
        packer.packDouble(value.targetDistance());
        packer.packString("perceptible_npc_count");
        packer.packInt(value.perceptibleNpcCount());
        packer.packString("perceptible_npc_overflow");
        packer.packBoolean(value.perceptibleNpcOverflow());
        packer.packString("perceptible_npc_uuids");
        packer.packArrayHeader(value.perceptibleNpcUuids().size());
        for (UUID uuid : value.perceptibleNpcUuids()) {
            binary(packer, uuidBytes(uuid));
        }
        packer.packString("perceptible_entity_count");
        packer.packInt(value.perceptibleEntityCount());
        packer.packString("perceptible_entity_overflow");
        packer.packBoolean(value.perceptibleEntityOverflow());
        packer.packString("perceptible_entity_indices");
        packer.packArrayHeader(value.perceptibleEntityIndices().size());
        for (int index : value.perceptibleEntityIndices()) packer.packInt(index);
        packer.packString("status_failure_bits");
        packer.packInt(value.statusFailureBits());
    }

    private static void attackActionList(
        MessagePacker packer,
        List<NpcAttackActionSnapshot> values
    ) throws IOException {
        packer.packArrayHeader(values.size());
        for (NpcAttackActionSnapshot value : values) {
            attackAction(packer, value);
        }
    }

    private static void internalStates(
        MessagePacker packer,
        String key,
        List<NpcTraceFrame> frames,
        int boundary
    ) throws IOException {
        packer.packString(key);
        packer.packArrayHeader(frames.size());
        for (NpcTraceFrame frame : frames) {
            NpcInternalSnapshot value = switch (boundary) {
                case 0 -> frame.internalState();
                case 1 -> frame.decisionInternalState();
                case 2 -> frame.nextInternalState();
                default -> throw new IllegalArgumentException("unknown internal boundary");
            };
            internalState(packer, value);
        }
    }

    private static void internalState(
        MessagePacker packer,
        NpcInternalSnapshot value
    ) throws IOException {
        packer.packArrayHeader(17);
        packer.packString(value.stateName());
        packer.packInt(value.stateIndex());
        packer.packInt(value.subStateIndex());
        packer.packBoolean(value.busy());
        packer.packBoolean(value.transitioning());
        packer.packBoolean(value.roleChangeRequested());
        packer.packBoolean(value.terminalAction());
        packer.packBoolean(value.backingAway());
        packer.packString(value.steeringMotion());
        packer.packBoolean(value.motionControllerPresent());
        packer.packBoolean(value.motionInProgress());
        packer.packBoolean(value.obstructed());
        packer.packDouble(value.currentSpeed());
        packer.packDouble(value.maximumSpeed());
        doubles(packer, value.avoidanceSteering());
        doubles(packer, value.separationSteering());
        packer.packArrayHeader(value.markedTargets().size());
        for (NpcInternalSnapshot.MarkedTarget target : value.markedTargets()) {
            packer.packArrayHeader(3);
            packer.packInt(target.slot());
            packer.packString(target.name());
            binary(packer, uuidBytes(target.uuid()));
        }
    }

    private static void attackActions(
        MessagePacker packer,
        String key,
        List<NpcTraceFrame> frames,
        boolean next
    ) throws IOException {
        packer.packString(key);
        packer.packArrayHeader(frames.size());
        for (NpcTraceFrame frame : frames) {
            List<NpcAttackActionSnapshot> row = next
                ? frame.nextAttackActions()
                : frame.attackActions();
            packer.packArrayHeader(row.size());
            for (NpcAttackActionSnapshot value : row) attackAction(packer, value);
        }
    }

    private static void attackAction(
        MessagePacker packer,
        NpcAttackActionSnapshot value
    ) throws IOException {
        packer.packArrayHeader(NpcAttackActionSnapshot.WIDTH);
        packer.packString(value.label());
        packer.packBoolean(value.active());
        packer.packBoolean(value.triggered());
        packer.packBoolean(value.ready());
        packer.packFloat(value.aimingSecondsRemaining());
        packer.packFloat(value.chargeSeconds());
        packer.packString(value.interactionType());
        packer.packString(value.interactionId());
        packer.packString(value.path());
    }

    private static void localPerception(
        MessagePacker packer,
        NativePerceptionChannels value
    ) throws IOException {
        packer.packMapHeader(11);
        string(packer, "schema", NativePerceptionChannels.SCHEMA);
        integer(packer, "version", NativePerceptionChannels.VERSION);
        string(packer, "cell_order", "geometry_cell_index_dx_dy_dz");
        integer(packer, "sample_count", value.sampleCount());
        packBinary(packer, "available_u8", value.available());
        packBinary(
            packer,
            "channel_validity_u8_bits",
            value.channelValidity()
        );
        packBinary(
            packer,
            "heightmap_i16_le",
            encodeInt16LittleEndian(value.heightmap())
        );
        packBinary(packer, "sky_light_u8", value.skyLight());
        packBinary(packer, "block_light_rgb_u8", value.blockLightRgb());
        packBinary(
            packer,
            "environment_i32_le",
            encodeInt32LittleEndian(value.environment())
        );
        packBinary(
            packer,
            "tint_argb_i32_le",
            encodeInt32LittleEndian(value.tintArgb())
        );
    }

    private static void actorEvidence(
        MessagePacker packer,
        String key,
        List<NpcTraceFrame> frames,
        boolean next
    ) throws IOException {
        packer.packString(key);
        packer.packArrayHeader(frames.size());
        for (NpcTraceFrame frame : frames) {
            ObservationCodec.packNativeActor(
                packer,
                next ? frame.nextActorEvidence() : frame.actorEvidence()
            );
        }
    }

    private static void targetActorEvidence(
        MessagePacker packer,
        String key,
        List<NpcTraceFrame> frames,
        boolean next
    ) throws IOException {
        packer.packString(key);
        packer.packArrayHeader(frames.size());
        for (NpcTraceFrame frame : frames) {
            ObservationCodec.packNativeActor(
                packer,
                next
                    ? frame.nextTargetActorEvidence()
                    : frame.targetActorEvidence()
            );
        }
    }

    private static void worldviews(
        MessagePacker packer,
        String key,
        List<NpcTraceFrame> frames,
        boolean next
    ) throws IOException {
        packer.packString(key);
        packer.packArrayHeader(frames.size());
        for (NpcTraceFrame frame : frames) {
            worldview(packer, next ? frame.nextWorldview() : frame.worldview());
        }
    }

    private static void worldview(
        MessagePacker packer,
        NpcWorldSnapshot value
    ) throws IOException {
        packer.packArrayHeader(14);
        packer.packLong(value.worldTick());
        packer.packLong(value.gameTimeEpochSecond());
        packer.packInt(value.gameTimeNano());
        packer.packFloat(value.dayProgress());
        packer.packFloat(value.sunlightFactor());
        packer.packInt(value.moonPhase());
        packer.packInt(value.npcCount());
        packer.packBoolean(value.overflow());
        packer.packArrayHeader(value.actors().size());
        for (NpcWorldSnapshot.Actor actor : value.actors()) {
            packer.packArrayHeader(7);
            binary(packer, uuidBytes(actor.uuid()));
            packer.packString(actor.role());
            doubles(packer, actor.state());
            packer.packString(actor.stateName());
            if (actor.targetUuid() == null) {
                packer.packNil();
            } else {
                binary(packer, uuidBytes(actor.targetUuid()));
            }
            packer.packBoolean(actor.combatAttack());
            packer.packFloat(actor.attackPauseSeconds());
        }
        packer.packInt(value.entityCount());
        packer.packBoolean(value.entityOverflow());
        packer.packArrayHeader(value.entities().size());
        for (NpcWorldSnapshot.Entity entity : value.entities()) {
            packer.packArrayHeader(18);
            packer.packInt(entity.entityIndex());
            nullableUuid(packer, entity.uuid());
            packer.packInt(entity.flags());
            packer.packString(entity.assetId());
            packer.packString(entity.modelAssetId());
            doubles(packer, entity.position());
            doubles(packer, entity.rotation());
            packer.packBoolean(entity.velocityAvailable());
            doubles(packer, entity.velocity());
            packer.packBoolean(entity.boundsAvailable());
            doubles(packer, entity.bounds());
            nullableUuid(packer, entity.ownerUuid());
            packer.packString(entity.physicsState());
            packer.packBoolean(entity.lifecycleTimingAvailable());
            packer.packLong(entity.lifecycleStartEpochSecond());
            packer.packInt(entity.lifecycleStartNano());
            packer.packLong(entity.lifecycleEndEpochSecond());
            packer.packInt(entity.lifecycleEndNano());
        }
        packer.packLong(value.simulationTimeEpochSecond());
        packer.packInt(value.simulationTimeNano());
    }

    private static void nullableUuid(MessagePacker packer, UUID value)
        throws IOException {
        if (value == null) packer.packNil();
        else binary(packer, uuidBytes(value));
    }

    private static void interactions(
        MessagePacker packer,
        String key,
        List<NpcTraceFrame> frames,
        boolean next
    ) throws IOException {
        packer.packString(key);
        packer.packArrayHeader(frames.size());
        for (NpcTraceFrame frame : frames) {
            List<NpcInteractionSnapshot> row = next
                ? frame.nextInteractions()
                : frame.interactions();
            packer.packArrayHeader(row.size());
            for (NpcInteractionSnapshot value : row) interaction(packer, value);
        }
    }

    private static void damageEvents(
        MessagePacker packer,
        List<NpcTraceFrame> frames
    ) throws IOException {
        packer.packString("damage_events");
        packer.packArrayHeader(frames.size());
        for (NpcTraceFrame frame : frames) {
            packer.packArrayHeader(frame.damageEvents().size());
            for (NpcDamageEventSnapshot value : frame.damageEvents()) {
                packer.packArrayHeader(NpcDamageEventSnapshot.WIDTH);
                packer.packString(value.sourceType());
                packer.packString(value.environmentType());
                packer.packInt(value.damageCauseIndex());
                packer.packString(value.damageCauseId());
                packer.packFloat(value.initialAmount());
                packer.packFloat(value.finalAmount());
                packer.packBoolean(value.cancelled());
                packer.packBoolean(value.blocked());
                packer.packBoolean(value.actorSource());
                packer.packBoolean(value.actorTarget());
                packer.packInt(value.sourceEntityIndex());
                nullableUuid(packer, value.sourceUuid());
                packer.packInt(value.targetEntityIndex());
                nullableUuid(packer, value.targetUuid());
                packer.packInt(value.projectileEntityIndex());
                nullableUuid(packer, value.projectileUuid());
                packer.packBoolean(value.hitLocationAvailable());
                doubles(packer, value.hitLocation());
                packer.packBoolean(value.healthAvailable());
                packer.packFloat(value.targetHealthAfter());
                packer.packFloat(value.targetMaxHealth());
                packer.packBoolean(value.lethal());
            }
        }
    }

    private static void lifecycleEvents(
        MessagePacker packer,
        List<NpcTraceFrame> frames
    ) throws IOException {
        packer.packString("lifecycle_events");
        packer.packArrayHeader(frames.size());
        for (NpcTraceFrame frame : frames) {
            packer.packArrayHeader(frame.lifecycleEvents().size());
            for (NpcLifecycleEventSnapshot value : frame.lifecycleEvents()) {
                packer.packArrayHeader(NpcLifecycleEventSnapshot.WIDTH);
                packer.packString(value.kind());
                packer.packString(value.origin());
                packer.packString(value.subject());
                packer.packString(value.key());
                packer.packInt(value.index());
                packer.packDouble(value.valueBefore());
                packer.packDouble(value.valueAfter());
                packer.packDouble(value.auxiliary0());
                packer.packDouble(value.auxiliary1());
                packer.packString(value.textBefore());
                packer.packString(value.textAfter());
                packer.packBoolean(value.successful());
                packer.packBoolean(value.complete());
                packer.packInt(value.entityIndex());
                nullableUuid(packer, value.entityUuid());
                nullableUuid(packer, value.ownerUuid());
                packer.packBoolean(value.positionAvailable());
                doubles(packer, value.position());
                packer.packInt(value.flags());
            }
        }
    }

    private static void interaction(
        MessagePacker packer,
        NpcInteractionSnapshot value
    ) throws IOException {
        packer.packArrayHeader(NpcInteractionSnapshot.WIDTH);
        packer.packString(value.source());
        packer.packString(value.type());
        packer.packString(value.baseType());
        packer.packInt(value.chainId());
        packer.packString(value.initialRootId());
        packer.packString(value.rootId());
        packer.packString(value.serverState());
        packer.packString(value.clientState());
        packer.packString(value.finalState());
        packer.packFloat(value.timeSeconds());
        packer.packFloat(value.timeShift());
        packer.packInt(value.operationCounter());
        packer.packInt(value.simulatedOperationCounter());
        packer.packInt(value.operationIndex());
        packer.packInt(value.clientOperationIndex());
        packer.packInt(value.callDepth());
        packer.packInt(value.simulatedCallDepth());
        packer.packBoolean(value.predicted());
        packer.packBoolean(value.requiresClient());
        packer.packBoolean(value.firstRun());
        packer.packBoolean(value.preTicked());
        packer.packBoolean(value.desynced());
        if (value.targetUuid() == null) {
            packer.packNil();
        } else {
            byte[] bytes = uuidBytes(value.targetUuid());
            packer.packBinaryHeader(bytes.length);
            packer.writePayload(bytes);
        }
    }

    private static byte[] uuidBytes(UUID uuid) {
        return ByteBuffer.allocate(16)
            .order(ByteOrder.BIG_ENDIAN)
            .putLong(uuid.getMostSignificantBits())
            .putLong(uuid.getLeastSignificantBits())
            .array();
    }

    private static void string(MessagePacker packer, String key, String value)
        throws IOException {
        packer.packString(key);
        packer.packString(value);
    }

    private static void integer(MessagePacker packer, String key, int value)
        throws IOException {
        packer.packString(key);
        packer.packInt(value);
    }

    private static void strings(
        MessagePacker packer,
        String key,
        List<String> values
    ) throws IOException {
        packer.packString(key);
        packer.packArrayHeader(values.size());
        for (String value : values) packer.packString(value);
    }

    private static void stringLists(
        MessagePacker packer,
        String key,
        List<List<String>> values
    ) throws IOException {
        packer.packString(key);
        packer.packArrayHeader(values.size());
        for (List<String> row : values) {
            packer.packArrayHeader(row.size());
            for (String value : row) packer.packString(value);
        }
    }

    private static void doubles(MessagePacker packer, double[] values)
        throws IOException {
        packer.packArrayHeader(values.length);
        for (double value : values) packer.packDouble(value);
    }

    private static void binary(MessagePacker packer, byte[] values)
        throws IOException {
        packer.packBinaryHeader(values.length);
        packer.writePayload(values);
    }

    private static byte[] booleans(boolean[] values) {
        byte[] result = new byte[values.length];
        for (int index = 0; index < values.length; index++) {
            result[index] = (byte) (values[index] ? 1 : 0);
        }
        return result;
    }

    private static void stringMap(
        MessagePacker packer,
        String key,
        java.util.Map<String, String> values
    ) throws IOException {
        packer.packString(key);
        packer.packMapHeader(values.size());
        for (var entry : values.entrySet()) {
            packer.packString(entry.getKey());
            packer.packString(entry.getValue());
        }
    }
}
