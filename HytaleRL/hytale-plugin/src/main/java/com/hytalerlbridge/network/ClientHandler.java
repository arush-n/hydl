package com.hytalerlbridge.network;

import com.hytalerlbridge.action.group.NativeGroupAction;
import com.hytalerlbridge.environment.EnvironmentManager;
import com.hytalerlbridge.environment.EnvironmentOptions;
import com.hytalerlbridge.environment.StepResult;
import com.hytalerlbridge.entity.PrivilegedEntityQuery;
import com.hytalerlbridge.entity.PrivilegedNpcSnapshot;
import com.hytalerlbridge.entity.PrivilegedNpcUuidQuery;
import com.hytalerlbridge.network.codec.ResourceOverrideCodec;
import com.hytalerlbridge.observation.NativeActorEvidenceRequest;
import com.hytalerlbridge.worldgen.WorldgenStructureMarkerQuery;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.UUID;
import org.msgpack.value.Value;
import static com.hytalerlbridge.network.wire.BinaryCodec.decodeFloat64LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.decodeInt32LittleEndianTriples;
import static com.hytalerlbridge.network.wire.MessageFields.checkedInt;
import static com.hytalerlbridge.network.wire.MessageFields.getBinaryField;
import static com.hytalerlbridge.network.wire.MessageFields.getBooleanField;
import static com.hytalerlbridge.network.wire.MessageFields.getField;
import static com.hytalerlbridge.network.wire.MessageFields.getLongField;
import static com.hytalerlbridge.network.wire.MessageFields.getMapField;
import static com.hytalerlbridge.network.wire.MessageFields.getOptionalDoubleField;
import static com.hytalerlbridge.network.wire.MessageFields.getOptionalIntListField;
import static com.hytalerlbridge.network.wire.MessageFields.getOptionalStringField;
import static com.hytalerlbridge.network.wire.MessageFields.getOptionalStringListField;
import static com.hytalerlbridge.network.wire.MessageFields.getStringField;
import static com.hytalerlbridge.network.wire.FrameWriter.readFrame;
import static com.hytalerlbridge.network.wire.FrameWriter.sendAck;
import static com.hytalerlbridge.network.wire.FrameWriter.sendError;
import static com.hytalerlbridge.network.codec.SnapshotCodec.sendPrivilegedEntitySnapshot;
import static com.hytalerlbridge.network.codec.SnapshotCodec.sendPrivilegedNpcSnapshot;
import static com.hytalerlbridge.network.codec.NpcTraceCodec.sendNpcTrace;
import static com.hytalerlbridge.network.codec.WorldgenStructureMarkerCodec.sendWorldgenStructureMarkers;
import static com.hytalerlbridge.network.codec.ProbeCodec.sendNavigationPathProbe;
import static com.hytalerlbridge.network.codec.ProbeCodec.sendNavigationSuccessorProbe;
import static com.hytalerlbridge.network.codec.ProbeCodec.sendTraversalEdgeProbe;
import static com.hytalerlbridge.network.codec.ProbeCodec.sendTraversalProbe;
import static com.hytalerlbridge.network.codec.RegionCodec.sendRegionBlockSemantics;
import static com.hytalerlbridge.network.codec.RegionCodec.sendRegionFluidSemantics;
import static com.hytalerlbridge.network.codec.RegionCodec.sendRegionManifest;
import static com.hytalerlbridge.network.codec.RegionCodec.sendRegionLightSection;
import static com.hytalerlbridge.network.codec.RegionCodec.sendRegionSection;
import static com.hytalerlbridge.network.codec.EvidenceCodec.sendCraftingCatalogEvidence;
import static com.hytalerlbridge.network.codec.EvidenceCodec.sendDoorTransitionEvidence;
import static com.hytalerlbridge.network.codec.EvidenceCodec.sendDropProgramEvidence;
import static com.hytalerlbridge.network.codec.EvidenceCodec.sendItemInteractionEvidence;
import static com.hytalerlbridge.network.codec.EvidenceCodec.sendLineOfSightEvidence;
import static com.hytalerlbridge.network.codec.EvidenceCodec.sendMutableBlockCells;
import static com.hytalerlbridge.network.codec.EvidenceCodec.sendMutableBlockEvidence;
import static com.hytalerlbridge.network.codec.EvidenceCodec.sendPerceptionChannels;
import static com.hytalerlbridge.network.codec.ExplosionCodec.sendExplosionCandidateProbe;
import static com.hytalerlbridge.network.handler.ExplosionMutationHandler.handleExplosionMutation;
import static com.hytalerlbridge.network.handler.ExplosionDynamicsHandler.handleExplosionDynamics;
import static com.hytalerlbridge.network.handler.worldactions.PolicyWorldActionCaptureHandler.handle;
import static com.hytalerlbridge.network.codec.ObservationCodec.getNativeActorEvidenceRequest;
import static com.hytalerlbridge.network.codec.ObservationCodec.sendObservation;
import static com.hytalerlbridge.network.codec.PolicyCombatBindingCodec.decode;

/** Handles one length-prefixed MessagePack client connection. */
public final class ClientHandler implements Runnable {


    private static final System.Logger LOGGER = System.getLogger(ClientHandler.class.getName());

    private final Socket socket;
    private final EnvironmentManager environmentManager;
    private final int bufferSize;
    private String environmentId;
    private int ticksPerStep;
    private int maxEpisodeSteps;

    public ClientHandler(
        Socket socket,
        EnvironmentManager environmentManager,
        int bufferSize
    ) {
        this.socket = socket;
        this.environmentManager = environmentManager;
        this.bufferSize = bufferSize;
        this.ticksPerStep = environmentManager.getDefaultTicksPerStep();
        this.maxEpisodeSteps = environmentManager.getDefaultMaxEpisodeSteps();
    }

    @Override
    public void run() {
        try (
            DataInputStream input = new DataInputStream(
                new BufferedInputStream(socket.getInputStream(), bufferSize)
            );
            DataOutputStream output = new DataOutputStream(
                new BufferedOutputStream(socket.getOutputStream(), bufferSize)
            )
        ) {
            while (!socket.isClosed()) {
                Value raw;
                try {
                    raw = readFrame(input);
                } catch (EOFException exception) {
                    break;
                } catch (IllegalArgumentException exception) {
                    sendError(output, exception.getMessage());
                    break;
                }

                if (!raw.isMapValue()) {
                    sendError(output, "Expected map message");
                    continue;
                }

                Map<Value, Value> message = raw.asMapValue().map();
                String type = getStringField(message, "type");
                try {
                    if (!dispatch(type, message, output)) {
                        break;
                    }
                } catch (IllegalArgumentException | IllegalStateException exception) {
                    sendError(output, exception.getMessage());
                } catch (RuntimeException exception) {
                    LOGGER.log(
                        System.Logger.Level.ERROR,
                        "Unhandled bridge request failure for type " + type,
                        exception
                    );
                    sendError(
                        output,
                        "Bridge request failed: "
                            + exception.getClass().getSimpleName()
                            + ": "
                            + exception.getMessage()
                    );
                } catch (Error error) {
                    LOGGER.log(
                        System.Logger.Level.ERROR,
                        "Fatal bridge request failure for type " + type,
                        error
                    );
                    sendError(
                        output,
                        "Bridge request failed: "
                            + error.getClass().getSimpleName()
                            + ": "
                            + error.getMessage()
                    );
                    throw error;
                }
            }
        } catch (IOException exception) {
            LOGGER.log(System.Logger.Level.DEBUG,
                "Bridge client disconnected: " + exception.getMessage());
        } finally {
            cleanup();
        }
    }

    private boolean dispatch(
        String type,
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        return switch (type) {
            case "reset" -> {
                handleReset(message, output);
                yield true;
            }
            case "step" -> {
                handleStep(message, output);
                yield true;
            }
            case "config" -> {
                handleConfig(message, output);
                yield true;
            }
            case "region_manifest" -> {
                handleRegionManifest(message, output);
                yield true;
            }
            case "region_section" -> {
                handleRegionSection(message, output);
                yield true;
            }
            case "region_light_section" -> {
                handleRegionLightSection(message, output);
                yield true;
            }
            case "region_block_semantics" -> {
                handleRegionBlockSemantics(message, output);
                yield true;
            }
            case "region_fluid_semantics" -> {
                handleRegionFluidSemantics(message, output);
                yield true;
            }
            case "perception_channels" -> {
                handlePerceptionChannels(message, output);
                yield true;
            }
            case "los_evidence" -> {
                handleLineOfSightEvidence(output);
                yield true;
            }
            case "mutable_block_evidence" -> {
                handleMutableBlockEvidence(output);
                yield true;
            }
            case "mutable_block_cells" -> {
                handleMutableBlockCells(message, output);
                yield true;
            }
            case "drop_program_evidence" -> {
                handleDropProgramEvidence(message, output);
                yield true;
            }
            case "crafting_catalog_evidence" -> {
                handleCraftingCatalogEvidence(output);
                yield true;
            }
            case "item_interaction_evidence" -> {
                handleItemInteractionEvidence(message, output);
                yield true;
            }
            case "policy_world_action_capture" -> {
                handle(message, output, environmentManager, environmentId);
                yield true;
            }
            case "door_evidence" -> {
                handleDoorTransitionEvidence(output);
                yield true;
            }
            case "traversal_probe" -> {
                handleTraversalProbe(message, output);
                yield true;
            }
            case "traversal_edge_probe" -> {
                handleTraversalEdgeProbe(message, output);
                yield true;
            }
            case "navigation_path_probe" -> {
                handleNavigationPathProbe(message, output);
                yield true;
            }
            case "navigation_successor_probe" -> {
                handleNavigationSuccessorProbe(message, output);
                yield true;
            }
            case "privileged_entity_snapshot" -> {
                handlePrivilegedEntitySnapshot(message, output);
                yield true;
            }
            case "privileged_npc_snapshot" -> {
                handlePrivilegedNpcSnapshot(message, output);
                yield true;
            }
            case "privileged_npc_snapshot_by_uuid" -> {
                handlePrivilegedNpcSnapshotByUuid(message, output);
                yield true;
            }
            case "npc_transition_trace", "npc_imitation_trace" -> {
                handleNpcImitationTrace(message, output);
                yield true;
            }
            case "worldgen_structure_markers" -> {
                handleWorldgenStructureMarkers(message, output);
                yield true;
            }
            case "explosion_candidate_probe" -> {
                handleExplosionCandidateProbe(message, output);
                yield true;
            }
            case "explosion_mutation_probe" -> {
                handleExplosionMutation(
                    message,
                    output,
                    environmentManager,
                    environmentId
                );
                yield true;
            }
            case "explosion_dynamics_probe" -> {
                handleExplosionDynamics(
                    message,
                    output,
                    environmentManager,
                    environmentId
                );
                yield true;
            }
            case "close" -> {
                sendAck(output);
                yield false;
            }
            default -> {
                sendError(output, "Unknown message type: " + type);
                yield true;
            }
        };
    }

    private void handleReset(Map<Value, Value> message, DataOutputStream output)
        throws IOException {
        String taskId = getStringField(message, "task_id");
        long seed = getLongField(message, "seed", System.currentTimeMillis());
        Map<Value, Value> options = getMapField(message, "options");
        int curriculumPhase = checkedInt(
            getLongField(options, "curriculum_phase", -1),
            "curriculum_phase"
        );
        if (curriculumPhase < -1 || curriculumPhase > 3) {
            throw new IllegalArgumentException("curriculum_phase must be -1..3");
        }
        Double nativeTimeDilation = getOptionalDoubleField(
            options,
            "native_time_dilation"
        );
        NativeActorEvidenceRequest actorEvidenceRequest =
            getNativeActorEvidenceRequest(options);
        EnvironmentOptions environmentOptions = new EnvironmentOptions(
            getStringField(options, "backend"),
            getStringField(options, "world"),
            getStringField(options, "npc_role"),
            getOptionalDoubleField(options, "spawn_x"),
            getOptionalDoubleField(options, "spawn_y"),
            getOptionalDoubleField(options, "spawn_z"),
            getBooleanField(options, "combat_target_active", true),
            getStringField(options, "fidelity_fixture"),
            getStringField(options, "native_combat_item_id"),
            getStringField(options, "native_combat_interaction_id"),
            getStringField(options, "native_combat_interaction_type"),
            getOptionalStringListField(
                options,
                "native_agent_armor_item_ids"
            ),
            getBooleanField(options, "native_navigation_trace", false),
            getOptionalStringListField(
                options,
                "native_combat_ability_interaction_ids"
            ),
            getOptionalStringListField(
                options,
                "native_combat_ability_interaction_types"
            ),
            getStringField(options, "native_guard_interaction_id"),
            getStringField(options, "native_guard_interaction_type"),
            actorEvidenceRequest,
            getBooleanField(options, "native_world_verbs", false),
            getOptionalIntListField(options, "native_world_hotbar_slots"),
            getOptionalStringListField(options, "native_world_hotbar_item_ids"),
            getOptionalIntListField(options, "native_world_hotbar_quantities"),
            getStringField(options, "worldgen_structure"),
            decode(options),
            checkedInt(
                getLongField(
                    options,
                    "native_tick_rate",
                    EnvironmentOptions.DEFAULT_NATIVE_TICK_RATE
                ),
                "native_tick_rate"
            ),
            nativeTimeDilation == null
                ? EnvironmentOptions.DEFAULT_NATIVE_TIME_DILATION
                : nativeTimeDilation.floatValue(),
            ResourceOverrideCodec.decode(options),
            getStringField(options, "combat_target_role")
        );

        environmentManager.destroy(environmentId);
        environmentId = null;
        environmentId = environmentManager.create(
            taskId,
            seed,
            curriculumPhase,
            ticksPerStep,
            maxEpisodeSteps,
            environmentOptions
        );
        sendObservation(output, environmentManager.observe(environmentId));
    }

    private void handleStep(Map<Value, Value> message, DataOutputStream output)
        throws IOException {
        if (environmentId == null) {
            throw new IllegalStateException("No environment active - call reset first");
        }
        Value action = getField(message, "action");
        Value actions = getField(message, "actions");
        if ((action == null) == (actions == null)) {
            throw new IllegalArgumentException(
                "step requires exactly one of action or actions"
            );
        }
        StepResult result = actions == null
            ? environmentManager.step(
                environmentId,
                getMapField(message, "action")
            )
            : environmentManager.step(
                environmentId,
                NativeGroupAction.fromValue(actions)
            );
        sendObservation(output, result);
    }

    private void handleConfig(Map<Value, Value> message, DataOutputStream output)
        throws IOException {
        int requestedTicks = checkedInt(
            getLongField(message, "tick_rate", ticksPerStep),
            "tick_rate"
        );
        int requestedMaxSteps = checkedInt(
            getLongField(message, "max_episode_steps", maxEpisodeSteps),
            "max_episode_steps"
        );
        if (requestedTicks < 1 || requestedTicks > 1000) {
            throw new IllegalArgumentException("tick_rate must be 1..1000");
        }
        if (requestedMaxSteps < 0) {
            throw new IllegalArgumentException("max_episode_steps cannot be negative");
        }
        ticksPerStep = requestedTicks;
        maxEpisodeSteps = requestedMaxSteps;
        sendAck(output);
    }

    private void handleRegionManifest(
        Map<Value, Value> message,
        DataOutputStream output
    )
        throws IOException {
        requireEnvironment();
        Value coreXValue = getField(message, "core_min_chunk_x");
        Value coreZValue = getField(message, "core_min_chunk_z");
        if ((coreXValue == null) != (coreZValue == null)) {
            throw new IllegalArgumentException(
                "core_min_chunk_x and core_min_chunk_z must be supplied together"
            );
        }
        Integer coreX = coreXValue == null
            ? null
            : checkedInt(
                getLongField(message, "core_min_chunk_x", Long.MIN_VALUE),
                "core_min_chunk_x"
            );
        Integer coreZ = coreZValue == null
            ? null
            : checkedInt(
                getLongField(message, "core_min_chunk_z", Long.MIN_VALUE),
                "core_min_chunk_z"
            );
        sendRegionManifest(
            output,
            environmentManager.regionManifest(environmentId, coreX, coreZ)
        );
    }

    private void handleRegionSection(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        int chunkX = checkedInt(
            getLongField(message, "chunk_x", Long.MIN_VALUE),
            "chunk_x"
        );
        int chunkZ = checkedInt(
            getLongField(message, "chunk_z", Long.MIN_VALUE),
            "chunk_z"
        );
        int sectionY = checkedInt(
            getLongField(message, "section_y", Long.MIN_VALUE),
            "section_y"
        );
        sendRegionSection(
            output,
            environmentManager.captureRegionSection(
                environmentId,
                chunkX,
                chunkZ,
                sectionY
            )
        );
    }

    private void handleRegionBlockSemantics(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        int chunkX = checkedInt(
            getLongField(message, "chunk_x", Long.MIN_VALUE),
            "chunk_x"
        );
        int chunkZ = checkedInt(
            getLongField(message, "chunk_z", Long.MIN_VALUE),
            "chunk_z"
        );
        int sectionY = checkedInt(
            getLongField(message, "section_y", Long.MIN_VALUE),
            "section_y"
        );
        sendRegionBlockSemantics(
            output,
            environmentManager.captureRegionBlockSemanticSection(
                environmentId,
                chunkX,
                chunkZ,
                sectionY
            )
        );
    }

    private void handlePerceptionChannels(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        int[] positions = decodeInt32LittleEndianTriples(
            getBinaryField(message, "positions_i32_le_xyz")
        );
        sendPerceptionChannels(
            output,
            environmentManager.capturePerceptionChannels(
                environmentId,
                positions
            )
        );
    }

    private void handleLineOfSightEvidence(DataOutputStream output)
        throws IOException {
        requireEnvironment();
        sendLineOfSightEvidence(
            output,
            environmentManager.captureLineOfSightEvidence(environmentId)
        );
    }

    private void handleMutableBlockEvidence(DataOutputStream output)
        throws IOException {
        requireEnvironment();
        sendMutableBlockEvidence(
            output,
            environmentManager.captureMutableBlockEvidence(environmentId)
        );
    }

    private void handleRegionFluidSemantics(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        int chunkX = checkedInt(
            getLongField(message, "chunk_x", Long.MIN_VALUE),
            "chunk_x"
        );
        int chunkZ = checkedInt(
            getLongField(message, "chunk_z", Long.MIN_VALUE),
            "chunk_z"
        );
        int sectionY = checkedInt(
            getLongField(message, "section_y", Long.MIN_VALUE),
            "section_y"
        );
        sendRegionFluidSemantics(
            output,
            environmentManager.captureRegionFluidSemanticSection(
                environmentId,
                chunkX,
                chunkZ,
                sectionY
            )
        );
    }

    private void handleRegionLightSection(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        int chunkX = checkedInt(
            getLongField(message, "chunk_x", Long.MIN_VALUE),
            "chunk_x"
        );
        int chunkZ = checkedInt(
            getLongField(message, "chunk_z", Long.MIN_VALUE),
            "chunk_z"
        );
        int sectionY = checkedInt(
            getLongField(message, "section_y", Long.MIN_VALUE),
            "section_y"
        );
        sendRegionLightSection(
            output,
            environmentManager.captureRegionLightSection(
                environmentId,
                chunkX,
                chunkZ,
                sectionY
            )
        );
    }

    private void handleMutableBlockCells(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        int[] positions = decodeInt32LittleEndianTriples(
            getBinaryField(message, "positions_i32_le_xyz")
        );
        String expectedWorldEpoch = getOptionalStringField(
            message,
            "expected_world_epoch"
        );
        sendMutableBlockCells(
            output,
            expectedWorldEpoch == null
                ? environmentManager.captureMutableBlockCells(
                    environmentId,
                    positions
                )
                : environmentManager.captureMutableBlockCells(
                    environmentId,
                    positions,
                    expectedWorldEpoch
                )
        );
    }

    private void handleDropProgramEvidence(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        String blockAssetId = getStringField(
            message,
            "block_asset_id"
        );
        String route = getStringField(message, "route");
        int sampleCount = checkedInt(
            getLongField(message, "sample_count", Long.MIN_VALUE),
            "sample_count"
        );
        sendDropProgramEvidence(
            output,
            environmentManager.captureDropProgramEvidence(
                environmentId,
                blockAssetId,
                route,
                sampleCount
            )
        );
    }

    private void handleCraftingCatalogEvidence(DataOutputStream output)
        throws IOException {
        requireEnvironment();
        sendCraftingCatalogEvidence(
            output,
            environmentManager.captureCraftingCatalogEvidence(environmentId)
        );
    }

    private void handleItemInteractionEvidence(
        Map<Value, Value> message,
        DataOutputStream output
    )
        throws IOException {
        requireEnvironment();
        int equippedSlot = checkedInt(
            getLongField(message, "equipped_slot", 0),
            "equipped_slot"
        );
        sendItemInteractionEvidence(
            output,
            environmentManager.captureItemInteractionEvidence(
                environmentId,
                equippedSlot
            )
        );
    }

    private void handleDoorTransitionEvidence(DataOutputStream output)
        throws IOException {
        requireEnvironment();
        sendDoorTransitionEvidence(
            output,
            environmentManager.captureDoorTransitionEvidence(environmentId)
        );
    }

    private void handleTraversalProbe(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        double[] positions = decodeFloat64LittleEndian(
            getBinaryField(message, "positions_f64_le_xyz"),
            3,
            "positions_f64_le_xyz"
        );
        double[] upwardLimits = decodeFloat64LittleEndian(
            getBinaryField(message, "upward_limits_f64_le"),
            1,
            "upward_limits_f64_le"
        );
        if (positions.length / 3 != upwardLimits.length) {
            throw new IllegalArgumentException(
                "Traversal positions and upward limits must have equal sample counts"
            );
        }
        sendTraversalProbe(
            output,
            environmentManager.captureTraversalProbe(
                environmentId,
                positions,
                upwardLimits
            )
        );
    }

    private void handleTraversalEdgeProbe(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        double[] starts = decodeFloat64LittleEndian(
            getBinaryField(message, "start_positions_f64_le_xyz"),
            3,
            "start_positions_f64_le_xyz"
        );
        double[] targets = decodeFloat64LittleEndian(
            getBinaryField(message, "target_positions_f64_le_xyz"),
            3,
            "target_positions_f64_le_xyz"
        );
        double[] horizontalTolerances = decodeFloat64LittleEndian(
            getBinaryField(
                message,
                "horizontal_arrival_tolerances_f64_le"
            ),
            1,
            "horizontal_arrival_tolerances_f64_le"
        );
        double[] verticalTolerances = decodeFloat64LittleEndian(
            getBinaryField(
                message,
                "vertical_arrival_tolerances_f64_le"
            ),
            1,
            "vertical_arrival_tolerances_f64_le"
        );
        if (
            starts.length != targets.length
                || starts.length / 3 != horizontalTolerances.length
                || horizontalTolerances.length != verticalTolerances.length
        ) {
            throw new IllegalArgumentException(
                "Traversal edge starts, targets, and tolerances must agree"
            );
        }
        sendTraversalEdgeProbe(
            output,
            environmentManager.captureTraversalEdgeProbe(
                environmentId,
                starts,
                targets,
                horizontalTolerances,
                verticalTolerances
            )
        );
    }

    private void handleNavigationPathProbe(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        double[] starts = decodeFloat64LittleEndian(
            getBinaryField(message, "start_positions_f64_le_xyz"),
            3,
            "start_positions_f64_le_xyz"
        );
        double[] targets = decodeFloat64LittleEndian(
            getBinaryField(message, "target_positions_f64_le_xyz"),
            3,
            "target_positions_f64_le_xyz"
        );
        if (starts.length != targets.length) {
            throw new IllegalArgumentException(
                "Navigation path starts and targets must agree"
            );
        }
        sendNavigationPathProbe(
            output,
            environmentManager.captureNavigationPathProbe(
                environmentId,
                starts,
                targets,
                checkedInt(
                    getLongField(
                        message,
                        "maximum_path_length",
                        Long.MIN_VALUE
                    ),
                    "maximum_path_length"
                ),
                checkedInt(
                    getLongField(
                        message,
                        "open_nodes_limit",
                        Long.MIN_VALUE
                    ),
                    "open_nodes_limit"
                ),
                checkedInt(
                    getLongField(
                        message,
                        "total_nodes_limit",
                        Long.MIN_VALUE
                    ),
                    "total_nodes_limit"
                ),
                checkedInt(
                    getLongField(
                        message,
                        "nodes_per_iteration",
                        Long.MIN_VALUE
                    ),
                    "nodes_per_iteration"
                )
            )
        );
    }

    private void handleNavigationSuccessorProbe(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        double[] starts = decodeFloat64LittleEndian(
            getBinaryField(message, "start_positions_f64_le_xyz"),
            3,
            "start_positions_f64_le_xyz"
        );
        byte[] directions = getBinaryField(message, "direction_indices_i8");
        if (starts.length / 3 != directions.length) {
            throw new IllegalArgumentException(
                "Navigation successor starts and directions must agree"
            );
        }
        sendNavigationSuccessorProbe(
            output,
            environmentManager.captureNavigationSuccessorProbe(
                environmentId,
                starts,
                directions
            )
        );
    }

    private void handlePrivilegedEntitySnapshot(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        double[] bounds = decodeFloat64LittleEndian(
            getBinaryField(message, "bounds_f64_le_min_max_xyz"),
            6,
            "bounds_f64_le_min_max_xyz"
        );
        if (bounds.length != 6) {
            throw new IllegalArgumentException(
                "bounds_f64_le_min_max_xyz must contain one AABB"
            );
        }
        int capacity = checkedInt(
            getLongField(message, "capacity", Long.MIN_VALUE),
            "capacity"
        );
        PrivilegedEntityQuery query = new PrivilegedEntityQuery(
            getStringField(message, "component_filter"),
            bounds,
            capacity
        );
        sendPrivilegedEntitySnapshot(
            output,
            environmentManager.capturePrivilegedEntitySnapshot(
                environmentId,
                query
            )
        );
    }

    private void handlePrivilegedNpcSnapshot(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        if (
            !PrivilegedNpcSnapshot.COMPONENT_FILTER.equals(
                getStringField(message, "component_filter")
            )
        ) {
            throw new IllegalArgumentException(
                "Unsupported privileged NPC component filter"
            );
        }
        double[] bounds = decodeFloat64LittleEndian(
            getBinaryField(message, "bounds_f64_le_min_max_xyz"),
            6,
            "bounds_f64_le_min_max_xyz"
        );
        if (bounds.length != 6) {
            throw new IllegalArgumentException(
                "bounds_f64_le_min_max_xyz must contain one AABB"
            );
        }
        int capacity = checkedInt(
            getLongField(message, "capacity", Long.MIN_VALUE),
            "capacity"
        );
        PrivilegedEntityQuery query = new PrivilegedEntityQuery(
            PrivilegedEntityQuery.COMPONENT_FILTER,
            bounds,
            capacity
        );
        sendPrivilegedNpcSnapshot(
            output,
            environmentManager.capturePrivilegedNpcSnapshot(
                environmentId,
                query
            )
        );
    }

    private void handlePrivilegedNpcSnapshotByUuid(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        double[] bounds = decodeFloat64LittleEndian(
            getBinaryField(message, "bounds_f64_le_min_max_xyz"),
            6,
            "bounds_f64_le_min_max_xyz"
        );
        if (bounds.length != 6) {
            throw new IllegalArgumentException(
                "bounds_f64_le_min_max_xyz must contain one AABB"
            );
        }
        PrivilegedNpcUuidQuery query = new PrivilegedNpcUuidQuery(
            getStringField(message, "component_filter"),
            uuidField(message, "npc_uuid_bytes"),
            bounds
        );
        sendPrivilegedNpcSnapshot(
            output,
            environmentManager.capturePrivilegedNpcSnapshotByUuid(
                environmentId,
                query
            )
        );
    }

    private void handleNpcImitationTrace(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        String command = getStringField(message, "command");
        switch (command) {
            case "start" -> sendNpcTrace(
                output,
                environmentManager.startNpcTrace(
                    environmentId,
                    uuidField(message, "npc_uuid_bytes"),
                    getOptionalStringField(message, "expected_role"),
                    checkedInt(
                        getLongField(
                            message,
                            "capacity",
                            com.hytalerlbridge.imitation.NpcTraceBatch.DEFAULT_CAPACITY
                        ),
                        "capacity"
                    )
                )
            );
            case "poll", "stop" -> sendNpcTrace(
                output,
                environmentManager.drainNpcTrace(
                    environmentId,
                    uuidField(message, "trace_uuid_bytes"),
                    checkedInt(
                        getLongField(
                            message,
                            "max_frames",
                            com.hytalerlbridge.imitation.NpcTraceBatch.MAX_DRAIN
                        ),
                        "max_frames"
                    ),
                    command.equals("stop")
                )
            );
            default -> throw new IllegalArgumentException(
                "NPC trace command must be start, poll, or stop"
            );
        }
    }

    private static UUID uuidField(Map<Value, Value> message, String name) {
        byte[] bytes = getBinaryField(message, name);
        if (bytes.length != 16) {
            throw new IllegalArgumentException(name + " must contain one RFC-4122 UUID");
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
        return new UUID(buffer.getLong(), buffer.getLong());
    }

    private void handleWorldgenStructureMarkers(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        if (
            !WorldgenStructureMarkerQuery.COMPONENT_FILTER.equals(
                getStringField(message, "component_filter")
            )
        ) {
            throw new IllegalArgumentException(
                "Unsupported WorldGen structure marker component filter"
            );
        }
        double[] bounds = decodeFloat64LittleEndian(
            getBinaryField(message, "bounds_f64_le_min_max_xyz"),
            6,
            "bounds_f64_le_min_max_xyz"
        );
        if (bounds.length != 6) {
            throw new IllegalArgumentException(
                "bounds_f64_le_min_max_xyz must contain one AABB"
            );
        }
        int capacity = checkedInt(
            getLongField(message, "capacity", Long.MIN_VALUE),
            "capacity"
        );
        WorldgenStructureMarkerQuery query =
            new WorldgenStructureMarkerQuery(
                WorldgenStructureMarkerQuery.COMPONENT_FILTER,
                bounds,
                capacity,
                getOptionalStringListField(message, "marker_asset_ids")
            );
        sendWorldgenStructureMarkers(
            output,
            environmentManager.captureWorldgenStructureMarkers(
                environmentId,
                query
            )
        );
    }

    private void handleExplosionCandidateProbe(
        Map<Value, Value> message,
        DataOutputStream output
    ) throws IOException {
        requireEnvironment();
        double[] origin = decodeFloat64LittleEndian(
            getBinaryField(message, "origin_f64_le_xyz"),
            3,
            "origin_f64_le_xyz"
        );
        if (origin.length != 3) {
            throw new IllegalArgumentException(
                "origin_f64_le_xyz must contain one XYZ row"
            );
        }
        int blockDamageRadius = checkedInt(
            getLongField(message, "block_damage_radius", Long.MIN_VALUE),
            "block_damage_radius"
        );
        Double entityDamageRadius = getOptionalDoubleField(
            message,
            "entity_damage_radius"
        );
        if (entityDamageRadius == null || !Double.isFinite(entityDamageRadius)) {
            throw new IllegalArgumentException(
                "entity_damage_radius must be finite"
            );
        }
        int capacity = checkedInt(
            getLongField(message, "capacity", Long.MIN_VALUE),
            "capacity"
        );
        sendExplosionCandidateProbe(
            output,
            environmentManager.captureExplosionCandidateProbe(
                environmentId,
                origin,
                blockDamageRadius,
                entityDamageRadius.floatValue(),
                getBooleanField(message, "ignore_controlled_actor", true),
                capacity
            )
        );
    }

    private void requireEnvironment() {
        if (environmentId == null) {
            throw new IllegalStateException(
                "No environment active - call reset first"
            );
        }
    }

    private void cleanup() {
        environmentManager.destroy(environmentId);
        environmentId = null;
        try {
            socket.close();
        } catch (IOException ignored) {
            // Nothing else to release.
        }
    }

}
