package com.hytalerlbridge.network.codec;

import com.hytalerlbridge.worldgen.NativeTraversalProbe;
import com.hytalerlbridge.worldgen.NativeTraversalEdgeProbe;
import com.hytalerlbridge.worldgen.NativeNavigationPathProbe;
import com.hytalerlbridge.worldgen.NativeNavigationSuccessorProbe;
import java.io.DataOutputStream;
import java.io.IOException;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeInt32LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeFloat32LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.encodeFloat64LittleEndian;
import static com.hytalerlbridge.network.wire.BinaryCodec.packBinary;
import static com.hytalerlbridge.network.wire.FrameWriter.sendFrame;

/**
 * Extracted verbatim from {@code ClientHandler}.
 */
public final class ProbeCodec {

    private ProbeCodec() {}

    public static void sendTraversalProbe(
        DataOutputStream output,
        NativeTraversalProbe probe
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(14);
            packer.packString("type");
            packer.packString("traversal_probe");
            packer.packString("schema");
            packer.packString(NativeTraversalProbe.SCHEMA);
            packer.packString("version");
            packer.packInt(NativeTraversalProbe.VERSION);
            packer.packString("server_version");
            packer.packString(probe.serverVersion());
            packer.packString("world");
            packer.packString(probe.worldName());
            packer.packString("worldgen_provider");
            packer.packString(probe.worldgenProvider());
            packer.packString("worldgen_version");
            packer.packString(probe.worldgenVersion());
            packer.packString("seed");
            packer.packLong(probe.seed());
            packer.packString("sample_count");
            packer.packInt(probe.sampleCount());
            packBinary(
                packer,
                "positions_f64_le_xyz",
                encodeFloat64LittleEndian(probe.positions())
            );
            packBinary(
                packer,
                "upward_limits_f64_le",
                encodeFloat64LittleEndian(probe.upwardLimits())
            );
            packBinary(
                packer,
                "actor_bounds_f64_le",
                encodeFloat64LittleEndian(probe.actorBounds())
            );
            packBinary(
                packer,
                "validation_codes_i8",
                probe.validationCodes()
            );
            packBinary(
                packer,
                "upward_collision_distances_f64_le",
                encodeFloat64LittleEndian(
                    probe.upwardCollisionDistances()
                )
            );
        });
    }


    public static void sendTraversalEdgeProbe(
        DataOutputStream output,
        NativeTraversalEdgeProbe probe
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(21);
            packer.packString("type");
            packer.packString("traversal_edge_probe");
            packer.packString("schema");
            packer.packString(NativeTraversalEdgeProbe.SCHEMA);
            packer.packString("version");
            packer.packInt(NativeTraversalEdgeProbe.VERSION);
            packer.packString("server_version");
            packer.packString(probe.serverVersion());
            packer.packString("world");
            packer.packString(probe.worldName());
            packer.packString("worldgen_provider");
            packer.packString(probe.worldgenProvider());
            packer.packString("worldgen_version");
            packer.packString(probe.worldgenVersion());
            packer.packString("seed");
            packer.packLong(probe.seed());
            packer.packString("sample_count");
            packer.packInt(probe.sampleCount());
            packBinary(
                packer,
                "start_positions_f64_le_xyz",
                encodeFloat64LittleEndian(probe.startPositions())
            );
            packBinary(
                packer,
                "target_positions_f64_le_xyz",
                encodeFloat64LittleEndian(probe.targetPositions())
            );
            packBinary(
                packer,
                "horizontal_arrival_tolerances_f64_le",
                encodeFloat64LittleEndian(
                    probe.horizontalArrivalTolerances()
                )
            );
            packBinary(
                packer,
                "vertical_arrival_tolerances_f64_le",
                encodeFloat64LittleEndian(
                    probe.verticalArrivalTolerances()
                )
            );
            packBinary(
                packer,
                "actor_bounds_f64_le",
                encodeFloat64LittleEndian(probe.actorBounds())
            );
            packBinary(
                packer,
                "direction_component_selector_f64_le_xyz",
                encodeFloat64LittleEndian(
                    probe.directionComponentSelector()
                )
            );
            packer.packString("maximum_climb_height");
            packer.packDouble(probe.maximumClimbHeight());
            packer.packString("maximum_drop_height");
            packer.packDouble(probe.maximumDropHeight());
            packBinary(packer, "reachable_u8", probe.reachable());
            packBinary(packer, "edge_blocked_u8", probe.edgeBlocked());
            packBinary(
                packer,
                "final_positions_f64_le_xyz",
                encodeFloat64LittleEndian(probe.finalPositions())
            );
            packBinary(
                packer,
                "travelled_distances_f64_le",
                encodeFloat64LittleEndian(probe.travelledDistances())
            );
        });
    }


    public static void sendNavigationPathProbe(
        DataOutputStream output,
        NativeNavigationPathProbe probe
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(24);
            packer.packString("type");
            packer.packString("navigation_path_probe");
            packer.packString("schema");
            packer.packString(NativeNavigationPathProbe.SCHEMA);
            packer.packString("version");
            packer.packInt(NativeNavigationPathProbe.VERSION);
            packer.packString("server_version");
            packer.packString(probe.serverVersion());
            packer.packString("world");
            packer.packString(probe.worldName());
            packer.packString("worldgen_provider");
            packer.packString(probe.worldgenProvider());
            packer.packString("worldgen_version");
            packer.packString(probe.worldgenVersion());
            packer.packString("seed");
            packer.packLong(probe.seed());
            packer.packString("sample_count");
            packer.packInt(probe.sampleCount());
            packBinary(
                packer,
                "start_positions_f64_le_xyz",
                encodeFloat64LittleEndian(probe.startPositions())
            );
            packBinary(
                packer,
                "target_positions_f64_le_xyz",
                encodeFloat64LittleEndian(probe.targetPositions())
            );
            packBinary(
                packer,
                "actor_bounds_f64_le",
                encodeFloat64LittleEndian(probe.actorBounds())
            );
            packBinary(
                packer,
                "direction_component_selector_f64_le_xyz",
                encodeFloat64LittleEndian(
                    probe.directionComponentSelector()
                )
            );
            packer.packString("maximum_path_length");
            packer.packInt(probe.maximumPathLength());
            packer.packString("open_nodes_limit");
            packer.packInt(probe.openNodesLimit());
            packer.packString("total_nodes_limit");
            packer.packInt(probe.totalNodesLimit());
            packer.packString("nodes_per_iteration");
            packer.packInt(probe.nodesPerIteration());
            packBinary(packer, "progress_i8", probe.progress());
            packBinary(
                packer,
                "iterations_i32_le",
                encodeInt32LittleEndian(probe.iterations())
            );
            packBinary(
                packer,
                "visited_counts_i32_le",
                encodeInt32LittleEndian(probe.visitedCounts())
            );
            packBinary(
                packer,
                "open_counts_i32_le",
                encodeInt32LittleEndian(probe.openCounts())
            );
            packBinary(
                packer,
                "path_node_counts_i32_le",
                encodeInt32LittleEndian(probe.pathNodeCounts())
            );
            packBinary(
                packer,
                "path_positions_f64_le_xyz",
                encodeFloat64LittleEndian(probe.pathPositions())
            );
            packBinary(
                packer,
                "path_travel_costs_f32_le",
                encodeFloat32LittleEndian(probe.pathTravelCosts())
            );
        });
    }


    public static void sendNavigationSuccessorProbe(
        DataOutputStream output,
        NativeNavigationSuccessorProbe probe
    ) throws IOException {
        sendFrame(output, packer -> {
            packer.packMapHeader(22);
            packer.packString("type");
            packer.packString("navigation_successor_probe");
            packer.packString("schema");
            packer.packString(NativeNavigationSuccessorProbe.SCHEMA);
            packer.packString("version");
            packer.packInt(NativeNavigationSuccessorProbe.VERSION);
            packer.packString("server_version");
            packer.packString(probe.serverVersion());
            packer.packString("world");
            packer.packString(probe.worldName());
            packer.packString("worldgen_provider");
            packer.packString(probe.worldgenProvider());
            packer.packString("worldgen_version");
            packer.packString(probe.worldgenVersion());
            packer.packString("seed");
            packer.packLong(probe.seed());
            packer.packString("sample_count");
            packer.packInt(probe.sampleCount());
            packBinary(
                packer,
                "start_positions_f64_le_xyz",
                encodeFloat64LittleEndian(probe.startPositions())
            );
            packBinary(
                packer,
                "direction_indices_i8",
                probe.directionIndices()
            );
            packBinary(
                packer,
                "directions_f64_le_xyz",
                encodeFloat64LittleEndian(probe.directions())
            );
            packBinary(
                packer,
                "actor_bounds_f64_le",
                encodeFloat64LittleEndian(probe.actorBounds())
            );
            packBinary(
                packer,
                "direction_component_selector_f64_le_xyz",
                encodeFloat64LittleEndian(
                    probe.directionComponentSelector()
                )
            );
            packBinary(
                packer,
                "direction_distances_f64_le",
                encodeFloat64LittleEndian(probe.directionDistances())
            );
            packBinary(
                packer,
                "travelled_distances_f64_le",
                encodeFloat64LittleEndian(probe.travelledDistances())
            );
            packBinary(
                packer,
                "reached_half_step_u8",
                probe.reachedHalfStep()
            );
            packBinary(
                packer,
                "reached_full_step_u8",
                probe.reachedFullStep()
            );
            packBinary(
                packer,
                "valid_positions_u8",
                probe.validPositions()
            );
            packBinary(packer, "edge_blocked_u8", probe.edgeBlocked());
            packBinary(
                packer,
                "half_step_positions_f64_le_xyz",
                encodeFloat64LittleEndian(probe.halfStepPositions())
            );
            packBinary(
                packer,
                "successor_positions_f64_le_xyz",
                encodeFloat64LittleEndian(probe.successorPositions())
            );
        });
    }
}
