package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.NativeTraversalProbe;
import com.hytalerlbridge.worldgen.NativeTraversalEdgeProbe;
import com.hytalerlbridge.worldgen.NativeNavigationPathProbe;
import com.hytalerlbridge.worldgen.NativeNavigationSuccessorProbe;

/** Native-only stationary actor collision/clearance probe capability. */
public interface NativeTraversalProbeSource {

    NativeTraversalProbe captureTraversalProbe(
        double[] positions,
        double[] upwardLimits
    );

    NativeTraversalEdgeProbe captureTraversalEdgeProbe(
        double[] startPositions,
        double[] targetPositions,
        double[] horizontalArrivalTolerances,
        double[] verticalArrivalTolerances
    );

    NativeNavigationPathProbe captureNavigationPathProbe(
        double[] startPositions,
        double[] targetPositions,
        int maximumPathLength,
        int openNodesLimit,
        int totalNodesLimit,
        int nodesPerIteration
    );

    NativeNavigationSuccessorProbe captureNavigationSuccessorProbe(
        double[] startPositions,
        byte[] directionIndices
    );
}
