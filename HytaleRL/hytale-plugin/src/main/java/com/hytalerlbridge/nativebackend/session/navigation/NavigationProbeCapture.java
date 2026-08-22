package com.hytalerlbridge.nativebackend.session.navigation;

import com.hypixel.hytale.component.ComponentAccessor;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.shape.Box;
import com.hypixel.hytale.server.core.modules.collision.BlockCollisionData;
import com.hypixel.hytale.server.core.modules.collision.CollisionModule;
import com.hypixel.hytale.server.core.modules.collision.CollisionResult;
import com.hypixel.hytale.server.core.modules.entity.component.BoundingBox;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.movement.controllers.MotionController;
import com.hypixel.hytale.server.npc.movement.controllers.MotionControllerWalk;
import com.hypixel.hytale.server.npc.movement.controllers.ProbeMoveData;
import com.hypixel.hytale.server.npc.navigation.AStarBase;
import com.hypixel.hytale.server.npc.navigation.AStarEvaluator;
import com.hypixel.hytale.server.npc.navigation.AStarNode;
import com.hypixel.hytale.server.npc.navigation.AStarNodePoolProviderSimple;
import com.hypixel.hytale.server.npc.navigation.AStarWithTarget;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.util.NPCPhysicsMath;
import com.hytalerlbridge.worldgen.NativeNavigationPathProbe;
import com.hytalerlbridge.worldgen.NativeNavigationSuccessorProbe;
import com.hytalerlbridge.worldgen.NativeTraversalEdgeProbe;
import com.hytalerlbridge.worldgen.NativeTraversalProbe;
import org.joml.Vector3d;
import static com.hytalerlbridge.nativebackend.support.NativeKeys.implementationVersion;
import static com.hytalerlbridge.nativebackend.support.SupportMath.navigationSearchDirections;

/**
 * World-thread builders for the four native navigation probes.
 *
 * <p>Extracted from {@code NativeEnvironmentSession}, which keeps the public
 * {@code capture*} entry points: those validate their arguments on the caller's
 * thread and marshal onto the world thread via {@code runOnWorld}. Everything
 * here runs ON the world thread and touches live native state, so none of it is
 * safe to call directly.
 *
 * <p>Three of the four builders resolve the actor's {@link MotionControllerWalk}
 * and collision AABB before doing anything else, and throw
 * {@link IllegalStateException} when either is missing - a probe against an
 * actor without a walk controller would silently report "unreachable" for every
 * sample rather than failing, so the checks are load-bearing.
 */
public final class NavigationProbeCapture {

    private static final String NATIVE_SERVER_VERSION = implementationVersion();

    private NavigationProbeCapture() {}

    public static NativeTraversalProbe buildTraversalProbe(
        NavigationProbeContext context,
        double[] positions,
        double[] upwardLimits
    ) {
        Store<EntityStore> store = context.world().getEntityStore().getStore();
        Ref<EntityStore> agent = context.agentRef();
        BoundingBox component = agent == null
            ? null
            : store.getComponent(agent, BoundingBox.getComponentType());
        Box collider = component == null ? null : component.getBoundingBox();
        if (collider == null) {
            throw new IllegalStateException(
                "Native traversal probe requires the actor collision AABB"
            );
        }
        double[] bounds = {
            collider.min.x,
            collider.min.y,
            collider.min.z,
            collider.max.x,
            collider.max.y,
            collider.max.z
        };
        int samples = upwardLimits.length;
        byte[] validation = new byte[samples];
        double[] upwardDistances = new double[samples];
        for (int sample = 0; sample < samples; sample++) {
            int offset = sample * 3;
            Vector3d position = new Vector3d(
                positions[offset],
                positions[offset + 1],
                positions[offset + 2]
            );
            int code = CollisionModule.get().validatePosition(
                context.world(),
                collider,
                position,
                new CollisionResult()
            );
            validation[sample] = (byte) code;
            double limit = upwardLimits[sample];
            if (code < 0 || limit == 0.0) continue;

            CollisionResult result = new CollisionResult(false, false);
            CollisionModule.findCollisions(
                collider,
                position,
                new Vector3d(0.0, limit, 0.0),
                result,
                store
            );
            BlockCollisionData collision = result.getFirstBlockCollision();
            upwardDistances[sample] = collision == null
                ? limit
                : limit * Math.max(0.0, Math.min(1.0, collision.collisionStart));
        }
        return new NativeTraversalProbe(
            NATIVE_SERVER_VERSION,
            context.worldName(),
            context.worldgenProvider(),
            context.worldgenVersion(),
            context.seed(),
            positions,
            upwardLimits,
            bounds,
            validation,
            upwardDistances
        );
    }

    public static NativeTraversalEdgeProbe buildTraversalEdgeProbe(
        NavigationProbeContext context,
        double[] startPositions,
        double[] targetPositions,
        double[] horizontalArrivalTolerances,
        double[] verticalArrivalTolerances
    ) {
        Store<EntityStore> store = context.world().getEntityStore().getStore();
        Ref<EntityStore> agent = context.agentRef();
        NPCEntity npc = agent == null
            ? null
            : store.getComponent(agent, NPCEntity.getComponentType());
        Role role = npc == null ? null : npc.getRole();
        MotionController controller = role == null
            ? null
            : role.getActiveMotionController();
        if (!(controller instanceof MotionControllerWalk walk)) {
            throw new IllegalStateException(
                "Traversal edge probe requires MotionControllerWalk"
            );
        }
        BoundingBox component = store.getComponent(
            agent,
            BoundingBox.getComponentType()
        );
        Box collider = component == null ? null : component.getBoundingBox();
        if (collider == null) {
            throw new IllegalStateException(
                "Native traversal edge probe requires the actor collision AABB"
            );
        }
        double[] bounds = {
            collider.min.x,
            collider.min.y,
            collider.min.z,
            collider.max.x,
            collider.max.y,
            collider.max.z
        };
        Vector3d selector = walk.getComponentSelector();
        double[] selectorValues = {selector.x, selector.y, selector.z};
        int samples = horizontalArrivalTolerances.length;
        byte[] reachable = new byte[samples];
        byte[] edgeBlocked = new byte[samples];
        double[] finalPositions = new double[samples * 3];
        double[] travelledDistances = new double[samples];
        for (int sample = 0; sample < samples; sample++) {
            int offset = sample * 3;
            Vector3d start = new Vector3d(
                startPositions[offset],
                startPositions[offset + 1],
                startPositions[offset + 2]
            );
            Vector3d target = new Vector3d(
                targetPositions[offset],
                targetPositions[offset + 1],
                targetPositions[offset + 2]
            );
            ProbeMoveData probe = new ProbeMoveData()
                .setPosition(start)
                .setTargetPosition(target);
            travelledDistances[sample] = walk.probeMove(agent, probe, store);
            double horizontalTolerance =
                horizontalArrivalTolerances[sample];
            double verticalTolerance = verticalArrivalTolerances[sample];
            boolean horizontalReached =
                walk.waypointDistanceSquared(target, probe.probePosition)
                    <= horizontalTolerance * horizontalTolerance;
            double verticalDifference = NPCPhysicsMath.getProjectedDifference(
                target,
                probe.probePosition,
                selector
            );
            reachable[sample] = (byte) (
                horizontalReached
                    && Math.abs(verticalDifference) <= verticalTolerance
                    ? 1
                    : 0
            );
            edgeBlocked[sample] = (byte) (probe.edgeBlocked ? 1 : 0);
            finalPositions[offset] = probe.probePosition.x;
            finalPositions[offset + 1] = probe.probePosition.y;
            finalPositions[offset + 2] = probe.probePosition.z;
        }
        return new NativeTraversalEdgeProbe(
            NATIVE_SERVER_VERSION,
            context.worldName(),
            context.worldgenProvider(),
            context.worldgenVersion(),
            context.seed(),
            startPositions,
            targetPositions,
            horizontalArrivalTolerances,
            verticalArrivalTolerances,
            bounds,
            selectorValues,
            walk.getMaxClimbHeight(),
            walk.getMaxDropHeight(),
            reachable,
            edgeBlocked,
            finalPositions,
            travelledDistances
        );
    }

    public static NativeNavigationSuccessorProbe buildNavigationSuccessorProbe(
        NavigationProbeContext context,
        double[] startPositions,
        byte[] directionIndices
    ) {
        Store<EntityStore> store = context.world().getEntityStore().getStore();
        Ref<EntityStore> agent = context.agentRef();
        NPCEntity npc = agent == null
            ? null
            : store.getComponent(agent, NPCEntity.getComponentType());
        Role role = npc == null ? null : npc.getRole();
        MotionController controller = role == null
            ? null
            : role.getActiveMotionController();
        if (!(controller instanceof MotionControllerWalk walk)) {
            throw new IllegalStateException(
                "Navigation successor probe requires MotionControllerWalk"
            );
        }
        BoundingBox component = store.getComponent(
            agent,
            BoundingBox.getComponentType()
        );
        Box collider = component == null ? null : component.getBoundingBox();
        if (collider == null) {
            throw new IllegalStateException(
                "Navigation successor probe requires the actor collision AABB"
            );
        }
        Vector3d[] searchDirections = navigationSearchDirections(walk);
        int samples = directionIndices.length;
        double[] directions = new double[samples * 3];
        double[] directionDistances = new double[samples];
        double[] travelledDistances = new double[samples];
        byte[] reachedHalfStep = new byte[samples];
        byte[] reachedFullStep = new byte[samples];
        byte[] validPositions = new byte[samples];
        byte[] edgeBlocked = new byte[samples];
        double[] halfStepPositions = new double[samples * 3];
        double[] successorPositions = new double[samples * 3];
        for (int sample = 0; sample < samples; sample++) {
            int directionIndex = directionIndices[sample];
            if (
                directionIndex < 0
                    || directionIndex >= searchDirections.length
            ) {
                throw new IllegalArgumentException(
                    "Navigation direction index is unavailable"
                );
            }
            int offset = sample * 3;
            Vector3d start = new Vector3d(
                startPositions[offset],
                startPositions[offset + 1],
                startPositions[offset + 2]
            );
            Vector3d direction = searchDirections[directionIndex];
            double directionDistance = direction.length();
            ProbeMoveData probe = new ProbeMoveData()
                .setPosition(start)
                .setDirection(direction);
            probe.setSaveSegments(true);
            double travelled = walk.probeMove(agent, probe, store);
            boolean half = travelled
                >= directionDistance * 0.49999995D;
            boolean full = travelled
                >= directionDistance * 0.9999999D;
            Vector3d halfPosition = new Vector3d(start);
            Vector3d successor = new Vector3d(start);
            boolean valid = false;
            if (half) {
                probe.computePosition(
                    directionDistance * 0.5D,
                    halfPosition
                );
                successor.set(full ? probe.probePosition : halfPosition);
                valid = walk.isValidPosition(successor, store);
            }
            directions[offset] = direction.x;
            directions[offset + 1] = direction.y;
            directions[offset + 2] = direction.z;
            directionDistances[sample] = directionDistance;
            travelledDistances[sample] = travelled;
            reachedHalfStep[sample] = (byte) (half ? 1 : 0);
            reachedFullStep[sample] = (byte) (full ? 1 : 0);
            validPositions[sample] = (byte) (valid ? 1 : 0);
            edgeBlocked[sample] = (byte) (probe.edgeBlocked ? 1 : 0);
            halfStepPositions[offset] = halfPosition.x;
            halfStepPositions[offset + 1] = halfPosition.y;
            halfStepPositions[offset + 2] = halfPosition.z;
            successorPositions[offset] = successor.x;
            successorPositions[offset + 1] = successor.y;
            successorPositions[offset + 2] = successor.z;
        }
        Box bounds = collider;
        Vector3d selector = walk.getComponentSelector();
        return new NativeNavigationSuccessorProbe(
            NATIVE_SERVER_VERSION,
            context.worldName(),
            context.worldgenProvider(),
            context.worldgenVersion(),
            context.seed(),
            startPositions,
            directionIndices,
            directions,
            new double[] {
                bounds.min.x,
                bounds.min.y,
                bounds.min.z,
                bounds.max.x,
                bounds.max.y,
                bounds.max.z
            },
            new double[] {selector.x, selector.y, selector.z},
            directionDistances,
            travelledDistances,
            reachedHalfStep,
            reachedFullStep,
            validPositions,
            edgeBlocked,
            halfStepPositions,
            successorPositions
        );
    }

    public static NativeNavigationPathProbe buildNavigationPathProbe(
        NavigationProbeContext context,
        double[] startPositions,
        double[] targetPositions,
        int maximumPathLength,
        int openNodesLimit,
        int totalNodesLimit,
        int nodesPerIteration
    ) {
        Store<EntityStore> store = context.world().getEntityStore().getStore();
        Ref<EntityStore> agent = context.agentRef();
        NPCEntity npc = agent == null
            ? null
            : store.getComponent(agent, NPCEntity.getComponentType());
        Role role = npc == null ? null : npc.getRole();
        MotionController controller = role == null
            ? null
            : role.getActiveMotionController();
        if (!(controller instanceof MotionControllerWalk walk)) {
            throw new IllegalStateException(
                "Navigation path probe requires MotionControllerWalk"
            );
        }
        BoundingBox component = store.getComponent(
            agent,
            BoundingBox.getComponentType()
        );
        Box collider = component == null ? null : component.getBoundingBox();
        if (collider == null) {
            throw new IllegalStateException(
                "Navigation path probe requires the actor collision AABB"
            );
        }
        AStarNodePoolProviderSimple poolProvider = store.getResource(
            AStarNodePoolProviderSimple.getResourceType()
        );
        if (poolProvider == null) {
            throw new IllegalStateException(
                "Navigation path probe requires the native AStar node pool"
            );
        }
        double[] bounds = {
            collider.min.x,
            collider.min.y,
            collider.min.z,
            collider.max.x,
            collider.max.y,
            collider.max.z
        };
        Vector3d selector = walk.getComponentSelector();
        double[] selectorValues = {selector.x, selector.y, selector.z};
        int samples = startPositions.length / 3;
        byte[] progress = new byte[samples];
        int[] iterations = new int[samples];
        int[] visitedCounts = new int[samples];
        int[] openCounts = new int[samples];
        int[] pathNodeCounts = new int[samples];
        double[] pathPositions = new double[
            samples * maximumPathLength * 3
        ];
        float[] pathTravelCosts = new float[
            samples * maximumPathLength
        ];
        AStarEvaluator evaluator = new AStarEvaluator() {
            @Override
            public boolean isGoalReached(
                Ref<EntityStore> ref,
                AStarBase pathfinder,
                AStarNode node,
                MotionController motionController,
                ComponentAccessor<EntityStore> componentAccessor
            ) {
                return node.getPositionIndex()
                    == ((AStarWithTarget) pathfinder).getTargetPositionIndex();
            }

            @Override
            public float estimateToGoal(
                AStarBase pathfinder,
                Vector3d fromPosition,
                MotionController motionController
            ) {
                Vector3d target = ((AStarWithTarget) pathfinder)
                    .getTargetPosition();
                double dx = fromPosition.x - target.x;
                double dy = fromPosition.y - target.y;
                double dz = fromPosition.z - target.z;
                return (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
            }
        };
        for (int sample = 0; sample < samples; sample++) {
            int offset = sample * 3;
            Vector3d start = new Vector3d(
                startPositions[offset],
                startPositions[offset + 1],
                startPositions[offset + 2]
            );
            Vector3d target = new Vector3d(
                targetPositions[offset],
                targetPositions[offset + 1],
                targetPositions[offset + 2]
            );
            AStarWithTarget pathfinder = new AStarWithTarget();
            pathfinder.setMaxPathLength(maximumPathLength);
            pathfinder.setOpenNodesLimit(openNodesLimit);
            pathfinder.setTotalNodesLimit(totalNodesLimit);
            pathfinder.setCanMoveDiagonal(true);
            pathfinder.setOptimizedBuildPath(false);
            ProbeMoveData probe = new ProbeMoveData();
            AStarBase.Progress state = pathfinder.initComputePath(
                agent,
                start,
                target,
                evaluator,
                walk,
                probe,
                poolProvider,
                store
            );
            int calls = 0;
            while (state == AStarBase.Progress.COMPUTING) {
                if (++calls > NativeNavigationPathProbe.MAX_TOTAL_NODES + 32) {
                    pathfinder.clearPath();
                    throw new IllegalStateException(
                        "Navigation path probe exceeded its bounded compute calls"
                    );
                }
                state = pathfinder.computePath(
                    agent,
                    walk,
                    probe,
                    nodesPerIteration,
                    store
                );
            }
            progress[sample] = (byte) state.ordinal();
            iterations[sample] = pathfinder.getIterations();
            visitedCounts[sample] = pathfinder.getVisitedBlocks().size();
            openCounts[sample] = pathfinder.getOpenCount();
            AStarNode node = pathfinder.getPath();
            int count = 0;
            while (node != null) {
                if (count >= maximumPathLength) {
                    pathfinder.clearPath();
                    throw new IllegalStateException(
                        "Native navigation path exceeds requested capacity"
                    );
                }
                int nodeOffset = sample * maximumPathLength + count;
                int positionOffset = nodeOffset * 3;
                Vector3d position = node.getPosition();
                pathPositions[positionOffset] = position.x;
                pathPositions[positionOffset + 1] = position.y;
                pathPositions[positionOffset + 2] = position.z;
                pathTravelCosts[nodeOffset] = node.getTravelCost();
                count++;
                node = node.getNextPathNode();
            }
            pathNodeCounts[sample] = count;
            pathfinder.clearPath();
        }
        return new NativeNavigationPathProbe(
            NATIVE_SERVER_VERSION,
            context.worldName(),
            context.worldgenProvider(),
            context.worldgenVersion(),
            context.seed(),
            startPositions,
            targetPositions,
            bounds,
            selectorValues,
            maximumPathLength,
            openNodesLimit,
            totalNodesLimit,
            nodesPerIteration,
            progress,
            iterations,
            visitedCounts,
            openCounts,
            pathNodeCounts,
            pathPositions,
            pathTravelCosts
        );
    }
}
