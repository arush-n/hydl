package com.hytalerlbridge.policy.perception.acquisition.motion;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.shape.Box;
import com.hypixel.hytale.server.core.modules.collision.BlockCollisionData;
import com.hypixel.hytale.server.core.modules.collision.CollisionModule;
import com.hypixel.hytale.server.core.modules.collision.CollisionResult;
import com.hypixel.hytale.server.core.modules.entity.component.BoundingBox;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.policy.perception.acquisition.ServerActorReader;
import java.util.ArrayList;
import java.util.List;
import org.joml.Vector3d;

/** Public-engine swept-AABB acquisition for live Dodge direction masks. */
public final class ServerDodgeCorridorReader {

    private static final double BOUNDS_EPSILON = 1.0e-6;

    /** Actor/world acquisition boundary, before any directional sweep. */
    public enum AcquisitionReason {
        READY,
        INVALID_REFERENCE,
        WORLD_UNAVAILABLE,
        COLLISION_MODULE_UNAVAILABLE,
        COLLISION_MODULE_DISABLED,
        BOUNDING_BOX_UNAVAILABLE,
        TICK_RATE_MISMATCH,
        TIMING_PROFILE_REQUIRES_LOCAL_CLOCK,
        INVALID_BOUNDING_BOX,
        INVALID_ACTOR_POSITION,
        ENGINE_EXCEPTION,
    }

    /** Public-engine evidence for one continuous actor-AABB sweep. */
    public record SweepEvidence(
        DodgeCorridorEvaluator.Direction direction,
        DodgeCorridorEvaluator.SweepResult result,
        int blockCollisionCount,
        boolean supportingSlide,
        double slideStart,
        double slideEnd,
        double firstCollisionStart,
        double firstCollisionEnd,
        double firstNormalX,
        double firstNormalY,
        double firstNormalZ,
        boolean firstUpwardNormal,
        boolean firstTouching,
        boolean firstOverlapping,
        int firstBlockX,
        int firstBlockY,
        int firstBlockZ,
        int firstBlockId,
        boolean obstructed,
        double stopFraction,
        int blockingCollisionIndex,
        int ignoredSlideContacts,
        int ignoredGrazingContacts,
        int lateApproachContacts,
        String unavailableType
    ) {
        public SweepEvidence {
            if (direction == null || result == null || blockCollisionCount < 0
                || unavailableType == null || blockingCollisionIndex < -1
                || ignoredSlideContacts < 0 || ignoredGrazingContacts < 0
                || lateApproachContacts < 0) {
                throw new IllegalArgumentException("invalid Dodge sweep evidence");
            }
        }
    }

    /** Typed live acquisition result; policy still consumes only {@code clear}. */
    public record Capture(
        boolean[] clear,
        AcquisitionReason acquisitionReason,
        DodgeCorridorEvaluator.Evaluation evaluation,
        SweepEvidence[] sweeps,
        int worldTicksPerSecond,
        int expectedTicksPerSecond,
        boolean grounded,
        double boundsWidth,
        double boundsHeight,
        double boundsDepth
    ) {
        public Capture {
            if (clear == null
                || clear.length != DodgeCorridorEvaluator.DIRECTION_COUNT
                || acquisitionReason == null || evaluation == null
                || sweeps == null
                || sweeps.length != DodgeCorridorEvaluator.DIRECTION_COUNT) {
                throw new IllegalArgumentException(
                    "invalid Dodge corridor capture");
            }
            clear = clear.clone();
            sweeps = sweeps.clone();
        }

        @Override public boolean[] clear() {
            return clear.clone();
        }

        @Override public SweepEvidence[] sweeps() {
            return sweeps.clone();
        }
    }

    private ServerDodgeCorridorReader() {}

    /** Missing actor/world/bounds/geometry data closes every direction. */
    public static boolean[] capture(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        ServerActorReader.Actor actor,
        DodgeMotionParameters parameters
    ) {
        return captureDetailed(ref, store, actor, parameters).clear();
    }

    /** Capture with typed precondition and continuous-contact diagnostics. */
    public static Capture captureDetailed(
        Ref<EntityStore> ref,
        Store<EntityStore> store,
        ServerActorReader.Actor actor,
        DodgeMotionParameters parameters
    ) {
        if (ref == null || !ref.isValid() || store == null || actor == null
            || parameters == null) {
            return unavailable(AcquisitionReason.INVALID_REFERENCE, parameters);
        }
        World world;
        CollisionModule collision;
        BoundingBox component;
        try {
            world = store.getExternalData().getWorld();
            collision = CollisionModule.get();
            component = store.getComponent(ref, BoundingBox.getComponentType());
        } catch (RuntimeException | LinkageError unavailable) {
            return unavailable(AcquisitionReason.ENGINE_EXCEPTION, parameters);
        }
        if (world == null) {
            return unavailable(AcquisitionReason.WORLD_UNAVAILABLE, parameters);
        }
        if (collision == null) {
            return unavailable(
                AcquisitionReason.COLLISION_MODULE_UNAVAILABLE, parameters);
        }
        if (collision.isDisabled()) {
            return unavailable(
                AcquisitionReason.COLLISION_MODULE_DISABLED, parameters);
        }
        if (component == null) {
            return unavailable(
                AcquisitionReason.BOUNDING_BOX_UNAVAILABLE, parameters);
        }
        int expectedTps = Math.round(parameters.serverTicksPerSecond());
        if (world.getTps() != expectedTps) {
            return unavailable(
                AcquisitionReason.TICK_RATE_MISMATCH,
                parameters,
                world.getTps(),
                expectedTps
            );
        }
        // JAX profiles 1/2 key alternating loaded_dt steps to the reset-local
        // combat tick. Absolute World.getTick() parity is not equivalent.
        if (parameters.motionTimingProfile() != 0) {
            return unavailable(
                AcquisitionReason.TIMING_PROFILE_REQUIRES_LOCAL_CLOCK,
                parameters,
                world.getTps(),
                expectedTps
            );
        }
        Box source = component.getBoundingBox();
        if (!valid(source)) {
            return unavailable(
                AcquisitionReason.INVALID_BOUNDING_BOX,
                parameters,
                world.getTps(),
                expectedTps
            );
        }
        if (!Double.isFinite(actor.x())
            || !Double.isFinite(actor.y())
            || !Double.isFinite(actor.z())) {
            return unavailable(
                AcquisitionReason.INVALID_ACTOR_POSITION,
                parameters,
                world.getTps(),
                expectedTps
            );
        }
        Box bounds = new Box(source);
        Vector3d position = new Vector3d(actor.x(), actor.y(), actor.z());
        SweepEvidence[] sweeps = emptySweeps();
        DodgeCorridorEvaluator.Evaluation evaluation =
            DodgeCorridorEvaluator.evaluateDetailed(
            actor.yawDegrees(),
            0L,
            actor.movement(15),
            parameters,
            (direction, displacement) -> {
                SweepEvidence evidence = sweep(
                    store, bounds, position, direction, displacement);
                sweeps[direction.ordinal()] = evidence;
                return evidence.result();
            }
        );
        return new Capture(
            evaluation.clear(),
            AcquisitionReason.READY,
            evaluation,
            sweeps,
            world.getTps(),
            expectedTps,
            actor.movement(15),
            source.max.x - source.min.x,
            source.max.y - source.min.y,
            source.max.z - source.min.z
        );
    }

    /**
     * Query one arbitrary actor-shaped world corridor. Exposed for the
     * pre-claim fixture-placement gate so live acquisition and placement use
     * one collision rule.
     */
    public static SweepEvidence sweep(
        Store<EntityStore> store,
        Box bounds,
        Vector3d position,
        DodgeCorridorEvaluator.Direction direction,
        DodgeCorridorEvaluator.Displacement displacement
    ) {
        Vector3d delta = new Vector3d(
            displacement.x(), displacement.y(), displacement.z());
        if (!Double.isFinite(delta.x) || !Double.isFinite(delta.y)
            || !Double.isFinite(delta.z)
            || delta.lengthSquared() <= BOUNDS_EPSILON * BOUNDS_EPSILON) {
            return unavailableSweep(direction, "degenerate_displacement");
        }
        try {
            // Slides stay enabled so the engine can identify supporting-floor
            // contacts. MotionObstructionClassifier then applies the exact
            // public NPC rail-step rule rather than treating raw contact count
            // as obstruction.
            // Character collision is disabled: the JAX corridor queries
            // immutable world geometry, not transient entity occupancy.
            CollisionResult result = new CollisionResult(true, false);
            CollisionModule.findCollisions(
                bounds,
                position,
                delta,
                false,
                result,
                store
            );
            int count = result.getBlockCollisionCount();
            BlockCollisionData first = result.getFirstBlockCollision();
            List<MotionObstructionClassifier.Contact> contacts =
                new ArrayList<>(count);
            for (int index = 0; index < count; index++) {
                BlockCollisionData hit = result.getBlockCollision(index);
                contacts.add(new MotionObstructionClassifier.Contact(
                    hit.collisionStart,
                    hit.collisionNormal.x,
                    hit.collisionNormal.y,
                    hit.collisionNormal.z
                ));
            }
            MotionObstructionClassifier.Classification classification =
                MotionObstructionClassifier.classify(
                    delta.x,
                    delta.y,
                    delta.z,
                    result.isSliding,
                    result.slideEnd,
                    contacts
                );
            return new SweepEvidence(
                direction,
                classification.obstructed()
                    ? DodgeCorridorEvaluator.SweepResult.BLOCKED
                    : DodgeCorridorEvaluator.SweepResult.CLEAR,
                count,
                result.isSliding,
                result.slideStart,
                result.slideEnd,
                first == null ? Double.NaN : first.collisionStart,
                first == null ? Double.NaN : first.collisionEnd,
                first == null ? Double.NaN : first.collisionNormal.x,
                first == null ? Double.NaN : first.collisionNormal.y,
                first == null ? Double.NaN : first.collisionNormal.z,
                first != null && first.collisionNormal.y > 0.5,
                first != null && first.touching,
                first != null && first.overlapping,
                first == null ? Integer.MIN_VALUE : first.x,
                first == null ? Integer.MIN_VALUE : first.y,
                first == null ? Integer.MIN_VALUE : first.z,
                first == null ? Integer.MIN_VALUE : first.blockId,
                classification.obstructed(),
                classification.stopFraction(),
                classification.blockingContactIndex(),
                classification.ignoredSlideContacts(),
                classification.ignoredGrazingContacts(),
                classification.lateApproachContacts(),
                ""
            );
        } catch (RuntimeException | LinkageError unavailable) {
            return unavailableSweep(
                direction, unavailable.getClass().getSimpleName());
        }
    }

    private static boolean valid(Box value) {
        return value != null
            && finite(value.min.x) && finite(value.min.y) && finite(value.min.z)
            && finite(value.max.x) && finite(value.max.y) && finite(value.max.z)
            && value.max.x - value.min.x > BOUNDS_EPSILON
            && value.max.y - value.min.y > BOUNDS_EPSILON
            && value.max.z - value.min.z > BOUNDS_EPSILON;
    }

    private static boolean finite(double value) {
        return Double.isFinite(value);
    }

    private static Capture unavailable(
        AcquisitionReason reason,
        DodgeMotionParameters parameters
    ) {
        return unavailable(reason, parameters, -1,
            parameters == null
                ? -1
                : Math.round(parameters.serverTicksPerSecond()));
    }

    private static Capture unavailable(
        AcquisitionReason reason,
        DodgeMotionParameters parameters,
        int worldTps,
        int expectedTps
    ) {
        DodgeCorridorEvaluator.Evaluation evaluation =
            DodgeCorridorEvaluator.evaluateDetailed(
                Double.NaN,
                0L,
                false,
                parameters,
                (_direction, _displacement) ->
                    DodgeCorridorEvaluator.SweepResult.UNAVAILABLE
            );
        return new Capture(
            closed(),
            reason,
            evaluation,
            emptySweeps(),
            worldTps,
            expectedTps,
            false,
            Double.NaN,
            Double.NaN,
            Double.NaN
        );
    }

    private static SweepEvidence unavailableSweep(
        DodgeCorridorEvaluator.Direction direction,
        String type
    ) {
        return new SweepEvidence(
            direction,
            DodgeCorridorEvaluator.SweepResult.UNAVAILABLE,
            0,
            false,
            Double.NaN,
            Double.NaN,
            Double.NaN,
            Double.NaN,
            Double.NaN,
            Double.NaN,
            Double.NaN,
            false,
            false,
            false,
            Integer.MIN_VALUE,
            Integer.MIN_VALUE,
            Integer.MIN_VALUE,
            Integer.MIN_VALUE,
            false,
            1.0,
            -1,
            0,
            0,
            0,
            type == null ? "" : type
        );
    }

    private static SweepEvidence[] emptySweeps() {
        SweepEvidence[] result =
            new SweepEvidence[DodgeCorridorEvaluator.DIRECTION_COUNT];
        for (DodgeCorridorEvaluator.Direction direction
                : DodgeCorridorEvaluator.Direction.values()) {
            result[direction.ordinal()] = unavailableSweep(direction, "not_run");
        }
        return result;
    }

    private static boolean[] closed() {
        return new boolean[DodgeCorridorEvaluator.DIRECTION_COUNT];
    }
}
