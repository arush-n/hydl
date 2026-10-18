package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.server.npc.corecomponents.movement.BodyMotionFindBase;
import com.hypixel.hytale.server.npc.instructions.BodyMotion;
import com.hypixel.hytale.server.npc.movement.Steering;
import com.hypixel.hytale.server.npc.movement.controllers.MotionController;
import com.hypixel.hytale.server.npc.navigation.IWaypoint;
import com.hypixel.hytale.server.npc.navigation.PathFollower;
import com.hypixel.hytale.server.npc.role.Role;
import java.lang.reflect.Field;
import java.util.Map;
import org.joml.Vector3dc;

/**
 * Opt-in, post-avoidance/pre-steering PathFollower evidence.
 *
 * <p>The public server API exposes the active wrapped body motion and
 * waypoints. Reflection is limited to source-pinned PathFollower state that
 * has no accessor. Source drift disables the trace rather than fabricating
 * values.</p>
 */
final class NativePathFollowerTrace {

    static final String SCHEMA = "hytalerl_native_path_follower_trace_v1";
    static final int VERSION = 1;
    static final String STAGE = "after_avoidance_before_steering";

    private static final Access ACCESS = Access.create();

    private final long worldTick;
    private final Frame frame;
    private final String reason;
    private final boolean transitionAvailable;
    private final String transitionReason;
    private final boolean waypointAdvanced;
    private final boolean pathFinished;

    private NativePathFollowerTrace(
        long worldTick,
        Frame frame,
        String reason,
        boolean transitionAvailable,
        String transitionReason,
        boolean waypointAdvanced,
        boolean pathFinished
    ) {
        this.worldTick = worldTick;
        this.frame = frame;
        this.reason = reason;
        this.transitionAvailable = transitionAvailable;
        this.transitionReason = transitionReason;
        this.waypointAdvanced = waypointAdvanced;
        this.pathFinished = pathFinished;
    }

    static void requireSupported() {
        if (ACCESS.error != null) {
            throw new IllegalStateException(
                "Native PathFollower trace is unavailable: " + ACCESS.error
            );
        }
    }

    static NativePathFollowerTrace pending() {
        return unavailable(0L, "not_captured");
    }

    static NativePathFollowerTrace capture(
        Role role,
        Vector3dc entityPosition,
        long worldTick,
        NativePathFollowerTrace previous
    ) {
        if (role == null || entityPosition == null) {
            return unavailable(worldTick, "target_role_or_position_missing");
        }
        if (ACCESS.error != null) {
            return unavailable(worldTick, ACCESS.error);
        }
        try {
            BodyMotion active = role.getLastBodySteeringMotion();
            BodyMotionFindBase<?> find = active == null
                ? null
                : active.findWrappedBodyMotion(BodyMotionFindBase.class);
            if (find == null) {
                String motion = active == null
                    ? "none"
                    : active.getClass().getSimpleName();
                return unavailable(
                    worldTick,
                    "active_motion_is_not_BodyMotionFindBase:" + motion
                );
            }
            PathFollower follower = (PathFollower) ACCESS.pathFollower.get(find);
            MotionController controller = role.getActiveMotionController();
            if (follower == null || controller == null) {
                return unavailable(
                    worldTick,
                    "path_follower_or_motion_controller_missing"
                );
            }
            IWaypoint current = follower.getCurrentWaypoint();
            IWaypoint next = follower.getNextWaypoint();
            Steering steering = role.getBodySteering();
            Vec3 currentPosition = waypointPosition(current);
            boolean bodyTranslationAvailable =
                steering != null && steering.hasTranslation();
            double bodyMaxDistance = steering == null
                ? 0.0
                : steering.getMaxDistance();
            boolean steeringComparable = current != null
                && bodyTranslationAvailable
                && !role.isAvoidingEntities()
                && !role.isApplySeparation()
                && Double.isFinite(bodyMaxDistance)
                && Math.abs(
                    bodyMaxDistance
                        - Vec3.distance(entityPosition, current.getPosition())
                ) <= 1.0e-9;
            Frame frame = new Frame(
                follower,
                current,
                next,
                active.getClass().getSimpleName(),
                Vec3.of(entityPosition),
                Vec3.of(controller.getComponentSelector()),
                currentPosition,
                waypointLength(current),
                waypointPosition(next),
                waypointLength(next),
                Vec3.of((Vector3dc) ACCESS.lastWaypointPosition.get(follower)),
                Vec3.of((Vector3dc) ACCESS.direction.get(follower)),
                Vec3.of((Vector3dc) ACCESS.rejection.get(follower)),
                ACCESS.currentWaypointDistanceSquared.getDouble(follower),
                follower.getRelativeSpeed(),
                ACCESS.relativeSpeedWaypoint.getDouble(follower),
                ACCESS.waypointRadius.getDouble(follower),
                ACCESS.rejectionWeight.getDouble(follower),
                follower.shouldSmoothPath(),
                follower.isWaypointFrozen(),
                bodyTranslationAvailable,
                steering == null
                    ? Vec3.ZERO
                    : Vec3.of(steering.getTranslation()),
                bodyMaxDistance,
                steeringComparable,
                role.isAvoidingEntities(),
                role.isApplySeparation()
            );
            Transition transition = transition(
                previous == null ? null : previous.frame,
                frame
            );
            return new NativePathFollowerTrace(
                worldTick,
                frame,
                "",
                transition.available,
                transition.reason,
                transition.advanced,
                transition.finished
            );
        } catch (ReflectiveOperationException | RuntimeException exception) {
            return unavailable(
                worldTick,
                "capture_failed:" + exception.getClass().getSimpleName()
            );
        }
    }

    void putInto(Map<String, Object> info) {
        info.put("native_navigation_trace_schema", SCHEMA);
        info.put("native_navigation_trace_version", VERSION);
        info.put("native_navigation_trace_requested", true);
        info.put("native_navigation_trace_stage", STAGE);
        info.put("native_navigation_trace_world_tick", worldTick);
        info.put("native_navigation_trace_available", frame != null);
        info.put("native_navigation_trace_reason", reason);
        info.put(
            "native_path_follower_transition_available",
            transitionAvailable
        );
        info.put(
            "native_path_follower_transition_reason",
            transitionReason
        );
        info.put("native_path_follower_waypoint_advanced", waypointAdvanced);
        info.put("native_path_follower_path_finished", pathFinished);
        if (frame == null) return;

        info.put("native_path_follower_motion", frame.motion);
        putVector(info, "native_path_follower_entity", frame.entityPosition);
        putVector(
            info,
            "native_path_follower_component_selector",
            frame.componentSelector
        );
        putWaypoint(
            info,
            "native_path_follower_current_waypoint",
            frame.currentWaypoint,
            frame.currentWaypointLength
        );
        putWaypoint(
            info,
            "native_path_follower_next_waypoint",
            frame.nextWaypoint,
            frame.nextWaypointLength
        );
        putVector(
            info,
            "native_path_follower_last_waypoint",
            frame.lastWaypointPosition
        );
        putVector(info, "native_path_follower_direction", frame.direction);
        putVector(info, "native_path_follower_rejection", frame.rejection);
        info.put(
            "native_path_follower_current_distance_squared",
            frame.currentWaypointDistanceSquared
        );
        info.put(
            "native_path_follower_relative_speed",
            frame.relativeSpeed
        );
        info.put(
            "native_path_follower_relative_speed_waypoint",
            frame.relativeSpeedWaypoint
        );
        info.put(
            "native_path_follower_waypoint_radius",
            frame.waypointRadius
        );
        info.put(
            "native_path_follower_rejection_weight",
            frame.rejectionWeight
        );
        info.put(
            "native_path_follower_should_smooth_path",
            frame.shouldSmoothPath
        );
        info.put(
            "native_path_follower_waypoint_frozen",
            frame.waypointFrozen
        );
        info.put(
            "native_path_follower_body_translation_available",
            frame.bodyTranslationAvailable
        );
        putVector(
            info,
            "native_path_follower_body_translation",
            frame.bodyTranslation
        );
        info.put(
            "native_path_follower_body_max_distance",
            frame.bodyMaxDistance
        );
        info.put(
            "native_path_follower_steering_comparable",
            frame.steeringComparable
        );
        info.put(
            "native_path_follower_avoidance_enabled",
            frame.avoidanceEnabled
        );
        info.put(
            "native_path_follower_separation_enabled",
            frame.separationEnabled
        );
    }

    private static Transition transition(Frame previous, Frame current) {
        if (previous == null) {
            return Transition.unavailable("no_previous_frame");
        }
        if (previous.follower != current.follower) {
            return Transition.unavailable("path_follower_changed");
        }
        if (previous.currentIdentity == null) {
            return Transition.unavailable("previous_path_inactive");
        }
        if (current.currentIdentity == previous.currentIdentity) {
            return Transition.available(false, false);
        }
        if (current.currentIdentity == previous.nextIdentity) {
            return Transition.available(true, current.currentIdentity == null);
        }
        return Transition.unavailable("path_smoothed_or_replanned");
    }

    private static NativePathFollowerTrace unavailable(
        long worldTick,
        String reason
    ) {
        return new NativePathFollowerTrace(
            worldTick,
            null,
            reason,
            false,
            reason,
            false,
            false
        );
    }

    private static Vec3 waypointPosition(IWaypoint waypoint) {
        return waypoint == null ? Vec3.ZERO : Vec3.of(waypoint.getPosition());
    }

    private static int waypointLength(IWaypoint waypoint) {
        return waypoint == null ? 0 : waypoint.getLength();
    }

    private static void putWaypoint(
        Map<String, Object> info,
        String prefix,
        Vec3 position,
        int length
    ) {
        info.put(prefix + "_present", length > 0);
        info.put(prefix + "_length", length);
        putVector(info, prefix, position);
    }

    private static void putVector(
        Map<String, Object> info,
        String prefix,
        Vec3 vector
    ) {
        info.put(prefix + "_x", vector.x);
        info.put(prefix + "_y", vector.y);
        info.put(prefix + "_z", vector.z);
    }

    private record Frame(
        PathFollower follower,
        IWaypoint currentIdentity,
        IWaypoint nextIdentity,
        String motion,
        Vec3 entityPosition,
        Vec3 componentSelector,
        Vec3 currentWaypoint,
        int currentWaypointLength,
        Vec3 nextWaypoint,
        int nextWaypointLength,
        Vec3 lastWaypointPosition,
        Vec3 direction,
        Vec3 rejection,
        double currentWaypointDistanceSquared,
        double relativeSpeed,
        double relativeSpeedWaypoint,
        double waypointRadius,
        double rejectionWeight,
        boolean shouldSmoothPath,
        boolean waypointFrozen,
        boolean bodyTranslationAvailable,
        Vec3 bodyTranslation,
        double bodyMaxDistance,
        boolean steeringComparable,
        boolean avoidanceEnabled,
        boolean separationEnabled
    ) {}

    private record Transition(
        boolean available,
        String reason,
        boolean advanced,
        boolean finished
    ) {
        private static Transition available(
            boolean advanced,
            boolean finished
        ) {
            return new Transition(true, "", advanced, finished);
        }

        private static Transition unavailable(String reason) {
            return new Transition(false, reason, false, false);
        }
    }

    private record Vec3(double x, double y, double z) {
        private static final Vec3 ZERO = new Vec3(0.0, 0.0, 0.0);

        private static Vec3 of(Vector3dc value) {
            return value == null
                ? ZERO
                : new Vec3(value.x(), value.y(), value.z());
        }

        private static double distance(Vector3dc left, Vector3dc right) {
            double dx = left.x() - right.x();
            double dy = left.y() - right.y();
            double dz = left.z() - right.z();
            return Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    private record Access(
        Field pathFollower,
        Field currentWaypointDistanceSquared,
        Field lastWaypointPosition,
        Field direction,
        Field rejection,
        Field relativeSpeedWaypoint,
        Field waypointRadius,
        Field rejectionWeight,
        String error
    ) {
        private static Access create() {
            try {
                return new Access(
                    field(BodyMotionFindBase.class, "pathFollower"),
                    field(PathFollower.class, "currentWaypointDistanceSquared"),
                    field(PathFollower.class, "lastWaypointPosition"),
                    field(PathFollower.class, "direction"),
                    field(PathFollower.class, "rejection"),
                    field(PathFollower.class, "relativeSpeedWaypoint"),
                    field(PathFollower.class, "waypointRadius"),
                    field(PathFollower.class, "rejectionWeight"),
                    null
                );
            } catch (ReflectiveOperationException | RuntimeException exception) {
                return new Access(
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    exception.getClass().getSimpleName()
                );
            }
        }

        private static Field field(Class<?> owner, String name)
            throws ReflectiveOperationException {
            Field field = owner.getDeclaredField(name);
            field.setAccessible(true);
            return field;
        }
    }
}
