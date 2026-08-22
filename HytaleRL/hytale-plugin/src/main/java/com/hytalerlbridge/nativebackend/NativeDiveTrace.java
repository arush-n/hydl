package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.movement.Steering;
import com.hypixel.hytale.server.npc.movement.controllers.MotionController;
import com.hypixel.hytale.server.npc.movement.controllers.MotionControllerBase;
import com.hypixel.hytale.server.npc.movement.controllers.MotionControllerDive;
import com.hypixel.hytale.server.npc.role.Role;
import java.lang.reflect.Field;
import java.util.Map;
import org.joml.Vector3dc;

/** Opt-in controller-state evidence for the isolated Dive fidelity fixture. */
final class NativeDiveTrace {

    static final String SCHEMA = "hytalerl_native_dive_motion_trace_v1";
    static final int VERSION = 1;

    private static final Access ACCESS = Access.create();

    private final Frame before;
    private final Frame after;
    private final Input input;
    private final String reason;

    private NativeDiveTrace(
        Frame before,
        Frame after,
        Input input,
        String reason
    ) {
        this.before = before;
        this.after = after;
        this.input = input;
        this.reason = reason;
    }

    static void requireSupported() {
        if (ACCESS.error != null) {
            throw new IllegalStateException(
                "Native Dive trace is unavailable: " + ACCESS.error
            );
        }
    }

    static NativeDiveTrace captureBefore(
        MotionController controller,
        Steering steering,
        Role role,
        Ref<EntityStore> reference,
        Store<EntityStore> store
    ) {
        if (!(controller instanceof MotionControllerDive dive)) {
            return unavailable(
                "active_controller_is_not_Dive:"
                    + (controller == null
                        ? "none"
                        : controller.getClass().getSimpleName())
            );
        }
        if (steering == null) return unavailable("body_steering_missing");
        if (ACCESS.error != null) return unavailable(ACCESS.error);
        try {
            return new NativeDiveTrace(
                Frame.capture(dive, role, reference, store),
                null,
                Input.capture(steering),
                ""
            );
        } catch (ReflectiveOperationException | RuntimeException exception) {
            return unavailable(
                "capture_before_failed:"
                    + exception.getClass().getSimpleName()
            );
        }
    }

    NativeDiveTrace captureAfter(
        MotionController controller,
        Role role,
        Ref<EntityStore> reference,
        Store<EntityStore> store
    ) {
        if (before == null || input == null) return this;
        if (!(controller instanceof MotionControllerDive dive)) {
            return new NativeDiveTrace(
                before,
                null,
                input,
                "active_controller_changed"
            );
        }
        try {
            return new NativeDiveTrace(
                before,
                Frame.capture(dive, role, reference, store),
                input,
                ""
            );
        } catch (ReflectiveOperationException | RuntimeException exception) {
            return new NativeDiveTrace(
                before,
                null,
                input,
                "capture_after_failed:"
                    + exception.getClass().getSimpleName()
            );
        }
    }

    boolean availableBefore() {
        return before != null && input != null;
    }

    void putInto(Map<String, Object> info) {
        info.put("native_dive_trace_schema", SCHEMA);
        info.put("native_dive_trace_version", VERSION);
        info.put(
            "native_dive_trace_available",
            before != null && after != null && input != null
        );
        info.put("native_dive_trace_reason", reason);
        if (before == null || after == null || input == null) return;
        before.putInto(info, "native_dive_before_");
        after.putInto(info, "native_dive_after_");
        input.putInto(info);
    }

    private static NativeDiveTrace unavailable(String reason) {
        return new NativeDiveTrace(null, null, null, reason);
    }

    private record Frame(
        double heading,
        double pitch,
        double moveSpeed,
        double climbSpeed,
        int motionKind,
        boolean onGround,
        boolean inWater,
        boolean canSteer,
        boolean collisionWithSolid,
        double effectHorizontalSpeedMultiplier,
        double maximumHorizontalSpeed,
        double maximumVerticalSpeed,
        double acceleration,
        double maximumRotationSpeed,
        double maximumMoveTurnAngle,
        double epsilonAngle,
        double epsilonSpeed,
        double swimDepth
    ) {
        private static Frame capture(
            MotionControllerDive dive,
            Role role,
            Ref<EntityStore> reference,
            Store<EntityStore> store
        ) throws ReflectiveOperationException {
            return new Frame(
                dive.getYaw(),
                dive.getPitch(),
                ACCESS.moveSpeed.getDouble(dive),
                ACCESS.climbSpeed.getDouble(dive),
                dive.getMotionKind().ordinal(),
                role != null && role.isOnGround(),
                dive.inWater(),
                dive.canSteer(reference, store),
                ACCESS.collisionWithSolid.getBoolean(dive),
                ACCESS.effectHorizontalSpeedMultiplier.getDouble(dive),
                ACCESS.maximumHorizontalSpeed.getDouble(dive),
                ACCESS.maximumVerticalSpeed.getDouble(dive),
                ACCESS.acceleration.getDouble(dive),
                ACCESS.maximumRotationSpeed.getDouble(dive),
                ACCESS.maximumMoveTurnAngle.getFloat(dive),
                dive.getEpsilonAngle(),
                dive.getEpsilonSpeed(),
                ACCESS.swimDepth.getDouble(dive)
            );
        }

        private void putInto(
            Map<String, Object> info,
            String prefix
        ) {
            info.put(prefix + "heading", heading);
            info.put(prefix + "pitch", pitch);
            info.put(prefix + "move_speed", moveSpeed);
            info.put(prefix + "climb_speed", climbSpeed);
            info.put(prefix + "motion_kind", motionKind);
            info.put(prefix + "on_ground", onGround);
            info.put(prefix + "in_water", inWater);
            info.put(prefix + "can_steer", canSteer);
            info.put(prefix + "collision_with_solid", collisionWithSolid);
            info.put(
                prefix + "effect_horizontal_speed_multiplier",
                effectHorizontalSpeedMultiplier
            );
            info.put(
                prefix + "maximum_horizontal_speed",
                maximumHorizontalSpeed
            );
            info.put(
                prefix + "maximum_vertical_speed",
                maximumVerticalSpeed
            );
            info.put(prefix + "acceleration", acceleration);
            info.put(prefix + "maximum_rotation_speed", maximumRotationSpeed);
            info.put(
                prefix + "maximum_move_turn_angle",
                maximumMoveTurnAngle
            );
            info.put(prefix + "epsilon_angle", epsilonAngle);
            info.put(prefix + "epsilon_speed", epsilonSpeed);
            info.put(prefix + "swim_depth", swimDepth);
        }
    }

    private record Input(
        boolean hasTranslation,
        double translationX,
        double translationY,
        double translationZ,
        boolean hasYaw,
        double yaw,
        boolean hasPitch,
        double pitch
    ) {
        private static Input capture(Steering steering) {
            Vector3dc translation = steering.getTranslation();
            return new Input(
                steering.hasTranslation(),
                translation.x(),
                translation.y(),
                translation.z(),
                steering.hasYaw(),
                steering.hasYaw() ? steering.getYaw() : 0.0,
                steering.hasPitch(),
                steering.hasPitch() ? steering.getPitch() : 0.0
            );
        }

        private void putInto(Map<String, Object> info) {
            info.put("native_dive_input_has_translation", hasTranslation);
            info.put("native_dive_input_x", translationX);
            info.put("native_dive_input_y", translationY);
            info.put("native_dive_input_z", translationZ);
            info.put("native_dive_input_has_yaw", hasYaw);
            info.put("native_dive_input_yaw", yaw);
            info.put("native_dive_input_has_pitch", hasPitch);
            info.put("native_dive_input_pitch", pitch);
        }
    }

    private record Access(
        Field moveSpeed,
        Field effectHorizontalSpeedMultiplier,
        Field maximumHorizontalSpeed,
        Field climbSpeed,
        Field maximumVerticalSpeed,
        Field acceleration,
        Field maximumRotationSpeed,
        Field maximumMoveTurnAngle,
        Field collisionWithSolid,
        Field swimDepth,
        String error
    ) {
        private static Access create() {
            try {
                return new Access(
                    field(MotionControllerBase.class, "moveSpeed"),
                    field(
                        MotionControllerBase.class,
                        "effectHorizontalSpeedMultiplier"
                    ),
                    field(
                        MotionControllerBase.class,
                        "maxHorizontalSpeed"
                    ),
                    field(MotionControllerDive.class, "climbSpeed"),
                    field(MotionControllerDive.class, "maxVerticalSpeed"),
                    field(MotionControllerDive.class, "acceleration"),
                    field(MotionControllerDive.class, "maxRotationSpeed"),
                    field(MotionControllerDive.class, "maxMoveTurnAngle"),
                    field(MotionControllerDive.class, "collisionWithSolid"),
                    field(MotionControllerDive.class, "swimDepth"),
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
