package com.hytalerlbridge.policy.perception.acquisition.motion;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/** Checkpoint-owned scalars for the prospective native Dodge corridor. */
public record DodgeMotionParameters(
    int executionProfile,
    float authoredForce,
    float knockbackScale,
    float nominalDeltaSeconds,
    float loadedDeltaSeconds,
    float serverTicksPerSecond,
    int motionTimingProfile,
    float configuredAirResistance,
    float configuredAirResistanceMax,
    float configuredGroundResistance,
    float configuredGroundResistanceMax,
    float configuredResistanceThreshold,
    int configuredResistanceStyle,
    float nativeHorizontalFactor,
    float nativeMovementVelocityResistance,
    float nativeGroundDragBase,
    float nativeAirDragMin,
    float nativeAirDragMax,
    float nativeAirDragMinSpeed,
    float nativeAirDragMaxSpeed,
    float nativeReferenceTicksPerSecond,
    float nativePerAxisDeadzone,
    int nativeResistanceStyle,
    float velocityRemovalSquared,
    int clearanceTickCapacity
) {

    public static final String SCHEMA =
        "hytalerl_dodge_corridor_motion_v1";
    public static final int VERSION = 1;
    public static final int NATIVE_NPC_NULL_CONFIG = 0;
    public static final int AUTHORED_CLIENT_CONFIGURED = 1;
    public static final int LINEAR_RESISTANCE = 0;
    public static final int EXPONENTIAL_RESISTANCE = 1;
    public static final int FORCE_VELOCITY_RESISTANCE = 2;

    public DodgeMotionParameters {
        boolean profile = executionProfile == NATIVE_NPC_NULL_CONFIG
            || executionProfile == AUTHORED_CLIENT_CONFIGURED;
        boolean timing = motionTimingProfile >= 0 && motionTimingProfile <= 2;
        boolean styles = (configuredResistanceStyle == LINEAR_RESISTANCE
                || configuredResistanceStyle == EXPONENTIAL_RESISTANCE)
            && nativeResistanceStyle == FORCE_VELOCITY_RESISTANCE;
        boolean finite = finitePositive(authoredForce)
            && finitePositive(knockbackScale)
            && finitePositive(nominalDeltaSeconds)
            && finitePositive(loadedDeltaSeconds)
            && finitePositive(serverTicksPerSecond)
            && unit(configuredAirResistance)
            && unit(configuredAirResistanceMax)
            && unit(configuredGroundResistance)
            && unit(configuredGroundResistanceMax)
            && finitePositive(configuredResistanceThreshold)
            && finitePositive(nativeHorizontalFactor)
            && finitePositive(nativeMovementVelocityResistance)
            && unit(nativeGroundDragBase)
            && unit(nativeAirDragMin)
            && unit(nativeAirDragMax)
            && finiteNonnegative(nativeAirDragMinSpeed)
            && finitePositive(nativeAirDragMaxSpeed)
            && nativeAirDragMaxSpeed > nativeAirDragMinSpeed
            && finitePositive(nativeReferenceTicksPerSecond)
            && finiteNonnegative(nativePerAxisDeadzone)
            && finitePositive(velocityRemovalSquared);
        if (!profile || !timing || !styles || !finite
            || clearanceTickCapacity < 1
            || clearanceTickCapacity > 4096) {
            throw new IllegalArgumentException(
                "invalid Dodge motion parameter contract");
        }
    }

    /**
     * Load the optional migration file. Absence is a supported old-profile
     * state and causes production acquisition to fail closed; malformed
     * content is an error rather than a silent fallback to Java constants.
     */
    public static DodgeMotionParameters loadOptional(Path path)
        throws IOException {
        if (path == null || !Files.isRegularFile(path)) return null;
        Map<String, String> values = new HashMap<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] fields = line.split("\\t", -1);
            if (fields.length != 2 || fields[0].isBlank()
                || values.put(fields[0], fields[1]) != null) {
                throw new IOException("invalid Dodge motion row: " + line);
            }
        }
        if (!SCHEMA.equals(require(values, "schema"))) {
            throw new IOException("Dodge motion schema drift");
        }
        int version = integer(values, "version");
        if (version != VERSION) {
            throw new IOException("Dodge motion version drift: " + version);
        }
        try {
            DodgeMotionParameters result = new DodgeMotionParameters(
                integer(values, "execution_profile"),
                number(values, "authored_force"),
                number(values, "knockback_scale"),
                number(values, "nominal_delta_seconds"),
                number(values, "loaded_delta_seconds"),
                number(values, "server_ticks_per_second"),
                integer(values, "motion_timing_profile"),
                number(values, "configured_air_resistance"),
                number(values, "configured_air_resistance_max"),
                number(values, "configured_ground_resistance"),
                number(values, "configured_ground_resistance_max"),
                number(values, "configured_resistance_threshold"),
                integer(values, "configured_resistance_style"),
                number(values, "native_horizontal_factor"),
                number(values, "native_movement_velocity_resistance"),
                number(values, "native_ground_drag_base"),
                number(values, "native_air_drag_min"),
                number(values, "native_air_drag_max"),
                number(values, "native_air_drag_min_speed"),
                number(values, "native_air_drag_max_speed"),
                number(values, "native_reference_ticks_per_second"),
                number(values, "native_per_axis_deadzone"),
                integer(values, "native_resistance_style"),
                number(values, "velocity_removal_squared"),
                integer(values, "clearance_tick_capacity")
            );
            if (values.size() != 27) {
                throw new IOException(
                    "Dodge motion field-count drift: " + values.size());
            }
            return result;
        } catch (IllegalArgumentException invalid) {
            throw new IOException("invalid Dodge motion contract", invalid);
        }
    }

    private static String require(Map<String, String> values, String name)
        throws IOException {
        String value = values.get(name);
        if (value == null || value.isBlank()) {
            throw new IOException("Dodge motion contract has no " + name);
        }
        return value;
    }

    private static float number(Map<String, String> values, String name)
        throws IOException {
        try {
            float value = Float.parseFloat(require(values, name));
            if (!Float.isFinite(value)) throw new NumberFormatException();
            return value;
        } catch (NumberFormatException invalid) {
            throw new IOException("invalid Dodge motion scalar " + name,
                invalid);
        }
    }

    private static int integer(Map<String, String> values, String name)
        throws IOException {
        try {
            return Integer.parseInt(require(values, name));
        } catch (NumberFormatException invalid) {
            throw new IOException("invalid Dodge motion integer " + name,
                invalid);
        }
    }

    private static boolean finitePositive(float value) {
        return Float.isFinite(value) && value > 0.0f;
    }

    private static boolean finiteNonnegative(float value) {
        return Float.isFinite(value) && value >= 0.0f;
    }

    private static boolean unit(float value) {
        return Float.isFinite(value) && value >= 0.0f && value <= 1.0f;
    }
}
