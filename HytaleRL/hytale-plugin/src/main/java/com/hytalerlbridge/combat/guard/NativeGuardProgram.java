package com.hytalerlbridge.combat.guard;

import java.util.Map;

/**
 * Parseable Hytale 0.5.7 common-guard magnitudes.
 *
 * <p>The native session still executes the authored interaction chain. These
 * constants expose the asset values that the JAX mechanics contract mirrors;
 * they do not replace a live outcome differential.</p>
 */
public final class NativeGuardProgram {

    public static final String SCHEMA = "hytalerl_native_guard_program_v1";
    public static final int VERSION = 1;
    public static final double GUARD_ENTRY_STAMINA_COST = 0.5;
    public static final double GUARD_HALF_ANGLE_DEGREES = 90.0;
    public static final double GUARD_EXIT_REGEN_DELAY_SECONDS = -1.0;
    public static final double GUARD_BASH_STAMINA_COST = 2.0;

    private NativeGuardProgram() {}

    /** Publish the same declared values beside the native guard lifecycle. */
    public static void putInto(Map<String, Object> info) {
        info.put("native_guard_program_schema", SCHEMA);
        info.put("native_guard_program_version", VERSION);
        info.put(
            "native_guard_entry_stamina_cost",
            GUARD_ENTRY_STAMINA_COST
        );
        info.put(
            "native_guard_half_angle_degrees",
            GUARD_HALF_ANGLE_DEGREES
        );
        info.put(
            "native_guard_exit_regen_delay_seconds",
            GUARD_EXIT_REGEN_DELAY_SECONDS
        );
        info.put(
            "native_guard_bash_stamina_cost",
            GUARD_BASH_STAMINA_COST
        );
    }
}
