package com.hytalerlbridge.policy.combat.bridge;

import com.hytalerlbridge.policy.diagnostics.action.dodge.DodgeCooldownContract;
import java.lang.reflect.Field;

/** Reflection-only reader for the bridge's authored Dodge cooldown metadata. */
public final class ReflectiveBridgeDodgeContract {

    public static final int TICKS_PER_SECOND = 30;

    private static final String PROGRAM_CLASS =
        "com.hytalerlbridge.combat.dodge.NativeDodgeProgram";

    private ReflectiveBridgeDodgeContract() {}

    public static DodgeCooldownContract load()
        throws ReflectiveOperationException {
        ClassLoader loader = ReflectiveBridgeDodgeContract.class.getClassLoader();
        return load(Class.forName(PROGRAM_CLASS, true, loader));
    }

    /** Server-free seam used by the standalone contract gate. */
    static DodgeCooldownContract load(Class<?> program)
        throws ReflectiveOperationException {
        if (program == null) {
            throw new ClassNotFoundException("native Dodge program");
        }
        Field id = program.getField("COOLDOWN_ID");
        Field seconds = program.getField("COOLDOWN_SECONDS");
        Object rawId = id.get(null);
        Object rawSeconds = seconds.get(null);
        if (!(rawId instanceof String value)
                || !(rawSeconds instanceof Number duration)) {
            throw new NoSuchFieldException(
                "native Dodge cooldown metadata has incompatible types");
        }
        return new DodgeCooldownContract(
            value,
            duration.floatValue(),
            TICKS_PER_SECOND
        );
    }
}
