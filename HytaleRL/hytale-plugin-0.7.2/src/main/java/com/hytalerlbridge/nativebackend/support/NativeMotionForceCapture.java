package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.server.core.entity.knockback.KnockbackComponent;
import com.hypixel.hytale.server.npc.movement.controllers.MotionController;
import com.hypixel.hytale.server.npc.movement.controllers.MotionControllerBase;
import com.hytalerlbridge.observation.NativeActorEvidenceFrame;
import java.lang.reflect.Field;
import java.util.List;
import org.joml.Vector3d;

/** Captures the distinct native NPC force channels without inventing zeros. */
public final class NativeMotionForceCapture {
    private static final Field APPLIED_VELOCITIES_FIELD =
        resolveAppliedVelocitiesField();
    private static final Field APPLIED_VELOCITY_FIELD =
        resolveAppliedVelocityField();

    private NativeMotionForceCapture() {}

    public static NativeActorEvidenceFrame.MotionForce capture(
        MotionController controller,
        KnockbackComponent knockback
    ) {
        CapturedVector legacy = controller == null
            ? CapturedVector.unavailable()
            : snapshot(controller.getExternalVelocity());
        CapturedConfigured configured = captureConfigured(controller);
        // Component absence is authoritative: no knockback is staged.
        CapturedVector pending = knockback == null
            ? CapturedVector.availableZero()
            : snapshot(knockback.getVelocity());
        return NativeActorEvidenceFrame.MotionForce.resolve(
            legacy.available(),
            legacy.value(),
            configured.available(),
            configured.value(),
            configured.count(),
            pending.available(),
            pending.value()
        );
    }

    /** Source-layout gate used by the server-free bridge test. */
    public static boolean configuredVelocityLayoutAvailable() {
        return APPLIED_VELOCITIES_FIELD != null
            && APPLIED_VELOCITY_FIELD != null;
    }

    private static CapturedConfigured captureConfigured(
        MotionController controller
    ) {
        if (
            !(controller instanceof MotionControllerBase base)
                || APPLIED_VELOCITIES_FIELD == null
                || APPLIED_VELOCITY_FIELD == null
        ) {
            return CapturedConfigured.unavailable();
        }
        try {
            Object raw = APPLIED_VELOCITIES_FIELD.get(base);
            if (!(raw instanceof List<?> rows)) {
                return CapturedConfigured.unavailable();
            }
            double[] sum = new double[3];
            for (Object row : rows) {
                if (
                    row == null
                        || !APPLIED_VELOCITY_FIELD
                            .getDeclaringClass()
                            .isInstance(row)
                ) {
                    return CapturedConfigured.unavailable();
                }
                Object rawVelocity = APPLIED_VELOCITY_FIELD.get(row);
                if (!(rawVelocity instanceof Vector3d value)) {
                    return CapturedConfigured.unavailable();
                }
                if (!finite(value)) {
                    return CapturedConfigured.unavailable();
                }
                sum[0] += value.x;
                sum[1] += value.y;
                sum[2] += value.z;
            }
            if (!finite(sum)) return CapturedConfigured.unavailable();
            return new CapturedConfigured(true, sum, rows.size());
        } catch (IllegalAccessException | RuntimeException exception) {
            return CapturedConfigured.unavailable();
        }
    }

    private static Field resolveAppliedVelocityField() {
        try {
            Class<?> type = Class.forName(
                MotionControllerBase.class.getName() + "$AppliedVelocity"
            );
            Field field = type.getDeclaredField("velocity");
            return field.trySetAccessible() ? field : null;
        } catch (
            ClassNotFoundException
                | NoSuchFieldException
                | RuntimeException exception
        ) {
            return null;
        }
    }

    private static CapturedVector snapshot(Vector3d value) {
        if (!finite(value)) return CapturedVector.unavailable();
        return new CapturedVector(
            true,
            new double[] {value.x, value.y, value.z}
        );
    }

    private static boolean finite(Vector3d value) {
        return value != null
            && Double.isFinite(value.x)
            && Double.isFinite(value.y)
            && Double.isFinite(value.z);
    }

    private static boolean finite(double[] value) {
        return value != null
            && value.length == 3
            && Double.isFinite(value[0])
            && Double.isFinite(value[1])
            && Double.isFinite(value[2]);
    }

    private static Field resolveAppliedVelocitiesField() {
        try {
            Field field = MotionControllerBase.class.getDeclaredField(
                "appliedVelocities"
            );
            return field.trySetAccessible() ? field : null;
        } catch (NoSuchFieldException | RuntimeException exception) {
            return null;
        }
    }

    private record CapturedVector(boolean available, double[] value) {
        private static CapturedVector unavailable() {
            return new CapturedVector(false, new double[3]);
        }

        private static CapturedVector availableZero() {
            return new CapturedVector(true, new double[3]);
        }
    }

    private record CapturedConfigured(
        boolean available,
        double[] value,
        int count
    ) {
        private static CapturedConfigured unavailable() {
            return new CapturedConfigured(false, new double[3], 0);
        }
    }
}
