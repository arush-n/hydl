package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.server.npc.movement.controllers.MotionController;
import java.util.ArrayList;
import java.util.List;
import org.joml.Vector3d;

/**
 * Extracted verbatim from {@code NativeEnvironmentSession}.
 *
 * <p>Every method here touches no session instance field, so the move
 * required no state surgery.
 */
public final class SupportMath {

    private SupportMath() {}

    public static float normalizeRadians(float value) {
        while (value > Math.PI) value -= (float) (Math.PI * 2.0);
        while (value < -Math.PI) value += (float) (Math.PI * 2.0);
        return value;
    }

    public static double normalizeDegrees(double value) {
        double normalized = (value + 180.0) % 360.0;
        if (normalized < 0.0) normalized += 360.0;
        return normalized - 180.0;
    }

    public static float clamp(float value, float minimum, float maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    public static void requireFinite(double[] values, String label) {
        for (double value : values) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(label + " must be finite");
            }
        }
    }

    public static void requireNavigationCapacity(
        int value,
        int minimum,
        int maximum,
        String label
    ) {
        if (value < minimum || value > maximum) {
            throw new IllegalArgumentException(
                label + " must be in [" + minimum + ", " + maximum + "]"
            );
        }
    }

    public static Vector3d[] navigationSearchDirections(
        MotionController controller
    ) {
        Vector3d selector = controller.getComponentSelector();
        boolean twoDimensional = controller.is2D();
        boolean projectedX = twoDimensional && selector.x == 0.0D;
        boolean projectedY = twoDimensional && selector.y == 0.0D;
        boolean projectedZ = twoDimensional && selector.z == 0.0D;
        List<Vector3d> result = new ArrayList<>(twoDimensional ? 8 : 26);
        for (int x = -1; x <= 1; x++) {
            if (projectedX && x != 0) {
                continue;
            }
            for (int y = -1; y <= 1; y++) {
                if (projectedY && y != 0) {
                    continue;
                }
                for (int z = -1; z <= 1; z++) {
                    if (
                        (projectedZ && z != 0)
                            || (x == 0 && y == 0 && z == 0)
                    ) {
                        continue;
                    }
                    result.add(new Vector3d(x, y, z));
                }
            }
        }
        return result.toArray(Vector3d[]::new);
    }
}
