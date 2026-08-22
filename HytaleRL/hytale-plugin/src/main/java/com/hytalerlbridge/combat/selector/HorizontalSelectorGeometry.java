package com.hytalerlbridge.combat.selector;

import com.hytalerlbridge.combat.CombatPhase;
import com.hytalerlbridge.combat.MeleeAttackProfile;
import java.util.ArrayList;
import java.util.List;

/**
 * Fixed-shape reproduction of Hytale 0.5.7's horizontal melee selector.
 *
 * <p>The server advances a zero-width selector sample at time zero, then
 * creates a truncated three-dimensional frustum for each subsequent sample.
 * The projection is relative to the attacker's head rotation, including the
 * authored pitch and roll. Hytale's hit executor clips entity-box surfaces;
 * the SAT below is equivalent for an AABB except when the frustum is wholly
 * contained by the box, which is handled explicitly.</p>
 */
public final class HorizontalSelectorGeometry {

    private static final double AXIS_EPSILON = 1.0e-8;

    private HorizontalSelectorGeometry() {}

    /**
     * Test one authored selector sample against a world-space entity AABB.
     *
     * @param profile authored selector and attack-chain data
     * @param elapsedTicks one-based elapsed attack-chain ticks
     * @param nominalDeltaSeconds logical interaction tick duration
     * @param selectorPi float-precision PI constant used by server projection
     * @param origin selector origin (normally the attacker's effective eye)
     * @param headYawDegrees absolute Hytale head yaw
     * @param headPitchDegrees absolute Hytale head pitch
     * @param bounds world-space min/max AABB in x/y/z order
     */
    public static boolean intersectsAabb(
        MeleeAttackProfile profile,
        int elapsedTicks,
        double nominalDeltaSeconds,
        double selectorPi,
        double[] origin,
        double headYawDegrees,
        double headPitchDegrees,
        double[] bounds
    ) {
        requireVector(origin, 3, "origin");
        requireBounds(bounds);
        if (profile.phaseAt(elapsedTicks) != CombatPhase.SWEEP) return false;
        if (!Double.isFinite(nominalDeltaSeconds)
            || nominalDeltaSeconds <= 0.0
            || !Double.isFinite(selectorPi)
            || selectorPi <= 0.0
            || !Double.isFinite(headYawDegrees)
            || !Double.isFinite(headPitchDegrees)) {
            throw new IllegalArgumentException("selector timing/rotation is invalid");
        }

        int sample = profile.selectorSampleAt(elapsedTicks);
        double currentTime = sample * nominalDeltaSeconds;
        double previousTime = Math.max(sample - 1, 0) * nominalDeltaSeconds;
        double deltaPercentage =
            (currentTime - previousTime) / profile.selectorRuntimeSeconds();
        if (deltaPercentage <= 0.0) return false;
        double priorPercentage =
            previousTime / profile.selectorRuntimeSeconds();
        double arcRadians = Math.toRadians(profile.arcDegrees());
        double yawDelta = arcRadians * deltaPercentage;
        double priorYaw = arcRadians * priorPercentage;
        double directionModifier = switch (profile.direction()) {
            case LEFT -> 1.0;
            case RIGHT -> -1.0;
            case CENTER -> 0.0;
        };
        double selectorYaw = (
            priorYaw
                + yawDelta
                + Math.toRadians(profile.yawStartOffsetDegrees())
        ) * directionModifier;

        double near = profile.startDistance();
        double far = profile.endDistance();
        double nearFarRatio = near / far;
        double horizontalFar = 2.0 * far * yawDelta / selectorPi;
        double horizontalNear = horizontalFar * nearFarRatio;
        double bottomNear = profile.extendBottom() * nearFarRatio;
        double topNear = profile.extendTop() * nearFarRatio;
        double bottomFar = profile.extendBottom();
        double topFar = profile.extendTop();

        double[][] selectorRotation = multiply(
            rotationX(-Math.toRadians(profile.pitchOffsetDegrees())),
            multiply(
                rotationY(-selectorYaw),
                rotationZ(-Math.toRadians(profile.rollOffsetDegrees()))
            )
        );
        double headYaw = Math.toRadians(headYawDegrees);
        double headPitch = Math.toRadians(headPitchDegrees);
        double cosPitch = Math.cos(headPitch);
        double[] forward = new double[] {
            -Math.sin(headYaw) * cosPitch,
            Math.sin(headPitch),
            -Math.cos(headYaw) * cosPitch
        };
        double[] right = normalize(cross(forward, new double[] {0.0, 1.0, 0.0}));
        double[] cameraUp = cross(right, forward);
        double[][] cameraRotation = new double[][] {
            right,
            cameraUp,
            scale(forward, -1.0)
        };
        double[][] worldToSelector = multiply(
            selectorRotation,
            cameraRotation
        );

        double[][] boxCornersWorld = corners(bounds);
        double[][] boxCorners = new double[boxCornersWorld.length][3];
        for (int index = 0; index < boxCornersWorld.length; index++) {
            boxCorners[index] = transform(
                worldToSelector,
                subtract(boxCornersWorld[index], origin)
            );
        }
        double[][] frustumCorners = new double[][] {
            {-horizontalNear, -bottomNear, -near},
            { horizontalNear, -bottomNear, -near},
            {-horizontalNear,  topNear, -near},
            { horizontalNear,  topNear, -near},
            {-horizontalFar,  -bottomFar,  -far},
            { horizontalFar,  -bottomFar,  -far},
            {-horizontalFar,   topFar,     -far},
            { horizontalFar,   topFar,     -far}
        };

        List<double[]> axes = new ArrayList<>(27);
        int[][] faces = new int[][] {
            {0, 1, 3},
            {4, 6, 7},
            {0, 2, 6},
            {1, 5, 7},
            {0, 4, 5},
            {2, 3, 7}
        };
        for (int[] face : faces) {
            axes.add(cross(
                subtract(frustumCorners[face[1]], frustumCorners[face[0]]),
                subtract(frustumCorners[face[2]], frustumCorners[face[0]])
            ));
        }

        double[][] boxAxes = transpose(worldToSelector);
        for (double[] axis : boxAxes) axes.add(axis);
        double[][] frustumEdges = new double[][] {
            subtract(frustumCorners[1], frustumCorners[0]),
            subtract(frustumCorners[2], frustumCorners[0]),
            subtract(frustumCorners[4], frustumCorners[0]),
            subtract(frustumCorners[5], frustumCorners[1]),
            subtract(frustumCorners[6], frustumCorners[2]),
            subtract(frustumCorners[7], frustumCorners[3])
        };
        for (double[] boxAxis : boxAxes) {
            for (double[] edge : frustumEdges) {
                axes.add(cross(boxAxis, edge));
            }
        }
        for (double[] axis : axes) {
            double norm = norm(axis);
            if (norm <= AXIS_EPSILON) continue;
            double[] normalized = scale(axis, 1.0 / norm);
            double[] boxProjection = projection(boxCorners, normalized);
            double[] frustumProjection = projection(
                frustumCorners,
                normalized
            );
            if (boxProjection[1] < frustumProjection[0]
                || frustumProjection[1] < boxProjection[0]) {
                return false;
            }
        }

        // HitDetectionExecutor clips the entity cube's surfaces. A selector
        // fully inside the cube intersects its volume but no clipped surface.
        double[][] selectorToWorld = transpose(worldToSelector);
        boolean whollyInside = true;
        for (double[] corner : frustumCorners) {
            double[] world = add(origin, transform(selectorToWorld, corner));
            if (world[0] < bounds[0] || world[0] > bounds[3]
                || world[1] < bounds[1] || world[1] > bounds[4]
                || world[2] < bounds[2] || world[2] > bounds[5]) {
                whollyInside = false;
                break;
            }
        }
        return !whollyInside;
    }

    private static double[][] corners(double[] bounds) {
        double[][] result = new double[8][3];
        for (int index = 0; index < result.length; index++) {
            result[index][0] = (index & 1) == 0 ? bounds[0] : bounds[3];
            result[index][1] = (index & 2) == 0 ? bounds[1] : bounds[4];
            result[index][2] = (index & 4) == 0 ? bounds[2] : bounds[5];
        }
        return result;
    }

    private static double[] projection(double[][] points, double[] axis) {
        double minimum = Double.POSITIVE_INFINITY;
        double maximum = Double.NEGATIVE_INFINITY;
        for (double[] point : points) {
            double projection = dot(point, axis);
            minimum = Math.min(minimum, projection);
            maximum = Math.max(maximum, projection);
        }
        return new double[] {minimum, maximum};
    }

    private static double[][] rotationX(double angle) {
        double cosine = Math.cos(angle);
        double sine = Math.sin(angle);
        return new double[][] {
            {1.0, 0.0, 0.0},
            {0.0, cosine, -sine},
            {0.0, sine, cosine}
        };
    }

    private static double[][] rotationY(double angle) {
        double cosine = Math.cos(angle);
        double sine = Math.sin(angle);
        return new double[][] {
            {cosine, 0.0, sine},
            {0.0, 1.0, 0.0},
            {-sine, 0.0, cosine}
        };
    }

    private static double[][] rotationZ(double angle) {
        double cosine = Math.cos(angle);
        double sine = Math.sin(angle);
        return new double[][] {
            {cosine, -sine, 0.0},
            {sine, cosine, 0.0},
            {0.0, 0.0, 1.0}
        };
    }

    private static double[][] multiply(double[][] left, double[][] right) {
        double[][] result = new double[3][3];
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                for (int inner = 0; inner < 3; inner++) {
                    result[row][column] +=
                        left[row][inner] * right[inner][column];
                }
            }
        }
        return result;
    }

    private static double[][] transpose(double[][] matrix) {
        double[][] result = new double[3][3];
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                result[row][column] = matrix[column][row];
            }
        }
        return result;
    }

    private static double[] transform(double[][] matrix, double[] vector) {
        return new double[] {
            dot(matrix[0], vector),
            dot(matrix[1], vector),
            dot(matrix[2], vector)
        };
    }

    private static double[] normalize(double[] vector) {
        double length = norm(vector);
        if (length <= AXIS_EPSILON) {
            throw new IllegalArgumentException("rotation basis is degenerate");
        }
        return scale(vector, 1.0 / length);
    }

    private static double norm(double[] vector) {
        return Math.sqrt(dot(vector, vector));
    }

    private static double dot(double[] left, double[] right) {
        return left[0] * right[0]
            + left[1] * right[1]
            + left[2] * right[2];
    }

    private static double[] cross(double[] left, double[] right) {
        return new double[] {
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0]
        };
    }

    private static double[] add(double[] left, double[] right) {
        return new double[] {
            left[0] + right[0],
            left[1] + right[1],
            left[2] + right[2]
        };
    }

    private static double[] subtract(double[] left, double[] right) {
        return new double[] {
            left[0] - right[0],
            left[1] - right[1],
            left[2] - right[2]
        };
    }

    private static double[] scale(double[] vector, double scalar) {
        return new double[] {
            vector[0] * scalar,
            vector[1] * scalar,
            vector[2] * scalar
        };
    }

    private static void requireVector(
        double[] values,
        int length,
        String name
    ) {
        if (values == null || values.length != length) {
            throw new IllegalArgumentException(
                name + " must contain " + length + " values"
            );
        }
        for (double value : values) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(name + " must be finite");
            }
        }
    }

    private static void requireBounds(double[] bounds) {
        requireVector(bounds, 6, "bounds");
        for (int axis = 0; axis < 3; axis++) {
            if (bounds[axis + 3] <= bounds[axis]) {
                throw new IllegalArgumentException(
                    "bounds maximum must exceed minimum on every axis"
                );
            }
        }
    }
}
