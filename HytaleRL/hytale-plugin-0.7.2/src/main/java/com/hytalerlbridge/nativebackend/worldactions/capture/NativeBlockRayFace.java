package com.hytalerlbridge.nativebackend.worldactions.capture;

import com.hypixel.hytale.math.iterator.BlockIterator;
import org.joml.Vector3d;
import org.joml.Vector3i;

/** Derives protocol face from the exact voxel transitions of BlockIterator. */
public final class NativeBlockRayFace {

    private NativeBlockRayFace() {}

    public static int faceForTarget(
        Vector3d origin,
        Vector3d direction,
        double maximumDistance,
        Vector3i target
    ) {
        if (
            origin == null
                || direction == null
                || target == null
                || !Double.isFinite(maximumDistance)
                || maximumDistance <= 0.0
                || !finite(origin)
                || !finite(direction)
                || direction.lengthSquared() <= 0.0
        ) {
            return 0;
        }
        Cursor cursor = new Cursor(target);
        BlockIterator.iterate(
            origin.x,
            origin.y,
            origin.z,
            direction.x,
            direction.y,
            direction.z,
            maximumDistance,
            (x, y, z, _px, _py, _pz, _qx, _qy, _qz) ->
                cursor.visit(x, y, z)
        );
        return cursor.face;
    }

    private static boolean finite(Vector3d value) {
        return Double.isFinite(value.x)
            && Double.isFinite(value.y)
            && Double.isFinite(value.z);
    }

    private static final class Cursor {
        private final Vector3i target;
        private boolean previousPresent;
        private int previousX;
        private int previousY;
        private int previousZ;
        private int face;

        private Cursor(Vector3i target) {
            this.target = target;
        }

        private boolean visit(int x, int y, int z) {
            if (x == target.x && y == target.y && z == target.z) {
                face = previousPresent
                    ? faceFromDelta(
                        x - previousX,
                        y - previousY,
                        z - previousZ
                    )
                    : 0;
                return false;
            }
            previousPresent = true;
            previousX = x;
            previousY = y;
            previousZ = z;
            return true;
        }
    }

    static int faceFromDelta(int dx, int dy, int dz) {
        if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) != 1) return 0;
        if (dx == 1) return 6;  // entered +X cell through West
        if (dx == -1) return 5; // entered -X cell through East
        if (dy == 1) return 2;  // entered +Y cell through Down
        if (dy == -1) return 1; // entered -Y cell through Up
        if (dz == 1) return 3;  // entered +Z cell through North
        if (dz == -1) return 4; // entered -Z cell through South
        return 0;
    }
}
