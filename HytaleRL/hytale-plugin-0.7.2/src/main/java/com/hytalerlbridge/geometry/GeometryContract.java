package com.hytalerlbridge.geometry;

/**
 * Fixed, model-agnostic geometry contract shared by bridge backends.
 *
 * <p>The policy-facing grid is always a 9x9x9 cube anchored at the floored
 * agent position. Native block asset identifiers remain diagnostic only;
 * collision boxes and semantic flags are carried explicitly so consumers do
 * not need to guess physics from an unstable asset index.</p>
 */
public final class GeometryContract {

    public static final String SCHEMA = "hytale_geometry_v5";
    public static final int VERSION = 5;
    public static final int RADIUS = 4;
    public static final int SIDE = RADIUS * 2 + 1;
    public static final int CELL_COUNT = SIDE * SIDE * SIDE;

    /** Maximum detail-box count present in every bundled Hytale 0.5.7 asset. */
    public static final int MAX_DETAIL_BOXES_0_5_7 = 9;

    /**
     * Fixed contact capacity used by the Python/JAX representation.
     *
     * <p>The wire representation is not truncated. A client must reject a
     * frame exceeding this capacity instead of silently losing contacts.</p>
     */
    public static final int MAX_CONTACTS = 64;

    public static final int FLAG_SOLID = 1;
    /**
     * The block stops the native NPC base-opacity line-of-sight predicate.
     *
     * <p>This is intentionally broader than {@code Opacity.Solid}: native
     * 0.5.7 blocks LOS for every non-transparent opacity. The role's effective
     * BlockSet is a separate predicate.</p>
     */
    public static final int FLAG_OPAQUE = 1 << 1;
    public static final int FLAG_FLUID = 1 << 2;
    public static final int FLAG_DAMAGING = 1 << 3;
    public static final int FLAG_CLIMBABLE = 1 << 4;
    public static final int FLAG_BOUNCY = 1 << 5;
    public static final int FLAG_TRIGGER = 1 << 6;
    public static final int FLAG_PROTRUDES_CELL = 1 << 7;
    public static final int FLAG_HAS_MOVEMENT_SETTINGS = 1 << 8;
    public static final int FLAG_HAS_FLUID_MOVEMENT_SETTINGS = 1 << 9;

    private GeometryContract() {}

    public static int cellIndex(int dx, int dy, int dz) {
        if (Math.abs(dx) > RADIUS
            || Math.abs(dy) > RADIUS
            || Math.abs(dz) > RADIUS) {
            throw new IllegalArgumentException(
                "Geometry offset is outside radius " + RADIUS
            );
        }
        return (dx + RADIUS) * SIDE * SIDE
            + (dy + RADIUS) * SIDE
            + (dz + RADIUS);
    }
}
