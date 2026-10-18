package com.hytalerlbridge.geometry;

import com.hytalerlbridge.world.BlockType;
import java.util.ArrayList;
import java.util.List;

/**
 * One complete geometry/support/contact observation.
 *
 * <p>{@code available=false} is an explicit capability failure, never an
 * implicit all-air world. {@code exactCollisionShapes} distinguishes native
 * engine-derived geometry from the legacy unit-cube simulator fallback.</p>
 */
public record GeometryFrame(
    boolean available,
    boolean exactCollisionShapes,
    int originX,
    int originY,
    int originZ,
    List<GeometryCell> cells,
    double[] agentBounds,
    double[] targetBounds,
    double[] agentLosOffset,
    double[] targetLosOffset,
    List<GeometryContact> contacts,
    boolean grounded,
    boolean ceilingContact,
    boolean targetLineOfSight,
    boolean targetLineOfSightValid
) {
    public GeometryFrame {
        cells = cells == null ? List.of() : List.copyOf(cells);
        contacts = contacts == null ? List.of() : List.copyOf(contacts);
        agentBounds = agentBounds == null ? new double[6] : agentBounds.clone();
        targetBounds = targetBounds == null ? new double[6] : targetBounds.clone();
        agentLosOffset = agentLosOffset == null
            ? new double[3]
            : agentLosOffset.clone();
        targetLosOffset = targetLosOffset == null
            ? new double[3]
            : targetLosOffset.clone();
        if (agentBounds.length != 6 || targetBounds.length != 6) {
            throw new IllegalArgumentException(
                "agentBounds and targetBounds must each contain six values"
            );
        }
        if (agentLosOffset.length != 3 || targetLosOffset.length != 3) {
            throw new IllegalArgumentException(
                "LOS offsets must each contain three values"
            );
        }
    }

    @Override
    public double[] agentBounds() {
        return agentBounds.clone();
    }

    @Override
    public double[] targetBounds() {
        return targetBounds.clone();
    }

    @Override
    public double[] agentLosOffset() {
        return agentLosOffset.clone();
    }

    @Override
    public double[] targetLosOffset() {
        return targetLosOffset.clone();
    }

    public static GeometryFrame empty() {
        return new GeometryFrame(
            false,
            false,
            0,
            0,
            0,
            List.of(),
            new double[6],
            new double[6],
            new double[3],
            new double[3],
            List.of(),
            false,
            false,
            false,
            false
        );
    }

    /**
     * Backend-neutral fallback for the old sparse voxel simulator.
     *
     * <p>Its solid blocks are explicit unit cubes and are deliberately marked
     * non-exact so they cannot be mistaken for native 0.5.7 asset hitboxes.</p>
     */
    public static GeometryFrame fromSimulatorBlocks(
        double x,
        double y,
        double z,
        List<int[]> nearbyBlocks
    ) {
        int originX = (int) Math.floor(x);
        int originY = (int) Math.floor(y);
        int originZ = (int) Math.floor(z);
        List<GeometryCell> cells = new ArrayList<>();
        if (nearbyBlocks != null) {
            for (int[] encoded : nearbyBlocks) {
                if (encoded == null || encoded.length != 4) continue;
                int dx = encoded[0];
                int dy = encoded[1];
                int dz = encoded[2];
                if (Math.abs(dx) > GeometryContract.RADIUS
                    || Math.abs(dy) > GeometryContract.RADIUS
                    || Math.abs(dz) > GeometryContract.RADIUS) {
                    continue;
                }
                BlockType type = BlockType.fromId(encoded[3]);
                if (type == BlockType.AIR) continue;
                boolean solid = type.isSolid();
                int flags = 0;
                if (solid) flags |= GeometryContract.FLAG_SOLID;
                if (solid && type != BlockType.GLASS && type != BlockType.LEAVES) {
                    flags |= GeometryContract.FLAG_OPAQUE;
                }
                if (type == BlockType.WATER) flags |= GeometryContract.FLAG_FLUID;
                double[] boxes = solid
                    ? new double[] {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}
                    : new double[0];
                cells.add(new GeometryCell(
                    dx,
                    dy,
                    dz,
                    type.id(),
                    type == BlockType.WATER ? type.id() : 0,
                    type.id(),
                    0,
                    flags,
                    type == BlockType.WATER ? 8 : 0,
                    type == BlockType.WATER ? 1.0 : 0.0,
                    solid ? 255 : 0,
                    0,
                    0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    type == BlockType.WATER ? 1.0 : 0.0,
                    type == BlockType.WATER ? 1.0 : 0.0,
                    type == BlockType.WATER ? 1.0 : 0.0,
                    boxes
                ));
            }
        }
        return new GeometryFrame(
            true,
            false,
            originX,
            originY,
            originZ,
            cells,
            new double[6],
            new double[6],
            new double[3],
            new double[3],
            List.of(),
            false,
            false,
            false,
            false
        );
    }
}
