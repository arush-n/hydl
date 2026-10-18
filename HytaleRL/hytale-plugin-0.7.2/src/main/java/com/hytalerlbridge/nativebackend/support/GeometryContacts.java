package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.shape.Box;
import com.hypixel.hytale.protocol.Opacity;
import com.hypixel.hytale.server.core.asset.type.blockhitbox.BlockBoundingBoxes;
import com.hypixel.hytale.server.core.asset.type.blocktype.config.BlockType;
import com.hypixel.hytale.server.core.modules.entity.component.BoundingBox;
import com.hypixel.hytale.server.core.modules.entity.component.ModelComponent;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.geometry.GeometryCell;
import com.hytalerlbridge.geometry.GeometryContact;
import com.hytalerlbridge.geometry.GeometryContract;
import java.util.List;
import org.joml.Vector3d;

/**
 * Extracted verbatim from {@code NativeEnvironmentSession}.
 *
 * <p>Every method here touches no session instance field, so the move
 * required no state surgery.
 */
public final class GeometryContacts {

    private GeometryContacts() {}

    public static void appendTouchingGeometryContacts(
        List<GeometryCell> cells,
        Vector3d position,
        Box agentBox,
        int originX,
        int originY,
        int originZ,
        List<GeometryContact> contacts
    ) {
        double agentMinX = position.x + agentBox.min.x;
        double agentMinY = position.y + agentBox.min.y;
        double agentMinZ = position.z + agentBox.min.z;
        double agentMaxX = position.x + agentBox.max.x;
        double agentMaxY = position.y + agentBox.max.y;
        double agentMaxZ = position.z + agentBox.max.z;
        double tolerance = 1.0e-3;

        for (GeometryCell cell : cells) {
            if ((cell.flags() & GeometryContract.FLAG_SOLID) == 0) continue;
            double[] boxes = cell.collisionBoxes();
            for (int detail = 0; detail < boxes.length / 6; detail++) {
                int offset = detail * 6;
                double boxMinX = originX + cell.dx() + boxes[offset];
                double boxMinY = originY + cell.dy() + boxes[offset + 1];
                double boxMinZ = originZ + cell.dz() + boxes[offset + 2];
                double boxMaxX = originX + cell.dx() + boxes[offset + 3];
                double boxMaxY = originY + cell.dy() + boxes[offset + 4];
                double boxMaxZ = originZ + cell.dz() + boxes[offset + 5];
                double overlapX = Math.min(agentMaxX, boxMaxX)
                    - Math.max(agentMinX, boxMinX);
                double overlapY = Math.min(agentMaxY, boxMaxY)
                    - Math.max(agentMinY, boxMinY);
                double overlapZ = Math.min(agentMaxZ, boxMaxZ)
                    - Math.max(agentMinZ, boxMinZ);

                if (overlapY > tolerance && overlapZ > tolerance) {
                    if (Math.abs(agentMinX - boxMaxX) <= tolerance) {
                        contacts.add(touchingContact(
                            cell, detail, 1, 0, 0,
                            boxMaxX,
                            overlapMidpoint(agentMinY, agentMaxY, boxMinY, boxMaxY),
                            overlapMidpoint(agentMinZ, agentMaxZ, boxMinZ, boxMaxZ),
                            originX, originY, originZ
                        ));
                    }
                    if (Math.abs(agentMaxX - boxMinX) <= tolerance) {
                        contacts.add(touchingContact(
                            cell, detail, -1, 0, 0,
                            boxMinX,
                            overlapMidpoint(agentMinY, agentMaxY, boxMinY, boxMaxY),
                            overlapMidpoint(agentMinZ, agentMaxZ, boxMinZ, boxMaxZ),
                            originX, originY, originZ
                        ));
                    }
                }
                if (overlapX > tolerance && overlapZ > tolerance) {
                    if (Math.abs(agentMinY - boxMaxY) <= tolerance) {
                        contacts.add(touchingContact(
                            cell, detail, 0, 1, 0,
                            overlapMidpoint(agentMinX, agentMaxX, boxMinX, boxMaxX),
                            boxMaxY,
                            overlapMidpoint(agentMinZ, agentMaxZ, boxMinZ, boxMaxZ),
                            originX, originY, originZ
                        ));
                    }
                    if (Math.abs(agentMaxY - boxMinY) <= tolerance) {
                        contacts.add(touchingContact(
                            cell, detail, 0, -1, 0,
                            overlapMidpoint(agentMinX, agentMaxX, boxMinX, boxMaxX),
                            boxMinY,
                            overlapMidpoint(agentMinZ, agentMaxZ, boxMinZ, boxMaxZ),
                            originX, originY, originZ
                        ));
                    }
                }
                if (overlapX > tolerance && overlapY > tolerance) {
                    if (Math.abs(agentMinZ - boxMaxZ) <= tolerance) {
                        contacts.add(touchingContact(
                            cell, detail, 0, 0, 1,
                            overlapMidpoint(agentMinX, agentMaxX, boxMinX, boxMaxX),
                            overlapMidpoint(agentMinY, agentMaxY, boxMinY, boxMaxY),
                            boxMaxZ,
                            originX, originY, originZ
                        ));
                    }
                    if (Math.abs(agentMaxZ - boxMinZ) <= tolerance) {
                        contacts.add(touchingContact(
                            cell, detail, 0, 0, -1,
                            overlapMidpoint(agentMinX, agentMaxX, boxMinX, boxMaxX),
                            overlapMidpoint(agentMinY, agentMaxY, boxMinY, boxMaxY),
                            boxMinZ,
                            originX, originY, originZ
                        ));
                    }
                }
            }
        }
    }

    public static GeometryContact touchingContact(
        GeometryCell cell,
        int detailBoxIndex,
        double normalX,
        double normalY,
        double normalZ,
        double pointX,
        double pointY,
        double pointZ,
        int originX,
        int originY,
        int originZ
    ) {
        return new GeometryContact(
            cell.dx(),
            cell.dy(),
            cell.dz(),
            detailBoxIndex,
            normalX,
            normalY,
            normalZ,
            pointX - originX,
            pointY - originY,
            pointZ - originZ,
            0.0,
            0.0,
            true,
            false
        );
    }

    public static double[] collisionBoxes(
        BlockBoundingBoxes hitboxAsset,
        int rotationIndex
    ) {
        BlockBoundingBoxes.RotatedVariantBoxes variant = hitboxAsset.get(
            Math.max(0, rotationIndex)
        );
        Box[] boxes = variant.hasDetailBoxes()
            ? variant.getDetailBoxes()
            : new Box[] {variant.getBoundingBox()};
        if (boxes == null || boxes.length == 0) return new double[0];
        if (boxes.length > GeometryContract.MAX_DETAIL_BOXES_0_5_7) {
            throw new IllegalStateException(
                "Runtime block hitbox exceeds the certified Hytale 0.5.7 capacity: "
                    + boxes.length
            );
        }
        double[] flattened = new double[boxes.length * 6];
        int offset = 0;
        for (Box box : boxes) {
            flattened[offset++] = box.min.x;
            flattened[offset++] = box.min.y;
            flattened[offset++] = box.min.z;
            flattened[offset++] = box.max.x;
            flattened[offset++] = box.max.y;
            flattened[offset++] = box.max.z;
        }
        return flattened;
    }

    public static double overlapMidpoint(
        double firstMin,
        double firstMax,
        double secondMin,
        double secondMax
    ) {
        return (
            Math.max(firstMin, secondMin)
                + Math.min(firstMax, secondMax)
        ) * 0.5;
    }

    /** Reproduce PositionCache.testLineOfSightRays endpoint construction. */
    public static double[] lineOfSightOffset(
        Store<EntityStore> store,
        Ref<EntityStore> reference,
        boolean source
    ) {
        if (reference == null || !reference.isValid()) return new double[3];
        ModelComponent model = store.getComponent(
            reference,
            ModelComponent.getComponentType()
        );
        if (model != null && model.getModel() != null) {
            return new double[] {0.0, model.getModel().getEyeHeight(), 0.0};
        }
        if (source) return new double[3];
        BoundingBox bounds = store.getComponent(
            reference,
            BoundingBox.getComponentType()
        );
        Box box = bounds == null ? null : bounds.getBoundingBox();
        return box == null
            ? new double[3]
            : new double[] {
                (box.min.x + box.max.x) * 0.5,
                (box.min.y + box.max.y) * 0.5,
                (box.min.z + box.max.z) * 0.5
            };
    }

    /**
     * Mirrors PositionCache's native 0.5.7 base opacity predicate.
     *
     * <p>The effective role predicate also applies its BlockSet. BuilderRole
     * defaults that set to {@code Opaque}, whose installed asset selects
     * {@code *_Window}. The exact engine LOS label remains in every frame as
     * the final role-specific oracle.</p>
     */
    public static boolean blocksDefaultNpcLineOfSight(
        int runtimeBlockId,
        BlockType blockType
    ) {
        if (runtimeBlockId == 0) return false;
        return blockType == null
            || blockType == BlockType.UNKNOWN
            || blockType.getOpacity() == null
            || blockType.getOpacity() != Opacity.Transparent;
    }

    public static boolean isDiagonalLosFixture(String fixture) {
        return fixture.equals("los_diagonal_graze")
            || fixture.equals("los_diagonal_block");
    }
}
