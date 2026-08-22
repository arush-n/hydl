package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.server.core.util.FillerBlockUtil;
import com.hytalerlbridge.geometry.trigger.TriggerTransportCompiler.Roots;
import com.hytalerlbridge.geometry.trigger.TriggerTransportCompiler.SourceCell;
import java.util.Map;

/** Stateless projection of native trigger roots and collision volumes. */
public final class NativeTriggerCapture {

    private static final double FLUID_SURFACE_EPSILON = 0.03125;

    private NativeTriggerCapture() {}

    public static SourceCell sourceCell(
        int dx,
        int dy,
        int dz,
        int runtimeBlockId,
        int filler,
        double fluidFillHeight,
        Map<InteractionType, String> blockInteractions,
        Map<InteractionType, String> fluidInteractions,
        double[] blockHitboxBoxes
    ) {
        Roots blockRoots = roots(blockInteractions);
        Roots fluidRoots = roots(fluidInteractions);
        Roots resolved = Roots.resolve(blockRoots, fluidRoots);
        int baseOffsetX = filler == 0
            ? 0
            : -FillerBlockUtil.unpackX(filler);
        int baseOffsetY = filler == 0
            ? 0
            : -FillerBlockUtil.unpackY(filler);
        int baseOffsetZ = filler == 0
            ? 0
            : -FillerBlockUtil.unpackZ(filler);
        if (!resolved.hasAny()) {
            return new SourceCell(
                dx, dy, dz,
                baseOffsetX, baseOffsetY, baseOffsetZ,
                true,
                blockRoots,
                fluidRoots,
                new double[0]
            );
        }

        double[] boxes;
        boolean volumeAvailable;
        if (runtimeBlockId == 0) {
            if (!Double.isFinite(fluidFillHeight)
                || fluidFillHeight < 0.0
                || fluidFillHeight > 1.0) {
                throw new IllegalArgumentException(
                    "fluid trigger fill must be finite and in [0, 1]"
                );
            }
            boxes = new double[] {
                0.0,
                0.0,
                0.0,
                1.0,
                (1.0 - FLUID_SURFACE_EPSILON) * fluidFillHeight,
                1.0
            };
            volumeAvailable = true;
        } else {
            volumeAvailable = blockHitboxBoxes != null
                && blockHitboxBoxes.length != 0;
            boxes = volumeAvailable ? blockHitboxBoxes : new double[0];
        }
        return new SourceCell(
            dx, dy, dz,
            baseOffsetX, baseOffsetY, baseOffsetZ,
            volumeAvailable,
            blockRoots,
            fluidRoots,
            boxes
        );
    }

    private static Roots roots(Map<InteractionType, String> interactions) {
        if (interactions == null || interactions.isEmpty()) return Roots.none();
        return new Roots(
            interactions.get(InteractionType.CollisionEnter),
            interactions.get(InteractionType.Collision),
            interactions.get(InteractionType.CollisionLeave)
        );
    }
}
