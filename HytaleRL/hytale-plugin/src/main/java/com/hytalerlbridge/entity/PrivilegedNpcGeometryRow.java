package com.hytalerlbridge.entity;

/**
 * Authoritative runtime geometry aligned with one privileged NPC spatial row.
 *
 * <p>The bounds are the entity's current local, rotation-adjusted
 * {@code BoundingBox}. The LOS offset follows {@code PositionCache}: model eye
 * height when a model is present, otherwise the bounding-box centre. Entity
 * display scale is diagnostic evidence and is never applied to the physical
 * bounds a second time.</p>
 */
public record PrivilegedNpcGeometryRow(
    PrivilegedEntityRow spatial,
    boolean geometrySupported,
    double minimumX,
    double minimumY,
    double minimumZ,
    double maximumX,
    double maximumY,
    double maximumZ,
    boolean collidable,
    boolean blocksLineOfSight,
    boolean lineOfSightOffsetSupported,
    double lineOfSightOffsetX,
    double lineOfSightOffsetY,
    double lineOfSightOffsetZ,
    boolean modelPresent,
    String modelAssetId,
    double modelScale,
    double modelEyeHeight,
    boolean entityScalePresent,
    double entityScale
) {
    public PrivilegedNpcGeometryRow {
        if (spatial == null || spatial.typeAssetId().isBlank()) {
            throw new IllegalArgumentException(
                "NPC spatial row and role asset ID must be present"
            );
        }
        if (modelAssetId == null) {
            throw new IllegalArgumentException("modelAssetId must not be null");
        }
        requireFinite(minimumX, "minimumX");
        requireFinite(minimumY, "minimumY");
        requireFinite(minimumZ, "minimumZ");
        requireFinite(maximumX, "maximumX");
        requireFinite(maximumY, "maximumY");
        requireFinite(maximumZ, "maximumZ");
        requireFinite(lineOfSightOffsetX, "lineOfSightOffsetX");
        requireFinite(lineOfSightOffsetY, "lineOfSightOffsetY");
        requireFinite(lineOfSightOffsetZ, "lineOfSightOffsetZ");
        requireFinite(modelScale, "modelScale");
        requireFinite(modelEyeHeight, "modelEyeHeight");
        requireFinite(entityScale, "entityScale");
        if (
            geometrySupported
                && (minimumX >= maximumX
                    || minimumY >= maximumY
                    || minimumZ >= maximumZ)
        ) {
            throw new IllegalArgumentException(
                "Supported NPC geometry must have positive extent"
            );
        }
        if (!geometrySupported && (collidable || blocksLineOfSight)) {
            throw new IllegalArgumentException(
                "Unsupported NPC geometry cannot publish physical flags"
            );
        }
        if (!lineOfSightOffsetSupported && (
            lineOfSightOffsetX != 0.0
                || lineOfSightOffsetY != 0.0
                || lineOfSightOffsetZ != 0.0
        )) {
            throw new IllegalArgumentException(
                "Unsupported NPC LOS offset must be zero"
            );
        }
        if (modelPresent) {
            if (modelScale <= 0.0) {
                throw new IllegalArgumentException(
                    "Present NPC model scale must be positive"
                );
            }
        } else if (
            !modelAssetId.isEmpty()
                || modelScale != 0.0
                || modelEyeHeight != 0.0
        ) {
            throw new IllegalArgumentException(
                "Absent NPC model has contradictory evidence"
            );
        }
        if (entityScalePresent ? entityScale <= 0.0 : entityScale != 0.0) {
            throw new IllegalArgumentException(
                "NPC display-scale evidence is contradictory"
            );
        }
    }

    private static void requireFinite(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
    }
}
