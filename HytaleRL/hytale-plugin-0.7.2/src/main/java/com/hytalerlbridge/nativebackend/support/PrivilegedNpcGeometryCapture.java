package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.shape.Box;
import com.hypixel.hytale.server.core.asset.type.model.config.Model;
import com.hypixel.hytale.server.core.modules.entity.component.BoundingBox;
import com.hypixel.hytale.server.core.modules.entity.component.EntityScaleComponent;
import com.hypixel.hytale.server.core.modules.entity.component.Intangible;
import com.hypixel.hytale.server.core.modules.entity.component.ModelComponent;
import com.hypixel.hytale.server.core.modules.entity.damage.DeathComponent;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.entity.PrivilegedEntityRow;
import com.hytalerlbridge.entity.PrivilegedNpcGeometryRow;

/** Exact, allocation-light runtime geometry capture for privileged NPC rows. */
public final class PrivilegedNpcGeometryCapture {

    private PrivilegedNpcGeometryCapture() {}

    public static PrivilegedNpcGeometryRow capture(
        Store<EntityStore> store,
        Ref<EntityStore> reference,
        PrivilegedEntityRow spatial
    ) {
        BoundingBox boundingBoxComponent = store.getComponent(
            reference,
            BoundingBox.getComponentType()
        );
        Box bounds = boundingBoxComponent == null
            ? null
            : boundingBoxComponent.getBoundingBox();
        boolean geometrySupported = validBounds(bounds);

        ModelComponent modelComponent = store.getComponent(
            reference,
            ModelComponent.getComponentType()
        );
        Model model = modelComponent == null ? null : modelComponent.getModel();
        boolean modelPresent = model != null;
        String modelAssetId = !modelPresent || model.getModelAssetId() == null
            ? ""
            : model.getModelAssetId();
        double modelScale = modelPresent ? model.getScale() : 0.0;
        double modelEyeHeight = modelPresent ? model.getEyeHeight() : 0.0;

        boolean offsetSupported = modelPresent
            ? Double.isFinite(modelEyeHeight)
            : geometrySupported;
        double offsetX = 0.0;
        double offsetY = 0.0;
        double offsetZ = 0.0;
        if (offsetSupported) {
            if (modelPresent) {
                offsetY = modelEyeHeight;
            } else {
                offsetX = (bounds.min.x + bounds.max.x) * 0.5;
                offsetY = (bounds.min.y + bounds.max.y) * 0.5;
                offsetZ = (bounds.min.z + bounds.max.z) * 0.5;
            }
        }

        EntityScaleComponent scaleComponent = store.getComponent(
            reference,
            EntityScaleComponent.getComponentType()
        );
        boolean entityScalePresent = scaleComponent != null;
        double entityScale = entityScalePresent ? scaleComponent.getScale() : 0.0;
        boolean collidable = geometrySupported
            && store.getComponent(reference, Intangible.getComponentType()) == null
            && store.getComponent(reference, DeathComponent.getComponentType()) == null;

        return new PrivilegedNpcGeometryRow(
            spatial,
            geometrySupported,
            geometrySupported ? bounds.min.x : 0.0,
            geometrySupported ? bounds.min.y : 0.0,
            geometrySupported ? bounds.min.z : 0.0,
            geometrySupported ? bounds.max.x : 0.0,
            geometrySupported ? bounds.max.y : 0.0,
            geometrySupported ? bounds.max.z : 0.0,
            collidable,
            false,
            offsetSupported,
            offsetX,
            offsetY,
            offsetZ,
            modelPresent,
            modelAssetId,
            modelScale,
            modelEyeHeight,
            entityScalePresent,
            entityScale
        );
    }

    private static boolean validBounds(Box bounds) {
        return bounds != null
            && Double.isFinite(bounds.min.x)
            && Double.isFinite(bounds.min.y)
            && Double.isFinite(bounds.min.z)
            && Double.isFinite(bounds.max.x)
            && Double.isFinite(bounds.max.y)
            && Double.isFinite(bounds.max.z)
            && bounds.min.x < bounds.max.x
            && bounds.min.y < bounds.max.y
            && bounds.min.z < bounds.max.z;
    }
}
