package com.hytalerlbridge.policy.perception.acquisition;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.asset.type.entityeffect.config.EntityEffect;
import com.hypixel.hytale.server.core.entity.effect.ActiveEntityEffect;
import com.hypixel.hytale.server.core.entity.effect.EffectControllerComponent;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.policy.perception.model.PerceptionFrame;
import com.hytalerlbridge.policy.perception.profile.StatusProgramCatalog;
import com.hytalerlbridge.policy.perception.projection.MechanicsProjection;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/** Reads native effects without inventing programs for unknown asset IDs. */
public final class ServerStatusReader {

    private ServerStatusReader() {
    }

    /**
     * Return an entity-major status row, or {@code null} on semantic mismatch.
     *
     * <p>Visual/derived omissions are removed before the eight-slot capacity is
     * checked. Native effects are sorted by runtime index, matching the bridge
     * evidence order. Cycle elapsed remains zero because 0.5.7 exposes no
     * public per-effect cycle clock; this is also the current native adapter's
     * declared projection.
     */
    public static PerceptionFrame.Status capture(
        Store<EntityStore> store,
        Ref<EntityStore> agent,
        Ref<EntityStore> target,
        StatusProgramCatalog catalog,
        float[] maximumHealth
    ) {
        if (store == null || agent == null || !agent.isValid()
            || catalog == null || maximumHealth == null
            || maximumHealth.length != MechanicsProjection.ENTITY_COUNT) {
            return null;
        }
        for (float value : maximumHealth) {
            if (!Float.isFinite(value) || value <= 0.0f) {
                return null;
            }
        }

        float[] remaining = new float[MechanicsProjection.STATUS_SLOTS];
        float[] elapsed = new float[MechanicsProjection.STATUS_SLOTS];
        float[] cooldown = new float[MechanicsProjection.STATUS_SLOTS];
        float[] damage = new float[MechanicsProjection.STATUS_SLOTS];
        float[] healing = new float[MechanicsProjection.STATUS_SLOTS];
        int[] resourceId = new int[MechanicsProjection.STATUS_SLOTS];
        Arrays.fill(resourceId, -1);
        float[] resourceDelta = new float[MechanicsProjection.STATUS_SLOTS];
        float[] speed = new float[MechanicsProjection.STATUS_SLOTS];
        boolean[] active = new boolean[MechanicsProjection.STATUS_SLOTS];

        @SuppressWarnings("unchecked")
        Ref<EntityStore>[] actors = new Ref[] {agent, target};
        for (int entity = 0; entity < actors.length; entity++) {
            Ref<EntityStore> ref = actors[entity];
            if (ref == null || !ref.isValid()) {
                continue;
            }
            List<ActiveEntityEffect> effects = activeEffects(
                store, ref, catalog);
            if (effects == null
                || effects.size() > MechanicsProjection.STATUS_CAPACITY) {
                return null;
            }
            for (int local = 0; local < effects.size(); local++) {
                ActiveEntityEffect effect = effects.get(local);
                EntityEffect asset = EntityEffect.getAssetMap().getAsset(
                    effect.getEntityEffectIndex());
                if (asset == null || asset.getId() == null
                    || asset.getId().isBlank()) {
                    return null;
                }
                StatusProgramCatalog.Program program = catalog.program(
                    asset.getId());
                if (program == null || program.infinite() != effect.isInfinite()) {
                    return null;
                }
                float nativeRemaining = effect.getRemainingDuration();
                if (!effect.isInfinite()
                    && (!Float.isFinite(nativeRemaining)
                        || nativeRemaining < 0.0f)) {
                    return null;
                }
                int slot = entity * MechanicsProjection.STATUS_CAPACITY + local;
                float health = maximumHealth[entity];
                remaining[slot] = effect.isInfinite()
                    ? Float.MAX_VALUE
                    : nativeRemaining;
                cooldown[slot] = program.cooldown();
                damage[slot] = program.valuePercent()
                    ? program.damage() * health
                    : program.damage();
                healing[slot] = program.valuePercent()
                    ? program.healing() * health
                    : program.healing();
                resourceId[slot] = program.resourceId();
                resourceDelta[slot] = program.resourceDelta();
                speed[slot] = program.speedMultiplier();
                active[slot] = true;
            }
        }
        return new PerceptionFrame.Status(
            remaining,
            elapsed,
            cooldown,
            damage,
            healing,
            resourceId,
            resourceDelta,
            speed,
            active
        );
    }

    private static List<ActiveEntityEffect> activeEffects(
        Store<EntityStore> store,
        Ref<EntityStore> ref,
        StatusProgramCatalog catalog
    ) {
        EffectControllerComponent controller = store.getComponent(
            ref, EffectControllerComponent.getComponentType());
        ActiveEntityEffect[] raw = controller == null
            ? null
            : controller.getAllActiveEntityEffects();
        if (raw == null || raw.length == 0) {
            return List.of();
        }
        raw = raw.clone();
        for (ActiveEntityEffect effect : raw) {
            if (effect == null) {
                return null;
            }
        }
        Arrays.sort(raw, Comparator.comparingInt(
            ActiveEntityEffect::getEntityEffectIndex));
        List<ActiveEntityEffect> result = new ArrayList<>(raw.length);
        for (ActiveEntityEffect effect : raw) {
            EntityEffect asset = EntityEffect.getAssetMap().getAsset(
                effect.getEntityEffectIndex());
            if (asset == null || asset.getId() == null
                || asset.getId().isBlank()) {
                return null;
            }
            if (!catalog.omitted(asset.getId())) {
                result.add(effect);
            }
        }
        return result;
    }
}
