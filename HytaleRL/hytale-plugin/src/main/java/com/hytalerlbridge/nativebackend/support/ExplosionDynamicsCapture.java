package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.entity.ExplosionConfig;
import com.hypixel.hytale.server.core.entity.ExplosionUtils;
import com.hypixel.hytale.server.core.entity.UUIDComponent;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.entity.damage.Damage;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatMap;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatValue;
import com.hypixel.hytale.server.core.modules.entitystats.EntityStatsModule;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.worldgen.NativeExplosionCandidateProbe;
import com.hytalerlbridge.worldgen.NativeExplosionDynamicsProbe;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;
import org.joml.Vector3d;

/** Runs one terminal, health-restored native explosion damage fixture. */
public final class ExplosionDynamicsCapture {

    private ExplosionDynamicsCapture() {}

    public record Context(
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed,
        String worldEpoch
    ) {}

    public static NativeExplosionDynamicsProbe capture(
        World world,
        Ref<EntityStore> agent,
        Ref<EntityStore> target,
        Context context,
        int capacity
    ) {
        if (
            world == null
                || context == null
                || agent == null
                || !agent.isValid()
                || target == null
                || !target.isValid()
        ) {
            return failure(context, capacity, "controlled entities unavailable");
        }
        Store<EntityStore> entities = world.getEntityStore().getStore();
        Store<ChunkStore> chunks = world.getChunkStore().getStore();
        List<EntityState> before;
        try {
            before = List.of(snapshot(entities, agent), snapshot(entities, target));
        } catch (RuntimeException exception) {
            return failure(context, capacity, message(exception));
        }
        Vector3d origin = new Vector3d(before.get(0).position())
            .add(before.get(1).position())
            .mul(0.5);
        NativeExplosionCandidateProbe admission = ExplosionAdmissionProbe.capture(
            world,
            agent,
            context.serverVersion(),
            context.worldName(),
            context.worldgenProvider(),
            context.worldgenVersion(),
            context.seed(),
            new double[] {origin.x, origin.y, origin.z},
            (int) NativeExplosionDynamicsProbe.ENTITY_DAMAGE_RADIUS,
            NativeExplosionDynamicsProbe.ENTITY_DAMAGE_RADIUS,
            false,
            capacity
        );
        Set<UUID> expected = before.stream()
            .map(EntityState::uuid)
            .collect(Collectors.toUnmodifiableSet());
        Set<UUID> admitted = admission.candidates().stream()
            .map(NativeExplosionCandidateProbe.Candidate::uuid)
            .collect(Collectors.toUnmodifiableSet());
        if (admission.overflow() || !admitted.equals(expected)) {
            return failure(
                context,
                capacity,
                "native admission did not select exactly both fixture entities"
            );
        }

        List<EntityState> after;
        RuntimeException executionFailure = null;
        try {
            ExplosionUtils.performExplosion(
                Damage.NULL_SOURCE,
                origin,
                new ControlledConfig(),
                null,
                entities,
                chunks
            );
            after = List.of(snapshot(entities, agent), snapshot(entities, target));
        } catch (RuntimeException exception) {
            executionFailure = exception;
            after = List.of();
        } finally {
            for (EntityState state : before) {
                try {
                    state.stats().setStatValue(
                        state.healthIndex(),
                        state.health()
                    );
                } catch (RuntimeException restoreFailure) {
                    if (executionFailure == null) executionFailure = restoreFailure;
                    else executionFailure.addSuppressed(restoreFailure);
                }
            }
        }
        if (executionFailure != null) {
            return failure(context, capacity, message(executionFailure));
        }

        List<NativeExplosionDynamicsProbe.EntityEffect> effects =
            new ArrayList<>(2);
        for (int index = 0; index < before.size(); index++) {
            EntityState start = before.get(index);
            EntityState end = after.get(index);
            if (!start.uuid().equals(end.uuid())) {
                return failure(context, capacity, "entity identity changed");
            }
            double distance = start.position().distance(origin);
            float expectedDamage = (float) (
                NativeExplosionDynamicsProbe.ENTITY_DAMAGE
                    * Math.pow(
                        1.0
                            - distance
                                / NativeExplosionDynamicsProbe.ENTITY_DAMAGE_RADIUS,
                        NativeExplosionDynamicsProbe.ENTITY_DAMAGE_FALLOFF
                    )
            );
            effects.add(new NativeExplosionDynamicsProbe.EntityEffect(
                0,
                start.uuid(),
                start.position().x,
                start.position().y,
                start.position().z,
                distance,
                start.health(),
                end.health(),
                expectedDamage
            ));
        }
        effects.sort(Comparator.comparing(effect -> effect.uuid().toString()));
        List<NativeExplosionDynamicsProbe.EntityEffect> ordered =
            new ArrayList<>(effects.size());
        for (int index = 0; index < effects.size(); index++) {
            NativeExplosionDynamicsProbe.EntityEffect value = effects.get(index);
            ordered.add(new NativeExplosionDynamicsProbe.EntityEffect(
                index,
                value.uuid(),
                value.x(),
                value.y(),
                value.z(),
                value.distance(),
                value.healthBefore(),
                value.healthAfter(),
                value.expectedRawDamage()
            ));
        }
        return new NativeExplosionDynamicsProbe(
            context.serverVersion(),
            context.worldName(),
            context.worldgenProvider(),
            context.worldgenVersion(),
            context.seed(),
            context.worldEpoch(),
            origin.x,
            origin.y,
            origin.z,
            NativeExplosionDynamicsProbe.ENTITY_DAMAGE_RADIUS,
            NativeExplosionDynamicsProbe.ENTITY_DAMAGE,
            NativeExplosionDynamicsProbe.ENTITY_DAMAGE_FALLOFF,
            capacity,
            ordered.size(),
            ordered.size() > capacity,
            ordered.size() <= capacity,
            true,
            ordered.size() > capacity ? "entity capacity exceeded" : "",
            ordered.size() > capacity ? List.of() : ordered
        );
    }

    private static EntityState snapshot(
        Store<EntityStore> store,
        Ref<EntityStore> reference
    ) {
        UUIDComponent identity = store.getComponent(
            reference,
            UUIDComponent.getComponentType()
        );
        TransformComponent transform = store.getComponent(
            reference,
            TransformComponent.getComponentType()
        );
        EntityStatMap stats = store.getComponent(
            reference,
            EntityStatsModule.get().getEntityStatMapComponentType()
        );
        EntityStatValue health = stats == null ? null : stats.get("Health");
        if (identity == null || transform == null || health == null) {
            throw new IllegalStateException(
                "fixture entity lacks UUID, transform, or Health"
            );
        }
        return new EntityState(
            identity.getUuid(),
            new Vector3d(transform.getPosition()),
            stats,
            health.getIndex(),
            health.get()
        );
    }

    private static NativeExplosionDynamicsProbe failure(
        Context context,
        int capacity,
        String reason
    ) {
        if (context == null) {
            throw new IllegalArgumentException("context must be present");
        }
        return new NativeExplosionDynamicsProbe(
            context.serverVersion(),
            context.worldName(),
            context.worldgenProvider(),
            context.worldgenVersion(),
            context.seed(),
            context.worldEpoch(),
            0.0,
            0.0,
            0.0,
            NativeExplosionDynamicsProbe.ENTITY_DAMAGE_RADIUS,
            NativeExplosionDynamicsProbe.ENTITY_DAMAGE,
            NativeExplosionDynamicsProbe.ENTITY_DAMAGE_FALLOFF,
            capacity,
            0,
            false,
            false,
            true,
            reason,
            List.of()
        );
    }

    private static String message(RuntimeException exception) {
        String detail = exception.getMessage();
        String value = exception.getClass().getSimpleName()
            + (detail == null || detail.isBlank() ? "" : ": " + detail);
        return value.length() <= 256 ? value : value.substring(0, 256);
    }

    private record EntityState(
        UUID uuid,
        Vector3d position,
        EntityStatMap stats,
        int healthIndex,
        float health
    ) {}

    private static final class ControlledConfig extends ExplosionConfig {

        private ControlledConfig() {
            damageBlocks = false;
            damageEntities = true;
            blockDamageRadius =
                (int) NativeExplosionDynamicsProbe.ENTITY_DAMAGE_RADIUS;
            entityDamageRadius = NativeExplosionDynamicsProbe.ENTITY_DAMAGE_RADIUS;
            entityDamage = NativeExplosionDynamicsProbe.ENTITY_DAMAGE;
            entityDamageFalloff =
                NativeExplosionDynamicsProbe.ENTITY_DAMAGE_FALLOFF;
            blockDropChance = 0.0f;
            knockback = null;
            itemTool = null;
            particles = null;
            soundEventId = null;
            soundEventIndex = 0;
        }
    }
}
