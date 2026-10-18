package com.hytalerlbridge.nativebackend.support;

import com.hypixel.hytale.component.ComponentAccessor;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.entity.ExplosionConfig;
import com.hypixel.hytale.server.core.entity.ExplosionUtils;
import com.hypixel.hytale.server.core.entity.UUIDComponent;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.worldgen.NativeExplosionCandidateProbe;
import it.unimi.dsi.fastutil.objects.ReferenceOpenHashSet;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.joml.Vector3d;

/** Calls Hytale's own non-damaging explosion target-admission pass. */
public final class ExplosionAdmissionProbe {

    private static final Method PROCESS_TARGET_BLOCKS = resolveTargetMethod();

    private ExplosionAdmissionProbe() {}

    public static NativeExplosionCandidateProbe capture(
        World world,
        Ref<EntityStore> controlledActor,
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed,
        double[] origin,
        int blockDamageRadius,
        float entityDamageRadius,
        boolean ignoreControlledActor,
        int capacity
    ) {
        if (world == null) {
            throw new IllegalArgumentException("world must be present");
        }
        if (
            ignoreControlledActor
                && (controlledActor == null || !controlledActor.isValid())
        ) {
            throw new IllegalStateException(
                "controlled actor is unavailable for explosion exclusion"
            );
        }

        Store<EntityStore> entities = world.getEntityStore().getStore();
        Store<ChunkStore> chunks = world.getChunkStore().getStore();
        Set<Ref<EntityStore>> admitted = new ReferenceOpenHashSet<>();
        invokeNativeAdmission(
            new Vector3d(origin[0], origin[1], origin[2]),
            new ProbeExplosionConfig(blockDamageRadius, entityDamageRadius),
            ignoreControlledActor ? controlledActor : null,
            admitted,
            entities,
            chunks
        );

        List<NativeExplosionCandidateProbe.Candidate> matches =
            new ArrayList<>(admitted.size());
        for (Ref<EntityStore> reference : admitted) {
            UUIDComponent identity = entities.getComponent(
                reference,
                UUIDComponent.getComponentType()
            );
            TransformComponent transform = entities.getComponent(
                reference,
                TransformComponent.getComponentType()
            );
            if (identity == null || transform == null) {
                throw new IllegalStateException(
                    "native explosion admitted an entity without UUID/transform"
                );
            }
            Vector3d position = transform.getPosition();
            matches.add(new NativeExplosionCandidateProbe.Candidate(
                identity.getUuid(),
                position.x,
                position.y,
                position.z
            ));
        }
        return NativeExplosionCandidateProbe.fromMatches(
            serverVersion,
            worldName,
            worldgenProvider,
            worldgenVersion,
            seed,
            origin,
            blockDamageRadius,
            entityDamageRadius,
            ignoreControlledActor,
            capacity,
            matches
        );
    }

    private static void invokeNativeAdmission(
        Vector3d origin,
        ExplosionConfig config,
        Ref<EntityStore> ignored,
        Set<Ref<EntityStore>> admitted,
        ComponentAccessor<EntityStore> entities,
        ComponentAccessor<ChunkStore> chunks
    ) {
        try {
            PROCESS_TARGET_BLOCKS.invoke(
                null,
                origin,
                config,
                ignored,
                admitted,
                entities,
                chunks
            );
        } catch (IllegalAccessException exception) {
            throw new IllegalStateException(
                "Hytale explosion admission is inaccessible",
                exception
            );
        } catch (InvocationTargetException exception) {
            Throwable cause = exception.getCause();
            if (cause instanceof RuntimeException runtime) throw runtime;
            throw new IllegalStateException(
                "Hytale explosion admission failed",
                cause
            );
        }
    }

    private static Method resolveTargetMethod() {
        try {
            Method method = ExplosionUtils.class.getDeclaredMethod(
                "processTargetBlocks",
                Vector3d.class,
                ExplosionConfig.class,
                Ref.class,
                Set.class,
                ComponentAccessor.class,
                ComponentAccessor.class
            );
            method.setAccessible(true);
            return method;
        } catch (ReflectiveOperationException exception) {
            throw new ExceptionInInitializerError(exception);
        }
    }

    private static final class ProbeExplosionConfig extends ExplosionConfig {

        private ProbeExplosionConfig(
            int blockDamageRadius,
            float entityDamageRadius
        ) {
            this.damageEntities = true;
            this.damageBlocks = false;
            this.blockDamageRadius = blockDamageRadius;
            this.entityDamageRadius = entityDamageRadius;
        }
    }
}
