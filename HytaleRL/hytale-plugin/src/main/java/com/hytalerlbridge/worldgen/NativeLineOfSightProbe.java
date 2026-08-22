package com.hytalerlbridge.worldgen;

import com.hypixel.hytale.component.CommandBuffer;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.hitdetection.HitDetectionExecutor;
import com.hypixel.hytale.math.hitdetection.LineOfSightProvider;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.server.core.modules.entity.component.ModelComponent;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.selector.HorizontalSelector;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.selector.Selector;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.corecomponents.entity.filters.EntityFilterLineOfSight;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.role.support.PositionCache;
import java.lang.reflect.Field;
import org.joml.Vector3d;

/** Executes the native LOS consumers against one controlled unloaded edge. */
public final class NativeLineOfSightProbe {
    private static final int CACHE_TRIALS = 16;
    private static final double CACHE_STEP_SECONDS = 0.001;
    private static final int MAX_UNLOADED_SEARCH_CHUNKS = 8;

    private NativeLineOfSightProbe() {}

    public static NativeLineOfSightEvidence capture(
        World world,
        Store<EntityStore> store,
        Ref<EntityStore> agentRef,
        Ref<EntityStore> targetRef,
        Role role,
        String serverVersion,
        String worldName,
        String worldgenProvider,
        String worldgenVersion,
        long seed
    ) {
        if (
            world == null
                || store == null
                || agentRef == null
                || targetRef == null
                || role == null
                || !agentRef.isValid()
                || !targetRef.isValid()
                || role.getPositionCache() == null
        ) {
            throw new IllegalStateException(
                "native LOS evidence requires live agent, target, role, and PositionCache"
            );
        }
        TransformComponent agentTransform = requireTransform(store, agentRef, "agent");
        TransformComponent targetTransform = requireTransform(store, targetRef, "target");
        Vector3d originalTarget = new Vector3d(targetTransform.getPosition());
        double sourceEyeHeight = eyeHeight(store, agentRef);
        double targetEyeHeight = eyeHeight(store, targetRef);
        Vector3d agentPosition = agentTransform.getPosition();
        double[] start = {
            agentPosition.x,
            agentPosition.y + sourceEyeHeight,
            agentPosition.z,
        };
        int startChunkX = ChunkUtil.chunkCoordinate(start[0]);
        int startChunkZ = ChunkUtil.chunkCoordinate(start[2]);
        int unloadedChunkX = firstUnloadedChunkX(world, startChunkX, startChunkZ);
        int loadedChunkX = unloadedChunkX - 1;
        double boundaryX = (double) unloadedChunkX * ChunkUtil.SIZE;
        double[] loadedEnd = {boundaryX - 0.25, start[1], start[2]};
        double[] unloadedEnd = {boundaryX + 0.25, start[1], start[2]};
        long loadedChunkIndex = ChunkUtil.indexChunk(loadedChunkX, startChunkZ);
        long unloadedChunkIndex = ChunkUtil.indexChunk(unloadedChunkX, startChunkZ);
        boolean loadedPresent = world.getChunkIfInMemory(loadedChunkIndex) != null;
        boolean unloadedPresentBefore =
            world.getChunkIfInMemory(unloadedChunkIndex) != null;
        if (!loadedPresent || unloadedPresentBefore) {
            throw new IllegalStateException(
                "controlled native LOS edge does not separate loaded and unloaded chunks"
            );
        }

        PositionCache cache = role.getPositionCache();
        EntityFilterLineOfSight perceptionFilter = new EntityFilterLineOfSight();
        try {
            boolean perceptionLoaded = perception(
                perceptionFilter,
                cache,
                store,
                agentRef,
                targetRef,
                targetTransform,
                targetEyeHeight,
                loadedEnd,
                true
            );
            boolean perceptionUnloaded = perception(
                perceptionFilter,
                cache,
                store,
                agentRef,
                targetRef,
                targetTransform,
                targetEyeHeight,
                unloadedEnd,
                true
            );
            LineOfSightProvider selectorProvider = selectorProvider(
                store,
                agentRef,
                Math.abs(unloadedEnd[0] - start[0]) + 2.0
            );
            boolean selectorLoaded = selectorProvider.test(
                start[0],
                start[1],
                start[2],
                loadedEnd[0],
                loadedEnd[1],
                loadedEnd[2]
            );
            boolean selectorUnloaded = selectorProvider.test(
                start[0],
                start[1],
                start[2],
                unloadedEnd[0],
                unloadedEnd[1],
                unloadedEnd[2]
            );

            setTargetEye(targetTransform, targetEyeHeight, loadedEnd);
            cache.clear(1.0);
            requireVisible(
                perceptionFilter.matchesEntity(agentRef, targetRef, role, store),
                "loaded forward cache prime"
            );
            setTargetEye(targetTransform, targetEyeHeight, unloadedEnd);
            boolean forwardCached = perceptionFilter.matchesEntity(
                agentRef,
                targetRef,
                role,
                store
            );
            boolean inverseUncached = cache.hasInverseLineOfSight(
                agentRef,
                targetRef,
                store
            );

            int[] expirySteps = new int[CACHE_TRIALS];
            for (int trial = 0; trial < CACHE_TRIALS; trial++) {
                setTargetEye(targetTransform, targetEyeHeight, loadedEnd);
                cache.clear(1.0);
                requireVisible(
                    perceptionFilter.matchesEntity(agentRef, targetRef, role, store),
                    "loaded cache timing prime"
                );
                setTargetEye(targetTransform, targetEyeHeight, unloadedEnd);
                requireVisible(
                    perceptionFilter.matchesEntity(agentRef, targetRef, role, store),
                    "cached LOS before expiry"
                );
                expirySteps[trial] = firstExpiryStep(
                    perceptionFilter,
                    cache,
                    store,
                    agentRef,
                    targetRef,
                    role
                );
            }
            boolean unloadedPresentAfter =
                world.getChunkIfInMemory(unloadedChunkIndex) != null;
            return new NativeLineOfSightEvidence(
                serverVersion,
                worldName,
                worldgenProvider,
                worldgenVersion,
                seed,
                start,
                loadedEnd,
                unloadedEnd,
                loadedChunkX,
                startChunkZ,
                unloadedChunkX,
                startChunkZ,
                loadedPresent,
                unloadedPresentBefore,
                unloadedPresentAfter,
                perceptionLoaded,
                perceptionUnloaded,
                selectorLoaded,
                selectorUnloaded,
                forwardCached,
                inverseUncached,
                CACHE_STEP_SECONDS,
                expirySteps
            );
        } finally {
            targetTransform.setPosition(originalTarget);
            cache.clear(1.0);
        }
    }

    private static int firstExpiryStep(
        EntityFilterLineOfSight filter,
        PositionCache cache,
        Store<EntityStore> store,
        Ref<EntityStore> agentRef,
        Ref<EntityStore> targetRef,
        Role role
    ) {
        for (
            int step = 1;
            step <= NativeLineOfSightEvidence.MAX_CACHE_STEPS;
            step++
        ) {
            cache.clear(CACHE_STEP_SECONDS);
            if (!filter.matchesEntity(agentRef, targetRef, role, store)) {
                return step;
            }
        }
        throw new IllegalStateException(
            "PositionCache LOS did not expire within the bounded probe"
        );
    }

    private static boolean perception(
        EntityFilterLineOfSight filter,
        PositionCache cache,
        Store<EntityStore> store,
        Ref<EntityStore> agentRef,
        Ref<EntityStore> targetRef,
        TransformComponent targetTransform,
        double targetEyeHeight,
        double[] endpoint,
        boolean clearCache
    ) {
        setTargetEye(targetTransform, targetEyeHeight, endpoint);
        if (clearCache) cache.clear(1.0);
        return filter.matchesEntity(agentRef, targetRef, roleOf(store, agentRef), store);
    }

    private static Role roleOf(
        Store<EntityStore> store,
        Ref<EntityStore> reference
    ) {
        var npc = store.getComponent(
            reference,
            com.hypixel.hytale.server.npc.entities.NPCEntity.getComponentType()
        );
        if (npc == null || npc.getRole() == null) {
            throw new IllegalStateException("native LOS source has no role");
        }
        return npc.getRole();
    }

    private static LineOfSightProvider selectorProvider(
        Store<EntityStore> store,
        Ref<EntityStore> agentRef,
        double endDistance
    ) {
        try {
            ProbeHorizontalSelector configuration =
                new ProbeHorizontalSelector(endDistance);
            Selector runtime = configuration.newSelector();
            runtime.tick(
                new ProbeCommandBuffer(store),
                agentRef,
                0.01f,
                1.0f
            );
            Field executorField = runtime.getClass().getDeclaredField("executor");
            executorField.setAccessible(true);
            HitDetectionExecutor executor =
                (HitDetectionExecutor) executorField.get(runtime);
            Field providerField =
                HitDetectionExecutor.class.getDeclaredField("losProvider");
            providerField.setAccessible(true);
            Object provider = providerField.get(executor);
            if (!(provider instanceof LineOfSightProvider lineOfSightProvider)) {
                throw new IllegalStateException(
                    "HorizontalSelector installed no native LOS provider"
                );
            }
            return lineOfSightProvider;
        } catch (ReflectiveOperationException exception) {
            throw new IllegalStateException(
                "Hytale selector LOS internals differ from the pinned 0.5.7 source",
                exception
            );
        }
    }

    private static int firstUnloadedChunkX(
        World world,
        int startChunkX,
        int startChunkZ
    ) {
        for (int offset = 1; offset <= MAX_UNLOADED_SEARCH_CHUNKS; offset++) {
            int chunkX = startChunkX + offset;
            if (
                world.getChunkIfInMemory(
                    ChunkUtil.indexChunk(chunkX, startChunkZ)
                ) == null
            ) {
                return chunkX;
            }
        }
        throw new IllegalStateException(
            "no unloaded native chunk found within the bounded LOS probe"
        );
    }

    private static TransformComponent requireTransform(
        Store<EntityStore> store,
        Ref<EntityStore> reference,
        String name
    ) {
        TransformComponent transform = store.getComponent(
            reference,
            TransformComponent.getComponentType()
        );
        if (transform == null) {
            throw new IllegalStateException(name + " has no TransformComponent");
        }
        return transform;
    }

    private static double eyeHeight(
        Store<EntityStore> store,
        Ref<EntityStore> reference
    ) {
        ModelComponent model = store.getComponent(
            reference,
            ModelComponent.getComponentType()
        );
        return model == null || model.getModel() == null
            ? 0.0
            : model.getModel().getEyeHeight();
    }

    private static void setTargetEye(
        TransformComponent target,
        double eyeHeight,
        double[] endpoint
    ) {
        target.setPosition(
            new Vector3d(endpoint[0], endpoint[1] - eyeHeight, endpoint[2])
        );
    }

    private static void requireVisible(boolean value, String phase) {
        if (!value) {
            throw new IllegalStateException(
                "native LOS fixture lost its controlled clear ray during " + phase
            );
        }
    }

    private static final class ProbeHorizontalSelector
        extends HorizontalSelector {
        private ProbeHorizontalSelector(double endDistance) {
            startDistance = 0.01;
            this.endDistance = endDistance;
            yawLength = Math.PI / 2.0;
            direction = Direction.TO_RIGHT;
            testLineOfSight = true;
        }
    }

    private static final class ProbeCommandBuffer
        extends CommandBuffer<EntityStore> {
        private ProbeCommandBuffer(Store<EntityStore> store) {
            super(store);
        }
    }
}
