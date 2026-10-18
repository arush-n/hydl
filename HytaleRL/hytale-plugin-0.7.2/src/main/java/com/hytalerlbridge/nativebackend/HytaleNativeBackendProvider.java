package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.ComponentRegistryProxy;
import com.hypixel.hytale.component.ComponentType;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.modules.entity.damage.Damage;
import com.hypixel.hytale.server.core.asset.type.item.config.Item;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.environment.EnvironmentOptions;
import com.hytalerlbridge.environment.EnvironmentSession;
import com.hytalerlbridge.environment.NativeBackendProvider;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

/** Creates isolated Hytale-native worlds and routes their final tick callbacks. */
public final class HytaleNativeBackendProvider implements NativeBackendProvider {

    private static final System.Logger LOGGER =
        System.getLogger(HytaleNativeBackendProvider.class.getName());
    static final Set<String> SUPPORTED_WORLD_TEMPLATES = Set.of(
        "flat",
        "hytale",
        "hytale_generator"
    );

    private final ComponentType<EntityStore, NativeAgentMarker> markerType;
    private final ComponentType<EntityStore, NativeNpcTraceMarker> traceMarkerType;
    private final ConcurrentHashMap<World, NativeEnvironmentSession> sessions =
        new ConcurrentHashMap<>();
    private volatile Map<String, Integer> itemIds;

    public HytaleNativeBackendProvider(
        ComponentType<EntityStore, NativeAgentMarker> markerType,
        ComponentType<EntityStore, NativeNpcTraceMarker> traceMarkerType
    ) {
        this.markerType = markerType;
        this.traceMarkerType = traceMarkerType;
    }

    public void registerSystems(ComponentRegistryProxy<EntityStore> registry) {
        registry.registerSystem(
            new NativeNpcInteractionOrderingBarrierSystem()
        );
        registry.registerSystem(new NativeNpcTracePreSystem(traceMarkerType));
        registry.registerSystem(new NativeNpcTraceDamageSystem(this));
        registry.registerSystem(new NativeNpcTraceInventorySystem(traceMarkerType));
        registry.registerSystem(new NativeControlSystem(markerType));
        registry.registerSystem(new NativeNpcTraceIntentSystem(traceMarkerType));
        registry.registerSystem(new NativeSyntheticClientSystem(markerType));
        registry.registerSystem(new NativeFallingBlockTraceSystem(markerType));
        registry.registerSystem(new NativeTickCaptureSystem(this));
    }

    @Override
    public EnvironmentSession create(
        String taskId,
        long seed,
        int curriculumPhase,
        int ticksPerStep,
        int maxEpisodeSteps,
        EnvironmentOptions options
    ) {
        if (!taskId.equals("native_fidelity")
            && !taskId.equals("navigate")
            && !taskId.equals("survive")
            && !taskId.equals("kill_trork")) {
            throw new IllegalArgumentException(
                "Native backend currently supports native_fidelity, navigate, survive, "
                    + "and kill_trork; got: "
                    + taskId
            );
        }
        if (!SUPPORTED_WORLD_TEMPLATES.contains(options.world())) {
            throw new IllegalArgumentException(
                "Native backend supports isolated flat, hytale, or "
                    + "hytale_generator world generation; got: "
                    + options.world()
            );
        }

        NativeEnvironmentSession session = new NativeEnvironmentSession(
            this,
            markerType,
            traceMarkerType,
            taskId,
            seed,
            curriculumPhase,
            ticksPerStep,
            maxEpisodeSteps,
            options,
            itemIdMap()
        );
        try {
            session.reset();
            return session;
        } catch (RuntimeException exception) {
            session.close();
            throw exception;
        }
    }

    void attach(World world, NativeEnvironmentSession session) {
        NativeEnvironmentSession previous = sessions.putIfAbsent(world, session);
        if (previous != null) {
            throw new IllegalStateException("Native world is already attached to an RL session");
        }
    }

    void detach(World world, NativeEnvironmentSession session) {
        if (world != null) sessions.remove(world, session);
    }

    void afterNativeTick(Store<EntityStore> store, float deltaTime) {
        World world = store.getExternalData().getWorld();
        NativeEnvironmentSession session = sessions.get(world);
        if (session != null) session.afterNativeTick(store, deltaTime);
    }

    void captureNpcTraceDamage(
        Ref<EntityStore> target,
        Store<EntityStore> store,
        Damage damage
    ) {
        NativeEnvironmentSession session = sessions.get(
            store.getExternalData().getWorld()
        );
        if (session != null) session.captureNpcTraceDamage(target, store, damage);
    }

    private Map<String, Integer> itemIdMap() {
        Map<String, Integer> cached = itemIds;
        if (cached != null) return cached;
        synchronized (this) {
            if (itemIds != null) return itemIds;
            Map<String, Integer> built = new HashMap<>();
            int index = 1;
            for (String id : new TreeSet<>(Item.getAssetMap().getAssetMap().keySet())) {
                built.put(id, index++);
            }
            itemIds = Map.copyOf(built);
            LOGGER.log(
                System.Logger.Level.INFO,
                "Native observation item table contains " + built.size() + " Hytale assets"
            );
            return itemIds;
        }
    }
}
