package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.dependency.Dependency;
import com.hypixel.hytale.component.dependency.RootDependency;
import com.hypixel.hytale.component.system.tick.TickingSystem;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import java.util.Set;

/** Captures the authoritative result after every other entity system has ticked. */
final class NativeTickCaptureSystem extends TickingSystem<EntityStore> {

    private final HytaleNativeBackendProvider provider;

    NativeTickCaptureSystem(HytaleNativeBackendProvider provider) {
        this.provider = provider;
    }

    @Override
    public Set<Dependency<EntityStore>> getDependencies() {
        return RootDependency.lastSet();
    }

    @Override
    public void tick(float deltaTime, int systemIndex, Store<EntityStore> store) {
        provider.afterNativeTick(store, deltaTime);
    }
}
