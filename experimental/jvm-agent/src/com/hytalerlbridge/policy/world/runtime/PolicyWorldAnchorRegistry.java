package com.hytalerlbridge.policy.world.runtime;

import com.hypixel.hytale.server.core.universe.PlayerRef;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hytalerlbridge.policy.HeadlessTickAnchor;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.UUID;

/** World-scoped packet anchors and opaque epochs owned by the policy plugin. */
public final class PolicyWorldAnchorRegistry implements AutoCloseable {

    private final Map<World, Context> contexts = new IdentityHashMap<>();

    /** Create at most one anchor/epoch pair for this exact World instance. */
    public synchronized Context context(World world) {
        if (world == null) {
            throw new IllegalArgumentException("policy World is unavailable");
        }
        return contexts.computeIfAbsent(
            world,
            key -> new Context(
                "policy-world-" + UUID.randomUUID(),
                HeadlessTickAnchor.install(key)
            )
        );
    }

    @Override
    public synchronized void close() {
        for (Map.Entry<World, Context> entry : contexts.entrySet()) {
            HeadlessTickAnchor.uninstall(
                entry.getKey(), entry.getValue().packetAnchor());
        }
        contexts.clear();
    }

    public record Context(String worldEpoch, PlayerRef packetAnchor) {
        public Context {
            if (worldEpoch == null || worldEpoch.isBlank()
                || packetAnchor == null) {
                throw new IllegalArgumentException(
                    "invalid policy World context");
            }
        }
    }
}
