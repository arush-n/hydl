package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.Component;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;

/** Non-serialized marker that binds one native NPC to its bridge session. */
public final class NativeAgentMarker implements Component<EntityStore> {

    private final NativeEnvironmentSession session;
    private final int actorId;

    public NativeAgentMarker() {
        this(null, -1);
    }

    public NativeAgentMarker(NativeEnvironmentSession session) {
        this(session, -1);
    }

    public NativeAgentMarker(NativeEnvironmentSession session, int actorId) {
        this.session = session;
        this.actorId = actorId;
    }

    NativeEnvironmentSession session() {
        return session;
    }

    int actorId() {
        return actorId;
    }

    @Override
    public NativeAgentMarker clone() {
        return new NativeAgentMarker(session, actorId);
    }
}
