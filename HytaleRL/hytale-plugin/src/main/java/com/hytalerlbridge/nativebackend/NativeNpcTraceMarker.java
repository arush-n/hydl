package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.Component;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import java.util.UUID;

/** Non-serialized marker limiting trace systems to the selected NPC. */
public final class NativeNpcTraceMarker implements Component<EntityStore> {

    private final NativeNpcTraceRecorder recorder;
    private final UUID npcUuid;

    public NativeNpcTraceMarker() {
        this(null, null);
    }

    NativeNpcTraceMarker(NativeNpcTraceRecorder recorder, UUID npcUuid) {
        this.recorder = recorder;
        this.npcUuid = npcUuid;
    }

    NativeNpcTraceRecorder recorder() {
        return recorder;
    }

    UUID npcUuid() {
        return npcUuid;
    }

    @Override
    public NativeNpcTraceMarker clone() {
        return new NativeNpcTraceMarker(recorder, npcUuid);
    }
}
