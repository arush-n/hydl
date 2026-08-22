package com.hytalerlbridge.environment;

import com.hytalerlbridge.imitation.NpcTraceBatch;
import java.util.UUID;

/** Optional native-only source of UUID-pinned NPC imitation traces. */
public interface NativeNpcTraceSource {

    NpcTraceBatch startNpcTrace(UUID npcUuid, String expectedRole, int capacity);

    NpcTraceBatch drainNpcTrace(UUID traceUuid, int maxFrames, boolean stop);
}
