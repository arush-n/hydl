package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.NativePerceptionChannels;

/** Native-only fixed channel capture capability. */
public interface NativePerceptionChannelSource {

    NativePerceptionChannels capturePerceptionChannels(int[] positions);
}
