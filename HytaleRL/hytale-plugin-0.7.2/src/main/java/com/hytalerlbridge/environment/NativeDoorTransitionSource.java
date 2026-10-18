package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.NativeDoorTransitionEvidence;

/** Optional native capability for the controlled vertical-door fixture. */
public interface NativeDoorTransitionSource {

    NativeDoorTransitionEvidence captureDoorTransitionEvidence();
}
