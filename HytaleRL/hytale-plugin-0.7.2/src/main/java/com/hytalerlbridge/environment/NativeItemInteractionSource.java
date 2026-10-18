package com.hytalerlbridge.environment;

import com.hytalerlbridge.worldgen.NativeItemInteractionEvidence;

/** Native-only resolved item-trigger and interaction-chain evidence. */
public interface NativeItemInteractionSource {

    NativeItemInteractionEvidence captureItemInteractionEvidence(
        int equippedSlot
    );
}
