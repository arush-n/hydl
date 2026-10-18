package com.hytalerlbridge.nativebackend.model;

import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;

/** Extracted verbatim from {@code NativeEnvironmentSession}. */
public record NativeInteractionBinding(
    String id,
    InteractionType type,
    RootInteraction root
) {}
