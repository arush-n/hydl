package com.hytalerlbridge.combat.dodge;

import com.hypixel.hytale.protocol.InteractionCooldown;
import com.hypixel.hytale.protocol.InteractionType;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.Interaction;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.InteractionTypeUtils;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hytalerlbridge.nativebackend.model.NativeInteractionBinding;
import com.hytalerlbridge.nativebackend.support.InteractionSupport;
import java.util.List;

/** Resolves lateral Dodge payloads without losing the public root cooldown. */
public final class NativeDodgeBindingResolver {
    static final String ROOT_PREFIX = "HytaleRL_DodgeProbe_";

    private NativeDodgeBindingResolver() {}

    public static synchronized NativeInteractionBinding resolve(int direction) {
        String interactionId = NativeDodgeProgram.interactionId(direction);
        if (interactionId.isEmpty()) {
            throw new IllegalArgumentException(
                "Dodge direction has no authored payload: " + direction
            );
        }
        if (Interaction.getAssetMap().getAsset(interactionId) == null) {
            throw new IllegalArgumentException(
                "Native Dodge interaction asset not found: " + interactionId
            );
        }
        float engineDefault = InteractionTypeUtils.getDefaultCooldown(
            InteractionType.Dodge
        );
        if (Float.compare(engineDefault, NativeDodgeProgram.COOLDOWN_SECONDS) != 0) {
            throw new IllegalStateException(
                "Native Dodge default cooldown moved: " + engineDefault
            );
        }

        String rootId = ROOT_PREFIX + interactionId;
        RootInteraction root = RootInteraction.getAssetMap().getAsset(rootId);
        if (root == null) {
            var result = RootInteraction.getAssetStore().loadAssets(
                InteractionSupport.NATIVE_COMBAT_PROBE_ASSET_PACK,
                List.of(configuredRoot(rootId, interactionId))
            );
            root = RootInteraction.getAssetMap().getAsset(rootId);
            if (result.hasFailed() || root == null) {
                throw new IllegalStateException(
                    "Failed to register native Dodge probe root: " + rootId
                );
            }
        }
        return new NativeInteractionBinding(
            interactionId,
            InteractionType.Dodge,
            root
        );
    }

    static RootInteraction configuredRoot(String rootId, String interactionId) {
        return new RootInteraction(rootId, configuredCooldown(), interactionId);
    }

    static InteractionCooldown configuredCooldown() {
        return new InteractionCooldown(
            NativeDodgeProgram.COOLDOWN_ID,
            NativeDodgeProgram.COOLDOWN_SECONDS,
            false,
            null,
            false,
            false
        );
    }
}
