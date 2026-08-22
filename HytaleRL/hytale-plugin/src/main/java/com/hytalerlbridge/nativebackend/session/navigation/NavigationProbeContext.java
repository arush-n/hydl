package com.hytalerlbridge.nativebackend.session.navigation;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;

/**
 * The slice of {@code NativeEnvironmentSession} state every navigation probe
 * builder reads.
 *
 * <p>All four builders in {@link NavigationProbeCapture} need exactly this set:
 * the world and actor they probe against, plus the four worldgen identity values
 * that get stamped onto every emitted probe record so a capture can be traced
 * back to the world that produced it. Bundling them keeps the builder signatures
 * readable - threading six values individually pushed
 * {@code buildNavigationPathProbe} to twelve parameters.
 *
 * <p>This is a snapshot taken on the caller's thread at capture time. The
 * session's identity fields are mutable, so a context must not be cached across
 * captures.
 */
public record NavigationProbeContext(
    World world,
    Ref<EntityStore> agentRef,
    String worldName,
    String worldgenProvider,
    String worldgenVersion,
    long seed
) {}
