package com.hytalerlbridge.nativebackend.session.info;

import java.util.Map;
import com.hypixel.hytale.server.core.universe.PlayerRef;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hytalerlbridge.environment.EnvironmentOptions;
import com.hytalerlbridge.environment.fidelity.NativeResourceOverride;
import com.hytalerlbridge.nativebackend.model.NativeSnapshot;

/**
 * Engine tick and timing counters, tick rate, time dilation and fidelity resource overrides.
 *
 * <p>One section of the info map built each step by
 * {@code NativeEnvironmentSession.buildInfo}. See the package README for
 * why these sections are split out and what the ordering contract is.
 */
public final class EngineTimingInfo {

    private EngineTimingInfo() {}

    public static void contribute(
        Map<String, Object> info,
        NativeSnapshot snapshot,
        int executedTicks,
        double simulatedSeconds,
        double minimumDeltaSeconds,
        double maximumDeltaSeconds,
        World world,
        EnvironmentOptions options,
        PlayerRef tickAnchor,
        int episodeEngineTicks
    ) {
        info.put("engine_ticks_executed", executedTicks);
        info.put("engine_simulated_seconds", simulatedSeconds);
        info.put(
            "engine_mean_delta_seconds",
            executedTicks == 0 ? 0.0 : simulatedSeconds / executedTicks
        );
        info.put("engine_min_delta_seconds", minimumDeltaSeconds);
        info.put("engine_max_delta_seconds", maximumDeltaSeconds);
        info.put("native_tick_rate", world == null ? 0 : world.getTps());
        info.put("native_time_dilation", options.nativeTimeDilation());
        info.put(
            "native_fidelity_resource_override_schema",
            NativeResourceOverride.SCHEMA
        );
        info.put(
            "native_fidelity_resource_override_count",
            options.hasNativeFidelityResourceOverrides()
                ? options.nativeFidelityResourceOverrides().size()
                : 0
        );
        info.put(
            "native_fidelity_resource_overrides",
            options.hasNativeFidelityResourceOverrides()
                ? options.nativeFidelityResourceOverrides().stream()
                    .map(
                        row -> row.entityId()
                            + ":"
                            + row.resourceId()
                            + ":"
                            + row.value()
                    )
                    .reduce((left, right) -> left + "," + right)
                    .orElse("")
                : ""
        );
        info.put("headless_tick_anchor", tickAnchor != null);
        info.put("episode_engine_ticks", episodeEngineTicks);
        info.put("native_world_tick", snapshot.worldTick());
        info.put("world_paused_between_steps", true);
    }
}
