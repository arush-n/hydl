package com.hytalerlbridge;

import com.hypixel.hytale.server.core.plugin.JavaPlugin;
import com.hypixel.hytale.server.core.plugin.JavaPluginInit;
import com.hytalerlbridge.nativebackend.HytaleNativeBackendProvider;
import com.hytalerlbridge.nativebackend.NativeAgentMarker;
import com.hytalerlbridge.nativebackend.NativeNpcTraceMarker;
import com.hytalerlbridge.environment.EnvironmentManager;
import com.hytalerlbridge.network.BridgeServer;
import com.hytalerlbridge.task.TaskRegistry;
import com.hytalerlbridge.task.builtin.BuiltinTasks;
import java.time.Duration;

/** Hytale 0.5.7 plugin entry point for native and simulator RL backends. */
public final class HytaleRLBridgePlugin extends JavaPlugin {

    private static final System.Logger LOGGER =
        System.getLogger(HytaleRLBridgePlugin.class.getName());

    private BridgeServer bridgeServer;
    private EnvironmentManager environmentManager;
    private BridgeConfig config;

    public HytaleRLBridgePlugin(JavaPluginInit init) {
        super(init);
    }

    @Override
    protected void setup() {
        LOGGER.log(System.Logger.Level.INFO, "HytaleRLBridge v0.1.0 - setting up");

        config = BridgeConfig.loadFromResource();
        TaskRegistry taskRegistry = new TaskRegistry();
        BuiltinTasks.registerAll(taskRegistry);

        var markerType = getEntityStoreRegistry().registerComponent(
            NativeAgentMarker.class,
            NativeAgentMarker::new
        );
        var traceMarkerType = getEntityStoreRegistry().registerComponent(
            NativeNpcTraceMarker.class,
            NativeNpcTraceMarker::new
        );
        HytaleNativeBackendProvider nativeBackend =
            new HytaleNativeBackendProvider(markerType, traceMarkerType);
        nativeBackend.registerSystems(getEntityStoreRegistry());

        environmentManager = new EnvironmentManager(taskRegistry, config, nativeBackend);
        bridgeServer = new BridgeServer(
            config.getHost(),
            config.getPort(),
            environmentManager,
            config.getBufferSize()
        );
    }

    @Override
    protected void start() {
        bridgeServer.start();
        try {
            int boundPort = bridgeServer.awaitBoundPort(Duration.ofSeconds(10));
            LOGGER.log(
                System.Logger.Level.INFO,
                "HytaleRLBridge ready - listening on " + config.getHost() + ":" + boundPort
            );
        } catch (Exception exception) {
            bridgeServer.stop();
            throw new IllegalStateException("HytaleRLBridge failed to bind", exception);
        }
    }

    @Override
    protected void shutdown() {
        LOGGER.log(System.Logger.Level.INFO, "HytaleRLBridge shutting down");
        if (bridgeServer != null) {
            bridgeServer.stop();
        }
        if (environmentManager != null) {
            environmentManager.shutdown();
        }
        LOGGER.log(System.Logger.Level.INFO, "HytaleRLBridge stopped");
    }
}
