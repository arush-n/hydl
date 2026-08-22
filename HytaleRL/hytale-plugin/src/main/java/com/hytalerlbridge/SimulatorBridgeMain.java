package com.hytalerlbridge;

import com.hytalerlbridge.environment.EnvironmentManager;
import com.hytalerlbridge.network.BridgeServer;
import com.hytalerlbridge.task.TaskRegistry;
import com.hytalerlbridge.task.builtin.BuiltinTasks;
import java.time.Duration;
import java.util.concurrent.CountDownLatch;

/** Standalone entry point for testing the simulator bridge without Hytale. */
public final class SimulatorBridgeMain {

    private SimulatorBridgeMain() {
    }

    public static void main(String[] args) throws Exception {
        BridgeConfig config = BridgeConfig.loadFromResource();
        String host = argument(args, "--host", "127.0.0.1");
        int port = Integer.parseInt(argument(args, "--port", Integer.toString(config.getPort())));

        TaskRegistry registry = new TaskRegistry();
        BuiltinTasks.registerAll(registry);
        EnvironmentManager environmentManager = new EnvironmentManager(registry, config);
        BridgeServer server = new BridgeServer(host, port, environmentManager, config.getBufferSize());

        Runtime.getRuntime().addShutdownHook(Thread.ofPlatform().unstarted(() -> {
            server.stop();
            environmentManager.shutdown();
        }));

        server.start();
        int boundPort = server.awaitBoundPort(Duration.ofSeconds(10));
        System.out.println("HYTALERL_READY " + host + ":" + boundPort);
        System.out.flush();
        new CountDownLatch(1).await();
    }

    private static String argument(String[] args, String name, String fallback) {
        for (int index = 0; index < args.length - 1; index++) {
            if (args[index].equals(name)) {
                return args[index + 1];
            }
        }
        return fallback;
    }
}
