package com.hytalerlbridge;

import java.io.InputStream;
import java.util.Map;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;

/** Immutable-at-runtime configuration loaded from the bundled config.yml. */
public final class BridgeConfig {

    private static final System.Logger LOGGER = System.getLogger(BridgeConfig.class.getName());
    static final String PORT_PROPERTY = "hytalerl.bridge.port";

    private String host = "127.0.0.1";
    private int port = 5556;
    private int ticksPerStep = 4;
    private int maxEpisodeSteps = 0;
    private String defaultTask = "survive";
    private int bufferSize = 65536;

    public static BridgeConfig loadFromResource() {
        BridgeConfig config = new BridgeConfig();
        try (InputStream input = BridgeConfig.class.getClassLoader().getResourceAsStream("config.yml")) {
            if (input == null) {
                LOGGER.log(System.Logger.Level.WARNING, "config.yml not found; using defaults");
                config.applySystemOverrides();
                config.validate();
                return config;
            }

            LoaderOptions options = new LoaderOptions();
            options.setAllowDuplicateKeys(false);
            Object loaded = new Yaml(new SafeConstructor(options)).load(input);
            if (!(loaded instanceof Map<?, ?> root)) {
                throw new IllegalArgumentException("config.yml must contain a mapping");
            }

            Map<?, ?> network = section(root, "network");
            config.host = stringValue(network, "host", config.host);
            config.port = intValue(network, "port", config.port);

            Map<?, ?> environment = section(root, "environment");
            config.ticksPerStep = intValue(environment, "ticks_per_step", config.ticksPerStep);
            config.maxEpisodeSteps = intValue(
                environment,
                "max_episode_steps",
                intValue(environment, "max_episode_ticks", config.maxEpisodeSteps)
            );
            config.defaultTask = stringValue(environment, "default_task", config.defaultTask);

            Map<?, ?> performance = section(root, "performance");
            config.bufferSize = intValue(performance, "buffer_size", config.bufferSize);
            config.applySystemOverrides();
            config.validate();
            return config;
        } catch (Exception exception) {
            throw new IllegalStateException("Failed to load config.yml", exception);
        }
    }

    private void applySystemOverrides() {
        String portOverride = System.getProperty(PORT_PROPERTY);
        if (portOverride != null && !portOverride.isBlank()) {
            try {
                port = Integer.parseInt(portOverride);
            } catch (NumberFormatException exception) {
                throw new IllegalArgumentException(
                    "System property " + PORT_PROPERTY + " must be an integer",
                    exception
                );
            }
        }
    }

    private void validate() {
        if (host.isBlank()) throw new IllegalArgumentException("network.host cannot be blank");
        if (port < 0 || port > 65535) throw new IllegalArgumentException("network.port must be 0..65535");
        if (ticksPerStep < 1 || ticksPerStep > 1000) {
            throw new IllegalArgumentException("environment.ticks_per_step must be 1..1000");
        }
        if (maxEpisodeSteps < 0) {
            throw new IllegalArgumentException("environment.max_episode_steps cannot be negative");
        }
        if (bufferSize < 1024) throw new IllegalArgumentException("performance.buffer_size is too small");
    }

    private static Map<?, ?> section(Map<?, ?> root, String key) {
        Object value = root.get(key);
        return value instanceof Map<?, ?> map ? map : Map.of();
    }

    private static String stringValue(Map<?, ?> map, String key, String fallback) {
        Object value = map.get(key);
        return value instanceof String string ? string : fallback;
    }

    private static int intValue(Map<?, ?> map, String key, int fallback) {
        Object value = map.get(key);
        return value instanceof Number number ? number.intValue() : fallback;
    }

    public String getHost() { return host; }
    public int getPort() { return port; }
    public int getTicksPerStep() { return ticksPerStep; }
    public int getMaxEpisodeSteps() { return maxEpisodeSteps; }
    public String getDefaultTask() { return defaultTask; }
    public int getBufferSize() { return bufferSize; }
}
