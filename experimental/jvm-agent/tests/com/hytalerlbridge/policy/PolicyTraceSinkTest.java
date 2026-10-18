package com.hytalerlbridge.policy;

import com.hytalerlbridge.policy.perception.acquisition.trace.JsonLinesPolicyTraceSink;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;

/** Server-free contract test for native policy transition trace pairing. */
public final class PolicyTraceSinkTest {

    private PolicyTraceSinkTest() {}

    public static void main(String[] arguments) throws Exception {
        testMaximumTransitionsConfiguration();
        Path directory = Files.createTempDirectory("hydl-policy-trace-");
        try {
            JsonLinesPolicyTraceSink trace = new JsonLinesPolicyTraceSink(
                directory, 8, 2, 3, "role\"with\\escape", "profile-sha");
            float[] first = {1.0f, -2.5f};
            float[] second = {3.0f, 4.5f};
            boolean[] mask = {true, false, true};
            int[] factors = new int[Policy.HEAD_SIZES.length];
            trace.observe(
                0, 10L, 1.0f / 30.0f, first, mask, factors,
                ActionDecoder.REJECTED, true, false, "");
            trace.observe(
                0, 11L, 1.0f / 30.0f, second, mask, factors,
                ActionDecoder.REJECTED, true, false, "");
            trace.skip(0, 12L, "evidence_agent_dead", true);
            trace.close();

            Path output;
            try (DirectoryStream<Path> files = Files.newDirectoryStream(
                    directory, "policy-trace-*.jsonl")) {
                output = files.iterator().next();
            }
            List<String> lines = Files.readAllLines(output);
            require(lines.size() == 3, "header plus two transition rows");
            require(lines.get(0).contains(
                "\"event\":\"header\""), "header event");
            require(lines.get(0).contains(
                "\"maximum_transitions\":8"), "configured row budget");
            require(lines.get(0).contains("role\\\"with\\\\escape"),
                "escaped header role");

            String paired = lines.get(1);
            require(paired.contains("\"world_tick\":10"), "first tick");
            require(paired.contains("\"next_world_tick\":11"), "paired tick");
            require(paired.contains("\"next_observation_available\":true"),
                "paired observation");
            require(decodeFloat(paired, "observation_f32_le_b64", 0) == 1.0f,
                "little-endian first observation");
            require(decodeFloat(paired, "next_observation_f32_le_b64", 1)
                    == 4.5f,
                "little-endian next observation");
            require(paired.contains("\"action_legal\":false"),
                "semantic action");

            String terminal = lines.get(2);
            require(terminal.contains("\"world_tick\":11"), "terminal tick");
            require(terminal.contains("\"next_observation_available\":false"),
                "terminal gap");
            require(terminal.contains("\"terminal\":true"),
                "terminal disposition");
            require(trace.transitionsWritten() == 2, "transition count");
        } finally {
            try (DirectoryStream<Path> files = Files.newDirectoryStream(directory)) {
                for (Path file : files) Files.deleteIfExists(file);
            }
            Files.deleteIfExists(directory);
        }
        testTerminalAfterTransitionBudget();
        System.out.println(
            "PASS - native policy trace configures a bounded budget and retains terminal evidence");
    }

    private static void testMaximumTransitionsConfiguration() throws Exception {
        Path directory = Files.createTempDirectory("hydl-policy-trace-config-");
        try {
            Path marker = directory.resolve("policy-trace");
            Files.writeString(marker, "\n");
            require(JsonLinesPolicyTraceSink.readMaximumTransitions(marker)
                    == JsonLinesPolicyTraceSink.DEFAULT_MAXIMUM_TRANSITIONS,
                "empty marker preserves the 512 default");
            Files.writeString(marker, "640\n");
            require(JsonLinesPolicyTraceSink.readMaximumTransitions(marker) == 640,
                "positive marker overrides the default");
            for (String invalid : List.of("0", "-2", "invalid", "2147483648")) {
                Files.writeString(marker, invalid);
                boolean rejected = false;
                try {
                    JsonLinesPolicyTraceSink.readMaximumTransitions(marker);
                } catch (IllegalArgumentException expected) {
                    rejected = true;
                }
                require(rejected, "invalid marker must fail closed: " + invalid);
            }
        } finally {
            try (DirectoryStream<Path> files = Files.newDirectoryStream(directory)) {
                for (Path file : files) Files.deleteIfExists(file);
            }
            Files.deleteIfExists(directory);
        }
    }

    private static void testTerminalAfterTransitionBudget() throws Exception {
        Path directory = Files.createTempDirectory("hydl-policy-trace-budget-");
        try {
            JsonLinesPolicyTraceSink trace = new JsonLinesPolicyTraceSink(
                directory, 1, 2, 3, "role", "profile-sha");
            float[] observation = {1.0f, 2.0f};
            boolean[] mask = {true, false, true};
            int[] factors = new int[Policy.HEAD_SIZES.length];
            trace.observe(0, 1L, 0.03f, observation, mask, factors,
                ActionDecoder.REJECTED, true, false, "");
            trace.observe(0, 2L, 0.03f, observation, mask, factors,
                ActionDecoder.REJECTED, true, false, "");
            trace.observe(0, 3L, 0.03f, observation, mask, factors,
                ActionDecoder.REJECTED, true, false, "");
            trace.observe(0, 4L, 0.03f, observation, mask, factors,
                ActionDecoder.REJECTED, true, false, "");
            trace.skip(0, 5L, "evidence_agent_dead", true);

            Path output;
            try (DirectoryStream<Path> files = Files.newDirectoryStream(
                    directory, "policy-trace-*.jsonl")) {
                output = files.iterator().next();
            }
            List<String> lines = Files.readAllLines(output);
            require(lines.size() == 3,
                "one configured transition plus one terminal row");
            require(lines.get(1).contains("\"world_tick\":1"),
                "ordinary transition uses the configured budget");
            require(lines.get(2).contains("\"world_tick\":4"),
                "terminal row retains the latest pending observation");
            require(lines.get(2).contains("\"terminal\":true"),
                "terminal row remains explicit after the budget");
            require(lines.get(2).contains("terminal:evidence_agent_dead"),
                "terminal reason survives the budget");
            trace.observe(1, 6L, 0.03f, observation, mask, factors,
                ActionDecoder.REJECTED, true, false, "");
            trace.observe(1, 7L, 0.03f, observation, mask, factors,
                ActionDecoder.REJECTED, true, false, "");
            trace.skip(1, 8L, "second_actor_terminal", true);
            trace.close();
            require(Files.readAllLines(output).size() == 3,
                "terminal row is final across later callbacks from other actors");
            require(trace.transitionsWritten() == 2,
                "written count includes the single terminal allowance");
        } finally {
            try (DirectoryStream<Path> files = Files.newDirectoryStream(directory)) {
                for (Path file : files) Files.deleteIfExists(file);
            }
            Files.deleteIfExists(directory);
        }
    }

    private static float decodeFloat(String line, String key, int index) {
        String marker = "\"" + key + "\":\"";
        int start = line.indexOf(marker);
        require(start >= 0, "encoded field " + key);
        start += marker.length();
        int end = line.indexOf('"', start);
        byte[] bytes = Base64.getDecoder().decode(line.substring(start, end));
        return ByteBuffer.wrap(bytes)
            .order(ByteOrder.LITTLE_ENDIAN)
            .getFloat(index * Float.BYTES);
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
