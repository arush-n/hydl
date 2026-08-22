package com.hytalerlbridge.policy.perception.profile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** Checkpoint-pinned semantics for native active-effect evidence. */
public final class StatusProgramCatalog {

    public record Program(
        int semanticId,
        float damage,
        float healing,
        float cooldown,
        float resourceDelta,
        float speedMultiplier,
        int resourceId,
        boolean valuePercent,
        boolean infinite
    ) {}

    private final Map<Integer, Program> programs;
    private final Set<String> omittedEffectIds;

    private StatusProgramCatalog(
        Map<Integer, Program> programs,
        Set<String> omittedEffectIds
    ) {
        this.programs = Map.copyOf(programs);
        this.omittedEffectIds = Set.copyOf(omittedEffectIds);
    }

    public static StatusProgramCatalog load(Path root) throws IOException {
        Map<Integer, Program> programs = new HashMap<>();
        Path table = root.resolve("status_programs.tsv");
        int lineNumber = 0;
        for (String line : Files.readAllLines(table, StandardCharsets.UTF_8)) {
            lineNumber++;
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] values = line.split("\\t", -1);
            if (values.length != 9) {
                throw new IOException(
                    table + ":" + lineNumber + " has " + values.length
                        + " fields, expected 9");
            }
            try {
                Program program = new Program(
                    Integer.parseInt(values[0]),
                    finite(values[1]),
                    finite(values[2]),
                    finite(values[3]),
                    finite(values[4]),
                    finite(values[5]),
                    Integer.parseInt(values[6]),
                    binary(values[7]),
                    binary(values[8])
                );
                if (program.semanticId() <= 0
                    || programs.put(program.semanticId(), program) != null) {
                    throw new IOException(
                        "duplicate or invalid status semantic ID at " + table
                            + ":" + lineNumber);
                }
            } catch (NumberFormatException invalid) {
                throw new IOException(
                    "invalid status program number at " + table + ":"
                        + lineNumber,
                    invalid
                );
            }
        }

        Set<String> omitted = new HashSet<>();
        Path omissions = root.resolve("status_omissions.txt");
        for (String line : Files.readAllLines(omissions, StandardCharsets.UTF_8)) {
            String value = line.trim();
            if (!value.isEmpty() && !value.startsWith("#")) {
                omitted.add(value);
            }
        }
        return new StatusProgramCatalog(programs, omitted);
    }

    /** Resolve exactly as the JAX native decoder does: FNV-1a asset ID. */
    public Program program(String effectId) {
        return programs.get(semanticId(effectId));
    }

    public boolean omitted(String effectId) {
        return omittedEffectIds.contains(effectId);
    }

    public int size() {
        return programs.size();
    }

    public static int semanticId(String assetId) {
        if (assetId == null || assetId.isEmpty()) {
            throw new IllegalArgumentException("asset ID must be nonempty");
        }
        int value = 0x811C9DC5;
        byte[] bytes = assetId.getBytes(StandardCharsets.UTF_8);
        for (byte next : bytes) {
            value = (value ^ Byte.toUnsignedInt(next)) * 0x01000193;
        }
        value &= 0x7FFFFFFF;
        return value == 0 ? 1 : value;
    }

    private static float finite(String value) {
        float parsed = Float.parseFloat(value);
        if (!Float.isFinite(parsed)) {
            throw new NumberFormatException("non-finite float");
        }
        return parsed;
    }

    private static boolean binary(String value) {
        if (value.equals("0")) {
            return false;
        }
        if (value.equals("1")) {
            return true;
        }
        throw new NumberFormatException("boolean is not 0 or 1");
    }
}
