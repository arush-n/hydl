package com.hytalerlbridge.policy.world.model;

/** Trigger-specific native identities retained behind one visible block slot. */
public record BlockCandidateBinding(
    boolean available,
    int rawX,
    int rawY,
    int rawZ,
    int actionX,
    int actionY,
    int actionZ,
    String expectedBlockId,
    WorldVerbBinding primary,
    WorldVerbBinding secondary,
    WorldVerbBinding use
) {
    public BlockCandidateBinding {
        expectedBlockId = canonical(expectedBlockId);
        primary = primary == null ? WorldVerbBinding.empty() : primary;
        secondary = secondary == null ? WorldVerbBinding.empty() : secondary;
        use = use == null ? WorldVerbBinding.empty() : use;
        if (available != !expectedBlockId.isEmpty()) {
            throw new IllegalArgumentException("invalid block candidate identity");
        }
        if (!available && (primary.available() || secondary.available()
            || use.available())) {
            throw new IllegalArgumentException(
                "an unavailable candidate cannot expose action bindings");
        }
        if (available) {
            validateBinding(primary, rawX, rawY, rawZ, actionX, actionY,
                actionZ, expectedBlockId);
            validateBinding(secondary, rawX, rawY, rawZ, actionX, actionY,
                actionZ, expectedBlockId);
            validateBinding(use, rawX, rawY, rawZ, actionX, actionY,
                actionZ, expectedBlockId);
        }
    }

    public WorldVerbBinding forAction(boolean useRequested, int trigger) {
        if (useRequested) return use;
        if (trigger == 1) return primary;
        if (trigger == 2) return secondary;
        return WorldVerbBinding.empty();
    }

    public static BlockCandidateBinding empty() {
        return new BlockCandidateBinding(
            false, 0, 0, 0, 0, 0, 0, "",
            WorldVerbBinding.empty(), WorldVerbBinding.empty(),
            WorldVerbBinding.empty());
    }

    private static void validateBinding(
        WorldVerbBinding binding,
        int rawX,
        int rawY,
        int rawZ,
        int actionX,
        int actionY,
        int actionZ,
        String expectedBlockId
    ) {
        if (!binding.available()) return;
        if (binding.verb().equals("break_block")
            || binding.verb().equals("use")) {
            if (binding.targetX() != actionX
                || binding.targetY() != actionY
                || binding.targetZ() != actionZ
                || !binding.expectedBlockId().equals(expectedBlockId)) {
                throw new IllegalArgumentException(
                    "Break/Use binding must target the canonical action cell");
            }
            return;
        }
        if (!binding.verb().equals("place_block")) {
            throw new IllegalArgumentException("unsupported block binding verb");
        }
        int[] offset = faceOffset(binding.blockFace());
        if (binding.targetX() != rawX + offset[0]
            || binding.targetY() != rawY + offset[1]
            || binding.targetZ() != rawZ + offset[2]) {
            throw new IllegalArgumentException(
                "Place binding must target the selected raw click face");
        }
    }

    private static int[] faceOffset(int face) {
        return switch (face) {
            case 1 -> new int[] {0, 1, 0};
            case 2 -> new int[] {0, -1, 0};
            case 3 -> new int[] {0, 0, -1};
            case 4 -> new int[] {0, 0, 1};
            case 5 -> new int[] {1, 0, 0};
            case 6 -> new int[] {-1, 0, 0};
            default -> throw new IllegalArgumentException(
                "Place binding requires a protocol block face");
        };
    }

    private static String canonical(String value) {
        if (value == null) return "";
        if (!value.equals(value.trim()) || value.indexOf('\0') >= 0) {
            throw new IllegalArgumentException(
                "block candidate identity must be canonical");
        }
        return value;
    }
}
