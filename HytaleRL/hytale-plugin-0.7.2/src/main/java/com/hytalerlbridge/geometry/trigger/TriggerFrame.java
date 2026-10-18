package com.hytalerlbridge.geometry.trigger;

import com.hytalerlbridge.geometry.GeometryContract;
import java.util.HashSet;
import java.util.List;

/** Exact, bounded trigger volumes and phase-program bindings for one frame. */
public record TriggerFrame(
    boolean available,
    boolean dataUnavailable,
    boolean capacityExceeded,
    String programCatalogSha256,
    List<String> programIds,
    List<Cell> cells
) {
    public TriggerFrame {
        programCatalogSha256 = programCatalogSha256 == null
            ? ""
            : programCatalogSha256;
        programIds = programIds == null ? List.of() : List.copyOf(programIds);
        cells = cells == null ? List.of() : List.copyOf(cells);
        if (available == (dataUnavailable || capacityExceeded)) {
            throw new IllegalArgumentException(
                "trigger availability must agree with its failure flags"
            );
        }
        if (!available) {
            if (!programCatalogSha256.isEmpty()
                || !programIds.isEmpty()
                || !cells.isEmpty()) {
                throw new IllegalArgumentException(
                    "unavailable trigger frames must fail closed"
                );
            }
        } else if (!programCatalogSha256.matches("[0-9A-F]{64}")) {
            throw new IllegalArgumentException(
                "available trigger frames require a SHA-256 catalog identity"
            );
        } else if (programIds.size() > TriggerContract.MAX_PROGRAMS_0_5_7) {
            throw new IllegalArgumentException("trigger program capacity exceeded");
        } else if (new HashSet<>(programIds).size() != programIds.size()
            || programIds.stream().anyMatch(id -> id == null || id.isBlank())) {
            throw new IllegalArgumentException(
                "trigger program IDs must be unique and non-empty"
            );
        } else if (!programCatalogSha256.equals(
            TriggerContract.catalogSha256(programIds)
        )) {
            throw new IllegalArgumentException(
                "trigger program catalog identity does not match"
            );
        } else if (cells.size() > TriggerContract.MAX_CELLS) {
            throw new IllegalArgumentException("trigger cell capacity exceeded");
        }
        if (available) {
            HashSet<Integer> coordinates = new HashSet<>();
            for (Cell cell : cells) {
                int key = GeometryContract.cellIndex(
                    cell.dx(),
                    cell.dy(),
                    cell.dz()
                );
                if (!coordinates.add(key)) {
                    throw new IllegalArgumentException("duplicate trigger cell");
                }
                validateProgramIndex(cell.enterProgram(), programIds.size());
                validateProgramIndex(cell.collisionProgram(), programIds.size());
                validateProgramIndex(cell.leaveProgram(), programIds.size());
            }
        }
    }

    public static TriggerFrame unavailable(
        boolean dataUnavailable,
        boolean capacityExceeded
    ) {
        return new TriggerFrame(
            false,
            dataUnavailable,
            capacityExceeded,
            "",
            List.of(),
            List.of()
        );
    }

    private static void validateProgramIndex(int index, int size) {
        if (index < TriggerContract.NO_PROGRAM || index >= size) {
            throw new IllegalArgumentException("invalid trigger program index");
        }
    }

    /** Boxes are local to the canonical base cell, not the raw filler cell. */
    public record Cell(
        int dx,
        int dy,
        int dz,
        int baseOffsetX,
        int baseOffsetY,
        int baseOffsetZ,
        int enterProgram,
        int collisionProgram,
        int leaveProgram,
        double[] boxes
    ) {
        public Cell {
            GeometryContract.cellIndex(dx, dy, dz);
            boxes = boxes == null ? new double[0] : boxes.clone();
            if (enterProgram < TriggerContract.NO_PROGRAM
                || collisionProgram < TriggerContract.NO_PROGRAM
                || leaveProgram < TriggerContract.NO_PROGRAM) {
                throw new IllegalArgumentException(
                    "trigger program indices cannot be below -1"
                );
            }
            if (enterProgram == TriggerContract.NO_PROGRAM
                && collisionProgram == TriggerContract.NO_PROGRAM
                && leaveProgram == TriggerContract.NO_PROGRAM) {
                throw new IllegalArgumentException(
                    "trigger cells require at least one phase program"
                );
            }
            validateBoxes(boxes);
        }

        @Override
        public double[] boxes() {
            return boxes.clone();
        }

        public int boxCount() {
            return boxes.length / 6;
        }
    }

    static double[] validatedBoxes(double[] values) {
        double[] boxes = values == null ? new double[0] : values.clone();
        validateBoxes(boxes);
        return boxes;
    }

    private static void validateBoxes(double[] boxes) {
        if (boxes.length == 0 || boxes.length % 6 != 0) {
            throw new IllegalArgumentException(
                "trigger boxes must contain complete non-empty six-value rows"
            );
        }
        if (boxes.length / 6 > TriggerContract.MAX_BOXES_0_5_7) {
            throw new IllegalArgumentException("trigger box capacity exceeded");
        }
        for (int offset = 0; offset < boxes.length; offset += 6) {
            for (int axis = 0; axis < 3; axis++) {
                double minimum = boxes[offset + axis];
                double maximum = boxes[offset + axis + 3];
                if (!Double.isFinite(minimum)
                    || !Double.isFinite(maximum)
                    || minimum > maximum) {
                    throw new IllegalArgumentException(
                        "trigger box bounds must be finite and ordered"
                    );
                }
            }
        }
    }
}
