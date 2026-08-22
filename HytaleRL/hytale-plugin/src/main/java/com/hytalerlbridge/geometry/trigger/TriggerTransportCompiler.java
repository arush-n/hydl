package com.hytalerlbridge.geometry.trigger;

import com.hytalerlbridge.geometry.GeometryContract;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeSet;

/** Compiles native roots and exact volumes into a deterministic trigger frame. */
public final class TriggerTransportCompiler {

    private TriggerTransportCompiler() {}

    public static TriggerFrame compile(Collection<SourceCell> sourceCells) {
        Objects.requireNonNull(sourceCells, "sourceCells");
        if (sourceCells.size() > TriggerContract.MAX_CELLS) {
            return TriggerFrame.unavailable(false, true);
        }

        ArrayList<ResolvedCell> resolvedCells = new ArrayList<>();
        TreeSet<String> programIds = new TreeSet<>();
        HashSet<Integer> coordinates = new HashSet<>();
        for (SourceCell source : sourceCells) {
            Objects.requireNonNull(source, "source cell");
            int coordinate = GeometryContract.cellIndex(
                source.dx(),
                source.dy(),
                source.dz()
            );
            if (!coordinates.add(coordinate)) {
                throw new IllegalArgumentException("duplicate trigger source cell");
            }
            Roots roots = Roots.resolve(source.blockRoots(), source.fluidRoots());
            if (!roots.hasAny()) continue;
            if (!source.volumeAvailable() || source.boxes().length == 0) {
                return TriggerFrame.unavailable(true, false);
            }
            programIds.addAll(roots.ids());
            resolvedCells.add(new ResolvedCell(source, roots));
        }
        if (programIds.size() > TriggerContract.MAX_PROGRAMS_0_5_7) {
            return TriggerFrame.unavailable(false, true);
        }

        List<String> programs = List.copyOf(programIds);
        Map<String, Integer> indices = new HashMap<>();
        for (int index = 0; index < programs.size(); index++) {
            indices.put(programs.get(index), index);
        }
        resolvedCells.sort(Comparator.comparingInt(row -> GeometryContract.cellIndex(
            row.source().dx(),
            row.source().dy(),
            row.source().dz()
        )));
        List<TriggerFrame.Cell> cells = resolvedCells.stream()
            .map(row -> row.cell(indices))
            .toList();
        return new TriggerFrame(
            true,
            false,
            false,
            TriggerContract.catalogSha256(programs),
            programs,
            cells
        );
    }

    /** Native phase roots before block-first/fluid-fallback resolution. */
    public record Roots(String enter, String collision, String leave) {
        public Roots {
            validate(enter);
            validate(collision);
            validate(leave);
        }

        public static Roots none() {
            return new Roots(null, null, null);
        }

        public static Roots resolve(Roots block, Roots fluid) {
            Roots blockRoots = block == null ? none() : block;
            Roots fluidRoots = fluid == null ? none() : fluid;
            return new Roots(
                first(blockRoots.enter, fluidRoots.enter),
                first(blockRoots.collision, fluidRoots.collision),
                first(blockRoots.leave, fluidRoots.leave)
            );
        }

        public boolean hasAny() {
            return enter != null || collision != null || leave != null;
        }

        List<String> ids() {
            ArrayList<String> result = new ArrayList<>(3);
            if (enter != null) result.add(enter);
            if (collision != null) result.add(collision);
            if (leave != null) result.add(leave);
            return result;
        }

        private static String first(String preferred, String fallback) {
            return preferred == null ? fallback : preferred;
        }

        private static void validate(String value) {
            if (value != null && value.isBlank()) {
                throw new IllegalArgumentException(
                    "trigger root IDs cannot be blank"
                );
            }
        }
    }

    /** One native cell with boxes already translated to its canonical base. */
    public record SourceCell(
        int dx,
        int dy,
        int dz,
        int baseOffsetX,
        int baseOffsetY,
        int baseOffsetZ,
        boolean volumeAvailable,
        Roots blockRoots,
        Roots fluidRoots,
        double[] boxes
    ) {
        public SourceCell {
            GeometryContract.cellIndex(dx, dy, dz);
            blockRoots = blockRoots == null ? Roots.none() : blockRoots;
            fluidRoots = fluidRoots == null ? Roots.none() : fluidRoots;
            boxes = boxes == null ? new double[0] : boxes.clone();
            if (!volumeAvailable && boxes.length != 0) {
                throw new IllegalArgumentException(
                    "unavailable trigger volumes cannot carry boxes"
                );
            }
            if (boxes.length != 0) TriggerFrame.validatedBoxes(boxes);
        }

        @Override
        public double[] boxes() {
            return boxes.clone();
        }
    }

    private record ResolvedCell(SourceCell source, Roots roots) {
        TriggerFrame.Cell cell(Map<String, Integer> indices) {
            return new TriggerFrame.Cell(
                source.dx(),
                source.dy(),
                source.dz(),
                source.baseOffsetX(),
                source.baseOffsetY(),
                source.baseOffsetZ(),
                index(indices, roots.enter()),
                index(indices, roots.collision()),
                index(indices, roots.leave()),
                source.boxes()
            );
        }

        private static int index(Map<String, Integer> indices, String id) {
            return id == null ? TriggerContract.NO_PROGRAM : indices.get(id);
        }
    }
}
