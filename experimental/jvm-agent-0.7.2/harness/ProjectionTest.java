import com.hytalerlbridge.policy.Projection;
import com.hytalerlbridge.policy.perception.projection.MovementProjection;
import com.hytalerlbridge.policy.perception.projection.MechanicsProjection;
import com.hytalerlbridge.policy.perception.projection.ActionMaskProjection;
import com.hytalerlbridge.policy.perception.projection.AbilityProjection;
import com.hytalerlbridge.policy.perception.projection.InventoryProjection;
import com.hytalerlbridge.policy.perception.projection.GeometryProjection;
import com.hytalerlbridge.policy.perception.projection.LightProjection;
import com.hytalerlbridge.policy.perception.projection.WorldProjection;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Certifies {@link Projection} against JAX-derived reference states.
 *
 * <p>Where {@code AssembleTest} checks the *layout* step (structured fields ->
 * 8,271 columns) this checks the *derivation* step: raw simulation quantities
 * -> structured fields. Both are needed; a correct layout over wrong values is
 * exactly as broken as the reverse, and neither test catches the other's fault.
 *
 * <p>Rows come from a real rollout, so magnitudes are realistic. The test also
 * reports how many columns actually vary across the sample — a derivation test
 * where every column is constant would pass without exercising anything.
 */
public final class ProjectionTest {

    /** Float32 rounding, since JAX evaluates in float32 and Java in double. */
    private static final float TOLERANCE = 1.0e-6f;

    public static void main(String[] args) throws IOException {
        Path directory = Path.of(args.length > 0 ? args[0] : "case/projection");
        Projection.Params params = readParams(directory.resolve("params.txt"));
        if (params.targetAbilitySlotDivisor() != 15.0) {
            throw new IllegalArgumentException(
                "arsenal attack-slot divisor must be 15, got "
                    + params.targetAbilitySlotDivisor());
        }

        List<String> lines = Files.readAllLines(directory.resolve("self_features.txt"));
        int checked = 0;
        float worst = 0.0f;
        String worstWhere = "-";
        List<String> failures = new ArrayList<>();

        float[] minimum = new float[Projection.SELF_SIZE];
        float[] maximum = new float[Projection.SELF_SIZE];
        java.util.Arrays.fill(minimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(maximum, Float.NEGATIVE_INFINITY);

        for (String line : lines) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] f = line.split("\t");
            Projection.SelfState state = new Projection.SelfState(
                Double.parseDouble(f[2]), Double.parseDouble(f[3]),
                Double.parseDouble(f[4]), Double.parseDouble(f[5]),
                Double.parseDouble(f[6]), Double.parseDouble(f[7]),
                f[8].equals("1"), Double.parseDouble(f[9]),
                Double.parseDouble(f[10]), f[11].equals("1"),
                Double.parseDouble(f[12]), Double.parseDouble(f[13]),
                Double.parseDouble(f[14]), Double.parseDouble(f[15]));

            String[] expectedText = f[16].split(",");
            float[] actual = Projection.selfFeatures(state, params);

            for (int i = 0; i < Projection.SELF_SIZE; i++) {
                float expected = Float.parseFloat(expectedText[i]);
                float delta = Math.abs(expected - actual[i]);
                if (delta > worst) {
                    worst = delta;
                    worstWhere = Projection.SELF_FEATURES[i]
                        + " (step " + f[0] + " lane " + f[1] + ")";
                }
                if (delta > TOLERANCE) {
                    failures.add(String.format(
                        "step %s lane %s %s: expected %.9f got %.9f (delta %.3e)",
                        f[0], f[1], Projection.SELF_FEATURES[i],
                        expected, actual[i], delta));
                }
                minimum[i] = Math.min(minimum[i], expected);
                maximum[i] = Math.max(maximum[i], expected);
            }
            checked++;
        }

        int varying = 0;
        for (int i = 0; i < Projection.SELF_SIZE; i++) {
            if (maximum[i] - minimum[i] > 1.0e-9f) {
                varying++;
            }
        }

        System.out.printf("rows           : %d%n", checked);
        System.out.printf("columns        : %d%n", Projection.SELF_SIZE);
        System.out.printf("varying columns: %d/%d (constant ones prove little)%n",
            varying, Projection.SELF_SIZE);
        System.out.printf("worst delta    : %.3e at %s%n", worst, worstWhere);
        System.out.printf("tolerance      : %.1e%n", TOLERANCE);
        System.out.printf("failures       : %d%n", failures.size());
        for (int i = 0; i < Projection.SELF_SIZE; i++) {
            System.out.printf("  %-26s [%+.4f, %+.4f]%s%n",
                Projection.SELF_FEATURES[i], minimum[i], maximum[i],
                maximum[i] - minimum[i] > 1.0e-9f ? "" : "   constant");
        }
        checkTarget(directory, params, failures);
        checkCombat(directory, params, failures);
        checkMovement(directory, failures);
        checkActorWorld(directory, failures);
        checkResources(directory, failures);
        checkDefense(directory, failures);
        checkStatus(directory, failures);
        checkDodge(directory, failures);
        checkAbility(directory, failures);
        checkInventory(directory, failures);
        checkLight(directory, failures);
        checkGeometry(directory, failures);

        if (!failures.isEmpty()) {
            failures.stream().limit(12).forEach(x -> System.out.println("  " + x));
            System.out.println("FAIL");
            System.exit(1);
        }
        if (checked == 0) {
            System.out.println("FAIL - no rows; the fixture is empty");
            System.exit(1);
        }
        System.out.println(
            "PASS - Java derives base and structured projection groups as JAX does");
    }

    /**
     * The target derivation, which is harder than self in three ways: it reads
     * two entities, quantises the offset through the wire scale, and masks
     * everything by visibility. The masked half of the rows is counted
     * explicitly — a derivation that ignored the mask would still pass on
     * visible rows alone.
     */
    private static void checkTarget(
        Path directory, Projection.Params params, List<String> failures
    ) throws IOException {
        Path path = directory.resolve("target_features.txt");
        if (!Files.exists(path)) {
            System.out.println("target        : fixture absent, skipped");
            return;
        }
        int rows = 0;
        int visible = 0;
        float worst = 0.0f;
        String worstWhere = "-";
        float[] minimum = new float[Projection.TARGET_SIZE];
        float[] maximum = new float[Projection.TARGET_SIZE];
        java.util.Arrays.fill(minimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(maximum, Float.NEGATIVE_INFINITY);

        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] f = line.split("\t");
            Projection.TargetState state = new Projection.TargetState(
                Double.parseDouble(f[2]), Double.parseDouble(f[3]),
                Double.parseDouble(f[4]),
                Double.parseDouble(f[5]), Double.parseDouble(f[6]),
                Double.parseDouble(f[7]),
                Double.parseDouble(f[8]),
                Double.parseDouble(f[9]), Double.parseDouble(f[10]),
                Double.parseDouble(f[11]),
                Double.parseDouble(f[12]), Double.parseDouble(f[13]),
                Double.parseDouble(f[14]),
                Double.parseDouble(f[15]), Double.parseDouble(f[16]),
                Double.parseDouble(f[17]), Double.parseDouble(f[18]),
                Double.parseDouble(f[19]), Double.parseDouble(f[20]));

            String[] expectedText = f[21].split(",");
            boolean expectedMask = f[22].equals("1");
            float[] actual = Projection.targetFeatures(state, params);

            if (expectedMask) {
                visible++;
            }
            // The visible column and the mask are the same fact; if they ever
            // disagree the masking is being applied inconsistently.
            if ((actual[15] > 0.5f) != expectedMask) {
                failures.add("sweep " + f[0] + " lane " + f[1]
                    + " target visibility disagrees with mask");
            }
            for (int i = 0; i < Projection.TARGET_SIZE; i++) {
                float expected = Float.parseFloat(expectedText[i]);
                float delta = Math.abs(expected - actual[i]);
                if (delta > worst) {
                    worst = delta;
                    worstWhere = Projection.TARGET_FEATURES[i]
                        + " (sweep " + f[0] + " lane " + f[1] + ")";
                }
                if (delta > TOLERANCE) {
                    failures.add(String.format(
                        "sweep %s lane %s %s: expected %.9f got %.9f (delta %.3e)",
                        f[0], f[1], Projection.TARGET_FEATURES[i],
                        expected, actual[i], delta));
                }
                minimum[i] = Math.min(minimum[i], expected);
                maximum[i] = Math.max(maximum[i], expected);
            }
            rows++;
        }

        int varying = 0;
        for (int i = 0; i < Projection.TARGET_SIZE; i++) {
            if (maximum[i] - minimum[i] > 1.0e-9f) {
                varying++;
            }
        }
        System.out.println();
        System.out.printf("target rows    : %d (%d visible, %d masked)%n",
            rows, visible, rows - visible);
        System.out.printf("target varying : %d/%d%n", varying, Projection.TARGET_SIZE);
        System.out.printf("target worst   : %.3e at %s%n", worst, worstWhere);
        for (int i = 0; i < Projection.TARGET_SIZE; i++) {
            System.out.printf("  %-26s [%+.4f, %+.4f]%s%n",
                Projection.TARGET_FEATURES[i], minimum[i], maximum[i],
                maximum[i] - minimum[i] > 1.0e-9f ? "" : "   constant");
        }
    }

    /**
     * The 24-column combat row, exercising both ports at once: the fixture's
     * raw inputs go through {@code targetEvidence} and then the encoder, so a
     * failure in either shows up here. The phases actually reached are printed
     * because the one-hot block is five columns that a fixture stuck in IDLE
     * would leave almost entirely unexercised.
     */
    private static void checkCombat(
        Path directory, Projection.Params params, List<String> failures
    ) throws IOException {
        Path path = directory.resolve("combat_features.txt");
        if (!Files.exists(path)) {
            System.out.println("combat        : fixture absent, skipped");
            return;
        }
        int rows = 0;
        int visible = 0;
        float worst = 0.0f;
        String worstWhere = "-";
        float[] minimum = new float[Projection.COMBAT_SIZE];
        float[] maximum = new float[Projection.COMBAT_SIZE];
        java.util.Arrays.fill(minimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(maximum, Float.NEGATIVE_INFINITY);
        java.util.Set<Integer> phases = new java.util.TreeSet<>();
        java.util.Set<Boolean> primaries = new java.util.TreeSet<>();
        java.util.Set<Integer> reportedSlots = new java.util.TreeSet<>();

        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            // Column order, 0-based, matching export_projection.py's writer:
            // 0 sweep, 1 lane, 2 primaryTarget, 3-5 agentPos, 6-8 targetPos,
            // 9-11 targetVel, 12 targetHealth, 13 targetYaw, 14 targetHeadYaw,
            // 15 targetHeadPitch, 16 perceptible, 17 phase, 18 progress,
            // 19 reportedIndex, 20-22 agentVel, 23 agentYaw,
            // 24 agentHealthFraction, 25 attackExecuting, 26 expected[24].
            String[] f = line.split("\t");
            // Positional parsing silently reads the wrong column if the writer
            // grows a field, and every value would still parse. Fail loudly.
            if (f.length != 27) {
                failures.add("combat fixture row has " + f.length
                    + " fields, expected 27 -- the exporter's column order "
                    + "and this parser have diverged");
                break;
            }
            boolean primaryTarget = f[2].equals("1");
            primaries.add(primaryTarget);
            boolean perceptible = f[16].equals("1");
            int phase = Integer.parseInt(f[17]);
            phases.add(phase);
            reportedSlots.add(Integer.parseInt(f[19]));
            if (perceptible) {
                visible++;
            }

            Projection.TargetRaw raw = new Projection.TargetRaw(
                Double.parseDouble(f[3]), Double.parseDouble(f[4]),
                Double.parseDouble(f[5]),
                Double.parseDouble(f[6]), Double.parseDouble(f[7]),
                Double.parseDouble(f[8]),
                Double.parseDouble(f[9]), Double.parseDouble(f[10]),
                Double.parseDouble(f[11]),
                Double.parseDouble(f[12]), Double.parseDouble(f[13]),
                Double.parseDouble(f[14]), Double.parseDouble(f[15]),
                perceptible,
                primaryTarget,
                phase,
                Double.parseDouble(f[18]),
                Integer.parseInt(f[19]));
            Projection.CombatSelf self = new Projection.CombatSelf(
                Double.parseDouble(f[20]), Double.parseDouble(f[21]),
                Double.parseDouble(f[22]),
                Double.parseDouble(f[23]), Double.parseDouble(f[24]),
                f[25].equals("1"));

            String[] expectedText = f[26].split(",");
            float[] actual = Projection.combatFeatures(
                self, Projection.targetEvidence(raw, params), params);

            for (int i = 0; i < Projection.COMBAT_SIZE; i++) {
                float expected = Float.parseFloat(expectedText[i]);
                float delta = Math.abs(expected - actual[i]);
                if (delta > worst) {
                    worst = delta;
                    worstWhere = Projection.COMBAT_FEATURES[i]
                        + " (sweep " + f[0] + " lane " + f[1] + ")";
                }
                if (delta > TOLERANCE) {
                    failures.add(String.format(
                        "sweep %s lane %s %s: expected %.9f got %.9f (delta %.3e)",
                        f[0], f[1], Projection.COMBAT_FEATURES[i],
                        expected, actual[i], delta));
                }
                minimum[i] = Math.min(minimum[i], expected);
                maximum[i] = Math.max(maximum[i], expected);
            }
            rows++;
        }

        int varying = 0;
        for (int i = 0; i < Projection.COMBAT_SIZE; i++) {
            if (maximum[i] - minimum[i] > 1.0e-9f) {
                varying++;
            }
        }
        System.out.println();
        System.out.printf("combat rows    : %d (%d visible, %d masked)%n",
            rows, visible, rows - visible);
        System.out.printf("combat varying : %d/%d%n", varying, Projection.COMBAT_SIZE);
        System.out.printf("combat phases  : %s of [0..4]%n", phases);
        System.out.printf("combat primary : %s%n", primaries);
        System.out.printf("combat worst   : %.3e at %s%n", worst, worstWhere);
        for (int i = 0; i < Projection.COMBAT_SIZE; i++) {
            System.out.printf("  %-32s [%+.4f, %+.4f]%s%n",
                Projection.COMBAT_FEATURES[i], minimum[i], maximum[i],
                maximum[i] - minimum[i] > 1.0e-9f ? "" : "   constant");
        }
        if (primaries.size() < 2) {
            failures.add("combat fixture only covers primaryTarget="
                + primaries + "; the gate that suppresses phase, progress and "
                + "attack index for a non-primary entity is untested");
        }
        if (reportedSlots.stream().noneMatch(slot -> slot > 0)) {
            failures.add("combat fixture never reaches a non-zero authored "
                + "ability slot; /15 and the stale legacy /4 are indistinguishable");
        }
    }

    /** Native movement bitset to the exact 23 learner columns and masks. */
    private static void checkMovement(
        Path directory, List<String> failures
    ) throws IOException {
        Path path = directory.resolve("movement_features.txt");
        if (!Files.exists(path)) {
            failures.add("movement fixture absent");
            return;
        }
        int rows = 0;
        int invalidRows = 0;
        float[] minimum = new float[Projection.MOVEMENT_STATE_SIZE];
        float[] maximum = new float[Projection.MOVEMENT_STATE_SIZE];
        boolean[] sawMaskFalse = new boolean[Projection.MOVEMENT_STATE_SIZE];
        boolean[] sawMaskTrue = new boolean[Projection.MOVEMENT_STATE_SIZE];
        java.util.Arrays.fill(minimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(maximum, Float.NEGATIVE_INFINITY);

        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] fields = line.split("\t");
            if (fields.length != 6) {
                failures.add("movement fixture row has " + fields.length
                    + " fields, expected 6 -- exporter and parser diverged");
                break;
            }
            String label = fields[0];
            int bits = Integer.parseInt(fields[1]);
            boolean available = fields[2].equals("1");
            String[] expectedValues = fields[3].split(",");
            String[] expectedMask = fields[4].split(",");
            boolean expectedInvalid = fields[5].equals("1");
            if (expectedValues.length != Projection.MOVEMENT_STATE_SIZE
                || expectedMask.length != Projection.MOVEMENT_STATE_SIZE) {
                failures.add(label + " movement fixture array width drift");
                continue;
            }

            MovementProjection.Result actual =
                Projection.movementFeatures(bits, available);
            if (actual.invalid() != expectedInvalid) {
                failures.add(label + " movement invalid: expected "
                    + expectedInvalid + " got " + actual.invalid());
            }
            if (actual.invalid()) {
                invalidRows++;
            }
            float[] values = actual.values();
            boolean[] mask = actual.mask();
            for (int index = 0; index < Projection.MOVEMENT_STATE_SIZE; index++) {
                float expected = Float.parseFloat(expectedValues[index]);
                boolean expectedAvailable = expectedMask[index].equals("1");
                if (values[index] != expected) {
                    failures.add(label + " movement "
                        + Projection.MOVEMENT_STATE_FEATURES[index]
                        + ": expected " + expected + " got " + values[index]);
                }
                if (mask[index] != expectedAvailable) {
                    failures.add(label + " movement mask "
                        + Projection.MOVEMENT_STATE_FEATURES[index]
                        + ": expected " + expectedAvailable + " got " + mask[index]);
                }
                minimum[index] = Math.min(minimum[index], expected);
                maximum[index] = Math.max(maximum[index], expected);
                sawMaskFalse[index] |= !expectedAvailable;
                sawMaskTrue[index] |= expectedAvailable;
            }
            rows++;
        }

        int varyingValues = 0;
        int varyingMasks = 0;
        for (int index = 0; index < Projection.MOVEMENT_STATE_SIZE; index++) {
            if (maximum[index] > minimum[index]) {
                varyingValues++;
            } else {
                failures.add("movement value column never varies: "
                    + Projection.MOVEMENT_STATE_FEATURES[index]);
            }
            if (sawMaskFalse[index] && sawMaskTrue[index]) {
                varyingMasks++;
            } else {
                failures.add("movement mask column never varies: "
                    + Projection.MOVEMENT_STATE_FEATURES[index]);
            }
        }
        if (rows == 0 || invalidRows == 0) {
            failures.add("movement fixture lacks rows or invalid-evidence coverage");
        }
        System.out.println();
        System.out.printf("movement rows  : %d (%d invalid)%n", rows, invalidRows);
        System.out.printf("movement values: %d/%d varying%n",
            varyingValues, Projection.MOVEMENT_STATE_SIZE);
        System.out.printf("movement masks : %d/%d varying%n",
            varyingMasks, Projection.MOVEMENT_STATE_SIZE);
    }

    /** Fluid, submersion and fall/drop evidence with independent masks. */
    private static void checkActorWorld(
        Path directory, List<String> failures
    ) throws IOException {
        Path path = directory.resolve("actor_world_features.txt");
        if (!Files.exists(path)) {
            failures.add("actor-world fixture absent");
            return;
        }
        int rows = 0;
        float worst = 0.0f;
        float[] minimum = new float[WorldProjection.ACTOR_WORLD_VALUE_SIZE];
        float[] maximum = new float[WorldProjection.ACTOR_WORLD_VALUE_SIZE];
        boolean[] maskFalse = new boolean[WorldProjection.ACTOR_WORLD_MASK_SIZE];
        boolean[] maskTrue = new boolean[WorldProjection.ACTOR_WORLD_MASK_SIZE];
        java.util.Arrays.fill(minimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(maximum, Float.NEGATIVE_INFINITY);
        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] fields = line.split("\t");
            if (fields.length != 12) {
                failures.add("actor-world fixture row has " + fields.length
                    + " fields, expected 12 -- exporter and parser diverged");
                break;
            }
            WorldProjection.ActorWorldInput input =
                new WorldProjection.ActorWorldInput(
                    fields[1].equals("1"), fields[2].equals("1"),
                    fields[3].equals("1"), fields[4].equals("1"),
                    fields[5].equals("1"), fields[6].equals("1"),
                    fields[7].equals("1"), Float.parseFloat(fields[8]),
                    Float.parseFloat(fields[9]));
            WorldProjection.ActorWorldResult actual =
                Projection.actorWorldFeatures(input);
            String[] expectedValues = fields[10].split(",");
            String[] expectedMask = fields[11].split(",");
            if (expectedValues.length != WorldProjection.ACTOR_WORLD_VALUE_SIZE
                || expectedMask.length != WorldProjection.ACTOR_WORLD_MASK_SIZE) {
                failures.add("actor-world fixture array width drift at case "
                    + fields[0]);
                continue;
            }
            float[] values = actual.values();
            boolean[] mask = actual.mask();
            for (int index = 0; index < values.length; index++) {
                float expected = Float.parseFloat(expectedValues[index]);
                float delta = Math.abs(expected - values[index]);
                worst = Math.max(worst, delta);
                if (delta > TOLERANCE) {
                    failures.add("actor-world case " + fields[0] + " value "
                        + index + ": expected " + expected + " got "
                        + values[index]);
                }
                minimum[index] = Math.min(minimum[index], expected);
                maximum[index] = Math.max(maximum[index], expected);
            }
            for (int index = 0; index < mask.length; index++) {
                boolean expected = expectedMask[index].equals("1");
                if (mask[index] != expected) {
                    failures.add("actor-world case " + fields[0] + " mask "
                        + index + ": expected " + expected + " got " + mask[index]);
                }
                maskFalse[index] |= !expected;
                maskTrue[index] |= expected;
            }
            rows++;
        }
        for (int index = 0; index < minimum.length; index++) {
            if (!(maximum[index] > minimum[index])) {
                failures.add("actor-world value column never varies: " + index);
            }
        }
        for (int index = 0; index < maskFalse.length; index++) {
            if (!(maskFalse[index] && maskTrue[index])) {
                failures.add("actor-world mask column never varies: " + index);
            }
        }
        if (rows == 0) {
            failures.add("actor-world fixture is empty");
        }
        System.out.println();
        System.out.printf("actor-world rows: %d%n", rows);
        System.out.printf("actor-world worst: %.3e%n", worst);
    }

    /** Per-entity resources normalized by authored minima and maxima. */
    private static void checkResources(
        Path directory, List<String> failures
    ) throws IOException {
        Path path = directory.resolve("resource_features.txt");
        if (!Files.exists(path)) {
            failures.add("resource fixture absent");
            return;
        }
        int rows = 0;
        float worst = 0.0f;
        float[] minimum = new float[MechanicsProjection.RESOURCE_VALUES];
        float[] maximum = new float[MechanicsProjection.RESOURCE_VALUES];
        boolean[] maskFalse = new boolean[MechanicsProjection.RESOURCE_VALUES];
        boolean[] maskTrue = new boolean[MechanicsProjection.RESOURCE_VALUES];
        java.util.Arrays.fill(minimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(maximum, Float.NEGATIVE_INFINITY);
        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] fields = line.split("\t");
            if (fields.length != 6) {
                failures.add("resource fixture row has " + fields.length
                    + " fields, expected 6 -- exporter and parser diverged");
                break;
            }
            float[] current = parseFloats(fields[1]);
            float[] authoredMinimum = parseFloats(fields[2]);
            float[] authoredMaximum = parseFloats(fields[3]);
            float[] expectedValues = parseFloats(fields[4]);
            boolean[] expectedMask = parseBooleans(fields[5]);
            if (current.length != MechanicsProjection.RESOURCE_VALUES
                || authoredMinimum.length != MechanicsProjection.RESOURCE_VALUES
                || authoredMaximum.length != MechanicsProjection.RESOURCE_VALUES
                || expectedValues.length != MechanicsProjection.RESOURCE_VALUES
                || expectedMask.length != MechanicsProjection.RESOURCE_VALUES) {
                failures.add("resource fixture array width drift at case " + fields[0]);
                continue;
            }
            var actual = Projection.resourceFeatures(
                new MechanicsProjection.ResourceInput(
                    current, authoredMinimum, authoredMaximum));
            float[] values = actual.values();
            boolean[] mask = actual.mask();
            for (int index = 0; index < values.length; index++) {
                float delta = Math.abs(expectedValues[index] - values[index]);
                worst = Math.max(worst, delta);
                if (delta > TOLERANCE) {
                    failures.add("resource case " + fields[0] + " value "
                        + index + ": expected " + expectedValues[index]
                        + " got " + values[index]);
                }
                if (mask[index] != expectedMask[index]) {
                    failures.add("resource case " + fields[0] + " mask "
                        + index + ": expected " + expectedMask[index]
                        + " got " + mask[index]);
                }
                minimum[index] = Math.min(minimum[index], expectedValues[index]);
                maximum[index] = Math.max(maximum[index], expectedValues[index]);
                maskFalse[index] |= !expectedMask[index];
                maskTrue[index] |= expectedMask[index];
            }
            rows++;
        }
        for (int index = 0; index < minimum.length; index++) {
            if (!(maximum[index] > minimum[index])) {
                failures.add("resource value column never varies: " + index);
            }
            if (!(maskFalse[index] && maskTrue[index])) {
                failures.add("resource mask column never varies: " + index);
            }
        }
        if (rows == 0) {
            failures.add("resource fixture is empty");
        }
        System.out.println();
        System.out.printf("resource rows  : %d%n", rows);
        System.out.printf("resource worst : %.3e%n", worst);
    }

    /** Guard, dodge, applied force, stamina timing and liveness. */
    private static void checkDefense(
        Path directory, List<String> failures
    ) throws IOException {
        Path path = directory.resolve("defense_features.txt");
        if (!Files.exists(path)) {
            failures.add("defense fixture absent");
            return;
        }
        int rows = 0;
        float worst = 0.0f;
        float[] minimum = new float[MechanicsProjection.DEFENSE_VALUES];
        float[] maximum = new float[MechanicsProjection.DEFENSE_VALUES];
        java.util.Arrays.fill(minimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(maximum, Float.NEGATIVE_INFINITY);
        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] fields = line.split("\t");
            if (fields.length != 12) {
                failures.add("defense fixture row has " + fields.length
                    + " fields, expected 12 -- exporter and parser diverged");
                break;
            }
            boolean[] guard = parseBooleans(fields[1]);
            boolean[] broken = parseBooleans(fields[2]);
            float[] remaining = parseFloats(fields[3]);
            float[] duration = parseFloats(fields[4]);
            float[] velocity = parseFloats(fields[5]);
            float[] force = parseFloats(fields[6]);
            float[] regen = parseFloats(fields[7]);
            float[] immunity = parseFloats(fields[8]);
            float[] health = parseFloats(fields[9]);
            float[] locomotionStamina = parseFloats(fields[10]);
            float[] expected = parseFloats(fields[11]);
            if (guard.length != 2 || broken.length != 2
                || remaining.length != 2 || duration.length != 2
                || velocity.length != 6 || force.length != 2
                || regen.length != 2 || immunity.length != 2
                || health.length != 2 || locomotionStamina.length != 2
                || expected.length != MechanicsProjection.DEFENSE_VALUES) {
                failures.add("defense fixture array width drift at case " + fields[0]);
                continue;
            }
            float[] actual = Projection.defenseFeatures(
                new MechanicsProjection.DefenseInput(
                    guard, broken, remaining, duration, velocity, force,
                    regen, immunity, health, locomotionStamina)).values();
            for (int index = 0; index < actual.length; index++) {
                float delta = Math.abs(expected[index] - actual[index]);
                worst = Math.max(worst, delta);
                if (delta > TOLERANCE) {
                    failures.add("defense case " + fields[0] + " column "
                        + index + ": expected " + expected[index]
                        + " got " + actual[index]);
                }
                minimum[index] = Math.min(minimum[index], expected[index]);
                maximum[index] = Math.max(maximum[index], expected[index]);
            }
            rows++;
        }
        for (int index = 0; index < minimum.length; index++) {
            // The target's final locomotion-stamina value is intentionally a
            // constant full bar: its sprint budget is not observable.
            if (index == 15) {
                if (minimum[index] != 1.0f || maximum[index] != 1.0f) {
                    failures.add("target locomotion stamina is not full-bar: "
                        + index);
                }
                continue;
            }
            if (!(maximum[index] > minimum[index])) {
                failures.add("defense column never varies: " + index);
            }
        }
        if (rows == 0) {
            failures.add("defense fixture is empty");
        }
        System.out.println();
        System.out.printf("defense rows   : %d%n", rows);
        System.out.printf("defense worst  : %.3e%n", worst);
    }

    /** Active-effect timing, health/resource deltas, speed and slot masks. */
    private static void checkStatus(
        Path directory, List<String> failures
    ) throws IOException {
        Path path = directory.resolve("status_features.txt");
        if (!Files.exists(path)) {
            failures.add("status fixture absent");
            return;
        }
        int rows = 0;
        float worst = 0.0f;
        float[] minimum = new float[MechanicsProjection.STATUS_VALUES];
        float[] maximum = new float[MechanicsProjection.STATUS_VALUES];
        boolean[] maskFalse = new boolean[MechanicsProjection.STATUS_SLOTS];
        boolean[] maskTrue = new boolean[MechanicsProjection.STATUS_SLOTS];
        java.util.Arrays.fill(minimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(maximum, Float.NEGATIVE_INFINITY);

        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] fields = line.split("\t");
            if (fields.length != 14) {
                failures.add("status fixture row has " + fields.length
                    + " fields, expected 14 -- exporter and parser diverged");
                break;
            }
            float[] maximumHealth = parseFloats(fields[1]);
            float[] resourceSpan = parseFloats(fields[2]);
            float[] remaining = parseFloats(fields[3]);
            float[] cycleElapsed = parseFloats(fields[4]);
            float[] cycleCooldown = parseFloats(fields[5]);
            float[] damage = parseFloats(fields[6]);
            float[] healing = parseFloats(fields[7]);
            int[] resourceId = parseInts(fields[8]);
            float[] resourceDelta = parseFloats(fields[9]);
            float[] speedMultiplier = parseFloats(fields[10]);
            boolean[] active = parseBooleans(fields[11]);
            float[] expectedValues = parseFloats(fields[12]);
            boolean[] expectedMask = parseBooleans(fields[13]);
            if (maximumHealth.length != 2
                || resourceSpan.length != MechanicsProjection.RESOURCE_VALUES
                || remaining.length != MechanicsProjection.STATUS_SLOTS
                || cycleElapsed.length != MechanicsProjection.STATUS_SLOTS
                || cycleCooldown.length != MechanicsProjection.STATUS_SLOTS
                || damage.length != MechanicsProjection.STATUS_SLOTS
                || healing.length != MechanicsProjection.STATUS_SLOTS
                || resourceId.length != MechanicsProjection.STATUS_SLOTS
                || resourceDelta.length != MechanicsProjection.STATUS_SLOTS
                || speedMultiplier.length != MechanicsProjection.STATUS_SLOTS
                || active.length != MechanicsProjection.STATUS_SLOTS
                || expectedValues.length != MechanicsProjection.STATUS_VALUES
                || expectedMask.length != MechanicsProjection.STATUS_SLOTS) {
                failures.add("status fixture array width drift at case "
                    + fields[0]);
                continue;
            }
            var actual = Projection.statusFeatures(
                new MechanicsProjection.StatusInput(
                    maximumHealth, resourceSpan, remaining, cycleElapsed,
                    cycleCooldown, damage, healing, resourceId, resourceDelta,
                    speedMultiplier, active));
            float[] values = actual.values();
            boolean[] mask = actual.mask();
            for (int index = 0; index < values.length; index++) {
                float delta = Math.abs(expectedValues[index] - values[index]);
                worst = Math.max(worst, delta);
                if (delta > TOLERANCE) {
                    failures.add("status case " + fields[0] + " value "
                        + index + ": expected " + expectedValues[index]
                        + " got " + values[index]);
                }
                minimum[index] = Math.min(minimum[index], expectedValues[index]);
                maximum[index] = Math.max(maximum[index], expectedValues[index]);
            }
            for (int index = 0; index < mask.length; index++) {
                if (mask[index] != expectedMask[index]) {
                    failures.add("status case " + fields[0] + " mask "
                        + index + ": expected " + expectedMask[index]
                        + " got " + mask[index]);
                }
                maskFalse[index] |= !expectedMask[index];
                maskTrue[index] |= expectedMask[index];
            }
            rows++;
        }
        for (int index = 0; index < minimum.length; index++) {
            if (!(maximum[index] > minimum[index])) {
                failures.add("status value column never varies: " + index);
            }
        }
        for (int index = 0; index < maskFalse.length; index++) {
            if (!(maskFalse[index] && maskTrue[index])) {
                failures.add("status mask column never varies: " + index);
            }
        }
        if (rows == 0) {
            failures.add("status fixture is empty");
        }
        System.out.println();
        System.out.printf("status rows    : %d%n", rows);
        System.out.printf("status worst   : %.3e%n", worst);
    }

    /** Native-authored dodge directions plus movement, life and stamina gates. */
    private static void checkDodge(
        Path directory, List<String> failures
    ) throws IOException {
        Path path = directory.resolve("dodge_features.txt");
        if (!Files.exists(path)) {
            failures.add("dodge fixture absent");
            return;
        }
        int rows = 0;
        boolean[] sawFalse = new boolean[ActionMaskProjection.DODGE_DIRECTION_COUNT];
        boolean[] sawTrue = new boolean[ActionMaskProjection.DODGE_DIRECTION_COUNT];
        java.util.Set<String> namedBlockers = new java.util.HashSet<>();
        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] fields = line.split("\t");
            if (fields.length != 7) {
                failures.add("dodge fixture row has " + fields.length
                    + " fields, expected 7 -- exporter and parser diverged");
                break;
            }
            boolean[] corridor = parseBooleans(fields[1]);
            boolean[] expected = parseBooleans(fields[6]);
            if (corridor.length != ActionMaskProjection.DODGE_DIRECTION_COUNT
                || expected.length
                    != ActionMaskProjection.DODGE_DIRECTION_COUNT) {
                failures.add("dodge fixture array width drift at " + fields[0]);
                continue;
            }
            boolean[] actual = Projection.dodgeActionMask(
                new ActionMaskProjection.DodgeInput(
                    corridor,
                    fields[2].equals("1"),
                    fields[3].equals("1"),
                    Float.parseFloat(fields[4]),
                    Float.parseFloat(fields[5])));
            for (int direction = 0; direction < actual.length; direction++) {
                if (actual[direction] != expected[direction]) {
                    failures.add("dodge " + fields[0] + " direction "
                        + direction + ": expected " + expected[direction]
                        + " got " + actual[direction]);
                }
                sawFalse[direction] |= !expected[direction];
                sawTrue[direction] |= expected[direction];
            }
            if (fields[0].equals("movement_disabled")
                || fields[0].equals("dead")
                || fields[0].equals("stamina_below")) {
                namedBlockers.add(fields[0]);
                for (boolean value : expected) {
                    if (value) {
                        failures.add("dodge blocker " + fields[0]
                            + " unexpectedly leaves a direction legal");
                    }
                }
            }
            rows++;
        }
        // Forward/back are deliberately not authored for native NPCs. Requiring
        // them to vary would reward a fidelity regression; left/right must vary.
        if (sawTrue[0] || sawTrue[1]) {
            failures.add("unauthored forward/back dodge became legal");
        }
        for (int direction = 2; direction < 4; direction++) {
            if (!(sawFalse[direction] && sawTrue[direction])) {
                failures.add("authored dodge direction never varies: "
                    + direction);
            }
        }
        if (namedBlockers.size() != 3) {
            failures.add("dodge fixture misses a readiness blocker: "
                + namedBlockers);
        }
        if (rows == 0) {
            failures.add("dodge fixture is empty");
        }
        System.out.println();
        System.out.printf("dodge rows     : %d%n", rows);
        System.out.println("dodge authored : left/right; forward/back closed");
    }

    /** Dynamic duration/cooldown/progress/cost features for all ability slots. */
    private static void checkAbility(
        Path directory, List<String> failures
    ) throws IOException {
        Path path = directory.resolve("ability_features.txt");
        if (!Files.exists(path)) {
            failures.add("ability fixture absent");
            return;
        }
        int rows = 0;
        float worst = 0.0f;
        float[] minimum = new float[AbilityProjection.VALUE_COUNT];
        float[] maximum = new float[AbilityProjection.VALUE_COUNT];
        boolean[] maskFalse = new boolean[AbilityProjection.SLOT_COUNT];
        boolean[] maskTrue = new boolean[AbilityProjection.SLOT_COUNT];
        boolean[] legalFalse = new boolean[AbilityProjection.SLOT_COUNT];
        boolean[] legalTrue = new boolean[AbilityProjection.SLOT_COUNT];
        java.util.Arrays.fill(minimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(maximum, Float.NEGATIVE_INFINITY);

        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] fields = line.split("\t");
            if (fields.length != 17) {
                failures.add("ability fixture row has " + fields.length
                    + " fields, expected 17 -- exporter and parser diverged");
                break;
            }
            float[] resources = parseFloats(fields[1]);
            float[] spans = parseFloats(fields[2]);
            float[] duration = parseFloats(fields[3]);
            float[] authoredCooldown = parseFloats(fields[4]);
            float[] remainingCooldown = parseFloats(fields[5]);
            int[] activeSlot = parseInts(fields[6]);
            float[] elapsed = parseFloats(fields[7]);
            float[] authoredCost = parseFloats(fields[8]);
            int[] costKind = parseInts(fields[9]);
            float[] resourceMinimum = parseFloats(fields[10]);
            boolean[] authoredMask = parseBooleans(fields[11]);
            boolean[] equipped = parseBooleans(fields[12]);
            boolean[] overflow = parseBooleans(fields[13]);
            boolean[] authoritativeLegal = parseBooleans(fields[14]);
            float[] expectedValues = parseFloats(fields[15]);
            boolean[] expectedMask = parseBooleans(fields[16]);
            if (resources.length != AbilityProjection.RESOURCE_VALUE_COUNT
                || spans.length != AbilityProjection.RESOURCE_VALUE_COUNT
                || duration.length != AbilityProjection.SLOT_COUNT
                || authoredCooldown.length != AbilityProjection.SLOT_COUNT
                || remainingCooldown.length != AbilityProjection.SLOT_COUNT
                || activeSlot.length != AbilityProjection.ENTITY_COUNT
                || elapsed.length != AbilityProjection.ENTITY_COUNT
                || authoredCost.length
                    != AbilityProjection.ABILITY_RESOURCE_VALUE_COUNT
                || costKind.length
                    != AbilityProjection.ABILITY_RESOURCE_VALUE_COUNT
                || resourceMinimum.length
                    != AbilityProjection.ABILITY_RESOURCE_VALUE_COUNT
                || authoredMask.length != AbilityProjection.SLOT_COUNT
                || equipped.length != AbilityProjection.ENTITY_COUNT
                || overflow.length != AbilityProjection.ENTITY_COUNT
                || authoritativeLegal.length != AbilityProjection.SLOT_COUNT
                || expectedValues.length != AbilityProjection.VALUE_COUNT
                || expectedMask.length != AbilityProjection.SLOT_COUNT) {
                failures.add("ability fixture array width drift at case "
                    + fields[0]);
                continue;
            }
            var actual = Projection.abilityFeatures(
                new AbilityProjection.Input(
                    resources, spans, duration, authoredCooldown,
                    remainingCooldown, activeSlot, elapsed, authoredCost,
                    costKind, resourceMinimum, authoredMask, equipped,
                    overflow, authoritativeLegal, new int[] {-1, -1}));
            float[] values = actual.values();
            boolean[] mask = actual.mask();
            boolean[] legal = actual.legal();
            for (int index = 0; index < values.length; index++) {
                float delta = Math.abs(expectedValues[index] - values[index]);
                worst = Math.max(worst, delta);
                if (delta > TOLERANCE) {
                    failures.add("ability case " + fields[0] + " value "
                        + index + ": expected " + expectedValues[index]
                        + " got " + values[index]);
                }
                minimum[index] = Math.min(minimum[index], expectedValues[index]);
                maximum[index] = Math.max(maximum[index], expectedValues[index]);
            }
            for (int index = 0; index < mask.length; index++) {
                if (mask[index] != expectedMask[index]) {
                    failures.add("ability case " + fields[0] + " mask "
                        + index + ": expected " + expectedMask[index]
                        + " got " + mask[index]);
                }
                if (legal[index] != authoritativeLegal[index]) {
                    failures.add("ability case " + fields[0] + " legal "
                        + index + ": authoritative value was not preserved");
                }
                maskFalse[index] |= !expectedMask[index];
                maskTrue[index] |= expectedMask[index];
                legalFalse[index] |= !authoritativeLegal[index];
                legalTrue[index] |= authoritativeLegal[index];
            }
            rows++;
        }
        int varyingValues = 0;
        int varyingMasks = 0;
        int varyingLegal = 0;
        for (int index = 0; index < minimum.length; index++) {
            if (maximum[index] > minimum[index]) {
                varyingValues++;
            } else {
                failures.add("ability value column never varies: " + index);
            }
        }
        for (int index = 0; index < maskFalse.length; index++) {
            if (maskFalse[index] && maskTrue[index]) {
                varyingMasks++;
            } else {
                failures.add("ability mask column never varies: " + index);
            }
            if (legalFalse[index] && legalTrue[index]) {
                varyingLegal++;
            } else {
                failures.add("ability legal column never varies: " + index);
            }
        }
        if (rows == 0) {
            failures.add("ability fixture is empty");
        }
        System.out.println();
        System.out.printf("ability rows   : %d%n", rows);
        System.out.printf("ability values : %d/%d varying%n",
            varyingValues, AbilityProjection.VALUE_COUNT);
        System.out.printf("ability masks  : %d/%d varying%n",
            varyingMasks, AbilityProjection.SLOT_COUNT);
        System.out.printf("ability legal  : %d/%d varying%n",
            varyingLegal, AbilityProjection.SLOT_COUNT);
        System.out.printf("ability worst  : %.3e%n", worst);
    }

    /** Six container rows and 76 quantity-aware semantic stack tokens. */
    private static void checkInventory(
        Path directory, List<String> failures
    ) throws IOException {
        Path path = directory.resolve("inventory_features.txt");
        if (!Files.exists(path)) {
            failures.add("inventory fixture absent");
            return;
        }
        int rows = 0;
        float worst = 0.0f;
        boolean availableFalse = false;
        boolean availableTrue = false;
        float[] containerMinimum = new float[InventoryProjection.CONTAINER_COUNT];
        float[] containerMaximum = new float[InventoryProjection.CONTAINER_COUNT];
        float[] tokenMinimum = new float[InventoryProjection.TOKEN_VALUE_COUNT];
        float[] tokenMaximum = new float[InventoryProjection.TOKEN_VALUE_COUNT];
        boolean[] containerMaskFalse =
            new boolean[InventoryProjection.CONTAINER_COUNT];
        boolean[] containerMaskTrue =
            new boolean[InventoryProjection.CONTAINER_COUNT];
        boolean[] tokenMaskFalse = new boolean[InventoryProjection.TOKEN_CAPACITY];
        boolean[] tokenMaskTrue = new boolean[InventoryProjection.TOKEN_CAPACITY];
        java.util.Arrays.fill(containerMinimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(containerMaximum, Float.NEGATIVE_INFINITY);
        java.util.Arrays.fill(tokenMinimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(tokenMaximum, Float.NEGATIVE_INFINITY);

        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] fields = line.split("\t");
            if (fields.length != 16) {
                failures.add("inventory fixture row has " + fields.length
                    + " fields, expected 16 -- exporter and parser diverged");
                break;
            }
            boolean inputAvailable = fields[1].equals("1");
            boolean[] containerAvailable = parseBooleans(fields[2]);
            int[] containerCapacity = parseInts(fields[3]);
            boolean[] sourceTokenMask = parseBooleans(fields[4]);
            int[] containerId = parseInts(fields[5]);
            int[] containerSlot = parseInts(fields[6]);
            int[] itemId = parseInts(fields[7]);
            int[] quantity = parseInts(fields[8]);
            float[] durability = parseFloats(fields[9]);
            boolean[] active = parseBooleans(fields[10]);
            boolean expectedAvailable = fields[11].equals("1");
            float[] expectedContainer = parseFloats(fields[12]);
            boolean[] expectedContainerMask = parseBooleans(fields[13]);
            float[] expectedTokens = parseFloats(fields[14]);
            boolean[] expectedTokenMask = parseBooleans(fields[15]);
            if (containerAvailable.length != InventoryProjection.CONTAINER_COUNT
                || containerCapacity.length != InventoryProjection.CONTAINER_COUNT
                || sourceTokenMask.length != InventoryProjection.TOKEN_CAPACITY
                || containerId.length != InventoryProjection.TOKEN_CAPACITY
                || containerSlot.length != InventoryProjection.TOKEN_CAPACITY
                || itemId.length != InventoryProjection.TOKEN_CAPACITY
                || quantity.length != InventoryProjection.TOKEN_CAPACITY
                || durability.length != InventoryProjection.TOKEN_CAPACITY
                || active.length != InventoryProjection.TOKEN_CAPACITY
                || expectedContainer.length
                    != InventoryProjection.CONTAINER_COUNT
                || expectedContainerMask.length
                    != InventoryProjection.CONTAINER_COUNT
                || expectedTokens.length != InventoryProjection.TOKEN_VALUE_COUNT
                || expectedTokenMask.length
                    != InventoryProjection.TOKEN_CAPACITY) {
                failures.add("inventory fixture array width drift at " + fields[0]);
                continue;
            }
            var actual = Projection.inventoryFeatures(
                new InventoryProjection.Input(
                    inputAvailable, containerAvailable, containerCapacity,
                    sourceTokenMask, containerId, containerSlot, itemId,
                    quantity, durability, active));
            if (actual.available() != expectedAvailable) {
                failures.add("inventory " + fields[0] + " available: expected "
                    + expectedAvailable + " got " + actual.available());
            }
            availableFalse |= !expectedAvailable;
            availableTrue |= expectedAvailable;
            float[] container = actual.containerValues();
            boolean[] containerMask = actual.containerMask();
            float[] tokens = actual.tokenValues();
            boolean[] tokenMask = actual.tokenMask();
            for (int index = 0; index < container.length; index++) {
                float delta = Math.abs(expectedContainer[index] - container[index]);
                worst = Math.max(worst, delta);
                if (delta > TOLERANCE) {
                    failures.add("inventory " + fields[0] + " container "
                        + index + ": expected " + expectedContainer[index]
                        + " got " + container[index]);
                }
                if (containerMask[index] != expectedContainerMask[index]) {
                    failures.add("inventory " + fields[0] + " container mask "
                        + index + " differs");
                }
                containerMinimum[index] = Math.min(
                    containerMinimum[index], expectedContainer[index]);
                containerMaximum[index] = Math.max(
                    containerMaximum[index], expectedContainer[index]);
                containerMaskFalse[index] |= !expectedContainerMask[index];
                containerMaskTrue[index] |= expectedContainerMask[index];
            }
            for (int index = 0; index < tokens.length; index++) {
                float delta = Math.abs(expectedTokens[index] - tokens[index]);
                worst = Math.max(worst, delta);
                if (delta > TOLERANCE) {
                    failures.add("inventory " + fields[0] + " token value "
                        + index + ": expected " + expectedTokens[index]
                        + " got " + tokens[index]);
                }
                tokenMinimum[index] = Math.min(tokenMinimum[index], expectedTokens[index]);
                tokenMaximum[index] = Math.max(tokenMaximum[index], expectedTokens[index]);
            }
            for (int index = 0; index < tokenMask.length; index++) {
                if (tokenMask[index] != expectedTokenMask[index]) {
                    failures.add("inventory " + fields[0] + " token mask "
                        + index + " differs");
                }
                tokenMaskFalse[index] |= !expectedTokenMask[index];
                tokenMaskTrue[index] |= expectedTokenMask[index];
            }
            if (fields[0].equals("fixed_over_capacity") && expectedAvailable) {
                failures.add("fixed-container overflow did not close inventory");
            }
            rows++;
        }
        int varyingContainer = 0;
        int varyingContainerMask = 0;
        int varyingTokens = 0;
        int varyingTokenMask = 0;
        for (int index = 0; index < containerMinimum.length; index++) {
            if (containerMaximum[index] > containerMinimum[index]) {
                varyingContainer++;
            } else {
                failures.add("inventory container value never varies: " + index);
            }
            if (containerMaskFalse[index] && containerMaskTrue[index]) {
                varyingContainerMask++;
            } else {
                failures.add("inventory container mask never varies: " + index);
            }
        }
        for (int index = 0; index < tokenMinimum.length; index++) {
            if (tokenMaximum[index] > tokenMinimum[index]) {
                varyingTokens++;
            } else {
                failures.add("inventory token value never varies: " + index);
            }
        }
        for (int index = 0; index < tokenMaskFalse.length; index++) {
            if (tokenMaskFalse[index] && tokenMaskTrue[index]) {
                varyingTokenMask++;
            } else {
                failures.add("inventory token mask never varies: " + index);
            }
        }
        if (!(availableFalse && availableTrue)) {
            failures.add("inventory availability never varies");
        }
        if (rows == 0) {
            failures.add("inventory fixture is empty");
        }
        System.out.println();
        System.out.printf("inventory rows : %d%n", rows);
        System.out.printf("inventory containers: %d/%d values, %d/%d masks varying%n",
            varyingContainer, InventoryProjection.CONTAINER_COUNT,
            varyingContainerMask, InventoryProjection.CONTAINER_COUNT);
        System.out.printf("inventory tokens: %d/%d values, %d/%d masks varying%n",
            varyingTokens, InventoryProjection.TOKEN_VALUE_COUNT,
            varyingTokenMask, InventoryProjection.TOKEN_CAPACITY);
        System.out.printf("inventory worst: %.3e%n", worst);
    }

    /** Geometry-aligned sky, RGB block-light and tint projection. */
    private static void checkLight(
        Path directory, List<String> failures
    ) throws IOException {
        Path path = directory.resolve("light_features.txt");
        if (!Files.exists(path)) {
            failures.add("light fixture absent");
            return;
        }
        int rows = 0;
        float worst = 0.0f;
        boolean availableFalse = false;
        boolean availableTrue = false;
        float[] minimum = new float[LightProjection.VALUE_COUNT];
        float[] maximum = new float[LightProjection.VALUE_COUNT];
        boolean[] maskFalse = new boolean[LightProjection.TOKEN_CAPACITY];
        boolean[] maskTrue = new boolean[LightProjection.TOKEN_CAPACITY];
        java.util.Arrays.fill(minimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(maximum, Float.NEGATIVE_INFINITY);

        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] fields = line.split("\t");
            if (fields.length != 13) {
                failures.add("light fixture row has " + fields.length
                    + " fields, expected 13 -- exporter and parser diverged");
                break;
            }
            boolean sourceAvailable = fields[1].equals("1");
            boolean[] sourceMask = parseBooleans(fields[2]);
            boolean[] lightValid = parseBooleans(fields[3]);
            int[] sky = parseInts(fields[4]);
            int[] block = parseInts(fields[5]);
            int[] tint = parseInts(fields[6]);
            boolean[] geometryMask = parseBooleans(fields[7]);
            boolean geometryAvailable = fields[8].equals("1");
            boolean actorValid = fields[9].equals("1");
            boolean expectedAvailable = fields[10].equals("1");
            float[] expected = parseFloats(fields[11]);
            boolean[] expectedMask = parseBooleans(fields[12]);
            if (sourceMask.length != LightProjection.TOKEN_CAPACITY
                || lightValid.length
                    != LightProjection.TOKEN_CAPACITY
                        * LightProjection.CHANNEL_COUNT
                || sky.length != LightProjection.TOKEN_CAPACITY
                || block.length != LightProjection.TOKEN_CAPACITY * 3
                || tint.length != LightProjection.TOKEN_CAPACITY * 3
                || geometryMask.length != LightProjection.TOKEN_CAPACITY
                || expected.length != LightProjection.VALUE_COUNT
                || expectedMask.length != LightProjection.TOKEN_CAPACITY) {
                failures.add("light fixture array width drift at " + fields[0]);
                continue;
            }
            var actual = Projection.lightFeatures(new LightProjection.Input(
                sourceAvailable, sourceMask, lightValid, sky, block, tint,
                geometryMask, geometryAvailable, actorValid));
            if (actual.available() != expectedAvailable) {
                failures.add("light " + fields[0] + " available: expected "
                    + expectedAvailable + " got " + actual.available());
            }
            availableFalse |= !expectedAvailable;
            availableTrue |= expectedAvailable;
            float[] values = actual.values();
            boolean[] tokenMask = actual.tokenMask();
            for (int index = 0; index < values.length; index++) {
                float delta = Math.abs(expected[index] - values[index]);
                worst = Math.max(worst, delta);
                if (delta > TOLERANCE) {
                    failures.add("light " + fields[0] + " value " + index
                        + ": expected " + expected[index] + " got "
                        + values[index]);
                }
                minimum[index] = Math.min(minimum[index], expected[index]);
                maximum[index] = Math.max(maximum[index], expected[index]);
            }
            for (int index = 0; index < tokenMask.length; index++) {
                if (tokenMask[index] != expectedMask[index]) {
                    failures.add("light " + fields[0] + " token mask "
                        + index + " differs");
                }
                maskFalse[index] |= !expectedMask[index];
                maskTrue[index] |= expectedMask[index];
            }
            if ((fields[0].startsWith("malformed_")
                    || fields[0].equals("geometry_mask_mismatch"))
                && expectedAvailable) {
                failures.add("light fail-closed case remained available: "
                    + fields[0]);
            }
            rows++;
        }

        int varyingValues = 0;
        int varyingMasks = 0;
        for (int index = 0; index < minimum.length; index++) {
            if (maximum[index] > minimum[index]) {
                varyingValues++;
            } else {
                failures.add("light value never varies: " + index);
            }
        }
        for (int index = 0; index < maskFalse.length; index++) {
            if (maskFalse[index] && maskTrue[index]) {
                varyingMasks++;
            } else {
                failures.add("light token mask never varies: " + index);
            }
        }
        if (!(availableFalse && availableTrue)) {
            failures.add("light availability never varies");
        }
        if (rows == 0) {
            failures.add("light fixture is empty");
        }
        System.out.println();
        System.out.printf("light rows     : %d%n", rows);
        System.out.printf("light values   : %d/%d varying%n",
            varyingValues, LightProjection.VALUE_COUNT);
        System.out.printf("light masks    : %d/%d varying%n",
            varyingMasks, LightProjection.TOKEN_CAPACITY);
        System.out.printf("light worst    : %.3e%n", worst);
    }

    /** Fixed-shape geometry-token encoding; World still owns selection/LOS. */
    private static void checkGeometry(
        Path directory, List<String> failures
    ) throws IOException {
        Path path = directory.resolve("geometry_features.txt");
        if (!Files.exists(path)) {
            failures.add("geometry fixture absent");
            return;
        }
        int rows = 0;
        float worst = 0.0f;
        boolean availableFalse = false;
        boolean availableTrue = false;
        float[] minimum = new float[GeometryProjection.VALUE_COUNT];
        float[] maximum = new float[GeometryProjection.VALUE_COUNT];
        boolean[] maskFalse = new boolean[GeometryProjection.TOKEN_CAPACITY];
        boolean[] maskTrue = new boolean[GeometryProjection.TOKEN_CAPACITY];
        java.util.Arrays.fill(minimum, Float.POSITIVE_INFINITY);
        java.util.Arrays.fill(maximum, Float.NEGATIVE_INFINITY);

        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] fields = line.split("\t");
            if (fields.length != 22) {
                failures.add("geometry fixture row has " + fields.length
                    + " fields, expected 22 -- exporter and parser diverged");
                break;
            }
            boolean sourceAvailable = fields[1].equals("1");
            boolean[] tokenMask = parseBooleans(fields[2]);
            int[] tokenKind = parseInts(fields[3]);
            int[] provenance = parseInts(fields[4]);
            float[] relative = parseFloats(fields[5]);
            float[] clearance = parseFloats(fields[6]);
            int[] semantic = parseInts(fields[7]);
            boolean[] dynamic = parseBooleans(fields[8]);
            int edgeCapacity = Integer.parseInt(fields[9]);
            boolean[] edgeMask = parseBooleans(fields[10]);
            int[] destination = parseInts(fields[11]);
            float[] edgeCost = parseFloats(fields[12]);
            int[] edgeKind = parseInts(fields[13]);
            int[] edgeFlags = parseInts(fields[14]);
            int boxCapacity = Integer.parseInt(fields[15]);
            boolean[] boxMask = parseBooleans(fields[16]);
            float[] boxes = parseFloats(fields[17]);
            boolean actorValid = fields[18].equals("1");
            boolean expectedAvailable = fields[19].equals("1");
            float[] expected = parseFloats(fields[20]);
            boolean[] expectedMask = parseBooleans(fields[21]);
            if (tokenMask.length != GeometryProjection.TOKEN_CAPACITY
                || tokenKind.length != GeometryProjection.TOKEN_CAPACITY
                || provenance.length != GeometryProjection.TOKEN_CAPACITY
                || relative.length != GeometryProjection.TOKEN_CAPACITY * 3
                || clearance.length != GeometryProjection.TOKEN_CAPACITY
                || semantic.length != GeometryProjection.TOKEN_CAPACITY
                || dynamic.length != GeometryProjection.TOKEN_CAPACITY
                || edgeMask.length
                    != GeometryProjection.TOKEN_CAPACITY * edgeCapacity
                || destination.length != edgeMask.length
                || edgeCost.length != edgeMask.length
                || edgeKind.length != edgeMask.length
                || edgeFlags.length != edgeMask.length
                || boxMask.length
                    != GeometryProjection.TOKEN_CAPACITY * boxCapacity
                || boxes.length != boxMask.length * 6
                || expected.length != GeometryProjection.VALUE_COUNT
                || expectedMask.length != GeometryProjection.TOKEN_CAPACITY) {
                failures.add("geometry fixture array width drift at " + fields[0]);
                continue;
            }
            var actual = Projection.geometryFeatures(
                new GeometryProjection.Input(
                    sourceAvailable, tokenMask, tokenKind, provenance,
                    relative, clearance, semantic, dynamic, edgeMask,
                    destination, edgeCost, edgeKind, edgeFlags, boxMask,
                    boxes, actorValid));
            if (actual.available() != expectedAvailable) {
                failures.add("geometry " + fields[0] + " available: expected "
                    + expectedAvailable + " got " + actual.available());
            }
            availableFalse |= !expectedAvailable;
            availableTrue |= expectedAvailable;
            float[] values = actual.values();
            boolean[] actualMask = actual.tokenMask();
            for (int index = 0; index < values.length; index++) {
                float delta = Math.abs(expected[index] - values[index]);
                worst = Math.max(worst, delta);
                if (delta > TOLERANCE) {
                    failures.add("geometry " + fields[0] + " value " + index
                        + ": expected " + expected[index] + " got "
                        + values[index]);
                }
                minimum[index] = Math.min(minimum[index], expected[index]);
                maximum[index] = Math.max(maximum[index], expected[index]);
            }
            for (int index = 0; index < actualMask.length; index++) {
                if (actualMask[index] != expectedMask[index]) {
                    failures.add("geometry " + fields[0] + " token mask "
                        + index + " differs");
                }
                maskFalse[index] |= !expectedMask[index];
                maskTrue[index] |= expectedMask[index];
            }
            if (fields[0].equals("short_axes_padded") && !expectedAvailable) {
                failures.add("geometry rejected legal short edge/box axes");
            }
            rows++;
        }

        int varyingValues = 0;
        int varyingMasks = 0;
        for (int index = 0; index < minimum.length; index++) {
            if (maximum[index] > minimum[index]) {
                varyingValues++;
            } else {
                failures.add("geometry value never varies: " + index);
            }
        }
        for (int index = 0; index < maskFalse.length; index++) {
            if (maskFalse[index] && maskTrue[index]) {
                varyingMasks++;
            } else {
                failures.add("geometry token mask never varies: " + index);
            }
        }
        if (!(availableFalse && availableTrue)) {
            failures.add("geometry availability never varies");
        }
        if (rows == 0) {
            failures.add("geometry fixture is empty");
        }
        System.out.println();
        System.out.printf("geometry rows  : %d%n", rows);
        System.out.printf("geometry values: %d/%d varying%n",
            varyingValues, GeometryProjection.VALUE_COUNT);
        System.out.printf("geometry masks : %d/%d varying%n",
            varyingMasks, GeometryProjection.TOKEN_CAPACITY);
        System.out.printf("geometry worst : %.3e%n", worst);
    }

    private static float[] parseFloats(String csv) {
        String[] text = csv.split(",");
        float[] values = new float[text.length];
        for (int index = 0; index < text.length; index++) {
            values[index] = Float.parseFloat(text[index]);
        }
        return values;
    }

    private static boolean[] parseBooleans(String csv) {
        String[] text = csv.split(",");
        boolean[] values = new boolean[text.length];
        for (int index = 0; index < text.length; index++) {
            values[index] = text[index].equals("1");
        }
        return values;
    }

    private static int[] parseInts(String csv) {
        String[] text = csv.split(",");
        int[] values = new int[text.length];
        for (int index = 0; index < text.length; index++) {
            values[index] = Integer.parseInt(text[index]);
        }
        return values;
    }

    private static Projection.Params readParams(Path path) throws IOException {
        Map<String, Double> values = new HashMap<>();
        for (String line : Files.readAllLines(path)) {
            if (line.isBlank()) {
                continue;
            }
            String[] parts = line.split("\t");
            values.put(parts[0], Double.parseDouble(parts[1]));
        }
        return new Projection.Params(
            values.get("agent_max_speed"),
            values.getOrDefault("agent_maximum_gait_speed", values.get("agent_max_speed")),
            values.get("vertical_speed_scale"),
            values.get("agent_max_health"),
            values.get("agent_attack_pause_max_seconds"),
            values.get("regen_delay_ticks"),
            values.get("agent_walk_max_fall_speed"),
            values.get("loaded_dt"),
            values.get("wire_fixed_point_scale"),
            values.get("target_chase_speed"),
            values.get("target_max_health"),
            values.get("sensor_range"),
            values.get("facing_error_degrees_scale"),
            values.get("head_pitch_degrees_scale"),
            values.get("target_attack_index_divisor"));
    }
}
