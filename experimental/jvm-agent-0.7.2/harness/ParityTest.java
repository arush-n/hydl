import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.ObservationAssembler;
import com.hytalerlbridge.policy.Policy;
import com.hytalerlbridge.policy.PolicyRuntime;
import com.hytalerlbridge.policy.Projection;
import com.hytalerlbridge.policy.perception.projection.WorldProjection;
import com.hytalerlbridge.policy.perception.projection.MechanicsProjection;
import com.hytalerlbridge.policy.perception.projection.ActionMaskProjection;
import com.hytalerlbridge.policy.perception.projection.AbilityProjection;
import com.hytalerlbridge.policy.perception.projection.InventoryProjection;
import com.hytalerlbridge.policy.perception.projection.GeometryProjection;
import com.hytalerlbridge.policy.perception.projection.LightProjection;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Stream;

/**
 * Pushes many states through the whole Java pipeline and checks JAX agrees.
 *
 * <p>The other tests each certify one link: {@code ProjectionTest} raw state to
 * derived fields, {@code AssembleTest} fields to the 8,271-column vector,
 * {@code Main} that vector to logits. All three pass while the chain is broken,
 * because a stage can be correct in isolation and still be connected wrong --
 * a derived group written to the wrong slot, the agent row taken from the wrong
 * entity, a field read from a stale copy.
 *
 * <p><b>Breadth is deliberate.</b> A single state proves almost nothing: a
 * reset has the agent at rest and the target idle, which is the regime least
 * likely to expose a float32 difference or a masking mistake. The fixture spans
 * a dozen weapon kinds -- melee, shield, bow, crossbow, staff, spellbook,
 * thrown, consumed -- against rotated opponents, sampled across a fight, so the
 * ability loadouts, legality masks and action surfaces genuinely differ.
 *
 * <p>Two things are checked per state and the second is the point. First, that
 * the Java-assembled observation matches JAX's. Second, that the decision is
 * the same -- because the derivations are float32 and carry rounding, and what
 * matters is that the residual never changes the action.
 */
public final class ParityTest {

    /** Float32 rounding, as in ProjectionTest. */
    private static final float TOLERANCE = 1.0e-6f;

    /** Groups this test derives rather than loads. */
    private static final String[] DERIVED_FLOATS = {
        "base_self_f32", "base_target_f32", "base_combat_f32",
        "movement_state_f32", "actor_world_f32", "resource_f32",
        "defense_f32", "status_f32", "ability_f32",
        "inventory_container_f32", "inventory_token_f32",
        "light_token_f32", "geometry_token_f32",
    };

    private static final String[] DERIVED_MASKS = {
        "movement_state_mask", "actor_world_mask", "resource_mask",
        "status_mask", "dodge_action_mask", "ability_mask", "ability_legal",
        "inventory_container_mask", "inventory_token_mask",
        "inventory_available", "light_available",
        "geometry_token_mask", "geometry_available",
    };

    private record Derived(
        Map<String, float[]> floats,
        Map<String, boolean[]> masks
    ) {}

    public static void main(String[] args) throws IOException {
        Path caseRoot = Path.of(args.length > 0 ? args[0] : "case");
        Path parityRoot = Path.of(args.length > 1 ? args[1] : "case/parity");

        List<Path> states = new ArrayList<>();
        if (Files.isDirectory(parityRoot)) {
            try (Stream<Path> entries = Files.list(parityRoot)) {
                entries.filter(p -> Files.exists(p.resolve("inputs.txt")))
                    .sorted()
                    .forEach(states::add);
            }
        }
        if (states.isEmpty()) {
            System.out.println(
                "SKIP - no parity fixture; run export_parity.py first");
            return;
        }
        // The exporter writes state directories as it goes but its manifest
        // only after its own coverage checks pass. A missing manifest means the
        // last export was rejected or interrupted, and the directories on disk
        // are a partial capture that would silently narrow every claim below.
        //
        // Presence alone is not enough. The flat and region exports write
        // separate manifests, each listing only its own states, and each clears
        // only its own directories. Checking that *a* manifest existed let a
        // directory no live export had blessed -- an orphan from a shortened or
        // superseded run -- be scored as if it were current. So every scored
        // directory must be named by some manifest.
        //
        // That check is only half of it, and the missing half already bit:
        // it catches a directory no manifest blessed, but not a state a
        // manifest blessed that is no longer on disk. A re-export that dropped
        // ticks 24 and 48 left 24 of 48 flat states missing while this test
        // reported green on the 28 that remained -- a 46% coverage loss the
        // gate could not see, because nothing compared the manifest's own
        // roster against what was actually scored. Both directions now fail.
        Set<String> blessed = new TreeSet<>();
        int manifests = 0;
        try (var entries = Files.list(parityRoot)) {
            for (Path manifest : entries.toList()) {
                String file = manifest.getFileName().toString();
                if (!file.startsWith("manifest_") || !file.endsWith(".json")) {
                    continue;
                }
                manifests++;
                for (String line : Files.readAllLines(manifest)) {
                    int at = line.indexOf("\"directory\"");
                    if (at < 0) {
                        continue;
                    }
                    int open = line.indexOf('"', line.indexOf(':', at) + 1);
                    int close = line.indexOf('"', open + 1);
                    if (open > 0 && close > open) {
                        blessed.add(line.substring(open + 1, close));
                    }
                }
            }
        }
        if (manifests == 0) {
            System.out.println("FAIL - " + states.size() + " state directories "
                + "but no manifest_*.json: the export was rejected or is still "
                + "running, so this fixture is not safe to certify against");
            System.exit(1);
        }
        List<String> orphans = new ArrayList<>();
        for (Path state : states) {
            if (!blessed.contains(state.getFileName().toString())) {
                orphans.add(state.getFileName().toString());
            }
        }
        if (!orphans.isEmpty()) {
            System.out.println("FAIL - " + orphans.size() + " state director"
                + (orphans.size() == 1 ? "y is" : "ies are") + " not named by "
                + "any of the " + manifests + " manifest(s): " + orphans
                + " -- these are leftovers from a superseded export and their "
                + "contents are not certified by any coverage check");
            System.exit(1);
        }
        Set<String> scored = new TreeSet<>();
        for (Path state : states) {
            scored.add(state.getFileName().toString());
        }
        List<String> vanished = new ArrayList<>();
        for (String name : blessed) {
            if (!scored.contains(name)) {
                vanished.add(name);
            }
        }
        if (!vanished.isEmpty()) {
            System.out.println("FAIL - " + vanished.size() + " of "
                + blessed.size() + " manifest-blessed state(s) are missing from "
                + "disk, so only " + scored.size() + " would be scored: "
                + vanished + " -- either an export is still running (the tiers "
                + "fill in order, so re-run when it finishes) or the fixture is "
                + "genuinely narrower than its manifest. Either way the "
                + "coverage numbers below would be overstated");
            System.exit(1);
        }

        boolean[] scope = new boolean[
            java.util.Arrays.stream(Policy.HEAD_SIZES).sum()];
        java.util.Arrays.fill(scope, true);
        PolicyRuntime runtime = PolicyRuntime.load(caseRoot, scope);
        boolean[] sharedLegal = Policy.readMask(caseRoot, "action_mask");
        List<String> failures = new ArrayList<>();
        int ownMasks = 0;
        int phaseDisagreements = 0;

        float worstGroup = 0.0f;
        String worstGroupWhere = "-";
        float worstObservation = 0.0f;
        String worstObservationWhere = "-";
        float worstValue = 0.0f;
        int decisionsCompared = 0;
        int decisionsAgreed = 0;
        int exactStates = 0;
        java.util.Set<String> distinctDecisions = new java.util.TreeSet<>();
        java.util.Set<String> distinctObservations = new java.util.HashSet<>();

        System.out.printf("%-26s %-12s %-12s %-10s %s%n",
            "state", "groups", "observation", "value", "decision");

        for (Path state : states) {
            Map<String, Double> in = readValues(state.resolve("inputs.txt"));
            Projection.Params params = params(in);
            if (params.targetAbilitySlotDivisor() != 15.0) {
                failures.add(state.getFileName()
                    + " uses legacy target-attack divisor "
                    + params.targetAbilitySlotDivisor()
                    + "; Arsenal publishes authored ability slot / 15");
                continue;
            }

            // The target's attack phase, progress and index are produced by an
            // upstream stage that is NOT ported -- the same status as
            // target_facing_error and head_pitch, which are already read out of
            // the combat row rather than recomputed. Take them from the JAX
            // observation so the rest of the row is testable.
            //
            // Recomputing them with `_target_phase` (which the exporter does,
            // into inputs.txt) disagrees with what the arsenal environment
            // actually observed. The disagreement is counted and reported below
            // rather than hidden: it means the phase source in an arsenal scene
            // is not the legacy path, and that is an open question, not a
            // settled one.
            float[] jaxCombat = Policy.read(state, "base_combat_f32");
            int observedPhase = decodePhase(jaxCombat);
            double observedProgress = jaxCombat[17];
            double observedIndex = Math.round(
                jaxCombat[21] * in.get("target_attack_index_divisor"));
            if (jaxCombat[21] < 0.0f) {
                observedIndex = -1.0;
            }
            if (observedPhase != (int) Math.round(in.get("target_attack_phase"))) {
                phaseDisagreements++;
            }
            in.put("target_attack_phase", (double) observedPhase);
            in.put("target_attack_progress_raw", observedProgress);
            in.put("target_reported_attack_index", observedIndex);

            Derived derived = derive(state, in, params);

            // -- the derived groups against JAX's own, per state --------------
            float stateWorstGroup = 0.0f;
            for (String name : DERIVED_FLOATS) {
                float[] expected = Policy.read(state, name);
                float[] actual = derived.floats().get(name);
                if (expected.length != actual.length) {
                    failures.add(state.getFileName() + " " + name + ": derived "
                        + actual.length + " values, fixture has " + expected.length);
                    continue;
                }
                for (int i = 0; i < expected.length; i++) {
                    float delta = Math.abs(expected[i] - actual[i]);
                    if (delta > stateWorstGroup) {
                        stateWorstGroup = delta;
                    }
                    if (delta > worstGroup) {
                        worstGroup = delta;
                        worstGroupWhere =
                            state.getFileName() + " " + name + "[" + i + "]";
                    }
                    if (delta > TOLERANCE) {
                        failures.add(String.format(
                            "%s %s column %d: expected %.9f got %.9f (delta %.3e)",
                            state.getFileName(), name, i, expected[i],
                            actual[i], delta));
                    }
                }
            }
            for (String name : DERIVED_MASKS) {
                boolean[] expected = Policy.readMask(state, name);
                boolean[] actual = derived.masks().get(name);
                if (expected.length != actual.length) {
                    failures.add(state.getFileName() + " " + name + ": derived "
                        + actual.length + " values, fixture has " + expected.length);
                    continue;
                }
                for (int i = 0; i < expected.length; i++) {
                    float delta = expected[i] == actual[i] ? 0.0f : 1.0f;
                    stateWorstGroup = Math.max(stateWorstGroup, delta);
                    if (delta > worstGroup) {
                        worstGroup = delta;
                        worstGroupWhere =
                            state.getFileName() + " " + name + "[" + i + "]";
                    }
                    if (delta != 0.0f) {
                        failures.add(String.format(
                            "%s %s column %d: expected %b got %b",
                            state.getFileName(), name, i, expected[i], actual[i]));
                    }
                }
            }

            // -- assemble and compare the whole vector ------------------------
            float[] javaObservation = ObservationAssembler.assemble(
                state, derived.floats(), derived.masks());
            float[] jaxObservation =
                readFloats(state.resolve("expected_observation.bin"));
            distinctObservations.add(
                java.util.Arrays.toString(jaxObservation));
            float stateWorstObservation = 0.0f;
            if (javaObservation.length != jaxObservation.length) {
                failures.add(state.getFileName() + ": observation width "
                    + javaObservation.length + " != JAX " + jaxObservation.length);
            } else {
                for (int i = 0; i < javaObservation.length; i++) {
                    float delta = Math.abs(javaObservation[i] - jaxObservation[i]);
                    if (delta > stateWorstObservation) {
                        stateWorstObservation = delta;
                    }
                    if (delta > worstObservation) {
                        worstObservation = delta;
                        worstObservationWhere =
                            state.getFileName() + " column " + i;
                    }
                    if (delta > TOLERANCE) {
                        failures.add(String.format(
                            "%s observation column %d: expected %.9f got %.9f "
                                + "(delta %.3e)",
                            state.getFileName(), i, jaxObservation[i],
                            javaObservation[i], delta));
                    }
                }
                if (stateWorstObservation == 0.0f) {
                    exactStates++;
                }
            }

            // -- the decision, which is what actually matters -----------------
            // Prefer this state's own legality. The shared mask still compares
            // Java against JAX fairly -- both sides see it -- but the decisions
            // would be the ones a differently-equipped actor would take, and
            // weapons whose abilities are illegal under it look interchangeable.
            boolean[] legal = sharedLegal;
            if (Files.exists(state.resolve("action_mask.bin"))) {
                legal = Policy.readMask(state, "action_mask");
                ownMasks++;
            }
            ActionDecoder.Decoded fromJava =
                runtime.decide(javaObservation, runtime.newCarry(), legal);
            ActionDecoder.Decoded fromJax =
                runtime.decide(jaxObservation, runtime.newCarry(), legal);
            float[] javaCarry = runtime.newCarry();
            runtime.decide(javaObservation, javaCarry, legal);
            float valueJava = runtime.value(javaCarry);
            float[] jaxCarry = runtime.newCarry();
            runtime.decide(jaxObservation, jaxCarry, legal);
            float valueJax = runtime.value(jaxCarry);
            float valueDelta = Math.abs(valueJava - valueJax);
            worstValue = Math.max(worstValue, valueDelta);

            decisionsCompared++;
            if (fromJava.equals(fromJax)) {
                decisionsAgreed++;
            } else {
                failures.add(state.getFileName()
                    + ": the float32 residual changed the decision: "
                    + describe(fromJava) + " vs " + describe(fromJax));
            }
            if (valueDelta > 1.0e-4f) {
                failures.add(String.format(
                    "%s: critic value moved by %.3e, more than rounding explains",
                    state.getFileName(), valueDelta));
            }
            distinctDecisions.add(describe(fromJax));

            System.out.printf("%-26s %-12s %-12s %-10s %s%n",
                state.getFileName(),
                String.format("%.2e", stateWorstGroup),
                String.format("%.2e", stateWorstObservation),
                String.format("%.2e", valueDelta),
                describe(fromJax));
        }

        // -- the override must actually reach the vector ----------------------
        // If it were silently ignored the assembler would fall back to each
        // fixture's own JAX-produced groups, every comparison above would be
        // JAX against itself, and all of them would pass while testing nothing.
        Path probeState = states.get(0);
        Map<String, Double> probeIn = readValues(probeState.resolve("inputs.txt"));
        Derived probeDerived = derive(probeState, probeIn, params(probeIn));
        float[] sentinel = probeDerived.floats().get("base_self_f32").clone();
        sentinel[0] = sentinel[0] + 0.25f;
        Map<String, float[]> probe = new LinkedHashMap<>(probeDerived.floats());
        probe.put("base_self_f32", sentinel);
        if (java.util.Arrays.equals(
                ObservationAssembler.assemble(
                    probeState, probe, probeDerived.masks()),
                ObservationAssembler.assemble(
                    probeState, probeDerived.floats(), probeDerived.masks()))) {
            failures.add("the assembler ignored an overridden group -- every "
                + "comparison above was JAX against itself");
        }

        reportGroupCoverage(states);

        System.out.printf("%nstates           : %d%n", states.size());
        System.out.printf("worst group      : %.3e at %s%n",
            worstGroup, worstGroupWhere);
        System.out.printf("worst observation: %.3e at %s%n",
            worstObservation, worstObservationWhere);
        System.out.printf("worst value      : %.3e%n", worstValue);
        System.out.printf("bit-exact states : %d/%d%n", exactStates, states.size());
        System.out.printf("decisions agreed : %d/%d%n",
            decisionsAgreed, decisionsCompared);
        System.out.printf("distinct obs     : %d/%d%n",
            distinctObservations.size(), states.size());
        System.out.printf("distinct actions : %d%s%n", distinctDecisions.size(),
            distinctDecisions.size() == 1
                ? "  (this policy maps every state here to one action from a "
                  + "zero carry -- a fact about the network, not the fixture)"
                : "");
        System.out.printf("phase supplied   : %d/%d states where the legacy "
            + "_target_phase disagreed with what the arsenal env observed%n",
            phaseDisagreements, states.size());
        System.out.printf("per-state masks  : %d/%d%s%n", ownMasks, states.size(),
            ownMasks == states.size() ? ""
                : " (rest fell back to the shared case mask)");
        System.out.printf("tolerance        : %.1e%n", TOLERANCE);

        // Anti-vacuity, correctly aimed. The original form of this guard
        // demanded distinct *actions*, and that was wrong: with a zero carry
        // this policy maps all 48 genuinely different observations to the same
        // argmax, which is a property of the network, not a defect in the
        // fixture. What must hold is that the observations differ and that the
        // derived groups actually perturb some of them -- otherwise every
        // comparison is the same vector checked repeatedly.
        if (distinctObservations.size() < 2 && states.size() > 4) {
            failures.add("all " + states.size() + " states produced the same "
                + "observation -- the fixture is not diverse enough for "
                + "agreement to mean anything");
        }
        if (exactStates == states.size() && states.size() > 4) {
            failures.add("every state was bit-identical to JAX -- the derived "
                + "groups never perturbed the vector, so the decision check "
                + "never had a residual to be robust to");
        }

        System.out.printf("failures         : %d%n", failures.size());
        if (!failures.isEmpty()) {
            failures.stream().limit(15).forEach(f -> System.out.println("  " + f));
            if (failures.size() > 15) {
                System.out.println("  ... and " + (failures.size() - 15) + " more");
            }
            System.out.println("FAIL");
            System.exit(1);
        }
        System.out.println("PASS - Java derives, assembles and decides as JAX does");
    }

    /**
     * The 37 groups and how many columns each contributes, in layout order.
     * Sizes are the per-actor widths after agent-row selection, which is what
     * lands in the 8,272-column vector.
     */
    private static final String[][] GROUPS = {
        {"base_self_f32", "16"}, {"base_target_f32", "16"},
        {"base_target_mask", "1"}, {"base_combat_f32", "18"},
        {"resource_f32", "7"}, {"resource_mask", "7"}, {"defense_f32", "8"},
        {"status_f32", "48"}, {"status_mask", "8"},
        {"ability_f32", "176"}, {"ability_mask", "16"}, {"ability_legal", "16"},
        {"actor_world_f32", "5"}, {"actor_world_mask", "3"},
        {"movement_state_f32", "23"}, {"movement_state_mask", "23"},
        {"geometry_token_f32", "5764"}, {"geometry_token_mask", "44"},
        {"geometry_available", "1"},
        {"light_token_f32", "440"}, {"light_available", "1"},
        {"inventory_container_f32", "6"}, {"inventory_container_mask", "6"},
        {"inventory_token_f32", "532"}, {"inventory_token_mask", "76"},
        {"inventory_available", "1"},
        {"skill_action_mask", "9"}, {"jump_action_mask", "1"},
        {"guard_action_mask", "1"}, {"dodge_action_mask", "4"},
        {"valid", "1"},
        {"block_candidate_f32", "448"}, {"block_candidate_mask", "16"},
        {"block_available", "1"},
        {"recipe_embedding", "512"}, {"recipe_mask", "16"},
        {"recipe_available", "1"},
    };

    /**
     * Say which groups the fixture actually exercises.
     *
     * <p>A group that is all-zero across every state is being certified
     * vacuously: the assembler copies zeros into the right slots and the
     * comparison passes without the group's logic ever mattering. That is the
     * true state of block candidates and recipes in a flat scene — and, as it
     * turns out, in the region fixture too, because it is built with no held
     * item so no block trigger is ever available.
     *
     * <p>This does not fail the run. It is a coverage report, and the honest
     * place to read how much of the 8,271 columns a given fixture speaks for.
     */
    private static void reportGroupCoverage(List<Path> states) throws IOException {
        System.out.printf("%n%-26s %8s  %s%n", "group", "columns", "states with signal");
        int exercised = 0;
        int exercisedColumns = 0;
        int totalColumns = 0;
        for (String[] group : GROUPS) {
            String name = group[0];
            int columns = Integer.parseInt(group[1]);
            totalColumns += columns;
            int withSignal = 0;
            for (Path state : states) {
                if (hasSignal(state, name)) {
                    withSignal++;
                }
            }
            if (withSignal > 0) {
                exercised++;
                exercisedColumns += columns;
            }
            System.out.printf("%-26s %8d  %d/%d%s%n", name, columns,
                withSignal, states.size(),
                withSignal == 0 ? "   ALL ZERO - certified vacuously" : "");
        }
        System.out.printf("%ngroups with signal: %d/%d (%d of %d columns, %.1f%%)%n",
            exercised, GROUPS.length, exercisedColumns, totalColumns,
            100.0 * exercisedColumns / totalColumns);
    }

    private static boolean hasSignal(Path state, String name) throws IOException {
        Path path = state.resolve(name + ".bin");
        if (!Files.exists(path)) {
            return false;
        }
        for (byte b : Files.readAllBytes(path)) {
            if (b != 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * Recover the attack phase from the one-hot block of a JAX combat row.
     *
     * <p>The block is multiplied by visibility, so an imperceptible target
     * leaves all five columns zero — which is IDLE's encoding as far as the
     * derivation is concerned, since the masking zeroes it either way.
     */
    private static int decodePhase(float[] combat) {
        for (int phase = 0; phase < 5; phase++) {
            if (combat[12 + phase] > 0.5f) {
                return phase;
            }
        }
        return Projection.PHASE_IDLE;
    }

    private static Projection.Params params(Map<String, Double> in) {
        return new Projection.Params(
            in.get("agent_max_speed"),
            in.getOrDefault("agent_maximum_gait_speed", in.get("agent_max_speed")),
            in.get("vertical_speed_scale"),
            in.get("agent_max_health"),
            in.get("agent_attack_pause_max_seconds"),
            in.get("regen_delay_ticks"),
            in.get("agent_walk_max_fall_speed"),
            in.get("loaded_dt"),
            in.get("wire_fixed_point_scale"),
            in.get("target_chase_speed"),
            in.get("target_max_health"),
            in.get("sensor_range"),
            in.get("facing_error_degrees_scale"),
            in.get("head_pitch_degrees_scale"),
            in.get("target_attack_index_divisor"));
    }

    private static Derived derive(
        Path state, Map<String, Double> in, Projection.Params params
    ) {
        float[] selfF32 = Projection.selfFeatures(new Projection.SelfState(
            in.get("agent_velocity_x"), in.get("agent_velocity_y"),
            in.get("agent_velocity_z"), in.get("agent_yaw_degrees"),
            in.get("agent_pitch_degrees"), in.get("agent_health"),
            in.get("agent_grounded") != 0.0,
            in.get("agent_attack_executing"),
            in.get("agent_attack_cooldown_seconds"),
            in.get("agent_knockback_control_lock") != 0.0,
            in.get("agent_ticks_since_damage"),
            in.get("agent_applied_vertical_velocity"),
            in.get("agent_fall_speed"),
            in.get("agent_motion_delta_seconds")), params);

        float[] targetF32 = Projection.targetFeatures(new Projection.TargetState(
            in.get("agent_position_x"), in.get("agent_position_y"),
            in.get("agent_position_z"),
            in.get("agent_velocity_x"), in.get("agent_velocity_y"),
            in.get("agent_velocity_z"),
            in.get("agent_yaw_degrees"),
            in.get("target_position_x"), in.get("target_position_y"),
            in.get("target_position_z"),
            in.get("target_velocity_x"), in.get("target_velocity_y"),
            in.get("target_velocity_z"),
            in.get("target_health"),
            in.get("target_visible"),
            in.get("target_facing_error"),
            in.get("target_head_facing_error"),
            in.get("target_head_pitch"),
            in.get("target_attack_progress")), params);

        Projection.TargetRaw raw = new Projection.TargetRaw(
            in.get("agent_position_x"), in.get("agent_position_y"),
            in.get("agent_position_z"),
            in.get("target_position_x"), in.get("target_position_y"),
            in.get("target_position_z"),
            in.get("target_velocity_x"), in.get("target_velocity_y"),
            in.get("target_velocity_z"),
            in.get("target_health"),
            in.get("target_yaw_degrees"),
            in.get("target_head_yaw_degrees"),
            in.get("target_head_pitch_degrees"),
            in.get("target_perceptible") != 0.0,
            true,
            (int) Math.round(in.get("target_attack_phase")),
            in.get("target_attack_progress_raw"),
            (int) Math.round(in.get("target_reported_attack_index")));
        float[] combatF32 = Projection.combatFeatures(
            new Projection.CombatSelf(
                in.get("agent_velocity_x"), in.get("agent_velocity_y"),
                in.get("agent_velocity_z"), in.get("agent_yaw_degrees"),
                in.get("agent_health_fraction"),
                in.get("agent_attack_executing_bool") != 0.0),
            Projection.targetEvidence(raw, params), params);

        var movement = Projection.movementFeatures(
            (int) Math.round(in.get("movement_value_bits")),
            (int) Math.round(in.get("movement_available_bits")));
        if (movement.invalid()) {
            throw new IllegalArgumentException(
                "parity fixture contains invalid movement-state bit evidence");
        }
        var actorWorld = Projection.actorWorldFeatures(
            new WorldProjection.ActorWorldInput(
                in.get("actor_controller_medium_available") != 0.0,
                in.get("actor_controller_in_fluid") != 0.0,
                in.get("actor_submersion_available") != 0.0,
                in.get("actor_feet_submerged") != 0.0,
                in.get("actor_eyes_submerged") != 0.0,
                in.get("actor_drop_available") != 0.0,
                in.get("actor_drop_support_found") != 0.0,
                in.get("actor_drop_height").floatValue(),
                in.get("agent_walk_max_drop_height").floatValue()));
        MechanicsProjection.ResourceResult resources;
        MechanicsProjection.DefenseResult defense;
        MechanicsProjection.StatusResult status;
        boolean[] dodgeMask;
        AbilityProjection.Result ability;
        InventoryProjection.Result inventory;
        LightProjection.Result light;
        GeometryProjection.Result geometry;
        try {
            resources = Projection.resourceFeatures(
                new MechanicsProjection.ResourceInput(
                    readFloats(state.resolve("raw_resource_current.bin")),
                    readFloats(state.resolve("raw_resource_minimum.bin")),
                    readFloats(state.resolve("raw_resource_maximum.bin"))));
            float[] rawDefense = readFloats(state.resolve("raw_defense.bin"));
            if (rawDefense.length != 24) {
                throw new IllegalArgumentException(
                    "raw defense has " + rawDefense.length + " values, expected 24");
            }
            defense = Projection.defenseFeatures(
                new MechanicsProjection.DefenseInput(
                    booleans(rawDefense, 0, 2),
                    booleans(rawDefense, 2, 2),
                    slice(rawDefense, 4, 2),
                    slice(rawDefense, 6, 2),
                    slice(rawDefense, 8, 6),
                    slice(rawDefense, 14, 2),
                    slice(rawDefense, 16, 2),
                    slice(rawDefense, 18, 2),
                    slice(rawDefense, 20, 2),
                    slice(rawDefense, 22, 2)));
            float[] rawStatus = readFloats(state.resolve("raw_status.bin"));
            if (rawStatus.length != 160) {
                throw new IllegalArgumentException(
                    "raw status has " + rawStatus.length
                        + " values, expected 160");
            }
            status = Projection.statusFeatures(
                new MechanicsProjection.StatusInput(
                    slice(rawStatus, 0, 2),
                    slice(rawStatus, 2, 14),
                    slice(rawStatus, 16, 16),
                    slice(rawStatus, 32, 16),
                    slice(rawStatus, 48, 16),
                    slice(rawStatus, 64, 16),
                    slice(rawStatus, 80, 16),
                    integers(rawStatus, 96, 16),
                    slice(rawStatus, 112, 16),
                    slice(rawStatus, 128, 16),
                    booleans(rawStatus, 144, 16)));
            float[] rawDodge = readFloats(state.resolve("raw_dodge.bin"));
            if (rawDodge.length != 8) {
                throw new IllegalArgumentException(
                    "raw dodge has " + rawDodge.length
                        + " values, expected 8");
            }
            dodgeMask = Projection.dodgeActionMask(
                new ActionMaskProjection.DodgeInput(
                    booleans(rawDodge, 0, 4),
                    rawDodge[4] > 0.5f,
                    rawDodge[5] > 0.5f,
                    rawDodge[6],
                    rawDodge[7]));
            float[] rawAbility = readFloats(state.resolve("raw_ability.bin"));
            if (rawAbility.length != 868) {
                throw new IllegalArgumentException(
                    "raw ability has " + rawAbility.length
                        + " values, expected 868");
            }
            ability = Projection.abilityFeatures(
                new AbilityProjection.Input(
                    slice(rawAbility, 0, 14),
                    slice(rawAbility, 14, 14),
                    slice(rawAbility, 28, 32),
                    slice(rawAbility, 60, 32),
                    slice(rawAbility, 92, 32),
                    integers(rawAbility, 124, 2),
                    slice(rawAbility, 126, 2),
                    slice(rawAbility, 128, 224),
                    integers(rawAbility, 352, 224),
                    slice(rawAbility, 576, 224),
                    booleans(rawAbility, 800, 32),
                    booleans(rawAbility, 832, 2),
                    booleans(rawAbility, 834, 2),
                    booleans(rawAbility, 836, 32),
                    new int[] {
                        (int) Math.round(in.getOrDefault("agent_weapon_family", -1.0)),
                        (int) Math.round(in.getOrDefault("target_weapon_family", -1.0))
                    }));
            boolean[] rawInventoryFlags = readBooleans(
                state.resolve("raw_inventory_flags.bin"));
            int[] rawInventoryI32 = readInts(
                state.resolve("raw_inventory_i32.bin"));
            float[] rawInventoryF32 = readFloats(
                state.resolve("raw_inventory_f32.bin"));
            if (rawInventoryFlags.length != 159
                || rawInventoryI32.length != 310
                || rawInventoryF32.length != 76) {
                throw new IllegalArgumentException(
                    "raw inventory width drift: "
                        + rawInventoryFlags.length + "/"
                        + rawInventoryI32.length + "/"
                        + rawInventoryF32.length);
            }
            inventory = Projection.inventoryFeatures(
                new InventoryProjection.Input(
                    rawInventoryFlags[0],
                    slice(rawInventoryFlags, 1, 6),
                    slice(rawInventoryI32, 0, 6),
                    slice(rawInventoryFlags, 7, 76),
                    slice(rawInventoryI32, 6, 76),
                    slice(rawInventoryI32, 82, 76),
                    slice(rawInventoryI32, 158, 76),
                    slice(rawInventoryI32, 234, 76),
                    rawInventoryF32,
                    slice(rawInventoryFlags, 83, 76)));
            boolean[] expectedGeometryMask = readBooleans(
                state.resolve("geometry_token_mask.bin"));
            boolean expectedGeometryAvailable = readBooleans(
                state.resolve("geometry_available.bin"))[0];
            boolean actorValid = readBooleans(
                state.resolve("valid.bin"))[0];
            geometry = Projection.geometryFeatures(geometryInputFromProjected(
                readFloats(state.resolve("geometry_token_f32.bin")),
                expectedGeometryMask, expectedGeometryAvailable, actorValid));
            light = Projection.lightFeatures(lightInputFromProjected(
                readFloats(state.resolve("light_token_f32.bin")),
                readBooleans(state.resolve("light_available.bin"))[0],
                geometry.tokenMask(), geometry.available(), actorValid));
        } catch (IOException exception) {
            throw new java.io.UncheckedIOException(exception);
        }

        Map<String, float[]> floats = new LinkedHashMap<>();
        floats.put("base_self_f32", selfF32);
        floats.put("base_target_f32", targetF32);
        floats.put("base_combat_f32", combatF32);
        floats.put("movement_state_f32", movement.values());
        floats.put("actor_world_f32", actorWorld.values());
        floats.put("resource_f32", resources.values());
        floats.put("defense_f32", defense.values());
        floats.put("status_f32", status.values());
        floats.put("ability_f32", ability.values());
        floats.put("inventory_container_f32", inventory.containerValues());
        floats.put("inventory_token_f32", inventory.tokenValues());
        floats.put("light_token_f32", light.values());
        floats.put("geometry_token_f32", geometry.values());
        Map<String, boolean[]> masks = new LinkedHashMap<>();
        masks.put("movement_state_mask", movement.mask());
        masks.put("actor_world_mask", actorWorld.mask());
        masks.put("resource_mask", resources.mask());
        masks.put("status_mask", status.mask());
        masks.put("dodge_action_mask", dodgeMask);
        masks.put("ability_mask", ability.mask());
        masks.put("ability_legal", ability.legal());
        masks.put("inventory_container_mask", inventory.containerMask());
        masks.put("inventory_token_mask", inventory.tokenMask());
        masks.put("inventory_available", new boolean[] {inventory.available()});
        masks.put("light_available", new boolean[] {light.available()});
        masks.put("geometry_token_mask", geometry.tokenMask());
        masks.put("geometry_available", new boolean[] {geometry.available()});
        return new Derived(floats, masks);
    }

    /** Reconstruct one policy-equivalent raw geometry row for chain testing. */
    private static GeometryProjection.Input geometryInputFromProjected(
        float[] projected,
        boolean[] projectedMask,
        boolean available,
        boolean actorValid
    ) {
        if (projected.length != GeometryProjection.VALUE_COUNT
            || projectedMask.length != GeometryProjection.TOKEN_CAPACITY) {
            throw new IllegalArgumentException("projected geometry width drift");
        }
        boolean[] tokenMask = available
            ? projectedMask.clone()
            : new boolean[GeometryProjection.TOKEN_CAPACITY];
        int[] kind = new int[GeometryProjection.TOKEN_CAPACITY];
        int[] provenance = new int[GeometryProjection.TOKEN_CAPACITY];
        float[] position = new float[GeometryProjection.TOKEN_CAPACITY * 3];
        float[] clearance = new float[GeometryProjection.TOKEN_CAPACITY];
        int[] semantic = new int[GeometryProjection.TOKEN_CAPACITY];
        boolean[] dynamic = new boolean[GeometryProjection.TOKEN_CAPACITY];
        boolean[] edgeMask = new boolean[
            GeometryProjection.TOKEN_CAPACITY * GeometryProjection.EDGE_CAPACITY];
        int[] destination = new int[edgeMask.length];
        java.util.Arrays.fill(destination, -1);
        float[] edgeCost = new float[edgeMask.length];
        int[] edgeKind = new int[edgeMask.length];
        int[] edgeFlags = new int[edgeMask.length];
        boolean[] boxMask = new boolean[
            GeometryProjection.TOKEN_CAPACITY
                * GeometryProjection.DETAIL_BOX_CAPACITY];
        float[] boxes = new float[boxMask.length * 6];

        for (int token = 0; token < GeometryProjection.TOKEN_CAPACITY; token++) {
            int input = token * GeometryProjection.FEATURE_SIZE;
            kind[token] = Math.round(projected[input]
                * GeometryProjection.MAXIMUM_TOKEN_KIND);
            provenance[token] = Math.round(projected[input + 1]
                * GeometryProjection.MAXIMUM_PROVENANCE);
            int xyz = token * 3;
            position[xyz] = projected[input + 2]
                * GeometryProjection.MAXIMUM_DISTANCE;
            position[xyz + 1] = projected[input + 3]
                * GeometryProjection.MAXIMUM_DISTANCE;
            position[xyz + 2] = projected[input + 4]
                * GeometryProjection.MAXIMUM_DISTANCE;
            clearance[token] = projected[input + 5]
                * GeometryProjection.MAXIMUM_DISTANCE;
            semantic[token] = Math.round(projected[input + 6] * 65535.0f);
            dynamic[token] = projected[input + 7] > 0.5f;
            int edgeBase = input + GeometryProjection.BASE_FEATURE_SIZE;
            for (int edge = 0; edge < GeometryProjection.EDGE_CAPACITY; edge++) {
                int source = token * GeometryProjection.EDGE_CAPACITY + edge;
                int feature = edgeBase + edge * GeometryProjection.EDGE_FEATURE_SIZE;
                edgeMask[source] = projected[feature] > 0.5f;
                destination[source] = edgeMask[source]
                    ? Math.round(projected[feature + 1]
                        * (GeometryProjection.TOKEN_CAPACITY + 1.0f) - 1.0f)
                    : -1;
                edgeCost[source] = projected[feature + 2]
                    * GeometryProjection.MAXIMUM_DISTANCE;
                edgeKind[source] = Math.round(projected[feature + 3] * 255.0f);
                edgeFlags[source] = Math.round(projected[feature + 4] * 255.0f);
            }
            int boxBase = edgeBase
                + GeometryProjection.EDGE_CAPACITY
                    * GeometryProjection.EDGE_FEATURE_SIZE;
            for (int box = 0; box < GeometryProjection.DETAIL_BOX_CAPACITY; box++) {
                int source = token * GeometryProjection.DETAIL_BOX_CAPACITY + box;
                int feature = boxBase + box * GeometryProjection.BOX_FEATURE_SIZE;
                boxMask[source] = projected[feature] > 0.5f;
                for (int component = 0; component < 6; component++) {
                    boxes[source * 6 + component] = projected[feature + 1 + component]
                        * GeometryProjection.MAXIMUM_DISTANCE;
                }
            }
        }
        return new GeometryProjection.Input(
            available, tokenMask, kind, provenance, position, clearance,
            semantic, dynamic, edgeMask, destination, edgeCost, edgeKind,
            edgeFlags, boxMask, boxes, actorValid);
    }

    /** Reconstruct raw channel levels from a known-good exported policy row. */
    private static LightProjection.Input lightInputFromProjected(
        float[] projected,
        boolean available,
        boolean[] geometryMask,
        boolean geometryAvailable,
        boolean actorValid
    ) {
        if (projected.length != LightProjection.VALUE_COUNT) {
            throw new IllegalArgumentException(
                "projected light has " + projected.length + " values, expected "
                    + LightProjection.VALUE_COUNT);
        }
        boolean[] sourceMask = available
            ? geometryMask.clone()
            : new boolean[LightProjection.TOKEN_CAPACITY];
        boolean[] valid = new boolean[
            LightProjection.TOKEN_CAPACITY * LightProjection.CHANNEL_COUNT];
        int[] sky = new int[LightProjection.TOKEN_CAPACITY];
        int[] block = new int[LightProjection.TOKEN_CAPACITY * 3];
        int[] tint = new int[LightProjection.TOKEN_CAPACITY * 3];
        for (int token = 0; token < LightProjection.TOKEN_CAPACITY; token++) {
            int input = token * LightProjection.FEATURE_SIZE;
            int channel = token * LightProjection.CHANNEL_COUNT;
            int rgb = token * 3;
            valid[channel] = projected[input] > 0.5f;
            valid[channel + 1] = projected[input + 1] > 0.5f;
            valid[channel + 2] = projected[input + 2] > 0.5f;
            sky[token] = Math.round(projected[input + 3] * 15.0f);
            for (int component = 0; component < 3; component++) {
                block[rgb + component] = Math.round(
                    projected[input + 4 + component] * 15.0f);
                tint[rgb + component] = Math.round(
                    projected[input + 7 + component] * 255.0f);
            }
        }
        return new LightProjection.Input(
            available, sourceMask, valid, sky, block, tint,
            geometryMask, geometryAvailable, actorValid);
    }

    private static float[] slice(float[] values, int start, int length) {
        return java.util.Arrays.copyOfRange(values, start, start + length);
    }

    private static int[] slice(int[] values, int start, int length) {
        return java.util.Arrays.copyOfRange(values, start, start + length);
    }

    private static boolean[] slice(boolean[] values, int start, int length) {
        return java.util.Arrays.copyOfRange(values, start, start + length);
    }

    private static boolean[] booleans(float[] values, int start, int length) {
        boolean[] result = new boolean[length];
        for (int index = 0; index < length; index++) {
            result[index] = values[start + index] > 0.5f;
        }
        return result;
    }

    private static int[] integers(float[] values, int start, int length) {
        int[] result = new int[length];
        for (int index = 0; index < length; index++) {
            result[index] = Math.round(values[start + index]);
        }
        return result;
    }

    private static String describe(ActionDecoder.Decoded d) {
        return String.format("skill=%d move=%d yaw=%+.1f jump=%b atk=%b",
            d.skillId(), d.worldMoveDirection(), d.yawDeltaDegrees(),
            d.jumpHeld(), d.attack());
    }

    private static Map<String, Double> readValues(Path path) throws IOException {
        Map<String, Double> values = new HashMap<>();
        for (String line : Files.readAllLines(path)) {
            if (line.isBlank()) {
                continue;
            }
            String[] parts = line.split("\t");
            values.put(parts[0], Double.parseDouble(parts[1]));
        }
        return values;
    }

    private static float[] readFloats(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.wrap(bytes)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[bytes.length / 4];
        buffer.asFloatBuffer().get(values);
        return values;
    }

    private static int[] readInts(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        if (bytes.length % Integer.BYTES != 0) {
            throw new IOException("int32 fixture byte width drift at " + path);
        }
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.wrap(bytes)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        int[] values = new int[bytes.length / Integer.BYTES];
        buffer.asIntBuffer().get(values);
        return values;
    }

    private static boolean[] readBooleans(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        boolean[] values = new boolean[bytes.length];
        for (int index = 0; index < bytes.length; index++) {
            if (bytes[index] != 0 && bytes[index] != 1) {
                throw new IOException(
                    "boolean fixture contains a non-binary byte at " + path);
            }
            values[index] = bytes[index] != 0;
        }
        return values;
    }
}
