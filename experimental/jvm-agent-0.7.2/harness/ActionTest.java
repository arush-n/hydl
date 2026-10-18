import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.ObservationAssembler;
import com.hytalerlbridge.policy.Policy;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Certifies {@link ActionDecoder} against JAX-exported reference cases.
 *
 * <p>Every expectation in {@code case/actions.txt} was produced by running the
 * real Python decoders -- {@code decode_staged_action_surface_factors} for the
 * semantic fields and {@code translate_learner_arsenal_action_to_native} for the
 * fields the server applies. Nothing here is my reading of those functions.
 *
 * <p>Equality is exact, not approximate. Every value is either an integer, a
 * boolean, or a value from the contract's float32-normalized multiscale look
 * table, so any difference means a real disagreement rather than accumulated
 * error.
 *
 * <p>World-root execution is deliberately outside this decoder. A Use, block,
 * or recipe selection must be classified by {@code requestsWorldVerb()} so the
 * composed sink can either execute it from privileged evidence or count a
 * fail-closed drop. Adapter tests own native request construction; this test
 * owns lossless semantic routing and never treats a dropped verb as success.
 */
public final class ActionTest {

    private record Case(
        String name, int[] factors,
        boolean legal, int skill, int ability, boolean guard, int dodge,
        boolean jump, int move, double yaw, double pitch, boolean use,
        int blockTrigger, int recipe, int block,
        boolean nativeLegal,
        boolean nf, boolean nb, boolean nl, boolean nr, boolean nAttack, boolean nJump
    ) {}

    /** Record a field that takes the same value in every case. */
    private static void requireVaries(
        List<String> constant,
        List<Case> cases,
        String field,
        java.util.function.Function<Case, String> read
    ) {
        java.util.Set<String> seen = new java.util.HashSet<>();
        for (Case one : cases) {
            seen.add(read.apply(one));
            if (seen.size() > 1) {
                return;
            }
        }
        constant.add(field + "=" + (seen.isEmpty() ? "<none>"
            : seen.iterator().next()));
    }

    public static void main(String[] args) throws IOException {
        Path directory = Path.of(args.length > 0 ? args[0] : "case");
        List<Case> cases = load(directory.resolve("actions.txt"));

        // Every expectation below is exact, which makes this file only as
        // strong as the cases in it. A regenerated actions.txt that dropped
        // rows would still report PASS -- on whatever remained -- because
        // nothing here required the fixture to exercise anything. So the
        // decoded surface has to vary before a single comparison is trusted:
        // a field pinned to one value across all 59 cases is certified
        // vacuously, exactly the hole that let ParityTest score orphans and
        // let ProfileTest certify one state.
        List<String> constant = new ArrayList<>();
        requireVaries(constant, cases, "legal", c -> String.valueOf(c.legal()));
        requireVaries(constant, cases, "skill", c -> String.valueOf(c.skill()));
        requireVaries(constant, cases, "ability", c -> String.valueOf(c.ability()));
        requireVaries(constant, cases, "guard", c -> String.valueOf(c.guard()));
        requireVaries(constant, cases, "dodge", c -> String.valueOf(c.dodge()));
        requireVaries(constant, cases, "jump", c -> String.valueOf(c.jump()));
        requireVaries(constant, cases, "move", c -> String.valueOf(c.move()));
        requireVaries(constant, cases, "yaw", c -> String.valueOf(c.yaw()));
        requireVaries(constant, cases, "pitch", c -> String.valueOf(c.pitch()));
        requireVaries(constant, cases, "use", c -> String.valueOf(c.use()));
        requireVaries(constant, cases, "blockTrigger",
            c -> String.valueOf(c.blockTrigger()));
        requireVaries(constant, cases, "block", c -> String.valueOf(c.block()));
        requireVaries(constant, cases, "nativeLegal",
            c -> String.valueOf(c.nativeLegal()));
        if (!constant.isEmpty()) {
            System.out.println("FAIL - " + constant.size() + " decoded field(s) "
                + "never vary across the " + cases.size() + " cases, so they "
                + "are certified vacuously: " + constant
                + " -- regenerate case/actions.txt with coverage for them");
            System.exit(1);
        }

        // Recipe selection is intentionally not published by the current
        // combat action surface. It must be represented by the explicit
        // unsupported sentinel, rather than being forced to vary in a fixture
        // that cannot legally select it.
        for (Case one : cases) {
            if (one.recipe() != -1) {
                System.out.println("FAIL - recipe selection is unsupported by "
                    + "the current action surface but case " + one.name()
                    + " published recipe index " + one.recipe());
                System.exit(1);
            }
        }

        int checked = 0;
        int worldVerbCases = 0;
        List<String> failures = new ArrayList<>();
        boolean[] permissive = ActionDecoder.permissive();

        for (Case expected : cases) {
            ActionDecoder.Decoded actual =
                ActionDecoder.decode(expected.factors(), permissive);

            check(failures, expected.name(), "legal",
                expected.legal(), actual.actionLegal());
            check(failures, expected.name(), "skill",
                expected.skill(), actual.skillId());
            check(failures, expected.name(), "ability",
                expected.ability(), actual.abilitySlot());
            check(failures, expected.name(), "guard",
                expected.guard(), actual.guardHeld());
            check(failures, expected.name(), "dodge",
                expected.dodge(), actual.dodgeDirection());
            check(failures, expected.name(), "jump",
                expected.jump(), actual.jumpHeld());
            check(failures, expected.name(), "move",
                expected.move(), actual.worldMoveDirection());
            check(failures, expected.name(), "yaw",
                expected.yaw(), actual.yawDeltaDegrees());
            check(failures, expected.name(), "pitch",
                expected.pitch(), actual.pitchDeltaDegrees());
            check(failures, expected.name(), "use",
                expected.use(), actual.useRequested());
            check(failures, expected.name(), "blockTrigger",
                expected.blockTrigger(), actual.blockInteractionTrigger());
            check(failures, expected.name(), "recipeIndex",
                expected.recipe(), actual.recipeCandidateIndex());
            check(failures, expected.name(), "blockIndex",
                expected.block(), actual.blockCandidateIndex());

            boolean expectedWorldVerb = expected.use()
                || expected.recipe() >= 0
                || expected.block() >= 0;
            check(failures, expected.name(), "requestsWorldVerb",
                expectedWorldVerb, actual.requestsWorldVerb());
            if (expectedWorldVerb) {
                worldVerbCases++;
            } else {
                check(failures, expected.name(), "nativeLegal",
                    expected.nativeLegal(), actual.actionLegal());
                check(failures, expected.name(), "native.forward",
                    expected.nf(), actual.forward());
                check(failures, expected.name(), "native.back",
                    expected.nb(), actual.back());
                check(failures, expected.name(), "native.left",
                    expected.nl(), actual.left());
                check(failures, expected.name(), "native.right",
                    expected.nr(), actual.right());
                check(failures, expected.name(), "native.attack",
                    expected.nAttack(), actual.attack());
                check(failures, expected.name(), "native.jump",
                    expected.nJump(), actual.jumpHeld());
            }
            checked++;
        }

        // Mask handling: forbidding one logit must reject exactly the action
        // that selects it, and leave the others alone. Range checking alone
        // would pass this file's other cases while silently applying a
        // masked-off ability at runtime, so it is pinned separately.
        int maskChecked = 0;
        List<String> maskedLines =
            Files.readAllLines(directory.resolve("actions_masked.txt"));
        int forbidden = Integer.parseInt(maskedLines.get(0).split("\t")[1].trim());
        boolean[] restricted = ActionDecoder.permissive();
        restricted[forbidden] = false;
        for (String line : maskedLines.subList(1, maskedLines.size())) {
            if (line.isBlank()) {
                continue;
            }
            String[] parts = line.split("\t");
            ActionDecoder.Decoded actual =
                ActionDecoder.decode(parseFactors(parts[1]), restricted);
            check(failures, parts[0], "masked.legal",
                parts[2].equals("1"), actual.actionLegal());
            check(failures, parts[0], "masked.skill",
                Integer.parseInt(parts[3]), actual.skillId());
            maskChecked++;
        }

        System.out.printf("cases         : %d%n", checked);
        System.out.printf("world-verb    : %d (routed to composed sink)%n", worldVerbCases);
        System.out.printf("masked cases  : %d (forbidding logit %d)%n",
            maskChecked, forbidden);
        System.out.printf("failures      : %d%n", failures.size());
        if (!failures.isEmpty()) {
            for (String failure : failures) {
                System.out.println("  " + failure);
            }
            System.out.println("FAIL");
            System.exit(1);
        }
        System.out.println("PASS - Java action decoder reproduces the JAX decision");
    }

    private static void check(List<String> failures, String name, String field,
                              Object expected, Object actual) {
        if (!expected.equals(actual)) {
            failures.add(name + "." + field + ": expected " + expected
                + ", got " + actual);
        }
    }

    private static int[] parseFactors(String text) {
        String[] parts = text.split(",");
        int[] factors = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            factors[i] = Integer.parseInt(parts[i].trim());
        }
        return factors;
    }

    private static List<Case> load(Path path) throws IOException {
        List<Case> cases = new ArrayList<>();
        for (String line : Files.readAllLines(path)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] f = line.split("\t");
            cases.add(new Case(
                f[0], parseFactors(f[1]),
                f[2].equals("1"), Integer.parseInt(f[3]), Integer.parseInt(f[4]),
                f[5].equals("1"), Integer.parseInt(f[6]), f[7].equals("1"),
                Integer.parseInt(f[8]),
                Double.parseDouble(f[9]), Double.parseDouble(f[10]),
                f[11].equals("1"), Integer.parseInt(f[12]),
                Integer.parseInt(f[13]), Integer.parseInt(f[14]),
                f[15].equals("1"),
                f[16].equals("1"), f[17].equals("1"), f[18].equals("1"),
                f[19].equals("1"), f[20].equals("1"), f[21].equals("1")
            ));
        }
        return cases;
    }
}
