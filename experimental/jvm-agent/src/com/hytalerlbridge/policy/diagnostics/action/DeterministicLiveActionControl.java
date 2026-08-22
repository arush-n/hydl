package com.hytalerlbridge.policy.diagnostics.action;

import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.Policy;
import com.hytalerlbridge.policy.action.ActionSelectionControl;
import com.hytalerlbridge.policy.combat.bridge.BridgeCombatFacade;
import com.hytalerlbridge.policy.combat.bridge.BridgeDodgeCooldownStateSource;
import com.hytalerlbridge.policy.combat.bridge.ReflectiveBridgeDodgeContract;
import com.hytalerlbridge.policy.combat.bridge.ReflectiveBridgeDodgeCooldownStateSource;
import com.hytalerlbridge.policy.combat.runtime.CombatReceiptObserver;
import com.hytalerlbridge.policy.diagnostics.action.dodge.DodgeCooldownContract;
import com.hytalerlbridge.policy.diagnostics.action.dodge.DodgeCooldownDiagnostic;
import com.hytalerlbridge.policy.diagnostics.action.dodge.DodgeCooldownTraceSink;
import com.hytalerlbridge.policy.world.model.BlockCandidateBinding;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * One-shot, data-driven control for live interaction certification.
 *
 * <p>This is deliberately not a scripted action executor. It narrows the same
 * evidence-derived mask supplied to the trained policy, lets the policy rank
 * every remaining choice with its own logits, and then uses the normal decoder
 * and production sink. It never names a weapon, recipe, item, block, ability,
 * candidate slot, reach, duration, charge, or damage scalar.</p>
 *
 * <p>Attack and World probes are separate modes. Combining them into a timed
 * script would require inventing a delay and could interrupt the exact native
 * lifecycle the attack probe exists to measure.</p>
 */
public final class DeterministicLiveActionControl
    implements ActionSelectionControl, CombatReceiptObserver {

    public static final String SCHEMA = "hytalerl_live_action_probe_v1";
    public static final String FILE_NAME = "live-test-control.txt";

    public enum Mode {
        DISABLED,
        ATTACK_ONCE,
        WORLD_ONCE,
        DODGE_COOLDOWN
    }

    public enum WorldRoot {
        ANY,
        RECIPE,
        BLOCK
    }

    private final Mode mode;
    private final int actorSlot;
    private final WorldRoot worldRoot;
    private final DodgeCooldownDiagnostic dodgeDiagnostic;
    private final BridgeDodgeCooldownStateSource dodgeCooldownStateSource;
    private boolean emitted;
    private long waitingSamples;

    private DeterministicLiveActionControl(
        Mode mode,
        int actorSlot,
        WorldRoot worldRoot,
        DodgeCooldownDiagnostic dodgeDiagnostic,
        BridgeDodgeCooldownStateSource dodgeCooldownStateSource
    ) {
        if (mode == null || worldRoot == null || actorSlot < 0) {
            throw new IllegalArgumentException(
                "live action control requires a mode, root, and nonnegative slot");
        }
        if ((mode == Mode.DODGE_COOLDOWN) != (dodgeDiagnostic != null)) {
            throw new IllegalArgumentException(
                "Dodge cooldown mode and diagnostic must be supplied together");
        }
        this.mode = mode;
        this.actorSlot = actorSlot;
        this.worldRoot = worldRoot;
        this.dodgeDiagnostic = dodgeDiagnostic;
        this.dodgeCooldownStateSource = dodgeCooldownStateSource == null
            ? BridgeDodgeCooldownStateSource.NONE : dodgeCooldownStateSource;
    }

    public static DeterministicLiveActionControl disabled() {
        return new DeterministicLiveActionControl(
            Mode.DISABLED,
            0,
            WorldRoot.ANY,
            null,
            BridgeDodgeCooldownStateSource.NONE
        );
    }

    public static DeterministicLiveActionControl attackOnce(int actorSlot) {
        return new DeterministicLiveActionControl(
            Mode.ATTACK_ONCE,
            actorSlot,
            WorldRoot.ANY,
            null,
            BridgeDodgeCooldownStateSource.NONE
        );
    }

    public static DeterministicLiveActionControl worldOnce(
        int actorSlot,
        WorldRoot root
    ) {
        return new DeterministicLiveActionControl(
            Mode.WORLD_ONCE,
            actorSlot,
            root,
            null,
            BridgeDodgeCooldownStateSource.NONE
        );
    }

    /** Server-free construction seam for the deterministic Dodge gate. */
    public static DeterministicLiveActionControl dodgeCooldown(
        int actorSlot,
        long seed,
        DodgeCooldownContract contract
    ) {
        DodgeCooldownDiagnostic diagnostic = new DodgeCooldownDiagnostic(
            actorSlot,
            seed,
            contract,
            DodgeCooldownTraceSink.disabled()
        );
        return new DeterministicLiveActionControl(
            Mode.DODGE_COOLDOWN,
            actorSlot,
            WorldRoot.ANY,
            diagnostic,
            BridgeDodgeCooldownStateSource.NONE
        );
    }

    /** Load the opt-in control file, or return a disabled control if absent. */
    public static DeterministicLiveActionControl load(Path file)
        throws IOException {
        if (!Files.exists(file)) return disabled();
        Map<String, String> fields = new HashMap<>();
        for (String raw : Files.readAllLines(file)) {
            String line = raw.strip();
            if (line.isEmpty() || line.startsWith("#")) continue;
            int split = line.indexOf('=');
            if (split <= 0 || split == line.length() - 1) {
                throw new IOException(
                    FILE_NAME + " requires key=value lines, got '" + line + "'");
            }
            String key = line.substring(0, split).strip();
            String value = line.substring(split + 1).strip();
            if (fields.putIfAbsent(key, value) != null) {
                throw new IOException(
                    FILE_NAME + " repeats key '" + key + "'");
            }
        }
        for (String key : fields.keySet()) {
            if (!key.equals("schema") && !key.equals("mode")
                && !key.equals("actor_slot")
                && !key.equals("world_root")
                && !key.equals("seed")) {
                throw new IOException(
                    FILE_NAME + " has unknown key '" + key + "'");
            }
        }
        if (!SCHEMA.equals(fields.get("schema"))) {
            throw new IOException(FILE_NAME + " has an unsupported schema");
        }
        Mode mode = switch (required(fields, "mode")) {
            case "attack_once" -> Mode.ATTACK_ONCE;
            case "world_once" -> Mode.WORLD_ONCE;
            case "dodge_cooldown" -> Mode.DODGE_COOLDOWN;
            default -> throw new IOException(
                FILE_NAME
                    + " mode must be attack_once, world_once, or dodge_cooldown");
        };
        int actorSlot;
        try {
            actorSlot = Integer.parseInt(required(fields, "actor_slot"));
        } catch (NumberFormatException invalid) {
            throw new IOException(FILE_NAME + " actor_slot is not an integer", invalid);
        }
        if (actorSlot < 0) {
            throw new IOException(FILE_NAME + " actor_slot must be nonnegative");
        }
        if (mode == Mode.DODGE_COOLDOWN && actorSlot == 0) {
            throw new IOException(
                FILE_NAME + " dodge_cooldown requires a nonzero actor_slot");
        }
        String requestedRoot = fields.getOrDefault("world_root", "any");
        WorldRoot root = switch (requestedRoot) {
            case "any" -> WorldRoot.ANY;
            case "recipe" -> WorldRoot.RECIPE;
            case "block" -> WorldRoot.BLOCK;
            default -> throw new IOException(
                FILE_NAME + " world_root must be any, recipe, or block");
        };
        if (mode != Mode.WORLD_ONCE && fields.containsKey("world_root")) {
            throw new IOException(
                FILE_NAME + " world_root is valid only for world_once");
        }
        if (mode != Mode.DODGE_COOLDOWN && fields.containsKey("seed")) {
            throw new IOException(
                FILE_NAME + " seed is valid only for dodge_cooldown");
        }
        DodgeCooldownDiagnostic dodge = null;
        BridgeDodgeCooldownStateSource dodgeState =
            BridgeDodgeCooldownStateSource.NONE;
        if (mode == Mode.DODGE_COOLDOWN) {
            long seed;
            try {
                seed = Long.parseLong(required(fields, "seed"));
            } catch (NumberFormatException invalid) {
                throw new IOException(FILE_NAME + " seed is not an integer", invalid);
            }
            DodgeCooldownContract contract;
            try {
                contract = ReflectiveBridgeDodgeContract.load();
            } catch (ReflectiveOperationException invalid) {
                throw new IOException(
                    FILE_NAME + " cannot read deployed Dodge cooldown metadata",
                    invalid
                );
            }
            try {
                dodgeState = ReflectiveBridgeDodgeCooldownStateSource.load();
            } catch (ReflectiveOperationException invalid) {
                throw new IOException(
                    FILE_NAME
                        + " cannot read deployed Dodge cooldown state view",
                    invalid
                );
            }
            DodgeCooldownTraceSink trace = DodgeCooldownTraceSink.live(
                file.toAbsolutePath().getParent(), actorSlot, seed, 4096);
            dodge = new DodgeCooldownDiagnostic(
                actorSlot, seed, contract, trace);
        }
        return new DeterministicLiveActionControl(
            mode, actorSlot, root, dodge, dodgeState);
    }

    @Override
    public synchronized Plan plan(
        int slot,
        float[] logits,
        boolean[] legal,
        WorldActionEvidence evidence
    ) {
        validateInputs(logits, legal, evidence);
        if (mode == Mode.DODGE_COOLDOWN) {
            return dodgeDiagnostic.plan(slot, logits, legal, evidence);
        }
        if (mode == Mode.DISABLED || slot != actorSlot) {
            return Plan.pass(legal);
        }
        if (emitted) {
            Plan hold = holdPlan(legal);
            if (!hold.constrained()) {
                throw new IllegalStateException(
                    "live action probe cannot hold a neutral post-edge action");
            }
            return hold;
        }
        Plan result = mode == Mode.ATTACK_ONCE
            ? attackPlan(logits, legal)
            : worldPlan(logits, legal, evidence);
        if (!result.constrained()) waitingSamples++;
        return result;
    }

    @Override
    public synchronized void observe(
        int slot,
        Plan plan,
        ActionDecoder.Decoded action
    ) {
        if (mode == Mode.DODGE_COOLDOWN) {
            dodgeDiagnostic.observe(slot, plan, action);
            return;
        }
        if (!plan.constrained()) return;
        if (slot != actorSlot || action == null
            || !action.actionLegal()) {
            throw new IllegalStateException(
                "live action probe did not produce its constrained action");
        }
        boolean holding = plan.route().equals("hold_after_attack")
            || plan.route().equals("hold_after_world");
        boolean matches = switch (plan.route()) {
            case "attack" -> action.attack() && !action.requestsWorldVerb()
                && action.abilitySlot() < 0 && !action.guardHeld();
            case "recipe" -> action.recipeCandidateIndex() >= 0
                && action.blockCandidateIndex() < 0
                && !action.useRequested();
            case "use" -> action.useRequested()
                && action.blockCandidateIndex() >= 0
                && action.recipeCandidateIndex() < 0;
            case "primary" -> !action.useRequested()
                && action.blockCandidateIndex() >= 0
                && action.recipeCandidateIndex() < 0
                && action.blockInteractionTrigger() == 1;
            case "secondary" -> !action.useRequested()
                && action.blockCandidateIndex() >= 0
                && action.recipeCandidateIndex() < 0
                && action.blockInteractionTrigger() == 2;
            case "hold_after_attack", "hold_after_world" ->
                !action.attack() && action.abilitySlot() < 0
                && !action.guardHeld() && action.dodgeDirection() == 0
                && !action.jumpHeld() && action.worldMoveDirection() == 0
                && !action.forward() && !action.back()
                && !action.left() && !action.right()
                && !action.requestsWorldVerb()
                && action.yawDeltaDegrees() == 0.0
                && action.pitchDeltaDegrees() == 0.0;
            default -> false;
        };
        if (!matches) {
            throw new IllegalStateException(
                "live action probe route '" + plan.route()
                    + "' decoded to an incompatible action");
        }
        if (!holding) emitted = true;
    }

    public Mode mode() {
        return mode;
    }

    public int actorSlot() {
        return actorSlot;
    }

    public synchronized boolean emitted() {
        return emitted;
    }

    public synchronized long waitingSamples() {
        return waitingSamples;
    }

    @Override
    public synchronized String describe() {
        if (mode == Mode.DISABLED) return "disabled";
        if (mode == Mode.DODGE_COOLDOWN) {
            return "schema=" + SCHEMA + " mode=dodge_cooldown "
                + dodgeDiagnostic.describe();
        }
        return "schema=" + SCHEMA
            + " mode=" + mode.name().toLowerCase()
            + " actor_slot=" + actorSlot
            + (mode == Mode.WORLD_ONCE
                ? " world_root=" + worldRoot.name().toLowerCase() : "")
            + " emitted=" + emitted
            + " neutral_hold=" + emitted
            + " waiting_samples=" + waitingSamples;
    }

    @Override
    public synchronized void observeCombatReceipt(
        CombatReceiptObserver.Phase phase,
        long worldTick,
        int slot,
        ActionDecoder.Decoded action,
        boolean firstControlTick,
        BridgeCombatFacade.Receipt receipt,
        BridgeDodgeCooldownStateSource.State dodgeCooldownBefore,
        BridgeDodgeCooldownStateSource.State dodgeCooldownAfter
    ) {
        if (mode == Mode.DODGE_COOLDOWN) {
            dodgeDiagnostic.observeCombatReceipt(
                phase,
                worldTick,
                slot,
                action,
                firstControlTick,
                receipt,
                dodgeCooldownBefore,
                dodgeCooldownAfter
            );
        }
    }

    /** Read-only bridge state source; configured only for Dodge diagnostics. */
    public BridgeDodgeCooldownStateSource dodgeCooldownStateSource() {
        return dodgeCooldownStateSource;
    }

    public synchronized DodgeCooldownDiagnostic dodgeDiagnostic() {
        return dodgeDiagnostic;
    }

    private static Plan attackPlan(float[] logits, boolean[] legal) {
        boolean[] constrained = legal.clone();
        neutralizeStandardRoots(constrained);
        stabilizeIndependentHeads(constrained);
        int base = offset(ActionDecoder.BASE_ACTION);
        for (int index = 0; index < Policy.HEAD_SIZES[ActionDecoder.BASE_ACTION];
                index++) {
            boolean attack = index == ActionDecoder.SKILL_ATTACK
                || index == ActionDecoder.SKILL_APPROACH_ATTACK
                || index == ActionDecoder.SKILL_RETREAT_ATTACK;
            constrained[base + index] = attack && legal[base + index];
        }
        return viable(constrained)
            ? new Plan(constrained, true, "attack")
            : Plan.pass(legal);
    }

    private Plan worldPlan(
        float[] logits,
        boolean[] legal,
        WorldActionEvidence evidence
    ) {
        Candidate best = null;
        // The recipe route is gone with its head. `WorldRoot.RECIPE` remains a
        // selectable *request* so an operator asking for it gets an empty plan
        // rather than a silently substituted block route.
        if (worldRoot != WorldRoot.RECIPE) {
            best = better(best, blockCandidate(
                "use", logits, legal, evidence));
            best = better(best, blockCandidate(
                "primary", logits, legal, evidence));
            best = better(best, blockCandidate(
                "secondary", logits, legal, evidence));
        }
        return best == null
            ? Plan.pass(legal)
            : new Plan(best.mask(), true, best.route());
    }

    private Plan holdPlan(boolean[] legal) {
        boolean[] constrained = legal.clone();
        neutralizeStandardRoots(constrained);
        retainOnly(constrained, ActionDecoder.BASE_ACTION,
            ActionDecoder.SKILL_IDLE);
        stabilizeIndependentHeads(constrained);
        retainOnly(constrained, ActionDecoder.BLOCK_TRIGGER, 0);
        return viable(constrained)
            ? new Plan(
                constrained,
                true,
                mode == Mode.ATTACK_ONCE
                    ? "hold_after_attack" : "hold_after_world")
            : Plan.pass(legal);
    }

    private static Candidate blockCandidate(
        String route,
        float[] logits,
        boolean[] legal,
        WorldActionEvidence evidence
    ) {
        boolean[] constrained = legal.clone();
        neutralizeStandardRoots(constrained);
        retainOnly(constrained, ActionDecoder.BASE_ACTION,
            ActionDecoder.SKILL_IDLE);
        stabilizeIndependentHeads(constrained);
        int block = offset(ActionDecoder.BLOCK);
        constrained[block] = false;
        boolean any = false;
        for (int factor = 1;
                factor < Policy.HEAD_SIZES[ActionDecoder.BLOCK]; factor++) {
            BlockCandidateBinding binding = evidence.blockBinding(factor - 1);
            boolean operation = switch (route) {
                case "use" -> binding.use().available();
                case "primary" -> binding.primary().available();
                case "secondary" -> binding.secondary().available();
                default -> throw new IllegalArgumentException(
                    "unsupported block route '" + route + "'");
            };
            boolean keep = legal[block + factor]
                && binding.available() && operation;
            constrained[block + factor] = keep;
            any |= keep;
        }
        int use = offset(ActionDecoder.USE);
        int trigger = offset(ActionDecoder.BLOCK_TRIGGER);
        if (route.equals("use")) {
            constrained[use] = false;
            constrained[use + 1] = legal[use + 1];
        } else {
            constrained[use] = legal[use];
            constrained[use + 1] = false;
            int selectedTrigger = route.equals("primary") ? 0 : 1;
            for (int index = 0;
                    index < Policy.HEAD_SIZES[ActionDecoder.BLOCK_TRIGGER];
                    index++) {
                constrained[trigger + index] = index == selectedTrigger
                    && legal[trigger + index];
            }
        }
        if (!any || !viable(constrained)) return null;
        return candidate(route, constrained, logits);
    }

    private static void neutralizeStandardRoots(boolean[] mask) {
        int base = offset(ActionDecoder.BASE_ACTION);
        for (int index = 0; index < Policy.HEAD_SIZES[ActionDecoder.BASE_ACTION];
                index++) {
            boolean attack = index == ActionDecoder.SKILL_ATTACK
                || index == ActionDecoder.SKILL_APPROACH_ATTACK
                || index == ActionDecoder.SKILL_RETREAT_ATTACK;
            if (attack) mask[base + index] = false;
        }
        retainOnly(mask, ActionDecoder.ABILITY, 0);
        retainOnly(mask, ActionDecoder.GUARD, 0);
        retainOnly(mask, ActionDecoder.USE, 0);
        retainOnly(mask, ActionDecoder.BLOCK, 0);
    }

    private static void stabilizeIndependentHeads(boolean[] mask) {
        // Locomotion choice 0 is idle, which pins gait, compass AND dodge in
        // one head -- they used to be three separate retainOnly calls.
        retainOnly(mask, ActionDecoder.LOCOMOTION, ActionDecoder.LOCOMOTION_IDLE);
        retainOnly(mask, ActionDecoder.JUMP, 0);
        retainOnly(mask, ActionDecoder.HOTBAR, 0);
        // The middle bin of an odd signed-delta head is its zero.
        retainOnly(mask, ActionDecoder.YAW_BINS,
            Policy.HEAD_SIZES[ActionDecoder.YAW_BINS] / 2);
        retainOnly(mask, ActionDecoder.BODY_YAW_BINS,
            Policy.HEAD_SIZES[ActionDecoder.BODY_YAW_BINS] / 2);
        retainOnly(mask, ActionDecoder.PITCH_BINS,
            Policy.HEAD_SIZES[ActionDecoder.PITCH_BINS] / 2);
    }

    private static void retainOnly(boolean[] mask, int head, int selected) {
        int start = offset(head);
        for (int index = 0; index < Policy.HEAD_SIZES[head]; index++) {
            if (index != selected) mask[start + index] = false;
        }
    }

    private static Candidate candidate(
        String route,
        boolean[] mask,
        float[] logits
    ) {
        float[] masked = masked(logits, mask);
        int[] factors = Policy.factors(masked);
        ActionDecoder.Decoded action = ActionDecoder.decode(factors, mask);
        if (!action.actionLegal()) return null;
        float score = 0.0f;
        for (int head = 0; head < factors.length; head++) {
            score += logits[offset(head) + factors[head]];
        }
        return new Candidate(route, mask.clone(), score);
    }

    private static Candidate better(Candidate current, Candidate candidate) {
        if (candidate == null) return current;
        if (current == null || candidate.score() > current.score()) {
            return candidate;
        }
        // Route order is a deterministic tie-break only. Non-tied decisions
        // are selected entirely by the policy's joint logits.
        if (candidate.score() == current.score()
            && candidate.route().compareTo(current.route()) < 0) {
            return candidate;
        }
        return current;
    }

    private static float[] masked(float[] logits, boolean[] mask) {
        float[] result = logits.clone();
        for (int index = 0; index < result.length; index++) {
            if (!mask[index]) result[index] = -1.0e9f;
        }
        return result;
    }

    private static boolean viable(boolean[] mask) {
        for (int head = 0; head < Policy.HEAD_SIZES.length; head++) {
            boolean any = false;
            int start = offset(head);
            for (int index = 0; index < Policy.HEAD_SIZES[head]; index++) {
                any |= mask[start + index];
            }
            if (!any) return false;
        }
        return true;
    }

    private static int offset(int head) {
        int result = 0;
        for (int index = 0; index < head; index++) {
            result += Policy.HEAD_SIZES[index];
        }
        return result;
    }

    private static void validateInputs(
        float[] logits,
        boolean[] legal,
        WorldActionEvidence evidence
    ) {
        int width = 0;
        for (int size : Policy.HEAD_SIZES) width += size;
        if (logits == null || logits.length != width
            || legal == null || legal.length != width || evidence == null) {
            throw new IllegalArgumentException(
                "live action control input width drift");
        }
        for (float logit : logits) {
            if (!Float.isFinite(logit)) {
                throw new IllegalArgumentException(
                    "live action control requires finite policy logits");
            }
        }
    }

    private static String required(Map<String, String> fields, String key)
        throws IOException {
        String value = fields.get(key);
        if (value == null || value.isEmpty()) {
            throw new IOException(FILE_NAME + " requires '" + key + "'");
        }
        return value;
    }

    private record Candidate(String route, boolean[] mask, float score) {}
}
