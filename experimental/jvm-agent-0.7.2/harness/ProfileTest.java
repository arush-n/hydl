import com.hytalerlbridge.policy.ObservationAssembler;
import com.hytalerlbridge.policy.Policy;
import com.hytalerlbridge.policy.PolicyPerception;
import com.hytalerlbridge.policy.Projection;
import com.hytalerlbridge.policy.perception.LivePolicyPerception;
import com.hytalerlbridge.policy.perception.model.PerceptionFrame;
import com.hytalerlbridge.policy.perception.profile.PerceptionProfile;
import com.hytalerlbridge.policy.perception.profile.StatusProgramCatalog;
import com.hytalerlbridge.policy.perception.projection.InventoryProjection;
import com.hytalerlbridge.policy.perception.projection.GeometryProjection;
import com.hytalerlbridge.policy.perception.projection.LightProjection;
import com.hytalerlbridge.policy.perception.projection.RecipeCandidateProjection;
import com.hytalerlbridge.policy.perception.projection.WorldProjection;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/** Offline contract checks for the production perception-profile loader. */
public final class ProfileTest {

    /** Same threshold {@code ParityTest} holds the whole fixture to. */
    private static final float PARITY_TOLERANCE = 1.0e-6f;

    public static void main(String[] args) throws IOException {
        // The profile carries the weapon-pinned authored metadata (divisors,
        // bounds, ability bindings, status catalog); a state carries one tick
        // of raw evidence. They lived in one directory, so the production path
        // could only ever be certified on the single state that happened to
        // ship a profile.json -- the reset tick of the baseline weapon, which
        // export_parity.py's own docstring calls the regime *least* likely to
        // expose a float32 difference or a masking mistake.
        //
        // Splitting them lets one profile compose against every state of its
        // own weapon. The pairing is still not free: profile.json is pinned to
        // (profile, opponent), so pointing this at a state from a different
        // weapon fails on the authored metadata rather than on a real defect.
        //
        // `--all <profiles-root> <parity-root>` sweeps every exported weapon
        // against every state of its own pairing. That mode exists because
        // certifying one weapon says nothing about the other eleven: each has
        // its own authored ability table, resource bounds and status catalog,
        // and those are exactly the inputs the production path consumes.
        if (args.length > 0 && args[0].equals("--all")) {
            require(args.length == 3 || args.length == 4,
                "--all takes <profiles-root> <parity-root> [state-prefix]");
            sweep(Path.of(args[1]), Path.of(args[2]),
                args.length == 4 ? args[3] : "");
            return;
        }
        Path profileRoot = Path.of(args.length > 0 ? args[0] : "case/profile");
        List<Path> states = new ArrayList<>();
        for (int i = 1; i < args.length; i++) {
            states.add(Path.of(args[i]));
        }
        if (states.isEmpty()) {
            states.add(profileRoot);
        }
        PerceptionProfile profile = PerceptionProfile.load(profileRoot);

        float[] assembled = ObservationAssembler.assemble(
            profile.template(), Map.of(), Map.of());
        require(
            Arrays.equals(
                assembled, Policy.read(profileRoot, "expected_observation")),
            "cached profile template changed the observation");
        require(profile.sha256().matches("[0-9A-F]{64}"),
            "profile identity is not a canonical SHA-256");
        require(
            profile.sha256().equals(
                PerceptionProfile.load(profileRoot).sha256()),
            "profile identity is not deterministic");
        require(profile.params().targetAbilitySlotDivisor() == 15.0,
            "production profile uses the legacy attack-count divisor");

        Projection.TargetRaw nonzeroAbilitySlot = new Projection.TargetRaw(
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            profile.params().targetMaxHealth(),
            0.0, 0.0, 0.0,
            true, true, Projection.PHASE_IDLE, 0.0, 3);
        require(Math.abs(
                Projection.targetEvidence(nonzeroAbilitySlot, profile.params())
                    .normalizedAttackIndex()
                    - 0.2f) < 1.0e-7f,
            "non-zero authored ability slot is not normalized by 15");

        float[] minimum = profile.resourceMinimum();
        minimum[0] += 100.0f;
        require(profile.resourceMinimum()[0] != minimum[0],
            "profile exposed mutable resource metadata");
        // The table is policy-contract-wide: role-owned environmental
        // immunity plus the two entity-interaction crossbow combo effects.
        require(profile.statusPrograms().size() == 3,
            "current policy status program count drifted");
        require(
            profile.statusPrograms().program("Immunity_Environmental") != null,
            "semantic status lookup did not reproduce JAX FNV-1a");
        require(profile.statusPrograms().omitted("Red_Flash"),
            "visual-only native status was not omitted");
        require(
            StatusProgramCatalog.semanticId("Immunity_Environmental")
                == 1315835084,
            "Java semantic ID differs from JAX");

        float worst = 0.0f;
        String worstWhere = "-";
        for (Path state : states) {
            float delta = compose(profile, state, state.equals(profileRoot));
            if (delta > worst) {
                worst = delta;
                worstWhere = state.getFileName().toString();
            }
        }

        System.out.println("profile sha256 : " + profile.sha256());
        System.out.println("states composed: " + states.size());
        System.out.printf("worst delta    : %.3e (%s)%n", worst, worstWhere);
        System.out.println(
            "PASS - production perception composition reproduces exact JAX metadata");
    }

    /**
     * Compose every exported weapon profile against every parity state of its
     * own pairing.
     *
     * <p>The pass condition is deliberately two-sided. Reproducing the rows is
     * necessary but not sufficient: a sweep that silently found no profiles, or
     * paired a profile with none of its states, would also report success
     * without comparing anything. So this additionally requires that every
     * profile directory contributed at least one state, and prints the per-
     * weapon counts, because "12 weapons green" is only meaningful next to the
     * number of states each one actually covered.
     */
    private static void sweep(
        Path profilesRoot, Path parityRoot, String statePrefix
    ) throws IOException {
        require(Files.isDirectory(profilesRoot),
            "no profiles root at " + profilesRoot
                + " -- run export_profile.py <weapon> for each weapon");
        List<Path> profileDirs = new ArrayList<>();
        try (Stream<Path> entries = Files.list(profilesRoot)) {
            entries.filter(p -> Files.exists(p.resolve("profile.json")))
                .sorted()
                .forEach(profileDirs::add);
        }
        require(!profileDirs.isEmpty(),
            "profiles root " + profilesRoot + " contains no profile.json");

        float worst = 0.0f;
        String worstWhere = "-";
        int totalStates = 0;
        List<String> barren = new ArrayList<>();
        System.out.printf("%-18s %-8s %s%n", "profile", "states", "worst");
        for (Path profileDir : profileDirs) {
            String weapon = profileDir.getFileName().toString();
            PerceptionProfile profile = certify(profileDir);

            List<Path> states = new ArrayList<>();
            try (Stream<Path> entries = Files.list(parityRoot)) {
                entries.filter(Files::isDirectory)
                    .filter(p -> p.getFileName().toString().matches(
                        java.util.regex.Pattern.quote(statePrefix + weapon)
                            + "_t\\d+"))
                    .sorted()
                    .forEach(states::add);
            }
            if (states.isEmpty()) {
                barren.add(weapon);
                continue;
            }
            float weaponWorst = 0.0f;
            for (Path state : states) {
                weaponWorst = Math.max(
                    weaponWorst, compose(profile, state, false));
            }
            totalStates += states.size();
            if (weaponWorst > worst) {
                worst = weaponWorst;
                worstWhere = weapon;
            }
            System.out.printf(
                "%-18s %-8d %.3e%n", weapon, states.size(), weaponWorst);
        }

        require(barren.isEmpty(), "these profiles matched no parity state, so "
            + "they certified nothing: " + barren
            + " -- the parity fixture and the exported profiles disagree about "
            + "which weapons exist");
        require(totalStates >= profileDirs.size(),
            "swept " + profileDirs.size() + " profiles but only " + totalStates
                + " states");

        System.out.println("profiles swept : " + profileDirs.size());
        System.out.println("states composed: " + totalStates);
        System.out.printf("worst delta    : %.3e (%s)%n", worst, worstWhere);
        System.out.println("PASS - every exported weapon profile reproduces "
            + "the JAX row for every state of its own pairing");
    }

    /**
     * Profile-level invariants that hold for any weapon.
     *
     * <p>Weapon-specific pins (the iron-sword status-program count, say) stay
     * in the single-profile path: this mode certifies composition across
     * weapons, and re-pinning each weapon's catalog here would just encode
     * whatever the exporter produced.
     */
    private static PerceptionProfile certify(Path profileRoot)
        throws IOException {
        PerceptionProfile profile = PerceptionProfile.load(profileRoot);
        float[] assembled = ObservationAssembler.assemble(
            profile.template(), Map.of(), Map.of());
        require(
            Arrays.equals(
                assembled, Policy.read(profileRoot, "expected_observation")),
            profileRoot + ": cached template changed the observation");
        require(profile.sha256().matches("[0-9A-F]{64}"),
            profileRoot + ": profile identity is not a canonical SHA-256");
        require(profile.params().targetAbilitySlotDivisor() == 15.0,
            profileRoot + ": uses the legacy attack-count divisor");
        require(profile.statusPrograms().size() >= 1,
            profileRoot + ": exported no status programs at all");
        return profile;
    }

    /**
     * Drive the production path for one state and require it to reproduce that
     * state's own JAX row.
     *
     * <p>Everything here reads its evidence from {@code state} and its authored
     * metadata from {@code profile}. That separation is the point: it is what
     * lets the single shipped profile certify every tick of its own weapon
     * rather than only the tick it was exported beside.
     *
     * <p>{@code exact} is not a convenience dial. Against the state the profile
     * shipped beside, the production path reproduces every nonzero JAX value
     * bit-for-bit; IEEE {@code -0.0} and {@code +0.0} are treated as the same
     * observation. Against a state written by
     * {@code export_parity.py}, the reference itself came from a different
     * exporter that associates float32 differently: the two fixtures disagree
     * by 1 ULP at column 16 for what is nominally the same iron-sword reset
     * tick ({@code 0.10416666} vs {@code 0.10416667}), and Java matches the
     * profile's. Demanding bit-equality there would assert that the two
     * exporters agree, which is a claim about the fixtures, not about this
     * port. So those states are held to {@code ParityTest}'s own 1.0e-6, and
     * the worst delta is printed so drift stays visible instead of absorbed.
     *
     * @return the worst absolute observation delta seen for this state
     */
    private static float compose(
        PerceptionProfile profile, Path state, boolean exact
    ) throws IOException {
        float[] expected = Policy.read(state, "expected_observation");

        float[] rawAbility = Policy.read(state, "raw_ability");
        var ability = Projection.abilityFeatures(profile.abilities(
            slice(rawAbility, 0, 14),
            slice(rawAbility, 92, 32),
            integers(rawAbility, 124, 2),
            slice(rawAbility, 126, 2),
            booleans(rawAbility, 836, 32)
        ));
        float worst = agree(ability.values(), Policy.read(state, "ability_f32"),
            exact, state + ": profile ability metadata");
        require(Arrays.equals(ability.mask(), Policy.readMask(state, "ability_mask")),
            state + ": profile ability metadata does not reproduce JAX mask");
        require(Arrays.equals(ability.legal(), Policy.readMask(state, "ability_legal")),
            state + ": profile ability metadata does not reproduce JAX legality");

        float[] current = Policy.read(state, "raw_resource_current");
        var resources = Projection.resourceFeatures(profile.resources(current));
        worst = Math.max(worst, agree(
            resources.values(), Policy.read(state, "resource_f32"),
            exact, state + ": profile resource bounds"));
        require(Arrays.equals(resources.mask(), Policy.readMask(state, "resource_mask")),
            state + ": profile resource bounds do not reproduce JAX mask");

        PerceptionFrame frame = frame(state);
        LivePolicyPerception live = new LivePolicyPerception(
            profile,
            (ref, npc, store, slot, deltaTime) -> frame
        );
        PolicyPerception.Sample sample = live.sample(
            null, null, null, 0, 1.0f / 30.0f);
        require(sample != null, state + ": valid exact-JAX frame was skipped");
        worst = Math.max(worst, agree(sample.observation(), expected, exact,
            state + ": production live composition"));
        require(Arrays.equals(
                sample.actionMask(), Policy.readMask(state, "action_mask")),
            state + ": production live composition does not reproduce JAX "
                + "legality");
        return worst;
    }

    /**
     * Compare derived floats against JAX and return the worst absolute delta.
     *
     * <p>Masks are never compared this way -- a boolean has no ULP, so any
     * disagreement there is a real masking defect and stays exact.
     */
    private static float agree(
        float[] actual, float[] jax, boolean exact, String what
    ) {
        require(actual.length == jax.length, what + " has width "
            + actual.length + ", JAX wrote " + jax.length);
        if (exact) {
            require(sameValues(actual, jax), what + " is not value-bit-exact "
                + "against the fixture it shipped beside, at "
                + firstDifference(actual, jax));
            return 0.0f;
        }
        float worst = 0.0f;
        int where = -1;
        for (int i = 0; i < actual.length; i++) {
            float delta = Math.abs(actual[i] - jax[i]);
            if (delta > worst) {
                worst = delta;
                where = i;
            }
        }
        require(worst <= PARITY_TOLERANCE, what + " exceeds the parity "
            + "tolerance: worst " + worst + " at column " + where
            + " (" + (where < 0 ? "-" : actual[where] + " != " + jax[where])
            + ")");
        return worst;
    }

    private static PerceptionFrame frame(Path state) throws IOException {
        Map<String, Double> in = readValues(state.resolve("inputs.txt"));
        Projection.SelfState self = new Projection.SelfState(
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
            in.get("agent_motion_delta_seconds")
        );
        Projection.TargetState target = new Projection.TargetState(
            in.get("agent_position_x"), in.get("agent_position_y"),
            in.get("agent_position_z"),
            in.get("agent_velocity_x"), in.get("agent_velocity_y"),
            in.get("agent_velocity_z"), in.get("agent_yaw_degrees"),
            in.get("target_position_x"), in.get("target_position_y"),
            in.get("target_position_z"),
            in.get("target_velocity_x"), in.get("target_velocity_y"),
            in.get("target_velocity_z"), in.get("target_health"),
            in.get("target_visible"), in.get("target_facing_error"),
            in.get("target_head_facing_error"), in.get("target_head_pitch"),
            in.get("target_attack_progress")
        );
        Projection.CombatSelf combatSelf = new Projection.CombatSelf(
            in.get("agent_velocity_x"), in.get("agent_velocity_y"),
            in.get("agent_velocity_z"), in.get("agent_yaw_degrees"),
            in.get("agent_health_fraction"),
            in.get("agent_attack_executing_bool") != 0.0
        );
        float[] rawAbility = Policy.read(state, "raw_ability");
        Projection.TargetRaw targetRaw = new Projection.TargetRaw(
            in.get("agent_position_x"), in.get("agent_position_y"),
            in.get("agent_position_z"),
            in.get("target_position_x"), in.get("target_position_y"),
            in.get("target_position_z"),
            in.get("target_velocity_x"), in.get("target_velocity_y"),
            in.get("target_velocity_z"), in.get("target_health"),
            in.get("target_yaw_degrees"),
            in.get("target_head_yaw_degrees"),
            in.get("target_head_pitch_degrees"),
            in.get("target_perceptible") != 0.0,
            true,
            (int) Math.round(in.get("target_attack_phase")),
            in.get("target_attack_progress_raw"),
            targetReportedAbilitySlot(rawAbility)
        );

        float[] rawDefense = Policy.read(state, "raw_defense");
        PerceptionFrame.Defense defense = new PerceptionFrame.Defense(
            booleans(rawDefense, 0, 2), booleans(rawDefense, 2, 2),
            slice(rawDefense, 4, 2), slice(rawDefense, 8, 6),
            slice(rawDefense, 16, 2), slice(rawDefense, 18, 2),
            slice(rawDefense, 20, 2)
        );
        float[] rawStatus = Policy.read(state, "raw_status");
        PerceptionFrame.Status status = new PerceptionFrame.Status(
            slice(rawStatus, 16, 16), slice(rawStatus, 32, 16),
            slice(rawStatus, 48, 16), slice(rawStatus, 64, 16),
            slice(rawStatus, 80, 16), integers(rawStatus, 96, 16),
            slice(rawStatus, 112, 16), slice(rawStatus, 128, 16),
            booleans(rawStatus, 144, 16)
        );
        PerceptionFrame.Ability ability = new PerceptionFrame.Ability(
            slice(rawAbility, 92, 32), integers(rawAbility, 124, 2),
            slice(rawAbility, 126, 2), booleans(rawAbility, 836, 32)
        );
        float[] rawDodge = Policy.read(state, "raw_dodge");
        PerceptionFrame.Mechanics mechanics = new PerceptionFrame.Mechanics(
            Policy.read(state, "raw_resource_current"),
            Policy.readMask(state, "resource_mask"),
            defense, status, ability, booleans(rawDodge, 0, 4),
            rawDodge[4] > 0.5f, rawDodge[6]
        );

        boolean[] inventoryFlags = Policy.readMask(
            state, "raw_inventory_flags");
        int[] inventoryI32 = readInts(
            state.resolve("raw_inventory_i32.bin"));
        InventoryProjection.Input inventory = new InventoryProjection.Input(
            inventoryFlags[0], slice(inventoryFlags, 1, 6),
            slice(inventoryI32, 0, 6), slice(inventoryFlags, 7, 76),
            slice(inventoryI32, 6, 76), slice(inventoryI32, 82, 76),
            slice(inventoryI32, 158, 76), slice(inventoryI32, 234, 76),
            Policy.read(state, "raw_inventory_f32"),
            slice(inventoryFlags, 83, 76)
        );

        boolean[] geometryMask = Policy.readMask(
            state, "geometry_token_mask");
        boolean geometryAvailable = Policy.readMask(
            state, "geometry_available")[0];
        boolean actorValid = Policy.readMask(state, "valid")[0];
        PerceptionFrame.WorldGroups world = new PerceptionFrame.WorldGroups(
            geometryInputFromProjected(
                Policy.read(state, "geometry_token_f32"), geometryMask,
                geometryAvailable, actorValid),
            lightInputFromProjected(
                Policy.read(state, "light_token_f32"),
                Policy.readMask(state, "light_available")[0],
                geometryMask, geometryAvailable, actorValid),
            Policy.read(state, "block_candidate_f32"),
            Policy.readMask(state, "block_candidate_mask"),
            Policy.readMask(state, "block_available")[0],
            recipeInput(state)
        );
        PerceptionFrame.Actions actions = new PerceptionFrame.Actions(
            Policy.readMask(state, "skill_action_mask"), new boolean[3],
            Policy.readMask(state, "guard_action_mask")[0],
            Policy.readMask(state, "jump_action_mask")[0],
            true, true, false, false, new boolean[2]
        );
        return new PerceptionFrame(
            actorValid,
            Policy.readMask(state, "base_target_mask")[0],
            self, target, combatSelf, targetRaw,
            (int) Math.round(in.get("movement_value_bits")),
            (int) Math.round(in.get("movement_available_bits")),
            new WorldProjection.ActorWorldInput(
                in.get("actor_controller_medium_available") != 0.0,
                in.get("actor_controller_in_fluid") != 0.0,
                in.get("actor_submersion_available") != 0.0,
                in.get("actor_feet_submerged") != 0.0,
                in.get("actor_eyes_submerged") != 0.0,
                in.get("actor_drop_available") != 0.0,
                in.get("actor_drop_support_found") != 0.0,
                in.get("actor_drop_height").floatValue(),
                in.get("agent_walk_max_drop_height").floatValue()
            ),
            mechanics, inventory, world, actions
        );
    }

    /**
     * Load the actor-visible structured recipe row when the fixture carries it.
     *
     * <p>Older parity captures predate raw recipe export and are accepted only
     * when their final recipe surface is completely unavailable and zero. A
     * nonzero legacy row cannot be reconstructed from its embedding and fails
     * rather than silently certifying a fabricated inverse.
     */
    private static RecipeCandidateProjection.Input recipeInput(Path state)
        throws IOException {
        if (!Files.exists(state.resolve("raw_recipe_available.bin"))) {
            boolean available = Policy.readMask(state, "recipe_available")[0];
            boolean anyMask = false;
            for (boolean value : Policy.readMask(state, "recipe_mask")) {
                anyMask |= value;
            }
            boolean anyValue = false;
            for (float value : Policy.read(state, "recipe_embedding")) {
                anyValue |= value != 0.0f;
            }
            require(!available && !anyMask && !anyValue,
                state + ": nonzero legacy recipe row has no structured source");
            return RecipeCandidateProjection.Input.unavailable();
        }
        return new RecipeCandidateProjection.Input(
            Policy.readMask(state, "raw_recipe_available")[0],
            Policy.readMask(state, "raw_recipe_candidate_mask"),
            Policy.readMask(state, "raw_recipe_input_mask"),
            readInts(state.resolve("raw_recipe_input_item_id.bin")),
            readInts(state.resolve("raw_recipe_input_resource_type_id.bin")),
            readInts(state.resolve("raw_recipe_input_quantity.bin")),
            Policy.readMask(state, "raw_recipe_input_metadata_required"),
            readInts(state.resolve("raw_recipe_input_metadata_hash.bin")),
            Policy.readMask(state, "raw_recipe_output_mask"),
            readInts(state.resolve("raw_recipe_output_item_id.bin")),
            readInts(state.resolve("raw_recipe_output_quantity.bin")),
            readInts(state.resolve("raw_recipe_output_metadata_hash.bin")),
            Policy.readMask(state, "raw_recipe_requirement_mask"),
            readInts(state.resolve("raw_recipe_requirement_bench_type.bin")),
            readInts(state.resolve("raw_recipe_requirement_bench_id_hash.bin")),
            readInts(state.resolve("raw_recipe_requirement_tier_level.bin")),
            Policy.readMask(state, "raw_recipe_knowledge_required"),
            readInts(state.resolve("raw_recipe_required_memories_level.bin")),
            Policy.read(state, "raw_recipe_time_seconds")
        );
    }

    /**
     * Reconstruct the policy-visible target slot from the same raw ability
     * clocks JAX's Arsenal compatibility projection consumes.
     *
     * <p>The legacy target attack index in {@code inputs.txt} is not the
     * learner-v3 value for an Arsenal opponent. While an ability is active JAX
     * publishes that authored slot; during cooldown it publishes the slot with
     * the largest remaining cooldown fraction. Reading the legacy value here
     * made every stepped production-path fixture report {@code -1} even though
     * its same-tick raw lifecycle row named slot zero.</p>
     */
    private static int targetReportedAbilitySlot(float[] rawAbility) {
        require(rawAbility.length == 868, "raw ability width drifted");
        final int capacity = 16;
        final int targetOffset = capacity;
        final int authoredCooldownOffset = 60;
        final int remainingCooldownOffset = 92;
        final int activeSlotOffset = 124;
        final int authoredOffset = 800;
        int active = Math.round(rawAbility[activeSlotOffset + 1]);
        if (active >= 0 && active < capacity
            && rawAbility[authoredOffset + targetOffset + active] > 0.5f) {
            return active;
        }

        float bestFraction = 0.0f;
        int bestSlot = -1;
        for (int slot = 0; slot < capacity; slot++) {
            if (rawAbility[authoredOffset + targetOffset + slot] <= 0.5f) {
                continue;
            }
            float duration = Math.max(
                rawAbility[authoredCooldownOffset + targetOffset + slot],
                1.0e-6f);
            float fraction = rawAbility[
                remainingCooldownOffset + targetOffset + slot] / duration;
            if (fraction > bestFraction) {
                bestFraction = fraction;
                bestSlot = slot;
            }
        }
        return bestSlot;
    }

    /** Reconstruct one policy-equivalent raw geometry row for composition. */
    private static GeometryProjection.Input geometryInputFromProjected(
        float[] projected,
        boolean[] projectedMask,
        boolean available,
        boolean actorValid
    ) {
        require(projected.length == GeometryProjection.VALUE_COUNT,
            "projected geometry fixture width drifted");
        require(projectedMask.length == GeometryProjection.TOKEN_CAPACITY,
            "projected geometry mask width drifted");
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

    /**
     * Reconstruct raw integer channels for the composition fixture.
     * ProjectionTest independently certifies the non-invertible failure cases;
     * this helper exists only to push a known-good policy row through the live
     * composition boundary.
     */
    private static LightProjection.Input lightInputFromProjected(
        float[] projected,
        boolean available,
        boolean[] geometryMask,
        boolean geometryAvailable,
        boolean actorValid
    ) {
        require(projected.length == LightProjection.VALUE_COUNT,
            "projected light fixture width drifted");
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
        return Arrays.copyOfRange(values, start, start + length);
    }

    private static int[] integers(float[] values, int start, int length) {
        int[] result = new int[length];
        for (int index = 0; index < length; index++) {
            result[index] = Math.round(values[start + index]);
        }
        return result;
    }

    private static int[] slice(int[] values, int start, int length) {
        return Arrays.copyOfRange(values, start, start + length);
    }

    private static boolean[] slice(boolean[] values, int start, int length) {
        return Arrays.copyOfRange(values, start, start + length);
    }

    private static boolean[] booleans(float[] values, int start, int length) {
        boolean[] result = new boolean[length];
        for (int index = 0; index < length; index++) {
            result[index] = values[start + index] != 0.0f;
        }
        return result;
    }

    private static int[] readInts(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        if (bytes.length % Integer.BYTES != 0) {
            throw new IOException("int32 fixture width drift at " + path);
        }
        int[] result = new int[bytes.length / Integer.BYTES];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            .asIntBuffer().get(result);
        return result;
    }

    private static Map<String, Double> readValues(Path path) throws IOException {
        Map<String, Double> values = new HashMap<>();
        for (String line : Files.readAllLines(path)) {
            if (line.isBlank()) {
                continue;
            }
            String[] parts = line.split("\\t", -1);
            values.put(parts[0], Double.parseDouble(parts[1]));
        }
        return values;
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static String firstDifference(float[] actual, float[] expected) {
        for (int index = 0; index < Math.min(actual.length, expected.length); index++) {
            if (!sameValue(actual[index], expected[index])) {
                return index + ": " + actual[index] + " != " + expected[index];
            }
        }
        return "width " + actual.length + " != " + expected.length;
    }

    private static boolean sameValues(float[] actual, float[] expected) {
        if (actual.length != expected.length) {
            return false;
        }
        for (int index = 0; index < actual.length; index++) {
            if (!sameValue(actual[index], expected[index])) {
                return false;
            }
        }
        return true;
    }

    /** Exact IEEE value bits; the observationally identical zero signs agree. */
    private static boolean sameValue(float actual, float expected) {
        return (actual == 0.0f && expected == 0.0f)
            || Float.floatToIntBits(actual) == Float.floatToIntBits(expected);
    }
}
