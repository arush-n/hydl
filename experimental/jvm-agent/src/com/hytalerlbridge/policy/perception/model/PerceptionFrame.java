package com.hytalerlbridge.policy.perception.model;

import com.hytalerlbridge.policy.Projection;
import com.hytalerlbridge.policy.perception.projection.InventoryProjection;
import com.hytalerlbridge.policy.perception.projection.GeometryProjection;
import com.hytalerlbridge.policy.perception.projection.LightProjection;
import com.hytalerlbridge.policy.perception.projection.RecipeCandidateProjection;
import com.hytalerlbridge.policy.perception.projection.WorldProjection;
import com.hytalerlbridge.policy.world.model.WorldActionEvidence;
import com.hytalerlbridge.policy.world.model.BlockCandidateBinding;

/**
 * One actor-safe, same-tick input to the live policy projector.
 *
 * <p>This is deliberately raw rather than an 8,271-float row. Acquisition
 * reads server components into these facts; {@code LivePolicyPerception} owns
 * every normalization and layout operation. Keeping that boundary explicit
 * lets offline fixtures certify the arithmetic without a running server and
 * prevents server API work from smuggling in a second encoder.
 */
public record PerceptionFrame(
    boolean valid,
    boolean targetPresent,
    Projection.SelfState self,
    Projection.TargetState target,
    Projection.CombatSelf combatSelf,
    Projection.TargetRaw targetRaw,
    int movementValueBits,
    int movementAvailableBits,
    WorldProjection.ActorWorldInput actorWorld,
    Mechanics mechanics,
    InventoryProjection.Input inventory,
    WorldGroups world,
    Actions actions
) {
    public PerceptionFrame {
        if (self == null || target == null || combatSelf == null
            || targetRaw == null || actorWorld == null || mechanics == null
            || inventory == null || world == null || actions == null) {
            throw new IllegalArgumentException(
                "a perception frame cannot contain null evidence groups");
        }
    }

    /** Dynamic mechanics; authored scales stay in {@code PerceptionProfile}. */
    public record Mechanics(
        float[] resources,
        boolean[] resourceAvailable,
        Defense defense,
        Status status,
        Ability ability,
        boolean[] dodgeCorridorClear,
        boolean movementEnabled,
        float stamina
    ) {
        public Mechanics {
            require(resources, 14, "resources");
            require(resourceAvailable, 14, "resource availability");
            require(dodgeCorridorClear, 4, "dodge corridor");
            if (defense == null || status == null || ability == null) {
                throw new IllegalArgumentException(
                    "mechanics subgroups cannot be null");
            }
            resources = resources.clone();
            resourceAvailable = resourceAvailable.clone();
            dodgeCorridorClear = dodgeCorridorClear.clone();
        }
        @Override public float[] resources() { return resources.clone(); }
        @Override public boolean[] resourceAvailable() {
            return resourceAvailable.clone();
        }
        @Override public boolean[] dodgeCorridorClear() {
            return dodgeCorridorClear.clone();
        }
    }

    public record Defense(
        boolean[] guardActive,
        boolean[] staminaBroken,
        float[] dodgeInvulnerabilityRemaining,
        float[] appliedVelocity,
        float[] staminaRegenDelay,
        float[] controlImmunity,
        float[] health
    ) {
        public Defense {
            require(guardActive, 2, "guard active");
            require(staminaBroken, 2, "stamina broken");
            require(dodgeInvulnerabilityRemaining, 2, "dodge remaining");
            require(appliedVelocity, 6, "applied velocity");
            require(staminaRegenDelay, 2, "stamina regen delay");
            require(controlImmunity, 2, "control immunity");
            require(health, 2, "health");
            guardActive = guardActive.clone();
            staminaBroken = staminaBroken.clone();
            dodgeInvulnerabilityRemaining =
                dodgeInvulnerabilityRemaining.clone();
            appliedVelocity = appliedVelocity.clone();
            staminaRegenDelay = staminaRegenDelay.clone();
            controlImmunity = controlImmunity.clone();
            health = health.clone();
        }
        @Override public boolean[] guardActive() { return guardActive.clone(); }
        @Override public boolean[] staminaBroken() { return staminaBroken.clone(); }
        @Override public float[] dodgeInvulnerabilityRemaining() {
            return dodgeInvulnerabilityRemaining.clone();
        }
        @Override public float[] appliedVelocity() { return appliedVelocity.clone(); }
        @Override public float[] staminaRegenDelay() {
            return staminaRegenDelay.clone();
        }
        @Override public float[] controlImmunity() { return controlImmunity.clone(); }
        @Override public float[] health() { return health.clone(); }
    }

    public record Status(
        float[] remainingSeconds,
        float[] cycleElapsedSeconds,
        float[] cycleCooldownSeconds,
        float[] damagePerCycle,
        float[] healingPerCycle,
        int[] resourceId,
        float[] resourceDeltaPerCycle,
        float[] speedMultiplier,
        boolean[] active
    ) {
        public Status {
            require(remainingSeconds, 16, "status remaining");
            require(cycleElapsedSeconds, 16, "status cycle elapsed");
            require(cycleCooldownSeconds, 16, "status cycle cooldown");
            require(damagePerCycle, 16, "status damage");
            require(healingPerCycle, 16, "status healing");
            require(resourceId, 16, "status resource id");
            require(resourceDeltaPerCycle, 16, "status resource delta");
            require(speedMultiplier, 16, "status speed");
            require(active, 16, "status active");
            remainingSeconds = remainingSeconds.clone();
            cycleElapsedSeconds = cycleElapsedSeconds.clone();
            cycleCooldownSeconds = cycleCooldownSeconds.clone();
            damagePerCycle = damagePerCycle.clone();
            healingPerCycle = healingPerCycle.clone();
            resourceId = resourceId.clone();
            resourceDeltaPerCycle = resourceDeltaPerCycle.clone();
            speedMultiplier = speedMultiplier.clone();
            active = active.clone();
        }
        @Override public float[] remainingSeconds() { return remainingSeconds.clone(); }
        @Override public float[] cycleElapsedSeconds() {
            return cycleElapsedSeconds.clone();
        }
        @Override public float[] cycleCooldownSeconds() {
            return cycleCooldownSeconds.clone();
        }
        @Override public float[] damagePerCycle() { return damagePerCycle.clone(); }
        @Override public float[] healingPerCycle() { return healingPerCycle.clone(); }
        @Override public int[] resourceId() { return resourceId.clone(); }
        @Override public float[] resourceDeltaPerCycle() {
            return resourceDeltaPerCycle.clone();
        }
        @Override public float[] speedMultiplier() { return speedMultiplier.clone(); }
        @Override public boolean[] active() { return active.clone(); }
    }

    public record Ability(
        float[] remainingCooldownSeconds,
        int[] activeSlot,
        float[] elapsedSeconds,
        boolean[] legal
    ) {
        public Ability {
            require(remainingCooldownSeconds, 32, "ability cooldown");
            require(activeSlot, 2, "active ability slot");
            require(elapsedSeconds, 2, "ability elapsed");
            require(legal, 32, "ability legal");
            remainingCooldownSeconds = remainingCooldownSeconds.clone();
            activeSlot = activeSlot.clone();
            elapsedSeconds = elapsedSeconds.clone();
            legal = legal.clone();
        }
        @Override public float[] remainingCooldownSeconds() {
            return remainingCooldownSeconds.clone();
        }
        @Override public int[] activeSlot() { return activeSlot.clone(); }
        @Override public float[] elapsedSeconds() { return elapsedSeconds.clone(); }
        @Override public boolean[] legal() { return legal.clone(); }
    }

    /** Structured World producer output; unavailable producers use zeros. */
    public record WorldGroups(
        GeometryProjection.Input geometry,
        LightProjection.Input light,
        float[] blockCandidates,
        boolean[] blockMask,
        boolean blockAvailable,
        BlockCandidateBinding[] blockBindings,
        RecipeCandidateProjection.Input recipeCandidates,
        String[] recipeIds
    ) {
        /** Compatibility constructor for sources without privileged bindings. */
        public WorldGroups(
            GeometryProjection.Input geometry,
            LightProjection.Input light,
            float[] blockCandidates,
            boolean[] blockMask,
            boolean blockAvailable,
            RecipeCandidateProjection.Input recipeCandidates
        ) {
            this(
                geometry, light, blockCandidates, blockMask, blockAvailable,
                emptyBlockBindings(),
                recipeCandidates,
                new String[WorldActionEvidence.RECIPE_CAPACITY]
            );
        }

        /** Compatibility constructor for recipe-only privileged bindings. */
        public WorldGroups(
            GeometryProjection.Input geometry,
            LightProjection.Input light,
            float[] blockCandidates,
            boolean[] blockMask,
            boolean blockAvailable,
            RecipeCandidateProjection.Input recipeCandidates,
            String[] recipeIds
        ) {
            this(
                geometry, light, blockCandidates, blockMask, blockAvailable,
                emptyBlockBindings(), recipeCandidates, recipeIds
            );
        }

        public WorldGroups {
            if (geometry == null || light == null || recipeCandidates == null
                || blockBindings == null || recipeIds == null) {
                throw new IllegalArgumentException(
                    "world evidence cannot be null");
            }
            require(blockCandidates, 16 * 28, "block candidates");
            require(blockMask, 16, "block mask");
            if (blockBindings.length != WorldActionEvidence.BLOCK_CAPACITY) {
                throw new IllegalArgumentException(
                    "block identity width drift: " + blockBindings.length);
            }
            if (recipeIds.length != WorldActionEvidence.RECIPE_CAPACITY) {
                throw new IllegalArgumentException(
                    "recipe identity width drift: " + recipeIds.length);
            }
            blockCandidates = blockCandidates.clone();
            blockMask = blockMask.clone();
            blockBindings = blockBindings.clone();
            recipeIds = recipeIds.clone();
        }
        @Override public float[] blockCandidates() { return blockCandidates.clone(); }
        @Override public boolean[] blockMask() { return blockMask.clone(); }
        @Override public BlockCandidateBinding[] blockBindings() {
            return blockBindings.clone();
        }
        @Override public String[] recipeIds() { return recipeIds.clone(); }

        public static WorldGroups unavailable() {
            return new WorldGroups(
                GeometryProjection.Input.unavailable(false),
                LightProjection.Input.unavailable(
                    new boolean[LightProjection.TOKEN_CAPACITY], false, false),
                new float[16 * 28], new boolean[16], false,
                emptyBlockBindings(),
                RecipeCandidateProjection.Input.unavailable(),
                new String[WorldActionEvidence.RECIPE_CAPACITY]
            );
        }

        private static BlockCandidateBinding[] emptyBlockBindings() {
            BlockCandidateBinding[] result = new BlockCandidateBinding[
                WorldActionEvidence.BLOCK_CAPACITY];
            java.util.Arrays.fill(result, BlockCandidateBinding.empty());
            return result;
        }
    }

    /** Current server/sink capability and legality inputs. */
    public record Actions(
        boolean[] skill,
        boolean[] door,
        boolean guard,
        boolean jump,
        boolean movement,
        boolean look,
        boolean use,
        boolean craft,
        boolean[] blockTrigger
    ) {
        public Actions {
            require(skill, 9, "skill actions");
            require(door, 3, "door actions");
            require(blockTrigger, 2, "block trigger");
            skill = skill.clone();
            door = door.clone();
            blockTrigger = blockTrigger.clone();
        }
        @Override public boolean[] skill() { return skill.clone(); }
        @Override public boolean[] door() { return door.clone(); }
        @Override public boolean[] blockTrigger() { return blockTrigger.clone(); }
    }

    private static void require(float[] values, int expected, String name) {
        if (values.length != expected) {
            throw new IllegalArgumentException(
                name + " width drift: " + values.length + " != " + expected);
        }
    }

    private static void require(int[] values, int expected, String name) {
        if (values.length != expected) {
            throw new IllegalArgumentException(
                name + " width drift: " + values.length + " != " + expected);
        }
    }

    private static void require(boolean[] values, int expected, String name) {
        if (values.length != expected) {
            throw new IllegalArgumentException(
                name + " width drift: " + values.length + " != " + expected);
        }
    }
}
