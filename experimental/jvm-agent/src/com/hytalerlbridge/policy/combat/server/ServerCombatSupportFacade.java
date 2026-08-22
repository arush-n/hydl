package com.hytalerlbridge.policy.combat.server;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.server.core.entity.InteractionManager;
import com.hypixel.hytale.server.core.modules.interaction.InteractionModule;
import com.hypixel.hytale.server.core.modules.interaction.interaction.config.RootInteraction;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.role.Role;
import com.hypixel.hytale.server.npc.interactions.NPCInteractionSimulationHandler;
import com.hypixel.hytale.server.npc.role.support.CombatSupport;
import com.hytalerlbridge.policy.combat.bridge.BridgeCombatFacade;
import com.hypixel.hytale.logger.HytaleLogger;
import com.hytalerlbridge.policy.perception.profile.PerceptionProfile;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * Policy-selected combat through the server's own attack pipeline.
 *
 * <p>This is the fallback used when the bridge's combat facade is absent —
 * which is the deployed state: {@code NativePolicyCombatFacade} is not in the
 * shipped bridge jar, so without this the policy contributes movement only and
 * every attack you see in-game is the role's own AI.
 *
 * <h2>What it does and does not control</h2>
 *
 * <p>The policy chooses <b>which</b> ability; the engine keeps <b>when</b>.
 * That split is deliberate, and it is the reason this class is ~200 lines
 * instead of a reimplementation of Hytale combat. {@code ActionAttack} asks
 * {@link CombatSupport#getNextAttackOverride()} at the moment it has decided to
 * swing ({@code ActionAttack.java:136}); writing the policy's chosen
 * interaction there substitutes our decision for the authored one while
 * cooldowns, charge banks, aiming time, damage, and friendly-fire rules all
 * stay where they belong. The package README is explicit that this mod must not
 * copy those rules, and this respects that.
 *
 * <p>The honest limitation: attack <em>timing</em> stays with the role's AI, so
 * a policy that wants to attack right now waits for the next authored opening.
 * Full timing control needs the bridge facade.
 *
 * <h2>The tag constraint, checked up front</h2>
 *
 * <p>{@code ActionAttack} rejects any override whose id is not in
 * {@code RootInteraction.getAssetMap().getKeysForTag(ATTACK_TAG_INDEX)},
 * logging "using interaction %s that is not tagged as an 'Attack' usable by
 * NPCs" and cancelling the swing. A rejected override is therefore a silent
 * no-op wearing the costume of a working one. So every slot is validated once
 * against that tag set and the result is reported — an unusable slot is named
 * at startup rather than discovered as an attack that never lands.
 */
public final class ServerCombatSupportFacade implements BridgeCombatFacade {

    private static final HytaleLogger LOGGER = HytaleLogger.forEnclosingClass();

    /** How many times the policy's choice reached the engine, for /hydl status. */
    private int overridesApplied;

    private final PerceptionProfile profile;
    /** Slot to authored charge seconds, straight from the profile. */
    private final Map<Integer, Double> chargeSeconds = new LinkedHashMap<>();

    /** Slot to authored interaction id, only for slots that can actually fire. */
    private Map<Integer, String> usableSlots;
    /** Slots the profile declares but the engine will refuse, for reporting. */
    private Set<String> rejectedSlots;

    /** Sentinel for "guard is the standing override"; no ability slot is negative. */
    private static final int GUARD_SLOT = -2;

    /** Last slot written as an override; avoids rewriting it every tick. */
    private int appliedSlot = -1;

    private final String roleName;

    private ServerCombatSupportFacade(PerceptionProfile profile, String roleName) {
        this.profile = profile;
        this.roleName = roleName == null ? "" : roleName;
        for (PerceptionProfile.AbilityBinding binding
                : profile.agentAbilityBindings()) {
            chargeSeconds.put(binding.slot(), binding.requestedChargeSeconds());
        }
    }

    public static ServerCombatSupportFacade create(
        PerceptionProfile profile, String roleName
    ) {
        return new ServerCombatSupportFacade(profile, roleName);
    }

    /**
     * The direction-ish tail of an interaction id, used to pair the profile's
     * player-side interactions with the role's own attack roots.
     *
     * <p>{@code Weapon_Sword_Primary_Swing_Left} and
     * {@code Example_Role_Sword_Swing_Left} are different assets in
     * different spaces — the profile names {@code Interaction}s the player
     * uses, while {@code ActionAttack} only accepts Attack-tagged
     * {@code RootInteraction}s authored per NPC role. Nothing in the data links
     * them, so they are paired on the trailing verb, which is the only shared
     * vocabulary: {@code Swing_Left}, {@code Swing_Right}, {@code Swing_Down}.
     */
    private static String directionKey(String interactionId) {
        String id = interactionId.toLowerCase();
        if (id.endsWith("swing_left")) return "swing_left";
        if (id.endsWith("swing_right")) return "swing_right";
        if (id.endsWith("swing_down")) return "swing_down";
        if (id.endsWith("swing_up")) return "swing_up";
        if (id.endsWith("thrust")) return "thrust";
        return "";
    }

    /**
     * Resolve which slots the engine will accept, once, on first use.
     *
     * <p>Deliberately not done in the constructor. The plugin builds its action
     * runtime during setup, and at that point the asset store is not populated
     * — {@code getKeysForTag} returns an empty set and <em>every</em> slot would
     * be recorded as unusable, permanently. Resolving on first {@link #bind()}
     * instead means the lookup happens once an NPC is actually being claimed,
     * by which time the assets are loaded. This is a tag read, not
     * {@code loadAssets}, so it takes no write lock and is safe on the world
     * thread — unlike the native binding resolution, which is not.
     */
    private void ensureValidated() {
        if (usableSlots != null) {
            return;
        }
        Set<String> attackTagged;
        try {
            attackTagged = RootInteraction.getAssetMap()
                .getKeysForTag(CombatSupport.ATTACK_TAG_INDEX);
        } catch (RuntimeException unavailable) {
            attackTagged = Set.of();
        }
        // Blocks are Attack-tagged too. `CombatSupport` distinguishes the kinds
        // with their own tags ("Attack=Melee", "Attack=Ranged", "Attack=Block"),
        // so use those rather than reading intent out of the asset name.
        Set<String> blocks = keysForTag(CombatSupport.BLOCK_TAG_INDEX);

        // Every offensive root this role actually owns, in a stable order.
        // Scoped to the role: another role's swing is a different animation,
        // reach and damage, and substituting one would be silent.
        java.util.List<String> roleOffensive = new java.util.ArrayList<>();
        for (String key : new java.util.TreeSet<>(attackTagged)) {
            if (!roleName.isEmpty() && key.startsWith(roleName)
                    && !blocks.contains(key)) {
                roleOffensive.add(key);
            }
        }
        roleAttackNames = roleOffensive;

        // Pair on the trailing verb where both sides have one. An empty
        // direction key means "no shared vocabulary", not "matches anything":
        // without that guard every unclassified id collides on "", and measured
        // 2026-08-08 the policy's signature attack paired with
        // Outlander_Marauder_Sword_Block -- an attack executing as a guard,
        // invisible to the policy and unlearnable.
        Map<String, String> roleAttacks = new LinkedHashMap<>();
        for (String key : roleOffensive) {
            String direction = directionKey(key);
            if (!direction.isEmpty()) {
                roleAttacks.putIfAbsent(direction, key);
            }
        }

        Map<Integer, String> usable = new LinkedHashMap<>();
        Set<String> rejected = new LinkedHashSet<>();
        java.util.List<PerceptionProfile.AbilityBinding> unpairedSlots =
            new java.util.ArrayList<>();
        chargeOnly = new LinkedHashSet<>();
        for (PerceptionProfile.AbilityBinding binding
                : profile.agentAbilityBindings()) {
            // Exact match first: if a profile id ever *is* an Attack root, use
            // it as authored rather than remapping it.
            if (attackTagged.contains(binding.interactionId())) {
                usable.put(binding.slot(), binding.interactionId());
                continue;
            }
            String direction = directionKey(binding.interactionId());
            String paired = direction.isEmpty() ? null : roleAttacks.get(direction);
            if (paired != null) {
                usable.put(binding.slot(), paired);
            } else {
                unpairedSlots.add(binding);
            }
        }

        // Whatever the verb match could not place -- the profile's signature
        // and thrust slots -- goes to the role's remaining offensive roots,
        // which is where a role's *special* abilities live. Dropping them
        // instead would silently delete two of the policy's five choices and
        // cap it at basic swings forever.
        //
        // Assigned in sorted order so the mapping is stable across boots: an
        // ability slot that means one attack today and another tomorrow is not
        // something a policy can be evaluated against.
        java.util.List<String> spare = new java.util.ArrayList<>(roleOffensive);
        spare.removeAll(usable.values());
        int next = 0;
        for (PerceptionProfile.AbilityBinding binding : unpairedSlots) {
            if (next < spare.size()) {
                usable.put(binding.slot(), spare.get(next++));
            } else if (!roleOffensive.isEmpty()
                    && binding.requestedChargeSeconds() > 0.0) {
                // Last resort, and only legitimate because charge is now wired:
                // a slot with no distinct root still produces a distinct action
                // when held longer. Thrust IS the primary held 0.700 s rather
                // than 0.117 s, so pointing it at the role's primary root with
                // its authored charge reproduces the real mechanic instead of
                // dropping the slot. Recorded separately so nobody reads it as
                // an exact match.
                usable.put(binding.slot(), roleOffensive.get(0));
                chargeOnly.add(binding.slot() + ":" + binding.interactionId()
                    + "@" + binding.requestedChargeSeconds() + "s");
            } else {
                rejected.add(binding.slot() + ":" + binding.interactionId());
            }
        }

        usableSlots = usable;
        rejectedSlots = rejected;

        // Guard is a real policy head and was being dropped entirely. Unlike
        // the ability slots it needs no remapping: the profile's own guard root
        // (Root_Weapon_Sword_Secondary_Guard) is weapon-scoped rather than
        // role-scoped and is itself Attack-tagged, so the policy's *trained*
        // interaction can be used verbatim -- strictly better fidelity than the
        // verb-matched swings. Only if that is absent do we fall back to
        // whatever Block-tagged root this role owns.
        PerceptionProfile.GuardBinding guard = profile.agentGuardBinding();
        if (guard != null && attackTagged.contains(guard.interactionId())) {
            guardRoot = guard.interactionId();
        } else {
            for (String key : new java.util.TreeSet<>(blocks)) {
                if (!roleName.isEmpty() && key.startsWith(roleName)) {
                    guardRoot = key;
                    break;
                }
            }
        }
    }

    /** Root written while the policy holds guard; empty if it cannot block. */
    private String guardRoot = "";

    /** Slots that reuse a root and are distinguished only by charge time. */
    private Set<String> chargeOnly = new LinkedHashSet<>();

    /** Every offensive root this role owns; empty explains everything. */
    private java.util.List<String> roleAttackNames = java.util.List.of();

    /**
     * Roles ranked by how many offensive roots they own, richest first.
     *
     * <p>Answers the question the slot report raises but cannot settle: the
     * policy has five ability slots, and a role with three plain swings can
     * only ever use three of them. A role with a signature root is what makes
     * slot 4 mean anything. Blocks are excluded via the {@code Attack=Block}
     * tag, so the count is offence only.
     *
     * @param roleNames every spawnable role, from {@code NPCPlugin}
     */
    public static String rolesByAttackCount(
        java.util.List<String> roleNames, int limit
    ) {
        Set<String> attacks = keysForTag(CombatSupport.ATTACK_TAG_INDEX);
        Set<String> blocks = keysForTag(CombatSupport.BLOCK_TAG_INDEX);
        if (attacks.isEmpty()) {
            return "(assets not loaded)";
        }
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (String role : roleNames) {
            int owned = 0;
            for (String key : attacks) {
                if (key.startsWith(role) && !blocks.contains(key)) {
                    owned++;
                }
            }
            if (owned > 0) {
                counts.put(role, owned);
            }
        }
        return counts.entrySet().stream()
            .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
            .limit(limit)
            .map(e -> e.getKey() + "=" + e.getValue())
            .reduce((a, b) -> a + ", " + b)
            .orElse("(no role owns an offensive root)");
    }

    /** Tag lookup that yields an empty set rather than throwing pre-assets. */
    private static Set<String> keysForTag(int tagIndex) {
        try {
            return RootInteraction.getAssetMap().getKeysForTag(tagIndex);
        } catch (RuntimeException unavailable) {
            return Set.of();
        }
    }

    /** One line for the log: what the policy can and cannot fire. */
    /** Overrides actually written; zero means the policy never picked an ability. */
    public int overridesApplied() {
        return overridesApplied;
    }

    public String describe() {
        ensureValidated();
        String report = usableSlots.size() + " of "
            + (usableSlots.size() + rejectedSlots.size())
            + " ability slots usable"
            + (rejectedSlots.isEmpty()
                ? "" : "; not Attack-tagged: " + String.join(", ", rejectedSlots));
        report += " (role " + roleName + " offensive roots: " + roleAttackNames
            + "; guard=" + (guardRoot.isEmpty() ? "none" : guardRoot) + ")";
        if (!chargeOnly.isEmpty()) {
            report += " | charge-differentiated: " + chargeOnly;
        }
        if (!usableSlots.isEmpty()) {
            report += " -> " + usableSlots;
        } else if (roleAttackNames.isEmpty()) {
            // The decisive case, and worth naming: a role with no authored
            // attacks cannot be given one through the override list at all.
            report += " | this role has no authored attacks; the override path"
                + " cannot fire anything for it";
        }
        return report;
    }

    @Override
    public Binding bind() {
        ensureValidated();
        if (usableSlots.isEmpty()) {
            return Binding.rejected(
                "no profile ability is Attack-tagged: " + describe());
        }
        // There is no native handle to hold: the state lives on the NPC's own
        // CombatSupport, reached through the actor on every apply. The record
        // rejects null, so this is a presence token and nothing more.
        return new Binding(true, "", new Execution(this));
    }

    @Override
    public Receipt initialize(
        Execution execution,
        Ref<EntityStore> actor,
        NPCEntity npc,
        Store<EntityStore> store
    ) {
        appliedSlot = -1;
        clearOverrides(npc);
        return available();
    }

    @Override
    public Receipt prepare(
        Execution execution,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        float deltaTime,
        boolean externalRemoteClientActive
    ) {
        // Nothing to synchronise: the engine owns the interaction runtime.
        return available();
    }

    @Override
    public Receipt apply(
        Execution execution,
        Ref<EntityStore> actor,
        NPCEntity npc,
        Store<EntityStore> store,
        float deltaTime,
        boolean firstControlTick,
        boolean attackRequested,
        int abilitySlot,
        boolean guardHeld,
        int dodgeDirection,
        double requestedChargeSeconds,
        boolean externalRootActive,
        boolean externalRemoteClientActive
    ) {
        CombatSupport combat = combatSupport(npc);
        if (combat == null) {
            return Receipt.unavailable("npc has no combat support");
        }

        // Guard takes the override while held, and that is *not* because it
        // excludes attacking -- it does not. In-game you can attack while
        // blocking; doing so produces the weapon's block attack, and no other
        // attack type is reachable until the block is released. Writing the
        // guard root reproduces exactly that: hold it and the NPC blocks, and
        // if the role's AI swings during the hold it swings with the guard
        // root, which is the block attack. Letting an ability slot win here
        // instead would allow a normal swing mid-block, which the game does
        // not permit.
        if (guardHeld && !guardRoot.isEmpty()) {
            if (appliedSlot != GUARD_SLOT) {
                combat.clearAttackOverrides();
                combat.addAttackOverride(guardRoot);
                appliedSlot = GUARD_SLOT;
                overridesApplied++;
                LOGGER.at(java.util.logging.Level.INFO)
                    .atMostEvery(5, java.util.concurrent.TimeUnit.SECONDS)
                    .log("PolicyAgent chose guard -> %s", guardRoot);
            }
            return new ReceiptBuilder().guard(guardRoot).build();
        }

        String interactionId = usableSlots.get(abilitySlot);
        if (abilitySlot < 0 || interactionId == null) {
            // No ability this tick, or one the engine would refuse. Drop any
            // standing override so the role falls back to authored behaviour
            // rather than repeating our last choice indefinitely.
            if (appliedSlot != -1) {
                combat.clearAttackOverrides();
                appliedSlot = -1;
            }
            return new ReceiptBuilder()
                .requested(abilitySlot >= 0)
                .rejected(abilitySlot >= 0
                    ? "slot " + abilitySlot + " is not Attack-tagged" : "")
                .build();
        }

        // Charge, every tick the ability is held. `isCharging` compares elapsed
        // time against this value (NPCInteractionSimulationHandler:21), so it
        // has to be standing when the engine starts the swing, not only on the
        // tick the choice changes. This is what makes slot 3 (Thrust, 0.700 s)
        // a different action from slot 0 (Swing_Left, 0.117 s) even when both
        // resolve to the same authored root -- a held primary, exactly as a
        // player produces a thrust by holding the button.
        requestCharge(actor, store, (float) requestedChargeSeconds);

        if (appliedSlot != abilitySlot) {
            // The only direct evidence that the policy -- not the role's AI --
            // chose this attack. Without it "the NPC swung" is unattributable,
            // which is the exact confusion this whole path exists to resolve.
            // Rate-limited because a decision every 5 ticks would flood the log.
            LOGGER.at(java.util.logging.Level.INFO)
                .atMostEvery(5, java.util.concurrent.TimeUnit.SECONDS)
                .log("PolicyAgent chose ability slot %d -> %s",
                    abilitySlot, interactionId);
            overridesApplied++;
            // Replace rather than append: addAttackOverride cycles the list
            // round-robin (CombatSupport:105-113), so appending would make the
            // NPC rotate through every ability the policy had ever chosen
            // instead of using the one it is choosing now.
            combat.clearAttackOverrides();
            combat.addAttackOverride(interactionId);
            appliedSlot = abilitySlot;
        }
        return new ReceiptBuilder()
            .requested(true)
            .accepted(interactionId)
            .active(combat.isExecutingAttack())
            .build();
    }

    @Override
    public void close(
        Execution execution,
        Ref<EntityStore> actor,
        Store<EntityStore> store,
        boolean externalRemoteClientActive
    ) {
        appliedSlot = -1;
    }

    @Override
    public double requestedChargeSeconds(int abilitySlot) {
        Double charge = chargeSeconds.get(abilitySlot);
        return charge == null ? 0.0 : charge;
    }

    /**
     * Ask the engine to charge the next interaction for {@code seconds}.
     *
     * <p>The same call the engine's own {@code ActionAttack} makes
     * ({@code :228}) — the NPC interaction simulation handler holds one
     * requested charge time and {@code isCharging} returns
     * {@code time < requestedChargeTime}. Without this every ability fires at
     * zero charge, which collapses the profile's five slots into "swing" and
     * makes the policy's charge-differentiated choices unobservable.
     *
     * <p>Failures are swallowed deliberately: a role whose interaction handler
     * is not the NPC one still attacks fine, just uncharged, and taking the
     * whole combat path down over it would be a worse trade.
     */
    private void requestCharge(
        Ref<EntityStore> actor, Store<EntityStore> store, float seconds
    ) {
        if (seconds <= 0.0f) {
            return;
        }
        try {
            InteractionManager manager = store.getComponent(
                actor, InteractionModule.get().getInteractionManagerComponent());
            if (manager == null) {
                return;
            }
            if (manager.getInteractionSimulationHandler()
                    instanceof NPCInteractionSimulationHandler handler) {
                handler.requestChargeTime(seconds);
            }
        } catch (RuntimeException unavailable) {
            LOGGER.at(java.util.logging.Level.WARNING)
                .atMostEvery(1, java.util.concurrent.TimeUnit.MINUTES)
                .log("PolicyAgent could not request charge: %s", unavailable);
        }
    }

    private void clearOverrides(NPCEntity npc) {
        CombatSupport combat = combatSupport(npc);
        if (combat != null) {
            combat.clearAttackOverrides();
        }
    }

    private static CombatSupport combatSupport(NPCEntity npc) {
        if (npc == null) {
            return null;
        }
        Role role = npc.getRole();
        return role == null ? null : role.getCombatSupport();
    }

    private static Receipt available() {
        return new ReceiptBuilder().build();
    }

    /**
     * The {@link Receipt} record takes 35 positional arguments, most of which
     * describe bridge-side lifecycle this facade does not have. Building them
     * by hand at four call sites is how a boolean ends up in the wrong slot.
     */
    private static final class ReceiptBuilder {
        private boolean requested;
        private boolean accepted;
        private String interactionId = "";
        private String rejectReason = "";
        private boolean active;
        private String guardId = "";

        ReceiptBuilder requested(boolean value) {
            this.requested = value;
            return this;
        }

        ReceiptBuilder accepted(String id) {
            this.accepted = true;
            this.interactionId = id;
            return this;
        }

        ReceiptBuilder rejected(String reason) {
            this.rejectReason = reason;
            return this;
        }

        ReceiptBuilder guard(String id) {
            this.guardId = id;
            return this;
        }

        ReceiptBuilder active(boolean value) {
            this.active = value;
            return this;
        }

        Receipt build() {
            return new Receipt(
                true, "",
                // attack: the policy's attack head routes through the ability
                // slot on arsenal profiles, so nothing is reported here.
                false, false, "", "",
                requested, accepted, interactionId, rejectReason,
                // dodge and guard have no server-side equivalent that does not
                // reimplement the rules; absent rather than faked.
                false, false, "", "",
                !guardId.isEmpty(), !guardId.isEmpty(), guardId, "",
                false, false, false, false, -1,
                false, false, false, false, 0,
                false, false,
                active, false, false, false, 0
            );
        }
    }
}
