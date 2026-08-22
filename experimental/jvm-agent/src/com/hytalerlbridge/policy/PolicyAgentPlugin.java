package com.hytalerlbridge.policy;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.math.util.ChunkUtil;
import com.hypixel.hytale.math.vector.Rotation3f;
import com.hypixel.hytale.server.core.plugin.JavaPlugin;
import com.hypixel.hytale.server.core.plugin.JavaPluginInit;
import com.hypixel.hytale.server.core.universe.PlayerRef;
import com.hypixel.hytale.server.core.universe.Universe;
import com.hypixel.hytale.server.core.universe.world.World;
import com.hypixel.hytale.server.core.universe.world.chunk.ChunkFlag;
import com.hypixel.hytale.server.core.universe.world.chunk.WorldChunk;
import com.hypixel.hytale.server.core.universe.world.storage.ChunkStore;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.NPCPlugin;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hytalerlbridge.policy.perception.LivePolicyPerception;
import com.hytalerlbridge.policy.perception.acquisition.PublicServerEvidenceSource;
import com.hytalerlbridge.policy.perception.acquisition.bridge.ReflectiveBridgeCombatLifecycleEvidenceSource;
import com.hytalerlbridge.policy.perception.acquisition.crafting.ServerRecipeCandidateReader;
import com.hytalerlbridge.policy.perception.acquisition.trace.JsonLinesLifecycleTraceSink;
import com.hytalerlbridge.policy.perception.acquisition.trace.LifecycleTraceSink;
import com.hytalerlbridge.policy.perception.profile.PerceptionProfile;
import com.hytalerlbridge.policy.combat.bridge.BridgeCombatFacade;
import com.hytalerlbridge.policy.combat.bridge.ReflectiveBridgeCombatFacade;
import com.hytalerlbridge.policy.combat.server.ServerCombatSupportFacade;
import com.hytalerlbridge.policy.combat.runtime.CombatActionSink;
import com.hytalerlbridge.policy.combat.runtime.PolicyCombatPrepareSystem;
import com.hytalerlbridge.policy.diagnostics.action.DeterministicLiveActionControl;
import com.hytalerlbridge.policy.world.bridge.BridgeWorldVerbFacade;
import com.hytalerlbridge.policy.world.bridge.ReflectiveBridgeWorldVerbFacade;
import com.hytalerlbridge.policy.world.runtime.PolicyWorldAnchorRegistry;
import com.hytalerlbridge.policy.world.runtime.PolicyWorldVerbPrepareSystem;
import com.hytalerlbridge.policy.world.runtime.WorldVerbActionSink;
import com.hytalerlbridge.policy.testing.HeadlessCombatFixture;
import com.hytalerlbridge.policy.testing.HeadlessCombatFixtureSpec;
import com.hytalerlbridge.policy.testing.placement.ServerCombatFixturePlacement;
import it.unimi.dsi.fastutil.Pair;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.joml.Vector3d;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import com.hytalerlbridge.policy.commands.HydlCommand;
import com.hytalerlbridge.policy.bundle.BridgeTargetAttestation;

/**
 * A standalone mod that drives NPCs from a bundle-attested JAX policy.
 *
 * <p>Compile- and link-independent from {@code HytaleRLBridge}. It registers
 * its own component and systems and compiles against the public server jar
 * alone. Live checkpoint perception resolves a small versioned lifecycle
 * facade from the deployed bridge through Hytale's plugin classloader; the
 * older frozen-observation harness remains bridge-independent. Removing this
 * mod still leaves the bridge's RL socket and native sessions untouched.
 *
 * <p>Configuration is by directory layout next to the jar, so there is no config
 * format to get wrong:
 *
 * <pre>
 * mods/HytalePolicyAgent-0.1.0.jar
 * mods/policy-agent/                 &lt;- weights and the captured observation
 *     encoder_input_kernel.bin, ... , actor_bias.bin
 *     observation.bin, action_mask.bin
 *     agent.json, contract.txt         &lt;- required v4 identity/provenance
 *     profile/                       &lt;- live evidence profile; if present,
 *                                      the matching bridge facade is required
 *     role.txt                       &lt;- required by v4; exact trained role
 *     decision.txt                   &lt;- optional; ticks between decisions
 *     nojump                         &lt;- optional; presence suppresses jumping
 *     live-test-control.txt          &lt;- optional one-shot diagnostic control
 *     bridge-target.txt              &lt;- optional exact bridge JAR/ABI tuple;
 *                                      attested before weight load/actor claim
 * </pre>
 *
 * <p>If the weights directory is missing the plugin logs and stays inert rather
 * than throwing -- a mod that fails to load takes the whole server down with it,
 * and an absent policy should not cost anyone their world.
 */
public final class PolicyAgentPlugin extends JavaPlugin {

    private static final String DATA_DIRECTORY = "policy-agent";
    private PolicyControlSystem control;
    private PolicyAttachmentSystem attachment;
    private SteeringActionSink steeringSink;
    private PolicyActionSink sink;
    private WorldVerbActionSink worldVerbSink;
    private CombatActionSink combatSink;
    private PolicyWorldAnchorRegistry worldAnchors;
    private PolicyRuntime runtime;
    private PolicyPerception perception;
    private int captureNonZero = -1;
    private int captureLegal = -1;
    private String perceptionDescription = "unconfigured";
    private String role;
    private int decisionPeriod = 1;
    /** Profile whose ability bindings still need resolving off the world thread. */
    private volatile PerceptionProfile warmupProfile;
    /** Persistent profile identity used by explicit live fixture setup. */
    private volatile PerceptionProfile liveProfile;
    /** Set once the bindings resolve; until then the world thread can deadlock. */
    private volatile boolean bindingsWarmed;
    /** Non-null only when the bridge combat facade was absent and we fell back. */
    private volatile ServerCombatSupportFacade serverCombatFacade;
    /** Its slot report is only meaningful once assets exist; logged once. */
    private volatile boolean serverCombatReported;
    private DeterministicLiveActionControl liveActionControl =
        DeterministicLiveActionControl.disabled();
    private boolean jumpAllowed = true;
    private String inertReason;
    private Path dataDirectory;
    private volatile World world;
    private volatile PlayerRef tickAnchor;
    private volatile long anchorChunkKey;
    private volatile boolean haveAnchorChunk;
    private volatile long secondaryAnchorChunkKey;
    private volatile boolean haveSecondaryAnchorChunk;
    /** Strong refs whose WorldChunks also own one keep-loaded count each. */
    private final Map<Long, Ref<ChunkStore>> pinnedFixtureChunks =
        new ConcurrentHashMap<>();
    /** True when an explicit spawn.txt or combat-fixture.txt was installed. */
    private volatile boolean testSubjectConfigured;
    /** Prevents the heartbeat from scheduling the same fixture more than once. */
    private volatile boolean testSubjectSpawnPending;
    /** Set only after the fixture has completed on the world thread. */
    private volatile boolean testSubjectReady;
    /** Parse/setup failures are terminal; a missing startup world is not. */
    private volatile boolean testSubjectFailed;

    public PolicyAgentPlugin(JavaPluginInit init) {
        super(init);
    }

    @Override
    protected void setup() {
        Path data = getFile().getParent().resolve(DATA_DIRECTORY);
        if (!Files.isDirectory(data)) {
            inertReason = "no weights directory at " + data;
            getLogger().atWarning().log("PolicyAgent inert: " + inertReason);
            return;
        }

        dataDirectory = data;
        try {
            // Widths alone do not identify weights: all eleven raw tensors can
            // be replaced with same-shaped files and still load. AgentBundle
            // therefore recomputes the canonical tensor-content digest in
            // Java before the raw tensors are decoded or any control system is
            // registered. Old/no-contract bundles fail closed and must be
            // re-exported.
            AgentBundle declared = AgentBundle.read(data);
            java.util.List<String> mismatches = new java.util.ArrayList<>(
                declared.problems(null));
            if (mismatches.isEmpty()
                    && Files.isRegularFile(data.resolve(
                        BridgeTargetAttestation.FILE_NAME))) {
                // Resolve and hash the independently packaged bridge before
                // policy weights are loaded or any actor-control system exists.
                mismatches.addAll(
                    BridgeTargetAttestation.runtimeProblems(data));
            }
            if (mismatches.isEmpty()) {
                runtime = PolicyRuntime.load(data);
                // Recompute after loading as a narrow TOCTOU guard and compare
                // declared dimensions with the tensors Policy actually read.
                mismatches.addAll(declared.problems(runtime));
            }
            if (!mismatches.isEmpty()) {
                inertReason = "bundle does not match this build -- "
                    + String.join("; ", mismatches);
                getLogger().atSevere().log("PolicyAgent inert: " + inertReason);
                runtime = null;
                return;
            }
            getLogger().atInfo().log("PolicyAgent bundle: " + declared.describe());

            ActionRuntime actions = configurePerception(data);
            liveActionControl = DeterministicLiveActionControl.load(
                data.resolve(DeterministicLiveActionControl.FILE_NAME));

            role = readRole(data);
            // Read out of the bridge's own CombatRuleset.defaultRules() by
            // executing it (2026-08-04), not guessed: steeringRelativeTurnSpeed
            // 1.5, jumpVelocityGravityFloor 10.0, jumpHeightParameter 1.3. This
            // mod does not depend on the bridge, so they are inlined here;
            // re-read them if HytaleCombatAssets.RULESET changes.
            // `nojump` in the data directory suppresses the jump head; see
            // SteeringActionSink for why that is a diagnostic worth having.
            boolean allowJump = !Files.exists(data.resolve("nojump"));
            boolean combatAvailable = actions != null
                && actions.combatFacade() != null;
            steeringSink = new SteeringActionSink(
                1.5, 10.0, 1.3, allowJump, combatAvailable);
            sink = steeringSink;
            worldAnchors = new PolicyWorldAnchorRegistry();
            if (actions != null && actions.worldFacade() != null) {
                worldVerbSink = new WorldVerbActionSink(
                    steeringSink,
                    actions.worldFacade(),
                    actions.recipes(),
                    worldAnchors
                );
                sink = worldVerbSink;
                getLogger().atInfo().log(
                    "PolicyAgent Fieldcraft executor armed through bridge facade");
            }
            if (combatAvailable) {
                combatSink = new CombatActionSink(
                    sink,
                    actions.combatFacade(),
                    liveActionControl,
                    liveActionControl.dodgeCooldownStateSource()
                );
                sink = combatSink;
                getLogger().atInfo().log(
                    "PolicyAgent combat executor armed through bridge facade");
            }
            requireLiveActionDependencies(
                liveActionControl.mode(),
                combatSink != null && serverCombatFacade == null,
                worldVerbSink != null
            );
            if (liveActionControl.mode()
                    != DeterministicLiveActionControl.Mode.DISABLED) {
                getLogger().atWarning().log(
                    "PolicyAgent deterministic live diagnostic enabled: "
                        + liveActionControl.describe());
            }
            if (!allowJump) {
                getLogger().atWarning().log("PolicyAgent: jump suppressed (nojump present)");
            }

            var markerType = getEntityStoreRegistry().registerComponent(
                PolicyAgentMarker.class, PolicyAgentMarker::new);
            attachment = new PolicyAttachmentSystem(
                markerType, runtime, role,
                Files.exists(data.resolve("takeover")),
                this::readyToClaimActors);
            int decisionPeriod = readDecisionPeriod(data);
            control = new PolicyControlSystem(
                markerType,
                perception,
                sink,
                decisionPeriod,
                liveActionControl
            );
            if (decisionPeriod == 1) {
                getLogger().atWarning().log(
                    "PolicyAgent: deciding every tick (33 ms at 30 TPS). Correct "
                        + "for training, faster than any person for an inference "
                        + "comparison -- write decision.txt with 6-8 to throttle.");
            } else {
                getLogger().atInfo().log(String.format(
                    "PolicyAgent: decision period %d ticks (%.0f ms at 30 TPS)",
                    decisionPeriod, decisionPeriod / 30.0 * 1000.0));
            }
            getEntityStoreRegistry().registerSystem(attachment);
            if (worldVerbSink != null) {
                getEntityStoreRegistry().registerSystem(
                    new PolicyWorldVerbPrepareSystem(
                        markerType, worldVerbSink));
            }
            if (combatSink != null) {
                getEntityStoreRegistry().registerSystem(
                    new PolicyCombatPrepareSystem(markerType, combatSink));
            }
            getEntityStoreRegistry().registerSystem(control);

            this.decisionPeriod = decisionPeriod;
            this.jumpAllowed = allowJump;

            // `/hydl`, not a subcommand of `/npc`: that root belongs to
            // Hytale's own NPC plugin and a future update is free to add
            // whatever it likes there. Registered only once the policy has
            // armed, so the commands can assume a live runtime.
            getCommandRegistry().registerCommand(new HydlCommand(this));

            getLogger().atInfo().log(String.format(
                "PolicyAgent armed: role=%s observation=%d encoder=%d "
                    + "recurrent=%d actions=%d "
                    + "perception=%s",
                role, runtime.observationSize(), runtime.encoderSize(),
                runtime.recurrentSize(), runtime.actionSize(),
                perceptionDescription));
            getLogger().atInfo().log(
                "PolicyAgent commands: /hydl spawn, /hydl status");
        } catch (Exception exception) {
            // Never take the server down over this mod.
            inertReason = exception.toString();
            getLogger().atSevere().log(
                "PolicyAgent failed to arm: " + exception);
        }
    }

    // -- read-only surface for the /hydl commands -------------------------
    //
    // Accessors rather than handing the systems out: a command that could
    // reach into PolicyControlSystem could change decision state from a
    // command thread while the tick thread is reading it.

    /** Whether a policy actually loaded; everything else is meaningless if not. */
    public boolean isArmed() {
        return runtime != null && control != null;
    }

    /** Why the plugin is inert, in the words of whatever refused to arm. */
    public String inertReason() {
        return inertReason == null ? "no weights directory" : inertReason;
    }

    /** The exact {@code Role.getRoleName()} the policy attaches to. */
    public String controlledRole() {
        return role;
    }

    /** How many NPCs the attachment system has claimed so far. */
    public int claimedCount() {
        return attachment == null ? 0 : attachment.claimed();
    }

    /** Roles seen but not claimed -- the tell for a role-name mismatch. */
    public Map<String, Integer> unclaimedRoles() {
        return attachment == null ? Map.of() : attachment.unclaimedRoles();
    }

    public int decisionPeriod() {
        return decisionPeriod;
    }

    public boolean jumpAllowed() {
        return jumpAllowed;
    }

    public int observationSize() {
        return runtime == null ? 0 : runtime.observationSize();
    }

    public int actionSize() {
        return runtime == null ? 0 : runtime.actionSize();
    }

    /**
     * How many observation columns were non-zero at the capture probe, and how
     * many actions were legal.
     *
     * <p>This is the "does it actually comprehend the world" number. The
     * contract width (8271) says how much the policy *can* read; this says how
     * much carried signal when it last looked. A wide contract with a handful
     * of non-zero columns is an agent reading mostly padding -- which looks
     * identical to a working agent from every other diagnostic.
     */
    public int captureNonZeroColumns() {
        return captureNonZero;
    }

    public int captureLegalActions() {
        return captureLegal;
    }

    /** Human-readable perception wiring, e.g. "live profile=<sha>". */
    public String perceptionDescription() {
        return perceptionDescription;
    }

    /** Read-only state of the opt-in one-shot live action diagnostic. */
    public String liveActionControlDescription() {
        return liveActionControl.describe();
    }

    /** Snapshot of bridge-combat admission/lifecycle counters for `/hydl status`. */
    public long[] liveCombatCounters() {
        return combatSink == null ? new long[0] : combatSink.counters();
    }

    /** Snapshot of World-verb admission/lifecycle counters for `/hydl status`. */
    public long[] liveWorldVerbCounters() {
        return worldVerbSink == null ? new long[0] : worldVerbSink.counters();
    }

    @Override
    protected void start() {
        if (control == null) {
            return;
        }
        getLogger().atInfo().log("PolicyAgent running");
        Path data = getFile().getParent().resolve(DATA_DIRECTORY);
        testSubjectConfigured = Files.isRegularFile(
            data.resolve("combat-fixture.txt"))
            || Files.isRegularFile(data.resolve("spawn.txt"));
        startHeartbeat();
    }

    /**
     * Keep explicit fixture chunks active once per second and report world and
     * policy ticks side by side every five refreshes.
     *
     * <p>These two numbers separate the only two explanations for a hook that
     * goes quiet. If the world tick climbs while policy ticks stall, the
     * systems are fine and the entities have stopped matching -- unloaded,
     * despawned, or moved out of a live chunk. If the world tick is also flat,
     * the world itself went idle and no amount of hook correctness matters.
     */
    private void startHeartbeat() {
        Thread thread = new Thread(() -> {
            long previousWorld = -1;
            long previousPolicy = -1;
            int refreshesUntilReport = 5;
            while (!Thread.currentThread().isInterrupted()) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    return;
                }
                // Resolve the native ability bindings as soon as the assets
                // exist. This runs before the first heartbeat report and long
                // before anyone can type /hydl spawn, and it is deliberately
                // on this thread rather than the world thread -- doing it
                // there parks the world permanently on an asset-store lock it
                // is itself blocking.
                PerceptionProfile pending = warmupProfile;
                if (!bindingsWarmed && pending != null) {
                    NativeBindingWarmup.Result warmup =
                        NativeBindingWarmup.warm(pending);
                    CombatActionSink.WarmupResult combatWarmup =
                        combatSink == null
                            ? new CombatActionSink.WarmupResult(
                                true, "combat facade not armed")
                            : combatSink.warmupBindingAssets();
                    if (warmup.warmed() && combatWarmup.warmed()) {
                        bindingsWarmed = true;
                        warmupProfile = null;
                        getLogger().atInfo().log(
                            "PolicyAgent native bindings warmed: " + warmup
                                + "; " + combatWarmup.detail());
                    } else {
                        // Expected on early passes: the asset store is still
                        // filling. Reported anyway, because a permanent
                        // failure here is the difference between a working
                        // agent and a hung server.
                        getLogger().atWarning().log(
                            "PolicyAgent native bindings not warmed yet ("
                                + warmup + "; combat="
                                + combatWarmup.detail() + "); retrying");
                    }
                }

                // Which abilities the engine will actually accept as attack
                // overrides. Reported here rather than at setup because the
                // Attack tag set is empty until the assets load, and a slot
                // the engine refuses is a swing that never happens with only
                // a rate-limited warning to show for it.
                ServerCombatSupportFacade serverCombat = serverCombatFacade;
                if (!serverCombatReported && serverCombat != null) {
                    serverCombatReported = true;
                    getLogger().atInfo().log(
                        "PolicyAgent server combat: " + serverCombat.describe());
                }

                // Plugin start races Universe startup by a few milliseconds:
                // the policy can be fully armed before getDefaultWorld()
                // becomes non-null.  A one-shot fixture request therefore
                // turns a healthy headless gate into an empty run.  Retry only
                // that transient condition; parse, profile and spawn failures
                // remain terminal and visible rather than looping forever.
                if (testSubjectConfigured
                    && !testSubjectReady
                    && !testSubjectSpawnPending
                    && !testSubjectFailed) {
                    spawnTestSubject();
                }

                World current = world;
                if (current == null || control == null) {
                    continue;
                }
                // The packet anchor keeps the *world* at 30 TPS, but its
                // ChunkTracker has no Player or TransformComponent and cannot
                // make a fixture chunk HOT. A keep-loaded chunk is retained
                // yet still becomes non-ticking when its active timer expires.
                // Refresh once per second. ChunkUnloadingSystem polls every
                // 0.5 seconds and can consume up to three active-timer units
                // per poll under memory pressure, so a five-second refresh is
                // not safe even though the nominal threshold is 7.5 seconds.
                if (haveAnchorChunk) {
                    current.execute(() -> {
                        // Bounded safe-placement can retain more than the two
                        // final actor chunks. Refresh every chunk whose
                        // keep-loaded count this plugin owns so a valid Dodge
                        // corridor cannot cross into a deactivated neighbor.
                        for (long chunkKey : pinnedFixtureChunks.keySet()) {
                            refreshFixtureChunk(current, chunkKey);
                        }
                    });
                }
                refreshesUntilReport--;
                if (refreshesUntilReport > 0) continue;
                refreshesUntilReport = 5;
                long worldTick = current.getTick();
                long[] counters = control.counters();
                getLogger().atInfo().log(String.format(
                    "PolicyAgent heartbeat: worldTick=%d (+%d) policyTicks=%d (+%d) "
                        + "claimed=%d skipped=%d illegal=%d idleWorld=%b",
                    worldTick, previousWorld < 0 ? 0 : worldTick - previousWorld,
                    counters[0], previousPolicy < 0 ? 0 : counters[0] - previousPolicy,
                    attachment.claimed(), counters[1], counters[2],
                    !current.isTicking()));

                // `PolicyControlSystem` has tracked per-NPC path and
                // displacement all along and nothing ever printed it, so a run
                // could report thousands of clean policy ticks while every NPC
                // stood still -- which is indistinguishable, from the log
                // alone, from one that is working. Path and displacement are
                // both reported because they answer different questions: path
                // near zero means it never moved, while path >> displacement
                // means it moved but went nowhere (circling, or oscillating
                // against terrain).
                PolicyAgentMarker witness = control.witness();
                if (witness != null && witness.havePosition()) {
                    double[] delta = witness.displacement();
                    double[] at = witness.position();
                    getLogger().atInfo().log(String.format(
                        "PolicyAgent movement: samples=%d path=%.2f horiz=%.2f "
                            + "disp=%.2f d=(%+.2f,%+.2f,%+.2f) at=(%.1f,%.1f,%.1f)",
                        witness.positionSamples(), witness.pathLength(),
                        witness.horizontalPathLength(),
                        witness.horizontalDisplacement(),
                        delta[0], delta[1], delta[2], at[0], at[1], at[2]));
                }

                publishMetrics(
                    worldTick,
                    current.isTicking(),
                    previousWorld < 0 ? 0.0 : (worldTick - previousWorld) / 5.0,
                    previousPolicy < 0 ? 0.0 : (counters[0] - previousPolicy) / 5.0);

                previousWorld = worldTick;
                previousPolicy = counters[0];
            }
        }, "PolicyAgent-heartbeat");
        thread.setDaemon(true);
        thread.start();
    }

    /**
     * Optional test harness: spawn one NPC so there is something to control.
     *
     * <p>A server with no players connected never spawns NPCs -- no chunks are
     * active -- so an otherwise-correct hook has nothing to claim and looks
     * identical to a broken one. Enabled only by writing {@code spawn.txt} in
     * the data directory as {@code Role x y z}; absent, this does nothing.
     */
    private void spawnTestSubject() {
        Path data = getFile().getParent().resolve(DATA_DIRECTORY);
        Path combatFixture = data.resolve("combat-fixture.txt");
        Path file = data.resolve("spawn.txt");
        testSubjectConfigured = Files.isRegularFile(combatFixture)
            || Files.isRegularFile(file);
        if (!testSubjectConfigured
            || testSubjectReady
            || testSubjectSpawnPending
            || testSubjectFailed) {
            return;
        }
        if (Files.isRegularFile(combatFixture)) {
            spawnCombatTestSubjects(combatFixture);
            return;
        }
        try {
            String[] parts = Files.readString(file).trim().split("\\s+");
            if (parts.length != 4) {
                getLogger().atWarning().log(
                    "PolicyAgent spawn.txt must be 'Role x y z', got: " + parts.length
                        + " fields");
                return;
            }
            String role = parts[0];
            double x = Double.parseDouble(parts[1]);
            double y = Double.parseDouble(parts[2]);
            double z = Double.parseDouble(parts[3]);

            World world = Universe.get().getDefaultWorld();
            if (!fixtureWorldReady(world)) {
                getLogger().atInfo().log(
                    "PolicyAgent: test-subject spawn deferred until the default world is ready");
                return;
            }
            long key = ChunkUtil.indexChunkFromBlock((int) x, (int) z);
            pinFixtureChunks(world, key);
            testSubjectSpawnPending = true;
            world.execute(() -> {
                try {
                    // A spawn into an unloaded chunk yields exactly one tick and
                    // then "Entity has moved into a chunk that isn't currently
                    // loaded" -- with no player there is nothing keeping the
                    // chunk resident. Pull it in first.
                    anchorChunkKey = key;
                    haveAnchorChunk = true;
                    Object chunk = world.getChunk(key);
                    getLogger().atInfo().log(String.format(
                        "PolicyAgent chunk (%d,%d) key=%d loaded=%b",
                        ChunkUtil.chunkCoordinate((int) x),
                        ChunkUtil.chunkCoordinate((int) z),
                        key, chunk != null));

                    Store<EntityStore> store = world.getEntityStore().getStore();
                    Pair<Ref<EntityStore>, ?> spawned = NPCPlugin.get().spawnNPC(
                        store, role, null,
                        new Vector3d(x, y, z),
                        new Rotation3f(0.0f, 0.0f, 0.0f));
                    if (spawned == null || spawned.left() == null) {
                        getLogger().atWarning().log(
                            "PolicyAgent: server rejected spawn of " + role);
                        return;
                    }
                    NPCEntity npc = store.getComponent(
                        spawned.left(), NPCEntity.getComponentType());
                    if (npc != null) {
                        // Otherwise the engine reclaims it before it is useful.
                        npc.setDespawnTime(Float.POSITIVE_INFINITY);
                    }
                    // A world with no players may not be ticking at all, in
                    // which case a correct hook still never runs. Record it
                    // rather than guessing later from an empty log.
                    boolean ticking = world.isTicking();
                    if (!ticking) {
                        world.setTicking(true);
                    }
                    // isTicking() being true is not enough: with no tracked
                    // PlayerRef the world is *idle* and its ECS systems run at
                    // whatever rate the unlimited thread happens to reach,
                    // which in practice is once. Anchor it.
                    tickAnchor = worldAnchors.context(world).packetAnchor();
                    getLogger().atInfo().log(String.format(
                        "PolicyAgent spawned test subject %s at (%.1f, %.1f, %.1f); "
                            + "world ticking=%b (forced=%b) tick=%d tickAnchor=%b",
                        role, x, y, z, ticking, !ticking, world.getTick(),
                        tickAnchor != null));
                    this.world = world;
                    testSubjectReady = true;
                } catch (Exception exception) {
                    testSubjectFailed = true;
                    getLogger().atWarning().log("PolicyAgent spawn failed: " + exception);
                } finally {
                    testSubjectSpawnPending = false;
                }
            });
        } catch (Exception exception) {
            testSubjectFailed = true;
            getLogger().atWarning().log("PolicyAgent could not read spawn.txt: " + exception);
        }
    }

    /**
     * Refuse diagnostics whose production sink cannot attribute the edge.
     *
     * <p>Package-visible for the self-contained deployment gate. In
     * particular, the public-server combat fallback is not "full combat": it
     * installs an override but leaves the swing decision to role AI.</p>
     */
    static void requireLiveActionDependencies(
        DeterministicLiveActionControl.Mode mode,
        boolean fullCombatAvailable,
        boolean worldVerbAvailable
    ) {
        if (mode == null) {
            throw new IllegalArgumentException(
                "live action diagnostic mode cannot be null");
        }
        if (mode == DeterministicLiveActionControl.Mode.ATTACK_ONCE
                && !fullCombatAvailable) {
            throw new IllegalStateException(
                "attack_once requires the bridge combat facade; the "
                    + "server fallback only selects an attack override "
                    + "and cannot issue or certify the requested edge");
        }
        if (mode == DeterministicLiveActionControl.Mode.DODGE_COOLDOWN
                && !fullCombatAvailable) {
            throw new IllegalStateException(
                "dodge_cooldown requires the bridge combat facade; the "
                    + "server fallback exposes no Dodge receipt lifecycle");
        }
        if (mode == DeterministicLiveActionControl.Mode.WORLD_ONCE
                && !worldVerbAvailable) {
            throw new IllegalStateException(
                "world_once requires the bridge World-verb facade");
        }
    }

    /**
     * Explicit playerless two-actor gate for checkpoint-selected combat.
     *
     * <p>The file supplies only roles and placement. Held items come from the
     * live checkpoint profile, so a stale iron-sword policy cannot be made to
     * look healthy by equipping an unrelated weapon. This path is test-only
     * and cannot run unless {@code combat-fixture.txt} is deliberately
     * installed next to the checkpoint.</p>
     */
    private void spawnCombatTestSubjects(Path file) {
        try {
            HeadlessCombatFixtureSpec spec = HeadlessCombatFixtureSpec.parse(
                Files.readString(file)
            );
            PerceptionProfile profile = liveProfile;
            if (profile == null) {
                throw new IllegalStateException(
                    "combat-fixture.txt requires a live checkpoint profile"
                );
            }
            if (!spec.agentRole().equals(role)) {
                throw new IllegalStateException(
                    "combat fixture agent role " + spec.agentRole()
                        + " does not match controlled role " + role
                );
            }
            String agentItem = profile.agentItemId();
            String targetItem = profile.targetItemId();
            World current = Universe.get().getDefaultWorld();
            if (!fixtureWorldReady(current)) {
                getLogger().atInfo().log(
                    "PolicyAgent: combat fixture deferred until the default world is ready");
                return;
            }
            if (profile.dodgeMotion() == null) {
                throw new IllegalStateException(
                    "combat fixture profile has no Dodge motion contract");
            }
            Vector3d preview = ServerCombatFixturePlacement.previewOrigin(
                current, spec);
            long[] fixtureChunks =
                ServerCombatFixturePlacement.requiredChunkKeys(
                    preview, spec, profile.dodgeMotion());
            pinFixtureChunks(current, fixtureChunks);
            testSubjectSpawnPending = true;
            current.execute(() -> {
                try {
                    Store<EntityStore> store = current.getEntityStore().getStore();
                    HeadlessCombatFixture.Result result =
                        HeadlessCombatFixture.spawn(
                            current,
                            store,
                            spec,
                            agentItem,
                            targetItem,
                            profile.dodgeMotion()
                        );
                    anchorChunkKey = result.agentChunkKey();
                    haveAnchorChunk = true;
                    secondaryAnchorChunkKey = result.targetChunkKey();
                    haveSecondaryAnchorChunk = true;
                    boolean ticking = current.isTicking();
                    if (!ticking) current.setTicking(true);
                    tickAnchor = worldAnchors.context(current).packetAnchor();
                    world = current;
                    getLogger().atInfo().log(String.format(
                        "PolicyAgent combat fixture ready: %s[%s] -> %s[%s] "
                            + "at (%.3f, %.3f, %.3f) -> "
                            + "(%.3f, %.3f, %.3f), worldTick=%d "
                            + "tickAnchor=%b %s",
                        spec.agentRole(), agentItem,
                        spec.targetRole(), targetItem,
                        result.agentPosition().x,
                        result.agentPosition().y,
                        result.agentPosition().z,
                        result.targetPosition().x,
                        result.targetPosition().y,
                        result.targetPosition().z,
                        current.getTick(),
                        tickAnchor != null,
                        result.placementReceipt()
                    ));
                    testSubjectReady = true;
                } catch (RuntimeException failure) {
                    testSubjectFailed = true;
                    getLogger().atSevere().withCause(failure).log(
                        "PolicyAgent combat fixture failed: %s",
                        failure
                    );
                } finally {
                    testSubjectSpawnPending = false;
                }
            });
        } catch (Exception failure) {
            testSubjectFailed = true;
            getLogger().atSevere().withCause(failure).log(
                "PolicyAgent could not configure combat-fixture.txt: %s",
                failure
            );
        }
    }

    /**
     * Load and retain the exact fixture chunks off the world thread.
     *
     * <p>{@code World.getChunk()} is only a momentary lookup, and retaining the
     * returned component reference alone does not disable
     * {@code ChunkUnloadingSystem}. In a playerless world the chunk otherwise
     * unloads immediately, serializes the NPC without this plugin's
     * non-serialized marker, and reloads it as a newly claimed actor at the
     * next heartbeat. Mirror {@code NativeEnvironmentSession}: retain the
     * component reference, increment the {@link WorldChunk} keep-loaded count,
     * and mark the chunk ticking before spawning either actor.</p>
     */
    private void pinFixtureChunks(World current, long... chunkKeys) {
        for (long chunkKey : chunkKeys) {
            Ref<ChunkStore> existing = pinnedFixtureChunks.get(chunkKey);
            if (existing != null && existing.isValid()) continue;
            try {
                Ref<ChunkStore> loaded = current
                    .getChunkStore()
                    .getChunkReferenceAsync(chunkKey)
                    .get(30, TimeUnit.SECONDS);
                if (loaded == null || !loaded.isValid()) {
                    throw new IllegalStateException(
                        "fixture chunk did not produce a valid reference: "
                            + chunkKey);
                }
                retainFixtureChunk(current, loaded, chunkKey);
                pinnedFixtureChunks.put(chunkKey, loaded);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(
                    "fixture chunk pin was interrupted", interrupted);
            } catch (java.util.concurrent.ExecutionException
                     | java.util.concurrent.TimeoutException failure) {
                throw new IllegalStateException(
                    "fixture chunk pin failed: " + chunkKey, failure);
            }
        }
    }

    private static void retainFixtureChunk(
        World current,
        Ref<ChunkStore> loaded,
        long chunkKey
    ) {
        CompletableFuture<Void> retained = new CompletableFuture<>();
        current.execute(() -> {
            try {
                Store<ChunkStore> store = current.getChunkStore().getStore();
                WorldChunk chunk = store.getComponent(
                    loaded,
                    WorldChunk.getComponentType()
                );
                if (chunk == null) {
                    throw new IllegalStateException(
                        "loaded fixture chunk has no WorldChunk: " + chunkKey
                    );
                }
                chunk.addKeepLoaded();
                chunk.resetActiveTimer();
                chunk.setFlag(ChunkFlag.TICKING, true);
                retained.complete(null);
            } catch (Throwable failure) {
                retained.completeExceptionally(failure);
            }
        });
        awaitFixtureChunkTask(retained, "retain", chunkKey);
    }

    private static void refreshFixtureChunk(World current, long chunkKey) {
        WorldChunk chunk = current.getChunk(chunkKey);
        if (chunk == null) {
            throw new IllegalStateException(
                "pinned fixture chunk disappeared: " + chunkKey
            );
        }
        chunk.resetActiveTimer();
        chunk.setFlag(ChunkFlag.TICKING, true);
    }

    private void releaseFixtureChunks(World current) {
        List<Map.Entry<Long, Ref<ChunkStore>>> pinned =
            new ArrayList<>(pinnedFixtureChunks.entrySet());
        pinnedFixtureChunks.clear();
        if (current == null || pinned.isEmpty()) return;

        CompletableFuture<Void> released = new CompletableFuture<>();
        current.execute(() -> {
            try {
                Store<ChunkStore> store = current.getChunkStore().getStore();
                for (Map.Entry<Long, Ref<ChunkStore>> entry : pinned) {
                    Ref<ChunkStore> ref = entry.getValue();
                    if (ref == null || !ref.isValid()) continue;
                    WorldChunk chunk = store.getComponent(
                        ref,
                        WorldChunk.getComponentType()
                    );
                    if (chunk != null && chunk.shouldKeepLoaded()) {
                        chunk.removeKeepLoaded();
                    }
                }
                released.complete(null);
            } catch (Throwable failure) {
                released.completeExceptionally(failure);
            }
        });
        try {
            released.get(5, TimeUnit.SECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            getLogger().atWarning().log(
                "PolicyAgent fixture chunk release was interrupted"
            );
        } catch (java.util.concurrent.ExecutionException
                 | java.util.concurrent.TimeoutException failure) {
            getLogger().atWarning().withCause(failure).log(
                "PolicyAgent fixture chunk release failed"
            );
        }
    }

    private static void awaitFixtureChunkTask(
        CompletableFuture<Void> task,
        String operation,
        long chunkKey
    ) {
        try {
            task.get(30, TimeUnit.SECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(
                "fixture chunk " + operation + " was interrupted: " + chunkKey,
                interrupted
            );
        } catch (java.util.concurrent.ExecutionException
                 | java.util.concurrent.TimeoutException failure) {
            throw new IllegalStateException(
                "fixture chunk " + operation + " failed: " + chunkKey,
                failure
            );
        }
    }

    private static boolean fixtureWorldReady(World current) {
        return current != null
            && current.getChunkStore() != null
            && current.getChunkStore().getStore() != null;
    }

    @Override
    protected void shutdown() {
        if (control == null) {
            return;
        }
        long[] counters = control.counters();
        World current = world;
        if (combatSink != null) {
            combatSink.closeActive();
        }
        if (worldVerbSink != null) {
            worldVerbSink.closeActive();
        }
        if (worldAnchors != null) {
            worldAnchors.close();
        }
        releaseFixtureChunks(current);
        if (current != null) {
            // If worldTick is large but policy ticks are ~0, the hook is fine
            // and the NPC's chunk simply never became a ticking chunk.
            getLogger().atInfo().log(String.format(
                "PolicyAgent world: ticking=%b worldTick=%d",
                current.isTicking(), current.getTick()));
        }
        getLogger().atInfo().log(String.format(
            "PolicyAgent stopping: claimed=%d ticks=%d skipped=%d illegal=%d "
                + "worldVerbDropped=%d interactionsSkipped=%d unclaimedRoles=%s",
            attachment.claimed(), counters[0], counters[1], counters[2],
            counters[3], steeringSink.skippedInteractions(),
            attachment.unclaimedRoles()));
        getLogger().atInfo().log(
            "PolicyAgent live action diagnostic: "
                + liveActionControl.describe());
        if (worldVerbSink != null) {
            long[] verbs = worldVerbSink.counters();
            getLogger().atInfo().log(String.format(
                "PolicyAgent World verbs: requested=%d accepted=%d finished=%d "
                    + "failed=%d bindingRejected=%d busy=%d unsupported=%d "
                    + "heldEdges=%d contextFailures=%d",
                verbs[0], verbs[1], verbs[2], verbs[3], verbs[4], verbs[5],
                verbs[6], verbs[7], verbs[8]));
        }
        if (combatSink != null) {
            long[] combat = combatSink.counters();
            getLogger().atInfo().log(String.format(
                "PolicyAgent combat: bound=%d bindRejected=%d unavailable=%d "
                    + "attackAccepted=%d abilityAccepted=%d dodgeAccepted=%d "
                    + "guardStarted=%d abilityFinished=%d abilityFailed=%d "
                    + "dodgeFinished=%d dodgeFailed=%d",
                combat[0], combat[1], combat[2], combat[3], combat[4],
                combat[5], combat[6], combat[7], combat[8], combat[9],
                combat[10]));
        }
    }

    /**
     * Publish counters for the console. Never allowed to disturb the server:
     * a full disk or a locked file must not stop the heartbeat, let alone the
     * world, so failures are logged once per occurrence and swallowed.
     */
    private void publishMetrics(
        long worldTick,
        boolean worldTicking,
        double worldTps,
        double policyTps
    ) {
        if (dataDirectory == null || runtime == null || control == null) {
            return;
        }
        try {
            PolicyMetrics.write(dataDirectory, PolicyMetrics.snapshot(
                role, runtime, attachment, control, steeringSink,
                worldVerbSink == null ? null : worldVerbSink.counters(),
                worldTick, worldTicking, worldTps, policyTps,
                captureNonZero,
                captureLegal));
        } catch (Exception exception) {
            getLogger().atWarning().log("PolicyAgent metrics write failed: " + exception);
        }
    }

    /**
     * Prefer live perception when a checkpoint profile is installed.
     *
     * <p>The frozen capture remains an explicit backwards-compatible harness
     * for action-sink testing. A present profile never falls back to it: a
     * missing bridge facade or schema mismatch leaves the plugin inert instead
     * of silently running open-loop.</p>
     */
    private ActionRuntime configurePerception(Path data) throws Exception {
        Path profileDirectory = data.resolve("profile");
        if (Files.isDirectory(profileDirectory)) {
            PerceptionProfile profile = PerceptionProfile.load(
                profileDirectory);
            liveProfile = profile;
            var lifecycle =
                ReflectiveBridgeCombatLifecycleEvidenceSource.load(profile);
            // Every native ability binding must be resolved off the world
            // thread before any NPC is claimed. Left to the world thread it
            // deadlocks the server permanently -- the tick already holds a read
            // lock on the asset store and the first resolution asks for the
            // write lock, which ReentrantReadWriteLock will never grant. See
            // NativeBindingWarmup for the thread dump and why the memo in
            // InteractionSupport makes warming sufficient.
            //
            // Not done here: plugin setup runs before the asset store is
            // populated, so every id fails with "Native interaction asset not
            // found". The heartbeat thread retries until the assets exist --
            // it is not the world thread either, which is the only property
            // that actually matters.
            warmupProfile = profile;
            ServerRecipeCandidateReader recipes =
                ServerRecipeCandidateReader.load();
            BridgeWorldVerbFacade worldFacade = null;
            try {
                worldFacade = ReflectiveBridgeWorldVerbFacade.load();
                if (!worldFacade.policyEvidenceSupported()) {
                    worldFacade = null;
                    getLogger().atWarning().log(
                        "PolicyAgent Fieldcraft disabled: bridge World-verb "
                            + "transport requires atomic candidate evidence "
                            + "that the standalone capture does not produce");
                }
            } catch (ReflectiveOperationException unavailable) {
                getLogger().atWarning().log(
                    "PolicyAgent Fieldcraft disabled: bridge World-verb "
                        + "facade absent or incompatible: " + unavailable);
            }
            BridgeCombatFacade combatFacade = null;
            try {
                combatFacade = ReflectiveBridgeCombatFacade.load(profile);
                getLogger().atInfo().log(
                    "PolicyAgent combat: bridge facade (full control)");
            } catch (ReflectiveOperationException unavailable) {
                // The deployed bridge does not ship NativePolicyCombatFacade,
                // and without a fallback that left the policy driving movement
                // only -- every attack in-game was the role's own AI, which
                // looks identical to a working agent.
                //
                // The server-side facade selects the ability through
                // CombatSupport's attack override and lets the engine keep the
                // timing, cooldowns and damage. Strictly less control than the
                // bridge, and reported as such so nobody reads an attack as
                // proof of full policy combat.
                ServerCombatSupportFacade serverCombat =
                    ServerCombatSupportFacade.create(profile, readRole(data));
                serverCombatFacade = serverCombat;
                combatFacade = serverCombat;
                getLogger().atWarning().log(
                    "PolicyAgent combat: bridge facade absent (" + unavailable
                        + "); falling back to the server attack-override path "
                        + "-- the policy selects the ability, the role's AI "
                        + "still chooses when to swing");
            }
            LifecycleTraceSink lifecycleTrace = LifecycleTraceSink.disabled();
            if (Files.exists(data.resolve("lifecycle-trace"))) {
                try {
                    var recorder = new JsonLinesLifecycleTraceSink(data, 4096);
                    lifecycleTrace = recorder;
                    getLogger().atInfo().log(
                        "PolicyAgent lifecycle trace: " + recorder.path());
                } catch (IOException unavailable) {
                    getLogger().atWarning().log(
                        "PolicyAgent lifecycle trace unavailable: "
                            + unavailable);
                }
            }
            perception = new LivePolicyPerception(
                profile,
                new PublicServerEvidenceSource(
                    profile,
                    lifecycle,
                    lifecycleTrace,
                    recipes,
                    worldFacade != null
                )
            );
            perceptionDescription = "live profile=" + profile.sha256();
            return new ActionRuntime(recipes, worldFacade, combatFacade);
        }

        CapturedObservationPerception captured =
            CapturedObservationPerception.load(data);
        perception = captured;
        captureNonZero = captured.nonZeroColumns();
        captureLegal = captured.legalActions();
        perceptionDescription = String.format(
            "frozen capture=%d non-zero,%d legal",
            captureNonZero,
            captureLegal
        );
        getLogger().atWarning().log(
            "PolicyAgent is using frozen open-loop perception; install the "
                + "checkpoint profile at policy-agent/profile for live input");
        return null;
    }

    /** Frozen captures need no bindings; live profiles wait for full warm-up. */
    private boolean nativeBindingsReady() {
        return warmupProfile == null || bindingsWarmed;
    }

    /** Explicit fixtures must validate placement before either actor is claimed. */
    private boolean readyToClaimActors() {
        return actorClaimReadiness(
            nativeBindingsReady(), testSubjectConfigured, testSubjectReady);
    }

    static boolean actorClaimReadiness(
        boolean nativeReady,
        boolean fixtureConfigured,
        boolean fixtureReady
    ) {
        return nativeReady && (!fixtureConfigured || fixtureReady);
    }

    private record ActionRuntime(
        ServerRecipeCandidateReader recipes,
        BridgeWorldVerbFacade worldFacade,
        BridgeCombatFacade combatFacade
    ) {}

    /**
     * Ticks between fresh decisions, from {@code decision.txt}; 1 if absent.
     *
     * <p>Note this does not make the policy cheaper -- the network still runs
     * every tick so the GRU carry advances at the rate it was trained at. It
     * changes reaction latency, not throughput. See {@link PolicyControlSystem}.
     */
    private int readDecisionPeriod(Path data) {
        Path file = data.resolve("decision.txt");
        try {
            if (Files.isRegularFile(file)) {
                int value = Integer.parseInt(Files.readString(file).trim());
                if (value >= 1) {
                    return value;
                }
                getLogger().atWarning().log(
                    "PolicyAgent decision.txt must be >= 1, got " + value);
            }
        } catch (Exception exception) {
            getLogger().atWarning().log(
                "PolicyAgent could not read decision.txt: " + exception);
        }
        return 1;
    }

    private String readRole(Path data) throws IOException {
        Path file = data.resolve("role.txt");
        if (!Files.isRegularFile(file)) {
            throw new IOException("validated bundle has no role.txt");
        }
        String value = Files.readString(file).trim();
        if (value.isEmpty()) {
            throw new IOException("validated bundle has an empty role.txt");
        }
        return value.equals("*") ? null : value;
    }
}
