package com.hytalerlbridge.policy;

import com.hypixel.hytale.component.ArchetypeChunk;
import com.hypixel.hytale.component.CommandBuffer;
import com.hypixel.hytale.component.ComponentType;
import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.component.dependency.Dependency;
import com.hypixel.hytale.component.dependency.Order;
import com.hypixel.hytale.component.dependency.SystemDependency;
import com.hypixel.hytale.component.query.Query;
import com.hypixel.hytale.component.system.tick.EntityTickingSystem;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.modules.interaction.system.InteractionSystems;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hypixel.hytale.server.npc.components.StepComponent;
import com.hypixel.hytale.server.npc.entities.NPCEntity;
import com.hypixel.hytale.server.npc.systems.AvoidanceSystem;
import com.hypixel.hytale.server.npc.systems.SteeringSystem;
import com.hytalerlbridge.policy.action.ActionSelectionControl;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Drives every NPC carrying a {@link PolicyAgentMarker} from a trained policy.
 *
 * <p>This is the autonomous counterpart to the bridge's {@code
 * NativeControlSystem}: same query shape, same ordering, but the action comes
 * from a network running in this JVM rather than from a step delivered over the
 * socket. Optional interaction sinks call only public, stateless bridge
 * facades; they never enter a native environment session, so policy-driven
 * NPCs and socket-driven RL episodes can coexist in one server.
 *
 * <p><b>The ordering constraints are load-bearing and copied deliberately.</b>
 * Steering written before {@code AvoidanceSystem} would be overwritten by it;
 * written after {@code SteeringSystem} it would be ignored for a tick. Queuing
 * player-equivalent inputs before the tick's interaction drain makes the same
 * chain execute one engine tick early. These were established by the bridge's
 * fidelity work -- they are not a guess, and changing them silently changes
 * behaviour rather than failing.
 *
 * <p>Runs single-threaded ({@code isParallel} false) for the same reason the
 * bridge's does: applying control mutates role steering and queues interactions,
 * neither of which is safe to do concurrently across a chunk.
 */
public final class PolicyControlSystem extends EntityTickingSystem<EntityStore> {

    private static final System.Logger LOGGER =
        System.getLogger(PolicyControlSystem.class.getName());

    /** Report roughly every 10 seconds at 30 TPS. */
    private static final long LOG_EVERY_TICKS = 300;

    private final ComponentType<EntityStore, PolicyAgentMarker> markerType;
    private final PolicyPerception perception;
    private final PolicyActionSink sink;
    private final int decisionPeriod;
    private final ActionSelectionControl actionControl;
    private final Query<EntityStore> query;
    private final Set<Dependency<EntityStore>> dependencies;

    private volatile ActionDecoder.Decoded lastDecision;
    private volatile float lastValue;

    /**
     * The first NPC this system ever drove, kept as a stable subject for the
     * motion readout. Reporting "the last marker ticked" would hop between
     * entities every tick and make displacement meaningless.
     */
    private volatile PolicyAgentMarker witness;

    private final AtomicLong ticks = new AtomicLong();
    private final AtomicLong skippedNoObservation = new AtomicLong();
    private final AtomicLong rejectedIllegal = new AtomicLong();
    private final AtomicLong worldVerbRequests = new AtomicLong();
    private final AtomicLong locomotionRequests = new AtomicLong();
    private final AtomicLong droppedWorldVerb = new AtomicLong();
    private final AtomicLong heldDecisions = new AtomicLong();

    public PolicyControlSystem(
        ComponentType<EntityStore, PolicyAgentMarker> markerType,
        PolicyPerception perception,
        PolicyActionSink sink
    ) {
        this(markerType, perception, sink, 1, ActionSelectionControl.NONE);
    }

    /**
     * @param decisionPeriod ticks between fresh decisions; the action is held
     *     in between. 1 acts every tick -- 33 ms at 30 TPS, which is symmetric
     *     and correct for training but faster than any person can play. 6-8 is
     *     roughly human reaction latency. See {@code console/docs/DECISION-RATE.md}.
     */
    public PolicyControlSystem(
        ComponentType<EntityStore, PolicyAgentMarker> markerType,
        PolicyPerception perception,
        PolicyActionSink sink,
        int decisionPeriod
    ) {
        this(
            markerType,
            perception,
            sink,
            decisionPeriod,
            ActionSelectionControl.NONE
        );
    }

    public PolicyControlSystem(
        ComponentType<EntityStore, PolicyAgentMarker> markerType,
        PolicyPerception perception,
        PolicyActionSink sink,
        int decisionPeriod,
        ActionSelectionControl actionControl
    ) {
        if (decisionPeriod < 1) {
            throw new IllegalArgumentException(
                "decisionPeriod must be >= 1, got " + decisionPeriod);
        }
        this.markerType = markerType;
        this.perception = perception;
        this.sink = sink;
        this.decisionPeriod = decisionPeriod;
        this.actionControl = actionControl == null
            ? ActionSelectionControl.NONE : actionControl;
        this.query = Query.and(markerType, NPCEntity.getComponentType());
        this.dependencies = orderingDependencies();
    }

    /** Identical to {@code NativeControlSystem.orderingDependencies()}. */
    static Set<Dependency<EntityStore>> orderingDependencies() {
        return Set.of(
            new SystemDependency<>(Order.AFTER, AvoidanceSystem.class),
            new SystemDependency<>(
                Order.AFTER,
                InteractionSystems.TickInteractionManagerSystem.class
            ),
            new SystemDependency<>(Order.BEFORE, SteeringSystem.class)
        );
    }

    @Override
    public Query<EntityStore> getQuery() {
        return query;
    }

    @Override
    public Set<Dependency<EntityStore>> getDependencies() {
        return dependencies;
    }

    @Override
    public boolean isParallel(int entityCount, int chunkCount) {
        return false;
    }

    @Override
    public void tick(
        float deltaTime,
        int index,
        ArchetypeChunk<EntityStore> chunk,
        Store<EntityStore> store,
        CommandBuffer<EntityStore> commandBuffer
    ) {
        PolicyAgentMarker marker = chunk.getComponent(index, markerType);
        if (marker == null || !marker.active()) {
            return;
        }
        NPCEntity npc = chunk.getComponent(index, NPCEntity.getComponentType());
        if (npc == null) {
            return;
        }
        Ref<EntityStore> ref = chunk.getReferenceTo(index);
        PolicyRuntime runtime = marker.runtime();

        PolicyPerception.Sample sample =
            perception.sample(ref, npc, store, marker.slot(), deltaTime);
        if (sample == null) {
            // Leave the NPC to its authored behaviour. Substituting zeros here
            // would be worse than doing nothing: the network would still emit a
            // confident action, derived from an observation that never happened.
            skippedNoObservation.incrementAndGet();
            // Takeover actors carry Frozen, so returning without a one-tick
            // StepComponent also prevents this system from running next tick.
            // Keep the native pipeline alive while continuing to abstain;
            // otherwise one transient missing row freezes the actor forever.
            if (marker.stepFrozenNpc()) {
                commandBuffer.addComponent(
                    ref,
                    StepComponent.getComponentType(),
                    new StepComponent(deltaTime)
                );
            }
            return;
        }
        float[] observation = sample.observation();
        boolean[] legal = sample.actionMask();

        expect(observation.length == runtime.observationSize(),
            "observation width", observation.length, runtime.observationSize());
        expect(legal.length == runtime.actionSize(),
            "action mask width", legal.length, runtime.actionSize());

        // The network runs on EVERY tick, even when its output will be
        // discarded, and that is deliberate rather than wasteful.
        //
        // The carry is a GRU hidden state, and the policy was trained with it
        // advancing once per environment tick -- there is no action repeat
        // anywhere in the gym, so `decision_period` never existed during
        // training. Skipping the forward pass on held ticks would advance the
        // memory at 1/period the rate it was trained at, which is a silent
        // change to what the network *is*, not a speed optimisation. JAX's own
        // `_throttle` (console/core/runner.py) does the same thing: it
        // evaluates the inner policy every tick and selects with `jnp.where`.
        //
        // So throttling here buys behaviour, not throughput. It is still worth
        // having -- acting on a 33 ms loop is superhuman -- but anyone reading
        // this for a performance win should look at perception instead, which
        // costs far more per tick than the forward pass does.
        PolicyRuntime.ControlledDecision controlled = runtime.decide(
            observation,
            marker.carry(),
            legal,
            sample.actionEvidence(),
            marker.slot(),
            actionControl
        );
        ActionDecoder.Decoded fresh = controlled.action();
        // A one-shot diagnostic must reach the sink on the tick it constrains.
        // It does not alter the recurrent update above, and later decisions
        // return to the configured hold period.
        int selectedPeriod = controlled.constrained() ? 1 : decisionPeriod;
        PolicyAgentMarker.ChosenDecision chosen = marker.chooseDecision(
            selectedPeriod,
            fresh,
            sample.actionEvidence()
        );
        ActionDecoder.Decoded action = chosen.action();
        if (!chosen.fresh()) {
            heldDecisions.incrementAndGet();
        }
        long tick = ticks.incrementAndGet();
        lastDecision = action;
        lastValue = runtime.value(marker.carry());
        if (tick == 1 || tick % LOG_EVERY_TICKS == 0) {
            LOGGER.log(System.Logger.Level.INFO, () -> String.format(
                "policy tick %d slot=%d skill=%d move=%d yaw=%+.2f jump=%b "
                    + "attack=%b legal=%b value=%.4f",
                tick, marker.slot(), action.skillId(), action.worldMoveDirection(),
                action.yawDeltaDegrees(), action.jumpHeld(), action.attack(),
                action.actionLegal(), runtime.value(marker.carry())));
        }

        if (!action.actionLegal()) {
            // The decoder already collapsed this to the neutral action. Applying
            // it is still correct -- it releases held inputs -- but count it,
            // because a policy that is illegal every tick looks identical to one
            // that is merely idle.
            rejectedIllegal.incrementAndGet();
        }
        if (action.requestsWorldVerb()) {
            worldVerbRequests.incrementAndGet();
            if (!sink.supportsWorldVerb(action)) {
                droppedWorldVerb.incrementAndGet();
            }
        }
        if (action.forward() || action.back() || action.left() || action.right()
                || action.worldMoveDirection() != 0) {
            locomotionRequests.incrementAndGet();
        }

        // Sample before applying, so the first sample is the position the
        // policy started from and every later one reflects the previous tick's
        // action having been simulated. Only ticks we actually drive are
        // recorded -- motion while the NPC is under authored control is not
        // ours to claim.
        TransformComponent transform =
            store.getComponent(ref, TransformComponent.getComponentType());
        if (transform != null) {
            var position = transform.getPosition();
            marker.observePosition(position.x(), position.y(), position.z());
            if (witness == null) {
                witness = marker;
            }
        }

        marker.applyLookDelta(action.yawDeltaDegrees(), action.pitchDeltaDegrees());
        // Takeover actors carry Frozen so RoleSystems cannot issue a second,
        // competing decision. Frozen also suppresses every native
        // SteppableTickingSystem. We run after behaviour and avoidance and
        // before SteeringSystem, so this one-tick step enables only the
        // downstream native motion pipeline. Hytale's StepCleanupSystem
        // removes the component in the engine's last system set.
        if (marker.stepFrozenNpc()) {
            commandBuffer.addComponent(
                ref,
                StepComponent.getComponentType(),
                new StepComponent(deltaTime)
            );
        }
        sink.apply(
            ref,
            npc,
            store,
            deltaTime,
            marker,
            action,
            chosen.evidence(),
            chosen.fresh()
        );
    }

    /** The first NPC driven, whose motion the metrics report; may be null. */
    public PolicyAgentMarker witness() {
        return witness;
    }

    /** The most recent decision, across all actors; {@code null} before the first. */
    public ActionDecoder.Decoded lastDecision() {
        return lastDecision;
    }

    /** Critic estimate that accompanied {@link #lastDecision()}. */
    public float lastValue() {
        return lastValue;
    }

    /** Configured ticks between fresh decisions; 1 means act every tick. */
    public int decisionPeriod() {
        return decisionPeriod;
    }

    /** Counters for {@code ticks, skipped, illegal, worldVerbDropped, held}. */
    public long[] counters() {
        return new long[] {
            ticks.get(),
            skippedNoObservation.get(),
            rejectedIllegal.get(),
            droppedWorldVerb.get(),
            heldDecisions.get(),
        };
    }

    /** All decoded World-verb requests, including native rejections. */
    public long worldVerbRequests() {
        return worldVerbRequests.get();
    }

    /** Legal decoded ticks that requested horizontal locomotion. */
    public long locomotionRequests() {
        return locomotionRequests.get();
    }

    private static void expect(boolean ok, String what, int actual, int expected) {
        if (!ok) {
            throw new IllegalStateException(
                "policy " + what + " mismatch: got " + actual + ", expected " + expected
            );
        }
    }
}
