package com.hytalerlbridge.nativebackend;

import com.hypixel.hytale.component.Ref;
import com.hypixel.hytale.component.Store;
import com.hypixel.hytale.protocol.SoundCategory;
import com.hypixel.hytale.server.core.asset.type.soundevent.config.SoundEvent;
import com.hypixel.hytale.server.core.modules.entity.component.MovementAudioComponent;
import com.hypixel.hytale.server.core.modules.entity.component.TransformComponent;
import com.hypixel.hytale.server.core.universe.world.SoundUtil;
import com.hypixel.hytale.server.core.universe.world.storage.EntityStore;
import com.hytalerlbridge.observation.AudioFrame;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.IntToDoubleFunction;
import org.joml.Vector3d;

/**
 * Receives the live 3-D sound stream from the transformed {@code SoundUtil} and turns it into
 * learner-relative 3-D audio events.
 *
 * <p>The transformer adds {@code public static volatile Consumer hytalerl$soundProbe} to
 * {@code SoundUtil} and calls it at the entry of the terminal {@code playSoundEvent3d} overloads.
 * This bridge attaches by setting that field to its own {@link #sink}. Rows arrive as
 * {@code Object[]{Integer index, SoundCategory|null, Double x, y, z, Float volume, pitch,
 * predicate|null}} and are stored as
 * {@code {index, category, x, y, z, volume, pitch, owner Ref|null, Long nanoTime}}.
 */
public final class NativeSoundProbeBridge {

    static final String PROBE_FIELD = "hytalerl$soundProbe";
    static final int RING_CAPACITY = 256;

    private static final System.Logger LOGGER =
        System.getLogger(NativeSoundProbeBridge.class.getName());

    private final Consumer<Object> sink = this::accept;
    private final Field ownerField = ownerFieldOrNull();
    /** Guarded by {@code this}. Holds stored rows, oldest first. */
    private final ArrayDeque<Object[]> ring = new ArrayDeque<>(RING_CAPACITY);
    /** Guarded by {@code this}. */
    private long dropped;
    /** The field this bridge set in {@link #attach}; null when detached or never attached. */
    private Field attachedField;
    private boolean warned;

    /** One 3-D sound kept for the learner: already distance-filtered and slot-mapped. */
    public record Sample(
        int soundEventIndex,
        int category,
        double relativeX,
        double relativeY,
        double relativeZ,
        float volumeModifier,
        float pitchModifier,
        int sourceSlot
    ) {}

    /** Attaches to the live {@code SoundUtil}, or returns empty when the probe is not installed. */
    public static Optional<NativeSoundProbeBridge> attach() {
        return attach(SoundUtil.class);
    }

    /** Attaches to {@code target}'s {@link #PROBE_FIELD}; empty when that field is absent. */
    static Optional<NativeSoundProbeBridge> attach(Class<?> target) {
        NativeSoundProbeBridge bridge = new NativeSoundProbeBridge();
        try {
            Field field = target.getField(PROBE_FIELD);
            if (
                !Modifier.isStatic(field.getModifiers())
                    || !Consumer.class.isAssignableFrom(field.getType())
            ) {
                return Optional.empty();
            }
            field.set(null, bridge.sink);
            bridge.attachedField = field;
            return Optional.of(bridge);
        } catch (NoSuchFieldException absent) {
            return Optional.empty();
        } catch (ReflectiveOperationException | RuntimeException | LinkageError failed) {
            LOGGER.log(System.Logger.Level.WARNING,
                "Sound probe attach failed; 3-D audio stays off: " + failed);
            return Optional.empty();
        }
    }

    private NativeSoundProbeBridge() {}

    /** Clears the probe field, but only while it still holds this bridge's sink. */
    public synchronized void detach() {
        Field field = attachedField;
        attachedField = null;
        if (field == null) return;
        try {
            if (field.get(null) == sink) field.set(null, null);
        } catch (ReflectiveOperationException | RuntimeException failed) {
            LOGGER.log(System.Logger.Level.WARNING, "Sound probe detach failed: " + failed);
        }
    }

    /** Rows dropped because the ring was full, since attach. */
    public synchronized long dropped() {
        return dropped;
    }

    /**
     * Receives one probe row from a server world thread. Never throws, so a probe fault cannot
     * reach the server's sound path.
     */
    void accept(Object value) {
        try {
            if (!(value instanceof Object[] row) || row.length < 8) return;
            Object[] stored = {
                row[0], row[1], row[2], row[3], row[4], row[5], row[6],
                ownerOf(row[7]),
                System.nanoTime(),
            };
            synchronized (this) {
                if (ring.size() == RING_CAPACITY) {
                    ring.removeFirst();
                    dropped++;
                }
                ring.addLast(stored);
            }
        } catch (Throwable ignored) {
            // Deliberately silent: see the class comment.
        }
    }

    /** Returns and clears the stored rows, oldest first. */
    public synchronized List<Object[]> drain() {
        List<Object[]> rows = new ArrayList<>(ring);
        ring.clear();
        return rows;
    }

    private Object ownerOf(Object predicate) throws IllegalAccessException {
        if (ownerField == null
            || !(predicate instanceof MovementAudioComponent.ShouldHearPredicate)) {
            return null;
        }
        return ownerField.get(predicate);
    }

    private static Field ownerFieldOrNull() {
        try {
            Field field = MovementAudioComponent.ShouldHearPredicate.class.getDeclaredField("owner");
            field.setAccessible(true);
            return field;
        } catch (ReflectiveOperationException | RuntimeException | LinkageError failed) {
            LOGGER.log(System.Logger.Level.WARNING,
                "Sound probe cannot read the predicate owner; owners stay null: " + failed);
            return null;
        }
    }

    /**
     * Drains the probe into the step buffer at the given step-relative offset. Called once per
     * native tick, before the tick's simulated time is advanced, to match packet capture.
     * Never throws: a probe failure must not fail the step.
     */
    public void captureTick(
        AudioFrame.StepBuffer buffer,
        double offsetSeconds,
        Store<EntityStore> store,
        Ref<EntityStore> agent,
        Ref<EntityStore> target
    ) {
        try {
            List<Object[]> rows = drain();
            if (rows.isEmpty() || agent == null || !agent.isValid()) return;
            TransformComponent transform =
                store.getComponent(agent, TransformComponent.getComponentType());
            if (transform == null) return;
            Vector3d learner = transform.getPosition();
            for (Sample sample : keep(
                rows,
                learner.x,
                learner.y,
                learner.z,
                agent,
                target,
                NativeSoundProbeBridge::maxDistance
            )) {
                buffer.captureSpatial(
                    sample.soundEventIndex(),
                    sample.category(),
                    sample.relativeX(),
                    sample.relativeY(),
                    sample.relativeZ(),
                    sample.volumeModifier(),
                    sample.pitchModifier(),
                    sample.sourceSlot(),
                    offsetSeconds
                );
            }
        } catch (RuntimeException failed) {
            synchronized (this) {
                if (warned) return;
                warned = true;
            }
            LOGGER.log(System.Logger.Level.WARNING,
                "Sound probe capture failed; later ticks are skipped silently: " + failed);
        }
    }

    /**
     * Keeps the rows within their sound's max distance of the learner and maps each owner to a
     * slot. Pure: the max-distance lookup and the owner comparison are injected.
     *
     * @param maxDistance returns the sound's max distance, or a negative value when unknown
     */
    public static List<Sample> keep(
        List<Object[]> rows,
        double learnerX,
        double learnerY,
        double learnerZ,
        Object agent,
        Object target,
        IntToDoubleFunction maxDistance
    ) {
        List<Sample> kept = new ArrayList<>();
        for (Object[] row : rows) {
            int index = (Integer) row[0];
            if (index <= 0) continue;
            double x = (Double) row[2];
            double y = (Double) row[3];
            double z = (Double) row[4];
            double relativeX = x - learnerX;
            double relativeY = y - learnerY;
            double relativeZ = z - learnerZ;
            double distance = Math.sqrt(
                relativeX * relativeX + relativeY * relativeY + relativeZ * relativeZ
            );
            double limit = maxDistance.applyAsDouble(index);
            if (!(limit >= 0.0) || !(distance <= limit)) continue;
            kept.add(new Sample(
                index,
                categoryValue(row[1]),
                relativeX,
                relativeY,
                relativeZ,
                ((Number) row[5]).floatValue(),
                ((Number) row[6]).floatValue(),
                slot(row[7], agent, target)
            ));
        }
        return kept;
    }

    static int categoryValue(Object category) {
        return category instanceof SoundCategory value ? value.getValue() : Integer.MIN_VALUE;
    }

    static int slot(Object owner, Object agent, Object target) {
        if (sameEntity(owner, agent)) return AudioFrame.SOURCE_LEARNER;
        if (sameEntity(owner, target)) return AudioFrame.SOURCE_TARGET;
        return AudioFrame.SOURCE_UNKNOWN;
    }

    /**
     * Ref does not override equals (javap, Server-0.7.2), so two Refs name one entity when they
     * share a store and an index. Other objects fall back to equals.
     */
    static boolean sameEntity(Object a, Object b) {
        if (a == null || b == null) return false;
        if (a == b) return true;
        if (a instanceof Ref<?> left && b instanceof Ref<?> right) {
            return left.getStore() == right.getStore() && left.getIndex() == right.getIndex();
        }
        return a.equals(b);
    }

    /** Max distance of a sound-event index, or -1 when the asset is unknown. */
    static double maxDistance(int soundEventIndex) {
        SoundEvent sound = SoundEvent.getAssetMap().getAssetOrDefault(soundEventIndex, null);
        return sound == null ? -1.0 : sound.getMaxDistance();
    }
}
