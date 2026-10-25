package com.hytalerlbridge.observation;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

/**
 * Immutable, bounded auditory events delivered during one agent step.
 *
 * <p>The current headless backend observes only world-broadcast 2-D and
 * entity-attached packets. Capture flags are part of the observation so a
 * learner never interprets missing 3-D events as silence.
 */
public record AudioFrame(
    boolean available,
    boolean captures2d,
    boolean captures3d,
    boolean capturesEntity,
    List<Event> events
) {
    public static final String SCHEMA = "hytalerl_audio_frame_v1";
    public static final int VERSION = 1;
    /** Wire version when the frame captures 3-D sounds; adds the eleventh event column. */
    public static final int VERSION_SPATIAL = 2;
    public static final int MAX_EVENTS = 16;

    /** Event source slot: the learner body. */
    public static final int SOURCE_LEARNER = 0;
    /** Event source slot: the target body. */
    public static final int SOURCE_TARGET = 1;
    /** Event source slot: anything else, or unknown. */
    public static final int SOURCE_UNKNOWN = -1;

    public static final int KIND_2D = 0;
    public static final int KIND_3D = 1;
    public static final int KIND_ENTITY = 2;
    public static final int CATEGORY_NONE = -1;

    public AudioFrame {
        events = List.copyOf(events);
        if (events.size() > MAX_EVENTS) {
            throw new IllegalArgumentException(
                "Audio frame exceeds capacity " + MAX_EVENTS
            );
        }
        if (!available && (captures2d || captures3d || capturesEntity || !events.isEmpty())) {
            throw new IllegalArgumentException(
                "Unavailable audio frame cannot advertise capture or events"
            );
        }
        for (Event event : events) {
            boolean captured = switch (event.kind()) {
                case KIND_2D -> captures2d;
                case KIND_3D -> captures3d;
                case KIND_ENTITY -> capturesEntity;
                default -> false;
            };
            if (!captured) {
                throw new IllegalArgumentException(
                    "Audio event kind is not covered by the frame capture mask"
                );
            }
        }
    }

    public static AudioFrame unavailable() {
        return new AudioFrame(false, false, false, false, List.of());
    }

    public static AudioFrame headlessPartial(List<Event> events) {
        return new AudioFrame(true, true, false, true, events);
    }

    /** A frame that also captured 3-D sounds (the live sound probe is attached). */
    public static AudioFrame headlessSpatial(List<Event> events) {
        return new AudioFrame(true, true, true, true, events);
    }

    public record Event(
        int kind,
        int soundEventIndex,
        int category,
        boolean spatial,
        double relativeX,
        double relativeY,
        double relativeZ,
        double volumeModifier,
        double pitchModifier,
        double ageSeconds,
        /** {@link #SOURCE_LEARNER}, {@link #SOURCE_TARGET} or {@link #SOURCE_UNKNOWN}. */
        int sourceSlot
    ) {
        /** Pre-spatial constructor: the source slot is unknown. */
        public Event(
            int kind,
            int soundEventIndex,
            int category,
            boolean spatial,
            double relativeX,
            double relativeY,
            double relativeZ,
            double volumeModifier,
            double pitchModifier,
            double ageSeconds
        ) {
            this(
                kind,
                soundEventIndex,
                category,
                spatial,
                relativeX,
                relativeY,
                relativeZ,
                volumeModifier,
                pitchModifier,
                ageSeconds,
                SOURCE_UNKNOWN
            );
        }

        public Event {
            if (
                sourceSlot != SOURCE_LEARNER
                    && sourceSlot != SOURCE_TARGET
                    && sourceSlot != SOURCE_UNKNOWN
            ) {
                throw new IllegalArgumentException("Audio source slot is out of range");
            }
            if (kind < KIND_2D || kind > KIND_ENTITY) {
                throw new IllegalArgumentException("Unknown audio event kind");
            }
            if (soundEventIndex <= 0) {
                throw new IllegalArgumentException(
                    "Audio sound-event index must be positive"
                );
            }
            boolean categoryValid = kind == KIND_ENTITY
                ? category == CATEGORY_NONE
                : category >= 0 && category <= 3;
            if (!categoryValid) {
                throw new IllegalArgumentException(
                    "Audio category is not canonical for its packet kind"
                );
            }
            if (kind != KIND_3D && spatial) {
                throw new IllegalArgumentException(
                    "Only 3-D audio events can carry a spatial vector"
                );
            }
            if (
                !Double.isFinite(relativeX)
                    || !Double.isFinite(relativeY)
                    || !Double.isFinite(relativeZ)
                    || !Double.isFinite(volumeModifier)
                    || !Double.isFinite(pitchModifier)
                    || !Double.isFinite(ageSeconds)
                    || ageSeconds < 0.0
            ) {
                throw new IllegalArgumentException(
                    "Audio event numeric fields must be finite and age nonnegative"
                );
            }
            if (
                !spatial
                    && (relativeX != 0.0 || relativeY != 0.0 || relativeZ != 0.0)
            ) {
                throw new IllegalArgumentException(
                    "Non-spatial audio event must use a zero relative vector"
                );
            }
        }
    }

    /**
     * Mutable per-step staging owned by one native session.
     *
     * <p>It retains only the last {@link #MAX_EVENTS} events and converts
     * capture offsets into ages exactly once at the immutable step boundary.
     */
    public static final class StepBuffer {
        private final ArrayDeque<CapturedEvent> events =
            new ArrayDeque<>(MAX_EVENTS);
        private int overwrittenCount;
        private int invalidCount;
        private double lastOffsetSeconds = Double.NEGATIVE_INFINITY;
        private boolean spatialCaptured;

        /** Marks the frame as capturing 3-D sounds, so {@link #captureSpatial} is accepted. */
        public synchronized void markSpatialCaptured() {
            spatialCaptured = true;
        }

        public synchronized void captureSpatial(
            int soundEventIndex,
            int category,
            double relativeX,
            double relativeY,
            double relativeZ,
            float volumeModifier,
            float pitchModifier,
            int sourceSlot,
            double offsetSeconds
        ) {
            boolean categoryValid = category >= 0 && category <= 3;
            if (
                !spatialCaptured
                    || soundEventIndex <= 0
                    || !categoryValid
                    || sourceSlot < SOURCE_UNKNOWN
                    || sourceSlot > SOURCE_TARGET
                    || !Double.isFinite(relativeX)
                    || !Double.isFinite(relativeY)
                    || !Double.isFinite(relativeZ)
                    || !Float.isFinite(volumeModifier)
                    || !Float.isFinite(pitchModifier)
                    || !Double.isFinite(offsetSeconds)
                    || offsetSeconds < 0.0
                    || offsetSeconds < lastOffsetSeconds
            ) {
                invalidCount++;
                return;
            }
            lastOffsetSeconds = offsetSeconds;
            if (events.size() == MAX_EVENTS) {
                events.removeFirst();
                overwrittenCount++;
            }
            events.addLast(
                new CapturedEvent(
                    KIND_3D,
                    soundEventIndex,
                    category,
                    true,
                    relativeX,
                    relativeY,
                    relativeZ,
                    sourceSlot,
                    volumeModifier,
                    pitchModifier,
                    offsetSeconds
                )
            );
        }

        public synchronized void captureNonSpatial(
            int kind,
            int soundEventIndex,
            int category,
            float volumeModifier,
            float pitchModifier,
            double offsetSeconds
        ) {
            boolean kindValid = kind == KIND_2D || kind == KIND_ENTITY;
            boolean categoryValid = kind == KIND_ENTITY
                ? category == CATEGORY_NONE
                : category >= 0 && category <= 3;
            if (
                !kindValid
                    || soundEventIndex <= 0
                    || !categoryValid
                    || !Float.isFinite(volumeModifier)
                    || !Float.isFinite(pitchModifier)
                    || !Double.isFinite(offsetSeconds)
                    || offsetSeconds < 0.0
                    || offsetSeconds < lastOffsetSeconds
            ) {
                invalidCount++;
                return;
            }
            lastOffsetSeconds = offsetSeconds;
            if (events.size() == MAX_EVENTS) {
                events.removeFirst();
                overwrittenCount++;
            }
            events.addLast(
                new CapturedEvent(
                    kind,
                    soundEventIndex,
                    category,
                    false,
                    0.0,
                    0.0,
                    0.0,
                    SOURCE_UNKNOWN,
                    volumeModifier,
                    pitchModifier,
                    offsetSeconds
                )
            );
        }

        public synchronized AudioFrame finish(double durationSeconds) {
            if (!Double.isFinite(durationSeconds) || durationSeconds < 0.0) {
                throw new IllegalArgumentException(
                    "Audio step duration must be finite and nonnegative"
                );
            }
            List<Event> finalized = new ArrayList<>(events.size());
            for (CapturedEvent captured : events) {
                if (captured.offsetSeconds() > durationSeconds) {
                    invalidCount++;
                    continue;
                }
                finalized.add(
                    new Event(
                        captured.kind(),
                        captured.soundEventIndex(),
                        captured.category(),
                        captured.spatial(),
                        captured.relativeX(),
                        captured.relativeY(),
                        captured.relativeZ(),
                        captured.volumeModifier(),
                        captured.pitchModifier(),
                        durationSeconds - captured.offsetSeconds(),
                        captured.sourceSlot()
                    )
                );
            }
            return new AudioFrame(true, true, spatialCaptured, true, finalized);
        }

        public synchronized int overwrittenCount() {
            return overwrittenCount;
        }

        public synchronized int invalidCount() {
            return invalidCount;
        }

        public synchronized int retainedCount() {
            return events.size();
        }

        private record CapturedEvent(
            int kind,
            int soundEventIndex,
            int category,
            boolean spatial,
            double relativeX,
            double relativeY,
            double relativeZ,
            int sourceSlot,
            float volumeModifier,
            float pitchModifier,
            double offsetSeconds
        ) {}
    }
}
