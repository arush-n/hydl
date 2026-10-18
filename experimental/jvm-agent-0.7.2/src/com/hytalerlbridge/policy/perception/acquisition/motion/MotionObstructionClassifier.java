package com.hytalerlbridge.policy.perception.acquisition.motion;

import java.util.List;

/**
 * Pure obstruction rule extracted from Hytale's
 * {@code MotionControllerBase.applyRailStep}.
 *
 * <p>A swept collision result may contain supporting-floor contacts and
 * tangential contacts that do not shorten the requested movement. Counting
 * every contact as an obstruction therefore closes valid horizontal Dodge
 * corridors. This class deliberately owns only the engine's classification
 * rule; acquisition of collision contacts remains in
 * {@link ServerDodgeCorridorReader}.</p>
 */
public final class MotionObstructionClassifier {

    private static final double MIN_GRAZING_THRESHOLD = 1.0e-6;
    private static final double GRAZING_LENGTH_FRACTION = 0.05;
    private static final double COMPLETE_FRACTION = 0.999999;

    /** Minimal immutable view of one ordered engine block contact. */
    public record Contact(
        double collisionStart,
        double normalX,
        double normalY,
        double normalZ
    ) {
        public Contact {
            if (!Double.isFinite(collisionStart)
                || !Double.isFinite(normalX)
                || !Double.isFinite(normalY)
                || !Double.isFinite(normalZ)) {
                throw new IllegalArgumentException(
                    "motion contact values must be finite");
            }
        }

        /** Match the engine's exact {@code Vector3d.equals(UP)} check. */
        public boolean isWorldNormal() {
            return normalX == 0.0 && normalY == 1.0 && normalZ == 0.0;
        }
    }

    /** Typed receipt retaining why contacts did or did not block movement. */
    public record Classification(
        boolean obstructed,
        double stopFraction,
        int blockingContactIndex,
        int ignoredSlideContacts,
        int ignoredGrazingContacts,
        int lateApproachContacts
    ) {
        public Classification {
            if (!Double.isFinite(stopFraction)
                || stopFraction < 0.0 || stopFraction > 1.0
                || blockingContactIndex < -1
                || ignoredSlideContacts < 0
                || ignoredGrazingContacts < 0
                || lateApproachContacts < 0) {
                throw new IllegalArgumentException(
                    "invalid motion-obstruction classification");
            }
            if (obstructed != (blockingContactIndex >= 0)) {
                throw new IllegalArgumentException(
                    "blocking index and obstruction flag disagree");
            }
        }
    }

    private MotionObstructionClassifier() {}

    /**
     * Classify sorted block contacts exactly like the public NPC rail step.
     *
     * <p>The collision module returns contacts ordered by collision start.
     * Supporting contacts inside the active slide interval and grazing
     * contacts are ignored. The first approaching non-slide contact ends the
     * scan; it obstructs only when it occurs before the completed-path
     * sentinel.</p>
     */
    public static Classification classify(
        double displacementX,
        double displacementY,
        double displacementZ,
        boolean sliding,
        double slideEnd,
        List<Contact> contacts
    ) {
        if (!Double.isFinite(displacementX)
            || !Double.isFinite(displacementY)
            || !Double.isFinite(displacementZ)
            || !Double.isFinite(slideEnd)
            || contacts == null) {
            throw new IllegalArgumentException(
                "motion obstruction inputs must be finite and non-null");
        }
        double length = Math.sqrt(
            displacementX * displacementX
                + displacementY * displacementY
                + displacementZ * displacementZ);
        if (!(length > 0.0) || !Double.isFinite(length)) {
            throw new IllegalArgumentException(
                "motion obstruction displacement must be non-zero");
        }
        double grazingThreshold = Math.max(
            MIN_GRAZING_THRESHOLD,
            GRAZING_LENGTH_FRACTION * length
        );
        int ignoredSlide = 0;
        int ignoredGrazing = 0;
        int lateApproach = 0;
        for (int index = 0; index < contacts.size(); index++) {
            Contact hit = contacts.get(index);
            if (hit == null) {
                throw new IllegalArgumentException(
                    "motion contacts cannot contain null");
            }
            boolean inSlideWindow = sliding
                && hit.collisionStart() <= slideEnd
                && hit.isWorldNormal();
            if (inSlideWindow) {
                ignoredSlide++;
                continue;
            }
            double approach = displacementX * hit.normalX()
                + displacementY * hit.normalY()
                + displacementZ * hit.normalZ();
            if (approach >= -grazingThreshold) {
                ignoredGrazing++;
                continue;
            }
            double clampedStart = Math.max(0.0, hit.collisionStart());
            if (clampedStart < COMPLETE_FRACTION) {
                return new Classification(
                    true,
                    clampedStart,
                    index,
                    ignoredSlide,
                    ignoredGrazing,
                    lateApproach
                );
            }
            // MotionControllerBase breaks on this first approaching contact,
            // even though a contact at the completion sentinel does not
            // shorten the requested translation.
            lateApproach++;
            break;
        }
        return new Classification(
            false,
            1.0,
            -1,
            ignoredSlide,
            ignoredGrazing,
            lateApproach
        );
    }
}
