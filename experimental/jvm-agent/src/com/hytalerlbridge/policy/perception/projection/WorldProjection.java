package com.hytalerlbridge.policy.perception.projection;

/** Pure learner projections for actor-local world evidence. */
public final class WorldProjection {

    public static final int ACTOR_WORLD_VALUE_SIZE = 5;
    public static final int ACTOR_WORLD_MASK_SIZE = 3;

    /** Raw availability and values supplied by the World producer. */
    public record ActorWorldInput(
        boolean controllerMediumAvailable,
        boolean controllerInFluid,
        boolean submersionAvailable,
        boolean feetSubmerged,
        boolean eyesSubmerged,
        boolean dropAvailable,
        boolean dropSupportFound,
        float dropHeight,
        float maximumDropHeight
    ) {}

    /** Five values plus the three non-interchangeable availability channels. */
    public record ActorWorldResult(float[] values, boolean[] mask) {
        public ActorWorldResult {
            if (values.length != ACTOR_WORLD_VALUE_SIZE
                || mask.length != ACTOR_WORLD_MASK_SIZE) {
                throw new IllegalArgumentException("actor-world projection width drift");
            }
            values = values.clone();
            mask = mask.clone();
        }

        @Override
        public float[] values() {
            return values.clone();
        }

        @Override
        public boolean[] mask() {
            return mask.clone();
        }
    }

    private WorldProjection() {
    }

    /** Match learner-v3's actor-world stack and final [0, 1] clip. */
    public static ActorWorldResult actorWorld(ActorWorldInput input) {
        boolean[] mask = {
            input.controllerMediumAvailable(),
            input.submersionAvailable(),
            input.dropAvailable(),
        };
        float denominator = Math.max(
            input.maximumDropHeight(), Float.MIN_NORMAL);
        float[] values = {
            present(input.controllerMediumAvailable(), input.controllerInFluid()),
            present(input.submersionAvailable(), input.feetSubmerged()),
            present(input.submersionAvailable(), input.eyesSubmerged()),
            present(input.dropAvailable(), input.dropSupportFound()),
            input.dropAvailable() ? input.dropHeight() / denominator : 0.0f,
        };
        for (int index = 0; index < values.length; index++) {
            values[index] = Math.max(0.0f, Math.min(1.0f, values[index]));
        }
        return new ActorWorldResult(values, mask);
    }

    private static float present(boolean available, boolean value) {
        return available && value ? 1.0f : 0.0f;
    }
}
