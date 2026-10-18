import java.nio.file.Files;
import java.nio.file.Path;

import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.PlayerLocomotionProjection;

/**
 * Offline regression probe for the live NPC actuator boundary.
 *
 * <p>This keeps a small source-level smoke check beside the actual packaged
 * contract test. It is useful when inspecting a deployed class directory: it
 * proves the corrected body-relative path is present and that the old absolute
 * formula/fixed-speed call did not return. It does not require a Hytale server.
 */
public final class ActuatorSemanticsAudit {
    private record Vec(double x, double z) {}

    /** The vector produced by JAX after body-relative direction decoding. */
    private static Vec jaxVector(int direction, double bodyYawDegrees) {
        double localAngle = (direction - 1) * (Math.PI / 4.0);
        double longitudinal = Math.cos(localAngle);
        double lateral = Math.sin(localAngle);
        double yaw = Math.toRadians(bodyYawDegrees);
        return new Vec(
            longitudinal * -Math.sin(yaw) + lateral * Math.cos(yaw),
            longitudinal * -Math.cos(yaw) - lateral * Math.sin(yaw)
        );
    }

    /** The vector emitted by the corrected Java projection for world_move. */
    private static Vec currentJavaVector(int direction, double bodyYawDegrees) {
        PlayerLocomotionProjection.Vector projected =
            PlayerLocomotionProjection.project(
                false, false, false, false, direction,
                ActionDecoder.GAIT_IDLE, Math.toRadians(bodyYawDegrees));
        return new Vec(projected.x(), projected.z());
    }

    public static void main(String[] args) throws Exception {
        Path sinkPath = Path.of(args.length > 0
            ? args[0]
            : "experimental/jvm-agent/src/com/hytalerlbridge/policy/"
                + "SteeringActionSink.java");
        String sink = Files.readString(sinkPath);

        // At yaw zero the old and corrected frames coincide, so this check must
        // use a nonzero body yaw. Direction 8 at -45 degrees is straight
        // forward in the body frame: JAX (0,-1).
        Vec expected = jaxVector(8, -45.0);
        Vec actual = currentJavaVector(8, -45.0);
        double error = Math.hypot(expected.x() - actual.x(),
            expected.z() - actual.z());
        if (error > 1.0e-9) {
            throw new AssertionError("corrected world-move projection mismatch: "
                + expected + " vs " + actual);
        }
        for (int direction = 1; direction <= 8; direction++) {
            Vec zeroYawJax = jaxVector(direction, 0.0);
            Vec zeroYawJava = currentJavaVector(direction, 0.0);
            if (Math.hypot(zeroYawJax.x() - zeroYawJava.x(),
                    zeroYawJax.z() - zeroYawJava.z()) > 1.0e-12) {
                throw new AssertionError(
                    "zero-yaw control mismatch at direction " + direction);
            }
        }

        if (sink.contains("double worldAngle")
            || sink.contains("dx = Math.sin(worldAngle)")
            || sink.contains("dz = -Math.cos(worldAngle)")) {
            throw new AssertionError("SteeringActionSink still contains the old "
                + "absolute world-move formula");
        }
        if (!sink.contains("PlayerLocomotionProjection.project(action, yaw)")
            || sink.contains("body.setTranslationRelativeSpeed(1.0)")) {
            throw new AssertionError("SteeringActionSink does not preserve the "
                + "gait projection through the NPC steering vector");
        }

        System.out.printf(
            "PASS - world_move direction=8 bodyYaw=-45: "
                + "jax=(%.6f,%.6f) java=(%.6f,%.6f) error=%.6f%n",
            expected.x(), expected.z(), actual.x(), actual.z(), error);
        System.out.println(
            "PASS - SteeringActionSink uses body-relative projection and "
                + "retains gait magnitude");
    }
}
