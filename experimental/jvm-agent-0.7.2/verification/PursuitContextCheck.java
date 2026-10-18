import com.hytalerlbridge.policy.Projection;
import com.hytalerlbridge.policy.PolicyAgentMarker;
import com.hytalerlbridge.policy.perception.PursuitTargetContext;
import com.hytalerlbridge.policy.perception.profile.PerceptionProfile;
import java.nio.file.Path;
import java.util.Arrays;

/** Geometry and disclosure checks for the opt-in Arena pursuit supplement. */
public final class PursuitContextCheck {
    public static void main(String[] args) throws Exception {
        Projection.Params params = PerceptionProfile.load(Path.of(args[0])).params();
        float[] row = new float[16];
        Arrays.fill(row, 0.125f);
        row[15] = 0; // Behind an occluder: disclosure must remain unchanged.
        Projection.TargetState state = new Projection.TargetState(
            0, 0, 0, 0, 1, 0, 0,
            3, 0, -4, 1, 2, -2, 0, 0, 0, 0, 0, 0);
        PursuitTargetContext.apply(row, state, params);
        near(row[0], 4f / 24); near(row[1], 3f / 24);
        near(row[6], 5f / 24); near(row[7], 5f / 24);
        near(row[9], 0.6f); near(row[10], 0.8f);
        near(row[3], 2f / (float) params.targetChaseSpeed());
        near(row[4], 1f / (float) params.targetChaseSpeed());
        near(row[5], 1f / (float) params.verticalSpeedScale());
        for (int i : new int[] {8, 11, 12, 13, 14}) near(row[i], 0.125f);
        near(row[15], 0);
        state = new Projection.TargetState(
            0, 0, 0, 0, 0, 0, 90,
            3, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        PursuitTargetContext.apply(row, state, params);
        near(row[0], -3f / 24); near(row[1], 4f / 24);
        state = new Projection.TargetState(
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        PursuitTargetContext.apply(row, state, params);
        near(row[9], 0); near(row[10], 0);
        PolicyAgentMarker marker = new PolicyAgentMarker();
        marker.setPose(0, 0);
        marker.applyLookDelta(30, 10);
        marker.applyBodyYawDelta(-45);
        near(marker.desiredYaw(), (float) Math.toRadians(30));
        near(marker.desiredBodyYaw(), (float) Math.toRadians(-45));
        near(marker.clone().desiredBodyYaw(), marker.desiredBodyYaw());
        marker.reset();
        near(marker.desiredBodyYaw(), 0);
        System.out.println("PASS - pursuit context geometry, relative velocity, coincidence, and visibility preservation");
    }
    private static void near(float actual, float expected) {
        if (!Float.isFinite(actual) || Math.abs(actual - expected) > 1e-6f)
            throw new AssertionError(actual + " != " + expected);
    }
}
