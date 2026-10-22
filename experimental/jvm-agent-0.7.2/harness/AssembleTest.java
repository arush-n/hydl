import com.hytalerlbridge.policy.ActionDecoder;
import com.hytalerlbridge.policy.ObservationAssembler;
import com.hytalerlbridge.policy.Policy;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Acceptance test for the Java observation assembler: 15656/15656 or fail.
 *
 * <p>Exact equality is the bar, not a tolerance. The assembler does selection,
 * masking, reshaping and concatenation -- all order-preserving operations on
 * values JAX already produced -- so any difference means a column is in the
 * wrong place or a mask was applied wrongly, not float drift.
 */
public final class AssembleTest {

    public static void main(String[] args) throws Exception {
        Path groups = Paths.get(args.length > 0 ? args[0] : "case/groups");
        float[] built = ObservationAssembler.assemble(groups);
        float[] expected = Policy.read(groups, "expected_observation");

        if (built.length != expected.length) {
            System.out.printf("FAIL width: java=%d jax=%d%n", built.length, expected.length);
            System.exit(1);
        }

        int differing = 0;
        int firstBad = -1;
        float worst = 0.0f;
        for (int i = 0; i < built.length; i++) {
            if (built[i] != expected[i]) {
                differing++;
                if (firstBad < 0) {
                    firstBad = i;
                }
                worst = Math.max(worst, Math.abs(built[i] - expected[i]));
            }
        }

        int nonZero = 0;
        for (float value : expected) {
            if (value != 0.0f) {
                nonZero++;
            }
        }

        System.out.printf("columns      : %d%n", built.length);
        System.out.printf("non-zero     : %d (a vector of zeros would pass vacuously)%n",
                nonZero);
        System.out.printf("identical    : %d%n", built.length - differing);
        System.out.printf("differing    : %d%n", differing);
        if (differing > 0) {
            System.out.printf("first at     : %d (java=%.6f jax=%.6f)%n",
                    firstBad, built[firstBad], expected[firstBad]);
            System.out.printf("worst delta  : %.6f%n", worst);
        }

        int builtNonZero = 0;
        for (float value : built) {
            if (value != 0.0f) {
                builtNonZero++;
            }
        }
        boolean ok = differing == 0 && builtNonZero == nonZero && nonZero > 0;
        System.out.println();
        System.out.println(ok
                ? "PASS - Java assembles the observation exactly as JAX does"
                : "FAIL");
        if (!ok) {
            System.exit(1);
        }
    }
}
