#!/usr/bin/env bash
# Build and run every JVM-agent test, with the right arguments.
#
# The arguments are the reason this exists. The tests take different fixture
# paths -- `AssembleTest` takes a group directory, `Main` and `ActionTest` take
# the case root, `ParityTest` takes both -- and passing the wrong one produces a
# stack trace that reads exactly like a regression. That happened once already.
#
# Exit code is the number of failing tests, so CI can gate on it.
set -uo pipefail
cd "$(dirname "$0")"

# Defaults resolve the same way they always did, but relative to this checkout
# and the current user rather than one hardcoded machine. Override either with
#   JDK=... SERVER_JAR=... ./run_tests.sh
JDK="${JDK:-../../HytaleRL/.tools/jdk25/jdk-25.0.3+9/bin}"
_GRADLE_HOME="${GRADLE_USER_HOME:-${USERPROFILE:-$HOME}/.gradle}"
JAR="${SERVER_JAR:-$_GRADLE_HOME/caches/modules-2/files-2.1/com.hypixel.hytale/Server/0.7.2/9ef07dd419828247213bb08c28ee502473132d09/Server-0.7.2.jar}"
CP="out;$JAR"
CASE_ROOT="${CASE_ROOT:-case}"
JDK_EXEC="$JDK"
if command -v wslpath >/dev/null 2>&1 \
    && printf '%s' "$JDK" | grep -Eq '^[A-Za-z]:[/\\]'; then
    JDK_EXEC=$(wslpath -u "$JDK")
fi
JAVAC="$JDK_EXEC/javac"
JAVA="$JDK_EXEC/java"
if [ ! -x "$JAVAC" ] && [ -f "$JDK_EXEC/javac.exe" ]; then
    JAVAC="$JDK_EXEC/javac.exe"
fi
if [ ! -x "$JAVA" ] && [ -f "$JDK_EXEC/java.exe" ]; then
    JAVA="$JDK_EXEC/java.exe"
fi

if [ ! -x "$JAVAC" ]; then
    echo "javac not found at $JDK -- set JDK=<jdk>/bin" >&2
    echo "note: JAVA_HOME on this machine points at a path that does not exist" >&2
    exit 127
fi

echo "== compiling =="
# shellcheck disable=SC2046
"$JAVAC" -d out -cp "$CP" $(find src tests harness -name '*.java')
compile=$?
if [ $compile -ne 0 ]; then
    echo "compile FAILED (rc=$compile)"
    exit 127
fi

failed=0
run() {
    local name="$1"; shift
    local output rc verdict
    output=$("$JAVA" -cp "$CP" "$@" 2>&1)
    rc=$?
    verdict=$(printf '%s\n' "$output" | grep -E '^(PASS|FAIL|SKIP)' | head -1)
    printf '%-34s rc=%-3d %s\n' "$name" "$rc" "${verdict:-<no verdict line>}"
    # A test that neither passes nor fails cleanly is a failure: an exception
    # before the verdict prints is exactly how a broken fixture path looks.
    if [ $rc -ne 0 ] || [ -z "$verdict" ]; then
        failed=$((failed + 1))
        printf '%s\n' "$output" | tail -12 | sed 's/^/    /'
    fi
}

echo
echo "== tests =="
run "Main"                  Main "$CASE_ROOT"
run "PolicyShape"           com.hytalerlbridge.policy.PolicyShapeTest
run "AssembleTest flat"     AssembleTest "$CASE_ROOT/groups"
run "AssembleTest region"   AssembleTest "$CASE_ROOT/groups-region"
run "ActionTest"            ActionTest "$CASE_ROOT"
run "LookDeltaCodebook"     com.hytalerlbridge.policy.action.LookDeltaCodebookTest
run "PlayerLocomotionProjection" com.hytalerlbridge.policy.PlayerLocomotionProjectionTest
run "StandardRootSelector" com.hytalerlbridge.policy.action.StandardRootSelectorTest
run "DeterministicLiveAction" com.hytalerlbridge.policy.diagnostics.action.DeterministicLiveActionControlTest
run "DodgeCooldownDiagnostic" com.hytalerlbridge.policy.diagnostics.action.dodge.DodgeCooldownDiagnosticTest
run "BridgeDodgeContract" com.hytalerlbridge.policy.combat.bridge.ReflectiveBridgeDodgeContractTest
run "LiveActionSetupGate" com.hytalerlbridge.policy.PolicyAgentPluginLiveActionGateTest
run "HookTest"              HookTest "$CASE_ROOT"
run "NativeBindingWarmup"  com.hytalerlbridge.policy.NativeBindingWarmupTest "$CASE_ROOT/profile"
run "HeadlessCombatFixture" com.hytalerlbridge.policy.testing.HeadlessCombatFixtureSpecTest
run "FixturePlacementPlanner" com.hytalerlbridge.policy.testing.placement.FixturePlacementPlannerTest
run "PolicyMirrorFixture" com.hytalerlbridge.policy.PolicyMirrorFixtureContractTest \
    case/diagnostic-profiles/dodge-e6d-iron-sword-mirror \
    case/diagnostic-configs/dodge-e6d-iron-sword-mirror/combat-fixture.txt
run "PolicyMetrics"         com.hytalerlbridge.policy.PolicyMetricsTest
run "ProjectionTest"        ProjectionTest "$CASE_ROOT/projection"
run "DefenseProjectionContract" com.hytalerlbridge.policy.perception.projection.DefenseProjectionContractTest
run "RecipeCandidateProjection" com.hytalerlbridge.policy.perception.projection.RecipeCandidateProjectionTest "$CASE_ROOT/recipe-projection"
run "ServerRecipeCandidates" com.hytalerlbridge.policy.perception.acquisition.crafting.ServerRecipeCandidateReaderTest
run "ServerBlockCandidates" com.hytalerlbridge.policy.perception.acquisition.blocks.ServerBlockCandidateReaderTest
run "PerceptionTest"        PerceptionTest "$CASE_ROOT/parity"
# The profile is weapon-pinned (iron_sword vs iron_mace), so it can legitimately
# compose only its own weapon's states -- pairing it with another weapon fails on
# authored metadata, which is an artifact rather than a defect. Globbing keeps
# this honest during a re-export: ParityTest owns "is the fixture complete", so
# this one does not need to duplicate that check and does not go red merely
# because a later tick has not been written yet.
# shellcheck disable=SC2046
run "ProfileTest"           ProfileTest "$CASE_ROOT/profile" $(ls -d "$CASE_ROOT"/parity/iron_sword_t*/ 2>/dev/null)
# The canonical profile above carries the bit-exact and iron-sword-specific
# pins. These sweep breadth instead: every exported weapon against every state
# of its own pairing, which is what stops a defect in one weapon's authored
# ability table, resource bounds or status catalog from hiding behind
# iron_sword. Both fail closed if their profiles root is missing, so a
# forgotten export_profile.py run cannot pass as coverage.
run "ProfileTest sweep"     ProfileTest --all "$CASE_ROOT/profiles" "$CASE_ROOT/parity"
# Region carries the only terrain: 5,764 geometry and 440 light columns are all
# zero in a flat scene, so without this the production path is certified on ~25%
# of the observation. It needs its own profiles because the region scene swaps
# the world, the held item (a pickaxe) and the opponent (unarmed) all at once.
run "ProfileTest region"    ProfileTest --all "$CASE_ROOT/profiles-region" "$CASE_ROOT/parity" region_
run "AcquisitionTest"       com.hytalerlbridge.policy.perception.acquisition.AcquisitionTest
run "MotionObstruction"    com.hytalerlbridge.policy.perception.acquisition.motion.MotionObstructionClassifierTest
run "DodgeCorridor"        com.hytalerlbridge.policy.perception.acquisition.motion.DodgeCorridorEvaluatorTest case/dodge-motion/jax-randomized-v1.tsv
run "CatalogBackedActorStats" com.hytalerlbridge.policy.perception.acquisition.resources.CatalogBackedActorStatsTest
if [ -f case/diagnostic-profiles/dodge-e6d-iron-sword-mirror-v6/dodge_motion.tsv ]; then
    run "DodgeMotionProfile" com.hytalerlbridge.policy.perception.profile.DodgeMotionProfileTest case/diagnostic-profiles/dodge-e6d-iron-sword-mirror-v6
fi
run "BridgeLifecycleAdapter" com.hytalerlbridge.policy.perception.acquisition.bridge.ReflectiveBridgeCombatLifecycleEvidenceSourceTest "$CASE_ROOT/profile"
run "BridgeWorldVerbAdapter" com.hytalerlbridge.policy.world.bridge.ReflectiveBridgeWorldVerbFacadeTest
run "BridgeCombatAdapter" com.hytalerlbridge.policy.combat.bridge.ReflectiveBridgeCombatFacadeTest "$CASE_ROOT/profile"
run "CombatActionSink"     com.hytalerlbridge.policy.combat.runtime.CombatActionSinkTest
run "CombatBindingProfile" com.hytalerlbridge.policy.combat.bridge.CombatBindingProfileTest "$CASE_ROOT/profiles"
run "ParityTest"            ParityTest "$CASE_ROOT" "$CASE_ROOT/parity"
run "RecurrentTest"         RecurrentTest "$CASE_ROOT" "$CASE_ROOT/parity" "$CASE_ROOT/recurrent"
run "ConcurrencyTest"       ConcurrencyTest "$CASE_ROOT" "$CASE_ROOT/parity"
run "RobustnessTest"        RobustnessTest "$CASE_ROOT"
run "ContractTest"          ContractTest "$CASE_ROOT"
# Raw-weight provenance gate. The test owns an independent canonical-stream
# oracle and 256 seeded same-shape mutations; BundleTest then checks that the
# plugin-facing manifest rejects those changes before arming.
run "PolicyTensorIdentity" com.hytalerlbridge.policy.bundle.PolicyTensorIdentityTest
# Independently packaged bridge target: fixed current tuple, same-size JAR
# substitution, and 256 generic schema/version/hash/future-capacity cases.
run "BridgeTargetAttestation" com.hytalerlbridge.policy.bundle.BridgeTargetAttestationTest
# Deployment gate. `case` is passed as the weights directory so the
# manifest-vs-loaded-weights cases run rather than being skipped.
run "BundleTest"            BundleTest "$CASE_ROOT"

echo
if [ $failed -eq 0 ]; then
    echo "all green"
else
    echo "$failed failing"
fi
exit $failed
