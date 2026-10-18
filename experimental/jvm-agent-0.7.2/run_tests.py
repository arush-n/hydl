#!/usr/bin/env python3
"""Compile and run the JVM-agent verification suite without a shell."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


AGENT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOTS = ("src", "tests", "harness")
VERDICT_RE = re.compile(r"^(PASS|FAIL|SKIP)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-jar", required=True, type=Path)
    parser.add_argument(
        "--bridge-classpath",
        type=Path,
        help="compiled bridge classes or JAR; enables actual bridge ABI checks",
    )
    parser.add_argument("--jdk-bin", type=Path)
    parser.add_argument("--case-root", type=Path, default=Path("case"))
    return parser.parse_args()


def java_tools(jdk_bin: Path | None) -> tuple[str, str]:
    suffix = ".exe" if os.name == "nt" else ""
    if jdk_bin is not None:
        bin_dir = jdk_bin.expanduser().resolve()
    else:
        bin_dir = (AGENT_ROOT / "../../HytaleRL/.tools/jdk25/jdk-25.0.3+9/bin").resolve()
        if not all((bin_dir / f"{tool}{suffix}").is_file() for tool in ("javac", "java")):
            javac, java = shutil.which("javac"), shutil.which("java")
            if javac and java:
                return javac, java
    javac, java = bin_dir / f"javac{suffix}", bin_dir / f"java{suffix}"
    if javac.is_file() and java.is_file():
        return str(javac), str(java)
    raise FileNotFoundError(f"javac and java not found under {bin_dir}; pass --jdk-bin")


def test_cases(
    case_root: Path,
    *,
    actual_bridge: bool = False,
) -> list[tuple[str, str, tuple[str, ...]]]:
    def item(name: str, main: str, *args: str) -> tuple[str, str, tuple[str, ...]]:
        return name, main, args

    def c(*parts: str) -> str:
        return str(case_root.joinpath(*parts))

    def bridge_args(*args: str) -> tuple[str, ...]:
        return ("--actual", *args) if actual_bridge else args

    root = str(case_root)
    tests = [
        item("Main", "Main", root),
        item("PolicyShape", "com.hytalerlbridge.policy.PolicyShapeTest"),
        item("AssembleTest flat", "AssembleTest", c("groups")),
        item("AssembleTest region", "AssembleTest", c("groups-region")),
        item("ActionTest", "ActionTest", root),
        item("LookDeltaCodebook", "com.hytalerlbridge.policy.action.LookDeltaCodebookTest"),
        item("PlayerLocomotionProjection", "com.hytalerlbridge.policy.PlayerLocomotionProjectionTest"),
        item("StandardRootSelector", "com.hytalerlbridge.policy.action.StandardRootSelectorTest"),
        item("DeterministicLiveAction", "com.hytalerlbridge.policy.diagnostics.action.DeterministicLiveActionControlTest"),
        item("DodgeCooldownDiagnostic", "com.hytalerlbridge.policy.diagnostics.action.dodge.DodgeCooldownDiagnosticTest"),
        item("BridgeDodgeContract", "com.hytalerlbridge.policy.combat.bridge.ReflectiveBridgeDodgeContractTest",
             *bridge_args()),
        item("LiveActionSetupGate", "com.hytalerlbridge.policy.PolicyAgentPluginLiveActionGateTest"),
        item("HookTest", "HookTest", root),
        item("NativeBindingWarmup", "com.hytalerlbridge.policy.NativeBindingWarmupTest", c("profile")),
        item("HeadlessCombatFixture", "com.hytalerlbridge.policy.testing.HeadlessCombatFixtureSpecTest"),
        item("FixturePlacementPlanner", "com.hytalerlbridge.policy.testing.placement.FixturePlacementPlannerTest"),
        item("PolicyMirrorFixture", "com.hytalerlbridge.policy.PolicyMirrorFixtureContractTest",
             c("diagnostic-profiles", "dodge-e6d-iron-sword-mirror"),
             c("diagnostic-configs", "dodge-e6d-iron-sword-mirror", "combat-fixture.txt")),
        item("PolicyMetrics", "com.hytalerlbridge.policy.PolicyMetricsTest"),
        item("PolicyTraceSink", "com.hytalerlbridge.policy.PolicyTraceSinkTest"),
        item("ProjectionTest", "ProjectionTest", c("projection")),
        item("DefenseProjectionContract", "com.hytalerlbridge.policy.perception.projection.DefenseProjectionContractTest"),
        item("RecipeCandidateProjection", "com.hytalerlbridge.policy.perception.projection.RecipeCandidateProjectionTest",
             c("recipe-projection")),
        item("ServerRecipeCandidates", "com.hytalerlbridge.policy.perception.acquisition.crafting.ServerRecipeCandidateReaderTest"),
        item("ServerBlockCandidates", "com.hytalerlbridge.policy.perception.acquisition.blocks.ServerBlockCandidateReaderTest"),
        item("PerceptionTest", "PerceptionTest", c("parity")),
    ]

    states = sorted((case_root / "parity").glob("iron_sword_t*"))
    tests.append(item("ProfileTest", "ProfileTest", c("profile"),
                      *(f"{path}{os.sep}" for path in states if path.is_dir())))
    tests.extend([
        item("ProfileTest sweep", "ProfileTest", "--all", c("profiles"), c("parity")),
        item("ProfileTest region", "ProfileTest", "--all", c("profiles-region"), c("parity"), "region_"),
        item("AcquisitionTest", "com.hytalerlbridge.policy.perception.acquisition.AcquisitionTest"),
        item("MotionObstruction", "com.hytalerlbridge.policy.perception.acquisition.motion.MotionObstructionClassifierTest"),
        item("DodgeCorridor", "com.hytalerlbridge.policy.perception.acquisition.motion.DodgeCorridorEvaluatorTest",
             c("dodge-motion", "jax-randomized-v1.tsv")),
        item("CatalogBackedActorStats", "com.hytalerlbridge.policy.perception.acquisition.resources.CatalogBackedActorStatsTest"),
    ])

    dodge_profile = case_root / "diagnostic-profiles/dodge-e6d-iron-sword-mirror-v6/dodge_motion.tsv"
    if dodge_profile.is_file():
        tests.append(item("DodgeMotionProfile", "com.hytalerlbridge.policy.perception.profile.DodgeMotionProfileTest",
                          str(dodge_profile.parent)))

    tests.extend([
        item("BridgeLifecycleAdapter", "com.hytalerlbridge.policy.perception.acquisition.bridge.ReflectiveBridgeCombatLifecycleEvidenceSourceTest",
             *bridge_args(c("profile"))),
        item("BridgeWorldVerbAdapter", "com.hytalerlbridge.policy.world.bridge.ReflectiveBridgeWorldVerbFacadeTest",
             *bridge_args()),
        item("BridgeCombatAdapter", "com.hytalerlbridge.policy.combat.bridge.ReflectiveBridgeCombatFacadeTest",
             *bridge_args(c("profile"))),
        item("CombatActionSink", "com.hytalerlbridge.policy.combat.runtime.CombatActionSinkTest"),
        item("CombatBindingProfile", "com.hytalerlbridge.policy.combat.bridge.CombatBindingProfileTest", c("profiles")),
        item("ParityTest", "ParityTest", root, c("parity")),
        item("RecurrentTest", "RecurrentTest", root, c("parity"), c("recurrent")),
        item("ConcurrencyTest", "ConcurrencyTest", root, c("parity")),
        item("RobustnessTest", "RobustnessTest", root),
        item("ContractTest", "ContractTest", root),
        item("PolicyTensorIdentity", "com.hytalerlbridge.policy.bundle.PolicyTensorIdentityTest"),
        item("BridgeTargetAttestation", "com.hytalerlbridge.policy.bundle.BridgeTargetAttestationTest"),
        item("BundleTest", "BundleTest", root),
    ])
    return tests


def verdict(output: str) -> str | None:
    return next((line for line in output.splitlines() if VERDICT_RE.match(line)), None)


def main() -> int:
    args = parse_args()
    server_jar = args.server_jar.expanduser().resolve()
    if not server_jar.is_file():
        print(f"server JAR not found: {server_jar}", file=sys.stderr)
        return 2
    bridge_classpath = (
        args.bridge_classpath.expanduser().resolve()
        if args.bridge_classpath is not None
        else None
    )
    if bridge_classpath is not None and not bridge_classpath.exists():
        print(f"bridge classpath not found: {bridge_classpath}", file=sys.stderr)
        return 2
    case_root = args.case_root.expanduser()
    case_root = (case_root if case_root.is_absolute() else AGENT_ROOT / case_root).resolve()
    try:
        javac, java = java_tools(args.jdk_bin)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    sources = sorted(source for root in SOURCE_ROOTS for source in (AGENT_ROOT / root).rglob("*.java"))
    if not sources:
        print("no Java sources found under src, tests, or harness", file=sys.stderr)
        return 2
    no_console = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    passed = skipped = failed = 0
    with TemporaryDirectory(prefix="jvm-agent-tests-") as classes:
        classpath_entries = [classes, str(server_jar)]
        if bridge_classpath is not None:
            classpath_entries.append(str(bridge_classpath))
        classpath = os.pathsep.join(classpath_entries)
        compiled = subprocess.run(
            [javac, "-d", classes, "-cp", classpath, *(str(source) for source in sources)],
            cwd=AGENT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", shell=False, **no_console,
        )
        if compiled.returncode:
            print(f"compile rc={compiled.returncode} FAIL\n{compiled.stdout or ''}")
            return 1

        for name, main_class, test_args in test_cases(
            case_root,
            actual_bridge=bridge_classpath is not None,
        ):
            result = subprocess.run(
                [java, "-cp", classpath, main_class, *test_args], cwd=AGENT_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", shell=False, **no_console,
            )
            output = result.stdout or ""
            found = verdict(output)
            if result.returncode or found is None or found.startswith("FAIL"):
                status, failed = "FAIL", failed + 1
            elif found.startswith("SKIP"):
                status, skipped = "SKIP", skipped + 1
            else:
                status, passed = "PASS", passed + 1
            print(f"{name:<34} rc={result.returncode:<3} verdict={found or '<no verdict line>'} status={status}")
            if status == "FAIL":
                for line in output.splitlines()[-12:]:
                    print(f"    {line}")

    total = passed + skipped + failed
    print(f"\nsummary: {passed} passed, {skipped} skipped, {failed} failed ({total} tests)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
