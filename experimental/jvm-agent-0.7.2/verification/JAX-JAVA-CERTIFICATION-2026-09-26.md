# JVM-agent local conformance record - 2026-09-26

**Result: PASS for local JAX-to-Java conformance.** This is not a live
sim-to-server behavior certificate or a release attestation.

## Evidence

| Gate | Result |
| --- | --- |
| JVM-agent suite against pinned Server 0.5.7 (SHA-256 `43D9BCFF1DD31574577DBFC82147718DBE2AC16C19000071F965B521BA808CDC`) | 45 passed, 0 skipped, 0 failed |
| JVM-agent suite against active Server 0.5.9 (SHA-256 `AB54FAB93699AC34D353F97E927E6DB121471ABB0D1BD69D43A4ECE0BA828B8C`) | 46 passed, 0 skipped, 0 failed |
| HytaleRL bridge Gradle suite against active Server 0.5.9 (same SHA-256) | 383 passed across 111 suites; 0 skipped, 0 failed, 0 errors |
| JDK | Temurin 25.0.4+7 |

The pinned-0.5.7 run had 45 named checks. The latest run adds a bounded-trace
regression check and passes 46/46 against the active 0.5.9 server JAR. The suite
covers JAX-derived observation and action parity, projection, flat and region
profiles, policy inference, recurrence, concurrency, trace capture, and bridge
adapter contracts. The bridge Gradle suite was forced to recompile and rerun;
compilation succeeded with 94 deprecation/removal warnings from Hytale API use.

An interim full JVM-agent gate was **43/45**, with `ProjectionTest` and
`CombatBindingProfile` failing. The projection fixture lacked light/geometry
groups, and the Shortbow fixture had stale authored-slot metadata. Both tests
now pass in the final gate.

Reproduce from the repository root:

```powershell
python experimental/jvm-agent/run_tests.py --server-jar "<Hytale Server JAR>" --bridge-classpath "HytaleRL/hytale-plugin/build/classes/java/main" --jdk-bin "<JDK 25 bin>"
```

For the active server artifact, pass its JAR to Gradle explicitly. From
`HytaleRL/hytale-plugin`:

```powershell
.\gradlew.bat -PhytaleServerJar="<Hytale Server JAR>" test --rerun-tasks --no-daemon --console=plain
```

The JVM-agent suite uses generated fixtures. Their producer and regeneration
instructions are in [`tools/export/DEV.md`](../tools/export/DEV.md).

## Fixes covered by this run

- Java self-observation now uses head yaw and pitch, matching the JAX inputs.
- Java defense projection clamps normalized applied-force speed to the JAX
  `[0, 1]` range.
- Profile export preserves each authored ability slot and its charge duration;
  the Shortbow fixture now contains all nine authored bindings and all five
increasing charge tiers.
- Full projection export includes the light and geometry groups consumed by the
  Java projection contract tests.
- The bounded policy trace caps nonterminal rows, retains at most one terminal
  row, and prevents later callbacks from appending rows after terminal evidence.

## Active-server artifact compatibility check

The latest offline gates were rerun against the server JAR named in the active
server process's startup command: **46/46 JVM-agent checks** and **383/383
bridge tests** passed. The on-disk server JAR has SHA-256
`AB54FAB93699AC34D353F97E927E6DB121471ABB0D1BD69D43A4ECE0BA828B8C`; this
extends source/API compatibility evidence beyond the pinned 0.5.7 JAR. It
does not prove which bridge classes the running process loaded.

A read-only process check found the running server started on 2026-09-07 at
12:10:26 and points to the bridge `build/libs` directory. The bridge JAR
on disk is SHA-256
`7700D112F8BCE9A6AFFE5EEB765867E83138F67070FA82408D08E5AB948F8D7B`, last
modified at 12:10:02 that day. Because the process was not restarted or
redeployed, changes made to bridge source after startup are not verified as
loaded by it.

To reproduce the active-artifact checks, pass the active server JAR to
`run_tests.py` and to Gradle's `-PhytaleServerJar` property. These checks prove
source/API compatibility only; they do not exercise the code currently loaded
in the running process.

## Certification boundary

The latest result proves the tested Java source matches the included
JAX-generated fixtures and passes the bridge project's offline tests against
the 0.5.9 JAR identified above; the earlier pinned 0.5.7 suite is recorded
separately. It does **not** prove that a policy works on a live authoritative
server. The current bridge receipt source has not been loaded into the active
server, and this run did not start or modify a game server. Live Place/Break
candidate evidence, the public-player boundary, and an authoritative policy
transfer trace remain open; see the [sim-to-server MVP plan](../../../HytaleRL/docs/SIM-TO-SERVER-MVP-PLAN.md).

The separate v30 live pursuit run was rechecked with the correct training-stage
report. The gate still exits 1 and issues no behavior certificate. Its recorded
bridge hash matches the archived run-local JAR, while the current build has a
different hash. The policy died at a minimum target distance of 1.1538 blocks,
before the 1.0-block catch condition; the run records 554 policy ticks against
the declared 512-tick limit, and its trace has no terminal row. The gate lacks
native declarations for the objective, target controller, catch distance,
target context, horizon, and perception mode; it also reports accommodation
and exact artifact-snapshot mismatches. The 0.5.7 and 0.5.9 snapshots are within
the same 0.5 release line; this is an exact-content alignment gap, not a
major-version port. The detailed status is in
[`POLICY-PORT-STATUS.md`](../../../HytaleRL/POLICY-PORT-STATUS.md).

The observation contract is structured state, not RGB/video perception, and
the NPC locomotion path remains a surrogate for player movement. Throughput,
fidelity, and transfer have not yet been measured together in a live-server
evaluation. Therefore no full sim-to-server behavior certificate is issued.
