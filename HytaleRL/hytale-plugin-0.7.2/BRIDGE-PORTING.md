# Porting the bridge to a new Hytale server version

Runbook for the Java bridge in this folder. The 0.7.2 port is the worked example: [PORT-0.7.2.md](PORT-0.7.2.md)
(what changed and why), [FIDELITY-0.7.2.md](FIDELITY-0.7.2.md) (measurements by section; its index at the top is generated) and
[CRAFTING-ORACLE.md](CRAFTING-ORACLE.md) (the crafting probe contract). The simulator side is the "Version bump
checklist" in `HytaleRL/hytalegym/hytalegym/sim/PORTING.md`. Each rule names its source. Items marked
(inferred) are not confirmed in the repository.

Rule: copy the previous plugin, retarget only what the server API forces, and record each validation in the
port's own document (the model is PORT-0.7.2.md, "What 0.7.2 changed, and how the port maps it").

## Order

1. Copy the previous plugin folder. Keep packages, classes, the wire protocol and the evidence schemas (PORT-0.7.2.md).
2. Read the new server: its jar, then `javap -c -p` on each class you touch (section 1).
3. Retarget the compile dependency and build. Fix compile errors only where the API forces them (section 1).
4. Apply the wire, component, legality and evidence rules (sections 2 to 5).
5. Probe live with a private port pair, one launch per scenario (section 7).
6. Run the Java gates (section 8), then pin the jar (section 9).

## 1. Server jar, bytecode and build

- The installed server is `~/Library/Application Support/Hytale/install/release/package/game/latest/`, which holds
  `Server/HytaleServer.jar` and `Assets.zip` (`tools/launch_bridge_072.py`, `DEFAULT_INSTALL`). The 0.7.2 server jar
  is 112,406,344 bytes with SHA-256 `23F71F03…` (ASSET-DIFF-0.7.2.md line 13). Confirmed: `shasum -a 256` on the
  installed jar starts `23f71f03`.
- Read bytecode with `javap -c -p <class>` on that jar. FIDELITY-0.7.2.md (q) checks `InteractionAdmissionDryRun`
  this way: no `putfield`, no `putstatic`, and no `cancelChains`, `setServerState` or `sendCancelPacket` call.
- `build.gradle.kts` depends on `com.hypixel.hytale:Server:<version>` (Maven) and uses a Java 25 toolchain. To compile
  against an installed jar instead, run `./gradlew compileJava -PhytaleServerJar=<absolute path>`; it changes the
  compile classpath only. The jar task sets `isPreserveFileTimestamps = false` and `isReproducibleFileOrder = true`.
- `./gradlew jar --offline` writes `build/libs/HytaleRLBridge072-0.1.0.jar` (`archiveBaseName` in `build.gradle.kts`).

## 2. Wire numbering

- `NativeInteractionTypes.WIRE_TYPES` keeps the 0.5 numbering at indices 0 to 24. Wire messages, evidence and masks
  use it. 0.7.2 inserted `Ability4` at 5 and shifted `Use` through `GameModeSwap` up by one in the server enum; the
  bridge does not follow that order. It appends `Ability4` (25), `OnBreak` (26) and `OnBreakImpact` (27), so the wire
  table has 28 entries on 0.7.2 and 25 on 0.5 (the class header comment). The trigger capacity is
  `NativeItemInteractionEvidence.TRIGGER_CAPACITY` (28 on 0.7.2).
- A new constant travels by name. The crossbow reload was requested as `Ability4` by name, and
  `InteractionType.valueOf` resolves it (commit 9f12f0d, which first kept the 25-entry table). The Python
  `INTERACTION_TYPE_NAMES` and `NATIVE_ITEM_TRIGGER_CAPACITY` (`hytalegym/worldgen/native_interactions/`) are keyed by
  `server_profile().certified_version`: 25 and 25 on 0.5, 28 and 28 on 0.7.2. Gen-5 contract shas and rollout
  fingerprints do not move.

## 3. Components a new server adds to players

- The headless player needs `InventoryComponent.AbilitySlots`. `NativeHeadlessWorldVerbContext.java`, lines 104 to 122,
  adds it with 6 slots (the default of `PlayerSystems.PlayerInitSystem`). Without it, `Interaction.tickInternal` throws
  on an interrupted reload and the world stops (GEN7-PLAN.md phase 4c).
- On each new server, compare what the player init systems add to a real player with what the headless player gets.
  The list holds one entry so far (inferred to be incomplete).

## 4. Legality

- Legality comes from `nativebackend/support/InteractionAdmissionDryRun.java`, a side-effect-free replay of the
  engine's admission. It falls back to `InteractionManager.canRun` with one warning when the replay method is absent.
- `canRun` refuses Primary during the Ability4 crossbow reload, which the engine admits at the Repeat window
  (commit 6fd60a2; FIDELITY-0.7.2.md (p)). Use the replay for every legality check.

## 5. Agent levers and evidence rows

- Two levers. `camera_delta_yaw` feeds `desiredYaw`, which sets the head (`HeadRotation`). `body_delta_yaw` feeds
  `desiredBodyYaw`, which sets the body (`NativePolicyActorState.applyLookDelta` and `applyBodyYawDelta`;
  `NativeEnvironmentSession.java` lines 2971 to 2976, 3124 and 3131).
- A camera-only look leaves the body facing away, by design. A test that needs the agent to face a target steers
  `body_delta_yaw` (GEN7-PLAN.md phases 4c and 4f; `AgentAction.java` parses both keys).
- The actor evidence row is additive. Policy rows carry `head_yaw_degrees` (HeadRotation) beside `yaw_degrees`, 24 keys.
  NPC trace actor rows keep 23 keys, which `native_npc_traces.py` validates exactly (`ObservationCodec.java` lines 272
  to 300; commit 6d914c8). Adding a field left the schema version and the actor contract unchanged (FIDELITY-0.7.2.md (n)).

## 6. Size ratchet

- `src/test/java/com/hytalerlbridge/SourceFileSizeRatchetTest.java` caps `NativeEnvironmentSession.java` at 8,600 lines
  (lowered from 9,700 on 2026-10-18 after the split below). Other files get 1,500. The file is 8,439 lines (`wc -l`,
  2026-10-18), down from 9,691.
- The split lives in `nativebackend/session/` (package `com.hytalerlbridge.nativebackend.session`), beside the existing
  `region/`, `navigation/` and `info/` packages. Phase 4p moved 36 methods, 5 constants and 1 record into nine classes in
  `session/` and `region/RegionLightBuilder` (1,753 lines in total): PrivilegedSnapshotBuilders 251, PerceptionBuilders 206,
  WorldgenEvidenceBuilders 212, MutableBlockBuilders 342, SnapshotBuilders 180, InventoryReaders 116, SpawnEquipment 135,
  FixtureBuilders 144, PolicyCombatKeys 117, RegionLightBuilder 50.
- Rule for a move: the body stays byte-identical; the session fields it reads come first as parameters with the same names;
  the session calls `Class.method(state..., args...)`; a moved method that calls another moved method passes the state
  through. Classes in `session/` cannot see `nativebackend` package-private types (PendingStep, WorldVerbTelemetry, the
  trace systems), so a method that depends on them stays in the session. Not moved on purpose: the tick loop, the step
  state machine, the ability request gate and busy gate, the capability capture beside the guard fork decision, the spawn
  path around the headless Player removal, the reset sequence around the crafting oracle, and every method that writes a
  session field.
- Gate for a split: compile, the full suite unchanged (421 tests on 2026-10-18), the ratchet, then a live smoke
  (`tools/smoke_bridge_072.py`), the six-arrow reload test and a hold probe on a scratch jar before the pin moves. The live
  runs exercise reset, steps and the ability path, not every moved capture builder; those moves rest on body identity and
  the unit tests.

## 7. Fixtures, probes and the live protocol

- Fixtures are reset options (`environment/EnvironmentOptions.java`). `hit_interrupt_duel` (line 49) makes the target
  HOSTILE to the agent with a server attitude override, `overrideAttitude(agentRef, Attitude.HOSTILE, 1.0e6)`
  (`NativeEnvironmentSession.java` lines 9506 to 9509, marked "Phase 4b fixture"). Keep that override fixture-only.
- Crafting oracle probes run at reset (`crafting_oracle_requests`) through the legacy craft path, in scope
  `oracle_unbound_*`. The production step path and its evidence gates are unchanged, so the probes are the route
  until the policy capture offers bench contexts (CRAFTING-ORACLE.md, "Oracle route").
- Launch a private server:
  `python3 tools/launch_bridge_072.py --run-dir <new empty directory> --bridge-port <P> --game-port <G> [--jar <jar>] [--then "<command>"]`.
  The run directory must be outside the install tree (the server exits 11 otherwise) and new or empty. The launcher
  binds both ports to check they are free, waits for the boot and bridge-listening markers, runs `--then` with
  `HYTALERL_BRIDGE_PORT`, `HYTALERL_GAME_PORT` and `HYTALERL_SERVER_PID` set, stops the server (`stop` on stdin, then
  SIGTERM, then SIGKILL) and writes `launch-report.json` in the run directory.
- Ports: one port pair per agent or session, never another session's. The launcher defaults are 5556 and 25572, but the
  game ports 25570 to 25573 belong to another session's render bridge (5570), so always pass `--game-port`: this
  session's pairs are 5556/25574, 5558/25576, 5560/25578 and 5562/25580 (phase 5 ran 23 launches on 5556/25574; commit
  274aaec used 5558/25576; 9f12f0d used 5556/25572 before the reservation). The launcher does not retry a busy port. A port in TIME_WAIT is expected to fail its bind probe
  (inferred from the probe code, not run here), so choose another pair.
- One launch per scenario when native state persists across resets (commit 76839c5).
- Receipts stay outside the repository, in the run directory or a job directory. Name the receipt path in the
  FIDELITY section.

## 8. Gates

1. The touched test classes first, with Gradle's `--tests` filter (standard Gradle syntax; not run in this session).
2. `SourceFileSizeRatchetTest`, which has 2 tests.
3. The full suite: `./gradlew test --offline --rerun --no-build-cache` (FIDELITY-0.7.2.md section (m) records this form).
   Read the counts from `build/test-results/test/TEST-*.xml` by summing the `tests`, `failures` and `errors`
   attributes. Commit a534e83 reports 116 suites, 414 tests, 0 failures.
4. Python: `python -m pytest HytaleRL/hytalegym/tests/test_server_versions.py -q` (7 tests in the 0.7.2 commits).

`src/test` is gitignored (`**/test/` and `*Test.java` in `HytaleRL/.gitignore`), so the unit tests exist on disk only.
Say so in the commit message, as the 0.7.2 commits do.

## 9. Pin

- `hytalegym/hytalegym/server_versions/vN_M.py` holds `bridge_jar_sha256`: the SHA-256 (upper case) of the bridge jar
  in `build/libs/`.
- `python3 tools/pin_bridge.py --check` compares the jar with the working-tree pin and notes a difference from HEAD's pin.
  Exit 0 when equal; exit 1 prints both hashes.
- `python3 tools/pin_bridge.py --build` runs `./gradlew jar --offline --no-build-cache`, rewrites only the pin line,
  runs `tests/test_server_versions.py` and prints the old and new hash. Run it only for the final build. The reason is
  in the plugin README, "Pin and build".
- The native reset compares the hash the live bridge reports (`bridge_sha256` in the reset info) with the pin
  (`worldgen/region/stability.py`, `require_current_native_runtime_bridge`). A pin that moves ahead of a jar that other
  sessions run makes their live runs fail.
