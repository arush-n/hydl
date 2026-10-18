# Server 0.7.2 port

This folder is a copy of [`../jvm-agent`](../jvm-agent) (the 0.5.x policy agent,
including its uncommitted working-tree edits as of 2026-10-16; build outputs
and `skills/` were not copied) retargeted to Hytale Server 0.7.2. Packages,
classes, the wire protocol and the bridge contract are unchanged; only what the
0.7.2 API forces was changed. No file outside this folder was modified.

## Build and run

- Server: `com.hypixel.hytale:Server:0.7.2` as cached by Gradle at
  `~/.gradle/caches/modules-2/files-2.1/com.hypixel.hytale/Server/0.7.2/9ef07dd…/Server-0.7.2.jar`,
  byte-identical to the launcher's `HytaleServer.jar` (SHA-256 `23f71f03…`).
- Compile: no build file exists in this folder, so
  `javac -d <out> -cp <Server-0.7.2.jar> $(find src -name '*.java')`.
- Jar: `jar --create --file HytalePolicyAgent072-0.1.0.jar -C <out> . -C mod manifest.json`.
  `manifest.json` (`ServerVersion "=0.7.2"`) sits at the jar root. The 072 name
  keeps it apart from the 0.5.x jar on disk; `adk/deploy/isolated.py` takes the
  jar by path (`--policy-jar`, with `--bridge-jar`, `--server-jar`, `--assets`
  for 0.7.2) and installs it under its own runtime name.
- Tests: `python3 run_tests.py --server-jar <Server-0.7.2.jar> --bridge-classpath <HytaleRLBridge072-0.1.0.jar>`.
  The bridge classpath enables the `--actual` ABI checks. `run_tests.sh`'s
  default `JAR` now points at 0.7.2, but its `CP="out;$JAR"` uses the Windows
  separator, so it cannot run on macOS as written. That line is unchanged from
  the 0.5 tree.
- Live: `tools/launch_bridge_072.py` (in `../HytaleRL/hytale-plugin-0.7.2`) copies
  only its `--jar` into `mods/`. To boot with the agent, add the agent jar to
  `<run-dir>/mods` after that copy, for example through a wrapper around the
  launcher's `shutil.copy2`. The run directory must be new and outside the install.

## Validation (2026-10-16)

| Check | Result |
|---|---|
| Compile against Server 0.7.2 | 0 errors, 0 warnings (21 errors before the port) |
| Unit suite, original tree (baseline, Server 0.5.10 + 0.5 bridge, run on a scratch copy before any change) | 46 passed, 0 failed (46 tests) |
| Unit suite, port (Server 0.7.2 + 0.7.2 bridge) | 46 passed, 0 failed (46 tests); same 46 names in the same order |
| Bridge ABI checks, `--actual` (0.7.2 bridge jar) | `BridgeLifecycleAdapter`, `BridgeWorldVerbAdapter`, `BridgeCombatAdapter`, `BridgeDodgeContract` all pass |
| Reflective member names (`getMethod`/`getField` literals in `src`, `tests`, `harness`) | every name is a member of some class in the 0.7.2 bridge jar (`javap -p` over all 283 classes, nested included) |
| Live boot (0.7.2 + 0.7.2 bridge + agent, ports 5641 and 25651) | plugin enabled, bridge listening on 127.0.0.1:5641, clean stop (exit 0); the agent's only WARN is `PolicyAgent inert: no weights directory`, expected without a bundle; 0 SEVERE from the agent |
| SEVERE lines in the live boot | 13, all `[SERR] Reallocate: 131072 to 1179648` during asset loading, before the agent loads. A bridge-only control boot logged the same lines (14), so they are engine-side |

Not exercised: the agent is inert without a policy bundle, so no chunk pinning,
block candidates, combat facade, LOS, crafting catalog or fixture spawn ran on a
live 0.7.2 world. The unit suite covers pure logic and the ABI checks cover the
bridge's reflective surface. The live-path changes below are unverified.
A live run needs a bundle with a live profile, which only exports on a CUDA
GPU (`adk/deploy/bundle.py:754`). On 2026-10-17 a diagnostic bundle without one
failed to arm identically on 0.5.10 (original agent) and 0.7.2 (this copy).

## What 0.7.2 changed, and how the port maps it

| Compile error (0.7.2) | Fix |
|---|---|
| `HeadlessTickAnchor` `writePacket` returns `boolean` (2 errors) | returns `true`, as the bridge's `HeadlessPacketHandler` does; the anchor still discards every write |
| `World.getChunk(long)` gone (`PolicyAgentPlugin` ×2, `HeadlessCombatFixture` ×2) | `policy/world/NativeWorldAccess.getChunk` (new; same helper as the bridge's `nativebackend/support/NativeWorldAccess`, trimmed to the calls the agent makes) |
| `ChunkFlag.TICKING` removed (`PolicyAgentPlugin` ×2) | `NativeWorldAccess.Column.setTicking(true)`; the column's `resetActiveTimer` and `addKeepLoaded` still compile on `WorldChunk` |
| `World.getBlock` / `getBlockType` gone (`ServerBlockCandidateReader` ×5) | `NativeWorldAccess.getBlock` / `getBlockType` (world-thread hop as in the bridge) |
| `Role.getCombatSupport()` gone (`ServerCombatSupportFacade`) | `store.getComponent(actor, CombatSupport.getComponentType())`; `clearOverrides` and `combatSupport` take the actor ref and store |
| `Role.getMarkedEntitySupport()` gone (`ServerActorReader`, `HeadlessCombatFixture`) | `store.getComponent(ref, MarkedEntitySupport.getComponentType())` on the NPC's own ref |
| `Role.getPositionCache()` gone (`ServerActorReader`) | `store.getComponent(self, PositionCache.getComponentType())` and `PositionCache.hasLineOfSight(ref, target, store)` |
| `Role.setMarkedTarget(String, Ref)` replaced by `(ref, accessor, slot, target)` (`HeadlessCombatFixture`, `PublicServerEvidenceSource`) | passes the NPC's ref and the store |
| `ISpawnProvider.getSpawnPoint` removed (`ServerCombatFixturePlacement`) | `NativeWorldAccess.waitForFutureWithoutLock(world, getSpawnPointAsync(world, uuid))`: drains the world task queue on the world thread instead of blocking it; joins elsewhere. The fixture spawn runs under `current.execute`, so this path is the world-thread one |

## Runtime hazards

| ID | Finding | Status |
|---|---|---|
| H1 | `InteractionType.Use.getValue()` is 6 in 0.7.2 (`Ability4(5)`, `Use(6)`, decompile `protocol/InteractionType.java`); the wire index is 5 | fixed: `ServerBlockCandidateReader.USE_WIRE_INDEX = 5`, the bridge's `NativeInteractionTypes` numbering |
| H2 | recipe pin 1947; 0.7.2 ships 2047 | fixed: `ServerRecipeCandidateReader.NATIVE_RECIPE_CAPACITY = 2_047`. The bridge measured 2047 of 2047 live (PORT); the agent's reader was not run live |
| H3 | place counting: `DragPlaceBlockInteraction` extends `SimpleInteraction`, so a count by `PlaceBlockInteraction` misses the drag node | no change. `InteractionManager.walkInteraction` calls `collect` on every node it visits, and `DragPlaceBlockInteraction.walk` descends into its fork root, so `Place_Single` is counted once through the walk, as 0.5's chain counted it. Confirmed by decompile; the count on a live block chain is not measured |
| H4 | camera target ray and line-of-sight rules changed in 0.7.2 | not changed; behaviour differs as the bridge's PORT describes (LOS passes non-transparent blocks unless a detail hitbox is hit) |
| H5 | `PositionCache.hasLineOfSight` caches per target and clears only at the role's tick (`PositionCache.java:779`); the agent calls it from `perceptible` outside the role tick | not changed (the spec asks only for what 0.7.2 forces); the bridge reads LOS through a pooled `ExecutionSupport`. Possible one-tick staleness is INFERRED, not measured |
| H6 | the velocity call | no change. The 3-argument `setVelocity` delegates to the 4-argument form with pre-multiplier `1.0F` (`MotionControllerBase.java:1241-1248`). `SplitVelocity` is enabled on the live server (`server.log`, 04:40), which sets `SHOULD_MODIFY_VELOCITY = false`, so the call takes the zero-then-add path, as 0.5 did |
| H7 | `World.isTicking` / `setTicking` in `PolicyAgentPlugin` compile | unchanged; the 0.7.2 semantics are INFERRED to match, not checked |
| H8 | legacy `Inventory` calls (`PolicyActorLoadout`, `ServerInventoryReader`) | compile; 0.7.2 marks the API for removal; unchanged |
| H9 | reflective names | no mismatch (see Validation) |
| H10 | `getSpawnPoint` was synchronous | replaced as above; a join on the world thread would deadlock, so it drains the queue instead |

## Behaviour that differs from 0.5.x

- The server rules the bridge's PORT lists (LOS, camera ray, block health,
  explosion reactions, crafting clamps) apply to the agent's reads unchanged.
  See `../hytale-plugin-0.7.2/PORT-0.7.2.md`.
- Ticking is the absence of `NonTicking` on a column and each of its sections.
  `setTicking` syncs the sections, as 0.5's `ChunkSystems.OnNonTicking` did.

## What is left

- Live policy episodes: needs a policy bundle under `mods/policy-agent` (the
  agent is inert without one). Nothing in this folder was run that way.
- Deployment: the bridge target tuple in `bridge-target.txt` and the policy bundle
  pins belong to `adk/deploy` (outside this folder). A 0.7.2 bridge needs its own
  tuple and a regenerated bundle.
- The legacy `Inventory` path (H8) and H5's LOS caching, both above.
- `run_tests.sh` `CP` separator (pre-existing, unchanged).
- Comments that still say 0.5.7 (history and rule notes) are unchanged. The
  `Server-0.5.7.jar` reference in `ReflectiveBridgeCombatLifecycleEvidenceSource`
  is updated.
