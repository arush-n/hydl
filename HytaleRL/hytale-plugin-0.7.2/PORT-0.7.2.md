# Server 0.7.2 port

This folder is a copy of [`../hytale-plugin`](../hytale-plugin) (the 0.5.x bridge,
including its uncommitted working-tree edits as of 2026-10-16) retargeted to
Hytale Server 0.7.2. Packages, classes, the wire protocol and the evidence
schemas are unchanged; only what the 0.7.2 API forces was changed. The bridge
port changed nothing outside this folder. Running training and evaluation on
0.7.2 (the Python version switch, the policy-agent copy, and the gameplay
changes that matter for training) is in [`TRAINING-0.7.2.md`](TRAINING-0.7.2.md).

## Build and run

- Dependency: Maven `com.hypixel.hytale:Server:0.7.2`, byte-identical to the
  launcher's `install/release/.../Server/HytaleServer.jar` (SHA-256 `23f71f03…`).
- `./gradlew jar` → `build/libs/HytaleRLBridge072-0.1.0.jar` (renamed so it cannot
  be confused with the 0.5.x `HytaleRLBridge-0.1.0.jar`; the plugin identity in
  `manifest.json` is still `HytaleRL:HytaleRLBridge`, `ServerVersion "=0.7.2"`).
  The 0.5.9-only `manifest-0.5.9.json` variant was dropped.
- `./gradlew test` runs the copied suite against the 0.7.2 runtime. The test JVM
  sets `java.util.logging.manager` because 0.7.2's `HytaleLogger` requires it.
- `tools/launch_bridge_072.py` starts a 0.7.2 server with the jar in a scratch
  directory (0.7.2 refuses to run from inside the launcher's install tree), and
  `tools/smoke_bridge_072.py` drives a native session from `hytalegym`.

## Validation (2026-10-16)

| Check | Result |
|---|---|
| Compile against Server 0.7.2 | 0 errors (197 before the port) |
| Unit tests | 383/383, 111 suites (the 0.5.x tree: 383/383 on Server 0.5.10) |
| Live boot on 0.7.2 | plugin loads and enables, bridge listens, clean stop |
| Python end-to-end (`HytaleSurvive-v0`, `backend="headless"`) | config → ack, reset → full observation (`native_server_version` 0.7.2), 20 steps, no bridge errors |
| Same smoke, existing bridge on Server 0.5.10 | same behaviour, including the hotbar crash below |
| Side-by-side episodes (scripted 80 steps and random 40 steps, twice per server) | same wire trace, actions, rewards, terminations, dtypes and shapes; value differences are version labels, 0.7.2 content or run-to-run variation (below) |
| Live feature sweep: 77 features in 4 families (region and perception, world verbs and crafting, combat and NPC traces, probes and fixtures), same requests on 0.5.10 + existing jar and 0.7.2 + port | 44 same, 15 differ by 0.7.2 content or rules, 6 pre-existing on both, 3 port-bug findings (2 distinct bugs: world-verb place and crafting count, fixed below), 7 inconclusive (dodge first-tick speed, synthetic-client sync counts, a change-counter baseline, dive physics with a randomized NPC scale, light readiness, one lava cell, task timing), 1 not exercised (typed bench and pocket crafting), 1 harness error rerun clean |
| Asset limits the bridge pins | 0.7.2 content fits: detail boxes per hitbox max 9 of 9 (`Boat`, `Bed_Royal`, as on 0.5.10), trigger programs at most 53 of 64, trigger boxes 4 of 9 |
| Fix checks on the final jar (0.7.2 and 0.5.10 side by side) | world-verb place puts Rock_Stone at +x=1 and uses the held stack; crafting catalog returns 2047 of 2047; drag flag false for seeds and true for blocks, as on 0.5.10 |
| Original 0.5.x tree | byte-identical to the pre-port snapshot (457 files) |
| Repo publication-hygiene gate | passes |

Pre-existing, not a port change: with random actions, choosing hotbar slot ≥ 5
crashes the server's interaction tick on the default `Trork_Unarmed` (hotbar 5)
and removes the NPC, on 0.5.10 and 0.7.2 alike. The bridge accepts slots 0–8
(the Python action space is `Discrete(9)`).

## What 0.7.2 changed, and how the port maps it

| 0.7.2 change | Port |
|---|---|
| `World` no longer implements `ChunkAccessor`/`IWorldChunks`; `WorldChunk`/`BlockChunk` lost their block getters; block, fluid, light, environment and support data live on per-section components, the heightmap on a per-column component | `nativebackend/support/NativeWorldAccess` re-implements each 0.5.10 accessor the bridge used, with its 0.5.10 behaviour (synchronous load, ticking rules, out-of-range defaults); `NativeWorldAccess.Column` stands in for a 0.5 `WorldChunk` + `BlockChunk`. Reads hop to the world thread when called elsewhere, since 0.7.2 component reads assert it |
| `ChunkFlag.TICKING` removed; ticking is the absence of `NonTicking`, on columns and sections; 0.5's `ChunkSystems.OnNonTicking` (column marker copied to every section) is gone | `Column.isTicking/setTicking`; `setTicking` and `getChunk` sync the column's sections, reproducing 0.5's propagation |
| `HeightmapColumn.NO_HEIGHT` (`Integer.MIN_VALUE`) for empty columns | mapped to 0, the 0.5.10 value |
| NPC `Role` supports (`CombatSupport`, `StateSupport`, `PositionCache`, `EntitySupport`, `MarkedEntitySupport`, `WorldSupport`) became ECS components; some APIs take a pooled `ExecutionSupport` | component lookups on the NPC's own ref; `acquireExecutionSupport` + `clearForReuse()` in `finally` |
| `Role.setMarkedTarget`, velocity instructions and several other Role methods take the entity ref/accessor or a pre-multiplier | updated call sites; velocity pre-multiplier `1.0F`, as 0.7.2's `ApplyForceInteraction` queues it |
| Block health moved from per-column `BlockHealthChunk` (keyed by position) to per-section `BlockHealthSection` (keyed by block index) | section lookups; the last-damage time is read reflectively from the section's private map (no public accessor) |
| `InteractionType` renumbered: `Ability4` inserted at 5 (shifting `Use`..`GameModeSwap` up by one), `OnBreak`/`OnBreakImpact` appended | `nativebackend/support/NativeInteractionTypes` keeps the 0.5 numbering (indices 0 to 24) that wire messages and masks use. `Ability4`, `OnBreak` and `OnBreakImpact` are appended at 25 to 27, so no existing index moves; the trigger capacity is 28 on 0.7.2 (25 on 0.5). The server's own order (`Ability4` at 5) was rejected because it would renumber `Use`..`GameModeSwap` on the wire |
| `Damage.EnvironmentSource.getType()` removed | the one environment source (explosions) is recorded as `explosion`, the 0.5 value |
| `ExplosionConfig.detonateExplosiveBlocks` (new, default true) also opens the block-target loop | the bridge's controlled explosion configs set it false, restoring 0.5's gate |
| `Interaction.cancelOnItemChange` became `onItemChangeBehavior` (Cancel/Fail/Finish/Ignore) | evidence reports `true` only for `Cancel`, the one value that skips the interaction body |
| Block placement moved into a gesture wrapper: `Block_Secondary` → `PlaceModeSelect` → `Place_Drag` (`DragPlaceBlockInteraction`), which forks `Place_Single` once per block the client reports in `forkCounts`; the per-node `AllowDragPlacement` flag was removed | both synthetic clients (the session's World verbs and `NativePolicyWorldVerbFacade`) report one fork, feed the forked chain the same block data, finish the parent chain, and keep the World run open until the fork ends (a failed fork fails the run, as a failed 0.5 PlaceBlock did); shared helpers in `nativebackend/support/InteractionSupport`. The facade's place path is reached only by the jvm-agent (the policy capture offers no place candidates on either version), so only the session path was run live. Evidence `allow_drag_placement` is true exactly for place nodes forked by a `DragPlaceBlockInteraction`: blocks yes, seeds no, as in 0.5 |
| 0.7.2 ships 2047 crafting recipes (0.5.x: 1947) | `NativeCraftingCatalogEvidence.RECIPE_CAPACITY` is 2047, measured on a live 0.7.2 server |
| `PacketHandler.writePacket` returns `boolean`; explosion, interaction and door APIs changed | ported in place; see the code comments at each site |
| Region chunk contract | `ChunkApiContract`/`NativeRegionManifest` now certify the 0.7 line (the shape checks are unchanged and resolve to the same values) |

0.7.2 marks the legacy `Inventory` API (hotbar, armor, active slot) for removal.
The port still uses it, which shows as compile warnings; a later server may need
that path moved to the inventory components.

## Behaviour that differs from 0.5.x

Chosen in the port:

- Synthetic-client NPC forces go through 0.7.2's own `Role` velocity methods (the
  bridge's documented intent), so 0.7.2's knockback scaling and single
  set-velocity apply where 0.5.10 zeroed then added.
- Mutable-block fixture "repaired" row: 0.5.10's `repairBlock` dropped the damage
  entry unless repaired health exceeded 1.0 (checked in bytecode), so the row
  read 1.0; the port clears the entry to reproduce that for the fixture's values.

Server rules that changed in 0.7.2 (the bridge observes them; evidence recorded
on 0.5 will differ):

- Line of sight (`PositionCache`) passes non-transparent blocks unless the ray
  hits one of their detail hitboxes.
- Block health: an entry is dropped once health leaves (0, 1), and any block
  write clears the cell's health; no block-damage packets are sent on clear.
- `BlockOperations.setBlock` also runs block-health and block-animation
  replacement hooks and always invalidates lighting.
- The camera target ray no longer stops at y 0..319.
- Crafting: timed bench crafts use the previous tier's time reduction; requested
  quantities are clamped; recipes may require bench augment tags (the bridge's
  bench pre-check does not include them, so the server can reject a craft the
  bridge accepted).
- An actor carrying a block resolves interactions from the carried container.
- Explosion-reaction blocks react to explosions and end the target ray.
- Clearing marked targets also drops their rebind state.
- Section and column ticking expire on separate timers when no player is near.

Seen in the live comparison; the cause is 0.7.2 content or randomness, not port code:

- Mana starts at 100 (0.7.2 `Mana.json`; 0.5.10: 0).
- Block and item runtime ids are renumbered (Rock_Stone 1351 → 1530); asset
  names and the region identity hashes match.
- Attack windows, re-attack times and duel event order vary from run to run on
  both servers: the attack pause is drawn from unseeded `ThreadLocalRandom`
  (Trork range 1–2 s in both asset sets).
- The first step after reset takes 34–68 world ticks (0.5.10: 30); after that
  both run 4 ticks per step. Cause not isolated.
- The first move after landing covers about one tick less ground. The NPC
  motion controller moves the agent, and the port leaves that path unchanged.
- Region light: 188–201 of 250 sections are lit at the first poll (0.5.10: 3–5).
- Region content differs in 46 of 250 sections (tree and decoration worldgen,
  block `MovementSettings` such as `Soil_Mud` and spider webs).
- Item interaction graphs gain nodes from content (`PlaceModeSelect` branches,
  `Block_Break`'s new `Next`).
- World-thread bridge tasks are slower (median 57 ms vs 45 ms over about 1,300
  requests); not profiled.

Pre-existing on 0.5.10 and 0.7.2 alike (not changed): besides the hotbar crash
above, `mutable_block_cells` during a held break can fail with
`ArithmeticException: long overflow` (`Duration.toNanos` in
`RegionGeometryCapture.mutableBlockRow`).

## Outside this folder

Done (see `TRAINING-0.7.2.md`): `hytalegym/hytalegym/server_versions/` keys the
bridge hash, region release line, crafting recipe count, built bridge path and
native-eval checkpoint rule by server version; `experimental/jvm-agent-0.7.2`
is the policy agent's 0.7.2 copy.

Still 0.5-only, by design: the simulator certifications (World source-pin
audits, the 0.5.7 `Assets.zip` hash, the `hytale_0_5_7` rulesets, the JAX
crafting table at 1947 rows) and Python fixtures that assert `0.5.7`. They
need a 0.7.2 calibration of the simulator, not a version switch.

`../hytale-plugin` no longer builds from a clean Gradle cache: Maven dropped
`Server:0.5.7` (it now lists 0.5.10, 0.6.9, 0.7.0–0.7.2).
