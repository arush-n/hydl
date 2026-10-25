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
and removes the NPC, on 0.5.10 and 0.7.2 alike. The Python action space offers
slots 0–8 (`Discrete(9)`, the player's hotbar). Since 2026-10-21 (#122) the 0.7.2
bridge ignores a slot at or past the role's own hotbar capacity, so the active
slot stays where it was; the 0.5.10 bridge still accepts 0–8.

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

## Held charge on the learner path (`charge_held`, 2026-10-22)

`AgentAction` carries two additive keys: `charge_held` (boolean, default false)
and `charge_hold_seconds` (finite, nonnegative float, default 0). `continuation()`
keeps both, as it keeps `guardHeld`. The Python learner sends both on each tick
it requests a melee hold slot: an authored Charging table with rows, no
`AllowIndefiniteHold`, and no authored outer request (sword thrust, battleaxe
swings, mace charges). `charge_hold_seconds` is that table's top threshold. The
outer `requested_charge_time` stays 0 for these slots.

Bridge semantics (`NativeEnvironmentSession.requestNativeAbility`, `NativeChargeHold`):

- The first held request queues the slot's root with `charge_hold_seconds` as
  its charge and binds the synthetic client with the same value. The charge
  fires by itself at that threshold, as the simulator's auto-fire does.
- A held tick on the same slot, including continuation ticks, never re-queues
  and never resets the charge (the refusal path in `NativeInteractionQueue`
  would apply zero).
- The first tick without the level, or with another slot, releases the charge
  once. The synthetic client delivers the release, which sets the charge to
  zero, so a hold that ends before the top takes the base branch. The release
  value is zero, not the elapsed hold: the sword's and battleaxe's outer 0.2 s
  gates would route an elapsed release into the charged branch.
- While the level is set, the held slot stays legal in the advertised ability
  mask (`Ability.hostLegal`); every other slot keeps its refusal.

Why no indefinite hold: `InteractionManager` skips `simulationTick` in
remote-client mode, so `ChargingInteraction.simulateTick0`'s own auto-fire at
`highestChargeValue` never runs, and the synthetic cursor finishes a charge only
at its requested seconds. An infinite cursor would never reach the thrust, and
a release would swing. Bytecode: Server-0.7.2.jar,
`InteractionManager` (`hasRemoteClient` gates) and
`ChargingInteraction.simulateTick0`.

Not covered: `AllowIndefiniteHold` tables (staff, wand, spellbook, knife, spear,
guns) and slots with an authored outer request (flame staff) keep the
requested-charge path. Live (jar `3865F45F`, 2026-10-24, receipts
`gen7_probes/thrust_lunge/thrust_lunge_Kweebec_Razorleaf_jar14_{edge,hold10,hold30}.json`): a 30-tick
hold on slot 3 thrusts (lunge 2.78 blocks from r+21, every held tick legal); a 10-tick hold and a
one-tick press swing with no lunge.

Held timing (fix 1, 2026-10-23): a held charge (`NativeSyntheticCombatClient.markChargeHeld`, called
from `requestNativeAbility` when `held`) makes each Charging op finish at its own highest threshold
(`chargedNext` max key on the protocol packet: 0.174 for the swing op, 0.565 for the thrust op) and
send that value, capped by the requested hold time, so a release still finishes at 0. The thrust
then fires near 0.739 s (about 22 ticks at 30 Hz). Without this, each op waited the full hold (0.85 s)
and a 30-tick hold released before the thrust (see the 2026-10-23 live receipt). The learner keeps a
held slot legal in the evidence's `ability_legal` (`NativeActorEvidenceAssembler._held_charge_open`)
while the bridge binds the slot (`host_legal`, which includes `NativeChargeHold.holds`): the busy
charging chain closes the slot in the arsenal projection, and in the simulator a held charge admits
nothing until it fires, so the slot stays legal there. Fix 3 reopened only the policy head mask; the
decoder reads the observation's own ability mask, so the third held tick was refused and sent as a
no-op, which released the charge into a swing (2026-10-24 fix3 hold30 receipt: legal ticks 11-12,
refused from 13).

## Host contracts the 0.7.2 bridge pins (2026-10-19)

The bridge refuses a session when a host contract it pins differs from the value the Python host
computes. The 0.7.2 values below are computed with `HYTALERL_SERVER_VERSION=0.7.2` (full values in the
Java source); the 0.5.7 column is what the default line still computes.

| Host contract | Bridge pin (`src/main/java/com/hytalerlbridge/...`) | 0.7.2 host value | 0.5.7 host value |
|---|---|---|---|
| Learner-v3 actor evidence (reset negotiation, frame echo) | `observation/NativeActorEvidenceFrame.CONTRACT_SHA256` | `E93BBEBE…` (changed) | `D47E87B2…` |
| NPC observation contract | `imitation/NpcObservationSnapshot.CONTRACT_SHA256` | `3D080DE9…` (changed) | `1C53B327…` |
| NPC trace action / event / lifecycle | `imitation/NpcTraceBatch.*_CONTRACT_SHA256` (computed) | `DE0A63AD…` / `7065A3BF…` / `5D6D0C26…` | same |
| NPC inventory frame | `observation/NativeInventoryFrame.CONTRACT_SHA256` | `CC393B3E…` | same |
| Policy world-action capture | `worldgen/policyactions/capture/model/NativePolicyWorldActionCapture.CONTRACT_SHA256` | `5E6672AD…` | same |
| World-verb transport | `action/NativeWorldVerbRequest.CONTRACT_SHA256` | `72AA4849…` | same |
| World-verb lifecycle | `nativebackend/WorldVerbTelemetry.CONTRACT_SHA256` | `99E83A86…` | same |
| Synthetic world interaction | `nativebackend/SyntheticWorldInteractionEvidence.CONTRACT_SHA256` | `F41C1386…` | same |
| Falling-block trace | `nativebackend/NativeFallingBlockTrace.CONTRACT_SHA256` | `AB46D1D6…` | same |
| Group action (computed) | `action/group/NativeGroupActionContract.sha256()` | `BB94B65F…` | same |
| Block affordance dictionary | `worldgen/BlockAffordanceContract.DICTIONARY_SHA256` (already the 0.7.2 value) | `a09b9441…` | `4f2104af…` |
| Chunk API shape (not a hash; runtime `ChunkUtil`: BITS 5, SIZE 32, HEIGHT 320, chunk range -2^26..2^26-1) | `worldgen/ChunkApiContract` | equal (javap on the 0.7.2 server jar) | equal |

The first two rows changed on 2026-10-19. The block-affordance row also differs by line, but its Java pin
already held the 0.7.2 value. The Python pins that differ by line are keyed by
`server_profile().certified_version`: `hytalegym/worldgen/native_npc_traces.py` (the actor-evidence pin;
the observation contract follows it) and `tests/jax/combat/observation/v3/native/test_native_actor_evidence.py`
(the Java source it reads). The block-affordance pin was already keyed.

Rule: the 0.7.2 plugin pins the 0.7.2 host values. A Java pin that the host compares is set to the value the
host computes under 0.7.2, and the 0.5.7 value stays in a comment beside it with the date. The 0.5 plugin
(`../hytale-plugin`) is not changed. The bridge's own jar hash (`bridge_jar_sha256` in
`hytalegym/hytalegym/server_versions/v0_7.py`) is a separate pin and is re-pinned after the final build.

## Crafting oracle: granted augment tags and the augment verdict (2026-10-20)

The oracle reads the bench's granted augment tags and reports, for each `craft_recipe` at a bench, the augment
verdict. The record schema is `hytalerl_crafting_oracle_v2`. The Python reader
(`hytalegym/worldgen/native_world_actions.py`) reads v2 and still accepts v1 records, which carry no augment fields.

Wire:

- Reset info: `native_crafting_bench_augment_tags`, a list of strings, sorted. It sits beside the other
  `native_crafting_bench_*` keys and is read after `BenchBlock.refreshGrantedAugmentTags`, the refresh the native
  check runs before it reads the set.
- Record: `bench.augment_tags` (the same list, read when the request begins) and `augment`, which is
  `{"admitted": bool, "required_tags": [...], "missing_tags": [...]}` or `null` when the request names no bench.
  The verdict covers the recipe's requirements that match the bench's type, id and tier, because the native check
  reaches the augment clause only for those. It is admitted when one matching requirement has every tag granted,
  and then reports that requirement. Otherwise it reports the first matching requirement and the tags the bench
  does not grant.

Preflight: `NativeHeadlessCrafting.matchesBenchRequirements` gained the augment clause, so a recipe whose required
tags the bench does not grant is refused before any inventory work, with `craft_recipe_not_available_at_bench`. The
native manager refused the same request as `native_crafting_manager_rejected`: `isValidBenchForRecipe` runs before any
input is removed, in `queueCraft` and in `craftItem`. The outcome is the same (refused, no inventory change);
only the reason changes, on the production craft path too. The granted tags are read only when a matching
requirement names tags.

Server API (0.7.2 `HytaleServer.jar`, javap): `protocol.BenchRequirement.requiredAugmentTags` (String[]);
`BenchBlock.getGrantedAugmentTags()` (Set<String>); `BenchBlock.refreshGrantedAugmentTags(Store, Ref, Vector3i)`
(static, delegates to a refresher that `AugmentBlocksPlugin.setup()` installs). `CraftingManager`'s
`isValidBenchForRecipe`, `benchHasRequiredAugmentTags` and `getBenchGrantedAugmentTags` are private, so the bridge
mirrors them.

Open: the refresher is set by a builtin plugin's setup. If it were absent, the refresh would throw in the readout
and in the native check for a tagged recipe alike. The live run confirms it is set. The bench block for the
GoblinLab workbench is `GoblinLab_Tinkering_Bench` (its `Bench.Id` is `GoblinLab_Workbench`), from the asset
file name; the live run confirms the block id.

The bridge jar changes with this build, so `bridge_jar_sha256` in `hytalegym/hytalegym/server_versions/v0_7.py`
is re-pinned after the final build.

## Block-type table request and async loader (2026-10-23)

`block_type_table` (no environment needed; handled without `requireEnvironment`)
returns `{type, schema: "hytalerl_block_type_table_v1", version: 1, count,
entries: [[runtime_id, asset_id], ...]}` for every `BlockType` asset. Runtime ids
are `BlockType.getAssetMap().getIndex(assetId)`, the same ids `nearby_blocks`
rows carry (`column.getBlock`; `NativeWorldAccess` sets blocks with `getIndex`).
Codec: `network/codec/BlockTypeTableCodec.java`, dispatch in `ClientHandler`.

Python: `hytalegym/envs/block_type_table.py`. `BlockTypeIdLoader` starts a daemon
thread after `HytaleEnv.reset()` when the env is built with `block_type_ids=True`.
The fetch uses a new `BridgeConnection`, never the env socket, and the result is
cached on the class keyed by (host, port). Lookups never block: before the table
arrives they return 0. Errors are kept in `loader.error` and never raised.

Opt-in info keys (absent when the flag is off): `nearby_block_type_id` int32
[9, 9, 9] indexed `[dx+4, dy+4, dz+4]`, the same flattening as the
`nearby_blocks` array, so the window matches `nearby_blocks` cell for cell; air
is 0; `block_type_ids_ready` bool; `nearby_block_type_unknown_count` counts
nonzero runtime ids missing from the table.

Caveat: `geometry/GeometryFrame.java` maps runtime ids through the plugin's own
enum `world/BlockType.fromId`, not the asset map. This request does not change
that path.

## Live 3-D sounds (probe) and sound-event table (2026-10-24)

0.7.2 sends `PlaySoundEvent3D` (footsteps among them) only to `Player` entities,
so a headless bridge hears none of them. An early-plugin transformer adds
`public static volatile Consumer hytalerl$soundProbe` to `SoundUtil` and calls
it on every `playSoundEvent3d` (Revision 1 in the live-hearing design: the
server's TransformingClassLoader cannot see early-plugin classes, so the hook is
JDK-typed). `nativebackend/NativeSoundProbeBridge.java` owns the bounded ring
(256, drop oldest, `dropped()` counts) and the owner read from
`MovementAudioComponent$ShouldHearPredicate.owner`.

- Attach: at each `reset`, the bridge sets the field to its own consumer. If the
  field is absent, the probe is absent and nothing changes. `close()` detaches,
  clearing the field only while it still holds this bridge's consumer.
- Per native tick, before the tick's simulated time advances, the bridge drains
  the ring. It keeps each sound within `SoundEvent.getMaxDistance()` of the
  learner's `TransformComponent` and maps the owner `Ref` to a slot: 0 learner,
  1 target, -1 other. Refs have no `equals` override (javap), so they match by
  store and index.
- Per tick, not per step: the event age is then measured the same way as the
  packet capture (`step.simulatedSeconds` at the tick).
- Wire v2, only when the probe is attached: `capture_mask` `[true, true, true]`,
  `version` 2, and eleven columns per event with `source_slot` last. See
  `protocol/messages.md` ("Audio v2"). Probe absent: v1, ten columns, the same
  bytes as before (a golden test pins them).
- `sound_event_table` (no environment needed): `{type, schema:
  "hytalerl_sound_event_table_v1", version: 1, count, entries: [[index,
  asset_id], ...]}`, index = `SoundEvent.getAssetMap().getIndex(assetId)`.
  Codec: `network/codec/SoundEventTableCodec.java`.

## Fixture blocks reset option (`fixture_blocks`, 2026-10-24)

Opt-in reset option that places named blocks before the first observation, for live
probes such as a body walking through `Plant_Bush` or `Deco_SpiderWeb_Flat` (their
MoveIn/MoveOut sounds feed the hearing pipeline). `options["fixture_blocks"]` is a
list of `[x, y, z, "AssetId"]`: integer cells, `|x|`, `|z|` <= 32, `60 <= y <= 80`,
at most 64 entries, non-blank asset ids. Absent or empty changes nothing. Parse
(`network/codec/FixtureBlocksCodec.java`) and the record's compact constructor
(`environment/EnvironmentOptions.java`) refuse a bad reset; an id that is not a block
type fails the reset in `NativeEnvironmentSession.applyFixtureBlocks()`, which runs
after `createFidelityFixture()` and before the warmup. Python checks the shape in
`HytaleEnv.reset` (`_validated_fixture_blocks`) before connecting.

Nothing is cleared on a later reset: each reset builds a new world (`createWorld`,
a fresh UUID, empty chunk storage) and `ClientHandler` destroys the previous
environment first, so a placed cell cannot carry over.

Jar 16 (`49CDB408`, pinned 2026-10-24; jar 15 `626B10D2` plus this option). Live with the sound probe
(`gen7_probes/hearing_footstep_live_candidate49CDB408.json` and `..._approach.json`): the learner running into a
line of `Plant_Bush` hears its own `SFX_Bush_MoveIn` on entering the first bush cell (source slot 0, 0.13 blocks);
running backward through a band of bushes with the active Trork chasing, its own MoveIn arrives at tick 8 (slot 0) and
the Trork's at tick 61 (slot 1, 3.77 blocks), both in `heard_sounds` with those sources. Impact sounds carry no
footstep predicate, so their source stays -1.

## Block damage age without overflow (jar 17, 2026-10-25)

`RegionGeometryCapture.mutableBlockRow` reports each mutable cell's seconds since its last damage. It converted
`Duration.between(lastDamage, gameTime)` with `toNanos()`, which throws `ArithmeticException: long overflow` past about
292 years. A server `BlockHealth()` starts stamped `Instant.EPOCH` (1970, read from the 0.7.2 class) and a fresh world's
game time starts in year 1 (`GameTime: 0001-01-01T05:30:00Z` in `server.log`), so once a block was partly broken the
`policy_world_action_capture` request failed (jar 16, `gen7_probes/break_neighbours_live_jar16_hits.json`: both seeds).
Which stamp the failing cell carried was not read; any stamp after year 293 overflows the same way.
`damageAgeSeconds` returns 0 for a negative gap before converting. That is what `Math.max(0, ...)` gave wherever the
old code did not throw, so every other value is unchanged (`RegionGeometryCaptureDamageAgeTest`, bit-for-bit).

Jar 17 (`F4AA8EBA`, pinned 2026-10-25; jar 16 plus this fix; Java 489 -> 491 passed). Live
(`gen7_probes/break_neighbours_live_candidateF4AA8EBA.json`, seeds 570057 and 913771): no overflow; the first pickaxe
hit leaves the `Rock_Stone` cell standing, it reads `Air` from tick 32, the second hit finishes at tick 39, and the
`Soil_Pebbles` beside and above it stay.

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
