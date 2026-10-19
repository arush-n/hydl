# Gen 7 and the generation-agnostic simulator: the plan (2026-10-17)

Owner rulings this plan follows: agents train on Hytale 0.7.2 ("gen 7"); 0.5.7 is deprecated and its
recorded results must stay reproducible; the gen-7 simulator sits in its own clearly named folder; and
"the JAX sim should be generation agnostic, so when future generations release we can easily update
everything." The phase gates and the measured 0.7.2 changes of `HytaleRL/hytale-plugin-0.7.2/
SIMULATOR-0.7.2.md`, `TRAINING-0.7.2.md` and `ASSET-DIFF-0.7.2.md` remain the evidence base; this plan
replaces that document's structure with the engine-and-profiles target.

## 1. The target in one paragraph

One JAX simulator engine. A generation is a **profile** (`sim/genN/profile.py`: version, data
directory, construct switches and their parameters), its **content tables** (`sim/genN/data/`,
extracted from that version's `Assets.zip` with provenance) and its **constructs** (mechanics that
exist only in some versions, implemented once in the engine behind a profile field). The engine never
holds a game number; it reads tables. The engine never branches on a version string; it branches on a
profile field whose value a construct owns. Updating to a new release is: add a profile, run the
extractors, read the diff against the release notes, enable or add constructs, run the differentials
against a live server of that version, compute the stamp, retrain, flip the default.

## 2. What is engine, what is generation

| Concern | Engine (`jax/combat/` migrating in place, `sim/common/`) | Generation (`sim/genN/`) |
|---|---|---|
| Numbers: speeds, stats, weapon tables, projectiles, consumables, durability, block properties, recipes, NPC roles | the schema, the loader, typed records, the content sha | the extracted tables |
| Behaviours that differ between versions: LOS rule, dodge directions, knockback model, interrupt and flinch, mana and runes, hookshot, climbing, doors, swimming, gliding, riding | one construct per behaviour behind a profile field | the field's value; constructs that exist nowhere else yet live under `genN/mechanics/` until a second generation shares them |
| World: Region library, path tables, traversal, biome table, suites | the capture and build tools | the captures made on that version |
| Contracts | the stamp machinery | `hytale_version`, ruleset sha, content sha |
| Verification | the differential harness, the source-pin audit framework | the baselines and receipts recorded on that version's jar and assets |

The decision rule: a value that changes when the game updates is content; a behaviour that appears,
disappears or changes shape is a construct behind a switch; everything else is engine.

## 2b. Render, do not replace (owner, 2026-10-17, late)

Owner: "we should not replace the existing code rather make copies or some rendering that updates it
such that the new gen is properly implemented ... optimizing where we can and in the future try to make
updating to a newer generation easier and quicker." So the mechanism that makes the engine generation-
agnostic is a **renderer**: the gen-5 source stays untouched (frozen), and each generation's simulator
package is *rendered* from that source plus the generation's **overlay**:
- the overlay names the modules to copy, the constants and tables to redirect to the content layer (by
  symbol, from the swap plan of phase 1), the construct hooks to insert (profile-gated), and any patch
  the fidelity report requires (for 0.7.2: the detail-hitbox line-of-sight rule, slope and wall limits,
  the guard rule, knockback values, hit timing, interrupt / flinch, mana and runes, the crafting table
  capacity, the dodge head size);
- `sim/common/render.py` applies an overlay to the committed source (read from `git show HEAD:`, never
  from the working tree, so no uncommitted work is baked in) and writes `sim/genN/engine/`;
- the **identity gate**: rendering the gen-5 source with the gen-5 overlay (content tables and no
  constructs) must give byte-identical rollouts on the fixed seeds; the gen-7 render then differs only
  where its overlay says, and each difference has a differential;
- updating to generation N+1 is: extract, write the overlay delta, render, run the differentials. The
  overlay delta is the whole port, in one reviewable file.
Optimisation is a first-class requirement of the port (owner, 2026-10-17: "when porting you should also be
optimizing and trying to make it faster from the terrain worlds to other accuracy checks"), measured, not
assumed, and never at the cost of bit-exactness unless the owner sees the numbers:
- **baselines first**, on the committed gen-5 engine: step throughput (lane-steps per second at batch 8
  and 64), jit compile time of the combat step, Region library load time, path-table build time, the
  runtime of each fidelity differential and of the eleven source-pin audits, and the test suites' wall
  time; recorded as receipts so every render reports its delta;
- **targets in the render step**: content tables as static arrays, vectorised replacements for
  Python-side loops the copied modules expose, cached Region and light captures, incremental path tables
  (rebuild only the changed sections), differentials that run against a recorded native receipt instead
  of a live server where the mechanic has not changed, and audits that hash once per jar;
- **the rule from the speed-ups memory**: measure first; only bit-exact opt-in changes while other
  sessions run; a methodology change goes to the owner with the numbers.

**Status (2026-10-17, late; phase 2a done, `sim/common/RENDER.md`).** The renderer exists: `sim/common/overlay.py`
(the overlay spec: modules, redirects by symbol / expression / line with casts and a status, profile-gated
construct hooks, exact patches each tied to a fidelity row and a differential, validation before any write),
`sim/common/render.py` (reads the pinned commit with `git cat-file --batch`, relocates, redirects, hooks,
patches, checks that everything compiles and nothing imports `hytalegym.jax`, writes atomically with a
MANIFEST) and `sim/common/bench.py` (identity workloads, bench, profiler). The **identity gate passes bit for
bit**: gen-5 source + gen-5 overlay against a `git archive` of the same commit, 11 outputs, 2,307 arrays,
101,298 elements, 0 mismatched, equal SHA-256; a negative control (base speed 5.5 -> 5.75) is caught (82
arrays differ); the content agent's rollout fingerprints (`sim/gen5/identity.json`) are reproduced through
the render; original and render compile to identical HLO for 4 programs; the existing combat suite is
unchanged before and after (69 failed / 1000 passed / 1 skipped / 15 xfailed both times, none of the
failures the port's). The rendered engine is a build artifact: `sim/<gen>/engine/` is gitignored and
re-rendered on demand. **Gen-7 skeleton** (`sim/gen7/overlay.py`): renders and runs on the 1/30 s clock; 24
rows applied (13 step-clock redirects to `SimProfile.step_duration`; 11 patches from the fidelity report:
5b knockback 6/7/12 -> 4.3/5.0/10.8 with the upward component 1 -> 0.2, 5c dagger swings 4 -> 6, 5d guard
roots no longer interrupt Primary and the sword thrust gains InterruptedBy Secondary, 1d slope 30 degrees /
wall 15 degrees parameters, 11 crafting capacity 1,947 -> 2,047), 90 pending (85 renderable redirects from
`redirects.json`, 5 unreachable, 5 hook stubs 5a / 7a / 7b / 2b / 9). Not applied, and why: the dodge head
size indexes four directions in decode, encode, masks and codec (now hook `dodge_action_head`); shield bash
StaminaRegenDelay Set -> Min needs a construct; dagger materials live outside the render; sword thrust timing
is row 5f. **No 0.7.2 fidelity is claimed: no differential has run.** Bench (original vs render, HLO
identical, timing deltas noise under load 28-39): `sim/BASELINES.md` section 5. One opt-in bit-exact speed-up
(`sim/gen5/overlay.py` `OPTIMIZED`: the damage-resistance walk, thunk events 6,718 -> 4,200 per step at
batch 64, +6 % throughput); four more hot spots proposed, not implemented. A methodology question for the
owner: the planned (production) compile and the general compile are not bit-identical (6 of 315 arrays, max
2.4e-7 over 64 steps), so moving callers to the planned path is a decision, not a free speed-up. Ruled 2026-10-18:
acceptable (the difference is far below the native differential bars); gates compare like with like and a mode
change re-baselines the fingerprints (`sim/BASELINES.md` section 5).

**Phase 3 status (2026-10-18; receipts `~/.claude/jobs/31947bd7/tmp/phase3/receipts/`, ledger `FIDELITY-0.7.2.md`
section g).** Activated: 16 swap-plan rows (15 `PLAYER_*` locomotion constants from `movement.json`, 1 sprint
multiplier 1.273); gen-5 and gen-7 tables agree on all 63 shared MovementConfig keys, so the rendered literals
equal HEAD in value and type, and only `MaxSlopeAngleDegrees` 30 and `MaxWallAngleDegrees` 15 are new. Render: 41
rows applied (13 clock, 16 locomotion, 11 patches, 1 hook), 73 pending. Line of sight, hook 2b
(`Constructs.line_of_sight = detail_hitbox`, `sim/gen7/mechanics/line_of_sight.py`): per opaque cell with detail
boxes the ray must meet a valid box; full cubes reduce to the gen-5 slab test; gen 5 returns the original function;
300-ray comparison with a float64 gen-5 reference, 0 mismatches. Differentials against a live 0.7.2 bridge
(`tools/launch_bridge_072.py`, port 5556, no process killed): motion on the flat floor with a ledge, max position
error 1.14e-05 m against the 5e-4 bar, PASS; geometry compare, all eight stages PASS; LOS on full cubes 3 of 3
fixtures match native. Locomotion audit at the 0.7.2 baseline recorded (14 class files and 6 assets drift from
0.5.7; pins unchanged). Identity gate on gen 5 still bit-identical (4 passed, 24 min); tests/sim 86 passed.
**Not met, with the reason:** slope and wall limits have no reader anywhere in the 0.7.2 server (a scan of 38,436
classes finds them only in MovementConfig and the protocol; engine contact normals are axis-aligned), so the
construct is a no-op: decision, set `slope_limits` False and record that gen 7 matches gen 5 on this field, the
two parameters stay as rendered data; slope motion and partial-block LOS differentials need fixtures that do not
exist (a slope case in `NativeEnvironmentSession.java` changes the bridge jar and its pin); NPC jump: HEAD gives the
NPC learner `PLAYER_JUMP_FORCE` 11.8 while the native NPC first-step vertical speed is 5.099 (jump height 2.4 m
against the 5e-4 bar, on gen 5 and gen 7 alike): decision by the ADR's rule that Hytale is right where the two
disagree, gen 7 adopts the native NPC jump force behind its profile and gen 5 stays frozen; the production
perception LOS (`jax/world/perception/los.py`) is not yet hooked, only `world/geometry` is; `walk_descent`'s
premise is unverified; Region recapture not started (it writes shared world artifacts: it goes ahead into
gen-7-specific artifact paths, never over the gen-5 ones). Two defects for the content layer: the swap plan's
sprint record names the dict key, not the value (the overlay re-targets it); `sim/gen5/data` is a 0.5.10
extraction while `gen5/profile.py` certifies 0.5.7 (relabel, or re-extract from a 0.5.7 archive if one exists).

**Phase 3b and 4 status (2026-10-18; receipts `~/.claude/jobs/31947bd7/tmp/phase4/`, ledger `FIDELITY-0.7.2.md`
section h).** Renderer: anchors relocate onto `hytalegym.envs` (never the `hytalegym` package, which the repo-root
shim shadows under `python -m pytest` from HytaleRL/), with a test that both import routes find the artifact root;
this was the cause of the 25 "actor-evidence contract mismatch" aborts the lower layer measured (the contract was
right). Through the shim the gen-7 render now hashes D47E87B2 like the stock engine; the 23 aborting native compares
run again: 8 pass, 15 are measurements (14 learner-row observation mismatches of width 8272, 107 to 109 entries, max
abs error 1.0 in the world, resource and guard groups; one full-row float32 residual 7.45e-9 against an exact-zero
assertion), whose stock-engine attribution is the lower layer's baseline run. Constructs: `dagger_swing_damage`
(hook on the scalar variant profile; the 0.7.2 item rows for the swings): 15 of 15 materials equal the table and
match native first-contact damage live on two seeds (gen-5 control: bronze 5 vs 3, mithril 12 vs 8). `npc_jump`
(hook 1c, npc_walk only): first step exact at 5.0990195 m/s (sqrt(2 x 10 x 1.3), the server's measured value;
the 1.3 is inferred, the server shows JumpHeight 0.5 and MaxClimbHeight 1.3), but the arc still misses the 5e-4 bar
(0.025 m jump in place, 0.98 m jump then run): the residual fits native applying gravity with the previous tick's
duration on airborne steps 1 to 8 (F1c), an integrator change that would move every generation: owner decision,
row 1c stays MISMATCH. Production perception line of sight: a patch (the box test sits inside the ray walk, no hook
point), 3 of 3 captured frames match native on both renders. `slope_limits` False, recorded (no 0.7.2 reader).
Content: the swap plan's sprint record names the value (`redirects.py` writes `symbol_kind`; redirects.json
regenerated, one record changed; six text-named records still to check before activation); `gen5/profile.py`
relabelled to the 0.5.10 extraction it holds (no 0.5.7 archive exists on this machine). Not built: partial-box LOS
and slope fixtures (no discriminating geometry without a rotation API or an eye offset), walk_descent hook (premise
confirmed by javap, no differential), the profile-keyed surrogate LOS compare. **Phase 4 proper did not start**: hit
interrupt / stun / flinch need a bridge fixture with an eligible role and the attack state machine; knockback Set
vs Add needs a displacement harness; mana and runes need a mana readout the bridge does not publish; projectile
tiers cannot be measured while the shortbow registers no native hit (6a); the dodge head changes the learner
contract. Those bridge-side pieces are phase 4b. Gates: tests/sim 89 passed with the identity file ignored; the
identity gate 4 passed against 8707eec (36 min); against HEAD it failed to import until 6c0c33f (motion.py), and
passed against that HEAD as well (4 passed, 24 min, the coordinator's run): gen 5 renders bit-identical from the
repaired HEAD.


**Phase 4b status (2026-10-18; receipts `~/.claude/jobs/31947bd7/tmp/phase4b/`, ledger `FIDELITY-0.7.2.md` section j).**
Bridge: jar `8B09661F...AC740B` is pinned (reproducible from committed source); readouts for mana and NPC state names; the
`hit_interrupt_duel` fixture (mutual HOSTILE by a server attitude override on the target side). Gen 7: the mana pool
construct (`mechanics/mana_pool.py`, hook 7a active) matches native refill within 5 % (native 0.2982 s, JAX 0.3000 s,
bar 0.3 s); maximum and initial 100 read on the bridge. **Not met:** the melee interrupt (row 5a): two native goblin
club hits add no flinch, stun or recovery effect, so the construct is not built; the skeleton has no JAX profile
either. **Not built:** the knockback harness (5b; Set versus Add undecided), the runes (7b), the shortbow tiers (6a, no
run this phase; the tier-0 launch of 12 against 15 is a candidate cause, untested). The whole phase is in the
receipts; no fidelity is claimed beyond row 7a.

**Bridge defects on 0.7.2, measured by the fidelity session (2026-10-18; its receipts `fid/receipts/` in its scratchpad),
next for the bridge owner (phase 4c), none of them test bugs.** (1) A policy-bound group actor never turns: with jar
`8B09661F`, seed 570057, actor one's `combat_target_yaw_degrees` stays 0.00 through the face-each-other reset command
and six +30 degree `camera_delta_yaw` steps (37 steps), so it faces away; this one fault explains the dodge-overlap
control hit never landing, the group combat measuring 0 damage, and the guard pose staying at -180. The yaw code is the
same in both plugins, so 0.7.2 likely no longer drives an NPC body from that path (inference). Regression test, local to
that session: `tests/fidelity/combat/group/test_native_group_actor_turn.py`. (2) The crossbow reload interrupt stops the
world: `Interaction.tickInternal` (Interaction.java:534) throws "contextHeldItemContainer is null", the agent is removed,
the removal throws "Entity is already not in a world!" on the world thread, and the bridge reports "World thread is not
accepting tasks"; the fix is in how the bridge builds the held-item context for its headless player, or in the server.
Also open for 5f: the kunai +1 per throw may be measurement (native sets accepted one drain late,
NativeEnvironmentSession.java:5789-5797, and the test re-requests on finish); the sword +2 matches a player-backed
selector in JAX; probes P1 and P2 of that session decide before any 5f patch.

**Bridge status, phase 4c (2026-10-18; FIDELITY-0.7.2.md section (k)).** The headless context now gives the controlled
entity an `AbilitySlots` component (jar `8720BC76`), which removes the `Interaction.tickInternal` NPE and the world stop;
the slot-4 reload still fails because the bridge requests `Ability3` while the 0.7.2 crossbow binds that reload to
`Ability4` (owner decision: remap or known failure). Policy-bound yaw follows `body_delta_yaw` by design, on 0.5 as on
0.7.2, so the group facing tests need a lever decision before they can pass; defect (1) above is withdrawn as a bridge
fault.

**Bridge status, phase 4d (2026-10-18; FIDELITY-0.7.2.md section (m)).** The bridge accepts `Ability4` by name on the
request path (jar `32EF5EDD`), and the 0.7.2 crossbow reload runs to completion under it live, matching the Primary
reload's ammo transitions; the wire and evidence numbering stay 0.5's. Open: the gen-7 content binding of the reload row
as `Ability4` (peer session), and the interrupt case, where Primary is not offered during the reload (cause not isolated).

**Bridge status, phase 4e (2026-10-18; FIDELITY-0.7.2.md section (n)).** The actor evidence row carries
`head_yaw_degrees` (HeadRotation) beside the body `yaw_degrees`, and the decoder maps `agent_head_yaw` from it when
present (jar `B56FD38E`; schema and contract identities unchanged), so the native self row is the head frame the policy
trained on; the live learner-row look case passes on the new pin on both seeds (4 passed, every value and mask; it
failed at tick 1 with 13 values before).

**Construct status, the fidelity session's commits of 2026-10-18 (FIDELITY-0.7.2.md section (o)); sim/gen7, sim/common/profile.py
and tests/sim only, each passing tests/sim and the identity gate before committing.** `490b1d4` the 5d guard cancel window
(13 of 13 attacks match the 0.7.2 root RunTime; the thrust 3-6 is its own row); `69422d7` the 6c crossbow binding
(the render sends Primary, Ability1, Ability4); `e8a1fc7` F1c (the server's fall-speed update with the previous tick's
duration and cubic drag; jump in place, jump held, jump held with forward and ledge walk pass live on both seeds, stock
passes only the ledge walk; residual: native walks again only on the third tick after landing); `ff51431` and
`4661d5c` the 5f dodge (force on tick 3, stamina cost on tick 4, invulnerability and cooldown on the acceptance tick 2;
dodge_left passes every value and mask on the render, stock fails from tick 2); `6aaf21d` crafting stage 0 (the
bench-context commit: 197 of 197 offered recipes commit on the render against 2 on stock; gen 5 keeps its defect,
BASELINES.md section 6). tests/sim 102 -> 114. The learner rows on the final render pass 6 of 14 live on `B56FD38E`
(stock 4). Next there: crafting stage 1 (resource-type inputs with output exclusions, +66 recipes), then the stages of
section 2b above; bench types 1 to 3 need the bridge oracle first (this session's lane).

**Bridge status, phases 4f and 4g (2026-10-18; FIDELITY-0.7.2.md sections (p) and (q)).** The bridge's advertised
legality comes from a side-effect-free replay of the engine's admission instead of `canRun` (jar `A4487CF3`), so the
crossbow interrupt is offered from the Repeat window (tick 26) as the server admits it; guard and dodge legality share the
path (not yet re-measured live). Still open on the Python side: the accepted-edge execution child resolves from post-shot
ammo (the official interrupt test fails at line 342 on `execution` 4 versus 0) and the learner mask drops for one tick
when the reload adds an arrow. The combo hits land and are cancelled by the server before application (not a bridge
defect; filter not isolated); the signature cases were a harness ray on the body yaw (fixed in the fidelity session's
harness). Next for the bridge: the crafting oracle stages of CRAFTING-ORACLE.md.

**Phase 4h (2026-10-18; FIDELITY-0.7.2.md section (r)).** The Python evidence side now resolves an accepted root's child
from the pre-admission row and omits the reload's instant stat-modifier status, so the learner mask equals the wire's
legality on every tick and the official crossbow interrupt test passes with the `Ability4` binding; the Primary reload
test still passes. The crossbow row 6c is complete apart from the server-cancelled projectile damage (combo and
signature), the next bridge investigation.

**Crafting oracle, stages 1, 2 and 4 (2026-10-18; FIDELITY-0.7.2.md section (s); CRAFTING-ORACLE.md answers).** The bridge
places a bench at a tier, sets the memories level and runs craft probes at reset through an oracle-only scope (jar
`A6C310DD`); all ten contract scenarios match (structural crafts land one tick after the request, type-0 crafts on the
request tick, tier and memories refusals as the server rules say). The simulator side can now model bench tiers, the
memories level and the structural queue; stages 3 (timed) and 5 (processing, diagram) are next for the bridge.

**Crafting oracle, stage 3 (2026-10-18; FIDELITY-0.7.2.md section (t); jar `5FE609E2`).** Timed bench crafts are measured:
inputs leave on the first world tick after the request, the output lands when native progress reaches `TimeSeconds` (87 ticks
for 3.0 s), walking out of reach cancels with a refund, a second request queues behind, damage does not cancel. Stage 5 is
open: processing benches are container placement plus game time (the bridge world's clock is paused, so the bench never
advanced), the Armory is a click-craft; both need a different route than `craft_recipe`.

**Handoff of the simulator lane from the fidelity session (2026-10-18, via the owner).** Done on gen 7: the NPC jump arc (8
of 10 live runs pass, 2 of 10 on stock), daggers (15 of 15 live), guard rules and the guard-release choice, dodge timing,
the crossbow reload on `Ability4`, mana refill (0.298 s per point live against 0.300 s), line of sight on full blocks
(partial blocks unmeasured), sword knockback (values from the 0.7.2 files, never measured live), crafting at 1,168 of 2,047
recipes. Content values still on the gen-5 numbers: of 69 compared side by side, 50 are equal in both versions, 5 are
covered by existing patches, and 14 differ (13 applied as Redirects on 2026-10-18, FIDELITY-0.7.2.md section (w); the
potion values wait on the reader): sword timing (6 attack-window times and 4 hold times, for
example 0.85 to 0.739 s), shortbow damage 6/9/12/16 to 1/2/4/7 and launch speed 15/30/50/70 to 12/22/35/50, the iron
crossbow combo damage 27 to 13, and potions. Broken: the potion reader crashes on 0.7.2 (the stamina potion no longer applies
its 15 s cooldown effect, the simulator still does); weapon durability (57 maximum and 82 loss-per-hit changes across 143
weapons) lives in a data file the gen-7 build cannot override yet (both closed on 2026-10-18, FIDELITY-0.7.2.md section (y): the
reader reads each table's effect list and the 0.7.2 drink is a 30 s interaction cooldown; the durability table is served
through the `weapon_durability` hook). Open: timing that needs new state (walking resuming
after a landing, the 0.35 s wait before a new guard, melee and kunai attack start, the one-tick structural craft, shortbow
ammo at reset); mechanics that exist only as stubs (hit interrupt and flinch, runes, hookshot, climbing, gliding, riding,
player swimming, smaller rules); crafting bench tiers above 1, memories, timed and processing recipes on the simulator
side; recapturing the terrain library on 0.7.2 (all three saved libraries are 0.5.7 captures); the 11 source-pin audits
(0 of 11 pass on 0.7.2); gen-7 contracts and checkpoints. Dropped lead: the left and right sword swings' knockback mode
"Add" is not a 0.7.2 change (both versions say Add; gen 5 runs them as Set through an existing calibration). From here this
session owns both lanes and runs the simulator items in sequence (they share `overlay.py`) with the bridge items in
parallel.

**Phase 4j (2026-10-18; FIDELITY-0.7.2.md section (u); jar `CAFAD9B7`).** The crossbow damage that never reached the target
was the bridge's own doing: the world-verb fixture gave the Trork a headless Player and the PvP-off world cancelled player-on-
player damage. The target is an NPC again; the combo hits land (61 to 51 to 41 to 28) and the iron signature tests pass. Open:
the ancient-steel combo (no hits) and Big Arrow (82 applied against 105), and the combo test's third-contact expectation (27 is
the gen-5 value; 0.7.2 authors 13).

**Crafting oracle, stage 5 (2026-10-18; FIDELITY-0.7.2.md section (v); jar `4A5E5D6D`).** Processing benches advance on game
time once the probe runs the world and the clock unpaused (the stepped bridge world pauses both): Campfire fish 61.9 game
seconds, Furnace copper bar about 301 game seconds with three sticks and one charcoal; the Armory axe is a click-craft landing
two ticks after the request. The simulator side now has every crafting stage's native shape; the open bridge items are the
Tannery and Salvagebench, refusals and negative controls.

**Shield bash under a held guard (2026-10-18; FIDELITY-0.7.2.md section (x); jar `56D2B532`).** The bash is a fork of the
held guard Wielding on the server; the bridge now advertises it on every held-guard tick and a bash request while guarding
runs (accepted at ticks 4 and 6 live). The learner row's bash mismatch shrinks to tick 2 (JAX a tick early); its other
mismatches predate this. Open: the plain-root bash with no guard held, advertised by both layers though the asset makes it a
fork only.

**Timing constructs with state, item 1 (2026-10-18; FIDELITY-0.7.2.md section (z)).** After an accepted jump the gen-7 NPC walks
again on the third tick after touchdown (`landing_walk_resume`, a `CombatState` countdown leaf), matching native within
1.5e-5 m on two seeds where the unheld engine missed by 0.86 to 1.27 m; a landing without a jump is not held. Open: the guard
cooldown, the attack start split and the shortbow reset ammo (findings recorded in the section); the player motion model's
hold is unmeasured.

**Phase 4n (2026-10-18; FIDELITY-0.7.2.md section (aa)).** The remaining crossbow rows are not bridge defects: the ancient-steel
combo misses because the duel harness steers yaw only while the agent's head pitch is -44.7 degrees; the Big Arrow's 82 is the
authored 0.7.2 value (0.5.7: 123) and the iron combo's 13 likewise (0.5.7: 27). The harness pitch steer, the test expectations and
the ancient-steel values in the shared `weapon_variants` module are simulator-side items.

**Crafting on gen 7 from the oracle (2026-10-18; FIDELITY-0.7.2.md section (ab)).** Bench tiers (row 11b) and the memories level
(row 11m) are built as gen-7 constructs and replay all ten oracle scenarios; the render admits 1,264 recipes (1,235 commit alone;
1,168 at a fresh bench, 1,264 at tier 8 and level 6). Open: timed recipes (row 11j, progress in seconds at 1/30 s) and the
processing and diagram actions.

**Timed bench crafts on gen 7 (2026-10-18; FIDELITY-0.7.2.md section (ac)).** Row 11j is built as the construct
`timed_crafting` (`Constructs.crafting_time`: `instant` on gen 5, `timed_job` on gen 7): a timed recipe starts a job in seven
`CombatState` leaves, its inputs leave on the first progress tick, the output lands when the progress in seconds at 1/30 s reaches
the recipe's time (R+90 for 3.0 s; the native 87 to 89 after the first progress tick is tick jitter), a second request queues one
deep, more than 7 blocks from the bench cancels with the refund to the original slots. All seven stage-3 receipts replay; the
render admits 1,485 recipes (1,446 commit alone, 1,305 at a fresh bench; all 211 timed commits finish). Open: the processing and
diagram actions (stage 5, 447 timed rows need processing benches), a live Region-step run of the executor path, and the fieldcraft
timed recipe `Weapon_Axe_Crude` (no bench requirement; native timing unmeasured).

**Phase 5, the source-pin audits on 0.7.2 (2026-10-18; FIDELITY-0.7.2.md section (ad)).** 0 of 11 pass before and after; the
harness now fails closed on recorded 0.7.2 rows instead of crashing (absent-safe digests; pins keyed by the audited jar's
release line in `worldgen/source_pin_baselines.py` and `source_pin_baselines_072.json`; the 0.5 line byte-identical on the 0.5.10
control). Each audit's finding names the simulator work it implies (the table in (ad)); the terrain recapture stays gated on all
eleven. Next: re-derive the claims per audit or run a live differential, the inventory row first (the likeliest false alarm).

**Crafting oracle stage 5 on gen 7 (2026-10-18; FIDELITY-0.7.2.md section (ae)).** Processing benches (row 11p, construct
`processing_benches`: a request moves the inputs and a fuel stack into per-lane bench leaves and activates; progress and fuel
advance one recipe second per 30 ticks; the output and the inputs leave on completion; a charcoal on the second fuel item) and the
Armory click-craft (row 11q, construct `diagram_crafting`: queued on the request tick, committed two ticks later) replay all six
stage-5 receipts in order, fuel and seconds; the native tick counts come 2 to 15 ticks earlier under the loaded server's long
ticks (jitter, not modelled). The render admits 1,945 recipes (1,895 commit alone; 1,754 at a fresh bench); 437 of the 447
processing-only rows start and 433 finish in the count. Open: the live Region-step run of the executor paths (timed, processing,
diagram), the furnace's extra start tick, tiers 2 and up, fuel exhaustion and collection (inferred), the 17 timed Armory rows.

**Timing constructs, items 2 to 4 (2026-10-18; FIDELITY-0.7.2.md section (af)).** Measured with the learner-row harness on a
private bridge (24 guard runs, kunai and iron-sword captures, the kunai charge bank, the shortbow reset rows): the guard's entry
reopens 10 or 11 ticks after the claim on the loaded server, the kunai and iron sword claim on tick 1 and start on tick 2
(finishing on 11 and 14), the kunai charge returns 27 to 29 ticks after its claim, the shortbow's ammo is 0 on the reset row and
on tick 1 and 1 from tick 2. Built as three constructs behind switches with gen 5 unchanged: `guard_reentry_cooldown`
(0.35 s from the claim at 1/30 s, reopen on R+11), `ability_start_split` (the chain starts one drain after the claim; charges
spent and recovered on the claim, nominal 1.0 s), `shortbow_reset_ammo` (the reset row's ammo 0). Every capture ran on a
machine at load 50 to 60 with native ticks 5 to 9 percent long, so native timers finished in fewer ticks than the exact clock
predicts; recorded as load jitter, with the idle-machine re-measurement (#78) as the decisive check. The tick-0 head and
body yaw are both 0 for the scenarios' spawn yaw (#75: no construct).

**Phase 4o (2026-10-18; FIDELITY-0.7.2.md section (ag)).** Section (aa)'s cause is corrected: the ancient-steel combo missed
because the head yaw was pinned at the -45 degree edge of its window around a body that never turned (pitch about +2 degrees)
and because the harness read the compass as a world bearing while the bridge takes it body-relative. With the body steered
with the head, a body-relative compass and an aim gate on the target box, all eight live crossbow rows pass on the pinned jar
(6 failed and 2 passed before). The expectations follow the 0.7.2 table (13, 82) and the applied integer, since the kernel
already rounds a hit half up (kernel.py 320 to 322; no rounding construct). Construct `crossbow_variant`
(`Constructs.crossbow_scalar_variant`, `shared_rows` -> `gen7_table`) gives the ancient-steel triple (10.5, 14, 82) on the
gen-7 render; durability was already served by the durability construct. Open: native rounding beyond .5 values (none in
the catalogue), the effect-damage path's rounding (unmeasured).

**Phase 4p (2026-10-18; FIDELITY-0.7.2.md sections (ah) and (ai)).** The three crafting executors ran inside a live Region step
for the first time (`tests/sim/test_region_executor_crafting.py`, the production environment over a synthetic Region) and five
defects came out, three in the constructs: the timed hook read the wrong wrapper's globals (`unwrap`), the executor patch passed
the recipe rows where the bench scan needs the block candidates, and none of the three wrappers set `state_commit`, so a running
job's advance was discarded each step. The other two were harness facts (the Armory axe needs the archive's resource-type table;
two renders in one process bind through a stale parent attribute, now dropped by `support.unmount`). After the fixes the Region
step reproduces the replays (timed inputs out R+1, queue on R+11, output R+90; Campfire first progress R+2, output R+61; Armory
queue on R, commit R+2) and one live timed craft on the pinned jar finished at R+82, inside the recorded jitter. The shield bash
became a construct (`shield_bash_root`, Hook on `_ability_root_admission_mask`): the bash is legal and admissible only while the
guard is held and active, so the render offers it on ticks 3 to 12 of a hold as live 0.7.2 does; the stock mask offered it from
the held level (ticks 1 and 2 too), never as a plain root, which corrects the premise of row 5d. Open: a bash request on the
release tick 13 (admitted by stock and construct, native unmeasured); no render-side live learner-row run exists.

**Phase 4q (2026-10-18; FIDELITY-0.7.2.md section (ak)).** Head at reset, measured on live 0.7.2 (two seeds): a target spawned
facing the agent has its head on its body heading at tick 0 and keeps it with no input, while the simulator's reset zeroes both
heads whatever the body yaw (`combat/runtime/reset.py` 156 to 161). The native agent always spawns at yaw 0, so the defect shows
only for a non-zero `entity_yaw`, which the arena, multi-actor and self-play resets all pass. Construct designed, not yet wired:
`Constructs.reset_head_heading` (`none` -> `body_heading`), a Hook on `reset_arsenal_batch` that seeds both heads from
`entity_yaw` and is a no-op when none is given, so identity rows do not move. Next: a construct lane under the overlay lock.

## 2c. Time base: 30 Hz (owner, 2026-10-17, late)

Owner: "we will be operating on 30hz, the goal is that we are faster so that we can accommodate for the
difference in time manually instead of being behind." Consequences:
- The gen-7 engine steps at the game's tick, 1/30 s, everywhere: the per-step clock is one value from the
  profile, not the gen-5 mix of float32(1/30) and 0.045 s per lane; the belief's tick counters already
  count at 1/30 s (B5), so predicates such as the 40-tick damage window keep their meaning.
- Horizons, cadences and holds are restated in 30 Hz ticks: a 512-tick horizon is 17.1 s (was 23.0 s at
  0.045 s); the host cadence of 0.5 s is 15 ticks (was 11); a hold of 0.1 to 3 s is 3 to 90 ticks.
- The per-tick compute budget is a gate, not a wish: the whole per-tick path (belief update, the fast
  decision path, the combiner and the skill bodies) must finish well inside 33.3 ms of wall time on the
  deployment machine for the deployed batch, with margin, so the agent can pace itself to the game clock
  and absorb jitter rather than fall behind. Bar: p95 of the per-tick path <= 15 ms at the deployed batch
  (one NPC or one squad), measured with the bench of §2b; the middle layer's own share is 0.3 ms today.
- The live-server differentials and the NPC traces (one frame per 1/30 s) then need no resampling; the
  reset-timing probe already measures in engine ticks.

## 3. Getting there from today

Today the gen-5 numbers are hand-transcribed into the engine's code and 0.7.2 has no profile beyond a
stub. The migration, in order, each step gated:

1. **Content layer and extractors (phase 1, running).** Extractors per content family run on both
   versions; `sim/common/content/` loads a profile's tables into typed records; `sim/common/content/
   adapters.py` reproduces every constant and table the engine holds today from the gen-5 tables.
   Gates: A1, the 0.5.7 tables equal the code's values field by field (differences are transcription
   or reading errors, each listed); A2, the 0.5.7 to 0.7.2 diff reproduces the measured changes (sword
   knockback 6.0 to 4.3 and its direction, dagger +50 %, ranged damage cut 3-4x, mana 100 at +1 per
   0.3 s, durability, interrupt flags on 31 attacks). Output: the swap plan, one row per constant with
   file and line, and whether that file is clean or uncommitted.
2. **The engine reads the gen-5 profile (migration).** Each constant becomes a table read through the
   profile; each version-specific behaviour becomes a construct behind a switch, with the gen-5 value
   set to today's behaviour. Gate: byte-identical rollouts against the pre-migration code on fixed
   seeds for every existing combat test and a determinism fixture; the existing test suite unchanged.
   Then the pre-migration copy is retired; there is one engine. Blocker: this step edits files that
   other sessions hold uncommitted (`jax/combat/types.py`, `mechanics/runtime/kernel.py`,
   `motion.py`, `observation/v3/policy/layout.py`, among 53); it starts from a committed baseline,
   which needs the owner to have that work committed or claimed.
3. **Gen-7 profile and constructs.** `sim/gen7/profile.py` points at `sim/gen7/data/` and sets the
   switches; constructs new in 0.7.2 are implemented behind their fields: hit interrupt, stun and the
   0.5 s flinch on the flagged attacks; mana with the refill rate and the rune casts (fireball,
   wind strike, chain hook, ground slam, charged frost shot, enrage, poison); hookshot; the 0.7.2
   line-of-sight rule (occlusion only where a block's detail hitbox is hit, not whole opaque cells);
   slope 30° and wall 15° walking limits; climbing, doors, swimming, gliding, riding as the skills need
   them. (Dodge is not new: the 0.5.10 assets already allow left and right only and the simulator
   matches; only the action head's size changes.) Each construct names its differential before it is wired. Order follows the skills: locomotion
   and LOS (P2), combat and mana (P3), crafting (P4), the rest.
4. **World on 0.7.2.** Region recapture (500/500 sections), path tables, biome table, suites and
   catch-time horizons rebuilt; audit worlds re-selected and kept unseen; reset timing measured.
5. **Verification.** Every row of the mechanic checklist (`experimental/neural_agent/skill_agent/
   docs/BUILD.md` section 4) reaches MATCH only through its differential receipt; the eleven source-pin
   audits get 0.7.2 baselines (P5); `FIDELITY-0.7.2.md` is the ledger.
6. **Cutover (P6).** The stamp changes with the ruleset bytes and the content sha; every checkpoint
   retrains on gen 7; one skill checkpoint transfers natively (`execution_valid`); the default version
   flips; gen 5 remains selectable until the owner retires it.

## 3b. What the fidelity check measured (2026-10-17, `HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md`)

The live 0.7.2 server ran for 46 minutes under the bridge. Of 36 mechanic rows: 5 MATCH, 14 MISMATCH,
9 ABSENT, 8 UNMEASURED. None of the eleven source pins passes on 0.7.2 (four crash on removed or moved
classes and assets; seven drift). Locomotion constants all equal the 0.7.2 MovementConfig; the slope and
wall limits are new keys; flat walking and a ledge drop match to 1.1e-5 m but every jump differential
fails (4e-3 m in place, 0.3-0.7 m with forward motion; the suspected cause is which tick's duration
gravity uses, unconfirmed). The skills train with the "player" motion model while the bridge body is an
NPC at 5.0 m/s, so that model has no live oracle at all, on either generation. Melee: three sword
swings' knockback stale (6/7/12 should be 4.3/5.0/10.8 with the upward component 1 to 0.2), dagger
damage 3.0 against 5.0 live, the guard-cancels-attack rule removed in 0.7.2, 139 durability rows differ,
the agent's own hits land one to two ticks late; hit interrupt, stun and flinch absent (Trork and
Kweebec cannot be interrupted by their role files; skeletons and goblins can). Crafting: 2,047 recipes
against a table capacity of 1,947, 22 bench ids, campfire 2.0 s. All staged Region libraries were
captured on 0.5.7. Of the catalogue's 86 learned skills: 27 need a new construct, 44 need recalibration
first, 12 are unmeasured, 1 matches. Nothing trains on gen 7 before phases 1 and 2 land.

## 4. Phases and gates

| Phase | Work | Gate |
|---|---|---|
| 1 (running) | extractors both versions; content layer; adapters and the swap plan; `gen5/data/`, `gen7/data/`; construct designs and stubs; `PORTING.md`; `tests/sim/` | A1, A2; existing combat tests unchanged; one fixed rollout byte-identical |
| 2 | the engine behind the gen-5 profile | byte-identical rollouts; suite unchanged; the pre-migration copy retired |
| 3 | gen-7 profile, locomotion and LOS constructs, Region recapture | motion differential within 5e-4 per step flat and slopes; LOS differential; locomotion audit at the 0.7.2 baseline; suites regenerated |
| 4 | combat content and constructs, mana and runes, dodge | combat compares with 0.7.2 receipts; interrupt/flinch and mana differentials; projectile tiers within 2 % |
| 5 | crafting; source-pin baselines | crafting compare; all eleven audits pass on 0.7.2 |
| 6 | stamp, retraining lineage, native transfer, default flip | `execution_valid` with a skill checkpoint on 0.7.2 |
| on demand | climbing, doors, swimming, gliding, riding | a differential each, before the verb's results count |

## 5. The next generation, in one page

Add `server_versions/vN.py` and `sim/genN/profile.py` with the switches set from the release notes.
Run every extractor on the new `Assets.zip` into `sim/genN/data/`; diff against the previous
generation; read the diff against the notes. For each changed behaviour, set a switch or add a construct
with its differential named first. Run the source-pin audits and the differentials against a live
server of that version; record receipts. Compute the content sha and the stamp. Retrain the skill
lineage. Flip the default. The runbook `sim/PORTING.md` carries the commands and the gates; its
"hand-maintained" list is the measure of how agnostic the engine is, and it must shrink each port.

## 6. Ownership and rules of work

- One agent ports; the fidelity agent measures; neither edits the other's files. Results go to the
  ledgers before they are claimed (`FIDELITY-0.7.2.md` for mechanics; `experimental/neural_agent/docs/
  LEDGER.md` for the layers above).
- Until step 2's gate passes, the gen-5 code is the reference and is not edited for the port.
- Nothing external: the live server is the local 0.7.2 install through the bridge; assets are read from
  the local install.
- Commit in batches on `main`, never push; stage only the files one changed, never a directory.
