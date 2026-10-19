# source_pins: the World source-pin audit tool

`rebaseline.py` is the one-command path for a new release line. This README is the
runbook; it lives here (tracked) because `**/DEV.md` and `**/SKILL.md` are gitignored.
Pointers from the worldgen and tests DEV.md files lead here.

The gate is `verify_installed_world_source_pins` in `source_pins.py`. It runs the
eleven audits in `tests/fidelity/surrogate_*_source_audit.py` (gitignored, on disk)
against the installed `Server/HytaleServer.jar` and `Assets.zip`. The 0.5 line keeps
its module-constant pins; the 0.7 line reads `source_pin_baselines_072.json`. The
verifier stops at the first failing audit and never reads an audit's
`matches_0_5_7_baseline` flag, so audit-level consistency needs its own check.

1. Observe: `python tools/source_pins/rebaseline.py observe --install <install dir>
   --out-dir <scratch>/observe` runs the audits with the gate's own arguments and
   `HYTALERL_SERVER_VERSION` set to the build; per-audit stdout, stderr and exit go
   to the scratch directory.
2. Candidate: `python tools/source_pins/rebaseline.py candidate --install <install
   dir> --out <scratch>/candidate.json [--moved OLD=NEW ...]` writes the observed
   digests only. Pass `--moved` for every asset that moved (navigation's trace assets
   did on 0.7.2). The package JSON is never written by this step.
3. Verify by hand, audit by audit. List each audit's `SOURCE_CONTRACT` and
   `payload` claims, then read the installed bytes each claim cites:
   `rebaseline.py changed <0.5 jar> <build jar> <fqcn> [--method NAME]` (whitespace-
   normalised body diff against a 0.5.x control install: the 0.5.7 jar is not on every
   machine, and the phase-5 identity run found the pinned items byte-equal on 0.5.10),
   `javap -cp <jar> -p -constants <fqcn>` for constant values, and the bridge source
   for any transport claim.
4. Record only what you verified: `rebaseline.py accept --install <install dir>
   --audit NAME --semantic <observed hash> --evidence "<claim, class or file:line>"`.
   It refuses unless the hash equals the audit's observed semantic hash, the pinned
   digests still match, and an evidence line is given. An audit you did not verify
   keeps `EXPECTED_SEMANTIC_SHA256` null and names its gaps in `open_gaps`.
5. Gate: run `tests/worldgen/test_source_pins.py` and
   `tests/worldgen/test_source_pin_baselines.py` (23 tests), then the verifier. Do not
   edit the audit scripts' semantics or the 0.5 constants to make an audit pass.

Pitfalls found on 0.7.2:
- javap pads the constant-pool column by width; a raw body diff marks spacing-only
  changes as changed. The `changed` subcommand collapses whitespace.
- Compare by name and parameters, not by signature alone: a re-signatured method
  (`ExplosionUtils.performExplosion` gained `Rotation3f`) looks removed to a
  signature diff. Check the name across overloads too.
- Inner classes (`Outer$Inner`) are not hashed unless listed. A claim that runs
  through one is not verified by the hash, so name the class in the notes.
- Abstract methods have zero instructions; check the implementations too
  (`ItemContainer.getCapacity`).
- Parse-time defaults: the audits evaluate their `--assets` argparse default from `HYTALE_ASSETS_ZIP`
  (or `APPDATA`) before parsing, so a host without either variable failed inside argparse before any
  comparison. Since 2026-10-18 the gate (`verify_installed_world_source_pins`) passes the installed
  `Assets.zip` to the subprocesses as `HYTALE_ASSETS_ZIP`; `observe` and `accept` set a placeholder
  `APPDATA` (the default is unused). The audit scripts are unchanged.
- Static text in a hashed payload can go stale without the gate noticing (inventory's
  `transport.schema` says v1 while both bridges emit v2). Read the bridge for transport
  claims before accepting.

## Classification of the ten failing audits on 0.7.2 (2026-10-19)

The question: does an audit fail because the simulator is misaligned with 0.7.2, or because it is
a 0.5.7 test whose whole-class hash moved on bytes no claim depends on? Rule per audit: **A**, pin
only (every claim the simulator relies on holds on 0.7.2); **B**, at least one claim is false on
0.7.2; **C**, undecided claims remain. Each report under `evidence_072/` lists every claim with
its verdict, the method read on both jars (control 0.5.10, installed 0.7.2) and the evidence line;
`evidence_072/claim_methods.json` maps each classified claim to the methods that carry it. The
reports were written by the classification lanes and read, not re-derived, by the coordinator.

| audit | result | claims A / B / C | what decided it | gate |
|---|---|---|---|---|
| inventory | A | (2026-10-18 pass) | section IDs, class mapping, getters, active slots and capacity byte-identical; caveat: the hashed `transport.schema` text says v1 while both bridges emit v2 | verified |
| explosion_dynamics | A | 5 / 0 / 0 | phase order, hash-set iteration, the damage formula, `Math.round` before the subtraction and the UUID wire order all hold; caveats: the damage cause now comes from `ExplosionConfig.entityDamageCauseId` (default Environment), a zero-length knockback falls back to (0, 1, 0), and the host contract still names 0.5.7 digests so the audit's own `matches` flag stays false | verified (accepted 2026-10-19) |
| doors | B, content | 10 / 0 / 1, plus the census | 33 vertical door assets on 0.7.2 against the audit's literal 31 (horizontal 15 matches); the door mechanics are identical on both jars; the double-door fallback spans two reshaped methods | fail closed |
| explosion_mutation | B, one claim | 16 / 1 / 7 | block health is stored per `BlockHealthSection`, not in a `BlockHealthChunk` map; the destroyed predicate, the damage scale and the radius boundary hold; undecided: the loaded-chunk test, the soft-block force scale, the support cascade, the drop stack, the bridge publication claims | fail closed |
| navigation | C | 50 / 0 / 3 | no claim false; undecided: the native trace stage and two repo-side artifact claims that only a live run decides | fail closed |
| perception | B, content and LOS predicate | 16 / 3 / 6 of 25 claims (29 rows in the file) | genuine: the blocking branch of `PositionCache.lambda$hasLineOfSightInternal$1` adds a ray-versus-detail-hitbox test for non-transparent and role-set blocks (the control blocks those unconditionally; modelled by the `line_of_sight` construct), so the class digest moves with it; the projected-distance declarations are 17 in 12 files against the pinned 11 (3 moved to `Undead/`, 9 new; control not checkable); the light-initialisation trigger is not traced (C, bridge and repo side) | fail closed (the gate's first stop) |
| entity_geometry | B | 12 / 4 / 0 of 16 | genuine: `EntityScaleBoundingBoxSystem.tick` sets the box scale to the entity scale over the model scale every tick (absent on the control; unmodelled), and the BoundingBox scale methods are new (class digest); stale on the control too: the `DeathComponent` exclusion claim (the test lives in `EntityRefCollisionProvider.defaultEntityFilter`; an attribution error in the audit); held: the rotated box producer, the perception endpoint, the double-application rule (`resolveBoxScale` divides by `Model.getScale`), the transport schema and ordering (the label `physical_scene_unavailable_fail_closed` is emitted nowhere) | fail closed |
| actor_world_state | C | 30 / 0 / 3 of 33 | the six groups hold (fall state, breathing, controller medium, submersion, drop, stationary traversal); the three open leaves are repo-side provider claims; unclaimed differences noted, not verdicts: 0.5.10 `WorldUtil` returns Empty for y >= 320 (0.7.2 has no cap), `BodyMotionCharge` clears the fall distance on 0.7.2, `computeNewFallSpeed` moves the inverted-gravity sign, and `applyActiveProbeBlockCollisionFilter` is a no-op | fail closed |
| locomotion | B, protocol and dive roles | 44 / 3 / 16 of 63 leaves (payload rows 8 / 1 / 0) | genuine: `MovementStates` keeps its 23 ordered booleans and adds `extraJumpsUsed` (a byte) on 0.7.2; the bridge's `MovementStateFrame.from` never reads it, so the byte does not reach the bridge contract; dive roles 6 against 5 (`Test_Dive_Stationary.json` added; control not checkable by rule); stale on the control too: the sprint fallback 1.65 (no fallback path and no 1.65 literal on either jar; the only 1.65 is `Mount.json`); held: the sprint multiplier 1.273, the walk steering `computeHeading` (identical on both jars), the two new MovementConfig keys (no server motion reader); the 16 C rows are repo-side producer formulas and runtime-only oracles | fail closed |
| mutable_blocks | B | 144 / 18 / 41 of 203 units | genuine: block health per `BlockHealthSection`, cleared through `BlockOperations.setBlock` on a changing write or flag 0x40 (`WorldChunk.setBlock` is gone; the control has it); `InteractionType` has 28 constants on 0.7.2 against 25 on the control (Ability4, OnBreak, OnBreakImpact added); the content censuses: support census in 6 fields by the audit's reader (4 confirmed independently), 11 falling-block assets against 2, tick procedures 17 against 13, gather types 22 against 20; `block_health_reset_on_write` is still a design-only stub; the damage, repair, break and place claims hold (A), and the `BlockHealthModule$BlockHealthSystem` absence is a move to `BlockHealthSystems$Tick` (A, pin only) | fail closed |
| crafting | B, bench augment tags and content drift | 28 / 4 / 0 of 32 | genuine: `isValidBenchForRecipe` (the one bench predicate: known-recipe test, memories gate, bench match) adds `benchHasRequiredAugmentTags` to the per-requirement match; 50 generated recipes carry augment requirements and no explicit recipe does (unmodelled; a candidate note for the `bench_crafting` row); resource selectors 988 against 974 (zero tag and metadata selectors; the 2,047 total holds); knowledge-required recipes 25 against 19; stale on the control too: the bridge's `craftable_count` omits the output-room condition on both copies, and the learning window's id split is a model gap on both jars | fail closed |

Final count (2026-10-19, after the claim lanes' final files): A 1, B 7, C 2 over the ten failing audits; inventory is verified from the previous pass. A: explosion_dynamics. B: doors, explosion_mutation, perception, entity_geometry, locomotion, mutable_blocks, crafting. C: navigation, actor_world_state. Counts are claim rows per audit (the table shows each lane's unit). What the B verdicts are.
Genuine 0.7.2 changes, each a candidate for simulator work: block health stored per `BlockHealthSection` and cleared through `BlockOperations.setBlock` on a changing write or flag 0x40 (`WorldChunk.setBlock` is gone); entity scale made physical by `EntityScaleBoundingBoxSystem`; the LOS detail-hitbox test in the blocking branch of `PositionCache` (modelled by the `line_of_sight` construct); the bench gate's augment tags (`benchHasRequiredAugmentTags`, unmodelled, 50 generated recipes); the `MovementStates` `extraJumpsUsed` byte (does not reach the bridge); `InteractionType` at 28 constants against 25 (Ability4, OnBreak, OnBreakImpact); and the content censuses: doors 33 against 31, projected-distance declarations 17 against 11, the support census in 6 fields (4 confirmed independently), falling-block assets 11 against 2, tick procedures 17 against 13, gather types 22, resource selectors 988 against 974, dive roles 6 against 5. The census items are genuine against the 0.5.7 literals the audits passed with; the 0.5.10 archive was not read, so their rows in the JSON say control not checkable.
Claims already false on the 0.5.10 control, so gen-5 test artifacts rather than 0.7.2 drift: the `DeathComponent` exclusion (an attribution error: the test lives in `EntityRefCollisionProvider.defaultEntityFilter`), the sprint fallback 1.65 (no such path on either jar), `WorldChunk.breakBlock` (absent on both), the `craftable_count` output-room omission (both bridge copies), and the learning id split (both jars).
Each claim's verdict and evidence is in `source_pin_baselines_072.json` (`open_gaps` for B and C, `confirmed_on_072` for A), and `evidence_072/EVIDENCE.md` is the consolidated record, one block per claim. The verifier still stops at `perception`, the first audit, so 0 of 11 pass end to end while 2 of 11 carry a semantic baseline. The C rows that remain are repo-side producer formulas, bridge-side publication contracts and runtime-only oracles.

## Why whole-class digests make every refactor a failure

Each audit's semantic hash covers the SHA-256 of every pinned class file (`CLASS_SHA256`), every
pinned asset (`ASSET_SHA256`), a live census where one exists, and the static `SOURCE_CONTRACT`
text. The gate compares that one hash, so a change to any byte of a pinned class fails the audit,
whether or not a simulator claim depends on the changed bytes: a renamed private field, a reordered
method, a new unrelated overload, or a constant-pool re-index all move the class digest. The 0.7.2
pass showed the consequence: the ten failing audits could not be told apart by the gate, and a
claim that still held (explosion dynamics: phase order, damage formula, `Math.round`) failed in the
same way as a claim that was false (explosion mutation: the health storage moved to
`BlockHealthSection`). The gate cannot separate "the pin is stale" from "the behaviour moved".

## What a claim-level pin would look like (not implemented)

Record, per release line and per audit, one entry per claim: the verdict, the evidence line, and
the list of methods whose bytes carry the claim. Each method is keyed by class, name and
parameters, and its body is the whitespace-normalised javap listing that `rebaseline.py changed`
already produces (constant-pool numbers replaced by `#`, offsets dropped). For a claim, the pin is
the set of method digests in its closure. On a new release line:

1. Recompute only the listed method digests. A claim whose digests all match keeps its verdict
   without re-derivation.
2. A claim with a changed digest is re-derived, and its diff is printed with
   `rebaseline.py changed <control> <build> <class> --method NAME`.
3. The whole-class `CLASS_SHA256` table stays as the outer guard, so a method nobody listed still
   fails closed.

Known limits of method-level pins: a claim can rely on a callee outside its list (the closure
depth must be recorded per claim), and a wrapper chain such as `performBlockDamage` into
`damageSingleBlock` needs every link listed. The per-claim method lists built in the 2026-10-19
pass (`evidence_072/claim_methods.json`: 469 claim rows across the ten audits, 254 with at least
one class-method token) are the input for that design.

## Addendum from the claim-level pass (audits_classify)

- A semantic pin binds every value the audit hashes, including expectations the audit embeds. The
  mutable-block contract embeds the 0.5.7 support census, and the 0.7.2 census differs in six fields
  by the audit's reader (four confirmed independently), so a pin would have re-baselined a false
  census. Doors and perception keep their 0.5.7 counts in the matches flags, which the gate does not
  read, but their live censuses are hashed (doors: 33 vertical assets against 31; perception: 17
  projected-distance declarations against 11). Re-derive such expectations before pinning.
- The per-claim method list (evidence_072/claim_methods.json, all ten audits) gives a class-method
  token for 254 of 469 claim rows; the rest are repo-side, bridge-side or not yet located, and
  mutable_blocks is the thinnest (63 of 203). It is an input for the claim-level design, not a pin.


## Claim text per release line

An audit hashes its claim text: its module `SOURCE_CONTRACT`, or a payload claim constant (perception's
`PERCEPTION_CLAIMS`). The module text is the 0.5.7 line's and never changes. A claim that is false on a later
line is re-worded for that line only, in `source_pin_baselines_072.json`, under the constant's name (2026-10-19).

Rules:

1. Per-line record. An audit's 0.7 record may carry `claim_texts`, an object keyed by constant name.
   `pins_for` returns each entry under its name on the 0.7 line; the 0.5 line returns only its defaults and
   never receives one. An audit with no entry hashes its module text.
2. Same structure, re-worded wording. A recorded text has the constant's key structure: the same dictionary
   keys at every depth, and each list's records have the constant's record shape. A list's length is data (a
   census may grow), and every scalar (string, number, boolean, null) is wording that may change type. A claim
   is never dropped or added at any depth. The check cannot see a whole list record removed or added (census
   lists change length), so reviewers check list-record changes by hand.
3. Stale text needs its override before a semantic pin stands. `STALE_CLAIM_TEXT_072` (in the gitignored
   `tests/worldgen/test_source_pin_baselines.py`) names each audit whose module text states a claim false on
   0.7.2, with the constants to override. A 0.7 semantic pin for one of them needs those constants in its
   `claim_texts` first. `UNOVERRIDDEN_STALE_AT_RULE_072` is a ratchet: it lists the pins accepted before this
   rule that still lack an override (inventory today: its `transport.schema` text says v1, and both bridges
   emit v2). Recording that override changes inventory's observed 0.7 hash, so its semantic pin must be
   re-accepted in the same step, and the set emptied in the test.
4. Audit scripts change by one payload line, or by moving claim constants. Each payload carries
   `"source_contract": pins.get("SOURCE_CONTRACT", SOURCE_CONTRACT)`. Perception's claims were inline in its
   payload; they now sit in the module constant `PERCEPTION_CLAIMS` (content unchanged) and enter the payload as
   `**pins.get("PERCEPTION_CLAIMS", PERCEPTION_CLAIMS)`. Nothing else in an audit changes.

A fragment is the complete constant with every key present and the 0.7.2 wording in its leaves: for inventory,
the `transport.schema` leaf reads `hytalerl_native_inventory_v2`. A partial object fails rule 2.

Proof (2026-10-19, lane `contract_per_line`; each run's `summary.json` is in that lane's scratch folder, not in
this repository):

- Observe before and after the change, on both installs: all eleven semantic hashes identical on 0.5 and on
  0.7, and the package JSON carries no `claim_texts`.
- Positive checks, with scratch copies of the JSON read through a scratch interpreter that sets the loader's
  path in its own process (`POINTED_072_JSON`; no tracked hook). In each, only the target audit's 0.7 hash
  moved, and no 0.5 hash moved:
  - explosion_mutation, one re-worded leaf (`damage.health`): `00ccd47d61fb` to `fe088ae5b861`.
  - entity_geometry, its collector fragment: `237372ef1341` to `e3f58624b62b`.
  - mutable_blocks, its collector fragment: `e465a47137e5` to `a97af2381a28`.
  - perception, one re-worded `PERCEPTION_CLAIMS` leaf (`los_contract.perception.predicate`): `a86113082526` to `fb222a92d7a5`.
- Tests: 16 before the change, 23 after (seven new rule tests).
