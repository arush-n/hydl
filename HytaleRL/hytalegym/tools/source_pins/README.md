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
   `tests/worldgen/test_source_pin_baselines.py` (16 tests), then the verifier. Do not
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
| perception | B, content | 0 / 1 / 7 | 17 `UseProjectedDistance` declarations on 0.7.2 against the audit's 11 (3 expected absent, 9 new, no value change on the common paths); the selectors, the LOS hit and cache paths, the channel and light classes and the repo-side contracts are not re-derived (the lane stopped at the Haiku limit) | fail closed (the gate's first stop) |
| entity_geometry | B | 6 / 3 / 7 | `EntityScaleBoundingBoxSystem.tick` sets the box scale from `EntityScaleComponent` each tick (absent on the control: entity scale is physical on 0.7.2); the spatial query excludes `Intangible` only, with zero `DeathComponent` references on both jars (that exclusion claim was stale on the control too); rotation semantics, the collision system, the transport schema and ordering hold; the entity LOS endpoint and the double-application rule are undecided | fail closed |
| actor_world_state | C | 0 / 0 / 7 | fall, breathing, controller medium, submersion, drop and stationary traversal unread (the lane stopped) | fail closed |
| locomotion | B, protocol | 2 / 1 / 6 | `MovementStates` keeps its 23 booleans and adds `public byte extraJumpsUsed` on 0.7.2 (a 24th field outside the 23-bit mask); the sprint multiplier 1.273 and the Default gameplay config hold; walk steering, the sprint fallback 1.65 and the repo producer are undecided | fail closed |
| mutable_blocks | B | 1 / 5 / 9 | `WorldChunk.setBlock` is gone (the write path is `BlockOperations.setBlock`, which calls `BlockHealthModule.onBlockReplaced`); block health lives in `BlockHealthSection`; the embedded support census differs in 6 fields (resolved rows 5823 to 6216), 11 falling-block assets against 2, 29 `InteractionType` constants against the claimed 25 (26 on the control); the destroyed predicate holds; `block_health_reset_on_write` is a design-only switch today (stub module, no Hook); `mutable_blocks_classpins_072.txt` keeps the class census (78 pinned: 19 unchanged, 58 changed, 1 absent) | fail closed |
| crafting | C | 0 / 0 / 7 | the 0.7.2 `isValidBenchForRecipe` is the player's known-recipe test, not the bench-match predicate the audit cites, so the bench gate was not located; the recipe corpus was not recounted | fail closed |

Final count (the collector lane, 2026-10-19 00:31 CDT): A 1, B 6, C 3. What the B verdicts are.
Genuine 0.7.2 changes, each a candidate for simulator work: block health stored per
`BlockHealthSection` and cleared through `BlockOperations.setBlock` (`WorldChunk.setBlock` is gone),
entity scale made physical by `EntityScaleBoundingBoxSystem`, the `MovementStates` extra byte, and
the content censuses (doors 33 against 31, projected-distance declarations 17 against 11, the
support census in 6 fields, 11 falling-block assets against 2). Claims that were already false on
the 0.5.10 control, so gen-5 test artifacts rather than 0.7.2 drift: the `DeathComponent`
exclusion, the 25 `InteractionType` constants (26 on the control, 29 on 0.7.2), and
`WorldChunk.breakBlock` (absent on the control). Each claim's verdict and evidence is also in
`source_pin_baselines_072.json` (`open_gaps` for B and C, `confirmed_on_072` for A), and
`evidence_072/EVIDENCE.md` is the consolidated record, one block per claim. The verifier still
stops at `perception`, the first audit, so 0 of 11 pass end to end while 2 of 11 carry a semantic
baseline. Next: the C leaves (the lanes stopped at the Haiku weekly limit, which resets
2026-10-23), then each B claim re-worded for 0.7.2 and the constructs the genuine changes need.

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
pass (`evidence_072/claim_methods.json`: 149 claim lines across the ten audits, 102 with at least
one class-method token) are the input for that design.

## Addendum from the claim-level pass (audits_classify)

- A semantic pin binds every value the audit hashes, including expectations the audit embeds. The
  mutable-block contract embeds the 0.5.7 support census, and the 0.7.2 census differs in six fields,
  so a pin would have re-baselined a false census. Doors and perception keep their 0.5.7 counts in the
  matches flags, which the gate does not read, but their live censuses are hashed (doors: 33 vertical
  assets against 31; perception: 17 projected-distance declarations against 11). Re-derive such
  expectations before pinning.
- The per-claim method list built from the ten failing audits covers 102 of 149 claim lines with at
  least one class-method token; the rest are open (C) or repo-side. The full list is in the lane's
  `evidence_072/claim_methods.json`. It is an input for the claim-level design, not a pin.
