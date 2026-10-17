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
   wind strike, chain hook, ground slam, charged frost shot, enrage, poison); hookshot; two-direction
   dodge; the 0.7.2 line-of-sight rule; climbing, doors, swimming, gliding, riding as the skills need
   them. Each construct names its differential before it is wired. Order follows the skills: locomotion
   and LOS (P2), combat and mana (P3), crafting (P4), the rest.
4. **World on 0.7.2.** Region recapture (500/500 sections), path tables, biome table, suites and
   catch-time horizons rebuilt; audit worlds re-selected and kept unseen; reset timing measured.
5. **Verification.** Every row of the mechanic checklist (`experimental/neural_agent/skill_agent/
   BUILD.md` section 4) reaches MATCH only through its differential receipt; the eleven source-pin
   audits get 0.7.2 baselines (P5); `FIDELITY-0.7.2.md` is the ledger.
6. **Cutover (P6).** The stamp changes with the ruleset bytes and the content sha; every checkpoint
   retrains on gen 7; one skill checkpoint transfers natively (`execution_valid`); the default version
   flips; gen 5 remains selectable until the owner retires it.

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
