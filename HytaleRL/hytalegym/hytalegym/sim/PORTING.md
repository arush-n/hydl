# Porting the simulator to Hytale version N+1 (runbook, 2026-10-17)

## Version bump checklist (learned from the 0.7.2 port)

Variables as in the block under "The runbook" below: `ROOT=/Users/arush/hydl`, `PY=$ROOT/.venv/bin/python`, `NEW` the game
install. `$SCRATCH` is any directory outside the repository. Run pytest from `$ROOT/HytaleRL/hytalegym` (first pitfall).
Each step ends with its gate, a command with a pass bar.

1. **Profile (a).** Add `server_versions/vN_M.py` (`ServerProfile`: bridge jar and its SHA-256, `crafting_recipe_count`,
   `checkpoint_rulesets`). Copy `sim/gen7/profile.py` to `sim/genN/` (`__init__.py`, `profile.py`) and set its fields; add
   the `GENERATION_MODULES` entry in `sim/common/content/registry.py`. Gate: `$PY -m pytest tests/sim/test_registry.py -q`.
2. **Extract (a).** `cd $ROOT/HytaleRL/hytalegym/tools && $PY -m content_extract extract --assets "$NEW/Assets.zip" --out ../hytalegym/sim/genN/data`.
   The game version comes from `Server/HytaleServer.jar` beside the archive (`--game-version` overrides). Gate:
   `$PY -m pytest tests/sim/test_extractors.py tests/sim/test_schema.py -q`.
3. **Diff (a).** From `tools`: `$PY -m content_extract diff ../hytalegym/sim/genPREV/data ../hytalegym/sim/genN/data --json $SCRATCH/diff.json --markdown $SCRATCH/diff.md`.
   Read it against the release notes and the live measurements; encode each measured change as an assertion beside
   `tests/sim/test_a2_diff.py`. Gate: those assertions pass.
4. **Swap plan (b).** After every commit that touches a module under `hytalegym/jax/` (a comment counts), from `tools`:
   `$PY -m sim_port.regen_swap_plan --dry-run` reports the new head, the record counts, every moved `head_line` and the pin.
   Then `$PY -m sim_port.regen_swap_plan` writes `sim/common/content/redirects.json` and re-pins
   `test_step_clock_rows_are_the_swap_plan_source_sites` in `tests/sim/test_render.py` if the counts changed. Gate: the
   dry run then shows no moves, and `$PY -m pytest tests/sim/test_render.py -q` passes.
5. **Gates before every commit (c).** Quick gate: `$PY -m pytest tests/sim -q --ignore=tests/sim/test_render_identity.py`.
   Identity gate: `$PY -m pytest tests/sim/test_render_identity.py -q`; it renders gen 5 from a `git archive` of HEAD and
   compares every array byte for byte. Pass bar: 4 passed.
6. **Gen-5 identity record (d), only when the gen-5 engine changes.** Export HEAD with
   `git archive HEAD HytaleRL/hytalegym/hytalegym | tar -x -C $SCRATCH/head`. Then, twice, from `tools`:
   `JAX_PLATFORMS=cpu PYTHONPATH=$SCRATCH/head/HytaleRL/hytalegym:. $PY -m sim_bench.rollout --batches 8,64 --steps 64 --out $SCRATCH/run1.json`, then the same with `run2.json`
   (the `command` field of `sim/gen5/identity.json`). Pass bar: identical fingerprints in both runs. Then move the old
   values to `committed_engine.previous`, record the new ones and update `sim/BASELINES.md`. Procedure: commits
   baacdc9 and ede9a39.
7. **Construct (e).** One module per construct under `sim/genN/mechanics/` with `SWITCH` and `IMPLEMENTED`, registered in
   `CONSTRUCT_MODULES` (`mechanics/__init__.py`). Its function takes `(original, profile)` and returns `original` when the
   switch is off; the overlay's `Hook` names it as `module:function`. Each `Hook`, `Patch` or `Redirect` entry in
   `sim/genN/overlay.py` carries the fidelity row and the differential text. Add the switch to `IMPLEMENTED_SWITCHES` in
   `tests/sim/test_constructs.py`. Gate: step 5, before every commit.
8. **Content values (f).** Compare the old and new tables side by side with the swap-plan adapters
   (`hytalegym.sim.common.content.adapters`; the adapter command is item 1 of section 4 below). Where the plan names the
   site, use a `Redirect`; otherwise a `Patch` anchored on the exact literal. Pin each new literal in
   `tests/sim/test_render.py`, as `test_gen7_overlay_applies_its_patches_and_step_clock` does. Gate: step 5.
9. **Live differentials (g).** Use a private bridge: `$PY $ROOT/HytaleRL/hytale-plugin-0.7.2/tools/launch_bridge_072.py`
   (details in `HytaleRL/hytale-plugin-0.7.2/BRIDGE-PORTING.md`, section 7). One port pair per session or agent, never a
   peer's; one launch per scenario when native state persists across resets. Gate: the pass bars of section 6 below.
10. **Record (h).** A `FIDELITY-N.md` section per phase, with a table whose status is MATCH, OPEN, NOT MET or UNMEASURED;
    receipts outside the repository, with their paths named; a status paragraph in `GEN7-PLAN.md` section 2b. Then
    `$PY $ROOT/HytaleRL/hytale-plugin-0.7.2/tools/fidelity_index.py` (it reads `FIDELITY-0.7.2.md`; copy it for N+1).
11. **Lanes (i).** One owner per tree: the simulator (`sim/`) and the bridge (the plugin folder). Neither session edits the
    other's files (GEN7-PLAN section 6). Before a commit that moves HEAD under a renderer, post a hold notice; after it, a
    release notice that names the HEAD to render from. The owner asked for this protocol; no file in the repository
    defines the notice format.

**Pitfalls**

- Repo-root shim: under `python -m pytest` run from `HytaleRL/`, `hytalegym` resolves to the shim
  `HytaleRL/hytalegym/__init__.py`, so run from `HytaleRL/hytalegym`. Render anchors relocate onto `hytalegym.envs` for the
  same reason (RENDER.md; `tests/sim/test_render_anchors.py`).
- Uncommitted work: renders read `git show HEAD:`, but the renderer and the overlays are working-tree Python. An
  uncommitted engine or overlay edit changes what a render means. Check `git status -- HytaleRL/hytalegym` before reading
  a difference as a finding (BASELINES.md section 1 measured different fingerprints from the working tree).
- A commit of an engine module without the regeneration fails `test_overlays_validate_against_head` and the step-clock
  identity test (commit 5cb738c: "expression must occur once on combat/types.py:604"). A comment-only edit moved the
  anchors too (commit 6c30ba9).
- Check an engine commit on a clean `git archive` of HEAD: commit ede9a39 did not import until commit 879321e added three
  files that it needed.
- The engine's own tests under `tests/jax` are gitignored (`HytaleRL/.gitignore`) and partly stale: commit ede9a39
  records one failure that predates its batch.
- Gate times follow machine load. Recorded identity-gate times: 5 min 25 s to 7 min 42 s in most commit messages of
  2026-10-18; 28 min 17 s and 49 min 54 s (the latter at load 85 to 105); 24 and 36 min in GEN7-PLAN 2b. Gates run on the
  shared working tree, so other sessions' uncommitted test edits change the result.
- A bridge pin move invalidates every artefact keyed by the bridge SHA: the world-evidence history under
  `HytaleRL/artifacts/worldgen/world-evidence-history/<sha>`, the recordings' `bridge_sha256`, the traversal and
  token-contract cascade manifests and the stability baseline constant. On 2026-10-18, after six pin moves in one day,
  twelve untracked worldgen tests failed on exactly that (receipts under the session's `test_fixtures/`). Republish
  those artefacts once the pin settles, not after every bridge commit; until then read those failures as staleness.
- Five more untracked tests need `HYTALE_ASSETS_ZIP` (or `APPDATA`) set to the install's `Assets.zip`; without it
  they fail on the asset lookup before reaching any bridge code.
- Several construct builders at once share four files: `sim/gen7/overlay.py`, `sim/common/profile.py`,
  `sim/gen7/mechanics/__init__.py` and `tests/sim/test_constructs.py`. Lock protocol (2026-10-18): before editing any of them,
  create `sim/gen7/.overlay.lock` containing your lane name (fail if it exists: `set -o noclobber`); edit, run
  `tests/sim/test_constructs.py` (it renders gen 7 and imports every construct) plus your own test module, and remove the lock
  only when both pass: a hook that raises at render time breaks every lane's tests, not just yours (2026-10-18: a hook read
  `jnp` from a wrapped module's globals that had none, and 61 unrelated tests errored until it was fixed). If the lock exists, wait 60 s and retry, up to 30 min; never delete another lane's lock. The
  construct module, its tests and its fixture are per lane (`sim/gen7/mechanics/<name>.py`, `tests/sim/test_<name>.py`) and
  need no lock. The coordinator commits one lane at a time, staging that lane's files only.
- The gen-7 tools name gen 7: `sim_port.regen_swap_plan` checks `hytalegym.sim.gen7.overlay`, and `fidelity_index.py`
  reads `FIDELITY-0.7.2.md`.

## The runbook: the model and sections 1 to 8

The engine is the committed gen-5 source and is never edited. A generation is a profile
(`sim/genN/profile.py`), its extracted content tables (`sim/genN/data/`) and an overlay: the
redirect list (`sim/common/content/redirects.json`: which engine symbol reads which table field),
the construct switches and hooks (`sim/genN/mechanics/`), and the fidelity patches. A generation's
simulator is rendered from `git show HEAD:<path>` plus that overlay (the renderer is
`sim/common/render.py`, described in `sim/common/RENDER.md`). The next port is therefore: profile, extract, diff and review, overlay delta,
render, differentials, stamp, retrain, flip. Commands assume the repository root `/Users/arush/hydl`
and the local install only (nothing external). Measured costs of each step: `BASELINES.md`.

```sh
ROOT=/Users/arush/hydl; PY=$ROOT/.venv/bin/python
NEW="$HOME/Library/Application Support/Hytale/install/release/package/game/latest"   # Assets.zip, Server/HytaleServer.jar
```

## 1. Profile

1. `HytaleRL/hytalegym/hytalegym/server_versions/vN_M.py` with a `ServerProfile` (bridge jar and
   SHA-256, `crafting_recipe_count`, `checkpoint_rulesets`) and its `_MODULES` entry.
2. `HytaleRL/hytalegym/hytalegym/sim/genN/{__init__,profile}.py`: copy `sim/gen7/profile.py`, set
   `generation`, `certified_version`, `release_line`, `data_dir`, and start `constructs` from the
   previous generation's values; add the `GENERATION_MODULES` entry in
   `sim/common/content/registry.py`.
3. Step duration is content: `SimProfile.step_duration` (gen 5: nominal 1/30 s, loaded 0.045 s,
   three per-lane timing profiles, from the ruleset `engine` block; gen 7: one value, 1/30 s, owner
   ruling: the agent operates at 30 Hz). Every site that reads it (338 in the engine, 24 in
   `arena/` and `HytaleRL/examples/`; module, symbol, HEAD line) is in `redirects.json`
   `step_duration`, regenerated with the redirect list.

Gate: `tests/sim/test_registry.py` (every server release line has a generation; the profile binds
its `ServerProfile`; the bridge profile's recipe count equals the extracted one; gen 5's step duration
equals today's ruleset engine block).

## 2. Extract

```sh
cd $ROOT/HytaleRL/hytalegym/tools
$PY -m content_extract extract --assets "$NEW/Assets.zip" --out ../hytalegym/sim/genN/data
```

Reads the archive only (no extraction to disk); writes nine tables plus `MANIFEST.json` and prints
the content SHA-256. The game version comes from the jar manifest (`--game-version` overrides). Each
table records every source path with its SHA-256 (inline up to 400 sources, otherwise the digest and
count; every row names its `source`), the archive SHA-256 and the extractor version; no timestamps,
so re-running is byte-identical.

Gate: `tests/sim/test_extractors.py` (byte-exact round trip on each installed version, synthetic-zip
units) and `tests/sim/test_schema.py` (schema, recomputed row hashes, manifest).

## 3. Diff and review

```sh
$PY -m content_extract diff ../hytalegym/sim/genPREV/data ../hytalegym/sim/genN/data \
    --json /tmp/content-diff.json --markdown /tmp/content-diff.md
$PY $ROOT/HytaleRL/hytale-plugin-0.7.2/tools/diff_assets_072.py   # file-level asset and jar diff (P0); adapt its paths
```

The content diff lists, per family, rows added, removed, changed (field by field) and moved (only the
source path changed, e.g. 0.7.2's `Aggressive/` to `Faction/` roles). Read it against the release
notes and live measurements (for 0.7.2: `HytaleRL/hytale-plugin-0.7.2/TRAINING-0.7.2.md`). Encode
every measured change as an assertion in `tests/sim/test_a2_diff.py` (the A2 gate for 0.7.2 checks
sword knockback 6.0 at (0, 1, -1) to 4.3 at (0, 0.2, -1), 31 interrupt-flagged attack files, daggers
+50 %, the ranged cut, mana 100 at +1 per 0.3 s, durability 57/82 of 143 rows, webs 0.65 to 0.45,
slope/wall keys, 2,047 recipes and 22 benches, NPC roles unchanged but moved, guard roots no longer
interrupting Primary, dodge unchanged).

## 4. Overlay delta

1. Redirects. Run every adapter on the new tables; an adapter that raises or changes shape marks a
   content-shape change to review (0.7.2 example: the shortbow charge root gained a 0.0 branch, so
   tiers are mapped by the interaction each branch names, not by position):

   ```sh
   cd $ROOT && $PY -c "from hytalegym.sim.common.content import load, adapters as A; t=load('N.M.P'); [print(e.adapter, getattr(A, e.adapter)) for e in A.SWAP_PLAN]"
   cd HytaleRL/hytalegym/tools && $PY -m content_extract.redirects --repo $ROOT --out ../hytalegym/sim/common/content/redirects.json

   Rule (2026-10-18): the step-clock redirects anchor on HEAD line numbers recorded in `redirects.json`, so every
   commit that touches an engine module under `hytalegym/jax/` (even a comment) must be followed by this
   regeneration, by re-pinning the site counts in `tests/sim/test_render.py`
   (`test_step_clock_rows_are_the_swap_plan_source_sites`) if they moved, and by `tests/sim` plus
   `test_render_identity.py`; the engine batch of that day failed both gates until the plan was regenerated.
   ```

   The redirect list records, per engine symbol, the module, the line in `HEAD`, the line in the
   working tree, whether the file is clean or uncommitted, the adapter, how the value is held
   (`module_constant`, `inline_literal`, `function_default`, `json_table`) and the table fields read.
2. Constructs. For each behaviour that appears, disappears or changes shape, set a `Constructs`
   field in the new profile or add one (default = gen-5 behaviour), with a stub under
   `sim/genN/mechanics/` naming its switch and a row in `mechanics_NNN.md` (inputs, differential,
   phase). Gate: `tests/sim/test_constructs.py`.
3. Patches. Numbers the gen-5 engine encodes structurally (a guard root's interrupt mask) become
   patches in the overlay, each tied to a table field (0.7.2 list: `gen7/mechanics_072.md`).

## 5. Render (`sim/common/render.py`; built 2026-10-17, RENDER.md)

Read every module with `git show HEAD:<path>` (never the working tree, so no uncommitted work is
baked in), copy, apply the redirects, insert the construct hooks and patches. Identity gate: the
gen-5 overlay renders byte-identical rollouts on the fixed seeds:

```sh
cd $ROOT/HytaleRL/hytalegym/tools
JAX_PLATFORMS=cpu PYTHONPATH=<render root>:. $PY -m sim_bench.rollout --batches 8,64 --steps 64 --out /tmp/render.json
# fingerprints must equal sim/gen5/identity.json committed_engine.fingerprints
```

Every render also reports compile time and throughput deltas against `BASELINES.md`, measured back
to back with the committed engine under the same load.

## 6. Differentials against a live server of version N

```sh
$PY $ROOT/HytaleRL/hytale-plugin-0.7.2/tools/launch_bridge_072.py --run-dir /tmp/runN --bridge-port 5556 --game-port 25572
# eleven source-pin audits (record new baselines; four crash on 0.7.2 and need code changes first)
cd $ROOT/HytaleRL/hytalegym && HYTALE_ASSETS_ZIP="$NEW/Assets.zip" APPDATA=/tmp/appdata-dummy PYTHONPATH=. \
  $PY tests/fidelity/surrogate_<name>_source_audit.py --server-jar "$NEW/Server/HytaleServer.jar" [--assets "$NEW/Assets.zip"]
#   <name> = perception, actor_world_state, locomotion, navigation, door, mutable_block,
#            explosion_mutation, explosion_dynamics, crafting, inventory, entity_geometry
#   (--assets for all but actor_world_state, explosion_dynamics, inventory, entity_geometry;
#    both env variables only satisfy argparse defaults evaluated before --assets)
cd $ROOT && PYTHONPATH=.:HytaleRL/hytalegym $PY -m adk.tools.native_motion_differential --port 5556 --steps 30 \
  --expected-server-version N.M.P --output /tmp/runN/motion/forward-30.json
PYTHONPATH=.:HytaleRL/hytalegym $PY HytaleRL/hytalegym/tests/fidelity/native_geometry_compare.py --port 5556 --expected-server-version N.M.P
PYTHONPATH=.:HytaleRL/hytalegym $PY -m pytest -m native_server HytaleRL/hytalegym/tests/fidelity/combat -q --timeout 600
PYTHONPATH=.:HytaleRL/hytalegym $PY HytaleRL/hytalegym/tests/fidelity/surrogate_crafting_native_compare.py --port 5556 \
  --assets "$NEW/Assets.zip" --server-jar "$NEW/Server/HytaleServer.jar" --native-evidence-jar <bridge jar>
```

The native tools take a shared evidence lease; the 0.7.2 fidelity run redirected it with its own
harness (`R/scripts/run_tool.py`, `lease_redirect.py` under `~/.claude/jobs/31947bd7/tmp/fidelity072/`,
not in the repository). Record every result with its receipt in `FIDELITY-N.md`; a mechanic is
calibrated only when its row passes.

| Gate | Pass bar |
|---|---|
| source-pin audits | 11/11 semantic match on version N |
| motion differential, flat and slopes | position error <= 5e-4 m per step |
| line of sight | every fixture ray agrees |
| combat compares | exact damage and hit ticks; per-hit displacement within tolerance |
| projectiles | range error <= 2 % per tier |
| resources (stamina, oxygen, mana) | rates within 5 % |
| crafting | catalogue and timings equal |
| native transfer | `execution_valid` with one skill checkpoint |

## 7. Stamp

The effective content identity is `hytalegym.sim.common.content.content_sha256(profile)` (gen 5:
tables plus calibration overlay; gen 7: tables), pinned as `CONTENT_SHA256` in `sim/genN/profile.py`
(gen 5 `BF735B65...`, gen 7 `FE91CAEF...`; `tests/sim/test_registry.py` fails on drift). It joins `hytale_version` and the ruleset SHA in the
combat contract stamp (`jax/combat/contracts/publication.py`, `combat_contract_stamp`; that file is
uncommitted by another session, so the field is not wired yet). A new stamp invalidates every
checkpoint through `combat_checkpoint_contract_errors`.

## 8. Retrain, then flip

Retrain the skill lineage on generation N (owner rule 2026-10-17: no re-stamped carry-over). Then set
`DEFAULT_SERVER_VERSION` in `server_versions/__init__.py` and mark the previous generation
`deprecated=True`; keep its profile, data and overlay until the owner retires it.

## What stays bit-exact

- The committed gen-5 source (never edited; renders read `HEAD`).
- Gen-5 rollouts: the gen-5 overlay must render `sim/gen5/identity.json` exactly.
- Gen-5 content: `sim/gen5/data/` plus `calibration_057.json` reproduce today's 1,172 engine values
  (`tests/sim/test_adapters.py`); A1 lists the two engine values the 0.5.x assets contradict
  (`tests/sim/test_a1_gen5.py`).

## What is still hand-maintained (shrink it every port)

| Value | Where | Why not extracted |
|---|---|---|
| gen-5 data comes from 0.5.10, not 0.5.7 | `sim/gen5/data/MANIFEST.json` | no 0.5.7 archive on this machine; `gen5/evidence_057.json` proves 28 files, 240 weapon graphs and 41 auxiliary sources identical to the 0.5.7 pins; Stats files and projectile configs have no 0.5.7 pin |
| sword Swing_Left/Right knockback `Set` (assets: `Add`) | `gen5/calibration_057.json` | gen-5 engine value kept for identity |
| `VARIANT_DAMAGE_KEYS` / `VARIANT_DEFAULTED_KEYS`, guard default 7.0 | `common/content/adapters.py` | the variant table's own selection rules (mirrors `tests/combat/assets/weapon_variants/test_asset_scalar_variants.py`) |
| arithmetic of each constant (decimal vs float sums) | `adapters.py` comments | reproduces the engine's literal bits |
| block codec defaults (Empty, Solid, Full, not a door) | `common/content/records.py` `BLOCK_DEFAULTS` | engine defaults the assets omit |
| `TICKS_PER_SECOND` 30, `DODGE_COOLDOWN_SECONDS` 0.35, `DODGE_LAUNCH_TICK`, `DODGE_COST_TICK`, `DODGE_VELOCITY_REMOVAL_SQUARED`, `CONTROL_IMMUNITY_INCREMENT` 25 | engine | engine defaults or code, not assets |
| native-measured ruleset fields (`loaded_delta_seconds`, landing scale, activation ticks, hit delays, `max_speed`) | `rulesets/hytale_0_5_7/*.json` | measured on a live server; re-measure per version |
| NPC `jump_height_parameter` 1.3, attack pauses, chase and maintain-distance details | ruleset | instruction-tree and builder-default values; the role extractor does not expand `Instructions` |
| Region library, path tables, biome table, suites, catch-time horizons | `HytaleRL/artifacts/worldgen/...` | captured from a live server, not asset tables |
| modifier-rune keyword semantics (`Server/Abilities`) | none yet | not extracted |
| `MAX_RESOURCE_TYPES_PER_ITEM` 7 | `jax/crafting/contract.py` | item `ResourceTypes` not in the crafting table |
