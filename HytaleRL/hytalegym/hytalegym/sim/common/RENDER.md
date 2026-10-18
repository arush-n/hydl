# Rendering a generation's simulator

Owner rulings this follows (2026-10-17): "we should not replace the existing code rather make copies
or some rendering that updates it such that the new gen is properly implemented"; "the JAX sim should
be generation agnostic so when future generations release we can easily update everything"; "when
porting you should also be optimizing and trying to make it faster"; "the agent operates at 30 Hz, the
game's tick".

The gen-5 engine under `hytalegym/jax/` is frozen. A generation's simulator package is **rendered**
from that source *as committed* plus the generation's **overlay**, and written to
`sim/genN/engine/` (package `hytalegym.sim.genN.engine`) with a `MANIFEST.json`. The overlay is the
whole port in one reviewable file (`sim/genN/overlay.py`).

```
git <commit>:hytalegym/jax/**  --relocate-->  redirects  -->  hooks  -->  patches  -->  checks  -->  sim/genN/engine/
                                                                                         (all or nothing)
```

## The overlay (`sim/common/overlay.py`)

| Field | What it is | Validation (against the commit, before anything is written) |
|---|---|---|
| `modules` | source roots under `hytalegym/jax/` (`"combat/"` = every module below it) | each root exists; the renderer adds every `hytalegym.jax` module they import, transitively (import statements anywhere, plus `"hytalegym.jax..."` strings that `import_module` consumers use), and every parent `__init__.py` |
| `redirects` | a value the engine reads, replaced by a literal at render time: the right-hand side of a module-level assignment; or one exact `expression` inside a module-level function/assignment, or on one committed `line` (`template` with `{value}`). Value sources: a content table field (`family`/`row`/`field` through `sim.common.content.load(profile).effective_row`); `family="adapter"`: the swap plan's adapter output `adapters.<row>(tables)[field]`; `family="profile"`: an attribute path of the generation's `SimProfile` (`step_duration.loaded_seconds`). `cast` fixes the literal's type (`float`, `int`, ..., `reciprocal_int`/`reciprocal_float` for a tick rate from a step duration) | the module is rendered; a symbol has exactly one module-level definition; an expression occurs exactly once inside it or on its line; a profile attribute or adapter exists; `status` is `active` or `todo` (todo rows are validated for location and listed, not applied) |
| `hooks` | a module-level function replaced, right after its definition, by `construct(original, profile)` imported from `sim/genN/mechanics/`; the construct reads its own `Constructs` switch and returns `original` when the switch is off | the function has exactly one module-level `def`; `switch` is a field of `sim.common.profile.Constructs`; `status` is `active` or `stub` (stubs are listed, not applied) |
| `patches` | exact text replacements (`old` -> `new`), each with the FIDELITY row it answers and the differential that gates it | `old` occurs exactly once in the file at the commit, and still exactly once after the earlier patches to that file |
| `profile` | `"hytalegym.sim.genN.profile:PROFILE"`; the `SimProfile` binds the bridge `ServerProfile` (`server_versions/`) through `SimProfile.server_profile()` | recorded in the manifest with the constructs, the step duration and the server profile |
| `audit` | free-form facts recorded in the manifest (gen 7: step-clock site coverage, swap-plan counts) | - |

Ids are unique across redirects, hooks and patches. Every check runs before writing; `render()` raises
`OverlayError` / `RenderError` and writes nothing if any step did not apply.

**Rows from the swap plan** (`sim/common/content/redirects.json`, written by the content agent's
`tools/content_extract/redirects.py`). `swap_plan_redirects(plan)` turns its 90 records into rows
valued by their adapter: module constants located by symbol, defaults and inline literals by their
committed line and text; the 5 records a render cannot reach (JSON tables, the shared
`combat/assets/weapon_variants` module, a symbol absent at HEAD) are returned separately.
`step_clock_redirects(plan)` turns the plan's `step_duration` sites (362) into the clock redirects:
the sites in the rendered source that take the step clock from a constant (a read of the ruleset
`engine` block, a literal tick rate, a one-tick literal `1.0 / 30.0`, a literal `* 30.0`) become
redirects to `SimProfile.step_duration`; every other site reads one of those and follows.

## What the renderer does (`sim/common/render.py`)

1. Pins the commit (`HEAD` by default) and reads every needed file with `git cat-file --batch
   <commit>:<path>` (equivalent to `git show`; one process for ~400 files). The working tree is
   never read, so uncommitted work from other sessions is never baked in.
2. **Relocation** (automatic, recorded per file): dotted `hytalegym.jax` references become the
   rendered package (imports, `import_module` strings; `hytalegym.jax_cache_bootstrap` is left
   alone); `Path(__file__).resolve().parents[N]` that climb out of the package are re-anchored on
   `hytalegym.__file__` so they name the same directory (gen 5 has one: `world/tokens/__init__.py`,
   `parents[5]` -> the HytaleRL dir); `Path(__file__).with_name(...)` / `.parent` stay local and the
   tracked data next to rendered packages is copied (gen 5: `arsenal/assets/*.json`). Any other
   `__file__` use or `from hytalegym import jax` refuses the render.
3. Redirects (bottom-up per file, so an inserted `# render redirect` comment line never shifts a
   pending line anchor), hooks, patches, each exactly once.
4. Checks: every module compiles; no rendered module imports or names `hytalegym.jax`; every import
   of the rendered package resolves to a rendered module; every declared step applied.
5. Writes through a sibling staging directory and a rename. A previous render (a directory with a
   `MANIFEST.json` of this schema) is replaced; any other non-empty directory is refused.

`MANIFEST.json`: source commit, overlay sha256 (canonical JSON of every field) and audit, renderer
and overlay-module sha256, the profile (constructs, step duration, bound server profile), the
content sha256 when content redirects applied, per file `sha256_before` (the blob) and
`sha256_after` and the list of changes, the applied and pending steps, and `external_back_edges`.

**Isolation, precisely.** HEAD's `hytalegym/__init__.py` imports `envs.hytale_env`, which imports
the original `hytalegym.jax.combat`; every process that imports `hytalegym` therefore also loads the
gen-5 original. What the renderer guarantees is that no *rendered* module imports `hytalegym.jax`.
Shared modules outside `jax/` (`worldgen`, `rulesets`, `combat/assets`, ...) are not rendered; the
manifest lists the five that themselves import `hytalegym.jax` (for example
`worldgen.native_action_resolution` -> `jax.crafting.contract`): on those paths a rendered generation
still reaches gen-5 code. A change a generation needs there is a content redirect or a later
rendering scope, not a patch.

Commands:

```
python -m hytalegym.sim.common.render --generation gen7            # -> sim/gen7/engine/
python -m hytalegym.sim.common.render --generation gen5 --out DIR  # any directory
python -m hytalegym.sim.common.render --generation gen7 --check    # validate + resolve, write nothing
```

Using a render: in-repo renders import as `hytalegym.sim.genN.engine`. A render elsewhere:
`render.mount(DIR)`. Code written against `hytalegym.jax` (for example the existing tests):
`render.redirect_imports(DIR)` before `hytalegym` is first imported (load `render.py` by path; it is
stdlib-only at import). Every `hytalegym.jax.X` the render contains is then the very module object
`<package>.X`; modules it does not contain (`hytalegym.jax.training`, `.evaluation`) load from the
original tree and reach the engine through the same aliases.

## Adding generation N+1

1. `server_versions/vN.py`, `sim/genN/profile.py` (constructs, `step_duration`), the extracted
   `sim/genN/data/`, the swap plan regenerated (see `PORTING.md`).
2. Copy the previous overlay to `sim/genN/overlay.py`; set `generation` and `profile`. The clock rows
   and the swap-plan rows come from `redirects.json`.
3. Activate swap-plan rows whose value the generation changes (adapter output), turn behaviour
   changes into hooks (constructs in `sim/genN/mechanics/`, each naming its differential first), and
   the rest into patches citing the fidelity row and differential. `--check` until it validates.
4. Render, import the package, run the differentials; the overlay delta is the review.
5. The identity gate must stay green for gen 5 on every renderer change.

## The identity gate

`tests/sim/test_render_identity.py` renders gen 5, extracts the render's commit with
`git archive` (read-only), and in one process on that tree runs `bench.identity_rollouts` through
`hytalegym.jax` and through the render: the
`test_null_arsenal_is_bit_exact_certified_melee_across_seeded_rollout` rollout (melee core and
arsenal, 48 steps, batch 4, seeds 570057/913771/173205/271828, 314159, 161803) and the
`test_complete_arsenal_rollout_scan_matches_eager_jax` rollout (iron sword and bombs, keys 70/71,
jitted and eager), plus the params and arsenal runtime configs. Every array is compared byte for
byte (PRNG keys as key data). Renders checked the same way: gen 5 (identity); gen 5 plus the 13
gen-7 step-clock redirects resolved against the gen-5 profile (today's values: the redirect
machinery changes only what the profile changes); `gen5.overlay.OPTIMIZED` (opt-in optimisation
patches); and a negative control with `PLAYER_BASE_SPEED` 5.5 -> 5.75 that must differ. Manual run:

```
PYTHONPATH=<git archive of the commit>/HytaleRL/hytalegym JAX_PLATFORMS=cpu \
  python -P hytalegym/sim/common/bench.py identity --rendered DIR
```

Independent cross-check: the content agent's `tools/sim_bench/rollout.py` run through a render with
`redirect_imports` must print the committed-engine fingerprints of `sim/gen5/identity.json`.

The static half (`tests/sim/test_render.py`) checks that every rendered gen-5 file equals its blob
once the package name is mapped back (the one anchored line excepted), that renders are
byte-deterministic, the refusal rules, and that swap-plan rows activated for gen 5 render the
committed values through the adapters.

## The step clock (owner, 2026-10-17: the agent operates at 30 Hz)

Gen 5 keeps its per-lane clock: `CombatParams.nominal_dt` 1/30 s and `loaded_dt` 0.045 s (ruleset
`engine` block), and a timing profile drawn per lane at reset (`runtime/math.py::_motion_delta`:
profile 0 always nominal, 1 and 2 loaded on odd or even ticks). The value is the profile field
`SimProfile.step_duration` (gen 5: 1/30 s, 0.045 s, 3 profiles; gen 7: 1/30 s, 1/30 s, 1). Of the
swap plan's 362 `step_duration` sites, 314 are in the rendered source: 13 take the clock from a
constant and are redirected (`clock.py:19`, `mechanics/factory.py:77`, `arsenal/programs/timing.py:6`,
`types.py:593/594/595/597/604`, `arsenal/profiles/hytale_0_5_7.py:2465/2652/4244/4254/4784`); 301 read
one of them; 48 are outside the rendered source (shared modules, the ruleset JSON). Only
`loaded_delta_seconds` (0.045 -> 1/30) and the profile count (3 -> 1) change value. Gen 5 has no
redirects (the identity render).

## Optimisation (owner: optimise while porting, bit-exact unless the owner sees the numbers)

`gen5/overlay.py` keeps the identity `OVERLAY` patch-free and adds `OPTIMIZED`: opt-in patches,
offered only while they pass the identity gate. First one: the damage-resistance parent walk
(`mechanics/runtime/statuses.py`) ran `fori_loop(0, DAMAGE_COUNT=15)` per resolution; the walk can
visit at most the longest `inherits` chain of the damage-cause rules `cause_inherits` is built from
(1 at 0.5.7), so the bound is computed from those rules at import (a cycle keeps 15). Later
iterations cannot change the carry, so it is bit-exact by construction; checked on the identity
rollouts and on 10,240 random resolutions with every entry populated.

## The bench (`sim/common/bench.py`)

```
python -m hytalegym.sim.common.bench --rendered DIR [--original head|worktree] [--receipts DIR]
```

Each workload runs in a fresh worker process per variant (CPU, cold JIT, persistent compilation
cache off), the original first and the render right after, on the same tree (default: `git archive`
of the render's commit, so the delta is rendering alone). Two programs are measured, because they
differ by an order of magnitude: `planned` (the step specialised by `plan_arsenal_compilation`, as
`training/multi_actor/runtime.py` compiles it) and `general` (no flags, as `rollout_arsenal_batch`
and `step_learner_arsenal_batch` call it). They are not bit-identical: in a 64-step
sword-vs-battleaxe rollout at batch 8, 6 of 315 arrays differ by at most 2.4e-7 (null-loadout,
sword+bombs and shortbow scenes match), so moving a caller from one to the other is a methodology
change for the owner, not a bit-exact speed-up. Rows: combat-step compile time; lane-steps per second at
batch 8 and 64; the deployed per-tick path (engine step + learner observation-v3 encode) at batch 1
with 1 and 8 NPCs, p50/p95 (best of two interleaved rounds) against the 33.3 ms tick budget (target
p95 <= 15 ms) and the real-time factor (simulated seconds from the step clock the engine used, per
wall second); the Region library load and atlas/path-table build (read-only). It also fingerprints
each compiled step's HLO with source metadata stripped: equal fingerprints mean the original and the
render are the same program, which no load average can blur. Then a `jax.profiler` trace of the
rendered per-tick path with thunk time attributed to engine source lines through the compiled HLO's
stack frames. Receipts (JSON, traces) go to `--receipts` (default `DIR/../render_bench`).
