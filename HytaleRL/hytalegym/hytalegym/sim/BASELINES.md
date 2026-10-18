# Baselines every render reports its delta against (measured 2026-10-17)

Owner requirement: optimisation is first-class in the port, "from the terrain worlds to other
accuracy checks". These are receipts of the current gen-5 engine; nothing was optimised or changed to
take them. Each render measures the same commands back to back with the committed engine and reports
the delta. The machine was shared: other sessions held the load average between 7 and 17 on 18 cores
throughout, and throughput moved by up to 2.5x between samples, so compare paired runs, not these
absolute numbers alone.

Machine: Apple M5 Max, 18 cores, 128 GiB, macOS 26.5.1 (25F80); Python 3.12.14, JAX 0.11.2 (CPU
backend, `JAX_PLATFORMS=cpu`), NumPy 2.5.3. Repository HEAD during the measurements: `dedd088`
(committed-engine archive) then `b6cafa38` (other sessions committing).

## 1. Combat step: compile time and throughput

Fixed rollout (`HytaleRL/hytalegym/tools/sim_bench/rollout.py`): `default_combat_params(microticks=4,
target_active=True)`, `reset_batch` on `key(570057)`, 64 `step_batch` calls on a fixed action stream;
lowering and compilation timed separately in a fresh process (no persistent cache:
`jax_compilation_cache_dir` is None after import); throughput = batch x 64 / median of 5 passes.

```sh
cd /Users/arush/hydl && git rev-parse HEAD > /tmp/head/HEAD_SHA && git archive HEAD HytaleRL/hytalegym/hytalegym | tar -x -C /tmp/head
cd HytaleRL/hytalegym/tools
JAX_PLATFORMS=cpu PYTHONPATH=/tmp/head/HytaleRL/hytalegym:. ../../../.venv/bin/python -m sim_bench.rollout --batches 8,64 --steps 64 --repeats 5 --out head.json
JAX_PLATFORMS=cpu PYTHONPATH=. ../../../.venv/bin/python -m sim_bench.rollout --batches 8,64 --steps 64 --repeats 5 --out worktree.json
```

| Engine | Batch | Lower (s) | Compile (s) | Lane-steps/s (median) | Load 1/5/15 at start |
|---|---:|---:|---:|---:|---|
| committed (`git archive HEAD` at `dedd088`), sample 1 | 8 | 0.194 | 1.055 | 23,645 | 17.1 / 15.9 / 15.5 |
| | 64 | 0.210 | 1.296 | 70,004 | |
| committed, sample 2 (beside the full combat suite) | 8 | 0.407 | 2.766 | 9,167 | 7.6 / 11.0 / 13.2 |
| | 64 | 0.363 | 2.261 | 39,188 | |
| working tree (HEAD + 53 uncommitted files of other sessions) | 8 | 0.192 | 1.083 | 24,076 | 15.6 / 15.6 / 15.4 |
| | 64 | 0.193 | 1.148 | 90,848 | |

Output fingerprints (SHA-256 over every output leaf) are identical across the two committed samples
and are the identity gate's values (`sim/gen5/identity.json`): batch 8 `0b28448a...2079`, batch 64
`ace7a94e...2670`. The working tree gives different fingerprints (`d8477b63...`, `b02ee274...`): other
sessions' uncommitted changes alter melee-core outputs, which is why renders read `HEAD`.

### Real-time factor and per-step wall time at 30 Hz (production training step)

The agent operates at 30 Hz (owner ruling): one decision per engine tick, budget 33.3 ms. Scene:
`arena.jax_env.make_scene` (the JAX PPO/Arsenal environment the arena tasks train on), flat world,
`microticks=1`, loadout `iron_sword`, `inert_opponent_ability_slots`; legal random actions drawn from
each step's mask outside the timed region; 300 timed steps after 3 warm-up steps; committed engine
(`git archive HEAD HytaleRL/hytalegym/hytalegym arena worlds` at `6b00064`).

```sh
cd HytaleRL/hytalegym/tools
JAX_PLATFORMS=cpu PYTHONPATH=/tmp/head/HytaleRL/hytalegym:/tmp/head:. ../../../.venv/bin/python -m sim_bench.realtime --batches 1,8 --steps 300
```

| Batch | First step incl. compile (s) | p50 (ms) | p95 (ms) | max (ms) | p95 within 33.3 ms | Simulated s per lane-step | Real-time factor per lane | All lanes |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 11.408 | 1.829 | 8.609 | 44.755 | yes | 0.0297 | 10.2 | 10.2 |
| 8 | 15.747 | 2.211 | 2.681 | 11.677 | yes | 0.0307 | 13.6 | 109.0 |

Against the plan's per-tick bar (GEN7-PLAN.md §2c: p95 of the whole per-tick path <= 15 ms at the
deployed batch), the engine step alone used 8.6 ms of it at p95 for batch 1 (max 44.8 ms, above the
33.3 ms tick) under that load; batch 8 used 2.7 ms. Load average 23.6 / 17.0 / 14.7 at the start
(other sessions). Simulated time is each lane's tick
advance times its `last_motion_delta_seconds` (gen 5 mixes 1/30 s and 0.045 s ticks; lanes that end
an episode advance less), so a 1/30 s gen-7 step changes this column by construction. Scene build
4.5 s / 3.3 s and reset 1.8 s / 2.0 s (batch 1 / 8).

## 2. Terrain: Region library and path tables (working tree, read-only tools)

| Measurement | Result | Command |
|---|---|---|
| load all 288 traversal graphs of `region-library-v3-288` (`NativeRegionTraversalGraph.load`) | 1.587 s total, 5.5 ms per graph (import 0.45 s, manifests < 1 ms) | inline timer around `NativeRegionTraversalGraph.load(lib / entry["path"])` for every `traversal-manifest.json` entry |
| materialize one Region fixture, seed 1331158452, batch 16 (`arena.training.runs.pursuit_run._materialize_region`) | 12.92 s cold, 0.07 ms cached; process wall 14.36 s | `PYTHONPATH=.:HytaleRL/hytalegym .venv/bin/python -m agent_skills.training_runs.tools.benchmark_region_loading --seed 1331158452 --batch 16 --output /tmp/region_loading` |
| build one path-distance table, seed 1331158452 (37,406 nodes x 3,394 landmarks, 15.5 MiB) | 4.09 s wall | `PYTHONPATH=.:HytaleRL/hytalegym .venv/bin/python -m agent_skills.training_runs.tools.region_path_distance --seed 1331158452 --output /tmp/path_distance` |

Load average 13.3-14.4 during these runs.

## 3. Accuracy checks

### Source-pin audits (0.5.10 control install; all 11 match their 0.5.7 semantics)

```sh
cd HytaleRL/hytalegym && HYTALE_ASSETS_ZIP=<0.5 Assets.zip> APPDATA=/tmp/appdata-dummy PYTHONPATH=. \
  ../../.venv/bin/python tests/fidelity/surrogate_<name>_source_audit.py --server-jar <0.5 HytaleServer.jar> [--assets <0.5 Assets.zip>]
```

| Audit | This run (s) | Fidelity receipt, 0.5.10 (s) | Fidelity receipt, 0.7.2 (s, all fail) |
|---|---:|---:|---:|
| perception | 0.76 | 0.8 | 0.7 (crash) |
| actor_world_state | 0.64 | 0.7 | 0.7 |
| locomotion | 2.25 | 2.3 | 2.5 |
| navigation | 2.81 | 2.7 | 0.7 (crash) |
| doors | 0.79 | 0.7 | 0.6 (crash) |
| mutable_blocks | 49.15 | 48.4 | 0.6 (crash) |
| explosion_mutation | 0.82 | 0.7 | 0.7 |
| explosion_dynamics | 0.73 | 0.6 | 0.6 |
| crafting | 0.32 | 0.2 | 0.3 |
| inventory | 0.73 | 0.6 | 0.6 |
| entity_geometry | 0.81 | 0.6 | 0.6 |
| total | 59.8 | 58.3 | |

Load average 9.8 falling to 6.6 during this run. Fidelity receipts:
`~/.claude/jobs/31947bd7/tmp/fidelity072/01_source_audits.log`.

### Fidelity differentials on live 0.7.2 (from the fidelity agent's receipts, read-only)

| Run | Duration | How it was obtained |
|---|---:|---|
| server boot to bridge listening | 5.6 s | FIDELITY-0.7.2.md "Live server" |
| `pytest -m native_server tests/fidelity/combat` (42 tests) | 1,694.6 s | junit `R/live/03h_pytest_combat.xml` |
| selected combat compares (12 tests) | 270.0 s | junit `R/live/03i_pytest_selected.xml` |
| flame staff compares (2 tests) | 2.6 s | junit `R/live/03l_flame_staff.xml` |
| one native motion differential (30 or fewer steps) | 7-11 s | spacing of output files in `R/live/motion/` (inferred) |
| native geometry and LOS compare | about 35 s | start and end stamps in `R/live/03c_geometry.log` (inferred) |
| active-combat root-tick characterization (4 samples) | about 46 s | stamps in `R/live/03g_root_ticks.log` (inferred) |
| reset-timing probe | about 5 s | stamps in `R/live/03def.log` (inferred) |

### This port's own tools

| Run | Duration |
|---|---:|
| `content_extract extract`, all nine families, 0.5.10 archive (3.4 GB, hash included) | 6.57 s |
| same, 0.7.2 archive (3.9 GB) | 5.39 s |
| `content_extract.redirects` | 0.44 s |
| `tests/sim` (60 tests: both round trips, A1, A2, adapters, constructs) | 28.4-38.3 s |

## 4. Test suite wall time

`test_ruleset.py` plus `tests/jax/combat`, working tree (the suite is not in git: `HytaleRL/.gitignore`
ignores `hytalegym/tests/**`):

```sh
.venv/bin/python -m pytest HytaleRL/hytalegym/tests/test_ruleset.py HytaleRL/hytalegym/tests/jax/combat -q -p no:cacheprovider -rf
```

| Run | Result | Wall time | Load average |
|---|---|---:|---|
| before this port's changes (started 16:46) | 74 failed, 1006 passed, 1 skipped, 15 xfailed | 2,950.4 s | 11-17 (other sessions) |
| after this port's changes (started 17:40) | 73 failed, 1006 passed, 15 xfailed, 2 errors: the 2 errors are two live-native tests that hit another session's native lease on 127.0.0.1:5556 (pid 39875); run alone afterwards they fail and skip exactly as in the baseline | 2,876.6 s | 6.6 rising to 24.0 |

## 5. Render: identity gate and bench (2026-10-17 late, phase 2a)

Identity gate (`tests/sim/test_render_identity.py`; `sim/common/bench.py` workloads): gen-5 source rendered with
the gen-5 overlay against a `git archive` of the render's commit; 11 outputs, 2,307 arrays, 101,298 elements,
0 mismatched, max |diff| 0.0, equal SHA-256. Negative control (PLAYER_BASE_SPEED 5.5 -> 5.75): 82 arrays differ.
The content agent's rollout fingerprints (`sim/gen5/identity.json`, b8 `0b28448a...`, b64 `ace7a94e...`) are
reproduced through the render by import redirection. Compiled HLO (metadata stripped) is identical for the planned
and general arsenal and melee steps.

Bench, original vs render on a HEAD archive (load 28-39; the HLO is identical, so timing deltas are noise;
receipt `~/.claude/jobs/31947bd7/tmp/render_bench/bench-20261018T010007Z.json`):

| metric | original | rendered |
|---|---:|---:|
| planned step compile, batch 64 (s) | 9.57 | 9.79 |
| planned lane-steps/s, batch 8 / 64 | 6,866 / 23,100 | 7,731 / 24,390 |
| planned per-tick p95, 1 / 8 NPCs (ms) | 2.01 / 3.15 | 2.34 / 3.27 |
| real-time factor, 1 / 8 NPCs | 29.3 / 19.0 | 28.2 / 19.7 |
| general step compile (s) | 25.6 | 26.2 |
| general lane-steps/s, batch 64 | 630 | 624 |
| general per-tick p95, 8 NPCs (ms) | 23.7 | 24.4 |
| Region library load / atlas + path table (s) | 10.6 / 6.8 | 11.0 / 7.0 |

"Planned" is what production training compiles; "general" (no compile flags) is how `rollout_arsenal_batch` and
`step_learner_arsenal_batch` call the step. Planned meets the 15 ms per-tick target; general is over it at 8 NPCs
(under the 33.3 ms budget). Planned and general are not bit-identical (6 of 315 arrays, max 2.4e-7, 64-step
sword-vs-battleaxe rollout). Owner ruling 2026-10-18: a compile-mode difference far below the native differential
bars (5e-4 per step for motion) is acceptable, so callers may use the planned compile; bit-exactness gates keep
comparing like with like (same compile mode on both sides), and a change of mode re-baselines the fingerprints. Opt-in `OPTIMIZED` gen-5
overlay (damage-resistance walk): bit-exact on the identity rollouts and on 10,240 random inputs; thunk events
6,718 -> 4,200 per step at batch 64; un-profiled step 3.46 -> 2.40 ms; +6 % throughput under load.
