# Region recapture on a server line

Runbook for `recapture_region.py`, the gate that recaptures the Region terrain on a server line and checks that a fresh bridge reproduces it. The first line run through this tool is 0.7.2 (gen 7). The tool swaps nothing. The swap steps at the end belong to the owner. Paths are relative to `/Users/arush/hydl/HytaleRL/` unless they are absolute.

## When a recapture is needed

Recapture when a server line changes what the Region producer reads. The staged Region libraries checked here are 0.5.7 captures. On 0.7.2 the block ids are renumbered (Rock_Stone 1351 to 1530) and some terrain differs (`hytale-plugin-0.7.2/FIDELITY-0.7.2.md`, row 3a). A recapture is not needed for a training or policy change, or for a same-line change that the compare still passes.

## Before you run it

- Use a free bridge TCP port and game UDP port, and always pass both (`hytale-plugin-0.7.2/BRIDGE-PORTING.md` section 7). Check both with `lsof` first. Never use the reserved ranges that other sessions hold on this Mac (bridge 5570 to 5572, game 25570 to 25573).
- `--run-root` must be a new or empty directory outside any install tree. The logs, launch reports, `compare-difference-report.json`, the contract and `receipt.json` go there.
- The bridge jar must hash to the pin for the line in `hytalegym/hytalegym/server_versions/`. The tool refuses otherwise.
- The compare script, `hytalegym/tests/fidelity/native_region_capture_compare.py`, is in the git-ignored `hytalegym/tests/**` tree (`HytaleRL/.gitignore:50`) and is not tracked. A checkout without it cannot run the gate. This README is the versioned copy of the procedure, not of the compare code.
- `artifacts/**` is git-ignored (`.gitignore:136`). The NPZs, the contracts and their STATUS files stay local.

## Commands

Run from any directory with the repo venv. Replace `<run root>` with a new scratch directory.

    PY=/Users/arush/hydl/.venv/bin/python
    TOOL=/Users/arush/hydl/HytaleRL/hytalegym/tools/worldgen/recapture_region.py

1. Exact digest scope, the default. The capture is written, then a fresh server must reproduce its complete semantic digest. On seed 570057 this scope failed in both of its runs, so it is not a reliable gate on a seed with fluid drift (see below).

        $PY $TOOL --version 0.7.2 --seed 570057 --bridge-port 5556 --game-port 25574 --run-root <run root> --reference /Users/arush/hydl/HytaleRL/artifacts/worldgen/native-region-light-v1/seed-570057/region.npz

2. Contract scope with a fresh capture. `--fluid-only-cap N` writes a v2 seed-projection contract from the passing capture, then compares with it. Use this for a new seed.

        $PY $TOOL --version 0.7.2 --seed <seed> --bridge-port 5556 --game-port 25574 --run-root <run root> --fluid-only-cap 6

3. Compare only, on an NPZ that already exists. `--skip-capture` requires `artifacts/worldgen/region_<version>_seed_<seed>.npz` to exist. It recomputes that file's sha256 and semantic digest from the file, and does not trust an earlier receipt. Pair it with `--fluid-only-cap N`, or give a stored contract with `--seed-projection-contract <path>`. The two contract options are alternatives.

        $PY $TOOL --version 0.7.2 --seed 570057 --bridge-port 5556 --game-port 25574 --run-root <run root> --skip-capture --fluid-only-cap 6

`--preflight-only` checks the pin, the ports, the paths, and a given contract's seed and bridge, then exits before any launch. A run without `--skip-capture` refuses an existing NPZ, so a contract run on an existing capture needs `--skip-capture`. `--reference` adds a read-only per-section and per-cell diff against an NPZ. It is measurement only.

## What each scope certifies

- Exact scope: a fresh process reproduces the capture's complete semantic digest (`--expected-semantic-sha256`), and two captures in one server process agree across a reset.
- Contract scope: the same-world and reset-world stability checks still run. Each live capture's seed projection digest must equal the contract's. Up to `max_non_damaging_fluid_only_cells` cells may differ between the two captures, and only in non-damaging fluid state. The contract's file and complete-digest fields are provenance. The compare does not check them (`hytalegym/hytalegym/worldgen/region/stability.py`, `RegionSeedProjectionContract.require_snapshot_projection`). Every contract run prints `exact_seed_regeneration_certified=false` and `lifecycle_bounds_fluid_drift=false`.
- In a contract run, `exact_artifact_equal=true` compares the two live captures with each other. It does not compare them with the staged NPZ.

## Why the exact scope is unreliable on fluid-drift seeds

On seed 570057, two captures in one server process can differ at a water edge. Four paired compares changed 5, 6, 2 and 0 cells. In each run with changed cells, the changed cells agree once non-damaging fluid state is removed from both sides (`other_differing_cells` = 0). The changed fields were `flags`, `fluid_level`, `fluid_fill_height`, `movement` and `FluidFX`.

The complete digest moves with those cells. Five complete digests were seen across the live captures, and the staged NPZ's digest is one of them. The seed projection digest, which removes only non-damaging fluid occupancy, was the same in every compare that checked it. The exact scope failed in its two runs (5 and 6 cells). The run with no changed cells would also have failed the exact scope, by its expected-digest check: its two captures agreed, but their complete digest differed from the NPZ's. That is inferred from the code path and was not run.

The 0.5.7 evidence for this seed used the projection scope (`artifacts/worldgen/world-evidence-history/EBE226EC/fixtures/native_region_seed_projection_contract.json`, schema v1, cap 3). The contract scope here is the same method.

## How the cap is chosen

The cap is the largest non-damaging fluid-only count observed in the recorded paired compares. For seed 570057 that is 6, from run-compare2, over four compares (0, 2, 5 and 6 cells). The cap is written into the contract, and the compare prints `non_damaging_fluid_only=N maximum=M`. Each run is recorded in `artifacts/worldgen/world-evidence-history/460054A8/STATUS.md`.

The cap is an observed bound, not a statistical one. A later run can exceed 6, and then the compare fails. Re-derive the cap from more measured runs and record each one in that STATUS. Do not raise the cap to make a run pass. The 0.5.7 contract for this seed records cap 3, but its README and STATUS do not say how that cap was chosen.

## What the tool writes

- `artifacts/worldgen/region_<version>_seed_<seed>.npz`. A run without `--skip-capture` refuses to overwrite it.
- Under `<run root>`: `capture.launcher.log` and `compare.launcher.log`, the launch directories, `compare-difference-report.json` (on a mismatch), `native_region_seed_projection_contract.json` (from `--fluid-only-cap`), `offline-diff.json` (with `--reference`) and `receipt.json`.
- The compare takes the native evidence lease for the bridge port (`hytalegym/hytalegym/worldgen/native_lease.py`). Its files live under `HytaleRL/.codex-local/`.

`receipt.json` records the capture (`skipped`, `passed`, `semantic_sha256`, `file_sha256_on_disk`) and the compare: `scope` (`exact_semantic_sha256` or `seed_projection_contract`), the contract's path and file sha256 when one is used, the launcher exit and the pass marker. The top-level `passed` is capture and compare together. In a `--skip-capture` run, `capture.passed` is `null`, because no capture ran.

## Swap steps (owner's; the tool does none of these)

1. Read the receipt, the compare log and the offline diff (sections and cells differing, by field).
2. Publish the line's library through `hytalegym/hytalegym/worldgen/region/recertification.py`, which stages and then promotes. That path runs `verify_installed_world_source_pins` before it touches the listener. `FIDELITY-0.7.2.md` (ad) records it refusing 0.7.2 at audit 1 until the audits are closed. `create_region_artifact_library` writes `sha256_rank_v1` by default, which moves six held-out seeds on 0.7.2. With `inherit_split_from` set to the 0.5.7 manifest and the 20 0.7.2 NPZs as artifacts, it writes `inherited_seed_labels_v1` with the 0.5.7 labels (see "Loading a 0.7.2 library"). Its repository callers all use the rank default (`examples/capture_native_region_library.py`, the random-seed pilot, and `tests/jax/world/region/test_region_library.py`). `recertification.py` does not call it: that file only reads the 0.5.7 pilot (`_recapture_world_evidence_unlocked`, `derive_contract_cascade`) and writes candidate evidence, so the 0.7.2 publish does not pass the option yet.
3. Rebuild the products that depend on the terrain: path tables, task suites and catch-time horizons (`hytale-plugin-0.7.2/SIMULATOR-0.7.2.md`, P2). The repository generators are `tools/worldgen/region_path_distance.py`, `tools/worldgen/task_zoo.py` and `tools/worldgen/catch_time_horizons.py` (each takes `--server-version`, default 0.7.2; see "Products derived from a library" below, committed 3cd8c06 on 2026-10-18). The biome table has no consumer (below). The Arsenal Region suites are derived at call time and pinned by SHA in `jax/evaluation/_contracts.py`, so they need a new frozen version once a 0.7.2 traversal library exists, not a rebuild.
4. Flip the defaults that point at `region-library-pilot-v2`: `hytalegym/hytalegym/jax/world/tokens/__init__.py` (lines 87 to 95), `hytalegym/hytalegym/worldgen/region/recertification.py` (lines 141 and 791 to 797) and `hytalegym/hytalegym/sim/common/bench.py` (line 62).
5. Keep the 0.5.7 libraries and the staged NPZs. The provider finds a library by its semantic hash under `HYTALERL_WORLD_ARTIFACT_ROOT` (`hytalegym/hytalegym/jax/world/region/provider.py`), so old and new libraries can sit side by side. Ambiguous configured artifacts raise. The 0.7.2 library's split labels are inherited from the 0.5.7 library in the same folder, so that folder must stay beside it; without it the 0.7.2 manifest is refused.

Open question before step 2 (owner's call): each library manifest records every NPZ's complete `semantic_sha256`. A library built from the staged 0.7.2 NPZ would therefore carry the NPZ's complete digest, which a fresh capture on this bridge does not reproduce. The projection digest is the stable value for this seed. Which digest identifies the 0.7.2 library is not decided here.

## Identities

The 0.7.2 bridge pin, the seed, the staged NPZ and its digests, the contract file and the compare runs are recorded in `artifacts/worldgen/world-evidence-history/460054A8/STATUS.md`. They are volatile, so they stay out of this README.

## Products derived from a library

Scope as of 2026-10-18, from a read-only pass over the tracked code. It covers the four products that a Region recapture makes stale (`hytale-plugin-0.7.2/SIMULATOR-0.7.2.md:92-96`). Paths are relative to `/Users/arush/hydl/HytaleRL/` unless absolute. This section corrects step 3's statement that no generator exists for the task suites: the arsenal Region suites have one in the repo (`hytalegym/hytalegym/jax/evaluation/_contracts.py`).

**Runtime-derived, nothing stored.** The traversal atlas that `hytalegym/hytalegym/sim/common/bench.py` times as "atlas + path table build" is built at load time from the traversal graphs (`hytalegym/hytalegym/jax/world/region/library.py:233-253`). Rebuilding the library rebuilds it. There is nothing to regenerate.

### Path-distance table (frozen, keyed to one Region)

- Read by `arena/training/runs/pursuit_run.py:2677-2691` through `arena/training/skills/path_distance.py`. A table is refused unless its `region_semantic_sha256` equals the loaded Region's.
- Generator: private, `agent_skills/training_runs/tools/region_path_distance.py`, documented in `hytalegym/hytalegym/sim/BASELINES.md:77`. Not run or read in this lane. The command as documented, run from `/Users/arush/hydl`:

        PYTHONPATH=.:HytaleRL/hytalegym .venv/bin/python -m agent_skills.training_runs.tools.region_path_distance --seed <region seed> --output <new file>

- Before a 0.7.2 build, confirm which library the private tool reads (tracked code does not say). Do not build until the owner has chosen the 0.7.2 library digest (`hytale-plugin-0.7.2/FIDELITY-0.7.2.md:1750`).
- Repository generator, version-controlled: `tools/worldgen/region_path_distance.py`. Build one seed from any directory with the repo venv:

      /Users/arush/hydl/.venv/bin/python /Users/arush/hydl/HytaleRL/hytalegym/tools/worldgen/region_path_distance.py --seed <region seed> --server-version 0.5.7 --output <new file>.npz

  The default `--server-version` is 0.7.2 (gen 7), which reads `artifacts/worldgen/region-library-pilot-v2-072`. `--library DIR` names another root, and `--dry-run` stops before the search. The landmark rule and the quantization are in the tool's docstring, with the inferred constants named there. Checked on 2026-10-18 for 0.5.7 seed 1618574389 (train): 39,836 nodes, 3,584 landmarks, 25.5 MiB, search 162 s and write 26 s at load about 50. The consumer accepts it for that Region and refuses seed 1044320000. The 0.7.2 table is not built: that library has a manifest but no `traversal-v2` yet.
- Decisions (coordinator, 2026-10-18, tool output only, nothing trains on it yet): distances beyond 127 blocks stay 255, the
  consumer's documented "unreachable" (saturating at 254 would remove the -clip cliff in `arena/training/contracts/pursuit.py`
  625-642 but flatten progress past the band); landmark cells stay 4 blocks, the consumer's constant (3,584 landmarks on seed
  1618574389; 25 MiB; about 200 s to build at load 50). Both are documented in the tool's docstring and open to change.


### Arsenal Region suites (derived at call time, pinned by SHA)

- Generator in the repo: `region_vN_suite_contract(traversal_manifest_path=...)`, v1 to v6, in `hytalegym/hytalegym/jax/evaluation/_contracts.py`. The checked v6 wrapper raises on drift (`_contracts.py:809-821`).
- One-command check. It prints the success line and exits 0 when v6 reproduces its frozen SHA:

        PYTHONPATH=/Users/arush/hydl/HytaleRL/hytalegym /Users/arush/hydl/.venv/bin/python -c "from hytalegym.jax.evaluation import region_v6_suite_contract as f; f(traversal_manifest_path='/Users/arush/hydl/HytaleRL/artifacts/worldgen/region-library-pilot-v2/traversal-v2/traversal-manifest.json'); print('v6 reproduces its frozen SHA')"

- Status on 2026-10-18: this raises `drifted` for v6. An unchecked recomputation also fails for v1 to v5 on both pilot-v2 traversal manifests. The cause is not isolated. Fix this before any suite rebuild.
- A 0.7.2 suite cannot come from the current code. `REGION_LIBRARY_SEMANTIC_SHA256`, `REGION_CAPTURE_EVIDENCE_BRIDGE_SHA256` and `FROZEN_REGION_WORLDS` are 0.5.7 values (`_contracts.py:72-79, 134`), and a new frozen version is needed (`_contracts.py:816-821`). That is a code change in `hytalegym/jax/evaluation`, the owner's call.

### Reach/pursue task zoo (tracked tool, 2026-10-18)

- Generator: `tools/worldgen/task_zoo.py`, a tracked rebuild of the private zoo (`agent_skills/training_runs/tools/task_zoo.py`, which was not read or copied). It is imported by `arena/tests/test_private_task_suite.py:3`, and the test pins the suite shape: terrains, tasks, horizon tags and the reserved audit seeds. Commands, with `PY=/Users/arush/hydl/.venv/bin/python`:

        $PY /Users/arush/hydl/HytaleRL/hytalegym/tools/worldgen/task_zoo.py --server-version 0.5.7 --json <new file>
        $PY /Users/arush/hydl/HytaleRL/hytalegym/tools/worldgen/task_zoo.py --server-version 0.7.2 --json <new file>

- Counts on 2026-10-18, at the default 512-tick horizon: 0.5.7 gives 16 train terrains in each of 3 families (48 tasks). 0.7.2 gives 13 per family (39 tasks). Its manifest labels and the package's split rule disagree on six seeds, and the tool excludes the union (`split_disagreements` in the output).
- The horizon band is 512 to 1000 ticks, pinned by the test (1000 is the runner ceiling). The default 512 ticks is 23.04 s on 0.5.7 and 17.07 s on 0.7.2.
- The 0.7.2 library is staged, not promoted. Its STATUS.md says its manifest fails `RegionArtifactLibrary.load` (a chunk API fingerprint check, and the split labels), so the 0.7.2 output is provisional.
- The rules that the test does not pin, the reserved audit seeds and the task settings are stated in the module docstring.

### Catch-time horizons (constants in code; tracked tool for the rule)

- Consumers: `arena/curriculum/pursuit.py:373-378` and `:635-639` (rung values); `arena/training/runs/pursuit_run.py:647-649` (refuses more than 1000 ticks); `console/core/execution/jobs.py:585` (refuses more than 512).
- The 1024-tick rung values are refused by the runner; a constructor check confirmed it. Raise the ceilings or change the rungs before any rebuild.
- Generator: `tools/worldgen/catch_time_horizons.py`, a tracked stand-in for the private `catch_time.py` (`experimental/neural_agent/skill_agent/docs/BUILD.md:147, 221`; not read or copied). Input: a JSON array or JSON Lines file of `{"seconds": ...}`, one per successful episode (`--example` prints one). The repository holds no measured times yet. Command:

        $PY /Users/arush/hydl/HytaleRL/hytalegym/tools/worldgen/catch_time_horizons.py --input <times file> --server-version 0.7.2 --quantiles 0.5,0.9,1.0 --json <new file>

- The rule is an inferred choice, stated in the module docstring: nearest-rank quantiles of the measured ticks, rounded up to a power of two, then capped. Reproduction from the comment's printed times (4.3, 8.5 and 17 s, one record each, `--quantiles 1.0`): at 0.045 s the rungs are 128, 256 and 512; at 1/30 s they are 256, 256 and 512, because 4.3 s is 129 ticks. With a log-scale nearest-power-of-two rounding instead (a scratch check, not in the tool), both bases give 128, 256 and 512, so the rounding rule decides which base reproduces the band.
- Ceilings (confirmed by reading): runner 1000 (`arena/training/contracts/pursuit.py:60`, enforced at `arena/training/runs/pursuit_run.py:647-649`); console 512 (`console/core/execution/jobs.py:585`, refused at `:612-615`). The tool caps at the lower one, 512, and reports the cap.
- The tick base differs. The curriculum comments use 30 Hz (`arena/curriculum/pursuit.py:43-44`). The contract comment says one environment step is the loaded motion tick, 0.045 s (`arena/training/contracts/pursuit.py:151-152`). Gen 7 is one 1/30 s step (`hytalegym/hytalegym/sim/gen7/profile.py:71`). Gen 5 steps 1/30 s on profile 0 and 0.045 s on profiles 1 and 2, on alternating ticks (`hytalegym/hytalegym/sim/common/profile.py:155-164`). State the base in any rebuild.

### Biome table (no consumer)

- No tracked Region, sim, training, example or adk code reads a biome table, and the Region NPZ has no biome member. The console's `BIOMES` tuple (`console/core/worlds/worldgen.py:111`) is a design-preview palette, not a Region product. Name the consumer before anyone builds this table.

## Library recapture (pilot v2, 0.7.2)

The twenty seeds of `region-library-pilot-v2` were recaptured on 0.7.2 with `recapture_region.py --fluid-only-cap 6` and each seed's manifest core passed as `--core-min-chunk-x/-z`. Three bridge and game port pairs ran in parallel, one seed at a time per pair. Each seed ran one capture and one contract-scope compare. The compare script always uses the server's default core, so a manifest core matches only when the default equals it. The seed projection digest includes the core, so each compare pass confirms the match for its seed. One compare launch failed on `EADDRINUSE` at the launcher's bind probe, most likely TIME_WAIT left on the bridge port by the previous server (inferred from the timing, not logged), so the tool now waits up to 180 s for both ports before each launch. The failed seed's compare was re-run once on the same NPZ with `--skip-capture`. The cap stayed at 6 for every seed. The staged folder is `artifacts/worldgen/region-library-pilot-v2-072/` (NPZs, a 0.5.7-schema manifest with the 0.5.7 seed splits, one contract per seed). Its `STATUS.md` holds the per-seed table, the commands and the validation. The staged manifest loads through `RegionArtifactLibrary.load` since the loader change in the next section. The source-pin gate blocks recertification on 0.7.2, so nothing is promoted. The driver and staging scripts are not in the repository.

### Loading a 0.7.2 library

`RegionArtifactLibrary.load` (`hytalegym/hytalegym/worldgen/region/library.py`) certifies the chunk API for the manifest's own `capture_contract.server_version`. `certified_chunk_api_manifest(server_version)` (`hytalegym/hytalegym/worldgen/region/contract.py`) returns the certified manifest for 0.5.7 or 0.7.2 and raises for any other version, 0.6.0 included. `CERTIFIED_CHUNK_API_SHA256` pins the two fingerprints. The no-argument call still means 0.5.7, so callers that pass no version do not move. The second change is a second split method, `inherited_seed_labels_v1`. Its `split_contract` holds `method`, `salt_sha256` and `source_library_semantic_sha256`; the last two are copied from the source library. The loader finds that source as the provider finds libraries: exactly one sibling folder of the manifest's folder must hold a `manifest.json` whose `library_semantic_sha256` equals the recorded digest. The loader loads the source without its artifacts, so the source's own digest is checked. The source must use `sha256_rank_v1`, and it must have the same salt, the same seeds and the same held-out labels. `sha256_rank_v1` is unchanged. The staged 0.7.2 manifest uses the inherited method because `sha256_rank_v1` over the 0.7.2 hashes moves six held-out seeds (`artifacts/worldgen/region-library-pilot-v2-072/STATUS.md`). The 0.5.7 folder therefore has to stay beside the 0.7.2 folder in `artifacts/worldgen/`. `create_region_artifact_library(root, paths, inherit_split_from=<0.5.7 manifest>)` writes this method. It copies the salt, the held-out count and the source digest from the source, and it refuses unless the artifact seeds equal the source's seeds and the source uses `sha256_rank_v1`. It also refuses before writing unless the loader would find that source as the sibling by digest. Its default, `sha256_rank_v1`, is unchanged. The provider (`hytalegym/hytalegym/jax/world/region/provider.py`, unchanged) finds a library by `library_semantic_sha256` under `HYTALERL_WORLD_ARTIFACT_ROOT`; it does not read the server version, and the manifest's own version selects the certified chunk API. Its default `require_traversal=True` still falls back to the surrogate for 0.7.2, because no 0.7.2 traversal library is staged. `tools/worldgen/task_zoo.py` (`package_heldout_seeds`) follows the manifest's method: under `inherited_seed_labels_v1` the rule is the certified labels, so the 0.7.2 zoo reports no split disagreements (16 train terrains and 4 held out per suite, 2026-10-19); under `sha256_rank_v1` it still re-derives the rank, and any other method fails closed.

## Recertifying a server line

`hytalegym/hytalegym/worldgen/region/recertification.py` (schema `hytalerl_world_evidence_recertification_v12`) stages the bridge-bound world evidence of one server line against one Region library: the seed-570057 projection, the stationary traversal, the traversal-graph library (built natively from the live bridge), the navigation and walk probes, and nine world-action probes. It stages only. A run publishes nothing (`publication.automatic` is false, `recertification.py` L544). It holds the native-evidence lease for its listener, so one run at a time drives a listener.

Line numbers are for `recertification.py` after the first-capture change of 2026-10-19. Status is as of 2026-10-19.

Current state:

- 0.5.7: compare mode, unchanged. All four references exist, so the run is the same as before. The cascade output is byte-identical before and after the change. The real 0.5 dry run stops at the bridge-identity check, because no 0.5 bridge build exists on this machine (see "Dry runs").
- 0.7.2: first-capture mode, implemented and tested with synthetic fixtures (no server). A real 0.7.2 run has not been made. The source-pin gate stops it first: at the committed pins (HEAD 7a260d2) at audit 1 (perception); at the audit lane's uncommitted working-tree pins at audit 6 (mutable_blocks). See "Dry runs".

### Per-line fixture root

`world_fixture_root()` (`hytalegym/hytalegym/server_versions/__init__.py`) returns the recorded World evidence folder of a line. The 0.5 line reads `hytalegym/tests/fidelity/fixtures/world/`, unchanged. A line whose `ServerProfile.world_fixture_root` is set reads that subfolder: 0.7.2 reads `hytalegym/tests/fidelity/fixtures/world/v0_7/`. The folder does not exist yet, so a 0.7.2 run treats every reference as absent.

Routed readers (each opens a file under the fixture folder, and each now calls the helper; 0.5 paths are byte-identical):

- `hytalegym/hytalegym/worldgen/region/recertification.py` (`fixture_root`)
- `hytalegym/tests/worldgen/test_region_recertification.py`, `test_native_world_recordings.py`, `test_region_state_change_use.py`
- `hytalegym/tests/jax/world/navigation/test_navigation_native_recording.py`, `hytalegym/tests/jax/world/region/test_region_library.py`, `hytalegym/tests/jax/combat/arsenal/tests/agents/ppo/test_training.py`
- `hytalegym/tests/sim/gen5_live.py` (the crafting recording; a 0.5 script)

Not routed, because they are records and not readers: `hytalegym/hytalegym/sim/common/content/adapters.py` (the `SWAP_PLAN` row for the crafting catalog) and `hytalegym/hytalegym/sim/common/content/redirects.json` (generated from `SWAP_PLAN`), and the metadata names in `hytalegym/worldgen/native_traversal.py` and `hytalegym/jax/world/navigation/__init__.py`. The arena and console tests that name `region-library-pilot-v2` read the Region library folder, not the fixtures.

### Reference modes

The run decides its mode from four references of the line (`_reference_mode`, def L1026):

1. the projection contract, `fixture_root/native_region_seed_projection_contract.json`;
2. the stationary recording, `fixture_root/native_traversal_projection_v2_recording.json`;
3. the capacity report, `_world_capacity_report`: `artifacts/worldgen/world-token-capacity-v4.json` for 0.5, `artifacts/worldgen/region-library-pilot-v2-072/world-token-capacity.json` for 0.7.2;
4. the traversal manifest, `<library>/traversal-v2/traversal-manifest.json`.

- All four present: `compare`. The run is the same as before. The cascade is written before the listener, the candidates are relabelled from the frozen fixtures, and the stationary and graph steps compare against them.
- None present: `first_capture`. The run captures its own references, as described below.
- Some present: refused with `partial World reference set for this line; missing: ...`. A run never compares some references and captures the others.

First capture, in order (the report carries `reference_mode: first_capture` and a `first_capture` block):

1. The gate, bridge identity and mode decision run as before (L206-263). The run requires `--seed-projection-contract` and `--seed-projection-npz`; without them it stops before the listener.
2. The Region library is checked against the line's expected digest, `ServerProfile.region_library_sha256` (0.7.2: `9DCA3C71...DAAB95`). A different library stops the run before the listener (`_require_expected_library`, L269).
3. The listener is checked (L281). No cascade is written yet.
4. The staged projection contract is staged (`_first_capture_projection_contract`, def L1067). It must be for seed 570057 and for the bridge; its reference digest must equal `ServerProfile.seed_projection_sha256` (0.7.2: `408D5FB2...D1104D`); and `require_snapshot_projection` must accept the staged NPZ. The contract is then copied to `stage/fixtures/`. The projection reference therefore comes from the staged line contract, not from the 0.5.7 fixture.
5. The seed-570057 region projection is compared with that contract (`region_seed_projection`, as in compare mode).
6. The capacity report is produced for the line (`world_token_capacity`, `tests/fidelity/surrogate_world_token_benchmark.py` on the Region library, no server): `stage/world-token-capacity.json`.
7. The stationary traversal is captured twice (`stationary_traversal_1` and `stationary_traversal_2`, `--recording-directory stage/stationary-N`). Neither has `--expect-recording`. After both finish, `_require_stationary_repeat_stable` requires the reported fields (except `captured_at_utc`, the artifact and the report digest), each artifact's sha256, and the arrays to agree (floats within 1e-12) before anything is staged.
8. The traversal graph library is built natively (`region_traversal_graphs`), with no `--expect-manifest`.
9. The navigation and walk probes and the nine world-action probes run as in compare mode.
10. The cascade is written last, reading the staged `stage/traversal-v2/traversal-manifest.json` (`derive_contract_cascade(..., traversal_manifest=...)`).
11. The loaders and the graph's bridge check run, and the report is written with the 19 candidate rows (the 18 compare rows plus `world_capacity_report`).

The first capture uses expected digests only in this mode. In compare mode no digest is checked, as before.

### Inputs

| Input | Value | Read by |
|---|---|---|
| Server line | `HYTALERL_SERVER_VERSION=0.7.2`; unset means 0.5.7 | `server_profile()` (`hytalegym/server_versions/`) |
| Installed server jar | `/Users/arush/Library/Application Support/Hytale/install/release/package/game/latest/Server/HytaleServer.jar`, sha256 `23f71f03763422ce7c04df191db49cd0e7d87d43b12baad75c43fdc202410aff` | `--server-jar`; `world_source_pin_staleness_report` (`source_pins.py:159`) |
| Installed Assets.zip | `/Users/arush/Library/Application Support/Hytale/install/release/package/game/latest/Assets.zip`, sha256 `ee8f8d5b332874eb928144726077fa66f80c117a5412017a5cde6fce2d9bb2a0` (3.9 GB) | `--assets`; the audits that take it (`source_pins.py:20-75`) |
| Pinned bridge | `hytale-plugin-0.7.2/build/libs/HytaleRLBridge072-0.1.0.jar`, sha256 `460054A8E7508FE5E205AD527AFB08474B61626EE5C2212FC2A484ECFCCAD88D` (pin: `v0_7.py`) | `--expected-bridge-sha256` (required); the deployed file must hash to it |
| Deployed bridge | the copy the live server loaded. With `tools/launch_bridge_072.py` that is `<run dir>/mods/HytaleRLBridge072-0.1.0.jar` | `--deployed-bridge`. The default is an APPDATA path, a Windows install path, so pass it on macOS |
| Live listener | bridge TCP `127.0.0.1:5556` (`--host`, `--port`), already listening | `_require_listener` (def L1228). The CLI has no game-port flag; the game UDP port, 25574, is set when the server starts |
| Native-evidence lease | `HytaleRL/.codex-local/native-world-recertification-127_0_0_1-5556.lock` (the `.json` signal exists only while a run holds it) | `acquire_native_evidence_lease` (L126); a run refuses when another holds it |
| Control seed | `--control-seed N`: a positive int32, not 570057. Omitted, a random one is drawn and recorded | `_control_seed`, `_random_control_seed` |
| Region library | 0.7.2: `artifacts/worldgen/region-library-pilot-v2-072/` (manifest sha256 `c30459089c7f2c682ca411e5a8d9c46f81a8fa69eb39d2c82f9041310c2a8243`). 0.5.7: `artifacts/worldgen/region-library-pilot-v2/` | `--region-library DIR`; default by server line (`_default_region_library`, def L999) |
| Expected library digest | 0.7.2 `9DCA3C71D43A2C3BF4BE6AFF6E2BD24D2F6608C8BE01D1687A0B259F06DAAB95` (`ServerProfile.region_library_sha256`); 0.5.7 unchecked | first capture only (`_require_expected_library`) |
| Seed-570057 projection reference | 0.7.2 `408D5FB2DAA797E767AEE23248C6591FC226E1FE764C6258353D60FE49D1104D` (`ServerProfile.seed_projection_sha256`). The staged contract is `artifacts/worldgen/world-evidence-history/460054A8/fixtures/native_region_seed_projection_contract.json` (cap 6). 0.5.7 `DB4B9B85...` (fixtures, compare mode) | first capture: `--seed-projection-contract` (checked against `--seed-projection-npz`, the staged `artifacts/worldgen/region_0.7.2_seed_570057.npz`) |
| Frozen references (compare mode only) | `hytalegym/tests/fidelity/fixtures/world/` (0.5) | `build_evidence_candidates` |
| Traversal library | none to read; the run builds `traversal-v2` from the library's NPZs | `region_traversal_graphs` (`native_region_traversal_graph_compare.py`) |
| Python and Java | `/Users/arush/hydl/.venv/bin/python` (3.12); `java` on PATH (OpenJDK 25.0.2 here) | every step; `launch_bridge_072.py` |

Set `APPDATA="/Users/arush/Library/Application Support"` for every gate run on this Mac. The source-pin lane now gives the audit subprocesses a placeholder APPDATA, so the crafting audit no longer needs it, but the test process and any direct run still benefit from it.

### Outputs

The stage directory must be absent or empty (or reused with `--resume`), and it must lie outside `fixture_root` and the library's `traversal-v2` (`_prepare_stage`). Keep it under the scratch folder.

- `source-pin-staleness.json` on a pass, or `source-pin-staleness-failure.json` on a failure.
- `contract-cascade.json`: relationships, identities and dependency order.
- `native_region_candidate_seed_570057.npz` and `native_region_difference.json`: the seed-570057 projection capture and its difference report.
- `fixtures/`: the candidate JSON fixtures (the `publication.targets` in the manifest). In compare mode these are relabelled from the 0.5 fixtures. In first capture the only file is the staged projection contract.
- First capture only: `stationary-1/` and `stationary-2/` (each a JSON and an NPZ), `world-token-capacity.json`.
- `traversal-v2/`: `graphs/` (one `.traversal.npz` and one row `.json` per library entry, 40 files for 20 seeds), `traversal-manifest.json`, `native-traversal-recording.json`. Total 42 files.
- `recertification-manifest.json`: `status` `validated_candidate_not_published`, the candidate rows with sha256 (18 in compare mode, 19 in first capture), the publication targets, `publication.automatic` false. First capture adds `reference_mode` and a `first_capture` block (the file hashes of the staged contract, the stationary pair and the capacity report, and the two expected digests).

Nothing is written under `artifacts/worldgen/world-evidence-history/`, under the fixture folder, or in a library folder.

### What promote means here

Promotion is a manual copy into the line's own folders. No tracked code does it: a search of `hytalegym/`, `arena/` and `console/` for the fixed paths, the receipt schema and copy calls found no writer to these paths (the console's copy calls move agent evidence). The private `agent_skills/` tree, at the repository root, is out of bounds for this lane and was not searched. `recertification.py` keeps promotion separate on purpose (`publication.reason`).

For 0.7.2 (a new line, so nothing is overwritten and there is no superseded set to preserve):

1. `hytalegym/tests/fidelity/fixtures/world/v0_7/`: the 13 JSON targets under their names in `publication.targets`, and the stationary artifact `native_traversal_projection_v2_recording.npz`.
2. `artifacts/worldgen/region-library-pilot-v2-072/traversal-v2/`: the 42 candidate files.
3. `artifacts/worldgen/region-library-pilot-v2-072/world-token-capacity.json`: the candidate capacity report (first capture).

The 0.5 set is not touched: it stays at the fixture root and in `world-evidence-history/5A54A8F5/`. Pointers and defaults stay as they are; the swap (README step 4) is an owner decision. Write a receipt `publication-vN.json` in the line's history folder, and update the local STATUS files.

Rollback for 0.7.2: remove the three added locations. They are new, so nothing else is restored. Deleting them needs an owner yes.

#### Promotion tool (2026-10-19)

`tools/worldgen/promote_world_evidence.py <stage> --server-version 0.7.2` is the manual copy made checkable. Dry run by
default: it recomputes every candidate row's sha256, runs the pre-promotion checks below (manifest status, bridge pin,
installed line, the `first_capture` digests against `ServerProfile.region_library_sha256` and `seed_projection_sha256`,
the two stationary captures byte-identical, cascade relationships and identities, traversal manifest bridge and source
library, `graph_count` against the entries and the recording's `artifact_count`, `full_edge_census`, the seed-projection
difference report within its cap with zero other differing cells, the capacity report's library and seed count, the
stationary recording `exact`, the source-pin report status), prints the copy plan with each destination's state (absent,
identical, exists, differs), and refuses when any check fails or any destination is present with other bytes. With
`--apply` it copies the targets into the line's folders (`world_fixture_root(version)`, `<library>/traversal-v2/`, the
profile's capacity report path), copies the manifest, the source-pin report, the cascade and the difference report into
`world-evidence-history/<bridge8>/` with the schema version suffix, and writes `publication-v<N>.json` there with the
checks and `created_paths` (the rollback list). Smoke test on 2026-10-19: against the 0.5.7 publication folder
`5A54A8F5` as a stage, all 26 checks pass and the plan reports every fixture identical and `traversal-v2` present, so it
refuses, as it should. Pointers and defaults are never touched by the tool.

Who reads the promoted 0.7.2 evidence: the routed readers, when run with `HYTALERL_SERVER_VERSION=0.7.2`. The sim and the JAX world tokens read the Region library by fixed path (`hytalegym/jax/world/tokens/__init__.py:87-96`, `hytalegym/sim/common/bench.py:62-63`), and those pointers stay on 0.5 until the swap. `hytalegym/jax/world/region/provider.py` selects a library by its digest and does not read these paths.

### Gates, in order

Line numbers are in `recertification.py` unless a file is named.

1. Lease: `recapture_world_evidence` acquires the listener's lease (purpose `world_recertification`). It refuses when another run holds the listener.
2. Stage: `_prepare_stage`, outside the published roots; absent or empty unless `--resume`.
3. Paths: the deployed bridge, the assets and the server jar defaults (L171-192).
4. Source pins, L206 (`verify_installed_world_source_pins`, `source_pins.py:80`): 11 audits in `_AUDITS` (`source_pins.py:20`), perception first. The first audit whose semantic hash misses raises (`source_pins.py:133-138`). The except block writes the failure file and re-raises.
5. Installed build (`source_pins.py:159`): the release line must equal the profile's certified version, and each audit's server sha must equal the installed jar.
6. Bridge identity, L229-254: the line's pin must equal `--expected-bridge-sha256`, and the deployed file's sha must equal both.
7. Mode decision, L258 (`_reference_mode`); first capture also needs the two projection inputs, and the library digest check (L269).
8. Cascade: compare mode writes it before the listener (L271). First capture defers it to step 13.
9. Listener, L281.
10. Candidates, L296-313: compare relabels the frozen fixtures (`build_evidence_candidates`); first capture stages the projection contract (`_first_capture_projection_contract`).
11. Live compares, each `check=True`, in order (L330-386): `region_seed_projection`; `world_token_capacity` (first capture only); `stationary_traversal` (compare) or two captures (first capture); `region_traversal_graphs` (with `--expect-manifest` in compare only); the navigation probes; the nine world-action probes.
12. First capture only: `_require_stationary_repeat_stable` (L428), then the cascade (L427-446).
13. Loaders, L446 and after: `RegionArtifactLibrary.load` and `RegionTraversalGraphLibrary.load`; the graph's `native_evidence_jar_sha256` must equal the expected bridge.
14. Report, L637: `recertification-manifest.json`.

### Duration

The 0.5.7 history (file times, local CDT, from the source-pin report to the candidate manifest, since no start time is recorded): `D954BD79` 12 min 44 s; `1C063331` 24 min 31 s; `5A54A8F5` (v12) 23 min 23 s. The graph step inside them is about 8 minutes (compile 423.6 s plus native capture 68.6 s for 20 artifacts).

First capture adds the capacity benchmark (measured 2026-10-19 on the 0.7.2 library: 43 s real time, exit 0, 20 seeds, JAX CPU) and the second stationary capture. The second capture is not timed on the 0.7.2 bridge. Planning figure: 25 to 35 minutes from the gate to the manifest, plus up to 180 s for the server to boot.

### Dry runs (2026-10-19)

- 0.7.2 (first-capture inputs given, so the gate is the first stop): `HYTALERL_SERVER_VERSION=0.7.2`, pinned bridge via `--deployed-bridge`, `--expected-bridge-sha256 460054A8...CAD88D`, `--port 5556`, `--control-seed 481374900`, `--seed-projection-contract .../460054A8/fixtures/native_region_seed_projection_contract.json`, `--seed-projection-npz artifacts/worldgen/region_0.7.2_seed_570057.npz`. No server was launched. Result: Final run (20:35 UTC, the final code): `RuntimeError: World source-pin audit failed (mutable_blocks): mutable block sources differ from Hytale 0.5.7`, exit 1, raised by the gate's verify call (L206). The stage holds `source-pin-staleness-failure.json` only. The gate passed perception, actor_world_state, locomotion, navigation and doors, because the audit lane's working-tree pins (`hytalegym/hytalegym/worldgen/source_pin_baselines_072.json`, modified and uncommitted) now pin them. The committed pins (HEAD 7a260d2) pin only inventory and explosion_dynamics, so there the gate stops at perception. The first-capture inputs were parsed; the gate stopped the run before they were checked. Before the audit lane's working-tree pins (20:27 UTC, same code except the report line), the stop was perception.
- 0.5.7 (the real 0.5 install, `v0.5` Assets and server jar, `--expected-bridge-sha256 7700D112...8D7B`): the gate passes on the 0.5.10 control. The run then stops at the bridge-identity check, because no 0.5 bridge build is on this machine. Result: `RuntimeError: deployed bridge differs from canonical world evidence: observed=460054A8E7508FE5E205AD527AFB08474B61626EE5C2212FC2A484ECFCCAD88D expected=7700D112F8BCE9A6AFFE5EEB765867E83138F67070FA82408D08E5AB948F8D7B`, exit 1, raised at L253 after the change (L232 before; the same message, moved by the helpers added above it). The gate passed on the 0.5.10 control, and the stage holds `source-pin-staleness.json` only. The same line appeared before the change.

### Blockers and what changed

The earlier blockers (the cascade before the traversal library, the 0.5.7 projection reference, the absent expected manifest, the 0.5.7 stationary recording, the 0.5.7 capacity report, the single-line fixture paths) are answered by first capture: the staged 0.7.2 contract, the repeat check, the deferred cascade, the produced capacity report, and the per-line folders.

Still open, and not for this change:

- The 0.7.2 audits: the gate stops at the first unpinned audit. At the committed pins that is perception; the audit lane's uncommitted working-tree pins move it to mutable_blocks (explosion_mutation is also unpinned there).
- The first-capture native steps have not run on the 0.7.2 bridge. A live run is the first real test of the deferred cascade and of the two stationary captures.
- The capacity report from the tracked producer is schema `hytalerl_world_token_capacity_benchmark_v1`. The 0.5.7 artifact is `v2`. The stationary compare reads only the fields both share (`library_semantic_sha256`, `artifact_seeds`, `natural_crops.exception_cell_counts`, `report_semantic_sha256`).
- Promotion and the swap are owner decisions.

### Live-run plan (first capture, after the audits pass)

Coordinator checks before the run:

1. `verify_installed_world_source_pins` passes for the 0.7.2 install, meaning the audit lane has accepted the 0.7.2 pins.
2. `artifacts/worldgen/region-library-pilot-v2-072/traversal-v2/`, `world-token-capacity.json` and `hytalegym/tests/fidelity/fixtures/world/v0_7/` are absent. Otherwise the mode is `compare` or the partial refusal applies.
3. The 5556 lease is free: no `native-world-recertification-127_0_0_1-5556.json` and no lock holder (`lsof`). Nothing listens on 5556 or on UDP 25574.
4. `hytale-plugin-0.7.2/build/libs/HytaleRLBridge072-0.1.0.jar` hashes to `460054A8...CAD88D`. The launcher copies it into the run directory.
5. `<run>` and `<stage>` are new, empty, absolute directories under `/Users/arush/.claude/jobs/31947bd7/tmp/recert_072/`.

Command. The launcher starts the server, runs the recertification as `--then`, and stops the server. Replace `<run>` and `<stage>`. Write no `$VARIABLES`, because `--then` is split with `shlex`. Add `--control-seed N` for a repeatable run.

    cd /Users/arush/hydl/HytaleRL/hytale-plugin-0.7.2
    env HYTALERL_SERVER_VERSION=0.7.2 APPDATA="/Users/arush/Library/Application Support" PYTHONPATH=/Users/arush/hydl/HytaleRL/hytalegym /Users/arush/hydl/.venv/bin/python tools/launch_bridge_072.py --run-dir <run> --bridge-port 5556 --game-port 25574 --then-timeout 5400 --then "/Users/arush/hydl/.venv/bin/python -m hytalegym.worldgen.region.recertification <stage> --port 5556 --deployed-bridge <run>/mods/HytaleRLBridge072-0.1.0.jar --expected-bridge-sha256 460054A8E7508FE5E205AD527AFB08474B61626EE5C2212FC2A484ECFCCAD88D --assets '/Users/arush/Library/Application Support/Hytale/install/release/package/game/latest/Assets.zip' --server-jar '/Users/arush/Library/Application Support/Hytale/install/release/package/game/latest/Server/HytaleServer.jar' --seed-projection-contract /Users/arush/hydl/HytaleRL/artifacts/worldgen/world-evidence-history/460054A8/fixtures/native_region_seed_projection_contract.json --seed-projection-npz /Users/arush/hydl/HytaleRL/artifacts/worldgen/region_0.7.2_seed_570057.npz"

The launcher's default game port is 25572, so `--game-port 25574` is required. Its `--then` exit code is recorded in `<run>/launch-report.json`.

Expected duration: up to 180 s to boot, then about 25 to 35 minutes (the planning figure above), so the `--then-timeout` is 5400 s.

While it runs: the 0.7.2 server holds 5556 and 25574 (`<run>/server.log` shows `Bridge server listening on`). The run holds the 5556 lease, which appears as `.codex-local/native-world-recertification-127_0_0_1-5556.json` until exit.

Writes: `<run>` (`server.log`, `launch-report.json`, `mods/`, `universe/`), `<stage>` (the Outputs list), and the lease signal, removed at exit. Nothing under `artifacts/worldgen/` or the fixture folder is written.

Checks before promotion:

1. `<run>/launch-report.json`: `ready` true, `then_exit_code` 0, `error` null.
2. `recertification-manifest.json`: `reference_mode` `first_capture`; 19 candidate rows; recompute each row's sha256 from the file in the stage.
3. `first_capture` block: the contract hash equals the staged 460054A8 contract; the two stationary artifact hashes are recorded; `region_library_sha256` and `projection_reference_sha256` equal the two expected digests.
4. `stationary-1/` and `stationary-2/`: the reported fields agree (only `captured_at_utc` differs) and the arrays agree within 1e-12.
5. `contract-cascade.json`: every relationship true; `identities.region_library` equals `9DCA3C71...DAAB95` (the cascade uppercases); `identities.traversal_library` equals the semantic sha256 of `traversal-v2/traversal-manifest.json`.
6. `traversal-v2/traversal-manifest.json`: `native_evidence_jar_sha256` `460054A8...`, `source_region_library_semantic_sha256` `9DCA3C71...`, `graph_count` 20. `native-traversal-recording.json`: `artifact_count` 20, `full_edge_census` true.
7. `native_region_difference.json` and the compare logs: the seed-projection check passes; the fluid-only changed cells are within the contract cap (6); the other differing cells are zero.
8. `world-token-capacity.json`: `library_semantic_sha256` `9DCA3C71...`, `artifact_seeds` of length 20.
9. A second independent run or control for the same bridge, as the 0.5.7 publications used (5A54A8F5: control seed 1148448982). The owner decides whether one run is enough.
10. The stage is outside the fixed paths, and `git status` shows no new tracked change from the run.

Then the owner decides the promotion (the three 0.7.2 locations above) and the swap (README steps 3 to 5). Promotion writes shared state, so it needs an explicit owner yes.
