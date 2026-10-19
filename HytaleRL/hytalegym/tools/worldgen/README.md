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
