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

`hytalegym/hytalegym/worldgen/region/recertification.py` (schema `hytalerl_world_evidence_recertification_v12`) stages the bridge-bound world evidence of one server line against one Region library: the seed-570057 projection, the stationary traversal, the traversal-graph library (built natively from the live bridge), the navigation and walk probes, and nine world-action probes. It stages only. A run publishes nothing (`publication.automatic` is false, `recertification.py:472`). It takes the native-evidence lease for its listener, so one run at a time drives a listener.

Line numbers are for `recertification.py` after the region-library change of 2026-10-19. Status is as of 2026-10-19.

Current state: on 0.7.2 the run stops at its first gate (audit 1, perception). Three more steps stop it once the audits pass (B1 to B3 below), and B4 and B5 may stop it later. Nothing from a 0.7.2 run is promoted.

### Inputs

| Input | Value | Read by |
|---|---|---|
| Server line | `HYTALERL_SERVER_VERSION=0.7.2`; unset means 0.5.7 | `server_profile()` (`hytalegym/server_versions/`) |
| Installed server jar | `/Users/arush/Library/Application Support/Hytale/install/release/package/game/latest/Server/HytaleServer.jar`, sha256 `23f71f03763422ce7c04df191db49cd0e7d87d43b12baad75c43fdc202410aff` (the build the 0.7.2 audits read) | `--server-jar`; `world_source_pin_staleness_report` (`source_pins.py:159`) |
| Installed Assets.zip | `/Users/arush/Library/Application Support/Hytale/install/release/package/game/latest/Assets.zip`, sha256 `ee8f8d5b332874eb928144726077fa66f80c117a5412017a5cde6fce2d9bb2a0` (3.9 GB) | `--assets`; the audits that take it (`source_pins.py:20-75`) |
| Pinned bridge | `hytale-plugin-0.7.2/build/libs/HytaleRLBridge072-0.1.0.jar`, sha256 `460054A8E7508FE5E205AD527AFB08474B61626EE5C2212FC2A484ECFCCAD88D` (pin: `hytalegym/hytalegym/server_versions/v0_7.py`) | `--expected-bridge-sha256` (required); the deployed file must hash to it (L219-234) |
| Deployed bridge | the copy the live server loaded. With `tools/launch_bridge_072.py` that is `<run dir>/mods/HytaleRLBridge072-0.1.0.jar` | `--deployed-bridge`. Without it the default is an APPDATA path under `Server/mods`, a Windows install path (`_default_deployed_bridge`, L922), so pass it on macOS |
| Live listener | bridge TCP `127.0.0.1:5556` (`--host`, `--port`), already listening | `_require_listener` (L247, def L940). The CLI has no game-port flag; the game UDP port, 25574, is set when the server starts |
| Native-evidence lease | `HytaleRL/.codex-local/native-world-recertification-127_0_0_1-5556.lock` (the `.json` signal exists only while a run holds it) | `acquire_native_evidence_lease` (L109); a run refuses when another holds it (`native evidence already active`) |
| Control seed | `--control-seed N`: a positive int32, not 570057. Omitted, a random one is drawn and recorded as `random_interposed_control_seed` | L249; `_control_seed` (L957), `_random_control_seed` (L950) |
| Region library | 0.7.2: `artifacts/worldgen/region-library-pilot-v2-072/` (manifest sha256 `c30459089c7f2c682ca411e5a8d9c46f81a8fa69eb39d2c82f9041310c2a8243`). 0.5.7: `artifacts/worldgen/region-library-pilot-v2/` (manifest sha256 `9b2fcfcecc98d1e8f8bee535fce1fc8fe86f08976ed81fdf6da6cbf526980903`) | `--region-library DIR`; default by server line (`_default_region_library`, L903) |
| Expected library digest | 0.7.2 `9dca3c71d43a2c3bf4be6aff6e2bd24d2f6608c8be01d1687a0b259f06daab95` (chunk API `cb82867f...`, split `inherited_seed_labels_v1`). 0.5.7 `f0ea0526ac1ee6184fc575c085607ee7b3f9843d78b0d4f806fff15b51c8616c` | recorded in `contract-cascade.json` (`identities.region_library`); the loader checks the manifest's own digest (the cascade loads it at L809) |
| Seed-570057 projection reference | 0.7.2 `408D5FB2DAA797E767AEE23248C6591FC226E1FE764C6258353D60FE49D1104D` (the staged NPZ and `world-evidence-history/460054A8/fixtures/native_region_seed_projection_contract.json` agree). 0.5.7 `DB4B9B85E0B40FF1D65827B2A0E1545651301D3B5CC52DF6D8A713E472F8F708` (`fixtures/world/native_region_seed_projection_contract.json`) | `require_snapshot_projection` (`stability.py:248`) |
| Frozen references | 0.5.7 `hytalegym/tests/fidelity/fixtures/world/` (`fixture_root`, L150) | `build_evidence_candidates` (L261, def L729) |
| Capacity report | `artifacts/worldgen/world-token-capacity-v4.json` (0.5.7, L270). No 0.7.2 report is staged | stationary step |
| Traversal library | none to read; the run builds `traversal-v2` from the library's NPZs | region_traversal_graphs step (`native_region_traversal_graph_compare.py`) |
| Python and Java | `/Users/arush/hydl/.venv/bin/python` (3.12); `java` on PATH (OpenJDK 25.0.2 here) | every step; `launch_bridge_072.py` |

Set `APPDATA="/Users/arush/Library/Application Support"` for every gate run on this Mac. The crafting audit's argparse default reads `APPDATA` only (`hytalegym/tests/fidelity/surrogate_crafting_source_audit.py:494-512`), so without it the recertification test fails at that audit.

### Outputs

The stage directory must be absent or empty (or reused with `--resume`), and it must lie outside `fixture_root` and the library's `traversal-v2` (`_prepare_stage`, L172, def L874). Keep it under the scratch folder.

- `source-pin-staleness.json` on a pass, or `source-pin-staleness-failure.json` on a failure.
- `contract-cascade.json`: relationships, identities and dependency order.
- `native_region_candidate_seed_570057.npz` and `native_region_difference.json`: the seed-570057 projection capture and its difference report.
- `fixtures/`: the 13 candidate JSON fixtures (the `publication.targets` in the manifest), the stationary traversal artifact `native_traversal_projection_v2_recording.npz`, and the candidate projection contract.
- `traversal-v2/`: `graphs/` (one `.traversal.npz` and one row `.json` per library entry, 40 files for 20 seeds), `traversal-manifest.json`, `native-traversal-recording.json`. Total 42 files, the same shape as the 0.5.7 folder.
- `recertification-manifest.json` (written at L544): `status` `validated_candidate_not_published`, 18 candidate rows with sha256, 14 publication targets (13 JSON plus the traversal directory), `publication.automatic` false.

Nothing is written under `artifacts/worldgen/world-evidence-history/`, under `hytalegym/tests/fidelity/fixtures/world/`, or in a library folder.

### What promote means here

Promotion is a manual copy. No tracked code does it: a search of `hytalegym/`, `arena/` and `console/` for the fixed paths, the receipt schema and copy calls found no writer to these paths (the console's copy calls move agent evidence). The private `agent_skills/` tree, at the repository root, is out of bounds for this lane and was not searched. `recertification.py` keeps it separate on purpose (`publication.reason`, L473-476). No tracked code reads the receipt schema `hytalerl_world_evidence_publication_v1` (same search); the receipts are hand-written (`world-evidence-history/5A54A8F5/publication-v12.json` is the model).

The evidence that tests and the sim read as "current" sits at two fixed paths:

1. The 13 JSON fixtures in `hytalegym/tests/fidelity/fixtures/world/`, under their names in the candidate manifest's `publication.targets`.
2. The traversal directory `artifacts/worldgen/<library>/traversal-v2/` (42 files).

The stationary NPZ arrays stay as they are. The v12 publication recorded `stationary_npz_arrays_unchanged: true`.

Manual steps, owner only, in order:

1. Review the candidate (the checks under "Before promotion").
2. Preserve the superseded set in `artifacts/worldgen/world-evidence-history/<bridge8>/` before the fixed paths change (the history README). For 0.5.7 that set already exists in `world-evidence-history/5A54A8F5/` (`fixtures/` holds the 13 JSON plus the stationary NPZ, `traversal-v2/` the 42 files, and the v12 manifest and publication receipt).
3. Copy the 13 JSON into `hytalegym/tests/fidelity/fixtures/world/` and replace the library's `traversal-v2/` with the candidate's 42 files.
4. Write `publication-vN.json` in the new bridge's history folder, and update the local STATUS files (`world-evidence-history/STATUS.md`, `fixtures/world/STATUS.md`, the library STATUS).
5. Promotion does not change pointers or digests. Those are the swap (README step 4), an owner decision.

Rollback: copy the preserved files from `world-evidence-history/<bridge8>/` back over the fixed paths. `hytalegym/tests/**` is git-ignored, so git cannot restore those files.

Who reads the fixed paths:

- Tests (names found by grep): `tests/worldgen/test_region_recertification.py` (`_FIXTURES`, L37), `tests/worldgen/test_native_world_recordings.py`, `tests/worldgen/test_region_state_change_use.py`, `tests/jax/world/navigation/test_navigation_native_recording.py`, `tests/jax/world/region/test_region_library.py`, `arena/tests/test_region_library_pin.py` and `console/tests/test_scene_cache.py` (both name the 0.5.7 library folder).
- Code: `hytalegym/worldgen/native_traversal.py:564` (names the stationary recording), `hytalegym/jax/world/navigation/__init__.py:485` (names the navigation recording), `hytalegym/sim/common/content/adapters.py:714` and `hytalegym/sim/common/content/redirects.json:1366` (the crafting catalog recording).
- Pointers, changed only at the swap: `hytalegym/jax/world/tokens/__init__.py:87-96` (the 0.5.7 library and traversal manifests by fixed path), `hytalegym/sim/common/bench.py:62-63`, `hytalegym/jax/evaluation/_contracts.py:72-79` (the pins `F0EA0526...` and the capture bridge `1644DC99...`, read by the frozen Arsenal suites), and `hytalegym/worldgen/region/stability.py:33-35` (the 0.5 bridge pin `7700D112...`).
- Not a reader: `hytalegym/jax/world/region/provider.py` selects a library by its semantic digest under `HYTALERL_WORLD_ARTIFACT_ROOT` (L30, L233). It does not read these paths.

Layout hazard (B6): the 13 fixture paths have no per-line folder. Copying 0.7.2 JSON there makes every 0.5.7 reader above read 0.7.2 evidence at once. The owner must choose a layout (per-line folders, or a cut-over at the swap) before any copy.

### Gates, in order

Line numbers are in `recertification.py` unless a file is named.

1. Lease, L109: `acquire_native_evidence_lease` for host and port, purpose `world_recertification`. Refuses when another run holds the listener.
2. Stage, L172 (`_prepare_stage`, L874): outside the published roots; absent or empty unless `--resume`.
3. Paths, L158-170: the deployed bridge, the assets and the server jar defaults.
4. Source pins, L185 (`verify_installed_world_source_pins`, `source_pins.py:80`): 11 audits in `_AUDITS` (`source_pins.py:20`), perception first. The first audit whose semantic hash does not match raises (`source_pins.py:133-138`). The except block at L190 writes the failure file and re-raises.
5. Installed build (`source_pins.py:159`): the server's release line must equal the profile's certified version, and each audit's server sha must equal the installed jar.
6. Bridge identity, L219-234: the line's pin (`current_native_evidence_jar_sha256`) must equal `--expected-bridge-sha256`, and the deployed file's sha must equal both.
7. Contract cascade, L245 (`derive_contract_cascade`, def L777): loads the region library (L809) and the traversal library (L810), checks the relationships, writes `contract-cascade.json`.
8. Listener, L247 (`_require_listener`): a TCP connect to host and port.
9. Control seed, L249.
10. Candidates, L261 (`build_evidence_candidates`, def L729): relabels the frozen projection contract (L746-747) and recording (L765-766) to the bridge and the version, and checks `require_evidence_bridge`.
11. Live compares, each `check=True` so a failure stops the run, in this order (L276-357): `region_seed_projection` (L278, `tests/fidelity/native_region_capture_compare.py`, checks the projection with `require_snapshot_projection`); `stationary_traversal` (L303); `region_traversal_graphs` (L320, `tests/fidelity/native_region_traversal_graph_compare.py`, which captures one graph per library entry and writes the traversal manifest through `create_region_traversal_graph_library`, `traversal_library.py:183`); `native_navigation_paths` and `native_walk_probe_subset` (L336, `_navigation_probe_commands` def L555); nine world-action probes (L346, `_world_action_probe_commands` def L612).
12. Loaders, L374-375: `RegionArtifactLibrary.load` and `RegionTraversalGraphLibrary.load`. The graph's `native_evidence_jar_sha256` must equal the expected bridge (L379-380).
13. Report, L544: `recertification-manifest.json`.

### Duration from the 0.5.7 history

No start time is recorded. These are local (CDT) file times in the history folders, from the source-pin report (written when the gate exits) to the candidate manifest:

- `D954BD79` (2026-08-06): 19:26:33 to 19:39:17, 12 min 44 s.
- `1C063331` (2026-08-03): 14:09:34 to 14:34:05, 24 min 31 s.
- `5A54A8F5` (2026-08-01, v12): 00:53:33 to 01:16:56, 23 min 23 s. Its publication receipt followed at 01:21:23 (`publication-v12.json`).
- The graph step inside the 5A54A8F5 run (`traversal-v2/native-traversal-recording.json`): 20 artifacts, compile 423.6 s plus native capture 68.6 s, about 8 minutes.

Planning figure: 15 to 25 minutes from the gate to the manifest, plus up to 180 s for the server to boot (the launcher's `--ready-timeout`). The gate's own duration is not recorded.

### Keyed and hand-maintained constants on this path

Keyed by the region-library change (one place): the library folder (`_default_region_library`, L903; `ServerProfile.region_library`, set for 0.7.2 in `v0_7.py`; the 0.5 line keeps `region-library-pilot-v2`). `derive_contract_cascade` takes `region_root`. The arsenal evidence bridge is already keyed to the line.

Still hand-maintained:

- `_TARGET_SEED = 570057` (L40). A 0.7.2 NPZ for it is staged (`artifacts/worldgen/region_0.7.2_seed_570057.npz`), but it is not a library member.
- `"570058"` in `_navigation_probe_commands` (L602). It is a library selection key, not a seed.
- The 0.5.7 fixture folder (`fixture_root`, L150) and the 13 fixture names (B2, B4, B6).
- The capacity report path (L270), which is 0.5.7 (B5).
- `_default_deployed_bridge` (L922), an APPDATA Windows path.
- No expected library digest is checked on this path; the cascade records one.
- Swap items outside the path: the pins listed under "Who reads the fixed paths".

### 0.7.2 dry run, 2026-10-19

Script: `/Users/arush/.claude/jobs/31947bd7/tmp/recert_072/scripts/dryrun_072.sh`; log `.../recert_072/dryrun.log`. Command: `HYTALERL_SERVER_VERSION=0.7.2`, `python -m hytalegym.worldgen.region.recertification <stage> --port 5556 --deployed-bridge hytale-plugin-0.7.2/build/libs/HytaleRLBridge072-0.1.0.jar --expected-bridge-sha256 460054A8...CAD88D --assets <Assets.zip> --server-jar <HytaleServer.jar> --control-seed 481374900`. No server was launched and no port was used; the lease was taken and released.

Result: the run exits 1 at L185 with

    RuntimeError: World source-pin audit failed (perception): installed LOS inputs differ from the 0.5.7 baseline

The stage holds only `source-pin-staleness-failure.json`. No library was loaded, because the library and the cascade come after the gate.

### Blockers after the audits pass

Checked on 2026-10-19 unless marked otherwise.

- B1 (confirmed by running the code): under `HYTALERL_SERVER_VERSION=0.7.2` the default folder is `region-library-pilot-v2-072`, and the cascade (L245, traversal load L810) raises `FileNotFoundError` for `region-library-pilot-v2-072/traversal-v2/traversal-manifest.json`. The traversal library is built later, in the graph step, so a first 0.7.2 run cannot pass the cascade.
- B2 (confirmed by running `require_snapshot_projection` on the staged NPZ): the candidate projection contract keeps the 0.5.7 reference `DB4B9B85...`, because only the bridge and the version are relabelled (L746-747). The staged 0.7.2 seed-570057 NPZ projects to `408D5FB2...`, and `require_snapshot_projection` raises `Region seed-projection digest changed` (`stability.py:248`). A live 0.7.2 capture gives the same digest (FIDELITY (aj)). The candidate's fluid-only cap is also the 0.5.7 value, 4; the 0.7.2 contract uses 6.
- B3 (inferred, not run): the graph step passes `--expect-manifest` (L332) at the absent 0.7.2 `traversal-manifest.json`. The compare loads it only after capturing all 20 graphs (`native_region_traversal_graph_compare.py:164-170`), so the first 0.7.2 build would fail after about 8 minutes of captures. The loader is the one that raised in B1.
- B4 (inferred, not run): the stationary step (`--expect-recording`, L315) requires the live report and arrays to equal the relabelled 0.5.7 recording (`surrogate_traversal_native_compare.py:148-149`, `_require_recording_matches` in that script, L421-470: stable fields, artifact hash, each array within 1e-12). The 0.7.2 terrain differs (FIDELITY (aj): 5,927 cells on seed 570057), so this should fail.
- B5 (not checked): the stationary step takes the 0.5.7 capacity report (L270). No 0.7.2 report is staged.
- B6 (owner decision): the single-line fixture paths (see "What promote means here").
- Design (owner decision): the candidate builder reuses 0.5.7 references by design (`test_candidate_repin_retains_reference_arrays_and_changes_only_evidence`). A new line needs references from its own first capture. The options are a first-capture mode (references and graph expectations from the run itself, as `recapture_region.py` builds its v2 contract, with the 0.5.7 path unchanged) or 0.7.2 reference recordings staged by hand first. B1 and B3 need an order change or a flag either way.

### Live-run plan, after the audits pass

Coordinator checks before the run:

1. `verify_installed_world_source_pins` passes for the 0.7.2 install, meaning the audit lane has accepted the 0.7.2 pins.
2. B1, B2 and B3 are resolved by an owner decision. Otherwise the run stops at the cascade, the projection, or after about 8 minutes at the graph step. B4 and B5 may stop it at the stationary step.
3. The 5556 lease is free: no `native-world-recertification-127_0_0_1-5556.json` and no lock holder (`lsof`). Nothing listens on 5556 or on UDP 25574.
4. `hytale-plugin-0.7.2/build/libs/HytaleRLBridge072-0.1.0.jar` hashes to `460054A8...CAD88D`. The launcher copies it into the run directory.
5. `<run>` and `<stage>` are new, empty, absolute directories under `/Users/arush/.claude/jobs/31947bd7/tmp/recert_072/`.

Command. The launcher starts the server, runs the recertification as `--then`, and stops the server. Replace `<run>` and `<stage>` with absolute paths. Write no `$VARIABLES`, because `--then` is split with `shlex`, not a shell. Add `--control-seed N` for a repeatable run.

    cd /Users/arush/hydl/HytaleRL/hytale-plugin-0.7.2
    env HYTALERL_SERVER_VERSION=0.7.2 APPDATA="/Users/arush/Library/Application Support" PYTHONPATH=/Users/arush/hydl/HytaleRL/hytalegym /Users/arush/hydl/.venv/bin/python tools/launch_bridge_072.py --run-dir <run> --bridge-port 5556 --game-port 25574 --then-timeout 3600 --then "/Users/arush/hydl/.venv/bin/python -m hytalegym.worldgen.region.recertification <stage> --port 5556 --deployed-bridge <run>/mods/HytaleRLBridge072-0.1.0.jar --expected-bridge-sha256 460054A8E7508FE5E205AD527AFB08474B61626EE5C2212FC2A484ECFCCAD88D --assets '/Users/arush/Library/Application Support/Hytale/install/release/package/game/latest/Assets.zip' --server-jar '/Users/arush/Library/Application Support/Hytale/install/release/package/game/latest/Server/HytaleServer.jar'"

The launcher's default game port is 25572, so `--game-port 25574` is required. Its `--then` exit code is recorded in `<run>/launch-report.json`.

Expected duration: up to 180 s to boot, then about 15 to 25 minutes (the planning figure above), so the 3600 s `--then-timeout` leaves room.

While it runs: the 0.7.2 server holds 5556 and 25574 (`<run>/server.log` shows `Bridge server listening on`). The run holds the 5556 lease, which appears as `.codex-local/native-world-recertification-127_0_0_1-5556.json` until exit. Its child processes join the lease through the environment token.

Writes: `<run>` (`server.log`, `launch-report.json`, `mods/`, `universe/`), `<stage>` (the Outputs list), and the lease signal `.json` under `HytaleRL/.codex-local/`, removed at exit (the `.lock` file stays). Nothing under `artifacts/worldgen/` or `hytalegym/tests/fidelity/fixtures/world/` is written.

### Before promotion

1. `<run>/launch-report.json`: `ready` true, `then_exit_code` 0, `error` null.
2. `recertification-manifest.json`: `status` `validated_candidate_not_published`, 18 candidate rows. Recompute each row's sha256 from the file in the stage.
3. `source-pin-staleness.json`: 11 audits, and the report's semantic sha256 equal to the accepted 0.7.2 pins.
4. `contract-cascade.json`: every relationship true. `identities.region_library` equals `9DCA3C71...DAAB95` (the cascade uppercases). `identities.traversal_library` equals the semantic sha256 of the new traversal manifest.
5. `traversal-v2/traversal-manifest.json`: `native_evidence_jar_sha256` `460054A8...`, `source_region_library_semantic_sha256` `9DCA3C71...`, `graph_count` 20. `native-traversal-recording.json`: `artifact_count` 20, `full_edge_census` true.
6. `native_region_difference.json` and the compare logs: the seed-projection check passes, the fluid-only changed cells are within the contract cap, and the other differing cells are zero. The 0.5.7 publications recorded exactly this (5A54A8F5: three fluid-only cells, zero other).
7. A second independent run or control for the same bridge, as the 0.5.7 publications used (5A54A8F5: control seed 1148448982, a separate seed from the target). The owner decides whether one run is enough.
8. The stage is outside the fixed paths, and `git status` shows no new tracked change from the run.

Then the owner decides the layout (B6), the swap (README steps 3 to 5), and the promotion itself. Promotion writes shared state, so it needs an explicit owner yes.
