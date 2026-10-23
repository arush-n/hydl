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

4. Physical only, on purpose. `--physical-only` skips the block-semantic sidecar. The receipt records the opt-out (`block_semantics.opt_out`), and `publish-library` refuses that seed.

        $PY $TOOL --version 0.7.2 --seed <seed> --bridge-port 5556 --game-port 25574 --run-root <run root> --fluid-only-cap 6 --physical-only

Items 1 and 2 write the block-semantic sidecar in the same run as the NPZ. Item 3 writes none, and neither does a `--skip-capture` run. Use the backfill in the block-semantic section for those.

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
- `artifacts/worldgen/region_<version>_seed_<seed>.block-semantics.npz` and `.receipt.json` beside it, written in the capture run. Not written with `--physical-only`, and not by a `--skip-capture` run. A run refuses to overwrite it.
- Under `<run root>`: `capture.launcher.log` and `compare.launcher.log`, the launch directories, `compare-difference-report.json` (on a mismatch), `native_region_seed_projection_contract.json` (from `--fluid-only-cap`), `offline-diff.json` (with `--reference`) and `receipt.json`.
- The compare takes the native evidence lease for the bridge port (`hytalegym/hytalegym/worldgen/native_lease.py`). Its files live under `HytaleRL/.codex-local/`.

`receipt.json` records the capture (`skipped`, `passed`, `semantic_sha256`, `file_sha256_on_disk`) and the compare: `scope` (`exact_semantic_sha256` or `seed_projection_contract`), the contract's path and file sha256 when one is used, the launcher exit and the pass marker. The top-level `passed` is capture and compare together. In a `--skip-capture` run, `capture.passed` is `null`, because no capture ran. The receipt also has `identity_complete` (true only when a sidecar's receipt says its reload matched, for this seed, and the file is unchanged) and `block_semantics` (`opt_out`, `passed`, `sidecar`).

## Swap steps (owner's; the tool does none of these)

1. Read the receipt, the compare log and the offline diff (sections and cells differing, by field).
2. Publish the line's library through `hytalegym/hytalegym/worldgen/region/recertification.py`, which stages and then promotes. That path runs `verify_installed_world_source_pins` before it touches the listener. `FIDELITY-0.7.2.md` (ad) records it refusing 0.7.2 at audit 1 until the audits are closed. `create_region_artifact_library` writes `sha256_rank_v1` by default, which moves six held-out seeds on 0.7.2. With `inherit_split_from` set to the 0.5.7 manifest and the 20 0.7.2 NPZs as artifacts, it writes `inherited_seed_labels_v1` with the 0.5.7 labels (see "Loading a 0.7.2 library"). Its repository callers all use the rank default (`examples/capture_native_region_library.py`, the random-seed pilot, and `tests/jax/world/region/test_region_library.py`). `recertification.py` does not call it: that file only reads the 0.5.7 pilot (`_recapture_world_evidence_unlocked`, `derive_contract_cascade`) and writes candidate evidence, so the 0.7.2 publish does not pass the option yet.
3. Rebuild the products that depend on the terrain: path tables, task suites and catch-time horizons (`hytale-plugin-0.7.2/SIMULATOR-0.7.2.md`, P2). The repository generators are `tools/worldgen/region_path_distance.py`, `tools/worldgen/task_zoo.py` and `tools/worldgen/catch_time_horizons.py` (each takes `--server-version`, default 0.7.2; see "Products derived from a library" below, committed 3cd8c06 on 2026-10-18). The biome table has no consumer (below). The Arsenal Region suites are derived at call time and pinned by SHA in `jax/evaluation/_contracts.py`, so they need a new frozen version once a 0.7.2 traversal library exists, not a rebuild.
4. Flip the defaults that point at `region-library-pilot-v2`: `hytalegym/hytalegym/jax/world/tokens/__init__.py` (lines 87 to 95), `hytalegym/hytalegym/worldgen/region/recertification.py` (lines 141 and 791 to 797) and `hytalegym/hytalegym/sim/common/bench.py` (line 62).
5. Keep the 0.5.7 libraries and the staged NPZs. The provider finds a library by its semantic hash under `HYTALERL_WORLD_ARTIFACT_ROOT` (`hytalegym/hytalegym/jax/world/region/provider.py`), so old and new libraries can sit side by side. Ambiguous configured artifacts raise. The 0.7.2 library's split labels are inherited from the 0.5.7 library in the same folder, so that folder must stay beside it; without it the 0.7.2 manifest is refused.

Open question before step 2 (owner's call): each library manifest records every NPZ's complete `semantic_sha256`. A library built from the staged 0.7.2 NPZ would therefore carry the NPZ's complete digest, which a fresh capture on this bridge does not reproduce. The projection digest is the stable value for this seed. Which digest identifies the 0.7.2 library is not decided here.

Pointers keyed by server line (2026-10-19). The Region library folder is chosen at call time from `server_profile().region_library`: unset or 0.5.7 read `artifacts/worldgen/region-library-pilot-v2/`, and 0.7.2 reads `artifacts/worldgen/region-library-pilot-v2-072/`, each holding `manifest.json` and `traversal-v2/traversal-manifest.json`. The readers are `_region_library_manifest()` and `_traversal_library_manifest()` in `hytalegym/hytalegym/jax/world/tokens/__init__.py` (the staging identity behind `world_geometry_token_contract`) and `_region_manifest()` in `hytalegym/hytalegym/sim/common/bench.py` (the `region` workload). `recertification.py` already resolves its folder the same way (`_default_region_library`), so the tokens and bench entries of step 4 are done, and its line numbers are stale. `HYTALERL_TRAVERSAL_MANIFEST`, when set to an absolute path, replaces the traversal manifest the identity reads; it is read at call time, must exist (otherwise the identity raises a `RuntimeError` naming it), and a relative value raises `ValueError`. The recertification exports it to name its stage's `traversal-v2` manifest during a first capture; bench does not read it. Without the override, a line whose Region manifest exists but whose traversal manifest does not (the first-capture state, before promotion) reports `STAGING_TRAVERSAL_ABSENT`, a non-hex sentinel, with the region digest and a count of 0, instead of raising; the 0.5 line never reaches that state. The identity is cached per process by the two manifest paths, so a process that reported the sentinel keeps it until it restarts or names the override. A bench `region` run on 0.7.2 still stops with a missing-file error on its traversal manifest.

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

- Generator in the repo: `region_vN_suite_contract(traversal_manifest_path=...)`, v1 to v6, in `hytalegym/hytalegym/jax/evaluation/_contracts.py`, and `region_v7_suite_contract()` (per line; see the v7 bullet). The checked v6 wrapper raises on drift.
- One-command check. It prints the success line and exits 0 when v6 reproduces its frozen SHA:

        PYTHONPATH=/Users/arush/hydl/HytaleRL/hytalegym /Users/arush/hydl/.venv/bin/python -c "from hytalegym.jax.evaluation import region_v6_suite_contract as f; f(traversal_manifest_path='/Users/arush/hydl/HytaleRL/artifacts/worldgen/region-library-pilot-v2/traversal-v2/traversal-manifest.json'); print('v6 reproduces its frozen SHA')"

- Status on 2026-10-18: this raises `drifted` for v6. An unchecked recomputation also fails for v1 to v5 on both pilot-v2 traversal manifests. The cause is not isolated. Fix this before any suite rebuild.
- Diagnosis (2026-10-19, read-only lane; scratch `suite_drift/`): two causes. Since the release, the suite input changed in
  `879321e` (2026-10-18): `worldgen/region/stability.py` moved `CURRENT_NATIVE_EVIDENCE_JAR_SHA256` from `21C4B1FA` to
  `7700D112` and its accessor now prefers `server_profile().bridge_jar_sha256`; the arsenal contract embeds that pin
  (`jax/combat/arsenal/schema/spec.py:1230`), so v6 recomputes to `46DF509E` in the working tree. Independently, the release
  code with the on-disk artifacts gives `58190258`, not the frozen `4A52B017`, and no on-disk traversal manifest, documented
  digest or known bridge pin reproduces the frozen value (1,587 combinations tried), so the freeze inputs are not recoverable.
  The library, the four frozen worlds and their edge selections all match. Under `HYTALERL_SERVER_VERSION=0.7.2` the pin
  becomes `460054A8` and v6 gives `D9DD0390`. Consequence: a suite is per server line because its content carries the line's
  bridge pin. The 0.7.2 suite (v7) is frozen (2026-10-19, see the v7 bullet) from the promoted 0.7.2 traversal manifest under the 0.7.2 profile and keyed by
  `server_profile().certified_version`; the 0.5.7 v6 stays historical (re-freezing it or binding a 0.5.7 v7 to the current
  pin is an owner decision; the lane's candidate patch is `suite_drift/proposed.patch`). Pins: `tests/jax/test_region_public_benchmark.py:131-142`,
  `evaluation/_artifact.py` (`FROZEN_REGION_SUITE_SHA256_BY_ID`), `docs/jax/arsenal-benchmark.md:221-222`.
- Resolved 2026-10-19 (owner's call: per-line v7). The three 0.5.7 identities are now inputs of the line table `ARSENAL_REGION_V7_LINES`; v1 to v6 keep them as defaults, and their suite dicts are unchanged (JSON compared before and after the change).
- Region v7 (0.7.2 line), frozen 2026-10-19, re-frozen the same night after the bridge batch dbd30f2 (pin 6F5B6021) and again after the host-contract batch b682e33 (pin 6EB44F30, run 8) and after the fixture-aim, augment-readout and request-tick batches (pin 1CB4F651, run 10 on 2026-10-20) and after the gait batch 3024fde (pin 6DFD8682, run 11 on 2026-10-21; the bridge pin enters the content twice, through the arsenal contract and through the promoted traversal manifest's `native_evidence_jar_sha256`, so every 0.7.2 bridge rebuild and every promotion re-freezes this value): `hytalerl_arsenal_region_v7` for `certified_version == "0.7.2"` is `CFB529B84C4A465619D2917FEB23D72F19466A53A1596E1BC2D4D647AE2D341F`.
  - Inputs: Region library `9DCA3C71D43A2C3BF4BE6AFF6E2BD24D2F6608C8BE01D1687A0B259F06DAAB95` (profile `region_library_sha256`; library manifest); traversal library `CC613B1202C44A2C351EAEA6027E057A4E844676B7DC361BCDD070CAE2A38C4C` (`region-library-pilot-v2-072/traversal-v2`, 20 graphs); capture bridge `460054A8E7508FE5E205AD527AFB08474B61626EE5C2212FC2A484ECFCCAD88D` (the library's capture contract, `capture_evidence_bridge_sha256` in the line table); the promoted traversal manifest's native evidence bridge (`native_evidence_jar_sha256`, today `6EB44F30`); the arsenal contract, which carries the current bridge pin (today `6EB44F30`).
  - Held-out worlds (`FROZEN_REGION_WORLDS_0_7_2`), by evaluation order: 0 slot 3 seed 1921897808; 1 slot 2 seed 1397671741; 2 slot 1 seed 1176664667; 3 slot 0 seed 1274646439. Each record carries the library's stored semantic digest, the traversal graph digest and the spawn edge, chosen by the same selection code as v6 (`_select_world_edge`, now taking the world table). The 0.5.7 order (slots 1, 3, 0, 2) does not apply to this library.
  - Fresh-capture drift is irrelevant to loading the frozen library. The identity is the library's stored NPZs and graphs; the held-out digests are one capture's complete digests (`artifacts/worldgen/region-library-pilot-v2-072/STATUS.md`), and a fresh capture need not reproduce them. Loading checks the stored artifacts, not a live capture.
  - Per-line rule: `region_v7_suite_contract()` and `load_region_v7_context()` read `server_profile().certified_version`. A line without an `ARSENAL_REGION_V7_LINES` entry raises "no frozen v7 suite on this line"; a changed input raises "drifted". The bridge pin is read at import (`hytalegym/jax/combat/mechanics/schema/contract.py:117`), so a process has one line: start 0.7.2 with `HYTALERL_SERVER_VERSION=0.7.2`. The 0.7.2 test cases skip on any other line.
  - One-command check (a 0.7.2 process): `HYTALERL_SERVER_VERSION=0.7.2 PYTHONPATH=/Users/arush/hydl/HytaleRL/hytalegym /Users/arush/hydl/.venv/bin/python -c "from hytalegym.jax.evaluation import region_v7_suite_contract as f; f(); print('v7 reproduces its frozen SHA')"`
  - How to freeze a later line (use a new id, `hytalerl_arsenal_region_v8`, because the registry keeps one value per id): (1) promote that line's Region and traversal libraries and add its profile entry in `hytalegym/server_versions`; (2) build its held-out world records through the selection code (`region_environments_from_artifact_manifest`, then `_select_world_edge` with the evaluation order set to the environment index and the graph digest taken from the traversal entry); (3) add the v8 content function and a line entry whose frozen value is computed in a process started with that line's `HYTALERL_SERVER_VERSION`; (4) register the id in `_artifact.py` and the package exports; (5) add tests that skip off the line, and a tampered-copy drift case.

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

### 0.7.2 products (2026-10-19)

Folder `artifacts/worldgen/region-library-pilot-v2-072/products/` (gitignored): `task_zoo_v1.json`, `path_distance/seed_<seed>.npz` (20 tables), `MANIFEST.json` and `STATUS.md`. The library is the staged 0.7.2 folder (semantic sha 9DCA3C71..., split method `inherited_seed_labels_v1`); `RegionArtifactLibrary.load(..., verify_artifacts=True)` accepts it (20 entries, 16 train, 4 heldout). The traversal library is the 0.7.2 first capture (run 4), promoted to `traversal-v2/` (library sha CC613B12..., source Region sha 9DCA3C71..., bridge pin 460054A8; 20 graphs, each checked against its Region entry). At 512 ticks the 0.7.2 zoo gives 16 train terrains per family and 48 tasks (a 512-tick horizon is 17.1 s at 1/30 s; the tool rounds the tag, so the 17.07 s above is unrounded). The 0.5.7 control gives 16 per family and 48 tasks (23.0 s at 0.045 s). These supersede the 2026-10-18 counts of 13 per family and 39 tasks. The old `sha256_rank_v1` rule still gives 13 and 39 read-only, with six disagreeing seeds; the tool's `split_disagreements` is empty now because the inherited labels are its rule, so that holds by construction. Path-distance tables: 20 of 20 built with `region_path_distance.py --server-version 0.7.2`, one sequential background loop (2026-10-19, 19:46 to 19:51). Each table's `region_semantic_sha256` equals its library entry's, its traversal graph digest equals the promoted record's, and each landmark column is 0 at its own node (74555 columns in 20 tables). Per seed (wall s / landmarks / MiB / finite %): 70970103 14 s / 3704 / 26.8 MiB / 11.2%; 260666958 11 s / 3761 / 20.2 MiB / 7.8%; 353404383 15 s / 3774 / 29.3 MiB / 12.0%; 385501745 15 s / 3721 / 25.5 MiB / 11.1%; 397077566 17 s / 3639 / 27.6 MiB / 12.3%; 470574333 11 s / 3651 / 20.3 MiB / 8.4%; 624460635 24 s / 4167 / 44.1 MiB / 15.0%; 712055254 11 s / 3687 / 20.3 MiB / 8.4%; 974971448 14 s / 4177 / 24.8 MiB / 7.5%; 1044320000 14 s / 3818 / 26.4 MiB / 10.7%; 1058723659 12 s / 3408 / 21.5 MiB / 11.7%; 1176664667 12 s / 3552 / 21.3 MiB / 11.0%; 1274646439 11 s / 3797 / 22.1 MiB / 8.5%; 1397671741 11 s / 3936 / 20.8 MiB / 7.4%; 1618574389 13 s / 3584 / 25.4 MiB / 12.3%; 1691108502 14 s / 3596 / 25.2 MiB / 12.2%; 1710062098 11 s / 3481 / 20.7 MiB / 10.5%; 1819617980 16 s / 3504 / 28.5 MiB / 14.3%; 1921663088 12 s / 3616 / 21.3 MiB / 9.6%; 1921897808 22 s / 3982 / 42.9 MiB / 16.0%. The wall times were measured on this run and are not comparable with the 2026-10-18 search time (see STATUS.md). The reader (`arena/training/skills/path_distance.py`, untracked) loads seed 1618574389's table with its own sha, refuses it for seed 1044320000's sha, and gives self-distance 0 for landmarks 0 to 2 (read-only, 2026-10-19). The finite share is 12.33% for that table, against 12.33% for the 2026-10-18 0.5.7 table of the same seed number (a different Region); the 127-block cap is the likely cause (inferred). For seed 1618574389 the 0.7.2 and 0.5.7 tables have identical node and landmark positions and differ in 3 of 142772224 entries; their traversal graph digests differ, so the two graphs differ slightly (cause not inspected). Decisions carried over unchanged: paths beyond 127 blocks stay 255 (a coordinator decision, 2026-10-18); the 4-block landmark cells and the 3.0-block layer gap are inferred (tool docstring); all open to change. Catch-time horizons are not built: the repository holds no measured episode times to derive them from.

## Library recapture (pilot v2, 0.7.2)

The twenty seeds of `region-library-pilot-v2` were recaptured on 0.7.2 with `recapture_region.py --fluid-only-cap 6` and each seed's manifest core passed as `--core-min-chunk-x/-z`. Three bridge and game port pairs ran in parallel, one seed at a time per pair. Each seed ran one capture and one contract-scope compare. The compare script always uses the server's default core, so a manifest core matches only when the default equals it. The seed projection digest includes the core, so each compare pass confirms the match for its seed. One compare launch failed on `EADDRINUSE` at the launcher's bind probe, most likely TIME_WAIT left on the bridge port by the previous server (inferred from the timing, not logged), so the tool now waits up to 180 s for both ports before each launch. The failed seed's compare was re-run once on the same NPZ with `--skip-capture`. The cap stayed at 6 for every seed. The staged folder is `artifacts/worldgen/region-library-pilot-v2-072/` (NPZs, a 0.5.7-schema manifest with the 0.5.7 seed splits, one contract per seed). Its `STATUS.md` holds the per-seed table, the commands and the validation. The staged manifest loads through `RegionArtifactLibrary.load` since the loader change in the next section. The source-pin gate blocks recertification on 0.7.2, so nothing is promoted. The driver and staging scripts are not in the repository.

### Loading a 0.7.2 library

`RegionArtifactLibrary.load` (`hytalegym/hytalegym/worldgen/region/library.py`) certifies the chunk API for the manifest's own `capture_contract.server_version`. `certified_chunk_api_manifest(server_version)` (`hytalegym/hytalegym/worldgen/region/contract.py`) returns the certified manifest for 0.5.7 or 0.7.2 and raises for any other version, 0.6.0 included. `CERTIFIED_CHUNK_API_SHA256` pins the two fingerprints. The no-argument call still means 0.5.7, so callers that pass no version do not move. The second change is a second split method, `inherited_seed_labels_v1`. Its `split_contract` holds `method`, `salt_sha256` and `source_library_semantic_sha256`; the last two are copied from the source library. The loader finds that source as the provider finds libraries: exactly one sibling folder of the manifest's folder must hold a `manifest.json` whose `library_semantic_sha256` equals the recorded digest. The loader loads the source without its artifacts, so the source's own digest is checked. The source must use `sha256_rank_v1`, and it must have the same salt, the same seeds and the same held-out labels. `sha256_rank_v1` is unchanged. The staged 0.7.2 manifest uses the inherited method because `sha256_rank_v1` over the 0.7.2 hashes moves six held-out seeds (`artifacts/worldgen/region-library-pilot-v2-072/STATUS.md`). The 0.5.7 folder therefore has to stay beside the 0.7.2 folder in `artifacts/worldgen/`. `create_region_artifact_library(root, paths, inherit_split_from=<0.5.7 manifest>)` writes this method. It copies the salt, the held-out count and the source digest from the source, and it refuses unless the artifact seeds equal the source's seeds and the source uses `sha256_rank_v1`. It also refuses before writing unless the loader would find that source as the sibling by digest. Its default, `sha256_rank_v1`, is unchanged. The provider (`hytalegym/hytalegym/jax/world/region/provider.py`, unchanged) finds a library by `library_semantic_sha256` under `HYTALERL_WORLD_ARTIFACT_ROOT`; it does not read the server version, and the manifest's own version selects the certified chunk API. Its default `require_traversal=True` still falls back to the surrogate for 0.7.2, because no 0.7.2 traversal library is staged. `tools/worldgen/task_zoo.py` (`package_heldout_seeds`) follows the manifest's method: under `inherited_seed_labels_v1` the rule is the certified labels, so the 0.7.2 zoo reports no split disagreements (16 train terrains and 4 held out per suite, 2026-10-19); under `sha256_rank_v1` it still re-derives the rank, and any other method fails closed.

## Recertifying a server line

`hytalegym/hytalegym/worldgen/region/recertification.py` (schema `hytalerl_world_evidence_recertification_v12`) stages the bridge-bound world evidence of one server line against one Region library: the seed-570057 projection, the stationary traversal, the traversal-graph library (built natively from the live bridge), the navigation and walk probes, and nine world-action probes. It stages only. A run publishes nothing (`publication.automatic` is false, `recertification.py` L544). It holds the native-evidence lease for its listener, so one run at a time drives a listener.

Line numbers are for `recertification.py` after the first-capture change of 2026-10-19. Status is as of 2026-10-19.

Current state:

- 0.5.7: compare mode, unchanged. All four references exist, so the run is the same as before. The cascade output is byte-identical before and after the change. The real 0.5 dry run stops at the bridge-identity check, because no 0.5 bridge build exists on this machine (see "Dry runs").
- 0.7.2: compare mode since 2026-10-19 19:45; re-promoted at 21:43 for the rebuilt bridge `6F5B6021`, at 23:38 for `6EB44F30`
  (the host-contract batch b682e33) and at 02:02 on 2026-10-20 for `1CB4F651` (the fixture-aim, augment-readout and request-tick
  batches; receipt `world-evidence-history/1CB4F651/publication-v12.json`) and at 19:00 on 2026-10-21 for `6DFD8682` (the gait
  batch 3024fde; receipt `world-evidence-history/6DFD8682/publication-v12.json`) and at 19:25 on 2026-10-22 for `1F874DFC` (the
  token-capacity and jump batch 04c3b18; receipt `world-evidence-history/1F874DFC/publication-v12.json`; each earlier bridge's evidence is under
  `world-evidence-history/<that bridge8>/superseded/`). The first
  capture (live run 4, 19:26 to 19:42) passed end to end and was promoted
  with `promote_world_evidence.py` (receipt `world-evidence-history/460054A8/publication-v12.json`, 62 files: 14 fixtures under
  `fixtures/world/v0_7/`, the 42-file `traversal-v2/` and `world-token-capacity.json` in `region-library-pilot-v2-072/`, the
  manifest, source-pin report, cascade and difference report copied beside the receipt). Identities: Region `9DCA3C71`,
  traversal `CC613B12` (identical across runs 1 to 4), world token `9A216202`; seed-570057 projection `408D5FB2` with 3
  fluid-only cells of a cap of 6; stationary captures byte-identical; block-use `typed_synthetic_status` `closed_by_bridge`
  (task #91). A dry run after promotion passes the gate, writes the cascade and stops at the listener, as the 0.5 line does.

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
8. The traversal graph library is built natively (`region_traversal_graphs`), with no `--expect-manifest`. From here on the
   run exports `HYTALERL_TRAVERSAL_MANIFEST=<stage>/traversal-v2/traversal-manifest.json` to the remaining probes (and sets it
   around its own cascade), because the world-token identity (`hytalegym/jax/world/tokens`) reads the traversal manifest
   beside the line's Region library, which in a first capture exists only in the stage until promotion; the promotion copies
   the same bytes, so the identity the probes and the cascade record is the line's identity after promotion.
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
refuses, as it should. Pointers and defaults are never touched by the tool. A new bridge on an already promoted line is
promoted with `--apply --supersede`: the present evidence (identified by the bridge recorded in the line's traversal manifest or
projection contract) is moved into `world-evidence-history/<old bridge8>/superseded/` first and the receipt lists the moves
under `superseded_paths`; the sequence for a rebuilt bridge is: pin the new sha in `server_versions/v0_7.py`, write the
seed-570057 projection contract for it (`recapture_region.py --skip-capture --fluid-only-cap 6`, one fresh server launch),
run the first capture into a fresh stage (the line's promoted state does not change the mode decision until the supersede
moves it aside, so run the capture with `HYTALERL_TRAVERSAL_MANIFEST` unset and the references temporarily absent, or run
it in compare mode against the previous bridge's evidence if the world did not change), then promote with `--supersede` and
re-freeze the Arsenal Region suite (its content carries the bridge pin).

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

### Live runs (2026-10-19, first capture on 0.7.2)

- Run 1 (16:04 to 17:18): gate, seed projection (`408D5FB2` reproduced, 5 fluid-only cells of a cap of 6), capacity report,
  two byte-identical stationary captures, 20 graphs; stopped at the mutable-block probe, whose gate compared the audit's
  0.5.7 literal match instead of the per-line pin (fixed in the twelve live compare scripts, `expected_source_pin`, `2d1e993`).
- Run 2 (18:07 to 18:20, compile cache warm: stationary captures under a minute, graphs in five): the same captures, then the
  navigation, walk and the first five world-action probes; stopped at the crafting probe, whose import chain computes the
  world-token identity from the traversal manifest beside the Region library (absent until promotion, once the pointer
  keyed by profile landed mid-run). Fixed by the exported `HYTALERL_TRAVERSAL_MANIFEST` above and the tokens module's
  override and intermediate state (task #89).
- Run 3 (18:39 to 18:54, pointers keyed and the override exported): the same captures, the graphs, navigation, walk and
  seven world-action probes including crafting and item interaction; stopped at the block-use probe, whose typed synthetic
  place and break requests carry no `expected_candidate_generation_sha256` and `expected_selected_semantic_sha256`, which
  the current typed transport (v5, on both bridges) requires; the bridge fails closed with
  `native_policy_world_action_evidence_required`. The 0.5.7 fixture predates that rule. A repair lane binds the expectations
  from the world-action capture channel in the block-use and capabilities probes.
- Run 4 (19:26 to 19:42): passed end to end (`WORLD_EVIDENCE_RECERTIFICATION_PASS published=false`), 19 candidate rows, the
  `first_capture` block with the expected digests. The repaired block-use probe binds its typed requests from the capture and
  records the bridge's closures verbatim (place: `rotation_zero_destination_geometry_unavailable`; the synthetic break cell:
  `block_primary_unavailable`) with the Use path certified; the capabilities probe passes unchanged. Promoted at 19:44 (the
  tool's 44 checks all passed; every destination was absent). Rollback: the receipt's `created_paths`.
- Runs 5 to 7 (20:55 to 21:41, the rebuilt bridge `6F5B6021` of the batch `dbd30f2`, pin `fda3b20`), compare mode against
  the 460054A8 promotion: run 5 stopped at the source-pin gate (the census refold had made the mutable_blocks pin stale; the
  final acceptance round `8773844` fixed it); run 6 stopped at the graph compare's 0.5-only literal
  (`is_supported_hytale_05_release` in `native_region_traversal_graph_compare.py`, now a release-line check against the
  profile); run 7 passed end to end (18 candidate rows, 5 fluid-only cells of a cap of 6, the traversal library byte-identical
  to the first capture). Promoted at 21:43 with `--apply --supersede`: 14 destinations of the 460054A8 promotion moved into
  `world-evidence-history/460054A8/superseded/`, 60 files written, receipt `world-evidence-history/6F5B6021/publication-v12.json`.
  The capacity report of the first capture stays (compare mode does not re-measure it).
- Run 8 (23:18 to 23:34, the bridge `6EB44F30` of the host-contract batch `b682e33`, which pins the 0.7.2 learner-v3
  actor-evidence contract `E93BBEBE` and the NPC observation contract `3D080DE9`), compare mode against the 6F5B6021 promotion:
  passed end to end (`WORLD_EVIDENCE_RECERTIFICATION_PASS published=false`); traversal library semantic `CC613B12` unchanged, 20
  graphs. Promoted at 23:38 with `--apply --supersede`: 36 checks passed, 14 destinations of the 6F5B6021 promotion moved into
  `world-evidence-history/6F5B6021/superseded/`, 60 files written, receipt `world-evidence-history/6EB44F30/publication-v12.json`.
  The frozen v7 suite was re-frozen after the promotion (`493FD07E`): the promoted traversal manifest's
  `native_evidence_jar_sha256` and the arsenal contract both carry the bridge, so the value moves with every rebuild and promotion.
- Run 9 (00:06 to 00:23 on 2026-10-20, the bridge `9AABD344` of the fixture-aim and augment-readout batches b0717c9 and
  e643be4): stopped at the Region capture compare, seven cross-reset fluid-only cells against the seed projection contract's
  cap of six (all non-damaging, `other` 0; the count has risen with the machine load: 3, 5, 6, 7 over runs 4, 7, 8, 9; the
  5-minute load was 15 under seven lanes). The cap is an owner knob; run 10 is queued for a quiet machine with the pin
  `1CB4F651` (the request-tick fix a5cf904). Measured live beside it: the block-use probe's break cell is advertised once the
  pickaxe's slot is active (acknowledged), closed at reset only because the tool is not held (`break_authored_tool_unavailable`),
  and a refused typed request's receipt now carries the decision's world tick (it was -1, which `from_info` rejects).
- Run 10 (01:33 to 01:58 on 2026-10-20, the bridge `1CB4F651` of the request-tick fix a5cf904, on a quiet machine: 1-minute
  load 3.0 at launch): passed end to end; the seed projection matched the reference exactly (`exact_artifact_equal=true`,
  `non_damaging_fluid_only=0` of the cap 6), so the capture compare wrote no difference report and the promotion tool now
  treats an absent report as the exact case (check "seed projection exact", facts `seed_projection_exact_artifact_equal`).
  The block-use recording regenerated with the probe's slot selection and binding face: `typed_synthetic_status` stays
  `closed_by_bridge` because the bound place is refused by the contract gate (`rotation_zero_destination_geometry_unavailable`),
  while the synthetic break now executes (`Pickaxe_Attack`, mutation applied, inventory applied; the typed headless Use
  transport certified). Promoted at 02:02 with `--apply --supersede`: 34 checks passed, the 14 destinations of the 6EB44F30
  promotion moved into `world-evidence-history/6EB44F30/superseded/`, 59 files written, receipt
  `world-evidence-history/1CB4F651/publication-v12.json`. v7 re-frozen after it (`703ED690`).
- Run 11 (18:41 to 19:00 on 2026-10-21, the bridge `6DFD8682` of the gait batch 3024fde, 1-minute load 4.6 at launch): passed
  end to end (`WORLD_EVIDENCE_RECERTIFICATION_PASS published=false`; capture differential, traversal replay, mutable block,
  explosions, drop program, crafting catalog, item interaction, block use and world-action capabilities all PASS). Promoted with
  `--apply --supersede`: 36 checks passed, the 14 destinations of the 1CB4F651 promotion moved into
  `world-evidence-history/1CB4F651/superseded/`, 60 files written, receipt `world-evidence-history/6DFD8682/publication-v12.json`.
  v7 re-frozen after it (`CFB529B8`).
- Run 12 (19:06 to 19:25 on 2026-10-22, the bridge `1F874DFC` of 04c3b18, token capacity 96 and the jump that keeps momentum;
  1-minute load 5.9 at launch): passed end to end (`WORLD_EVIDENCE_RECERTIFICATION_PASS published=false`, every stage PASS).
  Promoted with `--apply --supersede`: 36 checks passed, the 14 destinations of the 6DFD8682 promotion moved into
  `world-evidence-history/6DFD8682/superseded/`, 60 files written, receipt `world-evidence-history/1F874DFC/publication-v12.json`.
  `tests/worldgen/test_native_world_recordings.py` passes again. v8 re-frozen after it (`54B3057C` -> `9058D229`).

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

## Block-semantic sidecar (Region block identity), 2026-10-22

Lane region-072 (task #130 item 5). Diagnosis and plan only. Nothing was produced, promoted or published. The 0.7.2 library has no sidecar, so the perception channel's block candidates are closed there.

### What it is (confirmed)

- Per Region: a palette-compressed block identity for every cell. Each palette entry carries `semantic_key` and `asset_key` (8 big-endian uint32 words each, from SHA-256), `valid`, `affordance_tags`, `gather_type_index`, `required_tool_quality` and `rotation_index`. Cell codes are `uint16[25][10][32768]`. Entry zero is air.
- 0.5.7 folder: `HytaleRL/artifacts/worldgen/region-library-pilot-v2/block-semantics-v1/`. `manifest.json` has schema `hytalerl_region_block_semantic_library_v1`, with one row per Region: `path`, `file_sha256`, `source_region_semantic_sha256`, `semantic_sha256`, `evidence_bridge_sha256`. Sidecars live in `sidecars/<source sha>.block-semantics.npz`. Their keys are `_SNAPSHOT_KEYS` (`hytalegym/worldgen/region/block_semantics.py`): `metadata` (schema `hytalerl_native_region_block_semantic_snapshot_v1`), `core_min_chunk_xz`, `section_known`, `cell_code`, and seven `palette_*` arrays. The 20 sidecars carry 350 to 614 palette entries each (measured).
- Writer: `capture_native_region_block_semantics` (`block_semantics.py:499`) and `capture_projected_native_region_block_semantics` (`:530`). Both take a live bridge transport and call `capture_region_block_semantic_section` once per section, 25 chunks times 10 sections (`_capture_block_semantic_sections`, `:583`). `NativeRegionBlockSemanticSnapshot.save` (`:246`) writes the file. `create_region_block_semantic_library` (`hytalegym/worldgen/region/block_semantic_library.py:181`) writes the manifest.
- Reader: `RegionBlockSemanticLibrary.load` (`block_semantic_library.py:71`) checks the source library, the contracts, the one-to-one pairing and both hash levels. `NativeRegionBlockSemanticSnapshot.load` (`block_semantics.py:316`) reads one file.
- Runtime: `hytalegym/jax/world/region/library.py:143-156`. With no manifest, the runtime builds `region_block_semantic_atlas_without_evidence` (every block mask false, block heads closed; `jax/world/region/block_semantics.py:90-120`). With a manifest, it builds the atlas from the snapshots.
- Bound by: `worlds/region.py:96` (`BLOCK_SEMANTIC_MANIFEST`, under `region-library-v3-288`), `:120-127` (`block_semantics_available()`), `:545`. No 0.7.2 folder is bound.

### How the 0.5.7 sidecar was produced (live, confirmed for the mechanism)

- `block-semantics-v1/STATUS.md` records evidence bridge `CF0A5CD5...BD2F1` and "twelve live physical bases were artifact-exact". The writer takes a transport, so it cannot run from files alone.
- The live caller of the same writer is `experimental/worldgen-v2/tools/native_v2_region_capture.py:531` (`capture_native_region_block_semantics`, then save, reload and identity check at `:542-556`). Not confirmed: which script wrote the 0.5.7 manifest. No file in the repo names the `block-semantics-v1` writer.

### Offline production for the 0.7.2 library: not possible (confirmed)

- The stored 0.7.2 NPZs carry no block identity. Their keys are `__metadata_json__`, `__core_min_chunk_xz__`, `__section_known__`, `__cell_code__`, `cell_flags`, `cell_shape_index`, `cell_fluid_*`, `cell_support`, `cell_block_damage`, `cell_fluid_damage`, `cell_movement`, `cell_fluid_movement`, `shape_boxes`, `shape_box_mask` and `__filler_root_offset_packed__`. The metadata has no block names (checked on `region_0.7.2_seed_1044320000.npz`).
- `__cell_code__` indexes a physical palette, not a block-id table. The 0.5.7 NPZs of the 20 seeds use 85 to 134 physical entries per Region; the 0.5.7 sidecars for the same Regions hold 350 to 614 identities. The 0.7.2 NPZs checked use 126 (seed 70970103) and 161 (seed 1044320000). Identity is lost when the Region is stored (scratch `inspect_057_sidecar.py`; the 0.7.2 counts from the NPZ `cell_flags` length).
- The identity keys come from the live server's `BlockType` asset ids (`hytale-plugin-0.7.2/src/main/java/com/hytalerlbridge/nativebackend/session/region/RegionGeometryCapture.java:642-683`). Assets.zip has no per-cell block id, so it cannot map a stored cell to a block.
- Conclusion: producing `block-semantics-v1` for the 0.7.2 library needs a live 0.7.2 bridge. The lane wrote no artifact.

### Plan for a 0.7.2 sidecar (live; not run)

1. Preconditions. The bridge jar pinned in `hytalegym/server_versions/v0_7.py` (`bridge_jar_sha256`; `6DFD8682...` on 2026-10-22). The 0.7.2 Region NPZs are the physical base: `artifacts/worldgen/region_0.7.2_seed_<seed>.npz`. Ports per "Before you run it" above. The bridge must expose `region_block_semantics` (`hytale-plugin-0.7.2/.../network/ClientHandler.java:197`, present).
2. Per seed. Start the 0.7.2 bridge the same way the recapture does. Load the NPZ with `NativeRegionSnapshot.load`. Call `capture_projected_native_region_block_semantics(transport, snapshot, evidence_bridge_sha256=<the pinned jar>)`. Use the projected form. The exact form (`capture_native_region_block_semantics`) requires the live physical digest to equal the NPZ digest, which fails on fluid drift (see "Why the exact scope is unreliable" above). The projected form accepts a live pass whose seed projection matches, or whose differences are at most four dry-physical cells (`_block_semantic_physical_alignment`, `block_semantics.py:649`). The 0.7.2 compare table records matching seed projections for all 20 seeds (inferred that the check passes on each).
3. Save each sidecar with `save`, reload it with `load`, and verify `semantic_sha256()` against the saved value before any publish.
4. Publish once, into a new folder under a run root (`publish-library` now does this, with the Region manifest) (never the shared artifacts without the owner's yes): `create_region_block_semantic_library(root, [sidecar paths], region_library_0_7_2)`. It pairs one sidecar per Region by `source_region_semantic_sha256`. For the 0.7.2 library the pairing key is the NPZ's complete digest (library `9DCA3C71...`, `artifact.semantic_sha256`).
5. Validate. `RegionBlockSemanticLibrary.load(manifest, region_library_0_7_2)`, then a `region_action_runtime_from_artifact_libraries` build with `block_semantic_manifest_path` pointing at the new manifest. Check that a block cell from a known sidecar has `block_present` true.
6. Cost (inferred from the 0.7.2 recapture timings). 251 wire calls per seed (one physical pass plus 250 sections). About 1.3 to 2.3 minutes per seed, so roughly 30 to 50 minutes for 20 seeds with three bridges in parallel.
7. Decisions for the owner. Which digest identifies the 0.7.2 library (the same question as the open item in "Open question before step 2"). Where the manifest is bound: `worlds/region.py` binds only `region-library-v3-288`. Promotion, which writes shared state.

Not done by this lane: no server, no bridge, no capture, no artifact, no promotion. The live step is the coordinator's to schedule.

### Capture in the run, backfill and publish (2026-10-22)

Both routes use `capture_sidecar` in `capture_block_semantics.py`, so the block pass, the save, the reload and the receipt are one piece of code.

- In the run (new captures). `examples/capture_native_region.py` takes `--block-semantics-output`. After the physical NPZ is saved, it runs the block pass on the same env, against the NPZ it reads back. `recapture_region.py` passes this flag unless `--physical-only` is given. A seed is complete only when its sidecar's receipt says the reload matched and the file is unchanged. If the block pass fails, the capture fails, the compare does not run, and the physical NPZ stays on disk.
- Backfill (captures that have no sidecar). `capture_block_semantics.py` runs the same block pass on a stored NPZ, one fresh bridge per seed. The owner's ruling (2026-10-22) is capture to a stage only: `HytaleRL/artifacts/worldgen/stage/block-semantics-072-20261022/` (`run.sh SEED ...` launches one fresh bridge per seed on 5556/25574 after the load is below 6; `sidecars/`, `logs/`). Smoke, seed 1044320000 on bridge 6DFD8682: 250 of 250 sections, the reloaded digest equals the captured one (1c9d5d23...), the live physical pass agreed with the stored Region.


- Publish. `recapture_region.py publish-library --version 0.7.2 --seed S [--seed S ...] --library-root <new folder>` followed by either `--inherit-split-from <0.5.7 manifest.json>` or `--heldout-count N --split-salt-sha256 H`. It writes the Region library manifest and the block-semantics manifest in one step. It copies each NPZ to `<root>/artifacts/seed_<seed>.npz` and each sidecar to `<root>/block-semantics-v1/sidecars/`. It refuses, and writes nothing, if any seed lacks its NPZ or a complete sidecar. If the block manifest fails, it removes the Region manifest too. Inheriting the split needs the new folder to sit beside the 0.5.7 folder (`create_region_artifact_library` checks this).
- Why the sidecar sits beside the NPZ: the publisher needs sidecars under its own root and NPZs under the library root, so it copies both. Beside the NPZ, each identity stays next to its physical Region until that copy is made.
- Publishing writes only the new folder. Promotion into `region-library-pilot-v2-072/` and the binding in `worlds/region.py` still wait for the owner.
- Tests: `hytalegym/tests/worldgen/test_recapture_block_identity.py` (offline; the `tests/**` tree is git-ignored, so this file is local only).

## World geometry tokens: why the collision box fields (`box_*`) read 0, 2026-10-22

Diagnosis of the box fields that read 0 in the Region world and live. Lane region-072, 2026-10-22 (task #130 item 6). Diagnosis only: no simulator change was made. Every claim is marked confirmed (file:line, or a command or probe run here) or inferred.

### Where the fields are filled (confirmed)

- Source data: `NativeRegionSnapshot.load` reads `shape_boxes` and `shape_box_mask` (`hytalegym/worldgen/region/snapshot.py:442-443`). Region atlas: `hytalegym/jax/world/region/atlas.py:311-312` gathers `collision_boxes` and `collision_box_mask` by each cell's shape, and `:347-348` returns them.
- Exception candidates for a Region: `hytalegym/jax/world/tokens/_candidates.py:248` (`_region_collision_exception_candidates`), dispatched from `:105-111`. `box_mask` is gated by `available & filler_available` (`:275-278`). Full cubes are split out (`:280-286`) and become surface tokens.
- Surface and zero-box candidate builders write zeros or broadcast masks (`_candidates.py:441`, `:501`, `:604`). A full-cube token therefore has zero box fields by design. Pinned by `tests/jax/world/tokens/test_tokens.py:171-190` (`test_ordinary_full_cube_ground_is_visible_without_a_traversal_atlas`, shared tree, git-ignored).
- Token output: `hytalegym/jax/world/tokens/__init__.py:480-481`. Policy floats: `hytalegym/jax/combat/observation/v3/tokens/world.py:427-443`. Each of the 9 boxes is a mask followed by `x0 y0 z0 x1 y1 z1`, divided by the maximum distance and clipped to plus or minus 1. The skill contract names these `box_{i}_{field}` (`experimental/neural_agent/skill_agent/contract/tokens_v3.py:191`), so the layout matches.

### Why they read 0 in the Region world

Probe (confirmed, run in this lane; scripts in the lane's scratch folder, `census_traversal.py` and `overflow_split.py`). Region seed 1044320000 from `region-library-pilot-v2-072`. Actors stood on traversal nodes inside the traversal core. Producer: `produce_actor_region_world_geometry_tokens` (`hytalegym/jax/world/region/traversal.py:438`, the Region world's producer), with `token_capacity=44` and `maximum_distance=4.0`. Sample: 120 random in-core nodes.

- 65 of 120 rows (54%) are cleared by overflow: `capacity_exceeded` is true, `available` is false, and every field is zero, boxes included. The same 65 positions are available with zero overflow under the plain producer (no traversal) at capacity 44. So traversal tokens cause the overflow. At capacity 128 all 65 recover, and 10 of them carry boxes.
- 55 of 120 rows are available. 16 of those (29%) have at least one boxed token.
- Re-run by the coordinator (2026-10-22, the same script on a fresh 200-node sample, the lane's worktree at the HEAD engine): 98 of 200 rows (49%) cleared by overflow; 102 available, 24 of them boxed; at capacity 128, 94 of the 98 recover. So about half of the in-core positions see no geometry tokens at all, not only zero box fields.
- Capacity census (coordinator, 2026-10-22; `HytaleRL/artifacts/fidelity_evidence/region_token_capacity/`): random in-core positions with a random facing, capacity 512, 200 per seed. Tokens a row needs, median / p99 / max: seed 1044320000 51 / 68 / 74, 70970103 50.5 / 72 / 82, 1618574389 52 / 75 / 78. Rows a capacity keeps: 44 keeps 34 to 42 %, 64 keeps 94.5 to 98 %, 96 keeps 100 %. With a random facing the loss at 44 is larger than the fixed-facing probe's half.
- Mechanism. Traversal and exception candidates share one budget: `visible_count` sums the eligible candidates of both (`hytalegym/jax/world/tokens/__init__.py:362-376`). Overflow clears the complete row (`__init__.py:183`, "clears the complete actor row"). The budget is `WORLD_GEOMETRY_TOKEN_CAPACITY = 44` (`hytalegym/jax/combat/observation/v3/tokens/world.py:25`), which the arsenal environment normalises into the Region producer (`hytalegym/jax/combat/arsenal/environment.py:1445`, `:3171`; no Region override found in `worlds/region.py`).
- The staged capacity report says the same. `artifacts/worldgen/region-library-pilot-v2-072/world-token-capacity.json`: `interpretation.capacity_rule` = "overflow clears the complete actor row; no nearest-token truncation". Its `limitation` says the traversal tokens "require a separate capacity measurement", and no such measurement is staged (confirmed by reading the report).
- Non-full shapes are rare, so most rows have no boxed token. Confirmed: 7,858 of 8,192,000 cells (0.096%) in seed 1044320000 belong to a non-full boxed shape. Full cubes are about a third of cells. The 0.5.7 and 0.7.2 values agree to within 0.00001 for all 20 seeds.
- Not a library gap. The 0.7.2 NPZ holds the boxes (115 active, 514 non-zero floats; confirmed). The 0.5.7 pilot Region with the same probe gives 64 available, 19 with boxes and 56 cleared out of 120 (confirmed; `census_057.py`). So the pattern is not specific to 0.7.2.
- Other zero rows. An actor outside the traversal core, or at the capture edge, is unavailable (the probe shows this at the 0.7.2 capture edge).

### Live (inferred from code; not run on a live server)

- Flat world. `GeometryFrame.fromSimulatorBlocks` (`hytale-plugin-0.7.2/src/main/java/com/hytalerlbridge/geometry/GeometryFrame.java:100-135`) makes solid blocks unit cubes and gives non-solid blocks no boxes. It sets `exact_collision_shapes` false. Unit cubes are surface tokens with zero box fields, so zeros are expected for the flat world. The flat frame would also be refused by `geometry_state_from_numpy` (`hytalegym/jax/world/types.py:162`, `:208`, `require_exact=True` by default). No caller in `hytalegym` passes `require_exact=False`, so the flat frame probably never reaches the JAX token producer. Not confirmed.
- Multi-actor composer. `hytalegym/jax/combat/observation/v3/native/channels/actor_major/projection.py:237` sets `world_geometry_available = False` on every row, and `native/codec/decode.py:847` then returns no geometry. Any live multi-actor observation therefore has no world geometry tokens, and its box fields read 0. This is a Python-side choice, not a bridge gap (confirmed by reading; not run live).

### Decision for the owner (no change made)

The zeros come from a fail-closed capacity contract, not a missing field or a bug in the fill path. The candidate fixes all change a pinned contract and need recertification:

1. Raise `WORLD_GEOMETRY_TOKEN_CAPACITY` above 44. Done for 0.7.2 on 2026-10-22 (96); see "Token capacity by line" below. This item said the change moves `world_geometry_token_contract_sha256`; it does not (that contract records the capacity as caller-selected). The policy, learner-observation and frozen Region v7 digests do move.
2. Reserve a separate budget for exception tokens, so traversal cannot evict boxes. Also a contract change.
3. Keep the rule and document it. Nearest-token truncation is ruled out by the capacity report's rule.

The cheapest first step is to measure traversal visible-token counts over the training distribution, then choose 1 or 2. The skill-layer entry at `experimental/neural_agent/skill_agent/docs/PERCEPTION.md:405` lists this gap (task #130 item 6).

### Token capacity by line (2026-10-22)

Change made on 0.7.2 (task #130 item 6, owner-approved 2026-10-22). Confirmed by the code and the snapshot runs named in the lane report; the census is the evidence for the number.

- The capacity is keyed by server line in `hytalegym/server_versions/` (`world_geometry_token_capacity`, with `world_geometry_capacity_evidence`): 0.5.7 = 44 (unchanged), 0.7.2 = 96. `WORLD_GEOMETRY_TOKEN_CAPACITY` and `DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG` (`hytalegym/jax/combat/observation/v3/tokens/world.py`) read it at import, like the other per-line sizes, so set `HYTALERL_SERVER_VERSION` before importing. `produce_actor_region_world_geometry_tokens` defaults to it (`hytalegym/jax/world/region/traversal.py`). An explicit `WorldGeometryPolicyConfig(token_capacity=...)` still wins.
- Why: the census above. On 0.7.2, 44 keeps 34 to 42 % of in-core rows. The skill layer's wider census (`fidelity_evidence/region_token_capacity/skill_layer_census_*.log`): pilot-v2-072, all 20 seeds x 200 nodes, 3,983 rows: median 50, p99 72, p99.9 78, max 84; kept at 44 / 64 / 80 / 96: 38.0 / 96.5 / 99.95 / 100 %. region-library-v3-288 (0.5-era terrain), every 9th seed (32) x 100 nodes, 3,178 rows: max 81, 96 keeps 100 %. 96 leaves 12 tokens over the largest need seen. A sample, not an exhaustive census: rerun it before training on a new world type. An overflowed row is invisible to the agent (capacity_exceeded is not in the observation): it looks like terrain outside the captured core.
- What moves on 0.7.2: `arsenal_policy_contract_sha256` (6B8949A2... to 62AC787C...), `learner_observation_v3_contract_sha256` (A0970ED7... to 17B6FAAA...), and `ARSENAL_POLICY_OBSERVATION_SIZE` (8272 to 15656). `world_geometry_token_contract_sha256` does not move. Nothing moves on 0.5.7.
- Frozen Region suites. v7 (`ARSENAL_REGION_V7_LINES["0.7.2"]` in `hytalegym/jax/evaluation/_contracts.py`, recert run 11, CFB529B8...) stays a record. Its live contents embed the default config, so `region_v7_suite_contract()` now fails closed with a drift error (live digest 1F271FB8...). The active 0.7.2 suite is v8 (`ARSENAL_REGION_V8_LINES["0.7.2"]`, frozen `54B3057C...` at 04c3b18, re-frozen `9058D229...` after recertification run 12), entered through `load_region_v8_context` and `run_arsenal_region_v8_trained`. No default pointer selects a suite (grep found none); the caller chooses the runner. The artifact registry (`hytalegym/jax/evaluation/_artifact.py`) maps v8 to its digest.
- Existing 0.7.2 checkpoints become incompatible. A checkpoint saved with the old default carries `observation_size` 8272 and the 44-row digests. Under the new default the transfer contract compares each field (`hytalegym/jax/training/checkpoint.py:214-236`), so it reports the checkpoint as mismatched (read from the code; not run). Training is paused and these policies are throwaway, so retrain, or load one only with an explicit `WorldGeometryPolicyConfig(token_capacity=44)`.
- Deployment. `experimental/jvm-agent-0.7.2` moved to 96 rows and width 15656 together: `ObservationAssembler.OBSERVATION_SIZE` and `GEOMETRY_TOKENS`, the geometry and light `TOKEN_CAPACITY`, the `ParityTest` group widths, and the `export_projection.py` headers, which now derive from the line's capacity. `experimental/jvm-agent` (the 0.5 copy) is unchanged.
- Not changed: `REGION_TRAVERSAL_POLICY_STAGING_CAPACITY` (56, per-actor traversal-node staging), and the `token_capacity=44` default of `query_region_traversal_tokens` (node staging; every production caller passes 56).
- The bridge changes with it. The learner-v3 actor-evidence contract embeds the observation contract, so on 0.7.2 it moves E93BBEBE... to FC928136... (`NativeActorEvidenceFrame.CONTRACT_SHA256` in the 0.7.2 bridge, and the 0.7.2 entry in `worldgen/native_npc_traces.py`). The Python side and the jar land together, or every live handshake fails. The jar of this change also carries the jump momentum fix (task #132).
- Still on 44 or 8272, left to their owners: `experimental/neural_agent/skill_agent/contract/tokens_v3.py:253-256`, `state/senses.py:95, 478`, `state/memory.py:54` (the neural layer); `arena/training/imitation/native_replay_cache.py:371` (`8 * 44` slice); `console/docs/API.md:355` and `console/core/execution/jobs.py:798, 1164` (8272 in notes).
- The staged 0.7.2 capacity report (`artifacts/worldgen/region-library-pilot-v2-072/world-token-capacity.json`) has rows 4, 8, 16 and 32 only. It needs a first capture with 44 and 96 rows. Its producer, `tests/fidelity/surrogate_world_token_benchmark.py`, is gitignored and is not in the lane worktree.
