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
2. Publish the line's library through `hytalegym/hytalegym/worldgen/region/recertification.py`, which stages and then promotes. That path runs `verify_installed_world_source_pins` before it touches the listener. `FIDELITY-0.7.2.md` (ad) records it refusing 0.7.2 at audit 1 until the audits are closed.
3. Rebuild the products that depend on the terrain: path tables, task suites, the biome table and catch-time horizons (`hytale-plugin-0.7.2/SIMULATOR-0.7.2.md`, P2, names the tools). The path-table generator is named at SIMULATOR-0.7.2.md:95. That path is under the private skills tree, which this lane did not read. No generator for the biome table, the task suites or the horizons was found in `hytalegym/` outside `sim/`, in `examples/` or in `arena/`.
4. Flip the defaults that point at `region-library-pilot-v2`: `hytalegym/hytalegym/jax/world/tokens/__init__.py` (lines 87 to 95), `hytalegym/hytalegym/worldgen/region/recertification.py` (lines 141 and 791 to 797) and `hytalegym/hytalegym/sim/common/bench.py` (line 62).
5. Keep the 0.5.7 libraries and the staged NPZs. The provider finds a library by its semantic hash under `HYTALERL_WORLD_ARTIFACT_ROOT` (`hytalegym/hytalegym/jax/world/region/provider.py`), so old and new libraries can sit side by side. Ambiguous configured artifacts raise.

Open question before step 2 (owner's call): each library manifest records every NPZ's complete `semantic_sha256`. A library built from the staged 0.7.2 NPZ would therefore carry the NPZ's complete digest, which a fresh capture on this bridge does not reproduce. The projection digest is the stable value for this seed. Which digest identifies the 0.7.2 library is not decided here.

## Identities

The 0.7.2 bridge pin, the seed, the staged NPZ and its digests, the contract file and the compare runs are recorded in `artifacts/worldgen/world-evidence-history/460054A8/STATUS.md`. They are volatile, so they stay out of this README.
