# skill_agent/training/ — data and trainers

Training runs only from the plan the owner approved (`../docs/TRAINING_PLAN.md`). Private data (traces, datasets,
checkpoints) stays under `agent_skills/training_runs/` and is never staged.

| Module | Lines | What it holds |
|---|---:|---|
| `traces.py` | 222 | Native NPC trace ingestion: recorded `NpcTraceFrame` batches as arrays with a documented schema (A7). |
| `bc_dataset.py` | 751 | The behaviour-cloning dataset from those traces on the gen-7 surface; the compass labelled in the commanded body frame (v4); the self token's head yaw offset (v5). |
| `bc_smoke.py` | 235 | B2's smoke: a tiny behaviour-cloning body held out by world, and the trained body run through the combiner. |
| `distill.py` | 754 | The A2 reference trainer: DAgger distillation of walk_to and look_at together into one generalist head (runs A2 to C', README 2.16), kept to reproduce those runs; per-skill training (2.17) replaced it as the training design. It uses `blocks.py` and the two skills' `skill.py`. |
| `bars.py` | 189 | A report scored on its bars on held success, the first hit beside: a skill module's report (per lane kind: T5, held out, the skills beside it, E4) or an A2 reference report (per mode, with a skill filter). |
| `blocks.py` | 312 | The building blocks every trainer shares (moved from `distill.py`, 2026-10-19): frames, request features, resampling, the hold rule, legality, dataset chunks, saving and loading, run provenance. |
| `sequences.py` | 349 | Training through time on the v3 trunk (`../docs/TRAINING_PLAN.md` 8.2 item 2, stage G1; V1 adds the packed recording, `cut` with a stride and per-lane ends, and `take`): `record` keeps each rollout decision after the state layer (tokens, object slots, the stores' new objects, resets, the carried state), packed to its valid rows with `dtype` and the carry kept only at `carry_at`; `cut` makes sequences of up to L decisions that never cross a reset, each with its stored start, overlapping by L minus the stride; `take` picks sequences (packed or dense); `unroll` re-runs the trunk over them (`lax.scan` over `step`, pure JAX); `pack` and `unpack` keep the valid rows in float16 or float32. |
| `distill_v3.py` | 648 | The V1 reference trainer (`../docs/TRAINING_PLAN.md` 8.3 and 8.8; 2026-10-23): A2's walk_to and look_at task, teachers and evaluation (reused from `distill.py`), with run C''s options by default (walk_to v3's stop margin, the look direction in the head's frame; README 2.16), on the v3 state layer, with the agent's own pose as its channel, and a trunk with memory, trained through time on stored sequences (burn-in of 30 on 90, R2D2). Each batch is cut to the token slots the data fills (`_filled`: 39 of 356 on the flat world; the same values to float32 rounding, 6 to 13 times faster). `--control` is the memory-less run, `--smoke` one iteration of 300 steps, `--params-out` saves the trained parameters. |
| `skill_dagger.py` | 519 | DAgger for one skill's module on the frozen trunk (2.17): lanes, requests, the teachers of the skill and of the skills beside it, rollout, loss, training, evaluation per lane kind, and the command line every skill's `train.py` shares (`run_main`). |
| `run_distill.sh` | 25 | The A2 reference trainer's launcher: one run at nice 10 on the CPU, `run_distill.sh NAME ITERATIONS LANES STEPS EVAL_LANES [options]`, into the private runs folder, refusing a name that exists, on the gen-7 render by default (`engine.sh`, rendered into the run's `engine/`). Skills train through their own `train.sh`. |
| `engine.sh` | 23 | The engine a launcher runs on (2026-10-20, owner: train on gen 7): sourced with `ENGINE_DIR` set, it sets `PYTHON_M`; GEN=7, the default, renders gen 7 from the committed source into `ENGINE_DIR` and mounts it (`../runtime/gen7.py`), GEN=5 keeps the stock engine. The reference, the skills' `train.sh` and `certify.sh` use it. |

Run:

```sh
PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym .venv/bin/python -m \
    experimental.neural_agent.skill_agent.training.traces [--root DIR] [--json]
PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym .venv/bin/python -m \
    experimental.neural_agent.skill_agent.training.bc_dataset [--root DIR] [--out PATH]
PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu .venv/bin/python -m \
    experimental.neural_agent.skill_agent.training.bc_smoke \
    [--steps 1500] [--batch 256] [--seed 0] [--dataset NPZ] [--out PATH]
PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python -m \
    experimental.neural_agent.skill_agent.training.distill --iterations 6 --lanes 128 --steps 2000 --seed 0 \
    --eval-seed 77001 --eval-lanes 256 --out REPORT.json [--params-out PARAMS.npz] \
    [--walk-stop-margin 0.25] [--look-direction [--look-frame head]]
PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym .venv/bin/python -m \
    experimental.neural_agent.skill_agent.training.bars REPORT.json [--skill walk_to|look_at]
```

Gates: `test_sa_traces.py`, `test_sa_bc_dataset.py`, `test_sa_bc_smoke.py`, `test_sa_distill.py`, `test_sa_bars.py`,
`test_sa_skill_modules.py`, `test_sa_sequences.py`, `test_sa_distill_v3.py`. A skill trains by its own strategy: `../neural_skills/<skill>/train.sh NAME TRUNK.npz`
(its `train.py`).
