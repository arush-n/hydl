# skill_agent/training/ — data and trainers

Training runs only from the plan the owner approved (`../docs/TRAINING_PLAN.md`). Private data (traces, datasets,
checkpoints) stays under `agent_skills/training_runs/` and is never staged.

| Module | Lines | What it holds |
|---|---:|---|
| `traces.py` | 222 | Native NPC trace ingestion: recorded `NpcTraceFrame` batches as arrays with a documented schema (A7). |
| `bc_dataset.py` | 751 | The behaviour-cloning dataset from those traces on the gen-7 surface; the compass labelled in the commanded body frame (v4); the self token's head yaw offset (v5). |
| `bc_smoke.py` | 235 | B2's smoke: a tiny behaviour-cloning body held out by world, and the trained body run through the combiner. |
| `distill.py` | 1067 | DAgger distillation of the request-conditioned generalist: walk_to and look_at from the two scripted teachers, with held-out argument regions and a report of student against teacher, first hit and held 5 ticks (A2; README 2.16). Opt-in: walk_to v3's stop margin, the look request's direction, in the body's or the head's frame. |
| `bars.py` | 115 | A distillation report scored on A2's bars on held success, the first hit beside: T5, held out and E4 per mode, composition reported, a skill filter (README 2.16). |
| `run_distill.sh` | 21 | One DAgger run at nice 10 on the CPU: `run_distill.sh NAME ITERATIONS LANES STEPS EVAL_LANES [options]`, outputs to the private runs folder, refusing a name that exists. The skills' `train.sh` scripts call it through `../neural_skills/train_generalist.sh`. |

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

Gates: `test_sa_traces.py`, `test_sa_bc_dataset.py`, `test_sa_bc_smoke.py`, `test_sa_distill.py`, `test_sa_bars.py`.
