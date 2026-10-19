# skill_agent/training/ — data and trainers

Training runs only from the plan the owner approved (`../docs/TRAINING_PLAN.md`). Private data (traces, datasets,
checkpoints) stays under `agent_skills/training_runs/` and is never staged.

| Module | Lines | What it holds |
|---|---:|---|
| `traces.py` | 222 | Native NPC trace ingestion: recorded `NpcTraceFrame` batches as arrays with a documented schema (A7). |
| `bc_dataset.py` | 751 | The behaviour-cloning dataset from those traces on the gen-7 surface; the compass labelled in the commanded body frame (v4); the self token's head yaw offset (v5). |
| `bc_smoke.py` | 235 | B2's smoke: a tiny behaviour-cloning body held out by world, and the trained body run through the combiner. |
| `distill.py` | 953 | DAgger distillation of the request-conditioned generalist: walk_to and look_at from the two scripted teachers, with held-out argument regions and a report of student against teacher (A2; README 2.16). |

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
    --eval-seed 77001 --eval-lanes 256 --out REPORT.json [--params-out PARAMS.npz]
```

Gates: `test_sa_traces.py`, `test_sa_bc_dataset.py`, `test_sa_bc_smoke.py`, `test_sa_distill.py`.
