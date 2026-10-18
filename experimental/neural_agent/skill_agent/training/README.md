# skill_agent/training/ — data and trainers

Training runs only from the plan the owner approved (`../docs/TRAINING_PLAN.md`). Private data (traces, datasets,
checkpoints) stays under `agent_skills/training_runs/` and is never staged.

| Module | Lines | What it holds |
|---|---:|---|
| `traces.py` | 222 | Native NPC trace ingestion: recorded `NpcTraceFrame` batches as arrays with a documented schema (A7). |
| `bc_dataset.py` | 745 | The behaviour-cloning dataset from those traces on the gen-7 surface; the compass labelled in the commanded body frame (v4). |
| `bc_smoke.py` | 235 | B2's smoke: a tiny behaviour-cloning body held out by world, and the trained body run through the combiner. |

Run:

```sh
PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym .venv/bin/python -m \
    experimental.neural_agent.skill_agent.training.traces [--root DIR] [--json]
PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym .venv/bin/python -m \
    experimental.neural_agent.skill_agent.training.bc_dataset [--root DIR] [--out PATH]
PYTHONPATH=/Users/arush/hydl:/Users/arush/hydl/HytaleRL/hytalegym JAX_PLATFORMS=cpu .venv/bin/python -m \
    experimental.neural_agent.skill_agent.training.bc_smoke \
    [--steps 1500] [--batch 256] [--seed 0] [--dataset NPZ] [--out PATH]
```

Gates: `test_sa_traces.py`, `test_sa_bc_dataset.py`, `test_sa_bc_smoke.py`.
