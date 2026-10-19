#!/bin/bash
# look_at's bars for one run (../../training/bars.py): T5 and held out on held success, the first hit beside.
# Usage: evaluate.sh <run name, or a run folder>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
RUN=${1:?usage: evaluate.sh <run name, or a run folder>}
if [ -d "$RUN" ]; then RUN=$(cd "$RUN" && pwd); else RUN=$ROOT/agent_skills/training_runs/skill_agent_a2/$RUN; fi
cd "$ROOT"
export PYTHONPATH=$ROOT:$ROOT/HytaleRL/hytalegym
exec .venv/bin/python -m experimental.neural_agent.skill_agent.training.bars "$RUN" --skill look_at
