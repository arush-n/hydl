#!/bin/bash
# look_at's bars for one run (../../training/bars.py): its own module's run (skill_modules/look_at/), or an A2
# reference run (skill_agent_a2/) read for look_at's modes. T5 and held out on held success, the first hit beside.
# Usage: evaluate.sh <run name, or a run folder>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
RUN=${1:?usage: evaluate.sh <run name, or a run folder>}
if [ -d "$RUN" ]; then RUN=$(cd "$RUN" && pwd)
elif [ -d "$ROOT/agent_skills/training_runs/skill_modules/look_at/$RUN" ]; then RUN=$ROOT/agent_skills/training_runs/skill_modules/look_at/$RUN
else RUN=$ROOT/agent_skills/training_runs/skill_agent_a2/$RUN; fi
cd "$ROOT"
export PYTHONPATH=$ROOT:$ROOT/HytaleRL/hytalegym
exec .venv/bin/python -m experimental.neural_agent.skill_agent.training.bars "$RUN" --skill look_at
