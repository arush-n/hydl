#!/bin/bash
# walk_to's own training strategy (train.py: its teacher, the skills beside it, its lanes and its sizes), which trains
# walk_to's module on the shared frozen trunk. Writes the module and its report to
# agent_skills/training_runs/skill_modules/walk_to/<run name>/ (private, never staged) with the log beside it, and
# refuses a name that exists. On the gen-7 render by default (owner, 2026-10-20 night; GEN=5: the stock engine;
# ../../training/engine.sh), rendered into <run name>.engine beside it.
# Usage: train.sh <run name> <trunk .npz> [train.py options, e.g. --iterations 2 --lanes 32 --steps 200]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
NAME=${1:?usage: train.sh <run name> <trunk .npz> [train.py options]}
TRUNK=${2:?usage: train.sh <run name> <trunk .npz> [train.py options]}
shift 2
OUT=$ROOT/agent_skills/training_runs/skill_modules/walk_to/$NAME
if [ -e "$OUT" ] || [ -e "$OUT.log" ]; then echo "refusing: $OUT exists" >&2; exit 2; fi
cd "$ROOT"
mkdir -p "$(dirname "$OUT")"
ENGINE_DIR=$OUT.engine
source experimental/neural_agent/skill_agent/training/engine.sh
export PYTHONPATH=$ROOT:$ROOT/HytaleRL/hytalegym JAX_PLATFORMS=cpu
{ echo "start $(date -u +%FT%TZ) head $(git rev-parse --short HEAD) gen $GEN trunk $TRUNK options: $*"; uptime; } > "$OUT.log"
nice -n 10 $PYTHON_M experimental.neural_agent.skill_agent.neural_skills.walk_to.train \
    --trunk "$TRUNK" --out "$OUT" "$@" > /dev/null 2>> "$OUT.log"
{ echo "end $(date -u +%FT%TZ)"; uptime; } >> "$OUT.log"
echo "done: $OUT"
