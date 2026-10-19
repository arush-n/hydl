#!/bin/bash
# walk_to's own training strategy (train.py: its teacher, the skills beside it, its lanes and its sizes), which trains
# walk_to's module on the shared frozen trunk. Writes the module and its report to
# agent_skills/training_runs/skill_modules/walk_to/<run name>/ (private, never staged) with the log beside it, and
# refuses a name that exists. Training is paused (owner, 2026-10-19).
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
export PYTHONPATH=$ROOT:$ROOT/HytaleRL/hytalegym JAX_PLATFORMS=cpu
{ echo "start $(date -u +%FT%TZ) head $(git rev-parse --short HEAD) trunk $TRUNK options: $*"; uptime; } > "$OUT.log"
nice -n 10 .venv/bin/python -u -m experimental.neural_agent.skill_agent.neural_skills.walk_to.train \
    --trunk "$TRUNK" --out "$OUT" "$@" > /dev/null 2>> "$OUT.log"
{ echo "end $(date -u +%FT%TZ)"; uptime; } >> "$OUT.log"
echo "done: $OUT"
