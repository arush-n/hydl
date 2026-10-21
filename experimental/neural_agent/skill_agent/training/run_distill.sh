#!/bin/bash
# One DAgger run of the shared generalist (distill.py) at nice 10 on the CPU, as every A2 run so far (README 2.16), on
# the gen-7 render by default (GEN=5: the stock engine; engine.sh), rendered into <run>/engine.
# Usage: run_distill.sh <run name> <iterations> <lanes> <steps> <eval lanes> [distill options, e.g. --look-direction]
# Writes report.json, params.npz and log.txt (the start line records the commit and the options, and the load) to
# agent_skills/training_runs/skill_agent_a2/<run name>/, which is private and never staged, and refuses a name that
# exists. It launches the A2 reference trainer (the joint network of runs A2 to C'); skills train through their own
# strategies (../neural_skills/<skill>/train.sh, README 2.17).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$ROOT"
NAME=$1 ITERATIONS=$2 LANES=$3 STEPS=$4 EVAL_LANES=$5
shift 5
OUT=agent_skills/training_runs/skill_agent_a2/$NAME
if [ -e "$OUT" ]; then echo "refusing: $OUT exists" >&2; exit 2; fi
mkdir -p "$OUT"
ENGINE_DIR=$OUT/engine
source experimental/neural_agent/skill_agent/training/engine.sh
export PYTHONPATH=$ROOT:$ROOT/HytaleRL/hytalegym JAX_PLATFORMS=cpu
{ echo "start $(date -u +%FT%TZ) head $(git rev-parse --short HEAD) gen $GEN options: $*"; uptime; } > "$OUT/log.txt"
nice -n 10 $PYTHON_M experimental.neural_agent.skill_agent.training.distill \
    --iterations "$ITERATIONS" --lanes "$LANES" --steps "$STEPS" --seed 0 --eval-seed 77001 --eval-lanes "$EVAL_LANES" \
    --out "$OUT/report.json" --params-out "$OUT/params.npz" "$@" > /dev/null 2>> "$OUT/log.txt"
{ echo "end $(date -u +%FT%TZ)"; uptime; } >> "$OUT/log.txt"
echo "done: $OUT"
