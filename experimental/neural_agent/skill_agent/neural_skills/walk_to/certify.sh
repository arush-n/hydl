#!/bin/bash
# walk_to's teacher certificate, T1 to T3 (../../docs/TRAINING_PLAN.md 2c), on the four certificate seeds x 256
# episodes, written to the private certificates folder. Refuses to overwrite a certificate.
# On the gen-7 render by default (GEN=5: the stock engine; ../../training/engine.sh), rendered beside the certificate.
# Usage: certify.sh [stop margin in blocks, default 0.25: v3; 0 is v2] [name, default walk_to_v3_gen<GEN>]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
MARGIN=${1:-0.25}
NAME=${2:-walk_to_v3_gen${GEN:-7}}
OUT=$ROOT/agent_skills/training_runs/skill_agent_a2/certificates/$NAME.json
if [ -e "$OUT" ]; then echo "refusing: $OUT exists" >&2; exit 2; fi
cd "$ROOT"
ENGINE_DIR=${OUT%.json}.engine
source experimental/neural_agent/skill_agent/training/engine.sh
export PYTHONPATH=$ROOT:$ROOT/HytaleRL/hytalegym JAX_PLATFORMS=cpu
nice -n 10 $PYTHON_M experimental.neural_agent.skill_agent.teaching.certificates --teacher walk_to \
    --seeds 77001,66001,66002,66003 --episodes 256 --walk-stop-margin "$MARGIN" --out "$OUT" --date "$(date +%F)"
