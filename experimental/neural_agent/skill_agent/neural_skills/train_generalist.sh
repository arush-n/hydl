#!/bin/bash
# The generalist's current stage, which every skill folder's train.sh runs: A2's size with the options of run C',
# which passes all of A2's bars (README 2.16): walk_to v3's stop margin and the look direction in the head's frame.
# The size and the options are set here and nowhere else; the ledger records every run.
# Usage: train_generalist.sh <run name> [extra distill options]
set -euo pipefail
NAME=${1:?usage: train.sh <run name> [extra distill options]}
shift
HERE=$(cd "$(dirname "$0")" && pwd)
exec "$HERE/../training/run_distill.sh" "$NAME" 6 128 2000 256 \
    --walk-stop-margin 0.25 --look-direction --look-frame head "$@"
