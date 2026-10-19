#!/bin/bash
# look_at trains inside the shared generalist together with walk_to, so this runs the generalist's current stage
# (../train_generalist.sh: its size and options). Usage: train.sh <run name> [extra distill options]
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/../train_generalist.sh" "$@"
