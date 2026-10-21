#!/bin/bash
# Sourced by the skill layer's launchers: sets PYTHON_M, the command that runs a Python module on the run's engine
# (owner, 2026-10-20 night: train on gen 7, not gen 5). GEN=7, the default, renders the gen-7 simulator from the
# committed source into ENGINE_DIR (about 10 s; its MANIFEST.json records the source) and mounts it
# (../runtime/gen7.py); GEN=5 runs the stock engine. ENGINE_DIR must be a new folder of the run, so the engine a result
# came from stays beside it. From the repository root:
#     ENGINE_DIR=<run>.engine source experimental/neural_agent/skill_agent/training/engine.sh
#     $PYTHON_M experimental.neural_agent.skill_agent.<module> [args]
GEN=${GEN:-7}
case "$GEN" in
    7)
        : "${ENGINE_DIR:?set ENGINE_DIR to a new folder for the gen-7 render}"
        if [ -e "$ENGINE_DIR" ]; then echo "refusing: $ENGINE_DIR exists" >&2; exit 2; fi
        PYTHON_M=".venv/bin/python -u -m experimental.neural_agent.skill_agent.runtime.gen7 --engine $ENGINE_DIR --render --"
        ;;
    5)
        PYTHON_M=".venv/bin/python -u -m"
        ;;
    *)
        echo "GEN must be 7 or 5, got $GEN" >&2
        exit 2
        ;;
esac
