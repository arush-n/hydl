"""CLI entrypoint for Basic-agent JAX training."""

from arena.training.runs.pursuit_run import main
from agents.basic.agent import DEFAULT_CHECKPOINT

if __name__ == "__main__":
    main(default_source_policy=DEFAULT_CHECKPOINT)
