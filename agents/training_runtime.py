"""Pre-JAX runtime selection shared by agent training entry points."""

from hytalegym.runtime_environment import configure_training_runtime


TRAINING_RUNTIME_ENVIRONMENT = configure_training_runtime()


__all__ = ["TRAINING_RUNTIME_ENVIRONMENT"]
