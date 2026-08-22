"""Apply shared accelerator defaults before ADK imports JAX."""

from hytalegym.runtime_environment import configure_runtime_environment


configure_runtime_environment()
