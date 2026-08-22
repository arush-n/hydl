"""Optional learning algorithms built on the general SDK runtime.

The runtime and collection contracts do not depend on this package.  An
algorithm consumes ``AgentHandle.collect`` (or the lower-level JAX environment)
and owns its parameters, optimizer state, batch projection, and update rule.
"""

from adk.algorithms.ppo import PPOTools

__all__ = ["PPOTools"]
