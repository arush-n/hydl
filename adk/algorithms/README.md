# Algorithms

`adk.algorithms` contains optional learning consumers for the ADK collection
and training contracts. The package owns optimizer/update integration; the
runtime remains usable without importing an algorithm.

[`ppo.py`](ppo.py) adapts recurrent PPO to an `AgentHandle`, while
[`neutral.py`](neutral.py) provides the algorithm-neutral update seam used by
other consumers. These modules depend on [`runtime/`](../runtime/README.md)
and the HytaleGym training primitives, not on the Console.

For the project’s supported Console-launched lesson, continue to
[`arena/training/`](../../arena/training/README.md).
