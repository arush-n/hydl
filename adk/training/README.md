# ADK training adapters

`adk.training` connects an `AgentHandle` and its compiled collection contract
to optional learning algorithms. It is an adapter layer: the environment and
policy surfaces remain in [`runtime/`](../runtime/README.md) and
[`policy/`](../policy/README.md).

The package is useful for direct ADK PPO workflows and lower-level experiments.
The supported project training lesson is owned by Arena's pursuit collector and
is launched through the Console.

Start with [`ppo_adapter.py`](ppo_adapter.py) and the optional algorithm entry
point in [`../algorithms/`](../algorithms/README.md).
