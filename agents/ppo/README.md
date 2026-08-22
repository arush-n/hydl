# PPO agents

`agents.ppo` contains broader policy workflows built on ADK and Arena/HytaleGym
interfaces: combat profiles, native behavior imitation and transfer, WorldGen
tasks, and lower-level PPO entry points. It is a family of experiments and
adapters, not the Console's `basic` profile.

The package depends on the ADK policy/runtime surface and can consume Arena
task/world contracts. Native paths additionally require the HytaleRL bridge.

## Entry points

- [`agent.py`](agent.py) exposes the general PPO agent surface.
- [`native/`](native) contains native imitation and fidelity workflows.
- [`worldgen/README.md`](worldgen/README.md) covers WorldGen-specific tasks.
- [`__main__.py`](__main__.py) provides the package command-line dispatch.

For supported project training, use [`agents/basic/`](../basic/README.md) via
the Console.
