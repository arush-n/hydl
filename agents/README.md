# Agents

`agents` contains concrete policy families, profiles, and Console-facing
adapters. Arena owns the task and training contracts; ADK supplies the public
agent runtime; this package chooses how a particular agent is configured and
run.

The Console drives the `basic` agent through its HTTP training workflow. The
agent adapter resolves a run profile into the Arena pursuit runner, while the
PPO family contains broader combat, native-transfer, and WorldGen examples.
Profiles are declarative JSON inputs rather than learned weights; checkpoints
and run output are not part of the published source surface.

## Entry points

- [`basic/`](basic/README.md) is the Console-launched recurrent pursuit agent.
- [`ppo/`](ppo/README.md) contains general PPO and WorldGen policy workflows.
- [`profiles/`](profiles/README.md) contains named configuration profiles.
- The root package also exposes [`training_runtime.py`](training_runtime.py)
  for shared runtime setup.

Start the supported training flow in the root [`README.md`](../README.md), then
read the Console documentation before launching a run.
