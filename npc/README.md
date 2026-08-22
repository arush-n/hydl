# Native NPC runner

`npc` is the live-server execution surface for a trained policy. It converts a
policy's structured actions into the HytaleRL bridge protocol, records the
authoritative response, and keeps the native session boundary separate from
JAX training.

It depends on ADK policy/runtime contracts and the HytaleRL bridge. It does not
own task definitions or PPO; Arena and the gym provide those layers.

## Entry points

- [`loop.py`](loop.py) contains `serve`, action-surface checks, and the native
  execution report.
- [`__main__.py`](__main__.py) is the command-line entry point.

Use the Console for training and use this package when a selected checkpoint is
ready to be evaluated as a native NPC.
