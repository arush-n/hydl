# ADK runtime

`adk.runtime` is the algorithm-neutral execution layer. It owns lossless
environment adaptation, structured actions, episode boundaries, compiled
collection, rollouts, recurrent carry, scene construction, and optional
imagination.

It depends on HytaleGym and ADK policy/architecture contracts. Algorithms such
as PPO consume it; the runtime does not import an optimizer, Arena task, or
Console service.

## Entry points

- [`env_adapter.py`](env_adapter.py) binds the environment step boundary.
- [`actions.py`](actions.py) defines factor transport and receipts.
- [`loop.py`](loop.py), [`rollout.py`](rollout.py), and
  [`hierarchical.py`](hierarchical.py) collect policies.
- [`scene_builder.py`](scene_builder.py) and [`scenes.py`](scenes.py) build
  named, stamped environments.
- [`episode.py`](episode.py) and [`imagination.py`](imagination.py) handle
  boundaries and branch rollouts.
