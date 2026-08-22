# ADK scenarios

`adk.scenarios` describes behavior-focused scenes: what a policy is asked to
do, what shaping term is added, which scripted probes can exercise it, and what
ceiling or evidence makes the result meaningful.

It depends on [`environments/`](../environments/README.md), the ADK runtime,
and HytaleGym. These scenarios are distinct from Arena's complete `Task`
catalog: an ADK scenario may be a small diagnostic or shaping composition.

## Entry points

- [`core.py`](core.py) defines scenario identity and trajectory scoring.
- [`tasks.py`](tasks.py) defines native combat task presets.
- [`shaping.py`](shaping.py) composes additive reward terms.
- [`minigames/README.md`](minigames/README.md) documents the reusable shaping
  objectives.
- [`probes.py`](probes.py) contains scripted scenario drivers.
