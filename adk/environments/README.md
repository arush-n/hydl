# Agent environments

`adk.environments` declares scenes and world-capability providers: open or
fail-closed combat, captured regions, synthetic frames, terrain surveys, and
water/geometry checks.

It depends on HytaleGym and the shared world library. It is upstream of
[`scenarios/`](../scenarios/README.md), because a scenario can only be trusted
when its scene can express the mechanic it measures. It does not define Arena
task success or Console job ownership.

## Entry points

- [`config.py`](config.py) provides `SceneConfig` and `BuiltScene`.
- [`frames.py`](frames.py) and [`synthetic.py`](synthetic.py) define compact
  geometry sources.
- [`water.py`](water.py), [`frames.py`](frames.py), and [`blocks.py`](blocks.py)
  inspect world capability.

Use the root [`worlds/README.md`](../../worlds/README.md) for captured-world
selection used by Arena training.
