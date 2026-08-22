# PPO WorldGen tasks

`agents.ppo.worldgen` defines small WorldGen-backed task compositions and the
imitation/benchmark workflows that exercise them. The task constructors use
Arena `Task`, goal, reward, and world contracts; the package location reflects
the agent workflow rather than a second task framework.

[`minigames.py`](minigames.py) contains custom pressure/control tasks,
[`worldgen_il.py`](worldgen_il.py) handles imitation learning, and
[`worldgen_trace.py`](worldgen_trace.py) handles trace construction. World
providers come from Arena and the WorldGen V2 integration.

For the reusable behavior-shaping minigames, see
[`adk/scenarios/minigames/`](../../../adk/scenarios/minigames/README.md).
