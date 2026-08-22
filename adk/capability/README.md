# Capability measurement

`adk.capability` isolates action heads and measures what a scene can actually
express. Its purpose is to distinguish an unavailable mechanic from a policy
that has not learned it.

The package depends on the runtime action masks and HytaleGym scene providers.
It is measurement infrastructure, not a reward or training objective. Use
[`forced.py`](forced.py) to pin unrelated factors neutral,
[`measure.py`](measure.py) for capability summaries, and
[`scenes.py`](scenes.py) for named capability surfaces.

Pair it with [`probes/`](../probes/README.md) when validating a new scene.
