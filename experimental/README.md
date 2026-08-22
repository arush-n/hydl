# Optional integrations

This tree holds integrations that are useful to the published runtime but are
not part of the stable top-level package names. The release surface here is
limited to the native policy-agent implementation and the WorldGen V2 source
consumed by Arena's world providers.

The native policy agent depends on the HytaleRL bridge and ADK deployment
contracts. WorldGen V2 supplies authoring, capture, compilation, and JAX
loading seams used by [`arena/worlds.py`](../arena/worlds.py) and
[`arena/publication_worlds.py`](../arena/publication_worlds.py). Captured
evidence and local test harnesses are not runtime dependencies.

## Entry points

- [`jvm-agent/README.md`](jvm-agent/README.md) covers the Java-side policy
  integration and export tools.
- [`worldgen-v2/README.md`](worldgen-v2/README.md) covers custom environment
  recipes and the JAX world-provider path.

Start with [`arena/README.md`](../arena/README.md) if you want to use these
integrations through the task and training layer.
