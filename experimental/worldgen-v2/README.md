# WorldGen V2 integration

This package contains the authoring, capture, compilation, and JAX loading
seams for bounded WorldGen V2 environments. Arena uses its bundle and world
modules to bind terrain-aware tasks while keeping captured geometry separate
from the rollout code.

The package depends on Hytale asset/server inputs for authoring or native
capture. Its published source can load declared bundles and recipes without
shipping those external inputs. Arena imports the public world-provider
modules; the package does not import Arena or ADK.

## Entry points

- [`custom_environments/`](custom_environments) defines recipes and publication
  helpers.
- [`design_compiler/`](design_compiler) turns an authored environment into a
  validated design.
- [`jax_port/`](jax_port) contains the runtime world, bundle, structure, and
  traversal consumers.
- [`native_graph_compiler/`](native_graph_compiler) compiles native graph
  evidence into portable bindings.

Arena's world binding starts at [`arena/worlds.py`](../../arena/worlds.py) and
[`arena/publication_worlds.py`](../../arena/publication_worlds.py).
