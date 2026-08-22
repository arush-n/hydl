# Console worlds

`console.core.worlds` is the Console's world-authoring and terrain-reference
boundary. It validates deterministic designs, custom-game inputs, captured
world references, and the terrain patches used by run and replay views. It
does not silently turn a preview into a native capture or redefine Arena's
world-provider contract.

The API and panels depend on this package. It depends on the WorldGen and
HytaleRL world seams, while execution consumes its normalized design and
terrain references.

## Entry points

- [`worldgen.py`](worldgen.py) normalizes designs, previews terrain, and manages
  authored world records.
- [`custom_games.py`](custom_games.py) validates custom minigame manifests.
- [`custom_goals.py`](custom_goals.py) supplies the finite goal vocabulary.
- [`terrain.py`](terrain.py) resolves terrain patches for map and replay views.
- [`replay_terrain.py`](replay_terrain.py) preserves the seed and semantic
  identity used to render the correct world.

Continue to [`../README.md`](../README.md) for the Console core and to
[`../../docs/API.md`](../../docs/API.md) for the HTTP endpoints.
