# Worlds

Training and evaluation run inside **world libraries**: folders of captured
terrain that agents move through. This page covers choosing a library, choosing
which worlds from it a run uses, and adding a library of your own.

## Listing what is available

Libraries are discovered by scanning a folder, so a library exists because its
directory does. Nothing needs registering.

```python
from worlds import libraries

libraries.describe()
```

```json
{
  "libraries": [
    {"id": "region-library-pilot-v1", "counts": {"train": 16, "heldout": 4}, "default": false},
    {"id": "region-library-pilot-v2", "counts": {"train": 16, "heldout": 4}, "default": false},
    {"id": "region-library-v3-288",   "counts": {"train": 256, "heldout": 32}, "default": true}
  ],
  "orders": ["digest", "flattest", "mixed"],
  "default_split": "train"
}
```

The Console serves the same thing at `GET /api/worlds`.

Each library reports how many worlds it holds per **split**. `train` worlds are
the ones a run may select; `heldout` worlds are reserved for evaluation and are
never chosen for training.

## Choosing worlds for a run

A run asks for a number of worlds and an **order**. The order decides which
worlds it gets:

| order | what you get | use it when |
| --- | --- | --- |
| `digest` | the library's own ranking, keyed | you want a representative sample |
| `flattest` | the flattest worlds first | an agent is still learning to move |
| `mixed` | flat and ranked, interleaved | you want easy ground and hard ground together |

```python
from worlds import libraries

# Sixteen of the flattest worlds in the default library.
seeds = libraries.select(16, order="flattest")

# The same count from a specific library, using its own ranking.
seeds = libraries.select(16, order="digest", library="region-library-pilot-v2")
```

`key` shifts the `digest` and `mixed` orderings, so two runs with different keys
draw different worlds:

```python
libraries.select(16, order="digest", key=1)
libraries.select(16, order="digest", key=2)   # a different sixteen
```

`flattest` ignores `key` by design — the flattest worlds are the flattest
worlds.

**Terrain matters more than it looks.** Worlds vary widely in how much open,
walkable ground they contain. A run drawn with `digest` can include worlds that
are almost entirely cliff, where an agent spends most of an episode in the air
and its movement scores describe the terrain rather than the agent. If you are
training movement from scratch, start with `flattest` and widen later.

Selection fails loudly rather than quietly returning fewer worlds:

```python
libraries.select(999, order="digest")
# ValueError: library 'region-library-v3-288' split 'train' has 256 worlds,
#             fewer than the 999 requested
```

## From the Console

The training request takes the count and the order:

```json
{
  "world_count": 16,
  "world_order": "flattest"
}
```

`world_count` above 1 spreads a run's environments across the selected worlds;
pair it with `environment_diversity: true` so each environment also draws its
own spawn and target inside its world.

Training runs currently always use the default library. `GET /api/worlds` lists
the others so you can see what exists, but choosing one per run is not wired up
yet — use `libraries.select(..., library=...)` directly if you need a specific
library today.

## Adding a library

Put a directory beside the existing libraries containing a `manifest.json` that
declares the region-artifact schema:

```json
{
  "schema": "hytalerl_region_artifact_library_v1",
  "library_semantic_sha256": "…",
  "entries": [
    {"seed": 1234, "split": "train", "semantic_sha256": "…", "path": "…"}
  ]
}
```

It appears in `describe()` and in `GET /api/worlds` immediately — no code change
and no restart of anything that reads the list.

## Adding an order

Orders live in one mapping, `libraries.ORDERS`, keyed by the name a run asks
for. An order is a function that receives the library's entries and a key and
returns every seed in the order it prefers; the caller takes as many as it
asked for. Adding one is adding an entry to that mapping.
