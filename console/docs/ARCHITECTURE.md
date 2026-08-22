# Architecture

```
console/
├── server.py          assembly + entry point. Wires router and static; no logic.
├── api/
│   ├── routes.py      HTTP endpoints. Each one is a thin translation onto core.
│   └── schemas.py     wire contract (pydantic), mirroring RunSpec.
├── core/
│   ├── runner.py      the engine: RunSpec -> trajectory + summary + warnings.
│   ├── worldgen.py    deterministic custom-world preview + saved design contract.
│   └── surfaces.py    read-only views of the ADK and the native bridge.
├── static/            UI. Hand-rolled orbit camera on Canvas 2D, no library.
└── docs/              you are here.
```

## The one rule

**`core/` must stay usable without FastAPI.** The browser is one consumer of the
engine, not its owner:

```python
from console.core.execution.runner import RunSpec, run
r = run(RunSpec(loadout="iron_mace", decision_period=8, ticks=512))
```

If a route grows logic of its own, that logic belongs in `core` instead. The
practical test: could a pytest call it without starting a server?

## Data flow

```
browser form
   │  POST /api/run  (RunRequest)
   ▼
api/routes.py ──► core/runner.RunSpec ──► adk.AgentKit.make_scene
                                             │
                                    handle.compile_collector(policy, record)
                                             │
                          trajectory + summary + warnings  ──► browser ──► 3D
```

Nothing is cached or pre-baked. Every press of **Run rollout** compiles and
executes against the live JAX environment, which is why the first run of a new
configuration costs ~40 s and subsequent ones are fast.

## It is a frontend for the ADK, not a reimplementation

`core/surfaces.py` reads every value from the ADK's own published surface:

| shown | read from |
| --- | --- |
| scene name, contract SHA, bound providers | `adk.describe_scene` |
| loadouts | `adk.list_loadouts` |
| action heads and logits | `adk.probes.policies.HEAD_SPANS` |
| observation groups and columns | `adk.policy.fields.GROUP_FEATURES` |
| provider seams | `adk.JAX_SCENE_PROVIDER_ARGUMENTS` |

Nothing is restated as a literal, so the console cannot silently drift from the
library. If the ADK's action surface changes, the status bar changes with it.

## Native bridge: read-only, on purpose

`surfaces.bridge_status()` reports **reachability only** — it checks whether
the local native listeners are available and returns.

It deliberately does **not** open a `NativeBridgeSession`, because opening one
takes the native evidence lease and would evict whatever lane currently holds
it. A real native differential goes through the native fidelity runner, which
manages the lease properly and reports "not evaluated" rather than stealing it.

## Why no 3D library

The CSP on published artifacts blocks external hosts, and there is no vendored
`three.js` in the tree. A hand-rolled orbit camera is ~60 lines
(`static/app.js`, `project()`), needs no build step, and works offline. The
rollout scene is a ground plane plus two bodies. WorldGen Studio adds a 49x49
semantic height field. Its default renderer quantizes cell tops and emits only
exposed vertical faces, producing a voxel surface; an optional smooth mode uses
triangles. Trees, structures, water, and the semantic cave graph are separate
painter-sorted layers. This remains height-field extrusion rather than a
volumetric occupancy renderer, so caves are shown as a cutaway overlay rather
than falsely rendered as exact native void voxels.

## WorldGen data flow and boundary

```
browser controls -> preview/batch API -> core/worldgen.py -> Canvas voxel/top view
       ^                |                  |     |
       |                v                  |     +-> compact seed plan (<=512)
 saved design <- grouped manifest         |
                                             +-> offline design_compiler
                                                   |
                                                   v
                                      pinned V2 pack + receipt (not deployed)
```

The core functions are independently testable (`worldgen.preview(config)` and
`worldgen.batch(request)`) and own normalization, rectangular geometry,
deterministic generation, content placement, cave topology, metrics, warnings,
hashing, caching, and seed plans. JavaScript owns controls and rendering.

The compiler is a deliberately narrow adapter onto hash-pinned installed
WorldGen V2 templates. It records mapped and preview-only controls rather than
equating the semantic Python preview with Hytale's native density graph. A
saved manifest or compiled pack records the next native steps; the console
never converts a preview into an exact Region, deploys it, or mutates a JAX
corpus pin. Neutral spawn/high/low/structure/cave anchors support the separate
minigame lane without implementing rules or rewards here.

Width/length are a downstream capture contract because native generation is
unbounded. `extent_capture_plan()` converts them to a grid of 96x96-core capture
tiles. A one-tile request can use a smaller rectangular bounds window. Larger
requests are now represented by the composite-world contract: adjacent cores
captured during the same native reset are validated at their physical, block,
and fluid overlaps and materialized as **one JAX world ID** with tiled lookup.

`exact_port_supported_now=true` means that loader and manifest contract exist;
it does not mean the requested native artifacts already exist. Every tile still
requires exact capture. Cross-core physical lookup is available when the
captured composite includes the required physical and traversal seams.
See the published
[`experimental/worldgen-v2/README.md`](../../experimental/worldgen-v2/README.md)
for the world-generation boundary.

## Four invariants `run()` has to hold

Each of these was a live bug found by auditing the run path, each was silent in
a different way, and each is now guarded. If you touch `core/runner.py`, these
are the things to not break.

**1. A policy is called with three arguments, always.** `collect_transitions`
passes `(carry, actor_input, key)`. The console wraps policies in two places —
`_throttle` and `_carried` — and both used to ask whether the policy had an
`initial_carry` attribute to decide which convention it used. That is a
different question: `uniform_legal` and `strike_when_ready()` both take a carry,
return it untouched, and set no attribute. Use `_takes_carry`, which reads the
arity. Guarded by `tests/test_policy_shapes.py` (17 cases, ~1.5 s, no scene
build).

**2. Nothing is counted before the episode is truncated.** The environment does
not auto-reset, so past the first `done` the agent is a corpse and the flag
repeats every tick. Measured on a thinned-health duel: the episode ended at tick
**60 of 256**, so 196 corpse ticks — 77% of the run — were being folded into
`damage_dealt`, `reward_total`, `accepted`, `hit_rate` and `stamina_floor` while
the trajectory the browser drew stopped at 60. The summary panel and the chart
underneath it disagreed, and the panel was the wrong one. Truncate first, then
count, and read every number back out of `trajectory` — a local list built
before the cut still holds the full length.

**3. A non-finite value is a finding, not a crash.** Starlette renders with
`allow_nan=False`, so a single NaN anywhere in the payload raised *"Out of range
float values are not JSON compliant"* at render time — an opaque HTTP 500 after
the entire rollout had been paid for, naming neither column nor tick. `int(nan)`
failed earlier still. Non-finite values now become `null`, are counted per
series into `summary.nonfinite`, and raise a warning; reductions go through
`finite()`. This matters because a NaN carried into policy state disables an NPC
permanently and nothing in the environment raises when it happens.

**4. One rollout at a time.** `_BUILD_LOCK` (reentrant — `run` holds it across
its own call to `_scene_for`) serialises build, compile and execute. Two
concurrent requests otherwise mutate `_SCENE_CACHE` and `_COLLECTOR_CACHE`
check-then-set and race the `_LAST_CACHE_HIT` global, and peak device memory is
set by how many compile at once. `checks.py` and `jobs.py` already refuse to run
two at a time; this is the same rule for the path that costs the most.

A fifth, at the HTTP edge: `/api/run` catches non-`ValueError` exceptions and
returns the type, message and last frames. A bare 500 whose body is the string
`Internal Server Error` is the one failure a debug console must not have — both
policy bugs above were only findable by reading the server's stdout.

## What this is not

- **Not a fidelity harness.** Everything here is JAX-side.
- **One environment row is *drawn*.** `ENVIRONMENT = 0` of a rollout's batch of
  2 is what the map and the per-step readouts record. Training is a separate
  width — see `jobs.TRAIN_ENVS` — because a batch sized for drawing collects too
  little to learn from.
