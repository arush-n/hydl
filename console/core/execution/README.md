# Console execution

`console.core.execution` turns a validated request into a rollout, training,
self-play, check, or native-server job. It owns admission, runner selection,
worker lifecycle, metrics, and completion reporting; it delegates task and
environment semantics to Arena, ADK, and HytaleGym.

The HTTP API and panels call this package. It depends on the catalog, evidence,
telemetry, and local runtime configuration, but those packages do not launch
jobs themselves.

## Entry points

- [`runner.py`](runner.py) defines the ordinary rollout specification and
  validation.
- [`jobs.py`](jobs.py) owns preflight, queueing, launch, status, replay, and
  checkpoint routes.
- [`runners.py`](runners.py) selects the available local or WSL compute path.
- [`ladder.py`](ladder.py) drives the objective curriculum: it launches ONE
  rung, grades it on the success rate that run's own selection reported, and
  lets `arena.curriculum.ladder.Ladder` decide whether the next run promotes,
  repeats the rung or demotes. Queueing every rung up front is what made the
  gate thresholds decoration, since two of the three outcomes are not "the next
  rung". A run the driver did not launch is ignored, and a cancelled or faulted
  one stops the ladder rather than being recorded as a failed rung.
- [`selfplay.py`](selfplay.py) adapts the self-play job boundary.
- [`hytale.py`](hytale.py) manages an explicitly requested native server
  session.

Continue to [`../README.md`](../README.md) for the shared core services and to
[`../../docs/API.md`](../../docs/API.md) for the HTTP surface.
