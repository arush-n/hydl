# Console

The Console is the FastAPI service that turns a training or evaluation request
into a validated, recorded job. It owns preflight, runner selection, evidence,
archive views, replay data, and the browser interface; it does not define the
environment physics or the task reward laws.

The Console depends on Arena for pursuit and task execution and on ADK for
ordinary scene construction, probes, telemetry, and shaped objectives. The
current recurrent pursuit path is launched through `POST /api/train`; the
background worker then receives a resolved run specification. Direct trainer
module invocation is outside this interface.

## Entry points

- [`server.py`](server.py) starts the service.
- [`api/`](api/README.md) defines request schemas and HTTP routes.
- [`core/execution/`](core/README.md) resolves devices, jobs, workers, and
  pursuit configuration.
- [`core/evidence/`](core/evidence/README.md) stores reports, archives, and
  replay/world evidence.
- [`panels/`](panels/README.md) supplies browser-facing panel payloads.
- [`static/`](static/README.md) contains the live browser application.
- [`docs/`](docs/README.md) contains the HTTP and operator walkthroughs.

Run configuration and the supported request shape are documented in
[`docs/API.md`](docs/API.md); the root README shows the supported pursuit
workflow.
