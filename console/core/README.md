# Console core services

`console.core` contains the service-side orchestration beneath the HTTP routes:
catalogs, evidence storage, job execution, telemetry, world authoring, and
small shared storage helpers.

It depends on Arena, ADK, HytaleRL, and the local filesystem, but the dependency
direction remains downward from the Console. Core services translate a request
into an owned job or report; they do not redefine the underlying task or
physics contracts.

## Main areas

- [`catalog/`](catalog) resolves agent profiles and contract identities.
- [`evidence/`](evidence/README.md) stores reports, archives, and replay data.
- [`execution/`](execution) validates devices and launches workers.
- [`telemetry/`](telemetry) exposes live and persisted runtime streams.
- [`worlds/`](worlds) authors designs, custom games, and replay terrain.

The browser-facing payloads are in [`../panels/`](../panels/README.md).
