# Console panels

`console.panels` turns core service state into browser panel payloads. Panels
are presentation adapters: they read jobs, devices, worlds, evidence, Java
status, and telemetry, then return stable data for the static application.

They depend on [`console.core`](../core/README.md) and should not contain task
rewards, PPO, or environment stepping. The panel registry is in
[`__init__.py`](__init__.py); individual modules group related views such as
compute, profiles, runs, receipts, Java, and diagnostics.

The client that consumes these payloads is in [`../static/`](../static/README.md).
