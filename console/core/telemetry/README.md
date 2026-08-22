# Console telemetry

`console.core.telemetry` is the Console's read-and-publish layer for live job
events, metric streams, runtime logs, and service-status summaries. It keeps
operational observations separate from the training contracts and from the
files that a server writes on disk.

The Console API and panels depend on this package. It depends on the Console
configuration and storage conventions, but it does not own Arena tasks or
agent policies.

## Entry points

- [`live.py`](live.py) publishes events and returns snapshots for streaming
  clients.
- [`streams.py`](streams.py) records and reads named metric series.
- [`logs.py`](logs.py) indexes available log sources and tails them.
- [`surfaces.py`](surfaces.py) reports the status of the ADK and native bridge
  surfaces.

For the surrounding execution flow, continue to [`../README.md`](../README.md)
and [`../../../README.md`](../../../README.md).
