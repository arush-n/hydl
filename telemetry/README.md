# Telemetry

This directory is the local landing place for runtime telemetry emitted by the
server and bridge. Telemetry is operational output, not a source dependency;
the published code that consumes live streams lives under
[`console/core/telemetry/`](../console/core/telemetry/README.md).

Readers who need the data path should start with the Console telemetry package.
This directory itself does not define a schema or a training interface.
