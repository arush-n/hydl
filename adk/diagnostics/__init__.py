"""Tracing, failure decoding, and views for debugging an agent's inputs.

Split by cost, because the distinction matters in a rollout:

* **Traced, sync-free** -- safe inside ``jit``/``scan``:
  :func:`failure_flags`, :func:`observation_failures`, :func:`any_failure`,
  :func:`standard_trace` and the other recorders, :func:`summarize`,
  :func:`top_down_occupancy`, :func:`top_down_rgb`.
* **Host-side** -- transfers data and stalls, for startup checks and debugging:
  :func:`explain_failures` and every ``render_*``.

Nothing here is tuned to an algorithm; it reads the published observation and
returns plain arrays or plain text.
"""

from adk.diagnostics.failures import (
    ARSENAL_FAILURES,
    CAPACITY_OVERFLOWS,
    DIAGNOSTIC_FIELDS,
    FAILURE_FIELDS,
    MECHANICS_FAILURES,
    OBSERVATION_FAILURES,
    OVERFLOW_FIELDS,
    any_failure,
    capacity_overflows,
    explain_failures,
    failure_flags,
    failure_names,
    observation_failures,
)
from adk.diagnostics.render import (
    DENSITY_RAMP,
    render_ascii,
    render_availability,
    render_fields,
    render_summary,
    token_positions,
    top_down_occupancy,
    top_down_rgb,
)
from adk.diagnostics.trace import (
    action_trace,
    boundary_trace,
    combine,
    failure_trace,
    standard_trace,
    summarize,
)

__all__ = [
    "ARSENAL_FAILURES",
    "CAPACITY_OVERFLOWS",
    "DENSITY_RAMP",
    "DIAGNOSTIC_FIELDS",
    "FAILURE_FIELDS",
    "MECHANICS_FAILURES",
    "OBSERVATION_FAILURES",
    "OVERFLOW_FIELDS",
    "action_trace",
    "any_failure",
    "boundary_trace",
    "capacity_overflows",
    "combine",
    "explain_failures",
    "failure_flags",
    "failure_names",
    "failure_trace",
    "observation_failures",
    "render_ascii",
    "render_availability",
    "render_fields",
    "render_summary",
    "standard_trace",
    "summarize",
    "token_positions",
    "top_down_occupancy",
    "top_down_rgb",
]
