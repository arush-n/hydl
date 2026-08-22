"""Whatever is training, whatever its architecture.

The worked example of the `series` view, and the reason metric streams exist:
this panel does not know what a metric is. It charts the keys each stream
declared by logging them, so a Dreamer's world-model loss, a transformer's
attention entropy and PPO's four losses all land here without the console
learning about any of them.

Any process can feed it, including one the console never launched::

    curl -X POST localhost:8770/api/streams/my-run/log \\
         -H 'content-type: application/json' \\
         -d '{"step": 1, "metrics": {"loss": 0.4}, "meta": {"arch": "cnn"}}'
"""

from __future__ import annotations

from typing import Any

from console.core.telemetry import streams
from console.core.panels import panel


@panel("metric-streams", title="Metric streams", tab="train", order=10,
       agent="built-in", refresh=10.0,
       description="Any learner that logs here is charted, whatever it reports.")
def metric_streams() -> dict[str, Any]:
    index = streams.index()
    if not index:
        return {
            "view": "note",
            "text": (
                "No streams yet. POST to /api/streams/<id>/log and the stream "
                "is created by its first line — nothing has to be registered "
                "first, so an architecture that is not built yet can still "
                "appear here the moment it reports."
            ),
        }

    newest = index[0]
    data = streams.series(newest["id"]) or {"series": []}
    arch = (newest.get("meta") or {}).get("arch")

    note = f"{len(index)} stream(s); showing the most recent, {newest['id']}"
    if arch:
        note += f" ({arch})"
    if newest.get("non_finite_keys"):
        # A NaN loss is usually the finding in this repo. It is recorded as
        # rejected rather than plotted, so it must be said out loud -- on the
        # chart it would be indistinguishable from a gap.
        note += (". Non-finite values were rejected for: "
                 + ", ".join(newest["non_finite_keys"]))

    return {
        "view": "series",
        "series": data["series"],
        "count": f"{newest['points']} points · {len(newest['keys'])} keys",
        "note": note,
    }
