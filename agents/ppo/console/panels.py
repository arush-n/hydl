"""PPO-owned Evidence frame.

Only the artifact selection is PPO-specific.  The reader, missing-field
behavior, gate colors, and six-row card all come from the console's generic
declared-contract renderer.
"""

from __future__ import annotations

from pathlib import Path

from console.core.panels import panel
from console.panels.evidence import contract_payload

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _latest_summaries(limit: int = 2) -> list[Path]:
    if not ARTIFACTS.is_dir():
        return []
    return sorted(
        ARTIFACTS.rglob("*summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]


@panel(
    "ppo-evidence",
    title="PPO evidence",
    tab="evidence",
    order=20,
    agent="ppo",
    refresh=30.0,
    size="wide",
    tags=("JAX", "lane-owned"),
    description="The newest summaries PPO already writes, through the shared contract reader.",
)
def ppo_evidence():
    return contract_payload(
        _latest_summaries(),
        lane="ppo",
        selection="PPO newest *summary.json by file mtime",
        summary=(
            "PPO selects only its summary artifacts here. It does not define "
            "how status, gates, limitations, or identity are interpreted."
        ),
    )

