"""Small public API for configuring and training the Basic agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from arena.training.runs.pursuit_run import PursuitRunConfig, run_pursuit_training


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "basic_policy.npz"


@dataclass(frozen=True, slots=True)
class BasicAgent:
    """Movement and 3D-target-tracking foundation for later combat training."""

    checkpoint: Path = DEFAULT_CHECKPOINT

    def configure(self, output: Path, **overrides: Any) -> PursuitRunConfig:
        """Build the content-validated long-horizon JAX training request."""

        return PursuitRunConfig(
            source_policy=self.checkpoint,
            output=output,
            **overrides,
        )

    def train(
        self,
        output: Path,
        *,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
        **overrides: Any,
    ) -> dict[str, object]:
        """Run the configured shared-GPU pursuit curriculum."""

        return run_pursuit_training(
            self.configure(output, **overrides),
            progress=progress,
        )


__all__ = ["BasicAgent", "DEFAULT_CHECKPOINT", "ROOT"]
