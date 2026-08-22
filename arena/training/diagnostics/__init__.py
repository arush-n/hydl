"""Measurement, not training.

Nothing here changes a policy. These modules answer "is this signal even
learnable?" before a run spends hours proving it is not.

``timing_probe``
    On-device separability diagnostics for binary timing decisions.
``decision_interval``
    Executes one policy decision across physical combat ticks, so a decision
    period can be measured rather than assumed.
"""

from __future__ import annotations

from arena.training.diagnostics.decision_interval import (
    AdvancePhysicalStep,
    DECISION_INTERVAL_PROGRAM_SCHEMA,
    DecisionIntervalLifecycle,
    DecisionIntervalPhysicalStep,
    DecisionIntervalProgram,
    DecisionIntervalResult,
    DecisionIntervalTargetStep,
    TargetStep,
    run_decision_interval,
)
from arena.training.diagnostics.timing_probe import (
    BINARY_TIMING_SEPARABILITY_PROBE_SCHEMA,
    BinaryTimingProbeConfig,
    BinaryTimingProbeCounts,
    BinaryTimingProbeFit,
    BinaryTimingProbeGates,
    BinaryTimingProbeMetrics,
    BinaryTimingSeparabilityResult,
    LATENT_TIMING_AUROC_GATE,
    LATENT_TIMING_BALANCED_NLL_GATE,
    RAW_TIMING_AUROC_GATE,
    TimingNegativeKind,
    binary_timing_probe_metrics,
    binary_timing_probe_report,
    fit_binary_timing_separability_probe,
)


__all__ = [
    "AdvancePhysicalStep",
    "BINARY_TIMING_SEPARABILITY_PROBE_SCHEMA",
    "BinaryTimingProbeConfig",
    "BinaryTimingProbeCounts",
    "BinaryTimingProbeFit",
    "BinaryTimingProbeGates",
    "BinaryTimingProbeMetrics",
    "BinaryTimingSeparabilityResult",
    "DECISION_INTERVAL_PROGRAM_SCHEMA",
    "DecisionIntervalLifecycle",
    "DecisionIntervalPhysicalStep",
    "DecisionIntervalProgram",
    "DecisionIntervalResult",
    "DecisionIntervalTargetStep",
    "LATENT_TIMING_AUROC_GATE",
    "LATENT_TIMING_BALANCED_NLL_GATE",
    "RAW_TIMING_AUROC_GATE",
    "TargetStep",
    "TimingNegativeKind",
    "binary_timing_probe_metrics",
    "binary_timing_probe_report",
    "fit_binary_timing_separability_probe",
    "run_decision_interval",
]
