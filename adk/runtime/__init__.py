"""Lazy, algorithm-neutral runtime exports for agent authors."""

from __future__ import annotations

import importlib


_EXPORTS = {
    "CompositeActionHead": ("adk.runtime.actions", "ActionHead"),
    "CompositeActionSpec": ("adk.runtime.actions", "CompositeActionSpec"),
    "EnvironmentActionBoundary": (
        "adk.runtime.actions",
        "EnvironmentActionBoundary",
    ),
    "EnvironmentActionReceipt": (
        "adk.runtime.actions",
        "EnvironmentActionReceipt",
    ),
    "EpisodeBoundary": ("adk.runtime.episode", "EpisodeBoundary"),
    "HierarchicalActorInputFn": (
        "adk.runtime.hierarchical",
        "HierarchicalActorInputFn",
    ),
    "HierarchicalContextFn": (
        "adk.runtime.hierarchical",
        "HierarchicalContextFn",
    ),
    "HierarchicalLoopState": (
        "adk.runtime.hierarchical",
        "HierarchicalLoopState",
    ),
    "HierarchicalLoopTransition": (
        "adk.runtime.hierarchical",
        "HierarchicalLoopTransition",
    ),
    "HierarchicalRecordFn": (
        "adk.runtime.hierarchical",
        "HierarchicalRecordFn",
    ),
    "HierarchicalRuntimeDiagnostics": (
        "adk.runtime.hierarchical",
        "HierarchicalRuntimeDiagnostics",
    ),
    "JointActionCapability": (
        "adk.runtime.actions",
        "JointActionCapability",
    ),
    "JointActionRejected": ("adk.runtime.actions", "JointActionRejected"),
    "JointExecutionFailed": (
        "adk.runtime.actions",
        "JointExecutionFailed",
    ),
    "JointExecutionUnavailable": (
        "adk.runtime.actions",
        "JointExecutionUnavailable",
    ),
    "JointValidationUnavailable": (
        "adk.runtime.actions",
        "JointValidationUnavailable",
    ),
    "MaskedCompositeSample": (
        "adk.runtime.actions",
        "MaskedCompositeSample",
    ),
    "StructuredActionCodec": (
        "adk.runtime.actions",
        "StructuredActionCodec",
    ),
    "collect_hierarchical": (
        "adk.runtime.hierarchical",
        "collect_hierarchical",
    ),
    "compile_hierarchical_collector": (
        "adk.runtime.hierarchical",
        "compile_hierarchical_collector",
    ),
    "jax_hierarchical_context": (
        "adk.runtime.hierarchical",
        "jax_hierarchical_context",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "CompositeActionHead",
    "CompositeActionSpec",
    "EnvironmentActionBoundary",
    "EnvironmentActionReceipt",
    "EpisodeBoundary",
    "HierarchicalActorInputFn",
    "HierarchicalContextFn",
    "HierarchicalLoopState",
    "HierarchicalLoopTransition",
    "HierarchicalRecordFn",
    "HierarchicalRuntimeDiagnostics",
    "JointActionCapability",
    "JointActionRejected",
    "JointExecutionFailed",
    "JointExecutionUnavailable",
    "JointValidationUnavailable",
    "MaskedCompositeSample",
    "StructuredActionCodec",
    "collect_hierarchical",
    "compile_hierarchical_collector",
    "jax_hierarchical_context",
]
