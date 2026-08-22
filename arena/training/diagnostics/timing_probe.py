"""On-device separability diagnostics for binary timing decisions.

The probe is deliberately diagnostic-only.  It fits a small logistic readout on
group-disjoint rows and measures whether a timing label is present in raw,
policy-visible evidence and in a frozen policy latent.  It never changes the
policy, optimizer, action mask, or checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


BINARY_TIMING_SEPARABILITY_PROBE_SCHEMA = "arena-binary-timing-separability-probe-v1"
RAW_TIMING_AUROC_GATE = 0.99
LATENT_TIMING_AUROC_GATE = 0.90
LATENT_TIMING_BALANCED_NLL_GATE = 0.45


class TimingNegativeKind(IntEnum):
    """Mutually exclusive reasons that waiting is the teacher action."""

    NONE = 0
    PENDING_OR_PREVIOUS_ACTION = 1
    ALIGNMENT_OR_RANGE = 2
    LIFECYCLE_OR_UNSUPPORTED = 3
    OTHER = 4


def _hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def _label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical label")
    return value


@dataclass(frozen=True, slots=True)
class BinaryTimingProbeConfig:
    """Static, content-addressed fit and split contract for one timing probe."""

    raw_feature_names: tuple[str, ...]
    label_contract: str
    event_legal_contract: str
    wait_legal_contract: str
    updates: int = 256
    learning_rate: float = 0.05
    l2_coefficient: float = 1.0e-4
    split_seed: int = 0
    heldout_modulus: int = 4
    schema: str = BINARY_TIMING_SEPARABILITY_PROBE_SCHEMA

    def __post_init__(self) -> None:
        names = tuple(self.raw_feature_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("raw timing feature names must be non-empty and unique")
        for value in names:
            _label(value, "raw timing feature")
        object.__setattr__(self, "raw_feature_names", names)
        for name in (
            "label_contract",
            "event_legal_contract",
            "wait_legal_contract",
        ):
            object.__setattr__(self, name, _label(getattr(self, name), name))
        for name, minimum in (
            ("updates", 1),
            ("split_seed", 0),
            ("heldout_modulus", 2),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name.replace('_', ' ')} must be an integer")
            if int(value) < minimum:
                raise ValueError(f"{name.replace('_', ' ')} is below {minimum}")
            object.__setattr__(self, name, int(value))
        for name, allow_zero in (
            ("learning_rate", False),
            ("l2_coefficient", True),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name.replace('_', ' ')} must be real")
            value = float(value)
            if (
                not math.isfinite(value)
                or value < 0.0
                or (not allow_zero and value == 0.0)
            ):
                raise ValueError(f"{name.replace('_', ' ')} is outside its domain")
            object.__setattr__(self, name, value)
        if self.schema != BINARY_TIMING_SEPARABILITY_PROBE_SCHEMA:
            raise ValueError("binary timing probe schema is not current")

    def manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "negative_kinds": tuple(
                {"name": value.name.lower(), "value": int(value)}
                for value in TimingNegativeKind
            ),
            "split_law": (
                "uint32_mix(group_id,split_seed)%heldout_modulus==0;"
                "all_rows_from_one_group_share_one_partition"
            ),
            "fit_law": (
                "train_standardized_linear_logistic_probe;"
                "class_mean_balanced_nll_plus_l2;adam"
            ),
            "support_law": (
                "valid_and_event_legal_and_wait_legal_and_decision_eligible;"
                "pending_or_in_flight_rows_remain_counted_but_are_not_probe_rows"
            ),
            "metric_law": (
                "tie_grouped_exact_auroc;threshold_step_auprc;"
                "class_mean_balanced_nll;label_signed_logit_margins"
            ),
            "gates": {
                "raw_heldout_auroc_strictly_greater_than": RAW_TIMING_AUROC_GATE,
                "latent_heldout_auroc_minimum": LATENT_TIMING_AUROC_GATE,
                "latent_heldout_balanced_nll_maximum": (
                    LATENT_TIMING_BALANCED_NLL_GATE
                ),
            },
        }

    @property
    def contract_sha256(self) -> str:
        return _hash(self.manifest())


class BinaryTimingProbeMetrics(NamedTuple):
    rows: jax.Array
    positives: jax.Array
    negatives: jax.Array
    auroc: jax.Array
    auprc: jax.Array
    balanced_nll: jax.Array
    positive_nll: jax.Array
    negative_nll: jax.Array
    positive_margin_p10: jax.Array
    positive_margin_p50: jax.Array
    positive_margin_p90: jax.Array
    negative_margin_p10: jax.Array
    negative_margin_p50: jax.Array
    negative_margin_p90: jax.Array


class BinaryTimingProbeFit(NamedTuple):
    coefficients: jax.Array
    bias: jax.Array
    train: BinaryTimingProbeMetrics
    heldout: BinaryTimingProbeMetrics


class BinaryTimingProbeCounts(NamedTuple):
    valid_rows: jax.Array
    probe_rows: jax.Array
    excluded_unsupported_rows: jax.Array
    excluded_decision_rows: jax.Array
    train_rows: jax.Array
    heldout_rows: jax.Array
    train_groups: jax.Array
    heldout_groups: jax.Array
    positive_rows: jax.Array
    negative_rows: jax.Array
    pending_rows: jax.Array
    pending_positive_conflicts: jax.Array
    negative_pending_or_previous_action: jax.Array
    negative_alignment_or_range: jax.Array
    negative_lifecycle_or_unsupported: jax.Array
    negative_other: jax.Array
    unknown_negative_kind: jax.Array


class BinaryTimingProbeGates(NamedTuple):
    split_valid: jax.Array
    raw_label_learnable: jax.Array
    latent_label_separable: jax.Array


class BinaryTimingSeparabilityResult(NamedTuple):
    raw: BinaryTimingProbeFit
    latent: BinaryTimingProbeFit
    frozen_policy_train: BinaryTimingProbeMetrics
    frozen_policy_heldout: BinaryTimingProbeMetrics
    counts: BinaryTimingProbeCounts
    gates: BinaryTimingProbeGates


def _mixed_group_fold(
    group_ids: jax.Array,
    split_seed: int,
    modulus: int,
) -> jax.Array:
    """Return a stable pseudo-random fold without splitting a group."""

    value = jnp.asarray(group_ids, dtype=jnp.uint32) ^ jnp.uint32(split_seed)
    value ^= value >> jnp.uint32(16)
    value *= jnp.uint32(0x7FEB352D)
    value ^= value >> jnp.uint32(15)
    value *= jnp.uint32(0x846CA68B)
    value ^= value >> jnp.uint32(16)
    return value % jnp.uint32(modulus)


def _class_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    count = jnp.sum(mask)
    mean = jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(count, 1)
    return jnp.where(count > 0, mean, jnp.float32(jnp.nan))


def _masked_quantile(
    values: jax.Array,
    mask: jax.Array,
    quantile: float,
) -> jax.Array:
    count = jnp.sum(mask)
    ordered = jnp.sort(jnp.where(mask, values, jnp.inf))
    index = jnp.floor(jnp.float32(quantile) * jnp.maximum(count - 1, 0)).astype(
        jnp.int32
    )
    selected = ordered[jnp.clip(index, 0, ordered.shape[0] - 1)]
    return jnp.where(count > 0, selected, jnp.float32(jnp.nan))


def _curve_areas(
    scores: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Exact threshold AUROC and step-interpolated PR area, including ties."""

    safe_scores = jnp.where(mask, scores, -jnp.inf)
    order = jnp.argsort(safe_scores, descending=True)
    ordered_scores = safe_scores[order]
    ordered_valid = mask[order]
    ordered_positive = labels[order] & ordered_valid
    ordered_negative = ~labels[order] & ordered_valid
    positive_count = jnp.sum(ordered_positive)
    negative_count = jnp.sum(ordered_negative)
    cumulative_positive = jnp.cumsum(ordered_positive.astype(jnp.float32))
    cumulative_negative = jnp.cumsum(ordered_negative.astype(jnp.float32))
    next_scores = jnp.concatenate((ordered_scores[1:], jnp.asarray((-jnp.inf,))))
    next_valid = jnp.concatenate((ordered_valid[1:], jnp.asarray((False,))))
    threshold_end = ordered_valid & (~next_valid | (next_scores < ordered_scores))

    def accumulate(carry, item):
        previous_fpr, previous_tpr, previous_recall, roc_area, pr_area = carry
        tp, fp, boundary = item
        tpr = tp / jnp.maximum(positive_count, 1)
        fpr = fp / jnp.maximum(negative_count, 1)
        precision = tp / jnp.maximum(tp + fp, 1.0)
        next_roc = roc_area + jnp.where(
            boundary,
            (fpr - previous_fpr) * (tpr + previous_tpr) * 0.5,
            0.0,
        )
        next_pr = pr_area + jnp.where(
            boundary,
            (tpr - previous_recall) * precision,
            0.0,
        )
        return (
            jnp.where(boundary, fpr, previous_fpr),
            jnp.where(boundary, tpr, previous_tpr),
            jnp.where(boundary, tpr, previous_recall),
            next_roc,
            next_pr,
        ), None

    zero = jnp.float32(0.0)
    final, _ = jax.lax.scan(
        accumulate,
        (zero, zero, zero, zero, zero),
        (cumulative_positive, cumulative_negative, threshold_end),
    )
    valid_curve = (positive_count > 0) & (negative_count > 0)
    nan = jnp.float32(jnp.nan)
    return (
        jnp.where(valid_curve, final[3], nan),
        jnp.where(valid_curve, final[4], nan),
    )


def binary_timing_probe_metrics(
    scores: jax.Array,
    labels: jax.Array,
    mask: jax.Array,
) -> BinaryTimingProbeMetrics:
    """Measure one fixed timing score without moving data off device."""

    score = jnp.asarray(scores, dtype=jnp.float32).reshape((-1,))
    label = jnp.asarray(labels, dtype=jnp.bool_).reshape((-1,))
    selected = jnp.asarray(mask, dtype=jnp.bool_).reshape((-1,))
    if score.shape != label.shape or score.shape != selected.shape:
        raise ValueError("timing scores, labels, and mask must share one shape")
    positive = selected & label
    negative = selected & ~label
    positive_nll = _class_mean(jax.nn.softplus(-score), positive)
    negative_nll = _class_mean(jax.nn.softplus(score), negative)
    balanced = 0.5 * (positive_nll + negative_nll)
    auroc, auprc = _curve_areas(score, label, selected)
    positive_margin = score
    negative_margin = -score
    return BinaryTimingProbeMetrics(
        jnp.sum(selected),
        jnp.sum(positive),
        jnp.sum(negative),
        auroc,
        auprc,
        balanced,
        positive_nll,
        negative_nll,
        _masked_quantile(positive_margin, positive, 0.10),
        _masked_quantile(positive_margin, positive, 0.50),
        _masked_quantile(positive_margin, positive, 0.90),
        _masked_quantile(negative_margin, negative, 0.10),
        _masked_quantile(negative_margin, negative, 0.50),
        _masked_quantile(negative_margin, negative, 0.90),
    )


def _fit_logistic_probe(
    features: jax.Array,
    labels: jax.Array,
    train_mask: jax.Array,
    heldout_mask: jax.Array,
    config: BinaryTimingProbeConfig,
) -> BinaryTimingProbeFit:
    values = jnp.asarray(features, dtype=jnp.float32)
    if values.ndim != 2:
        raise ValueError("flattened timing features must have shape [rows,features]")
    count = jnp.sum(train_mask)
    mean = jnp.sum(jnp.where(train_mask[:, None], values, 0.0), axis=0) / jnp.maximum(
        count, 1
    )
    centered = values - mean
    variance = jnp.sum(
        jnp.where(train_mask[:, None], jnp.square(centered), 0.0), axis=0
    ) / jnp.maximum(count, 1)
    scale = jnp.sqrt(jnp.maximum(variance, 1.0e-6))
    standardized = centered / scale
    positive = train_mask & labels
    negative = train_mask & ~labels

    def objective(parameters):
        weights, bias = parameters
        score = standardized @ weights + bias
        positive_nll = _class_mean(jax.nn.softplus(-score), positive)
        negative_nll = _class_mean(jax.nn.softplus(score), negative)
        return 0.5 * (positive_nll + negative_nll) + 0.5 * jnp.float32(
            config.l2_coefficient
        ) * jnp.vdot(weights, weights)

    weights = jnp.zeros((values.shape[-1],), dtype=jnp.float32)
    bias = jnp.float32(0.0)
    first = (jnp.zeros_like(weights), jnp.float32(0.0))
    second = (jnp.zeros_like(weights), jnp.float32(0.0))

    def update(carry, index):
        parameters, first_moment, second_moment = carry
        gradients = jax.grad(objective)(parameters)
        next_first = jax.tree_util.tree_map(
            lambda previous, value: 0.9 * previous + 0.1 * value,
            first_moment,
            gradients,
        )
        next_second = jax.tree_util.tree_map(
            lambda previous, value: 0.999 * previous + 0.001 * jnp.square(value),
            second_moment,
            gradients,
        )
        step = index.astype(jnp.float32) + 1.0
        first_scale = 1.0 - jnp.float32(0.9) ** step
        second_scale = 1.0 - jnp.float32(0.999) ** step
        next_parameters = jax.tree_util.tree_map(
            lambda parameter, first_value, second_value: parameter
            - jnp.float32(config.learning_rate)
            * (first_value / first_scale)
            / (jnp.sqrt(second_value / second_scale) + 1.0e-8),
            parameters,
            next_first,
            next_second,
        )
        return (next_parameters, next_first, next_second), None

    (parameters, _, _), _ = jax.lax.scan(
        update,
        ((weights, bias), first, second),
        jnp.arange(config.updates, dtype=jnp.int32),
    )
    fitted_weights, fitted_bias = parameters
    scores = standardized @ fitted_weights + fitted_bias
    return BinaryTimingProbeFit(
        fitted_weights / scale,
        fitted_bias - jnp.vdot(mean / scale, fitted_weights),
        binary_timing_probe_metrics(scores, labels, train_mask),
        binary_timing_probe_metrics(scores, labels, heldout_mask),
    )


def _masked_unique_count(values: jax.Array, mask: jax.Array) -> jax.Array:
    maximum = jnp.iinfo(jnp.int32).max
    safe = jnp.where(mask, values, maximum)
    ordered = jnp.sort(safe)
    valid = ordered != maximum
    previous = jnp.concatenate((jnp.asarray((maximum,), jnp.int32), ordered[:-1]))
    return jnp.sum(valid & (ordered != previous))


def fit_binary_timing_separability_probe(
    raw_features: jax.Array,
    latent_features: jax.Array,
    frozen_policy_score: jax.Array,
    labels: jax.Array,
    valid: jax.Array,
    event_legal: jax.Array,
    wait_legal: jax.Array,
    decision_eligible: jax.Array,
    group_ids: jax.Array,
    negative_kind: jax.Array,
    pending: jax.Array,
    config: BinaryTimingProbeConfig,
) -> BinaryTimingSeparabilityResult:
    """Fit raw and latent probes on whole-group-disjoint supported rows."""

    if not isinstance(config, BinaryTimingProbeConfig):
        raise TypeError("timing separability probe requires BinaryTimingProbeConfig")
    raw = jnp.asarray(raw_features, dtype=jnp.float32)
    latent = jnp.asarray(latent_features, dtype=jnp.float32)
    if raw.ndim < 2 or latent.ndim < 2 or raw.shape[:-1] != latent.shape[:-1]:
        raise ValueError("raw and latent timing features must share leading dimensions")
    if raw.shape[-1] != len(config.raw_feature_names):
        raise ValueError("raw timing width does not match its feature contract")
    leading = raw.shape[:-1]
    label = jnp.asarray(labels, dtype=jnp.bool_)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    event = jnp.asarray(event_legal, dtype=jnp.bool_)
    wait = jnp.asarray(wait_legal, dtype=jnp.bool_)
    decision = jnp.asarray(decision_eligible, dtype=jnp.bool_)
    groups = jnp.asarray(group_ids)
    kinds = jnp.asarray(negative_kind)
    pending_rows = jnp.asarray(pending, dtype=jnp.bool_)
    policy_score = jnp.asarray(frozen_policy_score, dtype=jnp.float32)
    for name, value in (
        ("labels", label),
        ("valid", evidence),
        ("event legal", event),
        ("wait legal", wait),
        ("decision eligible", decision),
        ("group ids", groups),
        ("negative kind", kinds),
        ("pending", pending_rows),
        ("frozen policy score", policy_score),
    ):
        if value.shape != leading:
            raise ValueError(f"{name} must match timing feature rows")
    if not jnp.issubdtype(groups.dtype, jnp.integer):
        raise TypeError("timing probe group ids must be integers")
    if not jnp.issubdtype(kinds.dtype, jnp.integer):
        raise TypeError("timing negative kinds must be integers")

    flat_raw = raw.reshape((-1, raw.shape[-1]))
    flat_latent = latent.reshape((-1, latent.shape[-1]))
    flat_label = label.reshape((-1,))
    flat_valid = evidence.reshape((-1,))
    flat_event = event.reshape((-1,))
    flat_wait = wait.reshape((-1,))
    flat_decision = decision.reshape((-1,))
    flat_groups = groups.astype(jnp.int32).reshape((-1,))
    flat_kinds = kinds.astype(jnp.int32).reshape((-1,))
    flat_pending = pending_rows.reshape((-1,))
    flat_policy = policy_score.reshape((-1,))
    supported = flat_valid & flat_event & flat_wait
    probe_mask = supported & flat_decision
    heldout_group = (
        _mixed_group_fold(
            flat_groups,
            config.split_seed,
            config.heldout_modulus,
        )
        == 0
    )
    train_mask = probe_mask & ~heldout_group
    heldout_mask = probe_mask & heldout_group
    raw_fit = _fit_logistic_probe(
        flat_raw, flat_label, train_mask, heldout_mask, config
    )
    latent_fit = _fit_logistic_probe(
        flat_latent, flat_label, train_mask, heldout_mask, config
    )
    policy_train = binary_timing_probe_metrics(flat_policy, flat_label, train_mask)
    policy_heldout = binary_timing_probe_metrics(flat_policy, flat_label, heldout_mask)
    negative = flat_valid & ~flat_label
    known_kind = (flat_kinds >= int(TimingNegativeKind.NONE)) & (
        flat_kinds <= int(TimingNegativeKind.OTHER)
    )
    counts = BinaryTimingProbeCounts(
        jnp.sum(flat_valid),
        jnp.sum(probe_mask),
        jnp.sum(flat_valid & ~(flat_event & flat_wait)),
        jnp.sum(supported & ~flat_decision),
        jnp.sum(train_mask),
        jnp.sum(heldout_mask),
        _masked_unique_count(flat_groups, train_mask),
        _masked_unique_count(flat_groups, heldout_mask),
        jnp.sum(flat_valid & flat_label),
        jnp.sum(negative),
        jnp.sum(flat_valid & flat_pending),
        jnp.sum(flat_valid & flat_pending & flat_label),
        jnp.sum(
            negative
            & (flat_kinds == int(TimingNegativeKind.PENDING_OR_PREVIOUS_ACTION))
        ),
        jnp.sum(negative & (flat_kinds == int(TimingNegativeKind.ALIGNMENT_OR_RANGE))),
        jnp.sum(
            negative & (flat_kinds == int(TimingNegativeKind.LIFECYCLE_OR_UNSUPPORTED))
        ),
        jnp.sum(negative & (flat_kinds == int(TimingNegativeKind.OTHER))),
        jnp.sum(negative & ~known_kind),
    )
    split_valid = (
        (raw_fit.train.positives > 0)
        & (raw_fit.train.negatives > 0)
        & (raw_fit.heldout.positives > 0)
        & (raw_fit.heldout.negatives > 0)
        & (counts.train_groups > 0)
        & (counts.heldout_groups > 0)
        & (counts.unknown_negative_kind == 0)
    )
    gates = BinaryTimingProbeGates(
        split_valid,
        split_valid & (raw_fit.heldout.auroc > RAW_TIMING_AUROC_GATE),
        split_valid
        & (latent_fit.heldout.auroc >= LATENT_TIMING_AUROC_GATE)
        & (latent_fit.heldout.balanced_nll <= LATENT_TIMING_BALANCED_NLL_GATE),
    )
    return BinaryTimingSeparabilityResult(
        raw_fit,
        latent_fit,
        policy_train,
        policy_heldout,
        counts,
        gates,
    )


def binary_timing_probe_report(
    result: BinaryTimingSeparabilityResult,
    config: BinaryTimingProbeConfig,
) -> dict[str, Any]:
    """Materialize one completed device probe as a JSON-safe report."""

    if not isinstance(result, BinaryTimingSeparabilityResult):
        raise TypeError("timing probe report requires a separability result")
    if not isinstance(config, BinaryTimingProbeConfig):
        raise TypeError("timing probe report requires BinaryTimingProbeConfig")
    host = jax.device_get(result)

    def metrics(value: BinaryTimingProbeMetrics) -> dict[str, int | float]:
        return {
            name: int(item)
            if name in {"rows", "positives", "negatives"}
            else float(item)
            for name, item in zip(BinaryTimingProbeMetrics._fields, value, strict=True)
        }

    def fit(value: BinaryTimingProbeFit) -> dict[str, Any]:
        return {
            "coefficients": np.asarray(value.coefficients).tolist(),
            "bias": float(value.bias),
            "train": metrics(value.train),
            "heldout": metrics(value.heldout),
        }

    return {
        "schema": config.schema,
        "probe_contract_sha256": config.contract_sha256,
        "config": config.manifest(),
        "raw": fit(host.raw),
        "latent": fit(host.latent),
        "frozen_policy": {
            "train": metrics(host.frozen_policy_train),
            "heldout": metrics(host.frozen_policy_heldout),
        },
        "counts": {
            name: int(item)
            for name, item in zip(
                BinaryTimingProbeCounts._fields, host.counts, strict=True
            )
        },
        "gates": {
            name: bool(item)
            for name, item in zip(
                BinaryTimingProbeGates._fields, host.gates, strict=True
            )
        },
    }


__all__ = [
    "BINARY_TIMING_SEPARABILITY_PROBE_SCHEMA",
    "LATENT_TIMING_AUROC_GATE",
    "LATENT_TIMING_BALANCED_NLL_GATE",
    "RAW_TIMING_AUROC_GATE",
    "BinaryTimingProbeConfig",
    "BinaryTimingProbeCounts",
    "BinaryTimingProbeFit",
    "BinaryTimingProbeGates",
    "BinaryTimingProbeMetrics",
    "BinaryTimingSeparabilityResult",
    "TimingNegativeKind",
    "binary_timing_probe_metrics",
    "binary_timing_probe_report",
    "fit_binary_timing_separability_probe",
]
