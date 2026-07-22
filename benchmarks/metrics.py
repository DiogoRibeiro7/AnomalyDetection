"""Metric utilities for benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


DEFAULT_METRICS = ["roc_auc"]
SUPPORTED_METRICS = {
    "roc_auc",
    "average_precision",
    "precision_at_k",
    "recall_at_k",
    "f1_at_threshold",
    "best_f1",
    "runtime",
}


@dataclass(frozen=True)
class MetricConfig:
    """Normalized benchmark metric configuration."""

    names: list[str]
    positive_label: int | str = 1
    k: int | None = None
    threshold: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": self.names,
            "positive_label": self.positive_label,
            "k": self.k,
            "threshold": self.threshold,
        }


def resolve_metric_config(config: Any = None) -> MetricConfig:
    """Normalize CLI/YAML metric configuration."""

    if config is None:
        return MetricConfig(names=DEFAULT_METRICS.copy())
    if isinstance(config, str):
        return MetricConfig(names=_validate_metric_names([config]))
    if isinstance(config, (list, tuple)):
        return MetricConfig(names=_validate_metric_names(list(config)))
    if isinstance(config, dict):
        names = config.get("include") or config.get("names") or DEFAULT_METRICS
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, (list, tuple)):
            raise ValueError("metrics.include must be a string or list of strings.")
        k = config.get("k")
        if k is not None:
            k = int(k)
            if k <= 0:
                raise ValueError("metrics.k must be greater than zero.")
        threshold = float(config.get("threshold", 0.5))
        return MetricConfig(
            names=_validate_metric_names(list(names)),
            positive_label=config.get("positive_label", 1),
            k=k,
            threshold=threshold,
        )
    raise ValueError("metrics must be a string, list, mapping, or null.")


def evaluate_metrics(
    y_true: Any,
    scores: Any,
    *,
    runtime_seconds: float,
    config: MetricConfig | None = None,
) -> dict[str, float | None]:
    """Evaluate configured benchmark metrics."""

    metric_config = config or resolve_metric_config()
    labels = np.asarray(y_true)
    score_array = np.asarray(scores, dtype=float)
    positive = _positive_mask(labels, metric_config.positive_label)
    values: dict[str, float | None] = {}

    for name in metric_config.names:
        if name == "runtime":
            values[name] = float(runtime_seconds)
        elif name == "roc_auc":
            values[name] = _safe_metric(lambda: roc_auc_score(positive, score_array))
        elif name == "average_precision":
            values[name] = _safe_metric(
                lambda: average_precision_score(positive, score_array)
            )
        elif name == "precision_at_k":
            values[name] = precision_at_k(positive, score_array, metric_config.k)
        elif name == "recall_at_k":
            values[name] = recall_at_k(positive, score_array, metric_config.k)
        elif name == "f1_at_threshold":
            values[name] = _safe_metric(
                lambda: f1_score(
                    positive,
                    score_array >= metric_config.threshold,
                    zero_division=0,
                )
            )
        elif name == "best_f1":
            values[name] = best_f1_score(positive, score_array)
    return values


def precision_at_k(
    y_true_positive: Any,
    scores: Any,
    k: int | None = None,
) -> float | None:
    """Return precision among the top-k highest scores."""

    labels = np.asarray(y_true_positive, dtype=bool)
    score_array = np.asarray(scores, dtype=float)
    if labels.size == 0:
        return None
    effective_k = _effective_k(k, labels)
    selected = _top_k_mask(score_array, effective_k)
    return float(precision_score(labels, selected, zero_division=0))


def recall_at_k(
    y_true_positive: Any,
    scores: Any,
    k: int | None = None,
) -> float | None:
    """Return recall among the top-k highest scores."""

    labels = np.asarray(y_true_positive, dtype=bool)
    score_array = np.asarray(scores, dtype=float)
    if labels.size == 0:
        return None
    effective_k = _effective_k(k, labels)
    selected = _top_k_mask(score_array, effective_k)
    return float(recall_score(labels, selected, zero_division=0))


def best_f1_score(y_true_positive: Any, scores: Any) -> float | None:
    """Return the best F1 score across precision-recall thresholds."""

    labels = np.asarray(y_true_positive, dtype=bool)
    score_array = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    precision, recall, _thresholds = precision_recall_curve(labels, score_array)
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )
    return float(np.max(f1))


def _validate_metric_names(names: list[Any]) -> list[str]:
    normalized = [str(name) for name in names]
    unknown = sorted(set(normalized) - SUPPORTED_METRICS)
    if unknown:
        raise ValueError(f"Unsupported benchmark metric(s): {', '.join(unknown)}.")
    return normalized or DEFAULT_METRICS.copy()


def _positive_mask(labels: NDArray[Any], positive_label: int | str) -> NDArray[np.bool_]:
    return np.asarray(labels == positive_label, dtype=bool)


def _safe_metric(func) -> float | None:
    try:
        return float(func())
    except ValueError:
        return None


def _effective_k(k: int | None, labels: NDArray[np.bool_]) -> int:
    if k is not None:
        return min(k, labels.size)
    positives = int(np.sum(labels))
    return max(1, positives)


def _top_k_mask(scores: NDArray[np.floating[Any]], k: int) -> NDArray[np.bool_]:
    selected = np.zeros(scores.shape[0], dtype=bool)
    if scores.size == 0:
        return selected
    ranked = np.argsort(scores)[::-1][:k]
    selected[ranked] = True
    return selected

