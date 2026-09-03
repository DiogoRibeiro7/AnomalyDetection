"""Tests for benchmark metric utilities."""

from __future__ import annotations

import pytest
from dataexcept import ConfigurationError, DataValidationError

from analytics.base import OrientedScores
from benchmarks.metrics import (
    canonicalize_anomaly_scores,
    evaluate_metrics,
    precision_at_k,
    recall_at_k,
    resolve_metric_config,
)


def test_evaluate_metrics_returns_configured_values() -> None:
    config = resolve_metric_config(
        {
            "include": [
                "roc_auc",
                "average_precision",
                "precision_at_k",
                "recall_at_k",
                "f1_at_threshold",
                "best_f1",
                "runtime",
            ],
            "positive_label": 1,
            "k": 2,
            "threshold": 0.5,
        }
    )

    values = evaluate_metrics(
        [0, 1, 0, 1],
        [0.1, 0.9, 0.2, 0.8],
        runtime_seconds=0.25,
        config=config,
    )

    assert values["roc_auc"] == 1.0
    assert values["average_precision"] == 1.0
    assert values["precision_at_k"] == 1.0
    assert values["recall_at_k"] == 1.0
    assert values["f1_at_threshold"] == 1.0
    assert values["best_f1"] == 1.0
    assert values["runtime"] == 0.25


def test_lower_is_more_anomalous_scores_are_reoriented_before_metrics() -> None:
    labels = [0, 1, 0, 1]
    raw_scores = OrientedScores(
        [0.9, 0.1, 0.8, 0.2],
        "lower_is_more_anomalous",
    )
    config = resolve_metric_config({"include": ["roc_auc"], "positive_label": 1})

    values = evaluate_metrics(labels, raw_scores, runtime_seconds=0.0, config=config)

    assert values["roc_auc"] == 1.0


def test_binary_anomaly_scores_are_preserved() -> None:
    raw_scores = OrientedScores([0.0, 1.0], "binary_anomaly")
    canonical = canonicalize_anomaly_scores(raw_scores)
    assert canonical.tolist() == [0.0, 1.0]


def test_estimator_defined_orientation_is_rejected() -> None:
    raw_scores = OrientedScores([0.1, 0.9], "estimator_defined")
    with pytest.raises(DataValidationError, match="explicit score orientation"):
        canonicalize_anomaly_scores(raw_scores)


def test_top_k_metrics_default_k_to_number_of_positives() -> None:
    labels = [False, True, False, True]
    scores = [0.4, 0.3, 0.2, 0.9]

    assert precision_at_k(labels, scores) == 0.5
    assert recall_at_k(labels, scores) == 0.5


def test_invalid_metric_name_fails_fast() -> None:
    with pytest.raises(ConfigurationError, match="unsupported benchmark metric"):
        resolve_metric_config(["not_a_metric"])
