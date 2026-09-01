"""Edge cases for benchmark metric configuration and evaluation.

``test_benchmark_metrics`` covers the metric values themselves. The
configuration grammar, orientation handling, label alignment, and the
degenerate inputs that make a metric undefined are pinned here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from analytics.base import OrientedScores
from analytics.time_series import WindowedScores, WindowSpec
from benchmarks.metrics import (
    DEFAULT_METRICS,
    best_f1_score,
    canonicalize_anomaly_scores,
    evaluate_metrics,
    precision_at_k,
    recall_at_k,
    resolve_metric_config,
)


def test_resolve_defaults_when_no_configuration_is_given() -> None:
    assert resolve_metric_config().names == DEFAULT_METRICS


def test_resolve_accepts_a_bare_string() -> None:
    assert resolve_metric_config("roc_auc").names == ["roc_auc"]


def test_resolve_accepts_a_sequence() -> None:
    assert resolve_metric_config(["roc_auc", "runtime"]).names == [
        "roc_auc",
        "runtime",
    ]


def test_resolve_accepts_a_single_name_inside_a_mapping() -> None:
    assert resolve_metric_config({"names": "roc_auc"}).names == ["roc_auc"]


def test_resolve_reads_k_positive_label_and_threshold() -> None:
    config = resolve_metric_config(
        {
            "include": ["precision_at_k"],
            "k": 3,
            "positive_label": "anomaly",
            "threshold": 0.25,
        }
    )

    assert config.k == 3
    assert config.positive_label == "anomaly"
    assert config.threshold == pytest.approx(0.25)


def test_resolve_rejects_a_non_sequence_name_list() -> None:
    with pytest.raises(ValueError, match="metrics.include"):
        resolve_metric_config({"include": 5})


def test_resolve_rejects_a_non_positive_k() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        resolve_metric_config({"include": ["precision_at_k"], "k": 0})


def test_resolve_rejects_an_unsupported_metric_name() -> None:
    with pytest.raises(ValueError, match="Unsupported benchmark metric"):
        resolve_metric_config(["not_a_metric"])


def test_resolve_rejects_an_unusable_configuration_type() -> None:
    with pytest.raises(ValueError, match="string, list, mapping, or null"):
        resolve_metric_config(5)


def test_canonicalize_rejects_an_unknown_orientation() -> None:
    scores = OrientedScores([0.1, 0.2], "sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown score orientation"):
        canonicalize_anomaly_scores(scores)


def test_canonicalize_flips_lower_is_more_anomalous() -> None:
    scores = OrientedScores([1.0, -1.0], "lower_is_more_anomalous")
    np.testing.assert_allclose(canonicalize_anomaly_scores(scores), [-1.0, 1.0])


def test_window_aligned_labels_require_a_one_dimensional_label_vector() -> None:
    spec = WindowSpec(window_length=2)
    scores = WindowedScores([0.1, 0.2], label_indices=[0, 1], window_spec=spec)

    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate_metrics(np.zeros((2, 2)), scores, runtime_seconds=0.0)


def test_window_label_indices_must_fall_inside_the_label_vector() -> None:
    spec = WindowSpec(window_length=2)
    scores = WindowedScores([0.1, 0.2], label_indices=[0, 9], window_spec=spec)

    with pytest.raises(ValueError, match="outside the label vector"):
        evaluate_metrics([0, 1, 0], scores, runtime_seconds=0.0)


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k])
def test_top_k_metrics_are_undefined_without_labels(metric: Any) -> None:
    assert metric([], []) is None


def test_best_f1_is_undefined_when_only_one_class_is_present() -> None:
    assert best_f1_score([1, 1, 1], [0.1, 0.2, 0.3]) is None


def test_best_f1_is_undefined_without_labels() -> None:
    assert best_f1_score([], []) is None


def test_top_k_metrics_default_k_to_the_positive_count() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]

    assert precision_at_k(labels, scores) == pytest.approx(1.0)
    assert recall_at_k(labels, scores) == pytest.approx(1.0)


def test_explicit_k_is_capped_at_the_number_of_labels() -> None:
    labels = [1, 0]
    scores = [0.9, 0.1]

    assert recall_at_k(labels, scores, k=99) == pytest.approx(1.0)


def test_undefined_metrics_are_reported_as_none_rather_than_raising() -> None:
    """A single-class label vector makes ROC AUC undefined."""

    values = evaluate_metrics(
        [1, 1, 1],
        OrientedScores([0.1, 0.2, 0.3], "higher_is_more_anomalous"),
        runtime_seconds=1.5,
        config=resolve_metric_config(["roc_auc", "runtime"]),
    )

    assert values["roc_auc"] is None
    assert values["runtime"] == pytest.approx(1.5)


def test_undefined_metrics_serialize_as_valid_json() -> None:
    """NaN would serialize to the bare literal NaN, which strict parsers reject."""

    import json

    values = evaluate_metrics(
        [1, 1, 1],
        OrientedScores([0.1, 0.2, 0.3], "higher_is_more_anomalous"),
        runtime_seconds=1.0,
        config=resolve_metric_config(["roc_auc", "best_f1", "runtime"]),
    )
    encoded = json.dumps(values)

    def _reject(constant: str) -> float:
        raise AssertionError(f"non-standard JSON constant: {constant}")

    assert json.loads(encoded, parse_constant=_reject)["roc_auc"] is None
