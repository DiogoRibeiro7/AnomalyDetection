"""Regression tests for point-stream window alignment and reproducibility."""

from __future__ import annotations

import numpy as np
from dataexcept import DataValidationError

from anomalybench.analytics.base import OrientedScores
from anomalybench.analytics.time_series import WindowedScores, WindowSpec
from anomalybench.benchmarks.metrics import MetricConfig, evaluate_metrics
from anomalybench.benchmarks.reproducibility import build_manifest


def test_oriented_scores_preserve_window_alignment_metadata() -> None:
    spec = WindowSpec(window_length=3, stride=2, horizon=1)
    raw = WindowedScores(
        [-0.2, -0.9, -0.4],
        label_indices=[3, 5, 7],
        window_spec=spec,
    )

    oriented = OrientedScores(raw, "lower_is_more_anomalous")

    np.testing.assert_array_equal(oriented.label_indices, np.array([3, 5, 7]))
    assert oriented.window_spec == spec.as_dict()


def test_metrics_align_point_labels_before_evaluation() -> None:
    labels = np.array([0, 0, 0, 0, 0, 1, 0, 1])
    spec = WindowSpec(window_length=3, stride=2, horizon=1)
    raw = WindowedScores(
        [-0.1, -0.8, -0.9],
        label_indices=[3, 5, 7],
        window_spec=spec,
    )
    scores = OrientedScores(raw, "lower_is_more_anomalous")

    metrics = evaluate_metrics(
        labels,
        scores,
        runtime_seconds=0.01,
        config=MetricConfig(names=["roc_auc", "average_precision"]),
    )

    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0


def test_metrics_reject_unaligned_length_mismatch() -> None:
    labels = np.array([0, 1, 0, 1])
    scores = OrientedScores([0.2, 0.8], "higher_is_more_anomalous")

    try:
        evaluate_metrics(labels, scores, runtime_seconds=0.0)
    except DataValidationError as exc:
        assert "equal length after alignment" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected an explicit label/score length failure")


def test_manifest_expands_effective_window_defaults() -> None:
    manifest = build_manifest(
        run_id="window-run",
        timestamp="2026-08-31T17:00:00+00:00",
        config_hash="abc123",
        dataset_keys=["nab_machine_temperature"],
        detector_entries=[
            {
                "name": "lstm_autoencoder",
                "label": "lstm_autoencoder",
                "params": {"window_length": 8},
            }
        ],
        random_seed=None,
        n_jobs=1,
        output_directory=None,
        dataset_integrity=[],
    )

    assert manifest["detectors"][0]["windowing"] == {
        "window_length": 8,
        "stride": 1,
        "horizon": 0,
        "aggregation": "window",
        "label_alignment": "window_end",
    }
