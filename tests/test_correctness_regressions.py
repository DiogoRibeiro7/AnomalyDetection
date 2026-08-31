"""Regression tests for benchmark and detector correctness contracts."""

from __future__ import annotations

import numpy as np

from analytics.detectors import get_detector_class
from benchmarks.catalog import get_dataset_functions


def test_cardio_keeps_normal_and_pathological_cases_only() -> None:
    """Cardio must benchmark normal versus pathological, not suspect cases."""

    loader = get_dataset_functions()["cardio"]
    df, feature_cols, label_col, display_name = loader()

    assert display_name == "cardio"
    assert label_col == "Class"
    assert feature_cols
    assert set(df[label_col].unique()) == {0, 1}
    assert (df[label_col] == 0).any(), "Pathological anomaly class must be present."
    assert (df[label_col] == 1).any(), "Normal inlier class must be present."
    assert "NSP" not in df.columns


def test_dbscan_score_does_not_refit_on_evaluation_batch() -> None:
    """DBSCAN scoring must depend on fitted core points, not test-batch density."""

    detector_cls = get_detector_class("dbscan")
    train = np.array(
        [
            [0.00, 0.00],
            [0.02, 0.01],
            [-0.02, -0.01],
            [0.01, -0.02],
        ],
        dtype=float,
    )
    detector = detector_cls().fit(train, eps=0.08, min_samples=2)

    isolated = np.array([[5.0, 5.0]], dtype=float)
    crowded_far_batch = np.array(
        [
            [5.00, 5.00],
            [5.01, 5.00],
            [5.00, 5.01],
            [5.01, 5.01],
        ],
        dtype=float,
    )

    isolated_score = detector.score(isolated)
    crowded_scores = detector.score(crowded_far_batch)

    assert isolated_score.tolist() == [1.0]
    assert crowded_scores.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_dbscan_recognizes_points_near_fitted_core_samples() -> None:
    detector_cls = get_detector_class("dbscan")
    train = np.array(
        [
            [0.00, 0.00],
            [0.02, 0.01],
            [-0.02, -0.01],
            [0.01, -0.02],
        ],
        dtype=float,
    )
    detector = detector_cls().fit(train, eps=0.08, min_samples=2)

    scores = detector.score(np.array([[0.01, 0.01], [2.0, 2.0]], dtype=float))

    assert scores.tolist() == [0.0, 1.0]
