"""Lifecycle tests for the classical detectors not covered elsewhere.

``test_classical_detectors`` covers the Isolation Forest, PyOD adapter, and
distance-based detectors in depth. The registered detectors exercised here had
no direct tests, so their fit/score contract, declared score orientation, and
input handling are pinned.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from analytics.detectors import get_detector_class
from analytics.detectors.classical import EnsembleDetector, LOFDetector
from analytics.exceptions import DetectorNotFittedError

COLUMNS = list("abc")

# Parameters chosen so each detector trains on the small fixture below.
UNTESTED_DETECTORS: list[tuple[str, dict[str, Any]]] = [
    ("sos", {}),
    ("dbscan", {"eps": 1.5, "min_samples": 3}),
    ("elliptic_envelope", {"support_fraction": 0.9, "random_state": 0}),
    ("gaussian_mixture", {"n_components": 2, "random_state": 0}),
    ("sklearn_lof", {"n_neighbors": 5}),
    ("kmeans", {"n_clusters": 2, "random_state": 0, "n_init": 10}),
    ("mahalanobis", {}),
    ("kde", {"bandwidth": 0.5}),
]
DETECTOR_IDS = [key for key, _ in UNTESTED_DETECTORS]


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(11)
    train = pd.DataFrame(rng.normal(size=(60, 3)), columns=COLUMNS)
    test = pd.DataFrame(rng.normal(size=(15, 3)), columns=COLUMNS)
    return train, test


@pytest.mark.parametrize(
    ("detector_key", "fit_kwargs"), UNTESTED_DETECTORS, ids=DETECTOR_IDS
)
def test_fit_returns_self_and_marks_fitted(
    detector_key: str,
    fit_kwargs: dict[str, Any],
    frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = frames
    detector = get_detector_class(detector_key)()

    assert detector.fit(train, **fit_kwargs) is detector
    assert detector.is_fitted


@pytest.mark.parametrize(
    ("detector_key", "fit_kwargs"), UNTESTED_DETECTORS, ids=DETECTOR_IDS
)
def test_score_returns_one_finite_value_per_row(
    detector_key: str,
    fit_kwargs: dict[str, Any],
    frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = frames
    detector = get_detector_class(detector_key)().fit(train, **fit_kwargs)

    scores = np.asarray(detector.score(test))

    assert scores.shape == (len(test),)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    ("detector_key", "fit_kwargs"), UNTESTED_DETECTORS, ids=DETECTOR_IDS
)
def test_score_before_fit_is_refused(
    detector_key: str,
    fit_kwargs: dict[str, Any],
    frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, test = frames
    detector = get_detector_class(detector_key)()

    with pytest.raises(DetectorNotFittedError):
        detector.score(test)


@pytest.mark.parametrize(
    ("detector_key", "fit_kwargs"), UNTESTED_DETECTORS, ids=DETECTOR_IDS
)
def test_detect_anomalies_carries_the_declared_orientation(
    detector_key: str,
    fit_kwargs: dict[str, Any],
    frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = frames
    detector_cls = get_detector_class(detector_key)

    scores = detector_cls().detect_anomalies(train, **fit_kwargs)

    assert scores.score_orientation == detector_cls.score_orientation


@pytest.mark.parametrize(
    ("detector_key", "fit_kwargs"), UNTESTED_DETECTORS, ids=DETECTOR_IDS
)
def test_numpy_input_matches_dataframe_input(
    detector_key: str,
    fit_kwargs: dict[str, Any],
    frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Column labels must not change the scores."""

    train, test = frames
    from_frame = get_detector_class(detector_key)().fit(train, **fit_kwargs)
    from_array = get_detector_class(detector_key)().fit(train.to_numpy(), **fit_kwargs)

    np.testing.assert_allclose(
        np.asarray(from_frame.score(test)),
        np.asarray(from_array.score(test.to_numpy())),
    )


def test_dbscan_reports_binary_labels(
    frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Its declared orientation is binary_anomaly, so scores must be 0 or 1."""

    train, test = frames
    detector = get_detector_class("dbscan")().fit(train, eps=1.5, min_samples=3)

    scores = np.asarray(detector.score(test))

    assert set(np.unique(scores)) <= {0.0, 1.0}


def test_lof_detector_scores_every_row(
    frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = frames
    detector = LOFDetector().fit(train, min_pts=4)

    scores = detector.score(test)

    assert len(scores) == len(test)
    assert all(np.isfinite(score) for score in scores)


def test_lof_detector_supports_internal_normalization(
    frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = frames
    detector = LOFDetector().fit(train, normalize=True, min_pts=4)

    assert detector.lof.normalize is True
    assert len(detector.score(test)) == len(test)


def test_ensemble_detector_fits_and_averages_its_components(
    frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = frames
    detector = EnsembleDetector().fit(train)

    assert detector.knn.is_fitted
    assert detector.sos.is_fitted
    assert detector.hbos.is_fitted

    scores = np.asarray(detector.score(test))
    assert scores.shape == (len(test),)
    assert np.isfinite(scores).all()
