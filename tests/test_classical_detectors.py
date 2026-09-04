"""Comprehensive behaviour tests for classical anomaly detectors."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from dataexcept import DataValidationError

from anomalybench.analytics.base import BaseDetector
from anomalybench.analytics.detectors.classical import (
    ABODDetector,
    COPODDetector,
    FeatureBaggingDetector,
    HBOSDetector,
    IsolationForestDetector,
    KNNDetector,
    LODADetector,
    OneClassSVMDetector,
    PCAReconstructionDetector,
)


def test_isolation_forest_detector_comprehensive() -> None:
    rng = np.random.default_rng(42)
    train_array = rng.normal(size=(48, 4))
    detector = IsolationForestDetector().fit(
        train_array, n_estimators=32, random_state=0
    )
    assert detector is not None

    new_df = pd.DataFrame(rng.normal(size=(12, 4)), columns=list("abcd"))
    scores = detector.score(new_df)
    assert scores.shape == (12,)
    assert np.isfinite(scores).all()

    single_detector = IsolationForestDetector().fit(
        train_array[:1], n_estimators=8, random_state=0
    )
    single_scores = single_detector.score(train_array[:1])
    assert single_scores.shape == (1,)

    with pytest.raises(ValueError):
        IsolationForestDetector().fit(np.empty((0, 4)))

    nan_training = train_array.copy()
    nan_training[0, 0] = np.nan
    nan_scores = (
        IsolationForestDetector()
        .fit(nan_training, n_estimators=16, random_state=0)
        .score(nan_training)
    )
    assert np.isfinite(nan_scores).all()

    with pytest.raises(TypeError):
        IsolationForestDetector().fit(train_array, invalid_param=True)


def test_hbos_detector_comprehensive() -> None:
    rng = np.random.default_rng(7)
    train_df = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("xyz"))
    detector = HBOSDetector().fit(train_df, k=5)
    scores = detector.score(rng.normal(size=(15, 3)))
    assert scores.shape == (15,)
    assert np.isfinite(scores).all()

    single_scores = HBOSDetector().fit(train_df.iloc[:1], k=3).score(train_df.iloc[:1])
    assert single_scores.shape == (1,)

    with pytest.raises(IndexError):
        HBOSDetector().fit(np.empty((0, 3)))

    nan_training = train_df.to_numpy().copy()
    nan_training[0, 0] = np.nan
    with pytest.raises(ValueError):
        HBOSDetector().fit(nan_training, k=4)

    with pytest.raises(IndexError):
        HBOSDetector().fit(train_df.to_numpy(), k=0)


def test_knn_detector_comprehensive() -> None:
    rng = np.random.default_rng(11)
    train_array = rng.normal(size=(45, 2))
    detector = KNNDetector().fit(train_array, k=4)
    new_df = pd.DataFrame(rng.normal(size=(9, 2)), columns=["x", "y"])
    scores = detector.score(new_df)
    assert scores.shape == (9,)
    assert np.isfinite(scores).all()

    single_scores = KNNDetector().fit(new_df.iloc[:1], k=0).score(new_df.iloc[:1])
    assert single_scores.shape == (1,)

    with pytest.raises(ValueError):
        KNNDetector().fit(np.empty((0, 2)))

    nan_training = train_array.copy()
    nan_training[0, 0] = np.nan
    with pytest.raises(ValueError):
        KNNDetector().fit(nan_training, k=4)

    with pytest.raises(ValueError):
        KNNDetector().fit(train_array, k=-1)


def test_one_class_svm_detector_comprehensive() -> None:
    rng = np.random.default_rng(19)
    train_df = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("uvw"))
    detector = OneClassSVMDetector().fit(train_df, kernel="rbf", gamma=0.1)
    new_array = rng.normal(size=(10, 3))
    scores = detector.score(new_array)
    assert scores.shape == (10,)
    assert np.isfinite(scores).all()

    single_scores = (
        OneClassSVMDetector().fit(train_df.iloc[:1]).score(train_df.iloc[:1])
    )
    assert single_scores.shape == (1,)

    with pytest.raises(ValueError):
        OneClassSVMDetector().fit(np.empty((0, 3)))

    nan_training = train_df.to_numpy().copy()
    nan_training[0, 1] = np.nan
    with pytest.raises(ValueError):
        OneClassSVMDetector().fit(nan_training)

    with pytest.raises(ValueError):
        OneClassSVMDetector().fit(train_df, kernel="unknown")


def test_pca_reconstruction_detector_comprehensive() -> None:
    rng = np.random.default_rng(23)
    train_array = rng.normal(size=(70, 5))
    detector = PCAReconstructionDetector().fit(train_array, n_components=3)
    new_df = pd.DataFrame(rng.normal(size=(14, 5)), columns=list("pqrst"))
    scores = detector.score(new_df)
    assert scores.shape == (14,)
    assert np.isfinite(scores).all()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        single_scores = (
            PCAReconstructionDetector()
            .fit(train_array[:1], n_components=1)
            .score(train_array[:1])
        )
    assert single_scores.shape == (1,)

    with pytest.raises(ValueError):
        PCAReconstructionDetector().fit(np.empty((0, 5)))

    nan_training = train_array.copy()
    nan_training[0, 0] = np.nan
    with pytest.raises(ValueError):
        PCAReconstructionDetector().fit(nan_training, n_components=2)

    with pytest.raises(ValueError):
        PCAReconstructionDetector().fit(
            train_array, n_components=train_array.shape[1] + 1
        )


@pytest.mark.parametrize(
    ("detector_cls", "fit_kwargs"),
    [
        (COPODDetector, {}),
        (FeatureBaggingDetector, {"n_estimators": 5, "random_state": 0}),
        (LODADetector, {"n_random_cuts": 16}),
        (ABODDetector, {"method": "fast"}),
    ],
)
def test_pyod_detector_harmonized_interface(
    detector_cls: type[BaseDetector], fit_kwargs: dict[str, Any]
) -> None:
    rng = np.random.default_rng(31)
    columns = list("abc")
    train_df = pd.DataFrame(rng.normal(size=(40, 3)), columns=columns)
    test_array = rng.normal(size=(12, 3))

    detector = detector_cls().fit(train_df, **fit_kwargs)
    scores = detector.score(test_array)
    assert scores.shape == (12,)
    assert np.isfinite(scores).all()

    # Ensure DataFrame scoring also works and raises on malformed inputs
    refit = detector_cls().fit(test_array, **fit_kwargs)
    scores_df = refit.score(train_df)
    assert scores_df.shape == (40,)
    assert np.isfinite(scores_df).all()

    with pytest.raises(DataValidationError):
        detector_cls().fit(np.zeros(5), **fit_kwargs)
