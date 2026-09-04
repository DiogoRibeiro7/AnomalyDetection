"""Tests for the preprocessing pipeline utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomalybench.analytics.detectors.classical import IsolationForestDetector
from anomalybench.analytics.preprocessing import PreprocessingPipeline


def test_preprocessing_pipeline_handles_missing_values_and_categories() -> None:
    df = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 100.0],
            "category": ["a", "b", "a", None],
        }
    )
    pipeline = PreprocessingPipeline(clip_quantile=0.1)
    transformed = pipeline.fit_transform(df)

    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= 2
    assert not np.isnan(transformed).any()
    assert pipeline.feature_names_out_ is not None
    bounds = pipeline.numeric_clip_bounds_
    assert bounds is not None and "numeric" in bounds

    extreme = pd.DataFrame({"numeric": [5000], "category": ["a"]})
    clipped_value = bounds["numeric"][1]
    expected = pd.DataFrame({"numeric": [clipped_value], "category": ["a"]})
    transformed_extreme = pipeline.transform(extreme)
    transformed_expected = pipeline.transform(expected)
    np.testing.assert_allclose(transformed_extreme, transformed_expected)


def test_detector_detect_anomalies_with_preprocessing_pipeline() -> None:
    df = pd.DataFrame(
        {
            "numeric": [0.0, 1.0, np.nan, 3.0, 5.0],
            "category": ["x", "y", "x", "y", None],
        }
    )
    pipeline = PreprocessingPipeline()
    detector = IsolationForestDetector(preprocessing_pipeline=pipeline)
    scores = detector.detect_anomalies(df)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(df),)


def test_detector_accepts_pipeline_argument() -> None:
    df = pd.DataFrame(
        {
            "numeric": [0.0, 1.0, np.nan, 2.5],
            "category": ["x", "y", "x", "y"],
        }
    )
    pipeline = PreprocessingPipeline()
    detector = IsolationForestDetector()
    scores = detector.detect_anomalies(df, preprocessing_pipeline=pipeline)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(df),)
