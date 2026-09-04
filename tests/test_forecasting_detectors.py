from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dataexcept import DataValidationError

from anomalybench.analytics.detectors.forecasting import ARIMADetector, ProphetDetector


class _ArimaLikeModel:
    def predict(self, start: int, end: int) -> np.ndarray:
        length = end - start + 1
        return np.zeros(length, dtype=float)


class _ProphetLikeModel:
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"yhat": np.zeros(len(df), dtype=float)})


def test_arima_score_raises_when_series_count_mismatches_fit() -> None:
    detector = ARIMADetector()
    detector.models = [_ArimaLikeModel(), _ArimaLikeModel()]
    detector._mark_fitted()
    with pytest.raises(
        DataValidationError, match="received 1 series.*fitted on 2 series"
    ):
        detector.score(np.array([1.0, 2.0, 3.0], dtype=float))


def test_prophet_score_raises_when_series_count_mismatches_fit() -> None:
    detector = ProphetDetector()
    detector.models = [_ProphetLikeModel()]
    detector._mark_fitted()
    with pytest.raises(
        DataValidationError, match="received 2 series.*fitted on 1 series"
    ):
        detector.score(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))


def test_forecasting_scores_work_when_series_counts_match() -> None:
    arima = ARIMADetector()
    arima.models = [_ArimaLikeModel()]
    arima._mark_fitted()
    arima_scores = arima.score(np.array([1.0, 2.0, 3.0], dtype=float))
    assert arima_scores.shape == (1,)

    prophet = ProphetDetector()
    prophet.models = [_ProphetLikeModel(), _ProphetLikeModel()]
    prophet._mark_fitted()
    prophet_scores = prophet.score(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    assert prophet_scores.shape == (2,)
