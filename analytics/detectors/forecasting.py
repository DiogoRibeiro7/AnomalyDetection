"""Forecasting based anomaly detectors for time-series data."""

from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from analytics.base import BaseDetector


ArrayLike = Union[
    pd.DataFrame,
    Sequence[float],
    Sequence[Sequence[float]],
    NDArray[np.floating[Any]],
]
ScoreArray = NDArray[np.floating[Any]]


def _ensure_series_matrix(data: ArrayLike) -> NDArray[np.floating[Any]]:
    """Convert supported sequence inputs to a 2D NumPy array."""

    if isinstance(data, pd.DataFrame):
        arr = data.to_numpy(dtype=float)
    else:
        arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _validate_series_count(
    models: Sequence[Any],
    series_matrix: NDArray[np.floating[Any]],
    detector_name: str,
) -> None:
    expected = len(models)
    actual = int(series_matrix.shape[0])
    if expected != actual:
        raise ValueError(
            f"{detector_name} received {actual} series for scoring, "
            f"but was fitted on {expected} series."
        )


class ARIMADetector(BaseDetector):
    """Detect anomalies using ARIMA forecast residuals."""

    score_orientation = "higher_is_more_anomalous"

    def get_name(self) -> str:
        return "ARIMA"

    def fit(
        self,
        data: ArrayLike,
        order: tuple[int, int, int] = (5, 1, 0),
        **params: Any,
    ) -> ARIMADetector:
        from statsmodels.tsa.arima.model import ARIMA

        X = _ensure_series_matrix(data)
        self.models = [ARIMA(series, order=order, **params).fit() for series in X]
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        X = _ensure_series_matrix(data)
        _validate_series_count(self.models, X, self.get_name())
        scores = []
        for series, model in zip(X, self.models):
            pred = model.predict(start=0, end=len(series) - 1)
            resid = np.abs(series - pred)
            scores.append(resid.mean())
        return np.asarray(scores, dtype=float)


class ProphetDetector(BaseDetector):
    """Use Prophet forecasting to score time-series anomalies."""

    score_orientation = "higher_is_more_anomalous"

    def get_name(self) -> str:
        return "Prophet"

    def fit(self, data: ArrayLike, **params: Any) -> ProphetDetector:
        from prophet import Prophet

        X = _ensure_series_matrix(data)
        self.models = []
        for series in X:
            df = pd.DataFrame(
                {
                    "ds": pd.date_range(start="2000", periods=len(series), freq="D"),
                    "y": series,
                }
            )
            model = Prophet(**params).fit(df)
            self.models.append(model)
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        X = _ensure_series_matrix(data)
        _validate_series_count(self.models, X, self.get_name())
        scores = []
        for series, model in zip(X, self.models):
            df = pd.DataFrame(
                {
                    "ds": pd.date_range(start="2000", periods=len(series), freq="D"),
                    "y": series,
                }
            )
            forecast = model.predict(df)
            resid = np.abs(df["y"].values - forecast["yhat"].values)
            scores.append(resid.mean())
        return np.asarray(scores, dtype=float)


__all__ = [
    "ARIMADetector",
    "ProphetDetector",
]
