"""Forecasting based anomaly detectors for time-series data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.base import BaseDetector


class ARIMADetector(BaseDetector):
    """Detect anomalies using ARIMA forecast residuals."""

    def get_name(self) -> str:
        return "ARIMA"

    def fit(self, data, order=(5, 1, 0), **params):
        from statsmodels.tsa.arima.model import ARIMA

        X = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.models = [ARIMA(series, order=order).fit() for series in X]
        return self

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        scores = []
        for series, model in zip(X, self.models):
            pred = model.predict(start=0, end=len(series) - 1)
            resid = np.abs(series - pred)
            scores.append(resid.mean())
        return np.array(scores)


class ProphetDetector(BaseDetector):
    """Use Prophet forecasting to score time-series anomalies."""

    def get_name(self) -> str:
        return "Prophet"

    def fit(self, data, **params):
        from prophet import Prophet

        X = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)
        if X.ndim == 1:
            X = X.reshape(1, -1)
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

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)
        if X.ndim == 1:
            X = X.reshape(1, -1)
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
        return np.array(scores)


__all__ = [name for name in globals() if name.endswith("Detector")]
