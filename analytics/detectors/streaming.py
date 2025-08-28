"""Streaming and online anomaly detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.base import BaseDetector


class OnlineIsolationForestDetector(BaseDetector):
    """Streaming Isolation Forest using River's incremental implementation."""

    def get_name(self) -> str:
        return "Online Isolation Forest"

    def fit(self, data, **params):
        from river import anomaly  # lazy import

        self.model = anomaly.IsolationForest(**params)
        for row in self._to_dicts(data):
            self.model.learn_one(row)
        return self

    def score(self, data):
        return [self.model.score_one(row) for row in self._to_dicts(data)]

    def _to_dicts(self, data):
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient="records")
        arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        return [dict(enumerate(row)) for row in arr]


class RandomCutForestDetector(BaseDetector):
    """Random Cut Forest for streaming anomaly detection."""

    def get_name(self) -> str:
        return "Random Cut Forest"

    def fit(self, data, **params):
        from river import anomaly  # lazy import

        self.model = anomaly.RandomCutForest(**params)
        for row in self._to_dicts(data):
            self.model.learn_one(row)
        return self

    def score(self, data):
        return [self.model.score_one(row) for row in self._to_dicts(data)]

    def _to_dicts(self, data):
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient="records")
        arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        return [dict(enumerate(row)) for row in arr]


__all__ = [name for name in globals() if name.endswith("Detector")]
