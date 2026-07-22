"""Streaming and online anomaly detectors."""

from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from analytics.base import BaseDetector


ArrayLike = Union[pd.DataFrame, Sequence[Sequence[float]], NDArray[np.floating[Any]]]
RowDict = dict[str | int, float]


class OnlineIsolationForestDetector(BaseDetector):
    """Streaming Isolation Forest using River's incremental implementation."""

    def get_name(self) -> str:
        return "Online Isolation Forest"

    def fit(self, data: ArrayLike, **params: Any) -> OnlineIsolationForestDetector:
        from river import anomaly  # lazy import

        self.model = anomaly.IsolationForest(**params)
        for row in self._to_dicts(data):
            self.model.learn_one(row)
        return self

    def score(self, data: ArrayLike) -> list[float]:
        return [float(self.model.score_one(row)) for row in self._to_dicts(data)]

    def _to_dicts(self, data: ArrayLike) -> list[RowDict]:
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
            return [
                {key: float(value) for key, value in record.items()}
                for record in records
            ]
        arr: NDArray[np.floating[Any]]
        if isinstance(data, np.ndarray):
            arr = data.astype(float, copy=False)
        else:
            arr = np.asarray(data, dtype=float)
        return [
            {int(idx): float(value) for idx, value in enumerate(row)} for row in arr
        ]


class RandomCutForestDetector(BaseDetector):
    """Random Cut Forest for streaming anomaly detection."""

    def get_name(self) -> str:
        return "Random Cut Forest"

    def fit(self, data: ArrayLike, **params: Any) -> RandomCutForestDetector:
        from river import anomaly  # lazy import

        self.model = anomaly.RandomCutForest(**params)
        for row in self._to_dicts(data):
            self.model.learn_one(row)
        return self

    def score(self, data: ArrayLike) -> list[float]:
        return [float(self.model.score_one(row)) for row in self._to_dicts(data)]

    def _to_dicts(self, data: ArrayLike) -> list[RowDict]:
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
            return [
                {key: float(value) for key, value in record.items()}
                for record in records
            ]
        arr: NDArray[np.floating[Any]]
        if isinstance(data, np.ndarray):
            arr = data.astype(float, copy=False)
        else:
            arr = np.asarray(data, dtype=float)
        return [
            {int(idx): float(value) for idx, value in enumerate(row)} for row in arr
        ]


__all__ = [
    "OnlineIsolationForestDetector",
    "RandomCutForestDetector",
]
