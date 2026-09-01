"""Streaming and online anomaly detectors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from analytics.base import BaseDetector

type ArrayLike = pd.DataFrame | Sequence[Sequence[float]] | NDArray[np.floating[Any]]
RowDict = dict[str | int, float]


def _to_dicts(data: ArrayLike) -> list[RowDict]:
    if isinstance(data, pd.DataFrame):
        records = data.to_dict(orient="records")
        return [
            {str(key): float(value) for key, value in record.items()}
            for record in records
        ]

    if isinstance(data, np.ndarray):
        arr = data.astype(float, copy=False)
    else:
        arr = np.asarray(data, dtype=float)
    arr = np.atleast_2d(arr)
    return [{int(idx): float(value) for idx, value in enumerate(row)} for row in arr]


class HalfSpaceTreesDetector(BaseDetector):
    """Online anomaly detector backed by River's Half-Space Trees."""

    score_orientation = "higher_is_more_anomalous"

    def get_name(self) -> str:
        return "Half-Space Trees"

    def fit(self, data: ArrayLike, **params: Any) -> HalfSpaceTreesDetector:
        from river import anomaly  # lazy import

        self.model = anomaly.HalfSpaceTrees(**params)
        for row in _to_dicts(data):
            self.model.learn_one(row)
        return self

    def score(self, data: ArrayLike) -> list[float]:
        return [float(self.model.score_one(row)) for row in _to_dicts(data)]


class OnlineIsolationForestDetector(HalfSpaceTreesDetector):
    """Backward-compatible alias for River's online isolation-style detector."""

    def get_name(self) -> str:
        return "Online Isolation Forest"


class RandomCutForestDetector(BaseDetector):
    """Random Cut Forest for streaming anomaly detection.

    River does not currently expose ``anomaly.RandomCutForest``. The detector is
    retained so older configurations fail with an actionable message.
    """

    score_orientation = "higher_is_more_anomalous"

    def get_name(self) -> str:
        return "Random Cut Forest"

    def fit(self, data: ArrayLike, **params: Any) -> RandomCutForestDetector:
        from river import anomaly  # lazy import

        if not hasattr(anomaly, "RandomCutForest"):
            raise ImportError(
                "river.anomaly.RandomCutForest is not available in the installed "
                "River version. Use 'half_space_trees' or 'online_isolation_forest' "
                "for River-backed streaming anomaly detection."
            )
        self.model = anomaly.RandomCutForest(**params)
        for row in _to_dicts(data):
            self.model.learn_one(row)
        return self

    def score(self, data: ArrayLike) -> list[float]:
        return [float(self.model.score_one(row)) for row in _to_dicts(data)]


__all__ = [
    "HalfSpaceTreesDetector",
    "OnlineIsolationForestDetector",
    "RandomCutForestDetector",
]
