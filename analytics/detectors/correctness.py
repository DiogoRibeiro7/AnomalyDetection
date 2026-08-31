"""Correctness-focused detector implementations.

These detectors preserve public registry keys while fixing behavioural
contracts that cannot be implemented safely by the legacy wrappers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

from analytics.base import BaseDetector, coerce_tabular_2d

ScoreArray = NDArray[np.floating[Any]]


class InductiveDBSCANDetector(BaseDetector):
    """DBSCAN with an explicit out-of-sample scoring rule.

    Scikit-learn's DBSCAN is transductive and does not provide ``predict``.
    This wrapper therefore fits DBSCAN once, stores its core samples, and marks
    a new sample as an inlier when it lies within ``eps`` of at least one fitted
    core sample. Otherwise the sample is reported as an anomaly.

    This avoids refitting the clustering model on the evaluation batch and
    makes ``fit`` followed by ``score`` obey the common detector lifecycle.
    """

    score_orientation = "binary_anomaly"

    def get_name(self) -> str:
        return "DBSCAN"

    def fit(
        self,
        data: pd.DataFrame | Any,
        **params: Any,
    ) -> InductiveDBSCANDetector:
        """Fit DBSCAN and retain core samples for inductive scoring."""

        X = coerce_tabular_2d(data, detector_name="DBSCAN")
        self.model = DBSCAN(**params)
        self.model.fit(X)
        self.eps = float(self.model.eps)
        self.metric = self.model.metric
        self.metric_params = self.model.metric_params
        self.p = self.model.p
        self.core_samples_ = np.asarray(self.model.components_, dtype=float)
        return self

    def score(self, data: pd.DataFrame | Any) -> ScoreArray:
        """Return ``1`` for anomalies and ``0`` for fitted-density inliers."""

        X = coerce_tabular_2d(data, detector_name="DBSCAN")
        if self.core_samples_.shape[0] == 0:
            return np.ones(X.shape[0], dtype=float)

        metric_kwargs: dict[str, Any] = {}
        if self.metric_params:
            metric_kwargs.update(self.metric_params)
        if self.metric == "minkowski" and self.p is not None:
            metric_kwargs.setdefault("p", self.p)

        distances = pairwise_distances(
            X,
            self.core_samples_,
            metric=self.metric,
            **metric_kwargs,
        )
        is_inlier = np.any(distances <= self.eps, axis=1)
        return np.where(is_inlier, 0.0, 1.0)


__all__ = ["InductiveDBSCANDetector"]
