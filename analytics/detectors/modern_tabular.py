"""Modern tabular anomaly detectors with lightweight dependencies."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from analytics.base import BaseDetector, coerce_tabular_2d

ArrayLike = NDArray[np.floating[Any]]
type FrameOrArray = pd.DataFrame | ArrayLike
ScoreArray = NDArray[np.floating[Any]]


def _coerce(data: FrameOrArray) -> ArrayLike:
    return coerce_tabular_2d(data, detector_name="Modern tabular detector")


def _random_feature_map(
    data: ArrayLike,
    weights: ArrayLike,
    bias: NDArray[np.floating[Any]],
) -> ArrayLike:
    return np.tanh(data @ weights + bias)


class RandomNetworkDistillationDetector(BaseDetector):
    """Random Network Distillation for tabular anomaly scoring.

    A fixed random feature network defines a target representation. A compact
    predictor network is trained to reproduce that representation on training
    samples. High prediction error indicates inputs that do not match the
    learned normal structure.
    """

    score_orientation = "higher_is_more_anomalous"
    method_status = "native"
    implementation_provenance = "project-native"
    dependency_extra = "base"
    upstream_provider = None
    upstream_version_range = None
    preset_configs = {
        "smoke": {
            "representation_dim": 8,
            "hidden_layer_sizes": (8,),
            "max_iter": 1000,
        },
        "balanced": {
            "representation_dim": 32,
            "hidden_layer_sizes": (32,),
            "max_iter": 1000,
        },
        "research": {
            "representation_dim": 64,
            "hidden_layer_sizes": (64, 32),
            "max_iter": 2000,
        },
    }

    def get_name(self) -> str:
        return "Random Network Distillation"

    def fit(
        self,
        data: FrameOrArray,
        *,
        representation_dim: int | None = None,
        hidden_layer_sizes: tuple[int, ...] = (32,),
        max_iter: int = 500,
        random_state: int | None = None,
        solver: str = "lbfgs",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        **params: Any,
    ) -> RandomNetworkDistillationDetector:
        X = _coerce(data)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        rng = np.random.default_rng(random_state)
        n_features = X_scaled.shape[1]
        target_dim = (
            representation_dim
            if representation_dim is not None
            else min(max(n_features * 4, 8), 64)
        )
        if target_dim <= 0:
            raise ValueError("representation_dim must be greater than zero.")
        self.target_weights = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(1, n_features)),
            size=(n_features, target_dim),
        )
        self.target_bias = rng.normal(loc=0.0, scale=0.1, size=target_dim)
        target = _random_feature_map(X_scaled, self.target_weights, self.target_bias)
        self.predictor = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            **params,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.predictor.fit(X_scaled, target)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        X = self.scaler.transform(_coerce(data))
        target = _random_feature_map(X, self.target_weights, self.target_bias)
        predicted = self.predictor.predict(X)
        return np.mean((target - predicted) ** 2, axis=1)


class RandomFeatureIsolationForestDetector(BaseDetector):
    """Isolation Forest on random nonlinear tabular representations."""

    score_orientation = "higher_is_more_anomalous"
    method_status = "native"
    implementation_provenance = "project-native"
    dependency_extra = "base"
    upstream_provider = None
    upstream_version_range = None
    preset_configs = {
        "smoke": {"representation_dim": 16, "n_estimators": 24},
        "balanced": {"representation_dim": 64, "n_estimators": 100},
        "research": {"representation_dim": 128, "n_estimators": 300},
    }

    def get_name(self) -> str:
        return "Random Feature Isolation Forest"

    def fit(
        self,
        data: FrameOrArray,
        *,
        representation_dim: int | None = None,
        random_state: int | None = None,
        n_estimators: int = 100,
        contamination: str | float = "auto",
        **params: Any,
    ) -> RandomFeatureIsolationForestDetector:
        X = _coerce(data)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        rng = np.random.default_rng(random_state)
        n_features = X_scaled.shape[1]
        target_dim = (
            representation_dim
            if representation_dim is not None
            else min(max(n_features * 8, 16), 128)
        )
        if target_dim <= 0:
            raise ValueError("representation_dim must be greater than zero.")
        self.target_weights = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(1, n_features)),
            size=(n_features, target_dim),
        )
        self.target_bias = rng.normal(loc=0.0, scale=0.1, size=target_dim)
        features = _random_feature_map(X_scaled, self.target_weights, self.target_bias)
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            **params,
        )
        self.model.fit(features)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        X = self.scaler.transform(_coerce(data))
        features = _random_feature_map(X, self.target_weights, self.target_bias)
        return -np.asarray(self.model.decision_function(features), dtype=float)


class ECODDetector(BaseDetector):
    """Empirical-CDF Outlier Detection from PyOD."""

    score_orientation = "higher_is_more_anomalous"
    method_status = "adapter"
    implementation_provenance = "pyod-adapter"
    dependency_extra = "base"
    upstream_provider = "pyod"
    upstream_module = "pyod.models.ecod.ECOD"
    upstream_version_range = ">=2.0"
    preset_configs: dict[str, dict[str, Any]] = {
        "smoke": {},
        "balanced": {},
        "research": {},
    }

    def get_name(self) -> str:
        return "ECOD"

    def fit(self, data: FrameOrArray, **params: Any) -> ECODDetector:
        try:
            from pyod.models.ecod import ECOD
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise ImportError("ECODDetector requires PyOD to be installed.") from exc
        self.model = ECOD(**params)
        self.model.fit(_coerce(data))
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        return np.asarray(self.model.decision_function(_coerce(data)), dtype=float)


__all__ = [
    "ECODDetector",
    "RandomFeatureIsolationForestDetector",
    "RandomNetworkDistillationDetector",
]
