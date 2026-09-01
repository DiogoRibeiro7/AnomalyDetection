from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - optional dependency for typing only
    from analytics.preprocessing import PreprocessingPipeline

ScoreOrientation = Literal[
    "higher_is_more_anomalous",
    "lower_is_more_anomalous",
    "binary_anomaly",
    "estimator_defined",
]
TabularArray = NDArray[np.floating[Any]]


class OrientedScores(np.ndarray):
    """NumPy score array carrying detector score and alignment metadata."""

    score_orientation: ScoreOrientation
    label_indices: NDArray[np.int_]
    window_spec: dict[str, int | str]

    def __new__(
        cls,
        values: Any,
        score_orientation: ScoreOrientation,
    ) -> OrientedScores:
        obj = np.asarray(values, dtype=float).view(cls)
        obj.score_orientation = score_orientation
        for name in ("label_indices", "window_spec"):
            if hasattr(values, name):
                setattr(obj, name, getattr(values, name))
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.score_orientation = getattr(obj, "score_orientation", "estimator_defined")
        for name in ("label_indices", "window_spec"):
            if hasattr(obj, name):
                setattr(self, name, getattr(obj, name))


def coerce_tabular_2d(
    data: pd.DataFrame | Any,
    *,
    detector_name: str = "Detector",
    allow_empty: bool = False,
) -> TabularArray:
    """Return a dense 2-D floating-point array for tabular detectors."""

    if isinstance(data, pd.DataFrame):
        array = data.to_numpy(dtype=float, copy=False)
    else:
        array = np.asarray(data, dtype=float)

    if array.ndim != 2:
        raise ValueError(f"{detector_name} input must be a 2-D array.")
    if not allow_empty and array.shape[0] == 0:
        raise ValueError(f"{detector_name} input must contain at least one sample.")

    return array


class BaseDetector(ABC):
    """Common interface for all anomaly detectors.

    Detector implementations expose a three-step lifecycle:

    - ``fit(data, **params)`` trains the detector and marks it as fitted.
    - ``score(data)`` returns detector-specific anomaly scores and requires a
      successful prior fit.
    - ``detect_anomalies(data, **params)`` is the fit-and-score convenience
      path used by the benchmark CLI.
    """

    score_orientation: ScoreOrientation = "estimator_defined"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._wrap_lifecycle_method("fit", cls._wrap_fit)
        cls._wrap_lifecycle_method("score", cls._wrap_score)

    @classmethod
    def _wrap_lifecycle_method(
        cls,
        method_name: str,
        wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]],
    ) -> None:
        method = cls.__dict__.get(method_name)
        if method is None or getattr(method, "_detector_lifecycle_wrapped", False):
            return
        setattr(cls, method_name, wrapper_factory(method))

    @staticmethod
    def _wrap_fit(method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def _wrapped(self: BaseDetector, *args: Any, **kwargs: Any) -> Any:
            result = method(self, *args, **kwargs)
            self._mark_fitted()
            return result

        _wrapped._detector_lifecycle_wrapped = True  # type: ignore[attr-defined]
        return _wrapped

    @staticmethod
    def _wrap_score(method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def _wrapped(self: BaseDetector, *args: Any, **kwargs: Any) -> Any:
            self._require_fitted()
            return method(self, *args, **kwargs)

        _wrapped._detector_lifecycle_wrapped = True  # type: ignore[attr-defined]
        return _wrapped

    def __init__(
        self, preprocessing_pipeline: PreprocessingPipeline | None = None
    ) -> None:
        self._preprocessing_pipeline = preprocessing_pipeline
        self._preprocessing_fitted = False
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the detector has completed a successful fit."""

        return bool(getattr(self, "_is_fitted", False))

    def _mark_fitted(self) -> None:
        self._is_fitted = True

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.get_name()} must be fitted before calling score()."
            )

    @property
    def preprocessing_pipeline(self) -> PreprocessingPipeline | None:
        """Return the currently configured preprocessing pipeline."""

        return self._preprocessing_pipeline

    def set_preprocessing_pipeline(
        self, pipeline: PreprocessingPipeline | None
    ) -> None:
        """Attach a preprocessing pipeline used prior to model training."""

        self._preprocessing_pipeline = pipeline
        self._preprocessing_fitted = False

    def fit_preprocessed(self, data: Any, **params: Any):
        """Fit the detector after applying the preprocessing pipeline."""

        prepared = self._preprocess_for_fit(data)
        return self.fit(prepared, **params)

    def score_preprocessed(self, data: Any):
        """Score data after applying the preprocessing pipeline."""

        prepared = self._preprocess_for_score(data)
        return self.score(prepared)

    def _preprocess_for_fit(self, data: Any):
        if self._preprocessing_pipeline is None:
            return data
        transformed = self._preprocessing_pipeline.fit_transform(data)
        self._preprocessing_fitted = True
        return transformed

    def _preprocess_for_score(self, data: Any):
        if self._preprocessing_pipeline is None:
            return data
        if not self._preprocessing_fitted:
            raise RuntimeError(
                "Preprocessing pipeline must be fitted by calling fit_preprocessed "
                "before scoring."
            )
        return self._preprocessing_pipeline.transform(data)

    @abstractmethod
    def get_name(self) -> str:
        """Return human readable detector name."""

    @abstractmethod
    def fit(self, data, **params):
        """Fit the detector to the provided data."""

    @abstractmethod
    def score(self, data):
        """Return anomaly scores for the provided data."""

    def detect_anomalies(self, data, **params):
        """Fit and return raw scores carrying their orientation metadata."""

        pipeline = params.pop("preprocessing_pipeline", None)
        if pipeline is not None:
            self.set_preprocessing_pipeline(pipeline)
        prepared = self._preprocess_for_fit(data)
        self.fit(prepared, **params)
        raw_scores = (
            self.score(data)
            if self._preprocessing_pipeline is None
            else self.score(prepared)
        )
        return OrientedScores(raw_scores, self.score_orientation)
