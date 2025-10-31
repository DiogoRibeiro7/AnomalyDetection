from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - optional dependency for typing only
    from analytics.preprocessing import PreprocessingPipeline


class BaseDetector(ABC):
    """Common interface for all anomaly detectors."""

    def __init__(
        self, preprocessing_pipeline: PreprocessingPipeline | None = None
    ) -> None:
        self._preprocessing_pipeline = preprocessing_pipeline
        self._preprocessing_fitted = False

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
        """Convenience method that fits and scores in one step."""

        pipeline = params.pop("preprocessing_pipeline", None)
        if pipeline is not None:
            self.set_preprocessing_pipeline(pipeline)
        prepared = self._preprocess_for_fit(data)
        self.fit(prepared, **params)
        if self._preprocessing_pipeline is None:
            return self.score(data)
        return self.score(prepared)
