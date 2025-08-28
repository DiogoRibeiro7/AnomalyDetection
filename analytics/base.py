from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Common interface for all anomaly detectors."""

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
        self.fit(data, **params)
        return self.score(data)
