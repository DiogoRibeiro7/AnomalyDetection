"""Exceptions raised by this package, rooted in the DataExcept hierarchy.

DataExcept supplies structured exceptions for most data and model failures, and
they are used directly wherever one fits. The classes here cover the cases it
has no direct equivalent for, so every error raised by this package still
inherits from :class:`dataexcept.DataExceptError` and can be caught with it.

These exceptions do **not** inherit from :class:`ValueError`, :class:`KeyError`,
or :class:`TypeError`. Code that caught those from this package must catch the
DataExcept types instead.
"""

from __future__ import annotations

from collections.abc import Iterable

from dataexcept import DataScienceError, ResourceNotFoundError

__all__ = [
    "DetectorNotFittedError",
    "UnknownDatasetError",
    "UnknownDetectorError",
]


class DetectorNotFittedError(DataScienceError):
    """Raised when a detector is used before :meth:`fit` has run.

    Attributes:
        detector_name: Display name of the detector that was not fitted.
        method: The method that required a fitted detector.
    """

    def __init__(self, detector_name: str, method: str = "score") -> None:
        self.detector_name = detector_name
        self.method = method
        super().__init__(f"{detector_name} must be fitted before calling {method}().")


class _UnknownKeyError(ResourceNotFoundError):
    """Shared base for registry and catalog lookups that list what is available."""

    _resource_type = "resource"
    _plural = "resources"

    def __init__(self, name: str, available: Iterable[str]) -> None:
        self.available = sorted(available)
        super().__init__(self._resource_type, name)
        listing = ", ".join(self.available)
        self.args = (
            f"Unknown {self._resource_type} '{name}'. "
            f"Available {self._plural}: {listing}",
        )


class UnknownDetectorError(_UnknownKeyError):
    """Raised when a detector key is absent from the registry."""

    _resource_type = "detector"
    _plural = "detectors"


class UnknownDatasetError(_UnknownKeyError):
    """Raised when a dataset selector names something the catalog does not hold."""

    _resource_type = "dataset"
    _plural = "datasets"
