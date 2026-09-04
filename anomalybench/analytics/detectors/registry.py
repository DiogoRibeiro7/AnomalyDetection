from __future__ import annotations

from importlib import import_module

from dataexcept import ConfigurationError

from anomalybench.analytics.base import BaseDetector
from anomalybench.analytics.exceptions import UnknownDetectorError


class DetectorRegistry(dict[str, str]):
    """Detector mapping that can enforce strict lookups after initialization."""

    def __init__(self) -> None:
        super().__init__()
        self._strict = False

    def freeze(self) -> None:
        """Enable strict membership checks for user-facing detector selection."""

        self._strict = True

    def __contains__(self, key: object) -> bool:
        exists = super().__contains__(key)
        if self._strict and isinstance(key, str) and not exists:
            raise UnknownDetectorError(key, sorted(self))
        return exists


DETECTOR_REGISTRY = DetectorRegistry()


def _validate_registration(name: str, path: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("detector name", "must be a non-empty string")
    if not isinstance(path, str) or not path.strip():
        raise ConfigurationError("detector path", "must be a non-empty string")
    if ":" not in path:
        raise ConfigurationError(
            "detector path",
            f"must use 'module:ClassName' format (received: {path!r})",
        )
    module_path, class_name = path.split(":", 1)
    if not module_path or not class_name:
        raise ConfigurationError(
            "detector path",
            f"must include both module and class name (received: {path!r})",
        )


def register_detector(name: str, path: str, *, allow_override: bool = False) -> None:
    """Register a detector class by dotted module path."""

    _validate_registration(name, path)
    exists = dict.__contains__(DETECTOR_REGISTRY, name)
    if exists and not allow_override:
        existing = DETECTOR_REGISTRY[name]
        raise ConfigurationError(
            name,
            f"already registered with '{existing}'. "
            "Pass allow_override=True to replace it",
        )
    DETECTOR_REGISTRY[name] = path


def get_detector_class(name: str) -> type[BaseDetector]:
    """Return the detector class associated with *name*."""

    if not dict.__contains__(DETECTOR_REGISTRY, name):
        raise UnknownDetectorError(name, sorted(DETECTOR_REGISTRY))

    registration = DETECTOR_REGISTRY[name]
    _validate_registration(name, registration)
    module_path, class_name = registration.split(":", 1)
    module = import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ConfigurationError(
            name,
            f"points to missing class '{class_name}' " f"in module '{module_path}'",
        ) from exc
