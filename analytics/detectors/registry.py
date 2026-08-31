from __future__ import annotations

from importlib import import_module
from typing import Type

from analytics.base import BaseDetector


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
            available = ", ".join(sorted(self))
            raise ValueError(
                f"Unknown detector '{key}'. Available detectors: {available}"
            )
        return exists


DETECTOR_REGISTRY = DetectorRegistry()


def _validate_registration(name: str, path: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Detector name must be a non-empty string.")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Detector path must be a non-empty string.")
    if ":" not in path:
        raise ValueError(
            "Detector path must use 'module:ClassName' format " f"(received: {path!r})."
        )
    module_path, class_name = path.split(":", 1)
    if not module_path or not class_name:
        raise ValueError(
            "Detector path must include both module and class name "
            f"(received: {path!r})."
        )


def register_detector(name: str, path: str, *, allow_override: bool = False) -> None:
    """Register a detector class by dotted module path."""

    _validate_registration(name, path)
    exists = dict.__contains__(DETECTOR_REGISTRY, name)
    if exists and not allow_override:
        existing = DETECTOR_REGISTRY[name]
        raise ValueError(
            f"Detector '{name}' is already registered with '{existing}'. "
            "Pass allow_override=True to replace it."
        )
    DETECTOR_REGISTRY[name] = path


def get_detector_class(name: str) -> Type[BaseDetector]:
    """Return the detector class associated with *name*."""

    if not dict.__contains__(DETECTOR_REGISTRY, name):
        available = ", ".join(sorted(DETECTOR_REGISTRY))
        raise KeyError(f"Unknown detector '{name}'. Available detectors: {available}")

    registration = DETECTOR_REGISTRY[name]
    _validate_registration(name, registration)
    module_path, class_name = registration.split(":", 1)
    module = import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Detector '{name}' points to missing class '{class_name}' "
            f"in module '{module_path}'."
        ) from exc
