from __future__ import annotations

from importlib import import_module
from typing import MutableMapping, Type

from analytics.base import BaseDetector

DETECTOR_REGISTRY: MutableMapping[str, str] = {}


def register_detector(name: str, path: str) -> None:
    """Register a detector class by dotted module path.

    Parameters
    ----------
    name:
        Short key used to reference the detector.
    path:
        Dotted ``module:ClassName`` path.
    """
    DETECTOR_REGISTRY[name] = path


def get_detector_class(name: str) -> Type[BaseDetector]:
    """Return the detector class associated with *name*.

    The class is imported lazily to avoid importing optional dependencies
    until absolutely necessary.
    """
    module_path, class_name = DETECTOR_REGISTRY[name].split(":")
    module = import_module(module_path)
    return getattr(module, class_name)
