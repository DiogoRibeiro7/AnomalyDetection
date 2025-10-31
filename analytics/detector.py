"""Backwards compatible aggregator for detector classes.

Detectors are now organised under :mod:`analytics.detectors` submodules.  This
module re-exports all detector classes for code that previously imported from
``analytics.detector``.
"""

from .detectors import classical as _classical
from .detectors import deep as _deep
from .detectors import forecasting as _forecasting
from .detectors import graph as _graph
from .detectors import streaming as _streaming

_MODULES = (_classical, _deep, _streaming, _graph, _forecasting)

__all__ = [name for module in _MODULES for name in module.__all__]

for module in _MODULES:
    globals().update({name: getattr(module, name) for name in module.__all__})
