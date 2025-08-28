"""Backwards compatible aggregator for detector classes.

Detectors are now organised under :mod:`analytics.detectors` submodules.  This
module re-exports all detector classes for code that previously imported from
``analytics.detector``.
"""

from .detectors.classical import *  # noqa: F401,F403
from .detectors.deep import *  # noqa: F401,F403
from .detectors.streaming import *  # noqa: F401,F403
from .detectors.graph import *  # noqa: F401,F403
from .detectors.forecasting import *  # noqa: F401,F403
