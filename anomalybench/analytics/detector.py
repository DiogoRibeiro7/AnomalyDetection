"""Backwards compatible aggregator for detector classes.

Detectors are now organised under :mod:`anomalybench.analytics.detectors`
submodules. This module re-exports every detector class for code that
previously imported from ``anomalybench.analytics.detector``.

The re-exports are written out explicitly rather than installed into
``globals()`` at import time, so type checkers and IDEs can resolve them.
"""

from __future__ import annotations

from .detectors.classical import (
    ABODDetector,
    COPODDetector,
    DBSCANDetector,
    EllipticEnvelopeDetector,
    EnsembleDetector,
    FeatureBaggingDetector,
    GaussianMixtureDetector,
    HBOSDetector,
    IsolationForestDetector,
    KDEDetector,
    KMeansDetector,
    KNNDetector,
    LODADetector,
    LOFDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
    PCAReconstructionDetector,
    SklearnLOFDetector,
    SOSDetector,
)
from .detectors.correctness import (
    InductiveDBSCANDetector,
)
from .detectors.deep import (
    AnoGANDetector,
    AutoencoderDetector,
    DenoisingAutoencoderDetector,
    LSTMAutoencoderDetector,
    MADGANDetector,
    TransformerDetector,
    VariationalAutoencoderDetector,
)
from .detectors.forecasting import (
    ARIMADetector,
    ProphetDetector,
)
from .detectors.graph import (
    DegreeCentralityDetector,
    GraphIsolationForestDetector,
)
from .detectors.modern_tabular import (
    ECODDetector,
    RandomFeatureIsolationForestDetector,
    RandomNetworkDistillationDetector,
)
from .detectors.streaming import (
    HalfSpaceTreesDetector,
    OnlineIsolationForestDetector,
    RandomCutForestDetector,
)

__all__ = [
    "ABODDetector",
    "ARIMADetector",
    "AnoGANDetector",
    "AutoencoderDetector",
    "COPODDetector",
    "DBSCANDetector",
    "DegreeCentralityDetector",
    "DenoisingAutoencoderDetector",
    "ECODDetector",
    "EllipticEnvelopeDetector",
    "EnsembleDetector",
    "FeatureBaggingDetector",
    "GaussianMixtureDetector",
    "GraphIsolationForestDetector",
    "HBOSDetector",
    "HalfSpaceTreesDetector",
    "InductiveDBSCANDetector",
    "IsolationForestDetector",
    "KDEDetector",
    "KMeansDetector",
    "KNNDetector",
    "LODADetector",
    "LOFDetector",
    "LSTMAutoencoderDetector",
    "MADGANDetector",
    "MahalanobisDetector",
    "OneClassSVMDetector",
    "OnlineIsolationForestDetector",
    "PCAReconstructionDetector",
    "ProphetDetector",
    "RandomCutForestDetector",
    "RandomFeatureIsolationForestDetector",
    "RandomNetworkDistillationDetector",
    "SOSDetector",
    "SklearnLOFDetector",
    "TransformerDetector",
    "VariationalAutoencoderDetector",
]
