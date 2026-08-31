"""Detector registry and lazy-loading utilities."""

from .registry import (
    DETECTOR_REGISTRY,
    get_detector_class,
    register_detector,
)

# Built-in detector registrations grouped by theme
# Classical detectors
register_detector(
    "isolation_forest", "analytics.detectors.classical:IsolationForestDetector"
)
register_detector("sos", "analytics.detectors.classical:SOSDetector")
register_detector("knn", "analytics.detectors.classical:KNNDetector")
register_detector("hbos", "analytics.detectors.classical:HBOSDetector")
register_detector("ocsvm", "analytics.detectors.classical:OneClassSVMDetector")
register_detector("dbscan", "analytics.detectors.correctness:InductiveDBSCANDetector")
register_detector(
    "elliptic_envelope", "analytics.detectors.classical:EllipticEnvelopeDetector"
)
register_detector(
    "gaussian_mixture", "analytics.detectors.classical:GaussianMixtureDetector"
)
register_detector("sklearn_lof", "analytics.detectors.classical:SklearnLOFDetector")
register_detector("kmeans", "analytics.detectors.classical:KMeansDetector")
register_detector(
    "pca_reconstruction", "analytics.detectors.classical:PCAReconstructionDetector"
)
register_detector("mahalanobis", "analytics.detectors.classical:MahalanobisDetector")
register_detector("kde", "analytics.detectors.classical:KDEDetector")
register_detector("autoencoder", "analytics.detectors.deep:AutoencoderDetector")
register_detector("copod", "analytics.detectors.classical:COPODDetector")
register_detector(
    "feature_bagging", "analytics.detectors.classical:FeatureBaggingDetector"
)
register_detector("loda", "analytics.detectors.classical:LODADetector")
register_detector("abod", "analytics.detectors.classical:ABODDetector")
register_detector("ecod", "analytics.detectors.modern_tabular:ECODDetector")
register_detector(
    "random_network_distillation",
    "analytics.detectors.modern_tabular:RandomNetworkDistillationDetector",
)
register_detector(
    "random_feature_isolation_forest",
    "analytics.detectors.modern_tabular:RandomFeatureIsolationForestDetector",
)

# Streaming detectors
register_detector(
    "half_space_trees",
    "analytics.detectors.streaming:HalfSpaceTreesDetector",
)
register_detector(
    "online_isolation_forest",
    "analytics.detectors.streaming:OnlineIsolationForestDetector",
)
register_detector(
    "random_cut_forest", "analytics.detectors.streaming:RandomCutForestDetector"
)

# Deep learning detectors
register_detector(
    "denoising_autoencoder", "analytics.detectors.deep:DenoisingAutoencoderDetector"
)
register_detector(
    "variational_autoencoder", "analytics.detectors.deep:VariationalAutoencoderDetector"
)
register_detector(
    "lstm_autoencoder", "analytics.detectors.temporal:LSTMAutoencoderDetector"
)
register_detector(
    "tcn_autoencoder", "analytics.detectors.temporal:TCNAutoencoderDetector"
)
register_detector("transformer", "analytics.detectors.temporal:TransformerDetector")
register_detector("anogan", "analytics.detectors.deep:AnoGANDetector")
register_detector("madgan", "analytics.detectors.deep:MADGANDetector")

# Graph detectors
register_detector(
    "degree_centrality", "analytics.detectors.graph:DegreeCentralityDetector"
)
register_detector(
    "graph_isolation_forest", "analytics.detectors.graph:GraphIsolationForestDetector"
)

# Forecasting detectors
register_detector("arima", "analytics.detectors.forecasting:ARIMADetector")
register_detector("prophet", "analytics.detectors.forecasting:ProphetDetector")

# User-facing selection is strict after all built-ins have been registered.
DETECTOR_REGISTRY.freeze()

__all__ = [
    "DETECTOR_REGISTRY",
    "get_detector_class",
    "register_detector",
]
