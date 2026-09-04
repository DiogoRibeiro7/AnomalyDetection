"""Detector registry and lazy-loading utilities."""

from .registry import (
    DETECTOR_REGISTRY,
    get_detector_class,
    register_detector,
)

# Built-in detector registrations grouped by theme
# Classical detectors
register_detector(
    "isolation_forest",
    "anomalybench.analytics.detectors.classical:IsolationForestDetector",
)
register_detector("sos", "anomalybench.analytics.detectors.classical:SOSDetector")
register_detector("knn", "anomalybench.analytics.detectors.classical:KNNDetector")
register_detector("hbos", "anomalybench.analytics.detectors.classical:HBOSDetector")
register_detector(
    "ocsvm", "anomalybench.analytics.detectors.classical:OneClassSVMDetector"
)
register_detector(
    "dbscan", "anomalybench.analytics.detectors.correctness:InductiveDBSCANDetector"
)
register_detector(
    "elliptic_envelope",
    "anomalybench.analytics.detectors.classical:EllipticEnvelopeDetector",
)
register_detector(
    "gaussian_mixture",
    "anomalybench.analytics.detectors.classical:GaussianMixtureDetector",
)
register_detector(
    "sklearn_lof", "anomalybench.analytics.detectors.classical:SklearnLOFDetector"
)
register_detector("kmeans", "anomalybench.analytics.detectors.classical:KMeansDetector")
register_detector(
    "pca_reconstruction",
    "anomalybench.analytics.detectors.classical:PCAReconstructionDetector",
)
register_detector(
    "mahalanobis", "anomalybench.analytics.detectors.classical:MahalanobisDetector"
)
register_detector("kde", "anomalybench.analytics.detectors.classical:KDEDetector")
register_detector(
    "autoencoder", "anomalybench.analytics.detectors.deep:AutoencoderDetector"
)
register_detector("copod", "anomalybench.analytics.detectors.classical:COPODDetector")
register_detector(
    "feature_bagging",
    "anomalybench.analytics.detectors.classical:FeatureBaggingDetector",
)
register_detector("loda", "anomalybench.analytics.detectors.classical:LODADetector")
register_detector("abod", "anomalybench.analytics.detectors.classical:ABODDetector")
register_detector(
    "ecod", "anomalybench.analytics.detectors.modern_tabular:ECODDetector"
)
register_detector(
    "random_network_distillation",
    "anomalybench.analytics.detectors.modern_tabular:RandomNetworkDistillationDetector",
)
register_detector(
    "random_feature_isolation_forest",
    "anomalybench.analytics.detectors.modern_tabular:RandomFeatureIsolationForestDetector",
)

# Streaming detectors
register_detector(
    "half_space_trees",
    "anomalybench.analytics.detectors.streaming:HalfSpaceTreesDetector",
)
register_detector(
    "online_isolation_forest",
    "anomalybench.analytics.detectors.streaming:OnlineIsolationForestDetector",
)
register_detector(
    "random_cut_forest",
    "anomalybench.analytics.detectors.streaming:RandomCutForestDetector",
)

# Deep learning detectors
register_detector(
    "denoising_autoencoder",
    "anomalybench.analytics.detectors.deep:DenoisingAutoencoderDetector",
)
register_detector(
    "variational_autoencoder",
    "anomalybench.analytics.detectors.deep:VariationalAutoencoderDetector",
)
register_detector(
    "lstm_autoencoder",
    "anomalybench.analytics.detectors.temporal:LSTMAutoencoderDetector",
)
register_detector(
    "tcn_autoencoder",
    "anomalybench.analytics.detectors.temporal:TCNAutoencoderDetector",
)
register_detector(
    "transformer", "anomalybench.analytics.detectors.temporal:TransformerDetector"
)
register_detector("anogan", "anomalybench.analytics.detectors.deep:AnoGANDetector")
register_detector("madgan", "anomalybench.analytics.detectors.deep:MADGANDetector")

# Graph detectors
register_detector(
    "degree_centrality",
    "anomalybench.analytics.detectors.graph:DegreeCentralityDetector",
)
register_detector(
    "graph_isolation_forest",
    "anomalybench.analytics.detectors.graph:GraphIsolationForestDetector",
)

# Forecasting detectors
register_detector("arima", "anomalybench.analytics.detectors.forecasting:ARIMADetector")
register_detector(
    "prophet", "anomalybench.analytics.detectors.forecasting:ProphetDetector"
)

# User-facing selection is strict after all built-ins have been registered.
DETECTOR_REGISTRY.freeze()

__all__ = [
    "DETECTOR_REGISTRY",
    "get_detector_class",
    "register_detector",
]
