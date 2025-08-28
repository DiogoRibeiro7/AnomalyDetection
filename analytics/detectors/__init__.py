"""Detector registry and lazy-loading utilities."""

from .registry import (
    DETECTOR_REGISTRY,
    get_detector_class,
    register_detector,
)

# Built-in detector registrations grouped by theme
# Classical detectors
register_detector("isolation_forest", "analytics.detector:IsolationForestDetector")
register_detector("sos", "analytics.detector:SOSDetector")
register_detector("knn", "analytics.detector:KNNDetector")
register_detector("hbos", "analytics.detector:HBOSDetector")
register_detector("ocsvm", "analytics.detector:OneClassSVMDetector")
register_detector("dbscan", "analytics.detector:DBSCANDetector")
register_detector("elliptic_envelope", "analytics.detector:EllipticEnvelopeDetector")
register_detector("gaussian_mixture", "analytics.detector:GaussianMixtureDetector")
register_detector("sklearn_lof", "analytics.detector:SklearnLOFDetector")
register_detector("kmeans", "analytics.detector:KMeansDetector")
register_detector("pca_reconstruction", "analytics.detector:PCAReconstructionDetector")
register_detector("mahalanobis", "analytics.detector:MahalanobisDetector")
register_detector("kde", "analytics.detector:KDEDetector")
register_detector("autoencoder", "analytics.detector:AutoencoderDetector")
register_detector("copod", "analytics.detector:COPODDetector")
register_detector("feature_bagging", "analytics.detector:FeatureBaggingDetector")
register_detector("loda", "analytics.detector:LODADetector")
register_detector("abod", "analytics.detector:ABODDetector")

# Streaming detectors
register_detector(
    "online_isolation_forest", "analytics.detector:OnlineIsolationForestDetector"
)
register_detector("random_cut_forest", "analytics.detector:RandomCutForestDetector")

# Deep learning detectors
register_detector(
    "denoising_autoencoder", "analytics.detector:DenoisingAutoencoderDetector"
)
register_detector(
    "variational_autoencoder", "analytics.detector:VariationalAutoencoderDetector"
)
register_detector("lstm_autoencoder", "analytics.detector:LSTMAutoencoderDetector")
register_detector("transformer", "analytics.detector:TransformerDetector")
register_detector("anogan", "analytics.detector:AnoGANDetector")
register_detector("madgan", "analytics.detector:MADGANDetector")

# Graph detectors
register_detector("degree_centrality", "analytics.detector:DegreeCentralityDetector")
register_detector(
    "graph_isolation_forest", "analytics.detector:GraphIsolationForestDetector"
)

# Forecasting detectors
register_detector("arima", "analytics.detector:ARIMADetector")
register_detector("prophet", "analytics.detector:ProphetDetector")

__all__ = [
    "DETECTOR_REGISTRY",
    "get_detector_class",
    "register_detector",
]
