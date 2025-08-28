"""Classical anomaly detectors built on scikit-learn and PyOD.

Each detector exposes a :class:`~analytics.base.BaseDetector` interface with
``fit`` and ``score`` methods.  The implementations avoid importing optional
dependencies until required to keep the package lightweight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.base import BaseDetector
from analytics.lof import LOF

from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor, KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def _compute_variable_width_edges(feature: np.ndarray, k: int) -> np.ndarray:
    """Return bin edges so each bin has roughly equal frequency."""
    edges = np.percentile(feature, np.linspace(0, 100, k + 1))
    edges = np.unique(edges)
    if len(edges) - 1 < k:
        edges = np.histogram_bin_edges(feature, bins=k)
    return edges


class IsolationForestDetector(BaseDetector):
    """Wrapper around scikit-learn's :class:`IsolationForest`."""

    def get_name(self) -> str:
        return "Isolation Forest"

    def fit(self, data, **params):
        self.model = IsolationForest(verbose=1, **params)
        self.model.fit(data)
        return self

    def score(self, data):
        return self.model.decision_function(data)


class LOFDetector(BaseDetector):
    """Local Outlier Factor implementation allowing new data scoring."""

    def get_name(self) -> str:
        return "Local Outlier Factor"

    def fit(self, data, normalize: bool = False, **params):
        X = [tuple(x) for x in data.to_records(index=False)]
        self.lof = LOF(X, normalize=normalize)
        self.min_pts = params.get("min_pts", 3)
        return self

    def score(self, data):
        X = [tuple(x) for x in data.to_records(index=False)]
        return [-self.lof.local_outlier_factor(self.min_pts, point) for point in X]


class SOSDetector(BaseDetector):
    """Stochastic Outlier Selection using the ``sksos`` package."""

    def get_name(self) -> str:
        return "Stochastic Outlier Selection"

    def fit(self, data, **params):
        from sksos import SOS  # lazy import

        perplexity = params.get("perplexity", 30)
        metric = params.get("metric", "euclidean")
        eps = params.get("eps", 1e-5)
        self.model = SOS(perplexity=perplexity, metric=metric, eps=eps)
        self.X = data.values if isinstance(data, pd.DataFrame) else data
        return self

    def score(self, data=None):
        X = (
            self.X
            if data is None
            else (data.values if isinstance(data, pd.DataFrame) else data)
        )
        return -self.model.predict(X)


class EnsembleDetector(BaseDetector):
    """Simple ensemble averaging KNN, SOS and HBOS scores."""

    def get_name(self) -> str:
        return "Ensembled detector"

    def fit(self, data, **params):
        self.knn = KNNDetector().fit(data, **params)
        self.sos = SOSDetector().fit(data, **params)
        self.hbos = HBOSDetector().fit(data, **params)
        return self

    def score(self, data):
        knn_scores = self.knn.score(data)
        sos_scores = self.sos.score(data)
        hbos_scores = self.hbos.score(data)
        scores = np.vstack([knn_scores, sos_scores, hbos_scores])
        scores /= scores.max(axis=1, keepdims=True)
        return -scores.sum(axis=0)


class HBOSDetector(BaseDetector):
    """Histogram-Based Outlier Score with vectorized scoring."""

    def get_name(self) -> str:
        return "Histogram-Based Outlier Score"

    def fit(self, data, **params):
        k = params.get("k", 3)
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        self.histograms = {}
        for i in range(data.shape[1]):
            feature = data[:, i]
            edges = _compute_variable_width_edges(feature, k)
            self.histograms[i] = np.histogram(feature, bins=edges, density=True)
        return self

    def score(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        n_samples = data.shape[0]
        log_probs = np.zeros(n_samples)
        for j in range(data.shape[1]):
            hist, edges = self.histograms[j]
            idx = np.digitize(data[:, j], edges[1:-1], right=False)
            idx = np.clip(idx, 0, len(hist) - 1)
            log_probs += np.log(hist[idx] + 1e-12)
        return log_probs


class KNNDetector(BaseDetector):
    """K-Nearest Neighbors distance-based detector."""

    def get_name(self) -> str:
        return "K-Nearest Neighbors"

    def fit(self, data, **params):
        self.k = params.get("k", 3)
        self.neigh = NearestNeighbors(n_neighbors=self.k + 1).fit(data)
        return self

    def score(self, data):
        distances, _ = self.neigh.kneighbors(data)
        return -np.sum(distances, axis=1)


class OneClassSVMDetector(BaseDetector):
    """Wrapper around :class:`sklearn.svm.OneClassSVM`."""

    def get_name(self) -> str:
        return "One-Class SVM"

    def fit(self, data, **params):
        self.model = OneClassSVM(**params)
        self.model.fit(data)
        return self

    def score(self, data):
        return self.model.decision_function(data)


class DBSCANDetector(BaseDetector):
    """Density-based spatial clustering anomaly detector."""

    def get_name(self) -> str:
        return "DBSCAN"

    def fit(self, data, **params):
        self.model = DBSCAN(**params)
        self.model.fit(data)
        return self

    def score(self, data):
        labels = self.model.fit_predict(data)
        return np.where(labels == -1, 1.0, 0.0)


class EllipticEnvelopeDetector(BaseDetector):
    """Robust covariance estimate assuming Gaussian distributed data."""

    def get_name(self) -> str:
        return "Elliptic Envelope"

    def fit(self, data, **params):
        self.model = EllipticEnvelope(**params)
        self.model.fit(data)
        return self

    def score(self, data):
        return self.model.decision_function(data)


class GaussianMixtureDetector(BaseDetector):
    """Gaussian Mixture negative log-likelihood as anomaly score."""

    def get_name(self) -> str:
        return "Gaussian Mixture"

    def fit(self, data, **params):
        self.model = GaussianMixture(**params)
        self.model.fit(data)
        return self

    def score(self, data):
        return -self.model.score_samples(data)


class SklearnLOFDetector(BaseDetector):
    """Scikit-learn's LOF with novelty mode for scoring new data."""

    def get_name(self) -> str:
        return "Sklearn LOF"

    def fit(self, data, **params):
        self.model = LocalOutlierFactor(novelty=True, **params)
        self.model.fit(data)
        return self

    def score(self, data):
        return self.model.decision_function(data)


class KMeansDetector(BaseDetector):
    """Distance to nearest KMeans centroid as anomaly score."""

    def get_name(self) -> str:
        return "KMeans"

    def fit(self, data, **params):
        self.n_clusters = params.pop("n_clusters", 8)
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(data)
        return self

    def score(self, data):
        distances = self.kmeans.transform(data)
        min_dist = np.min(distances, axis=1)
        return -min_dist


class PCAReconstructionDetector(BaseDetector):
    """Use PCA reconstruction error as anomaly score."""

    def get_name(self) -> str:
        return "PCA Reconstruction"

    def fit(self, data, **params):
        self.n_components = params.pop("n_components", 0.95)
        self.pca = PCA(n_components=self.n_components).fit(data)
        return self

    def score(self, data):
        transformed = self.pca.transform(data)
        reconstructed = self.pca.inverse_transform(transformed)
        errors = np.linalg.norm(data - reconstructed, axis=1)
        return -errors


class MahalanobisDetector(BaseDetector):
    """Mahalanobis distance using empirical covariance."""

    def get_name(self) -> str:
        return "Mahalanobis"

    def fit(self, data, **params):
        self.cov = EmpiricalCovariance(**params)
        self.cov.fit(data)
        return self

    def score(self, data):
        distances = self.cov.mahalanobis(data)
        return -distances


class KDEDetector(BaseDetector):
    """Kernel Density Estimator returning log-density scores."""

    def get_name(self) -> str:
        return "Kernel Density"

    def fit(self, data, **params):
        self.kde = KernelDensity(**params)
        self.kde.fit(data)
        return self

    def score(self, data):
        return self.kde.score_samples(data)


class COPODDetector(BaseDetector):
    """Copula-based Outlier Detector from PyOD."""

    def get_name(self) -> str:
        return "COPOD"

    def fit(self, data, **params):
        from pyod.models.copod import COPOD

        self.model = COPOD(**params)
        self.X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(self.X)
        return self

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else data
        return -self.model.decision_function(X)


class FeatureBaggingDetector(BaseDetector):
    """Feature Bagging ensemble from PyOD."""

    def get_name(self) -> str:
        return "Feature Bagging"

    def fit(self, data, **params):
        from pyod.models.feature_bagging import FeatureBagging

        self.model = FeatureBagging(**params)
        self.X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(self.X)
        return self

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else data
        return -self.model.decision_function(X)


class LODADetector(BaseDetector):
    """Lightweight Online Detector of Anomalies from PyOD."""

    def get_name(self) -> str:
        return "LODA"

    def fit(self, data, **params):
        from pyod.models.loda import LODA

        self.model = LODA(**params)
        self.X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(self.X)
        return self

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else data
        return -self.model.decision_function(X)


class ABODDetector(BaseDetector):
    """Angle-Based Outlier Detector from PyOD."""

    def get_name(self) -> str:
        return "ABOD"

    def fit(self, data, **params):
        from pyod.models.abod import ABOD

        self.model = ABOD(**params)
        self.X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(self.X)
        return self

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else data
        return -self.model.decision_function(X)


__all__ = [name for name in globals() if name.endswith("Detector")]
