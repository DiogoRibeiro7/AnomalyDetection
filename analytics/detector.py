from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sksos import SOS
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EmpiricalCovariance

from analytics.lof import LOF

import numpy as np
import pandas as pd


def _compute_variable_width_edges(feature, k):
    """Return bin edges so that each bin has roughly equal frequency.

    When the data has many repeated values, ``np.percentile`` may return
    duplicate edges which leads to fewer than ``k`` bins. In that case, the
    method falls back to numpy's automatic bin edge computation to maintain the
    requested number of bins.
    """
    edges = np.percentile(feature, np.linspace(0, 100, k + 1))
    edges = np.unique(edges)
    if len(edges) - 1 < k:
        edges = np.histogram_bin_edges(feature, bins=k)
    return edges


class Detector(object):
    def __init__(self):
        pass

    def get_name(self):
        raise NotImplementedError("This function needs to be implemented")

    def detect_anomalies(self, data, **params):
        raise NotImplementedError("This function needs to be implemented")


class PCADetector(object):
    def __init__(self, detector):
        self.detector = detector

    def get_name(self):
        raise NotImplementedError("This function needs to be implemented")

    def detect_anomalies(self, data, n_components=3, **params):
        pca = PCA(n_components=n_components)
        return self.detector.detect_anomalies(pca.fit_transform(data), **params)


class TSNEDetector(object):
    def __init__(self, detector):
        self.detector = detector

    def get_name(self):
        raise NotImplementedError("This function needs to be implemented")

    def detect_anomalies(self, data, n_components=3, **params):
        tsne = TSNE(n_components=n_components)
        return self.detector.detect_anomalies(tsne.fit_transform(data), **params)


class IsolationForestDetector(Detector):
    def get_name(self):
        return "Isolation Forest"

    def detect_anomalies(self, data, **params):
        iso_forest = IsolationForest(verbose=1)
        iso_forest.set_params(**params)
        iso_forest.fit(data)
        return iso_forest.decision_function(
            data
        )  # The anomaly score. The lower, the more abnormal.


class LOFDetector(Detector):
    def get_name(self):
        return "Local Outlier Factor"

    def detect_anomalies(self, data, **params):
        data = [tuple(x) for x in data.to_records(index=False)]
        lof = LOF(data, normalize=False)
        min_pts = 3
        if "min_pts" in params:
            min_pts = params["min_pts"]
        return [
            -lof.local_outlier_factor(min_pts, tuple(data[i])) for i in range(len(data))
        ]


class SOSDetector(Detector):
    def get_name(self):
        return "Stochastic Outlier Selection"

    def detect_anomalies(self, data, **params):
        perplexity = 30
        metric = "euclidean"
        eps = 1e-5
        if "perplexity" in params:
            perplexity = params["perplexity"]
        if "metric" in params:
            metric = params["metric"]
        if "eps" in params:
            eps = params["eps"]
        sos = SOS(
            perplexity=perplexity, metric=metric, eps=eps
        )  # https://github.com/jeroenjanssens/scikit-sos
        if isinstance(data, pd.DataFrame):
            return -sos.predict(data.values)
        else:
            return -sos.predict(data)


class EnsembleDetector(Detector):
    def get_name(self):
        return "Ensembled detector"

    def detect_anomalies(self, data, **params):
        knn_scores = KNNDetector().detect_anomalies(data)
        sos_scores = SOSDetector().detect_anomalies(data)
        hbos_scores = HBOSDetector().detect_anomalies(data)
        scores = [knn_scores, sos_scores, hbos_scores]

        norm_scores = []
        for score in scores:
            norm_score = np.array(score)
            _max = max(norm_score)
            norm_score /= _max
            norm_scores.append(norm_score)

        return -np.sum(np.array(norm_scores), axis=0)


class HBOSDetector(Detector):
    def get_name(self):
        return "Histogram-Based Outlier Score"

    def detect_anomalies(self, data, **params):
        # http://www.dfki.de/KI2012/PosterDemoTrack/ki2012pd13.pdf

        k = 3  # How many bins do we use in each histogram?
        if "k" in params:
            k = params["k"]

        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        # Use variable-width binning as proposed in the HBOS paper. Each bin
        # contains approximately the same number of samples. This is
        # implemented by computing the quantiles for every feature and using
        # them as bin edges.
        histograms = {}
        for i in range(data.shape[1]):
            feature = data[:, i]
            edges = _compute_variable_width_edges(feature, k)
            histograms[i] = np.histogram(feature, bins=edges, density=True)

        scores = []
        for i in range(data.shape[0]):
            record = data[i, :]
            score = 0
            for j in range(len(record)):
                histogram = histograms[j]
                for k in range(len(histogram[1]) - 1):
                    if histogram[1][k] <= record[j] < histogram[1][k + 1]:
                        score += np.log(histogram[0][k])
                        break
            scores.append(score)

        return scores


class KNNDetector(Detector):
    def get_name(self):
        return "K-Nearest Neighbors"

    def detect_anomalies(self, data, **params):
        k = 3
        if "k" in params:
            k = params["k"]
        distances, indices = (
            NearestNeighbors(n_neighbors=k + 1).fit(data).kneighbors(data)
        )

        return -np.sum(distances, axis=1)


class OneClassSVMDetector(Detector):
    def get_name(self):
        return "One-Class SVM"

    def detect_anomalies(self, data, **params):
        svm = OneClassSVM(**params)
        svm.fit(data)
        return svm.decision_function(data)


class DBSCANDetector(Detector):
    def get_name(self):
        return "DBSCAN"

    def detect_anomalies(self, data, **params):
        db = DBSCAN(**params)
        labels = db.fit_predict(data)
        return np.where(labels == -1, 1.0, 0.0)


class EllipticEnvelopeDetector(Detector):
    # Important! assumes gaussian distributions of data
    # Important! assumes that the number of outliers in known in advance (contamination param)
    # Important! n_samples > n_features ** 2  (apply PCA if this is not the case)
    def get_name(self):
        return "Elliptic Envelope"

    def detect_anomalies(self, data, **params):
        envelope = EllipticEnvelope(**params)
        envelope.fit(data)
        return envelope.decision_function(data)


class GaussianMixtureDetector(Detector):
    def get_name(self):
        return "Gaussian Mixture"

    def detect_anomalies(self, data, **params):
        gmm = GaussianMixture(**params)
        gmm.fit(data)
        # Higher negative log likelihood indicates more anomalous points
        return -gmm.score_samples(data)


class SklearnLOFDetector(Detector):
    def get_name(self):
        return "Sklearn LOF"

    def detect_anomalies(self, data, **params):
        lof = LocalOutlierFactor(novelty=True, **params)
        lof.fit(data)
        return lof.decision_function(data)


class KMeansDetector(Detector):
    def get_name(self):
        return "KMeans"

    def detect_anomalies(self, data, **params):
        n_clusters = params.pop("n_clusters", 8)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        distances = kmeans.transform(data)
        min_dist = np.min(distances, axis=1)
        return -min_dist


class PCAReconstructionDetector(Detector):
    def get_name(self):
        return "PCA Reconstruction"

    def detect_anomalies(self, data, **params):
        n_components = params.pop("n_components", 0.95)
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        reconstructed = pca.inverse_transform(transformed)
        errors = np.linalg.norm(data - reconstructed, axis=1)
        return -errors


class MahalanobisDetector(Detector):
    def get_name(self):
        return "Mahalanobis"

    def detect_anomalies(self, data, **params):
        cov = EmpiricalCovariance(**params)
        cov.fit(data)
        distances = cov.mahalanobis(data)
        return -distances
