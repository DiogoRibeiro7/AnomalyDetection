from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sksos import SOS
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors, KernelDensity
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from pyod.models.copod import COPOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.loda import LODA
from pyod.models.abod import ABOD

from analytics.lof import LOF
from analytics.base import BaseDetector

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


class PCADetector(BaseDetector):
    def __init__(self, detector):
        self.detector = detector
        self.pca = None

    def get_name(self):
        return f"PCA({self.detector.get_name()})"

    def fit(self, data, n_components=3, **params):
        self.pca = PCA(n_components=n_components)
        transformed = self.pca.fit_transform(data)
        self.detector.fit(transformed, **params)
        return self

    def score(self, data):
        transformed = self.pca.transform(data)
        return self.detector.score(transformed)


class TSNEDetector(BaseDetector):
    def __init__(self, detector):
        self.detector = detector
        self.tsne = None

    def get_name(self):
        return f"TSNE({self.detector.get_name()})"

    def fit(self, data, n_components=3, **params):
        self.tsne = TSNE(n_components=n_components)
        transformed = self.tsne.fit_transform(data)
        self.detector.fit(transformed, **params)
        return self

    def score(self, data):
        transformed = self.tsne.transform(data)
        return self.detector.score(transformed)


class IsolationForestDetector(BaseDetector):
    def get_name(self):
        return "Isolation Forest"

    def fit(self, data, **params):
        self.model = IsolationForest(verbose=1, **params)
        self.model.fit(data)
        return self

    def score(self, data):
        return self.model.decision_function(data)


class LOFDetector(BaseDetector):
    def get_name(self):
        return "Local Outlier Factor"

    def fit(self, data, normalize=False, **params):
        X = [tuple(x) for x in data.to_records(index=False)]
        self.lof = LOF(X, normalize=normalize)
        self.data = X
        self.min_pts = params.get("min_pts", 3)
        return self

    def score(self, data):
        return [
            -self.lof.local_outlier_factor(self.min_pts, tuple(self.data[i]))
            for i in range(len(self.data))
        ]


class SOSDetector(BaseDetector):
    def get_name(self):
        return "Stochastic Outlier Selection"

    def fit(self, data, **params):
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
    def get_name(self):
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
        scores = [knn_scores, sos_scores, hbos_scores]

        norm_scores = []
        for score in scores:
            arr = np.array(score)
            arr /= arr.max()
            norm_scores.append(arr)

        return -np.sum(np.array(norm_scores), axis=0)


class HBOSDetector(BaseDetector):
    def get_name(self):
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
        scores = []
        for i in range(data.shape[0]):
            record = data[i, :]
            s = 0
            for j in range(len(record)):
                histogram = self.histograms[j]
                for k in range(len(histogram[1]) - 1):
                    if histogram[1][k] <= record[j] < histogram[1][k + 1]:
                        s += np.log(histogram[0][k])
                        break
            scores.append(s)
        return scores


class KNNDetector(BaseDetector):
    def get_name(self):
        return "K-Nearest Neighbors"

    def fit(self, data, **params):
        self.k = params.get("k", 3)
        self.neigh = NearestNeighbors(n_neighbors=self.k + 1).fit(data)
        return self

    def score(self, data):
        distances, _ = self.neigh.kneighbors(data)
        return -np.sum(distances, axis=1)


class OneClassSVMDetector(BaseDetector):
    def get_name(self):
        return "One-Class SVM"

    def fit(self, data, **params):
        self.model = OneClassSVM(**params)
        self.model.fit(data)
        return self

    def score(self, data):
        return self.model.decision_function(data)


class DBSCANDetector(BaseDetector):
    def get_name(self):
        return "DBSCAN"

    def fit(self, data, **params):
        self.model = DBSCAN(**params)
        self.model.fit(data)
        return self

    def score(self, data):
        labels = self.model.fit_predict(data)
        return np.where(labels == -1, 1.0, 0.0)


class EllipticEnvelopeDetector(BaseDetector):
    def get_name(self):
        return "Elliptic Envelope"

    def fit(self, data, **params):
        self.model = EllipticEnvelope(**params)
        self.model.fit(data)
        return self

    def score(self, data):
        return self.model.decision_function(data)


class GaussianMixtureDetector(BaseDetector):
    def get_name(self):
        return "Gaussian Mixture"

    def fit(self, data, **params):
        self.model = GaussianMixture(**params)
        self.model.fit(data)
        return self

    def score(self, data):
        return -self.model.score_samples(data)


class SklearnLOFDetector(BaseDetector):
    def get_name(self):
        return "Sklearn LOF"

    def fit(self, data, **params):
        self.model = LocalOutlierFactor(novelty=True, **params)
        self.model.fit(data)
        return self

    def score(self, data):
        return self.model.decision_function(data)


class KMeansDetector(BaseDetector):
    def get_name(self):
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
    def get_name(self):
        return "PCA Reconstruction"

    def fit(self, data, **params):
        self.n_components = params.pop("n_components", 0.95)
        self.pca = PCA(n_components=self.n_components)
        transformed = self.pca.fit_transform(data)
        self.reconstructed = self.pca.inverse_transform(transformed)
        self.data = data
        return self

    def score(self, data):
        errors = np.linalg.norm(self.data - self.reconstructed, axis=1)
        return -errors


class MahalanobisDetector(BaseDetector):
    def get_name(self):
        return "Mahalanobis"

    def fit(self, data, **params):
        self.cov = EmpiricalCovariance(**params)
        self.cov.fit(data)
        return self

    def score(self, data):
        distances = self.cov.mahalanobis(data)
        return -distances


class KDEDetector(BaseDetector):
    def get_name(self):
        return "Kernel Density"

    def fit(self, data, **params):
        self.kde = KernelDensity(**params)
        self.kde.fit(data)
        return self

    def score(self, data):
        return self.kde.score_samples(data)


class AutoencoderDetector(BaseDetector):
    def get_name(self):
        return "Autoencoder"

    def fit(self, data, **params):
        hidden_layer_sizes = params.pop("hidden_layer_sizes", (32, 32, 32))
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=2000)
        self.model.fit(data, data)
        self.data = data
        return self

    def score(self, data):
        reconstructed = self.model.predict(self.data)
        errors = np.linalg.norm(self.data - reconstructed, axis=1)
        return -errors


class COPODDetector(BaseDetector):
    def get_name(self):
        return "COPOD"

    def fit(self, data, **params):
        self.model = COPOD(**params)
        self.X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(self.X)
        return self

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else data
        return -self.model.decision_function(X)


class FeatureBaggingDetector(BaseDetector):
    def get_name(self):
        return "Feature Bagging"

    def fit(self, data, **params):
        self.model = FeatureBagging(**params)
        self.X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(self.X)
        return self

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else data
        return -self.model.decision_function(X)


class LODADetector(BaseDetector):
    def get_name(self):
        return "LODA"

    def fit(self, data, **params):
        self.model = LODA(**params)
        self.X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(self.X)
        return self

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else data
        return -self.model.decision_function(X)


class ABODDetector(BaseDetector):
    def get_name(self):
        return "ABOD"

    def fit(self, data, **params):
        self.model = ABOD(**params)
        self.X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(self.X)
        return self

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else data
        return -self.model.decision_function(X)


class DenoisingAutoencoderDetector(BaseDetector):
    def get_name(self):
        return "Denoising Autoencoder"

    def fit(self, data, noise_level=0.1, **params):
        hidden = params.pop("hidden_layer_sizes", (32, 32, 32))
        noisy = data + noise_level * np.random.normal(size=data.shape)
        self.model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=2000)
        self.model.fit(noisy, data)
        self.data = data
        return self

    def score(self, data):
        reconstructed = self.model.predict(self.data)
        errors = np.linalg.norm(self.data - reconstructed, axis=1)
        return -errors


class VariationalAutoencoderDetector(BaseDetector):
    def get_name(self):
        return "Variational Autoencoder"

    def fit(self, data, **params):
        try:
            from pyod.models.vae import VAE
        except Exception as e:  # pragma: no cover - dependency optional
            raise ImportError(
                "VariationalAutoencoderDetector requires pyod and TensorFlow"
            ) from e
        self.X = data.values if isinstance(data, pd.DataFrame) else data
        self.model = VAE(**params)
        self.model.fit(self.X)
        return self

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else data
        return -self.model.decision_function(X)


class LSTMAutoencoderDetector(BaseDetector):
    def get_name(self):
        return "LSTM Autoencoder"

    def fit(self, data, epochs=10, hidden_size=8, lr=1e-3, **params):
        try:
            import torch
            from torch import nn, optim
        except Exception as e:  # pragma: no cover - dependency optional
            raise ImportError("LSTMAutoencoderDetector requires PyTorch") from e
        X = data.values if isinstance(data, pd.DataFrame) else data
        tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

        class LSTMAE(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

            def forward(self, x):
                _, (h, _) = self.encoder(x)
                dec_input = h.repeat(x.size(1), 1, 1).transpose(0, 1)
                out, _ = self.decoder(dec_input)
                return out

        model = LSTMAE(tensor.size(-1), hidden_size)
        optimiz = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            optimiz.zero_grad()
            output = model(tensor)
            loss = loss_fn(output, tensor)
            loss.backward()
            optimiz.step()

        self.reconstructed = model(tensor).detach().numpy()
        self.X = tensor.detach().numpy()
        return self

    def score(self, data):
        errors = np.linalg.norm(self.X - self.reconstructed, axis=(1, 2))
        return -errors


class TransformerDetector(BaseDetector):
    def get_name(self):
        return "Transformer"

    def fit(self, data, epochs=10, d_model=16, nhead=2, lr=1e-3, **params):
        try:
            import torch
            from torch import nn, optim
        except Exception as e:  # pragma: no cover - dependency optional
            raise ImportError("TransformerDetector requires PyTorch") from e
        X = data.values if isinstance(data, pd.DataFrame) else data
        tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

        class TransAE(nn.Module):
            def __init__(self, input_dim, d_model, nhead):
                super().__init__()
                self.input = nn.Linear(input_dim, d_model)
                enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
                dec_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
                self.decoder = nn.TransformerDecoder(dec_layer, num_layers=1)
                self.output = nn.Linear(d_model, input_dim)

            def forward(self, x):
                z = self.input(x)
                h = self.encoder(z)
                dec = self.decoder(h, h)
                return self.output(dec)

        model = TransAE(tensor.size(-1), d_model, nhead)
        optimiz = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            optimiz.zero_grad()
            output = model(tensor)
            loss = loss_fn(output, tensor)
            loss.backward()
            optimiz.step()

        self.reconstructed = model(tensor).detach().numpy()
        self.X = tensor.detach().numpy()
        return self

    def score(self, data):
        errors = np.linalg.norm(self.X - self.reconstructed, axis=(1, 2))
        return -errors
