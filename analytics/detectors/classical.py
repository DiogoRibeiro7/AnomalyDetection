"""Classical anomaly detectors built on scikit-learn and PyOD.

Each detector exposes a :class:`~analytics.base.BaseDetector` interface with
``fit`` and ``score`` methods.  The implementations avoid importing optional
dependencies until required to keep the package lightweight.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, ClassVar, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from analytics.base import BaseDetector, coerce_tabular_2d
from analytics.lof import LOF

from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor, KernelDensity
from sklearn.svm import OneClassSVM


ArrayLike = NDArray[np.floating[Any]]
FrameOrArray = Union[pd.DataFrame, ArrayLike]
ScoreArray = NDArray[np.floating[Any]]


def _coerce_frame_or_array(data: FrameOrArray) -> ArrayLike:
    """Return a 2-D numpy array representation of *data*.

    This helper centralises the conversion logic used by the classical
    detectors so that future additions inherit consistent validation behaviour
    regardless of whether a :class:`pandas.DataFrame` or ``numpy.ndarray`` is
    supplied.  All classical detectors operate on dense floating point arrays,
    therefore inputs are promoted to ``float`` and verified to be two
    dimensional.
    """

    return coerce_tabular_2d(data)


def _compute_variable_width_edges(feature: ArrayLike, k: int) -> ArrayLike:
    """Return bin edges so each bin has roughly equal frequency."""
    edges = np.percentile(feature, np.linspace(0, 100, k + 1))
    edges = np.unique(edges)
    if len(edges) - 1 < k:
        edges = np.histogram_bin_edges(feature, bins=k)
    return edges


class IsolationForestDetector(BaseDetector):
    """Wrapper around scikit-learn's :class:`IsolationForest`.

    Args:
        None: Instances are created without constructor parameters. Provide
            estimator options when calling :meth:`fit`.

    Returns:
        IsolationForestDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: Decision function values returned by :meth:`score`.

    Attributes:
        model: The fitted ``IsolationForest`` estimator once :meth:`fit` is
            called.

    Examples:
        >>> detector = IsolationForestDetector()
        >>> _ = detector.fit(X_train)
        >>> scores = detector.score(X_test)
        >>> scores.shape
        (len(X_test),)
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "Isolation Forest"

    def fit(self, data: FrameOrArray, **params: Any) -> IsolationForestDetector:
        """Fit the isolation forest model to the provided data.

        Args:
            data (pandas.DataFrame or numpy.ndarray): A 2D array-like object
                containing the training observations, shaped ``(n_samples,
                n_features)``.
            **params: Additional keyword arguments forwarded to
                :class:`sklearn.ensemble.IsolationForest`.

        Returns:
            IsolationForestDetector: The fitted detector instance.

        Examples:
            >>> detector = IsolationForestDetector()
            >>> detector.fit(X_train, n_estimators=200)
            IsolationForestDetector(...)
        """
        self.model = IsolationForest(verbose=1, **params)
        X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(X)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Compute anomaly scores for new observations.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples shaped
                ``(n_samples, n_features)`` to evaluate with the fitted
                estimator.

        Returns:
            numpy.ndarray: The isolation forest decision function values where
            higher scores indicate more normal observations.

        Examples:
            >>> detector = IsolationForestDetector().fit(X_train)
            >>> detector.score(X_test)  # doctest: +SKIP
            array([...])
        """
        X = data.values if isinstance(data, pd.DataFrame) else data
        return np.asarray(self.model.decision_function(X))


class _PyODAdapter(BaseDetector):
    """Common adapter implementing :class:`BaseDetector` for PyOD models."""

    pyod_class_path: ClassVar[str]
    display_name: ClassVar[str]
    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return self.display_name

    def _load_pyod_class(self):
        module_path, class_name = self.pyod_class_path.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)

    def fit(self, data: FrameOrArray, **params: Any) -> _PyODAdapter:
        """Fit the wrapped PyOD estimator on the provided data."""

        pyod_cls = self._load_pyod_class()
        self.model = pyod_cls(**params)
        X = _coerce_frame_or_array(data)
        self.model.fit(X)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Return negative PyOD decision scores for compatibility."""

        X = _coerce_frame_or_array(data)
        scores = np.asarray(self.model.decision_function(X), dtype=float)
        return -scores


class LOFDetector(BaseDetector):
    """Local Outlier Factor implementation allowing new data scoring.

    Args:
        None: Instances require no constructor parameters; pass LOF settings to
            :meth:`fit`.

    Returns:
        LOFDetector: The fitted detector returned by :meth:`fit`.
        list[float]: Local outlier factor scores produced by :meth:`score`.

    Attributes:
        lof: The :class:`analytics.lof.LOF` instance trained during
            :meth:`fit`.
        min_pts: Minimum number of neighbors used when computing the local
            outlier factor.

    Examples:
        >>> detector = LOFDetector()
        >>> _ = detector.fit(X_train_df, min_pts=5)
        >>> detector.score(X_test_df)[0]  # doctest: +SKIP
        -0.23
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "Local Outlier Factor"

    def fit(
        self, data: pd.DataFrame, normalize: bool = False, **params: Any
    ) -> LOFDetector:
        """Compute the LOF model for the given tabular data.

        Args:
            data (pandas.DataFrame): A DataFrame containing the observations
                used for training. Column order is preserved in the internal
                representation.
            normalize (bool, optional): Whether to apply LOF's internal
                normalization routine. Defaults to ``False``.
            **params: Additional parameters such as ``min_pts`` forwarded to
                :class:`analytics.lof.LOF`.

        Returns:
            LOFDetector: The fitted detector instance.

        Examples:
            >>> detector = LOFDetector()
            >>> detector.fit(X_train_df, normalize=True, min_pts=10)
            LOFDetector(...)
        """
        X = [tuple(x) for x in data.to_records(index=False)]
        self.lof = LOF(X, normalize=normalize)
        self.min_pts = params.get("min_pts", 3)
        return self

    def score(self, data: pd.DataFrame) -> list[float]:
        """Evaluate new observations using the trained LOF model.

        Args:
            data (pandas.DataFrame): A DataFrame shaped ``(n_samples,
                n_features)`` with the same schema as the training data.

        Returns:
            list[float]: Negative local outlier factor values where larger
                values correspond to less anomalous points.

        Examples:
            >>> detector = LOFDetector().fit(X_train_df, min_pts=5)
            >>> detector.score(X_test_df)[:3]  # doctest: +SKIP
            [-0.2, -0.5, -0.1]
        """
        X = [tuple(x) for x in data.to_records(index=False)]
        return [-self.lof.local_outlier_factor(self.min_pts, point) for point in X]


class SOSDetector(BaseDetector):
    """Stochastic Outlier Selection using the ``sksos`` package.

    Args:
        None: Constructed without arguments; configure hyper-parameters via
            :meth:`fit`.

    Returns:
        SOSDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: SOS anomaly probabilities returned by :meth:`score`.

    Attributes:
        model: The wrapped :class:`sksos.SOS` estimator.
        X: Cached training data used when scoring without new data.

    Examples:
        >>> detector = SOSDetector()
        >>> _ = detector.fit(X_train_df, perplexity=15)
        >>> detector.score()[:3]  # doctest: +SKIP
        array([...])
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "Stochastic Outlier Selection"

    def fit(self, data: FrameOrArray, **params: Any) -> SOSDetector:
        """Fit the SOS model on dense feature data.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Training samples of shape
                ``(n_samples, n_features)``.
            **params: Optional SOS hyper-parameters such as ``perplexity`` or
                ``metric``.

        Returns:
            SOSDetector: The fitted detector instance.

        Examples:
            >>> detector = SOSDetector()
            >>> detector.fit(X_train_df, perplexity=50, metric="cosine")
            SOSDetector(...)
        """
        from sksos import SOS  # lazy import

        perplexity = params.get("perplexity", 30)
        metric = params.get("metric", "euclidean")
        eps = params.get("eps", 1e-5)
        self.model = SOS(perplexity=perplexity, metric=metric, eps=eps)
        self.X = data.values if isinstance(data, pd.DataFrame) else data
        return self

    def score(self, data: FrameOrArray | None = None) -> ScoreArray:
        """Score data using the fitted SOS model.

        Args:
            data (pandas.DataFrame or numpy.ndarray, optional): Samples to
                score. If ``None``, the training data provided to
                :meth:`fit` is used.

        Returns:
            numpy.ndarray: Negative SOS probabilities where larger values
                indicate more anomalous observations.

        Examples:
            >>> detector = SOSDetector().fit(X_train_df)
            >>> detector.score(X_test_df)  # doctest: +SKIP
            array([...])
        """
        X = (
            self.X
            if data is None
            else (data.values if isinstance(data, pd.DataFrame) else data)
        )
        return -np.asarray(self.model.predict(X))


class EnsembleDetector(BaseDetector):
    """Simple ensemble averaging KNN, SOS and HBOS scores.

    Args:
        None: Instances are parameter-free; supply detector configurations when
            calling :meth:`fit`.

    Returns:
        EnsembleDetector: The fitted ensemble returned by :meth:`fit`.
        numpy.ndarray: Aggregated anomaly scores returned by :meth:`score`.

    Attributes:
        knn: Fitted :class:`KNNDetector` component.
        sos: Fitted :class:`SOSDetector` component.
        hbos: Fitted :class:`HBOSDetector` component.

    Examples:
        >>> detector = EnsembleDetector()
        >>> _ = detector.fit(X_train)
        >>> detector.score(X_test)  # doctest: +SKIP
        array([...])
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "Ensembled detector"

    def fit(self, data: FrameOrArray, **params: Any) -> EnsembleDetector:
        """Train the constituent detectors and cache them for scoring.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Training data matrix of
                shape ``(n_samples, n_features)``.
            **params: Keyword arguments forwarded to each base detector.

        Returns:
            EnsembleDetector: The fitted ensemble instance.

        Examples:
            >>> detector = EnsembleDetector()
            >>> detector.fit(X_train, k=10)
            EnsembleDetector(...)
        """
        self.knn = KNNDetector().fit(data, **params)
        self.sos = SOSDetector().fit(data, **params)
        self.hbos = HBOSDetector().fit(data, **params)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Combine component detector scores by normalized averaging.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples to be scored,
                shaped ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Aggregated anomaly scores where lower values imply
                more anomalous points.

        Examples:
            >>> detector = EnsembleDetector().fit(X_train)
            >>> detector.score(X_test)[:5]  # doctest: +SKIP
            array([...])
        """
        knn_scores = self.knn.score(data)
        sos_scores = self.sos.score(data)
        hbos_scores = self.hbos.score(data)
        scores = np.vstack([knn_scores, sos_scores, hbos_scores])
        scores /= scores.max(axis=1, keepdims=True)
        return -scores.sum(axis=0)


class HBOSDetector(BaseDetector):
    """Histogram-Based Outlier Score with vectorized scoring.

    Args:
        None: Instantiate without arguments and configure via :meth:`fit`.

    Returns:
        HBOSDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: Log-density scores produced by :meth:`score`.

    Attributes:
        edges_left: Left bin edges for each feature.
        edges_right: Right bin edges for each feature.
        hist: Density estimates for each histogram bin.
        log_hist: Log-density used for additive scoring.

    Examples:
        >>> detector = HBOSDetector()
        >>> _ = detector.fit(X_train)
        >>> detector.score(X_test)  # doctest: +SKIP
        array([...])
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "Histogram-Based Outlier Score"

    def fit(self, data: FrameOrArray, **params: Any) -> HBOSDetector:
        """Estimate per-feature histograms with adaptive binning.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Training samples in a 2D
                structure of shape ``(n_samples, n_features)``.
            **params: Additional configuration such as ``k`` for the number of
                bins.

        Returns:
            HBOSDetector: The fitted detector instance.

        Examples:
            >>> detector = HBOSDetector()
            >>> detector.fit(X_train, k=5)
            HBOSDetector(...)
        """
        k = params.get("k", 3)
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        n_features = data.shape[1]
        self.edges_left = np.zeros((n_features, k))
        self.edges_right = np.zeros((n_features, k))
        self.hist = np.zeros((n_features, k))
        for i in range(n_features):
            feature = data[:, i]
            edges = _compute_variable_width_edges(feature, k)
            hist, _ = np.histogram(feature, bins=edges, density=True)
            self.hist[i] = hist
            self.edges_left[i] = edges[:-1]
            right = edges[1:]
            right[-1] = np.inf
            self.edges_right[i] = right
        self.log_hist = np.log(self.hist + 1e-12)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Evaluate the log-density based HBOS score for each sample.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples compatible with
                the histogram features used during fitting.

        Returns:
            numpy.ndarray: Log-probabilities where lower values indicate
                anomalies.

        Examples:
            >>> detector = HBOSDetector().fit(X_train)
            >>> detector.score(X_test)[:2]  # doctest: +SKIP
            array([-12.5, -10.7])
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        mask = (data[:, :, None] >= self.edges_left[None, :, :]) & (
            data[:, :, None] < self.edges_right[None, :, :]
        )
        log_probs = (mask * self.log_hist[None, :, :]).sum(axis=2)
        return log_probs.sum(axis=1)


class KNNDetector(BaseDetector):
    """K-Nearest Neighbors distance-based detector.

    Args:
        None: The detector is created without constructor parameters; set
            ``k`` and other options via :meth:`fit`.

    Returns:
        KNNDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: Summed distance scores returned by :meth:`score`.

    Attributes:
        k: Number of neighbors considered for scoring.
        neigh: Fitted :class:`sklearn.neighbors.NearestNeighbors` estimator.

    Examples:
        >>> detector = KNNDetector()
        >>> _ = detector.fit(X_train)
        >>> detector.score(X_test)  # doctest: +SKIP
        array([...])
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "K-Nearest Neighbors"

    def fit(self, data: FrameOrArray, **params: Any) -> KNNDetector:
        """Fit the nearest neighbor index on the training data.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Training matrix with
                shape ``(n_samples, n_features)``.
            **params: Optional parameters, including ``k`` for the number of
                neighbors.

        Returns:
            KNNDetector: The fitted detector instance.

        Examples:
            >>> detector = KNNDetector()
            >>> detector.fit(X_train, k=10)
            KNNDetector(...)
        """
        self.k = params.get("k", 3)
        X = data.values if isinstance(data, pd.DataFrame) else data
        self.neigh = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Score samples based on summed neighbor distances.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples to evaluate with
                shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Negative summed distances, where smaller values
                indicate potential anomalies.

        Examples:
            >>> detector = KNNDetector().fit(X_train, k=5)
            >>> detector.score(X_test)[:3]  # doctest: +SKIP
            array([-4.2, -3.1, -5.0])
        """
        X = data.values if isinstance(data, pd.DataFrame) else data
        distances, _ = self.neigh.kneighbors(X)
        return -np.sum(distances, axis=1)


class OneClassSVMDetector(BaseDetector):
    """Wrapper around :class:`sklearn.svm.OneClassSVM`.

    Args:
        None: Instances require no constructor parameters; configure the SVM via
            :meth:`fit`.

    Returns:
        OneClassSVMDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: Decision function scores produced by :meth:`score`.

    Attributes:
        model: The fitted one-class SVM estimator.

    Examples:
        >>> detector = OneClassSVMDetector()
        >>> _ = detector.fit(X_train)
        >>> detector.score(X_test)  # doctest: +SKIP
        array([...])
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "One-Class SVM"

    def fit(self, data: FrameOrArray, **params: Any) -> OneClassSVMDetector:
        """Train a one-class SVM on the input data.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Training samples with
                shape ``(n_samples, n_features)``.
            **params: Keyword arguments for :class:`sklearn.svm.OneClassSVM`.

        Returns:
            OneClassSVMDetector: The fitted detector instance.

        Examples:
            >>> detector = OneClassSVMDetector()
            >>> detector.fit(X_train, kernel="rbf", gamma=0.1)
            OneClassSVMDetector(...)
        """
        self.model = OneClassSVM(**params)
        X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(X)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Compute signed distance to the SVM decision boundary.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples to score with
                shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Decision function scores where larger values denote
                inliers.

        Examples:
            >>> detector = OneClassSVMDetector().fit(X_train)
            >>> detector.score(X_test)  # doctest: +SKIP
            array([...])
        """
        X = data.values if isinstance(data, pd.DataFrame) else data
        return np.asarray(self.model.decision_function(X))


class DBSCANDetector(BaseDetector):
    """Density-based spatial clustering anomaly detector.

    Args:
        None: Instantiate without arguments; configure clustering options in
            :meth:`fit`.

    Returns:
        DBSCANDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: Binary anomaly indicators returned by :meth:`score`.

    Attributes:
        model: The :class:`sklearn.cluster.DBSCAN` estimator fitted to the
            training data.

    Examples:
        >>> detector = DBSCANDetector()
        >>> _ = detector.fit(X_train, eps=0.5)
        >>> detector.score(X_test)  # doctest: +SKIP
        array([...])
    """

    score_orientation = "binary_anomaly"

    def get_name(self) -> str:
        return "DBSCAN"

    def fit(self, data: FrameOrArray, **params: Any) -> DBSCANDetector:
        """Cluster the training data using DBSCAN.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Observations of shape
                ``(n_samples, n_features)``.
            **params: Keyword arguments forwarded to
                :class:`sklearn.cluster.DBSCAN`.

        Returns:
            DBSCANDetector: The fitted detector instance.

        Examples:
            >>> detector = DBSCANDetector()
            >>> detector.fit(X_train, eps=0.8, min_samples=5)
            DBSCANDetector(...)
        """
        self.model = DBSCAN(**params)
        X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(X)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Assign anomaly labels based on DBSCAN clustering.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples to cluster, shaped
                ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Binary scores where ``1.0`` represents an outlier and
                ``0.0`` an inlier.

        Examples:
            >>> detector = DBSCANDetector().fit(X_train, eps=0.6)
            >>> detector.score(X_test)[:4]  # doctest: +SKIP
            array([0., 1., 0., 0.])
        """
        X = data.values if isinstance(data, pd.DataFrame) else data
        labels = self.model.fit_predict(X)
        return np.where(labels == -1, 1.0, 0.0)


class EllipticEnvelopeDetector(BaseDetector):
    """Robust covariance estimate assuming Gaussian distributed data.

    Args:
        None: Instances are created without arguments; pass covariance options
            via :meth:`fit`.

    Returns:
        EllipticEnvelopeDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: Decision scores produced by :meth:`score`.

    Attributes:
        model: The fitted :class:`sklearn.covariance.EllipticEnvelope`
            estimator.

    Examples:
        >>> detector = EllipticEnvelopeDetector()
        >>> _ = detector.fit(X_train)
        >>> detector.score(X_test)  # doctest: +SKIP
        array([...])
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "Elliptic Envelope"

    def fit(self, data: FrameOrArray, **params: Any) -> EllipticEnvelopeDetector:
        """Estimate a robust covariance model for Gaussian-like data.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Training matrix of shape
                ``(n_samples, n_features)``.
            **params: Parameters to initialize
                :class:`sklearn.covariance.EllipticEnvelope`.

        Returns:
            EllipticEnvelopeDetector: The fitted detector instance.

        Examples:
            >>> detector = EllipticEnvelopeDetector()
            >>> detector.fit(X_train, contamination=0.1)
            EllipticEnvelopeDetector(...)
        """
        self.model = EllipticEnvelope(**params)
        X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(X)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Compute distances to the robust covariance contour.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples to score shaped
                ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Decision function values where larger scores signify
                more typical observations.

        Examples:
            >>> detector = EllipticEnvelopeDetector().fit(X_train)
            >>> detector.score(X_test)  # doctest: +SKIP
            array([...])
        """
        X = data.values if isinstance(data, pd.DataFrame) else data
        return np.asarray(self.model.decision_function(X))


class GaussianMixtureDetector(BaseDetector):
    """Gaussian Mixture negative log-likelihood as anomaly score.

    Args:
        None: Instantiate without parameters; provide mixture settings via
            :meth:`fit`.

    Returns:
        GaussianMixtureDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: Negative log-likelihood scores returned by :meth:`score`.

    Attributes:
        model: The fitted :class:`sklearn.mixture.GaussianMixture` estimator.

    Examples:
        >>> detector = GaussianMixtureDetector()
        >>> _ = detector.fit(X_train, n_components=3)
        >>> detector.score(X_test)  # doctest: +SKIP
        array([...])
    """

    score_orientation = "higher_is_more_anomalous"

    def get_name(self) -> str:
        return "Gaussian Mixture"

    def fit(self, data: FrameOrArray, **params: Any) -> GaussianMixtureDetector:
        """Fit a Gaussian mixture model to the training samples.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Training data shaped
                ``(n_samples, n_features)``.
            **params: Parameters passed to
                :class:`sklearn.mixture.GaussianMixture`.

        Returns:
            GaussianMixtureDetector: The fitted detector instance.

        Examples:
            >>> detector = GaussianMixtureDetector()
            >>> detector.fit(X_train, n_components=2, covariance_type="diag")
            GaussianMixtureDetector(...)
        """
        self.model = GaussianMixture(**params)
        X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(X)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Return negative log-likelihood scores for the provided data.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples shaped
                ``(n_samples, n_features)`` to evaluate with the fitted model.

        Returns:
            numpy.ndarray: Negative log probabilities where larger values imply
                more anomalous points.

        Examples:
            >>> detector = GaussianMixtureDetector().fit(X_train)
            >>> detector.score(X_test)[:4]  # doctest: +SKIP
            array([...])
        """
        X = data.values if isinstance(data, pd.DataFrame) else data
        return -self.model.score_samples(X)


class SklearnLOFDetector(BaseDetector):
    """Scikit-learn's LOF with novelty mode for scoring new data.

    Args:
        None: Instances do not require constructor parameters; pass LOF options
            to :meth:`fit`.

    Returns:
        SklearnLOFDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: Novelty detection scores produced by :meth:`score`.

    Attributes:
        model: The :class:`sklearn.neighbors.LocalOutlierFactor` estimator
            configured with ``novelty=True``.

    Examples:
        >>> detector = SklearnLOFDetector()
        >>> _ = detector.fit(X_train)
        >>> detector.score(X_test)  # doctest: +SKIP
        array([...])
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "Sklearn LOF"

    def fit(self, data: FrameOrArray, **params: Any) -> SklearnLOFDetector:
        """Train scikit-learn's LOF implementation in novelty mode.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Training samples shaped
                ``(n_samples, n_features)``.
            **params: Additional parameters for
                :class:`sklearn.neighbors.LocalOutlierFactor`.

        Returns:
            SklearnLOFDetector: The fitted detector instance.

        Examples:
            >>> detector = SklearnLOFDetector()
            >>> detector.fit(X_train, n_neighbors=40)
            SklearnLOFDetector(...)
        """
        self.model = LocalOutlierFactor(novelty=True, **params)
        X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(X)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Score samples using the LOF decision function.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples for evaluation of
                shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Signed LOF scores where higher values indicate less
                anomalous observations.

        Examples:
            >>> detector = SklearnLOFDetector().fit(X_train)
            >>> detector.score(X_test)  # doctest: +SKIP
            array([...])
        """
        X = data.values if isinstance(data, pd.DataFrame) else data
        return np.asarray(self.model.decision_function(X))


class KMeansDetector(BaseDetector):
    """Distance to nearest KMeans centroid as anomaly score.

    Args:
        None: Construct instances without parameters; choose ``n_clusters`` via
            :meth:`fit`.

    Returns:
        KMeansDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: Negative centroid distances from :meth:`score`.

    Attributes:
        n_clusters: Number of centroids used during fitting.
        kmeans: The fitted :class:`sklearn.cluster.KMeans` estimator.

    Examples:
        >>> detector = KMeansDetector()
        >>> _ = detector.fit(X_train, n_clusters=5)
        >>> detector.score(X_test)  # doctest: +SKIP
        array([...])
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "KMeans"

    def fit(self, data: FrameOrArray, **params: Any) -> KMeansDetector:
        """Train KMeans on the provided data.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Observations shaped
                ``(n_samples, n_features)``.
            **params: Optional KMeans parameters, such as ``n_clusters``.

        Returns:
            KMeansDetector: The fitted detector instance.

        Examples:
            >>> detector = KMeansDetector()
            >>> detector.fit(X_train, n_clusters=3)
            KMeansDetector(...)
        """
        self.n_clusters = params.pop("n_clusters", 8)
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        X = data.values if isinstance(data, pd.DataFrame) else data
        self.kmeans.fit(X)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Score samples by their distance to the closest centroid.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples shaped
                ``(n_samples, n_features)`` consistent with the training data.

        Returns:
            numpy.ndarray: Negative minimum distances where lower values denote
                more anomalous points.

        Examples:
            >>> detector = KMeansDetector().fit(X_train, n_clusters=4)
            >>> detector.score(X_test)[:3]  # doctest: +SKIP
            array([-0.4, -0.7, -1.2])
        """
        X = data.values if isinstance(data, pd.DataFrame) else data
        distances = self.kmeans.transform(X)
        min_dist = np.min(distances, axis=1)
        return -min_dist


class PCAReconstructionDetector(BaseDetector):
    """Use PCA reconstruction error as anomaly score.

    Args:
        None: Instantiate without arguments; configure PCA via :meth:`fit`.

    Returns:
        PCAReconstructionDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: Negative reconstruction errors from :meth:`score`.

    Attributes:
        n_components: Number (or fraction) of principal components retained.
        pca: The fitted :class:`sklearn.decomposition.PCA` transformer.

    Examples:
        >>> detector = PCAReconstructionDetector()
        >>> _ = detector.fit(X_train)
        >>> detector.score(X_test)  # doctest: +SKIP
        array([...])
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "PCA Reconstruction"

    def fit(self, data: FrameOrArray, **params: Any) -> PCAReconstructionDetector:
        """Fit PCA to approximate the training data.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Training samples with
                shape ``(n_samples, n_features)``.
            **params: Optional PCA parameters including ``n_components``.

        Returns:
            PCAReconstructionDetector: The fitted detector instance.

        Examples:
            >>> detector = PCAReconstructionDetector()
            >>> detector.fit(X_train, n_components=0.9)
            PCAReconstructionDetector(...)
        """
        self.n_components = params.pop("n_components", 0.95)
        X = data.values if isinstance(data, pd.DataFrame) else data
        self.pca = PCA(n_components=self.n_components).fit(X)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Compute negative reconstruction error for each sample.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples shaped
                ``(n_samples, n_features)`` to evaluate.

        Returns:
            numpy.ndarray: Negative L2 reconstruction errors where smaller
                values indicate more anomalous observations.

        Examples:
            >>> detector = PCAReconstructionDetector().fit(X_train)
            >>> detector.score(X_test)[:5]  # doctest: +SKIP
            array([-0.2, -1.4, ...])
        """
        X = data.values if isinstance(data, pd.DataFrame) else data
        transformed = self.pca.transform(X)
        reconstructed = self.pca.inverse_transform(transformed)
        errors = np.linalg.norm(X - reconstructed, axis=1)
        return -errors


class MahalanobisDetector(BaseDetector):
    """Mahalanobis distance using empirical covariance.

    Args:
        None: Create detector without parameters; configure covariance options
            via :meth:`fit`.

    Returns:
        MahalanobisDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: Negative distance scores from :meth:`score`.

    Attributes:
        cov: The fitted :class:`sklearn.covariance.EmpiricalCovariance`
            estimator.

    Examples:
        >>> detector = MahalanobisDetector()
        >>> _ = detector.fit(X_train)
        >>> detector.score(X_test)  # doctest: +SKIP
        array([...])
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "Mahalanobis"

    def fit(self, data: FrameOrArray, **params: Any) -> MahalanobisDetector:
        """Estimate the covariance matrix for Mahalanobis scoring.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Training samples shaped
                ``(n_samples, n_features)``.
            **params: Optional arguments passed to
                :class:`sklearn.covariance.EmpiricalCovariance`.

        Returns:
            MahalanobisDetector: The fitted detector instance.

        Examples:
            >>> detector = MahalanobisDetector()
            >>> detector.fit(X_train)
            MahalanobisDetector(...)
        """
        self.cov = EmpiricalCovariance(**params)
        X = data.values if isinstance(data, pd.DataFrame) else data
        self.cov.fit(X)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Compute negative Mahalanobis distance for each sample.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples to evaluate with
                shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Negative distances where smaller (more negative)
                values indicate greater anomaly likelihood.

        Examples:
            >>> detector = MahalanobisDetector().fit(X_train)
            >>> detector.score(X_test)[:3]  # doctest: +SKIP
            array([-5.3, -4.1, -3.0])
        """
        X = data.values if isinstance(data, pd.DataFrame) else data
        distances = self.cov.mahalanobis(X)
        return -distances


class KDEDetector(BaseDetector):
    """Kernel Density Estimator returning log-density scores.

    Args:
        None: Instantiate without parameters; set KDE options via :meth:`fit`.

    Returns:
        KDEDetector: The fitted detector returned by :meth:`fit`.
        numpy.ndarray: Log-density scores returned by :meth:`score`.

    Attributes:
        kde: The fitted :class:`sklearn.neighbors.KernelDensity` estimator.

    Examples:
        >>> detector = KDEDetector()
        >>> _ = detector.fit(X_train)
        >>> detector.score(X_test)  # doctest: +SKIP
        array([...])
    """

    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "Kernel Density"

    def fit(self, data: FrameOrArray, **params: Any) -> KDEDetector:
        """Fit a kernel density estimator to the training data.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Training observations with
                shape ``(n_samples, n_features)``.
            **params: Keyword arguments for
                :class:`sklearn.neighbors.KernelDensity`.

        Returns:
            KDEDetector: The fitted detector instance.

        Examples:
            >>> detector = KDEDetector()
            >>> detector.fit(X_train, bandwidth=0.5)
            KDEDetector(...)
        """
        self.kde = KernelDensity(**params)
        X = data.values if isinstance(data, pd.DataFrame) else data
        self.kde.fit(X)
        return self

    def score(self, data: FrameOrArray) -> ScoreArray:
        """Evaluate log-density scores for the provided samples.

        Args:
            data (pandas.DataFrame or numpy.ndarray): Samples to score shaped
                ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Log-density values where lower scores correspond to
                potential anomalies.

        Examples:
            >>> detector = KDEDetector().fit(X_train)
            >>> detector.score(X_test)[:3]  # doctest: +SKIP
            array([-3.2, -4.5, -2.8])
        """
        X = data.values if isinstance(data, pd.DataFrame) else data
        return np.asarray(self.kde.score_samples(X))


class COPODDetector(_PyODAdapter):
    """Copula-based Outlier Detector from PyOD."""

    pyod_class_path: ClassVar[str] = "pyod.models.copod.COPOD"
    display_name: ClassVar[str] = "COPOD"


class FeatureBaggingDetector(_PyODAdapter):
    """Feature Bagging ensemble from PyOD."""

    pyod_class_path: ClassVar[str] = "pyod.models.feature_bagging.FeatureBagging"
    display_name: ClassVar[str] = "Feature Bagging"


class LODADetector(_PyODAdapter):
    """Lightweight Online Detector of Anomalies from PyOD."""

    pyod_class_path: ClassVar[str] = "pyod.models.loda.LODA"
    display_name: ClassVar[str] = "LODA"


class ABODDetector(_PyODAdapter):
    """Angle-Based Outlier Detector from PyOD."""

    pyod_class_path: ClassVar[str] = "pyod.models.abod.ABOD"
    display_name: ClassVar[str] = "ABOD"


__all__ = [
    "IsolationForestDetector",
    "LOFDetector",
    "SOSDetector",
    "EnsembleDetector",
    "HBOSDetector",
    "KNNDetector",
    "OneClassSVMDetector",
    "DBSCANDetector",
    "EllipticEnvelopeDetector",
    "GaussianMixtureDetector",
    "SklearnLOFDetector",
    "KMeansDetector",
    "PCAReconstructionDetector",
    "MahalanobisDetector",
    "KDEDetector",
    "COPODDetector",
    "FeatureBaggingDetector",
    "LODADetector",
    "ABODDetector",
]
