"""Visualization utilities for detector evaluation and analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

try:  # pragma: no cover - optional dependency
    import umap

    _UMAP_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    umap = None
    _UMAP_AVAILABLE = False


ArrayLike = np.ndarray | Sequence[float] | Sequence[int]


def plot_roc_curves(
    y_true: ArrayLike,
    detector_scores: Mapping[str, ArrayLike],
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot ROC curves for multiple detectors.

    Parameters
    ----------
    y_true : ArrayLike
        Ground-truth binary anomaly labels where ``1`` represents an anomaly.
    detector_scores : Mapping[str, ArrayLike]
        Mapping from detector names to their anomaly score arrays. Higher
        scores should indicate higher anomaly likelihood.
    ax : Axes, optional
        Existing Matplotlib axis to plot on. When omitted a new figure and axis
        are created.

    Returns
    -------
    Axes
        Axis containing the plotted ROC curves.
    """

    if ax is None:
        _, ax = plt.subplots()

    y_true_array = np.asarray(y_true)
    if y_true_array.ndim != 1:
        raise ValueError("y_true must be a one-dimensional array")

    for name, scores in detector_scores.items():
        score_array = np.asarray(scores)
        if score_array.shape[0] != y_true_array.shape[0]:
            raise ValueError(
                f"Detector '{name}' produced {score_array.shape[0]} scores but "
                f"expected {y_true_array.shape[0]}"
            )
        fpr, tpr, _ = roc_curve(y_true_array, score_array)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def create_score_histogram(
    detector_scores: Mapping[str, ArrayLike],
    *,
    bins: int = 30,
    ax: Axes | None = None,
) -> Axes:
    """Create a histogram comparing score distributions across detectors.

    Parameters
    ----------
    detector_scores : Mapping[str, ArrayLike]
        Mapping of detector names to their anomaly scores.
    bins : int, default=30
        Number of histogram bins to compute.
    ax : Axes, optional
        Existing Matplotlib axis to plot on. When omitted a new figure and axis
        are created.

    Returns
    -------
    Axes
        Axis containing the plotted histograms.
    """

    if ax is None:
        _, ax = plt.subplots()

    for name, scores in detector_scores.items():
        score_array = np.asarray(scores, dtype=float)
        ax.hist(
            score_array,
            bins=bins,
            alpha=0.6,
            label=name,
            density=True,
            edgecolor="white",
        )

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions")
    ax.legend()
    return ax


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Sequence[str] | None = None,
    normalize: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot a confusion matrix summarizing detector predictions.

    Parameters
    ----------
    y_true : ArrayLike
        Ground-truth binary anomaly labels.
    y_pred : ArrayLike
        Predicted binary labels produced by a detector.
    labels : Sequence[str], optional
        Class labels corresponding to ``[normal, anomaly]``. Defaults to
        ``["Normal", "Anomaly"]`` when omitted.
    normalize : {"true", "pred", "all"}, optional
        Normalization mode forwarded to
        :func:`sklearn.metrics.confusion_matrix`.
    ax : Axes, optional
        Existing Matplotlib axis to plot on. When omitted a new figure and axis
        are created.

    Returns
    -------
    Axes
        Axis displaying the confusion matrix.
    """

    if ax is None:
        _, ax = plt.subplots()

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    if labels is None:
        labels = ("Normal", "Anomaly")
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    return ax


def visualize_embedding(
    data: ArrayLike,
    scores: ArrayLike,
    *,
    method: str = "tsne",
    reducer_kwargs: Mapping[str, Any] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Visualize high-dimensional data using t-SNE or UMAP embeddings.

    Parameters
    ----------
    data : ArrayLike
        High-dimensional feature matrix of shape ``(n_samples, n_features)``.
    scores : ArrayLike
        Anomaly scores used to color points in the embedding.
    method : {"tsne", "umap"}, default="tsne"
        Dimensionality reduction algorithm to apply.
    reducer_kwargs : Mapping[str, Any], optional
        Additional keyword arguments forwarded to the selected reducer.
    ax : Axes, optional
        Existing Matplotlib axis to plot on. When omitted a new figure and axis
        are created.

    Returns
    -------
    Axes
        Axis displaying the two-dimensional embedding.

    Raises
    ------
    ValueError
        If an unsupported ``method`` is provided or the score length does not
        match ``data``.
    ImportError
        When ``method`` is "umap" but the optional dependency is unavailable.
    """

    embeddings = np.asarray(data)
    if embeddings.ndim != 2:
        raise ValueError("data must be a two-dimensional array")

    score_array = np.asarray(scores)
    if score_array.shape[0] != embeddings.shape[0]:
        raise ValueError("scores must have the same length as data")

    reducer_kwargs = dict(reducer_kwargs or {})

    if method.lower() == "tsne":
        reducer_kwargs.setdefault("random_state", 42)
        reducer = TSNE(n_components=2, **reducer_kwargs)
    elif method.lower() == "umap":
        if not _UMAP_AVAILABLE:  # pragma: no cover - exercised in tests
            raise ImportError("UMAP is not installed. Install umap-learn to use it.")
        reducer_kwargs.setdefault("random_state", 42)
        reducer = umap.UMAP(n_components=2, **reducer_kwargs)
    else:
        raise ValueError("method must be either 'tsne' or 'umap'")

    embedding = reducer.fit_transform(embeddings)

    if ax is None:
        _, ax = plt.subplots()

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=score_array,
        cmap="viridis",
        edgecolors="none",
    )
    plt.colorbar(scatter, ax=ax, label="Anomaly Score")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"{method.upper()} Embedding")
    return ax


__all__ = [
    "plot_roc_curves",
    "create_score_histogram",
    "plot_confusion_matrix",
    "visualize_embedding",
]
