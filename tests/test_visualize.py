"""Tests for visualization utilities."""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import analytics.visualize as visualize


@pytest.fixture(autouse=True)
def close_figures() -> None:
    """Ensure Matplotlib figures do not accumulate across tests."""

    yield
    plt.close("all")


def test_plot_roc_curves_draws_curves() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    scores = {
        "DetectorA": np.array([0.1, 0.9, 0.2, 0.7, 0.3, 0.8]),
        "DetectorB": np.array([0.3, 0.8, 0.4, 0.6, 0.5, 0.9]),
    }

    ax = visualize.plot_roc_curves(y_true, scores)

    # Two detectors plus the diagonal reference line.
    assert len(ax.lines) == 3
    assert ax.get_title() == "ROC Curves"


def test_create_score_histogram_overlays_distributions() -> None:
    scores = {
        "DetectorA": np.random.normal(size=50),
        "DetectorB": np.random.normal(loc=1.0, size=50),
    }

    ax = visualize.create_score_histogram(scores, bins=10)

    assert ax.get_title() == "Score Distributions"
    assert len(ax.patches) > 0


def test_plot_confusion_matrix_uses_defaults() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    ax = visualize.plot_confusion_matrix(y_true, y_pred)

    assert ax.get_title() == "Confusion Matrix"
    assert ax.images[0].get_array().shape == (2, 2)


def test_visualize_embedding_tsne_returns_scatter() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(10, 4))
    scores = rng.random(10)

    ax = visualize.visualize_embedding(
        data,
        scores,
        method="tsne",
        reducer_kwargs={"perplexity": 3, "max_iter": 250},
    )

    scatter = ax.collections[0]
    assert scatter.get_offsets().shape[0] == data.shape[0]


def test_visualize_embedding_umap_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=(12, 3))
    scores = rng.random(12)

    if visualize._UMAP_AVAILABLE:  # pragma: no branch - depends on environment
        ax = visualize.visualize_embedding(
            data,
            scores,
            method="umap",
            reducer_kwargs={"n_neighbors": 5, "min_dist": 0.1},
        )
        scatter = ax.collections[0]
        assert scatter.get_offsets().shape[0] == data.shape[0]
    else:
        with pytest.raises(ImportError):
            visualize.visualize_embedding(data, scores, method="umap")
