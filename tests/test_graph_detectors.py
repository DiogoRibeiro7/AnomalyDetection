"""Tests for the registered graph detectors.

Both detectors are in the registry but had no direct tests, so the lifecycle,
score orientation, and the parameter name they inherit from ``BaseDetector``
are pinned here.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from analytics.detectors import get_detector_class
from analytics.exceptions import DetectorNotFittedError

GRAPH_DETECTORS = ["degree_centrality", "graph_isolation_forest"]


@pytest.fixture
def star() -> nx.Graph:
    """A hub joined to six leaves: node 0 is the structural outlier."""

    return nx.star_graph(6)


@pytest.mark.parametrize("detector_key", GRAPH_DETECTORS)
def test_fit_accepts_the_base_class_parameter_name(
    detector_key: str, star: nx.Graph
) -> None:
    """``fit`` must bind by keyword through the ``BaseDetector`` interface."""

    detector = get_detector_class(detector_key)()
    assert detector.fit(data=star) is detector
    assert detector.is_fitted


@pytest.mark.parametrize("detector_key", GRAPH_DETECTORS)
def test_score_returns_one_value_per_node(detector_key: str, star: nx.Graph) -> None:
    detector = get_detector_class(detector_key)()
    detector.fit(star)
    scores = np.asarray(detector.score(star))

    assert scores.shape == (star.number_of_nodes(),)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("detector_key", GRAPH_DETECTORS)
def test_score_before_fit_is_refused(detector_key: str, star: nx.Graph) -> None:
    detector = get_detector_class(detector_key)()
    with pytest.raises(DetectorNotFittedError):
        detector.score(star)


@pytest.mark.parametrize("detector_key", GRAPH_DETECTORS)
def test_detect_anomalies_carries_the_declared_orientation(
    detector_key: str, star: nx.Graph
) -> None:
    detector_cls = get_detector_class(detector_key)
    scores = detector_cls().detect_anomalies(star)

    assert scores.score_orientation == detector_cls.score_orientation


def test_degree_centrality_ranks_the_hub_as_most_anomalous(star: nx.Graph) -> None:
    """Higher is more anomalous, and the hub's degree is the outlier."""

    detector = get_detector_class("degree_centrality")()
    scores = np.asarray(detector.fit(star).score(star))

    assert int(np.argmax(scores)) == 0
    # Every leaf has the same degree, so their scores are identical.
    assert np.allclose(scores[1:], scores[1])


def test_graph_isolation_forest_ranks_the_hub_as_most_anomalous(
    star: nx.Graph,
) -> None:
    """Lower is more anomalous for this detector, so the hub is the minimum."""

    detector = get_detector_class("graph_isolation_forest")()
    scores = np.asarray(detector.fit(star, random_state=0).score(star))

    assert int(np.argmin(scores)) == 0


def test_degree_centrality_is_stable_across_repeated_scoring(star: nx.Graph) -> None:
    detector = get_detector_class("degree_centrality")()
    detector.fit(star)

    first = np.asarray(detector.score(star))
    second = np.asarray(detector.score(star))

    np.testing.assert_allclose(first, second)
