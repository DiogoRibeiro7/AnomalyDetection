"""Graph and network based anomaly detectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from analytics.base import BaseDetector

if TYPE_CHECKING:
    import networkx as nx


ScoreArray = NDArray[np.floating[Any]]


class DegreeCentralityDetector(BaseDetector):
    """Flag nodes with unusual degree centrality in a graph."""

    def get_name(self) -> str:
        return "Degree Centrality"

    def fit(self, graph: "nx.Graph", **params: Any) -> DegreeCentralityDetector:
        import networkx as nx

        centrality = nx.degree_centrality(graph)
        values = np.array(list(centrality.values()), dtype=float).reshape(-1, 1)
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(values)
        return self

    def score(self, graph: "nx.Graph") -> ScoreArray:
        import networkx as nx

        centrality = nx.degree_centrality(graph)
        values = np.array(list(centrality.values()), dtype=float).reshape(-1, 1)
        z = np.abs(self.scaler.transform(values))
        return np.asarray(z.ravel(), dtype=float)


class GraphIsolationForestDetector(BaseDetector):
    """Apply Isolation Forest on basic graph structural features."""

    def get_name(self) -> str:
        return "Graph Isolation Forest"

    def _graph_features(self, graph: "nx.Graph") -> NDArray[np.floating[Any]]:
        import networkx as nx

        degrees = dict(graph.degree())
        clustering = nx.clustering(graph)
        return np.array(
            [[degrees[n], clustering[n]] for n in graph.nodes()], dtype=float
        )

    def fit(self, graph: "nx.Graph", **params: Any) -> GraphIsolationForestDetector:
        from sklearn.ensemble import IsolationForest

        self.nodes = list(graph.nodes())
        X = self._graph_features(graph)
        self.model = IsolationForest(**params).fit(X)
        return self

    def score(self, graph: "nx.Graph") -> ScoreArray:
        X = self._graph_features(graph)
        return np.asarray(self.model.decision_function(X), dtype=float)


__all__ = [
    "DegreeCentralityDetector",
    "GraphIsolationForestDetector",
]
