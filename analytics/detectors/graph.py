"""Graph and network based anomaly detectors."""

from __future__ import annotations

import numpy as np
from analytics.base import BaseDetector


class DegreeCentralityDetector(BaseDetector):
    """Flag nodes with unusual degree centrality in a graph."""

    def get_name(self) -> str:
        return "Degree Centrality"

    def fit(self, graph, **params):
        import networkx as nx

        centrality = nx.degree_centrality(graph)
        values = np.array(list(centrality.values())).reshape(-1, 1)
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(values)
        return self

    def score(self, graph):
        import networkx as nx

        centrality = nx.degree_centrality(graph)
        values = np.array(list(centrality.values())).reshape(-1, 1)
        z = np.abs(self.scaler.transform(values))
        return z.ravel()


class GraphIsolationForestDetector(BaseDetector):
    """Apply Isolation Forest on basic graph structural features."""

    def get_name(self) -> str:
        return "Graph Isolation Forest"

    def _graph_features(self, graph):
        import networkx as nx

        degrees = dict(graph.degree())
        clustering = nx.clustering(graph)
        return np.array(
            [[degrees[n], clustering[n]] for n in graph.nodes()], dtype=float
        )

    def fit(self, graph, **params):
        from sklearn.ensemble import IsolationForest

        self.nodes = list(graph.nodes())
        X = self._graph_features(graph)
        self.model = IsolationForest(**params).fit(X)
        return self

    def score(self, graph):
        X = self._graph_features(graph)
        return self.model.decision_function(X)


__all__ = [name for name in globals() if name.endswith("Detector")]
