#!/usr/bin/env python3
"""Command line interface to run anomaly detection benchmarks.

Use ``--summary`` to display information about the available datasets instead
of running the detectors.
"""
import argparse
from collections import Counter
import pandas as pd
import networkx as nx
from benchmarks.load_all_datasets import load_all_datasets
from analytics.detector import (
    IsolationForestDetector,
    SOSDetector,
    KNNDetector,
    HBOSDetector,
    EllipticEnvelopeDetector,
    GaussianMixtureDetector,
    SklearnLOFDetector,
    KMeansDetector,
    PCAReconstructionDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
    DBSCANDetector,
    KDEDetector,
    AutoencoderDetector,
    COPODDetector,
    FeatureBaggingDetector,
    LODADetector,
    ABODDetector,
    DenoisingAutoencoderDetector,
    VariationalAutoencoderDetector,
    LSTMAutoencoderDetector,
    TransformerDetector,
    AnoGANDetector,
    MADGANDetector,
    DegreeCentralityDetector,
    GraphIsolationForestDetector,
)
from sklearn.metrics import roc_auc_score


def summarize_datasets(datasets=None):
    """Print basic information about the available datasets."""
    for ds in load_all_datasets():
        name = ds["name"]
        if datasets and name not in datasets:
            continue
        df = ds["dataframe"]
        features = ds["feature_cols"]
        labels = ds["label_col"]
        print(f"Dataset: {name}")
        if isinstance(df, pd.DataFrame):
            counts = df[labels].value_counts().to_dict()
            print(f"  samples: {len(df)}")
            print(f"  features: {len(features)}")
            print(f"  label distribution: {counts}")
        else:
            counts = Counter(nx.get_node_attributes(df, labels).values())
            print(f"  nodes: {df.number_of_nodes()}")
            print(f"  edges: {df.number_of_edges()}")
            print(f"  label distribution: {dict(counts)}")
        print()


DETECTORS = {
    "isolation_forest": IsolationForestDetector(),
    "sos": SOSDetector(),
    "knn": KNNDetector(),
    "hbos": HBOSDetector(),
    "ocsvm": OneClassSVMDetector(),
    "dbscan": DBSCANDetector(),
    "elliptic_envelope": EllipticEnvelopeDetector(),
    "gaussian_mixture": GaussianMixtureDetector(),
    "sklearn_lof": SklearnLOFDetector(),
    "kmeans": KMeansDetector(),
    "pca_reconstruction": PCAReconstructionDetector(),
    "mahalanobis": MahalanobisDetector(),
    "kde": KDEDetector(),
    "autoencoder": AutoencoderDetector(),
    "denoising_autoencoder": DenoisingAutoencoderDetector(),
    "variational_autoencoder": VariationalAutoencoderDetector(),
    "lstm_autoencoder": LSTMAutoencoderDetector(),
    "transformer": TransformerDetector(),
    "copod": COPODDetector(),
    "feature_bagging": FeatureBaggingDetector(),
    "loda": LODADetector(),
    "abod": ABODDetector(),
    "anogan": AnoGANDetector(),
    "madgan": MADGANDetector(),
    "degree_centrality": DegreeCentralityDetector(),
    "graph_isolation_forest": GraphIsolationForestDetector(),
}


def run_benchmarks(datasets=None, detectors=None):
    """Run benchmarks for the specified datasets and detectors."""
    if detectors:
        selected = [(n, DETECTORS[n]) for n in detectors if n in DETECTORS]
    else:
        selected = DETECTORS.items()

    for ds in load_all_datasets():
        name = ds["name"]
        if datasets and name not in datasets:
            continue
        df = ds["dataframe"]
        features = ds["feature_cols"]
        labels = ds["label_col"]
        print(f"Dataset: {name}")
        for det_name, detector in selected:
            try:
                if isinstance(df, pd.DataFrame):
                    scores = detector.detect_anomalies(df[features])
                    y_true = df[labels]
                else:
                    scores = detector.detect_anomalies(df)
                    y_true = [data[labels] for _, data in df.nodes(data=True)]
                auc = roc_auc_score(y_true, scores)
                print(f"  {det_name}: AUC={auc:.3f}")
            except Exception as e:
                print(f"  {det_name}: skipped ({e})")
        print()


def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection benchmarks")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Names of datasets to benchmark. Defaults to all",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show dataset summaries instead of running benchmarks",
    )
    parser.add_argument(
        "--detectors",
        nargs="*",
        help="Detector names to run. Defaults to all",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML configuration specifying datasets and detectors",
    )
    args = parser.parse_args()
    if args.config:
        from benchmarks.config_benchmark import run_from_config

        run_from_config(args.config)
    elif args.summary:
        summarize_datasets(args.datasets)
    else:
        run_benchmarks(args.datasets, args.detectors)


if __name__ == "__main__":
    main()
