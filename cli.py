#!/usr/bin/env python3
"""Command line interface to run anomaly detection benchmarks.

Use ``--summary`` to display information about the available datasets instead
of running the detectors.
"""
import argparse
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
        counts = df[labels].value_counts().to_dict()
        print(f"Dataset: {name}")
        print(f"  samples: {len(df)}")
        print(f"  features: {len(features)}")
        print(f"  label distribution: {counts}")
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
}


def run_benchmarks(datasets=None):
    for ds in load_all_datasets():
        name = ds["name"]
        if datasets and name not in datasets:
            continue
        df = ds["dataframe"]
        features = ds["feature_cols"]
        labels = ds["label_col"]
        print(f"Dataset: {name}")
        for det_name, detector in DETECTORS.items():
            scores = detector.detect_anomalies(df[features])
            auc = roc_auc_score(df[labels], scores)
            print(f"  {det_name}: AUC={auc:.3f}")
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
    args = parser.parse_args()
    if args.summary:
        summarize_datasets(args.datasets)
    else:
        run_benchmarks(args.datasets)


if __name__ == "__main__":
    main()
