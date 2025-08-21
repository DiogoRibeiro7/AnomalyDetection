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
