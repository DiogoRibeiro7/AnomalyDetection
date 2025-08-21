# Anomaly Detection

This repository provides a collection of simple anomaly detection algorithms
and utilities for benchmarking them on several standard datasets. A small
command line interface is provided to run the included benchmarks.

Install the required dependencies with Poetry and set up the pre‑commit hooks:

```bash
poetry install
pre-commit install
```
Poetry manages the dependencies for this project. If you previously used a
`requirements.txt` file, it has been removed in favour of the `pyproject.toml`
configuration.

## Roadmap
- Implement variable width binning for HBOS (completed).
- Add caching to LOF calculations (completed).
- Provide a small CLI for running benchmarks on the included datasets (completed).
- Add a dataset summary option to the CLI (completed).
- Expand the detector library with additional algorithms and deep
  learning approaches (see `ROADMAP.md`).

## Usage
Run all benchmarks:

```bash
python cli.py
```

Select specific datasets by name:

```bash
python cli.py wisconsinBreast cardio
```

Run only particular detectors:

```bash
python cli.py --detectors knn hbos
```

Show dataset summaries instead of running benchmarks:

```bash
python cli.py --summary
```

Available detectors include Isolation Forest, Stochastic Outlier Selection,
K‑Nearest Neighbors, Histogram‑Based Outlier Score, One‑Class SVM, DBSCAN,
Elliptic Envelope, Gaussian Mixture, Sklearn LOF, KMeans,
PCA Reconstruction, Mahalanobis distance, Kernel Density, Autoencoder,
 Denoising Autoencoder, Variational Autoencoder, LSTM Autoencoder, Transformer,
 COPOD, Feature Bagging, LODA, ABOD, AnoGAN, and MAD-GAN.
