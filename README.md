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
python cli.py wisconsin_breast_cancer cardio
```

The suite now includes tabular, image, time-series, and graph datasets.
For example, run benchmarks on the Iris (tabular), Digits (image),
synthetic_timeseries (time-series), and karate_club_graph (graph) sets:

```bash
python cli.py iris digits fashion_mnist_sample nab_art_daily_small_noise synthetic_timeseries karate_club_graph
```

Legacy display names (for example, ``wisconsinBreast``) remain supported for
backwards compatibility, but the CLI now documents canonical loader keys such as
``wisconsin_breast_cancer`` to make configuration files and tags easier to
reason about. Newly added lightweight datasets like ``fashion_mnist_sample`` and
``nab_art_daily_small_noise`` broaden image and time-series coverage without
incurring large download requirements.

Run only particular detectors:

```bash
python cli.py --detectors knn hbos
```

Benchmarks can also be driven by a YAML configuration file that lists the
datasets and detectors to evaluate. An example configuration is provided in
`benchmarks/benchmark_config.yml`:

```yaml
datasets:
  - iris
  - digits
detectors:
  - isolation_forest
  - copod
```

Run benchmarks based on such a configuration with:

```bash
python cli.py --config benchmarks/benchmark_config.yml
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
 COPOD, Feature Bagging, LODA, ABOD, Online Isolation Forest, Random Cut Forest,
 AnoGAN, MAD-GAN, Degree Centrality, Graph Isolation Forest, ARIMA, and Prophet.

### Plugins and hyperparameter search

External detector packages can register themselves via the plugin interface.
Plugins must reside in modules whose names start with ``plugins.``:

```bash
python cli.py --plugins plugins.my_module --detectors my_custom_detector
```

The module ``plugins.my_module`` should call
``analytics.detectors.register_detector`` during import. Stratified
cross-validation utilities are available in :mod:`analytics.hyperparam`:

```python
from analytics.hyperparam import grid_search
best_params, score = grid_search(
    "isolation_forest", {"n_estimators": [50, 100]}, X, y, cv=5
)
```

### Leaderboard output

Benchmark results can be appended to a CSV leaderboard:

```bash
python cli.py --detectors isolation_forest --datasets iris --leaderboard results.csv
```

## Contributing
Contributions to expand the detector library are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new algorithms.
