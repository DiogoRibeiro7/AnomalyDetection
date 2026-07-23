# Anomaly Detection

[![CI](https://github.com/DiogoRibeiro7/AnomalyDetection/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/DiogoRibeiro7/AnomalyDetection/actions/workflows/ci.yml) [![Branch Policy](https://github.com/DiogoRibeiro7/AnomalyDetection/actions/workflows/branch-policy.yml/badge.svg?branch=develop)](https://github.com/DiogoRibeiro7/AnomalyDetection/actions/workflows/branch-policy.yml) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Anomaly Detection provides anomaly detection algorithms, benchmark dataset
loaders, and a command line interface for comparing detectors on standard
datasets.

## Python Support

This project supports Python `3.12` only. Python `3.13` is intentionally blocked
until the project runtime and dependency stack are validated there.

Install the project and development tooling with Poetry:

```bash
poetry install
poetry run pre-commit install
```

Poetry manages dependencies through `pyproject.toml`.

## Optional Extras

The base install supports classical detectors, ARIMA forecasting, graph
detectors, the CLI, and benchmark workflows. Optional detector stacks can be
enabled with Poetry extras:

```bash
# Deep learning detectors: PyTorch and TensorFlow
poetry install -E deep

# Streaming detectors: River
poetry install -E streaming

# Prophet detector support
poetry install -E forecasting

# Enable all optional detector stacks
poetry install -E all-detectors
```

## Usage

Run all benchmarks:

```bash
poetry run benchmark-cli
```

Select specific datasets by name:

```bash
poetry run benchmark-cli wisconsin_breast_cancer cardio
```

The suite includes tabular, image, time-series, and graph datasets:

```bash
poetry run benchmark-cli iris digits fashion_mnist_sample nab_art_daily_small_noise nab_machine_temperature synthetic_timeseries karate_club_graph
```

Run selected detectors:

```bash
poetry run benchmark-cli --detectors knn hbos
```

Run from a YAML configuration:

```bash
poetry run benchmark-cli --config benchmarks/benchmark_config.yml
```

Run a reproducible smoke benchmark with a manifest, JSON report, and enriched
leaderboard:

```bash
poetry run benchmark-cli --config benchmarks/benchmark_config.v0.3.0-smoke.yml
```

Run a multi-metric benchmark using dataset metadata selectors:

```bash
poetry run benchmark-cli --config benchmarks/benchmark_config.v0.4.0-metrics.yml
```

Show dataset summaries:

```bash
poetry run benchmark-cli --summary
```

Legacy display names such as `wisconsinBreast` remain supported, but canonical
loader keys such as `wisconsin_breast_cancer` should be preferred for scripts
and configuration files.

Available detectors include Isolation Forest, Stochastic Outlier Selection,
K-Nearest Neighbors, Histogram-Based Outlier Score, One-Class SVM, DBSCAN,
Elliptic Envelope, Gaussian Mixture, Sklearn LOF, KMeans, PCA Reconstruction,
Mahalanobis distance, Kernel Density, Autoencoder, Denoising Autoencoder,
Variational Autoencoder, LSTM Autoencoder, Transformer, COPOD, Feature Bagging,
LODA, ABOD, Half-Space Trees, Online Isolation Forest, AnoGAN, MAD-GAN, Degree
Centrality, Graph Isolation Forest, ARIMA, and Prophet.

## Plugins

External detector packages can register themselves through plugin modules whose
names start with `plugins.`:

```bash
poetry run benchmark-cli --plugins plugins.my_module --detectors my_custom_detector
```

The module should call `analytics.detectors.register_detector` during import.
Detector keys are protected against accidental collisions by default. To
replace an existing detector intentionally, pass `allow_override=True` to
`register_detector`.

## Detector API

All built-in detectors follow the same lifecycle:

- `fit(data, **params)` trains the detector and marks it as fitted.
- `score(data)` returns detector-specific anomaly scores and raises a
  `RuntimeError` if called before `fit`.
- `detect_anomalies(data, **params)` is the fit-and-score convenience path used
  by benchmark workflows.

Detectors expose an `is_fitted` property and a `score_orientation` value. Score
orientation is one of `higher_is_more_anomalous`, `lower_is_more_anomalous`,
`binary_anomaly`, or `estimator_defined`. Current score values are preserved for
backwards compatibility; use `score_orientation` when comparing detectors that
produce different score semantics.

## Hyperparameter Search

Stratified cross-validation utilities are available in `analytics.hyperparam`:

```python
from analytics.hyperparam import grid_search

best_params, score = grid_search(
    "isolation_forest",
    {"n_estimators": [50, 100]},
    X,
    y,
    cv=5,
)
```

## Leaderboards

Benchmark results can be appended to a CSV leaderboard:

```bash
poetry run benchmark-cli iris --detectors isolation_forest --leaderboard results.csv
```

The leaderboard CSV uses a structured schema with these columns:
`run_timestamp_utc`, `run_id`, `config_hash`, `dataset_name`, `dataset_key`,
`detector_name`, `detector_label`, `detector_params`, `random_seed`,
`runtime_seconds`, `failure_category`, `auc`, and `error`.

For reproducible runs, provide a stable run identifier, seed, and output path:

```bash
poetry run benchmark-cli iris \
  --detectors isolation_forest \
  --metrics roc_auc average_precision precision_at_k runtime \
  --metric-k 10 \
  --positive-label 1 \
  --random-seed 42 \
  --run-id paper-table-1 \
  --output-dir benchmark-results \
  --json-report benchmark-results/paper-table-1.json
```

The JSON report embeds a benchmark manifest with the package version, Python
version, selected dataset keys, detector keys and parameters, random seed,
metric configuration, configuration hash, timestamp, and bundled dataset file
integrity hashes.
Bundled dataset metadata in `benchmarks/datasets.yml` records source URLs,
license notes, modality, task type, label semantics, and local files used for
integrity checks.

Supported benchmark metrics are `roc_auc`, `average_precision`,
`precision_at_k`, `recall_at_k`, `f1_at_threshold`, `best_f1`, and `runtime`.
YAML configurations can select datasets by metadata fields such as `modality`,
`task`, and `label_type`.

## Quality Checks

Run the same core checks used by CI before opening a pull request:

```bash
poetry check
poetry build -f wheel
poetry run python -m pytest -q
poetry run pre-commit run --all-files
```

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff) and
[.zenodo.json](.zenodo.json). Zenodo will use `.zenodo.json` when archiving
GitHub releases for DOI creation.

To cite all versions of this software, use the Zenodo concept DOI:
[10.5281/zenodo.21496904](https://doi.org/10.5281/zenodo.21496904).

## Project Roadmap

- Implement variable width binning for HBOS. Completed.
- Add caching to LOF calculations. Completed.
- Provide a CLI for running included benchmarks. Completed.
- Add dataset summary output to the CLI. Completed.
- Expand the detector library with additional algorithms and deep learning
  approaches. See [ROADMAP.md](ROADMAP.md).

## Contributing

Contributions to expand the detector library are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
