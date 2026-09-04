# Detectors

## The lifecycle

Every detector follows the same three-method contract:

```python
detector.fit(data, **params)      # trains, marks the detector fitted
detector.score(data)              # detector-specific anomaly scores
detector.detect_anomalies(data)   # fit-and-score convenience path
```

`score()` before `fit()` raises `DetectorNotFittedError` rather than returning
nonsense. `detect_anomalies()` is what benchmark workflows call, and it returns
scores that carry their orientation as metadata.

## Score orientation

Detectors disagree about which direction means "more anomalous". Isolation
Forest returns lower values for outliers; a distance-based detector returns
higher ones. Comparing them naively inverts one of the two.

Every detector therefore declares a `score_orientation`:

| Value | Meaning |
| --- | --- |
| `higher_is_more_anomalous` | larger score, more anomalous |
| `lower_is_more_anomalous` | smaller score, more anomalous |
| `binary_anomaly` | scores are 0 or 1 |
| `estimator_defined` | orientation unknown; rejected by benchmark evaluation |

`score()` preserves each detector's native values, so nothing is silently
rescaled. Benchmark evaluation calls `canonicalize_anomaly_scores()` to put
everything on `higher_is_more_anomalous` before ranking.

!!! warning "This matters outside benchmarking too"
    `analytics.hyperparam.grid_search` once scored raw `score()` output. For a
    lower-is-more-anomalous detector that inverts the ROC AUC, so the search
    selected the *worst* parameters in the grid while reporting a plausible
    number. If you write your own evaluation loop, canonicalise first.

## Available detectors

35 detectors ship in the registry. The key in the first column is what
`--detectors` and `get_detector_class()` take.

### Classical

| Key | Detector |
| --- | --- |
| `isolation_forest` | Isolation Forest |
| `knn` | k-Nearest Neighbors distance |
| `hbos` | Histogram-Based Outlier Score |
| `ocsvm` | One-Class SVM |
| `elliptic_envelope` | Elliptic Envelope |
| `gaussian_mixture` | Gaussian Mixture |
| `sklearn_lof` | Local Outlier Factor (scikit-learn) |
| `kmeans` | KMeans centroid distance |
| `pca_reconstruction` | PCA reconstruction error |
| `mahalanobis` | Mahalanobis distance |
| `kde` | Kernel Density Estimation |
| `sos` | Stochastic Outlier Selection |
| `copod` | Copula-Based Outlier Detection |
| `abod` | Angle-Based Outlier Detection |
| `loda` | Lightweight On-line Detector of Anomalies |
| `feature_bagging` | Feature Bagging ensemble |
| `dbscan` | Inductive DBSCAN |

`dbscan` lives in a `correctness` module rather than with the others: DBSCAN is
transductive, so scoring unseen points needs an explicit inductive wrapper
instead of the usual `fit`/`predict` pairing.

### Modern tabular

| Key | Detector |
| --- | --- |
| `ecod` | Empirical Cumulative Distribution Outlier Detection |
| `random_feature_isolation_forest` | Random Feature Isolation Forest |
| `random_network_distillation` | Random Network Distillation |

Worth reaching for when distance, density, or covariance baselines are too rigid
for nonlinear feature interactions. Classical methods remain preferable for
small datasets, tight latency budgets, or when the score has to be explainable.

### Deep — `deep` extra

| Key | Detector |
| --- | --- |
| `autoencoder` | Autoencoder |
| `denoising_autoencoder` | Denoising Autoencoder |
| `variational_autoencoder` | Variational Autoencoder |
| `anogan` | AnoGAN |
| `madgan` | MAD-GAN |

### Temporal — `deep` extra

| Key | Detector |
| --- | --- |
| `lstm_autoencoder` | LSTM Autoencoder |
| `tcn_autoencoder` | Temporal Convolutional Network Autoencoder |
| `transformer` | Transformer |

### Streaming — `streaming` extra

| Key | Detector |
| --- | --- |
| `half_space_trees` | Half-Space Trees |
| `online_isolation_forest` | Online Isolation Forest |
| `random_cut_forest` | Random Cut Forest |

`random_cut_forest` is not present in every River release; when the installed
version does not provide it, selecting the key raises `DependencyError` naming
the missing estimator rather than failing at import.

### Graph

| Key | Detector |
| --- | --- |
| `degree_centrality` | Degree Centrality |
| `graph_isolation_forest` | Graph Isolation Forest |

### Forecasting

| Key | Detector |
| --- | --- |
| `arima` | ARIMA residuals |
| `prophet` | Prophet residuals (`forecasting` extra) |

Every entry is registered as a dotted `module:Class` string and imported only
when selected, so the heavy frameworks cost nothing until you ask for a
detector that needs them.

## Time-series semantics

Sequence detectors operate on genuine sequences rather than collapsing the time
axis. A `WindowSpec` declares the windowing explicitly:

```python
from anomalybench.analytics.time_series import WindowSpec

spec = WindowSpec(window_length=32, stride=1, horizon=0)
```

Window scores carry the point indices they align to, so a windowed score vector
can be compared against point labels without the caller having to reconstruct
the alignment. Mismatches raise rather than silently comparing unaligned
vectors.
