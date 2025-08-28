# Roadmap

The project's next major goal is to broaden the range of anomaly detectors.
This document sketches the path toward that goal so that contributors can
coordinate their efforts.

## Phase 1 – Expand classical methods
- Integrate additional scikit‑learn compatible detectors such as COPOD, Feature
  Bagging, LODA, ABOD, and others available in the literature.
- Refactor existing code so detectors share a common interface, easing future
  integrations.

## Phase 2 – Incorporate deep learning techniques
- Implement autoencoder variants (denoising and variational) and explore
  sequence models like LSTM or Transformer‑based detectors for time‑series data.
- Evaluate generative approaches, e.g. GAN‑based detectors like AnoGAN and
  MAD‑GAN.

## Phase 3 – Benchmarking and community tooling
- Extend the benchmark suite with additional tabular, image, and time‑series
  datasets.
- Provide configuration‑driven benchmarking utilities to compare new
  detectors.
- Publish contribution guidelines to make it easy for the community to add new
  algorithms.

## Phase 4 – Specialized domains and streaming data
- Add graph and network anomaly detectors to handle structured data.
- Integrate streaming/online algorithms such as Online Isolation Forest and
  Random Cut Forest.
- Explore forecasting-based approaches (e.g., ARIMA, Prophet) for time-series
  anomaly detection.

## Phase 5 – Automation and scalability
- Implement hyperparameter search utilities to standardize detector tuning.
- Develop a plugin interface so external libraries can register detectors
  dynamically.
- Publish benchmark leaderboards with reproducible configurations to guide
  community contributions.

Contributions are welcome at every stage of this roadmap.
