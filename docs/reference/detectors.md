# Detectors API

## Base classes

::: anomalybench.analytics.base
    options:
      members:
        - BaseDetector
        - OrientedScores
        - coerce_tabular_2d

## Registry

::: anomalybench.analytics.detectors.registry
    options:
      members:
        - register_detector
        - get_detector_class

## Detector implementations

### Classical

::: anomalybench.analytics.detectors.classical

### Modern tabular

::: anomalybench.analytics.detectors.modern_tabular

### Graph

::: anomalybench.analytics.detectors.graph

### Forecasting

::: anomalybench.analytics.detectors.forecasting

## Supporting modules

### Preprocessing

::: anomalybench.analytics.preprocessing

### Time series

::: anomalybench.analytics.time_series

### Hyperparameter search

::: anomalybench.analytics.hyperparam
