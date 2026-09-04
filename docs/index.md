# AnomalyBench

Anomaly detection algorithms, benchmark dataset loaders, and a command line
interface for comparing detectors on standard datasets under one reproducible
protocol.

The distinguishing feature is not the detector library — [PyOD] already does
that well, and several detectors here are adapters over it. It is that every
comparison runs under the same evaluation contract: labels canonicalised to
`1 = anomaly`, scores put on a common orientation before ranking, and a
manifest recording exactly what produced each number.

[PyOD]: https://pyod.readthedocs.io/

## Install

```bash
pip install anomalybench
```

The base install covers classical detectors, ARIMA forecasting, graph
detectors, the CLI, and the benchmark workflows. Deep learning, streaming, and
Prophet stacks are [optional extras](install.md).

## A first benchmark

```bash
benchmark-cli iris --detectors isolation_forest knn --random-seed 42
```

```
Dataset: iris
  isolation_forest (random_state=42): AUC=0.577
  knn: AUC=0.256
```

The seed annotation appears only for detectors that accept one, and `iris` is a
classification dataset pressed into service as an anomaly benchmark — the point
of the example is the mechanics, not the scores.

Or from Python:

```python
from anomalybench.analytics.detectors import get_detector_class

detector = get_detector_class("isolation_forest")()
detector.fit(X_train)
scores = detector.score(X_test)
```

## What to read next

<div class="grid cards" markdown>

- **[Detectors](guide/detectors.md)** — the lifecycle every detector follows,
  what is available, and why score orientation matters.
- **[Running benchmarks](guide/benchmarks.md)** — the CLI, YAML configuration,
  run manifests, and leaderboards.
- **[Metrics](guide/metrics.md)** — which metrics are supported and how scores
  are made comparable across detectors.
- **[Exceptions](guide/exceptions.md)** — the structured error hierarchy, and
  what changed for callers who used to catch `ValueError`.

</div>

## Citing

Cite the concept DOI to refer to the software generally, or the version DOI of
the release you actually ran:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21496904.svg)](https://doi.org/10.5281/zenodo.21496904)

Machine-readable metadata lives in `CITATION.cff` and `.zenodo.json`. For a
result you intend to publish, cite the version DOI and keep the run manifest —
together they pin the code and the configuration that produced it.
