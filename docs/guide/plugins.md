# Plugins

External packages can add detectors without being vendored into AnomalyBench.

## Registering a detector

A plugin module calls `register_detector` at import time, passing a dotted
`module:Class` path rather than the class itself — the registry loads lazily, so
a plugin that depends on a heavy framework costs nothing until its detector is
actually selected.

```python
# plugins/my_module.py
from anomalybench.analytics.detectors import register_detector

register_detector("my_custom_detector", "my_package.detectors:MyDetector")
```

The detector class subclasses `BaseDetector`, implements `fit` and `score`, and
declares its `score_orientation`:

```python
from anomalybench.analytics.base import BaseDetector


class MyDetector(BaseDetector):
    score_orientation = "higher_is_more_anomalous"

    def fit(self, data, **params):
        ...
        return self

    def score(self, data):
        ...
```

Leaving `score_orientation` at its `estimator_defined` default means benchmark
evaluation will refuse to score the detector. Declare it.

## Loading plugins

```bash
benchmark-cli --plugins plugins.my_module --detectors my_custom_detector
```

`--plugins` imports the named modules before resolving detectors, which is when
their `register_detector` calls run.

## Collision protection

The built-in registry is frozen once the shipped detectors are registered, and
re-registering an existing key raises `ConfigurationError` naming the path
already bound to it. Shadowing a built-in has to be deliberate:

```python
register_detector("knn", "my_package:MyKNN", allow_override=True)
```

This exists because a plugin that silently replaced `isolation_forest` would
produce a benchmark whose leaderboard rows are indistinguishable from ones
produced by the real thing.
