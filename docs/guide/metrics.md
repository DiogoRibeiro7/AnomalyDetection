# Metrics

## Supported metrics

| Name | Notes |
| --- | --- |
| `roc_auc` | default; threshold-free ranking quality |
| `average_precision` | area under the precision-recall curve |
| `precision_at_k` | precision among the top `k` scored points |
| `recall_at_k` | recall among the top `k` scored points |
| `f1_at_threshold` | F1 at an explicit score threshold |
| `best_f1` | best F1 over all thresholds on the PR curve |
| `runtime` | wall-clock seconds for fit and score |

Configure them with `--metrics`, or under `metrics.include` in YAML. `k`
defaults to the number of true anomalies when unset, which makes
`precision_at_k` comparable across datasets with different anomaly rates.

## Making detectors comparable

Two normalisations run before any metric is computed.

**Labels are canonicalised to `1 = anomaly`.** Datasets disagree: some mark
anomalies `1`, some `-1`, some use a class name. `positive_label` declares
which value is the anomaly class, and everything downstream sees a boolean
mask.

**Scores are canonicalised to `higher_is_more_anomalous`.** Each detector
declares its native `score_orientation`; `canonicalize_anomaly_scores()` flips
the ones that need it. A detector declaring `estimator_defined` — orientation
unknown — is rejected rather than scored on a guess, because a silently
inverted ROC AUC looks like a plausible bad result rather than a bug.

## Degenerate cases

A metric that cannot be computed yields `null`, not a crash and not a fake
number. This happens when a dataset split contains only one class, or when a
detector produces constant scores. A `k` larger than the sample count is not
one of these — it is clamped to the number of samples.

!!! note "Non-finite results"
    scikit-learn returns `NaN` rather than raising for some degenerate inputs.
    Catching `ValueError` alone let `NaN` through into the JSON report, which is
    not valid JSON. Metric results are now checked for finiteness and coerced to
    `null` when they are not.
