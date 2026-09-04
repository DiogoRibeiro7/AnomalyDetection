# Roadmap

This roadmap translates the project direction into versioned milestones. It is
intended to help contributors choose work that fits the current release stage
and to keep releases scoped enough to validate properly.

The project is currently in the `0.x` series. Minor versions may still adjust
APIs when needed, but each release should preserve documented workflows or
provide a migration note. The roadmap was reviewed on 2026-07-22 against the
current PyOD capability catalogue and expanded to include newer anomaly
detection families, stronger benchmark requirements, explicit implementation
provenance, and clearer acceptance criteria. It was reviewed again on
2026-09-04 to record the releases that have shipped since and to separate
milestone identity from version numbering.

## Milestone Numbering

Versions are no longer chosen by hand. release-please derives them from
Conventional Commit prefixes on `main`, so a milestone lands in whichever minor
comes next rather than in the number this document happens to print beside it.

The `v0.6.0` heading below is the first place that mattered: `v0.6.0` and
`v0.6.1` were consumed by the distribution work recorded under
[v0.6.x](#v06x---distribution-and-documentation), not by the time-series
detector pack the heading names. Headings are therefore read as an **ordered
sequence of milestones**, not as a commitment to a version string. When a
milestone ships, its section records the version it actually landed in.

## Planning Principles

- Keep the base install useful and lightweight; put GPU-heavy, graph-heavy, and
  foundation-model integrations behind extras.
- Prefer reproducible benchmark coverage before adding large method families.
- Add new algorithms through the public detector lifecycle: `fit`, `score`, and
  `detect_anomalies`.
- Record score orientation, fitted-state behavior, randomness controls, and
  optional dependency requirements for every detector.
- Treat recent research methods as experimental until they have stable
  dependencies, documented input shapes, deterministic smoke tests, and at
  least one benchmark configuration.
- Prefer adapters to established libraries when they are maintained and expose
  compatible semantics, for example PyOD, DeepOD, PyGOD, River, and
  time-series foundation-model packages.
- Classify every method as `native`, `adapter`, `baseline`, `experimental`,
  `benchmark-only`, or `watchlist`; do not present an existing upstream method
  as a new native implementation target.
- Review upstream capability snapshots before each minor release so roadmap
  targets do not drift behind maintained libraries.
- Do not make visual, multimodal, diffusion, causal, log, or hosted-LLM tracks
  hard blockers for a stable core `v1.0.0` release.

## Method Intake Criteria

Every new detector or benchmark integration should include:

- A canonical registry key, display label, score orientation, supported
  modalities, and implementation provenance.
- A method status of `native`, `adapter`, `baseline`, `experimental`,
  `benchmark-only`, or `watchlist`.
- For adapters and baselines: upstream provider, tested provider version range,
  upstream module or registry key, and compatibility notes.
- Dependency placement: base, `deep`, `streaming`, `forecasting`, `graph`,
  `vision`, `multimodal`, `foundation`, `diffusion`, `log`, or another
  explicit extra.
- Input contract: tabular matrix, multivariate time series window, graph, event
  stream, or mixed tabular/text representation.
- Determinism controls: `random_state`, `seed`, framework seeding, or an
  explicit statement that the method is not deterministic.
- Minimum tests: lifecycle, parameter forwarding, score shape, failure path for
  missing optional dependencies, and a smoke benchmark when practical.
- Documentation: method summary, expected use case, known limitations, and
  citation or upstream implementation link.
- Benchmark entry: at least one small configuration that can run in CI or a
  documented reason why CI coverage is not feasible.

## Upstream Capability Tracking

Maintain a versioned capability snapshot at
`benchmarks/catalogues/upstream_capabilities.yml`. The snapshot should record:

- Provider name and tested version range.
- Upstream detector key or module.
- Modality, supervision level, task level, and batch or streaming protocol.
- Project treatment: `adapter`, `baseline`, `benchmark-only`, or unsupported.
- Known semantic differences in score orientation, thresholding, input shape,
  fitted-state behavior, and persistence.

The snapshot is not a lockfile. It is a reviewed catalogue used to distinguish
new project work from maintained upstream coverage. At the 2026-07-22 review,
important PyOD reference baselines included:

- Tabular: DIF, DeepSVDD, and DevNet.
- Time series: Matrix Profile, Spectral Residual, KShape, SAND, LSTMAD, and
  Anomaly Transformer.
- Graph node anomalies: DOMINANT, CoLA, CONAD, AnomalyDAE, GUIDE, Radar,
  ANOMALOUS, and SCAN; these are primarily transductive reference methods.

A provider update must not silently change benchmark meaning. Changes to an
adapter's upstream implementation, defaults, or score semantics require a
manifest update, compatibility test, and migration note when user-visible.

## v0.1.x - Foundation And Release Hygiene

Status: released as `v0.1.0`; patch releases only.

The `0.1` line establishes the repository, packaging, release, and citation
baseline.

Delivered scope:

- Poetry packaging and installable `benchmark-cli` command.
- GitHub Actions for core tests and streaming-extra smoke tests.
- Zenodo and GitHub citation metadata.
- Repository hygiene files for licensing, contributing, editor defaults, pull
  requests, and releases.

Patch policy:

- Patch documentation, packaging, and CI issues found while preparing later
  releases.
- Avoid broad feature work unless it directly unblocks a later milestone.

Exit criteria for `v0.1.x` patches:

- `poetry check` passes.
- `poetry build -f wheel` passes.
- `poetry run python -m pytest -q` passes.
- GitHub CI passes on `main`.

## v0.2.0 - Detector API Stabilization

Status: released as `v0.2.0`; patch releases only.

Goal: make detector implementations predictable enough for users, plugins, and
benchmark automation.

Delivered scope:

- Supported detector lifecycle: `fit`, `score`, and `detect_anomalies`.
- Fitted-state enforcement so `score` fails clearly before `fit`.
- Score-orientation metadata across registered detectors.
- Shared tabular input validation utilities.
- Optional dependency behavior documented for deep, streaming, and forecasting
  detectors.

Patch policy:

- Fix lifecycle or score-orientation bugs.
- Add compatibility tests for detector behavior that is already documented.
- Avoid changing public detector semantics unless the migration note is clear.

## v0.3.0 - Benchmark Reproducibility

Status: released as `v0.3.0` on 2026-07-22; patch releases only.

Goal: make benchmark results reproducible, comparable, and easier to share.

Core scope:

- Benchmark run manifest with package version, Python version, platform,
  dataset keys, detector keys, detector parameters, random seed, timestamp,
  configuration hash, and dataset file integrity records.
- Versioned JSON report schema for benchmark results.
- Leaderboard rows with runtime, failure category, random seed, run ID, and
  configuration hash fields.
- CLI and YAML config options for output directory, JSON report path, stable
  run IDs, and random seeds.
- Dataset metadata with provenance, license notes, task type, and bundled file
  integrity checks.
- Versioned smoke benchmark configuration for CI and release validation.

Acceptance criteria:

- `benchmarks/benchmark_config.v0.3.0-smoke.yml` runs without optional GPU
  dependencies.
- Report and manifest JSON are deterministic in schema, even when timestamps and
  runtime values differ.
- Existing `--leaderboard` workflows remain backward-compatible enough for
  users to append new rows without changing command structure.
- Tests verify report schema, leaderboard schema, config option forwarding,
  seed injection, and dataset integrity checks.

Non-goals:

- Adding large new datasets.
- Adding GPU-only detectors.
- Declaring benchmark scores comparable across all modalities before metric
  selection is expanded.

## v0.4.0 - Metrics, Datasets, And Evaluation Protocols

Status: released as `v0.4.0` on 2026-07-23; patch releases only.

Goal: broaden evaluation coverage and make metric choice explicit before adding
many modern detectors.

Planned work:

- Add metric selection in YAML and CLI: ROC AUC, average precision, precision at
  k, recall at k, F1 at a configured threshold, and runtime.
- Add time-series metrics that avoid over-rewarding point-adjustment, including
  VUS-ROC and VUS-PR inspired by recent reliability work on time-series anomaly
  benchmarks.
- Add calibration utilities for score normalization, threshold selection, and
  contamination-aware reporting.
- Add dataset selectors for modality, task type, size, label type, license
  status, and optional dependency needs.
- Add compact benchmark examples for tabular, multivariate time series, graph
  node anomalies, and streaming data.
- Add clearer CLI output for skipped detectors, unsupported
  dataset-detector combinations, and missing optional dependencies.
- Add report-level warnings when a metric is inappropriate for a dataset label
  type or detector score orientation.

Candidate datasets:

- Tabular: ODDS-style compact datasets where redistribution is allowed or where
  loaders can fetch from the upstream source.
- Time series: small NAB-derived examples, SMD/MSL/SMAP-style loaders when
  licensing and size are acceptable, and synthetic contextual anomaly fixtures.
- Graph: Cora/Citeseer-style attributed graph fixtures, plus small synthetic
  node-anomaly graphs.
- Streaming: generated concept-drift streams with repeatable seeds.

Deliverables:

- Expanded `benchmarks/datasets.yml` schema.
- Metric selection documented in benchmark configuration examples.
- Stable benchmark report schema extension for multiple metrics.
- CI smoke configurations for each supported modality.

Acceptance criteria:

- Metric implementations have unit tests with known expected values.
- Each included dataset has source, license note, task type, modality, label
  semantics, and integrity metadata.
- Benchmark reports record metric configuration and threshold policy.

## v0.5.0 - Modern Tabular Detector Pack

Status: released as `v0.5.0` on 2026-09-03. Scope is frozen.

The release also carries two items outside this milestone's scope: the
migration of every raised exception onto the DataExcept hierarchy, which is
a breaking change, and the TCN baseline listed under `v0.6.0` below,
delivered early.

Goal: add contemporary tabular anomaly detection methods without making the
base install depend on heavy deep learning stacks.

Milestone closure policy:

- Do not reopen or replace completed `v0.5.0` implementation work because an
  equivalent or adjacent upstream method now exists.
- The detector registry, tests, and release notes are authoritative for the
  exact methods delivered by this milestone.
- Record every delivered method as `native`, `adapter`, or `baseline`, including
  its dependency extra and tested upstream version where applicable.
- Treat PyOD DIF, DeepSVDD, and DevNet as upstream reference baselines or
  adapters in future benchmarks rather than as new implementation targets.
- Move unfinished ideas from the original candidate list to the research
  watchlist or a later release instead of extending `v0.5.0`.

Historical target families:

- Deep isolation and one-class neural representation methods.
- Weakly supervised deviation-network methods with explicit supervision
  requirements.
- Geometric-transformation and contrastive/self-supervised tabular methods.
- Retrieval-augmented reconstruction for tabular data.
- Maintained PyOD or DeepOD adapters where licensing, dependencies, and public
  detector semantics are compatible.

Completed implementation requirements:

- Deep dependencies remain outside the base installation.
- Shared PyTorch training utilities provide deterministic seeds, CPU-friendly
  smoke configurations, early stopping, validation splits, and checkpoint
  handling where applicable.
- Parameter presets distinguish `smoke`, `balanced`, and `research` modes.
- Fit-time contamination is exposed only where the underlying method uses it.
- Score orientation and normalization behavior are covered by tests.

Release verification:

- At least two modern tabular detectors have CPU-friendly smoke tests.
- A benchmark configuration compares classical and modern tabular detectors.
- Documentation explains when modern tabular methods may help and when
  classical methods remain preferable.
- No GPU is required for the default tabular smoke suite.
- Missing deep dependencies fail with actionable `ImportError` messages.
- Runtime and randomness are captured in the v0.3 report manifest.

Post-release watchlist:

- DAGMM as an unsupervised deep mixture-model detector.
- DeepSAD as a semi-supervised extension of Deep SVDD.
- PReNet, FEAWAD, and REPEN for labelled or weakly supervised settings.
- GOAD and NeuTraL AD only when they were not already delivered in `v0.5.0`.

## v0.6.x - Distribution And Documentation

Status: released as `v0.6.0` and `v0.6.1` on 2026-09-04.

This milestone was not planned. It exists because making the project
installable from PyPI required a rename, and the rename was a breaking change
that had to be released rather than folded into a detector milestone.

Delivered scope:

- Renamed the distribution and the import namespace to `anomalybench`. The
  previous top-level `analytics` and `benchmarks` packages were too generic to
  claim on PyPI without colliding with unrelated projects. Every module now
  lives under `anomalybench/`, and the detector registry's dotted-path strings
  were rewritten with it.
- Published to PyPI using Trusted Publishing, so releases upload through a
  short-lived OIDC token rather than a stored API token.
- Replaced manual release preparation with release-please: version bumps,
  changelog, and tags are derived from Conventional Commit prefixes, and the
  bump propagates to `.zenodo.json` and `CITATION.cff` through `extra-files`.
- Collapsed `develop` and `main` into a single long-lived branch. Maintaining
  both under `required_linear_history` with squash promotion produced divergence
  that could not be reconciled by merging.
- Added a MkDocs site with narrative guides and an mkdocstrings API reference,
  built with `--strict` in CI and deployed to GitHub Pages from `main`:
  <https://diogoribeiro7.github.io/anomalybench/>.

Known defects, both since fixed:

- `v0.5.1` and `v0.6.1` are no-op releases. The `changelog-sections` config
  marked housekeeping commit types releasable, so `ci:`-only changes cut a
  version, a tag, a PyPI upload, and a Zenodo DOI for changes no user could
  observe. Housekeeping types are now `hidden` and non-releasing.
- `v0.4.0` shipped with an undated "Unreleased" changelog heading. The release
  workflow now rejects a release whose changelog section is undated.

Consequences for later milestones:

- Any citation or benchmark manifest produced before `v0.6.0` refers to import
  paths that no longer exist. The version DOI still resolves, so published
  results remain reproducible against the release they name.
- The documentation site is now a release surface. A detector added without a
  declared `score_orientation`, or with a docstring that griffe cannot parse,
  fails the docs build.

## v0.6.0 - Modern Time-Series Detector Pack

Status: planned. The `v0.6.0` version number was consumed by the distribution
work above, so this milestone ships in the next feature minor. The heading is
kept because later sections and the `v0.5.0` notes refer to it by name.

Goal: add current multivariate and contextual time-series anomaly detection
methods with explicit windowing, metrics, and reproducibility controls, while
using maintained PyOD implementations as reference baselines.

Upstream reference baselines:

- PyOD Matrix Profile, Spectral Residual, KShape, SAND, and LSTMAD.
- PyOD Anomaly Transformer with association discrepancy.
- These methods remain benchmark baselines or adapters; they are not the
  milestone's primary native implementation targets.

Priority implementation or adapter targets:

- TranAD-style transformer reconstruction and adversarial self-conditioning.
- USAD as a lighter adversarial autoencoder baseline.
- OmniAnomaly for stochastic recurrent latent-variable detection.
- DCdetector-style dual-attention contrastive representation learning.
- AnomalyBERT-style masked or self-supervised transformer detection.
- Series2Graph for graph-based subsequence anomaly detection.
- TimesNet-style temporal 2D-variation backbones where an upstream dependency
  or compact implementation is maintainable.
- Lightweight TCN encoder-decoder baseline for CPU-friendly comparison.
  Delivered ahead of schedule in `v0.5.0`.
- M2N2 or FITS as research candidates after the core pair is stable.
- Foundation-model adapters such as MOMENT or Chronos remain optional and may
  move to the experimental foundation-model milestone.

Implementation plan:

- Add a time-series windowing contract that records window length, stride,
  horizon, aggregation policy, and label alignment in benchmark manifests.
- Add dataset-level declarations for point anomalies, range anomalies,
  contextual anomalies, and multivariate channel semantics.
- Add metric policy for point-wise, range-wise, and volume-under-surface
  evaluation.
- Split deep time-series dependencies from other deep extras if PyTorch,
  TensorFlow, and foundation-model dependencies conflict.
- Add deterministic data-loader and training seeds.
- Record whether a detector is a project-native implementation, an upstream
  adapter, or a benchmark-only baseline.

Deliverables:

- Time-series detector API guide.
- At least one transformer-style target not already provided natively by PyOD.
- At least one non-transformer neural target, with USAD or a TCN baseline as the
  preferred CPU-friendly option. Satisfied by the TCN baseline shipped in
  `v0.5.0`.
- Time-series benchmark config comparing project targets with PyOD baselines.
- JSON report fields for windowing, aggregation, provenance, and upstream
  provider version.

Acceptance criteria:

- Small synthetic and real-world time-series smoke datasets run in CI.
- Benchmark output records enough information to reproduce windowed scores.
- Documentation warns against comparing point-adjusted metrics with
  non-adjusted metrics.
- Baseline and native methods are visibly distinguished in reports.
- An upstream PyOD update cannot silently change the benchmark configuration.

## v0.7.0 - Graph Anomaly Detection Pack

Status: planned.

Goal: extend graph anomaly detection beyond PyOD's existing transductive,
node-level reference coverage and support explicit evaluation settings.

Upstream reference baselines:

- PyOD DOMINANT, CoLA, CONAD, AnomalyDAE, GUIDE, Radar, ANOMALOUS, and SCAN.
- PyGOD adapters may be used where they add maintained methods or capabilities
  not exposed through the PyOD adapter surface.
- Existing transductive node detectors are benchmark baselines, not the main
  reason for this milestone.

Priority targets:

- Inductive node anomaly detection with clear train/test graph separation.
- Edge, subgraph, and whole-graph anomaly detection protocols.
- GAE, ONE, DONE, AdONE, GAAN, OCGNN, DMGD, GADNR, or CARD adapters where
  maintained implementations expose compatible semantics.
- Neighborhood-aggregation plus tree-ensemble baselines, motivated by GADBench
  findings that simple aggregated features can outperform specialized GNNs in
  some settings.
- Semi-supervised graph anomaly detection with normal-node labels, including
  GGAD-style pseudo-anomaly generation as an experimental target.
- Dynamic, temporal, heterogeneous, and mini-batch graph detection after static
  inductive support is stable.

Implementation plan:

- Introduce graph dataset metadata for node, edge, subgraph, and graph-level
  labels.
- Define graph detector input contracts for NetworkX and optional PyTorch
  Geometric data objects.
- Add adapters behind a `graph` extra so base users do not install graph deep
  learning dependencies.
- Add benchmark support for transductive, inductive, unsupervised,
  semi-supervised, and supervised graph settings.
- Add graph feature extraction baselines before heavy GNN models.
- Record graph split policy, message-passing access, label access, and whether
  unseen nodes or unseen graphs are evaluated.

Deliverables:

- Graph detector API guide covering task level and inference setting.
- At least one inductive or non-node-level detector capability.
- At least one non-GNN graph baseline.
- Small attributed graph benchmark fixture.
- Graph benchmark report fields for task level, training labels, graph split,
  message-passing access, and transductive or inductive setting.

Acceptance criteria:

- Graph extras install separately from base and time-series extras.
- Graph smoke tests run without requiring a large GPU.
- Reported graph metrics include the evaluation setting, not just the score.
- Transductive results are not presented as evidence of inductive performance.

## v0.8.0 - Streaming, Drift, And Online Detection

Status: planned.

Goal: make online anomaly detection practical for incremental data, delayed
labels, and concept drift.

Candidate methods:

- River-backed Half-Space Trees and related online anomaly models.
- xStream-style random projection streaming anomaly detection.
- Robust Random Cut Forest when a maintained Python dependency is available.
- KitNET for lightweight online ensemble reconstruction.
- IForestASD and KNNCAD-style adaptive streaming detectors.
- Incremental LODA where the update semantics are well defined.
- ADWIN and Page-Hinkley as companion drift diagnostics, not as replacements
  for anomaly detectors.
- Online conformal anomaly scoring for calibrated alert thresholds.

Implementation plan:

- Add `partial_fit`, `score_one`, or a separate streaming detector protocol if
  forcing all online methods through batch `fit` becomes misleading.
- Add stream replay utilities with deterministic event order and timestamps.
- Use prequential evaluation: score an event, record the prediction and
  latency, reveal the label when available, update the detector, and update
  online metrics.
- Distinguish score-before-update from score-after-update behavior.
- Add latency, throughput, peak memory, warm-up period, update frequency,
  adaptation delay, and drift-point metrics.
- Add benchmark configs for warm-up, online scoring, delayed labels, and
  concept-drift recovery.

Deliverables:

- Streaming detector protocol or documented adapter layer.
- Stream benchmark report schema extension.
- CLI mode for replaying stream fixtures.
- Tests for order stability and deterministic seeded streams.
- A prequential benchmark example with delayed labels and at least one drift
  segment.

Acceptance criteria:

- Batch and streaming detector protocols are clearly separated.
- Streaming reports record warm-up, update, delayed-label, and alert-threshold
  policies.
- Reports identify whether each score was produced before or after model update.
- False-alarm rate and recovery delay are reported before and after drift.
- Existing batch benchmarks remain unaffected.

## v0.9.0 - Foundation Models, LLM-Assisted Workflows, And Experimental Methods

Status: planned and experimental.

Goal: evaluate foundation-model and LLM-assisted anomaly detection without
turning experimental research into default behavior or duplicating existing
upstream advisory interfaces.

Candidate methods and studies:

- Time-series foundation-model adapters for MOMENT, Chronos, TimesFM, or later
  maintained open models.
- Zero-shot or few-shot tabular anomaly detection with serialized tables and
  LLM scoring as an experimental plugin.
- Conformal p-value scoring and adaptive conformal thresholds for calibrated
  false-alarm control.
- Foundation-model residuals with conformal calibration for signal monitoring.
- Comparative detector-selection evaluation across fixed heuristics,
  benchmark-based routing, PyOD-style advisory systems, and LLM-assisted
  recommendations.
- Log anomaly detection adapters for structured and semi-structured logs:
  parser-plus-classifier baselines, transformer classifiers, and LogLLM-style
  semantic sequence classifiers.

Detector-selection evaluation:

- Treat recommendation as an evaluated decision system, not merely an LLM
  interface.
- Record the candidate detector set, dataset summary supplied to the selector,
  prompt or policy, unavailable methods rejected, and final parameters.
- Compare recommendations against an oracle chosen from the same benchmark
  candidate set.
- Report selection regret as
  `metric(best benchmark detector) - metric(recommended detector)` for metrics
  where larger values are better, with the sign adjusted for cost metrics.
- Measure recommendation stability, latency, token or inference cost, and
  failure to respect modality, supervision, or dependency constraints.

Implementation plan:

- Keep all model downloads opt-in and cache-controlled.
- Record model name, revision, prompt or template, context window, decoding
  parameters, candidate detector catalogue, and hardware in manifests.
- Add privacy and data-exfiltration warnings for hosted LLM providers.
- Prefer local or open-weight adapters for reproducible research workflows.
- Require deterministic fixtures for prompt serialization and result parsing.
- Validate that a recommended detector exists, is installed, supports the
  dataset modality, and matches the allowed supervision setting.
- Add conformal calibration metadata: calibration split, target false-alarm
  rate, nonconformity score, update policy, and drift handling.
- Add log parsing metadata: parser, template extraction settings, sequence
  window length, sessionization key, and tokenizer or model revision.

Deliverables:

- Experimental namespace or plugin examples for foundation-model detectors.
- A detector-selection benchmark comparing at least one non-LLM policy and one
  LLM-assisted or upstream advisory policy.
- Manifest fields for model revisions, candidate sets, prompts or policy
  fingerprints, conformal calibration, and log parsing.
- Documentation on when these methods are exploratory and not production-ready.
- A small synthetic log anomaly fixture or documented external public log
  loader with licensing notes.

Acceptance criteria:

- No network calls occur during default tests.
- Hosted-model integrations are never enabled by default.
- Reports make model revision, candidate catalogue, and prompt or policy hashes
  visible.
- Recommendation benchmarks report selection regret and invalid-selection rate.
- Conformal outputs expose calibrated p-values or threshold decisions, not only
  raw anomaly scores.
- Log adapters can run a parser-plus-classifier baseline without a hosted LLM.

## v0.10.0 - Plugin And Extension Ecosystem

Goal: make external detector and dataset extensions practical.

Planned work:

- Formalize plugin loading contracts for detectors, datasets, metrics, reports,
  and visualization hooks.
- Add plugin metadata validation and collision reporting.
- Require plugin detector provenance, upstream provider information, and tested
  compatibility ranges.
- Support external dataset loader registration with integrity metadata.
- Add examples for a minimal detector plugin, dataset plugin, metric plugin,
  and report plugin.
- Add safer plugin import diagnostics in the CLI.
- Document compatibility expectations for plugin authors.

Deliverables:

- Plugin author guide.
- Example plugin package under documentation or examples.
- Tests for plugin registration, override behavior, and failure messages.

Acceptance criteria:

- Plugins can be loaded from config files and CLI flags.
- Registry collisions produce actionable diagnostics.
- Plugin metadata is included in benchmark manifests.

## v0.11.0 - Visualization And Reporting

Goal: turn benchmark outputs into useful reports for inspection and comparison.

Planned work:

- Add report generation from leaderboard or JSON benchmark outputs.
- Add plots for ROC curves, precision-recall curves, score distributions,
  runtime comparisons, calibration curves, and detector failure summaries.
- Add time-series visualizations for anomaly ranges, alert thresholds, and
  window aggregation.
- Add graph visualizations for node scores on small graph fixtures.
- Add dataset summary exports in Markdown and JSON.
- Improve visualization tests so plotting code is stable in headless CI.
- Keep plotting optional enough that non-report workflows remain lightweight.

Deliverables:

- CLI report command or report mode.
- Documented output examples.
- Tested visualization helpers for common benchmark outputs.

Acceptance criteria:

- Report generation works from a saved v0.3+ JSON report.
- Plots include metric, threshold, and score-orientation context.
- Headless CI verifies that report artifacts are created.

## v0.12.0 - Visual And Multimodal Anomaly Detection

Goal: add image, industrial inspection, and multimodal anomaly detection without
making computer-vision dependencies part of the base install. Explicitly
separate sample-level embedding detection from patch-level or pixel-level
industrial anomaly localisation.

Candidate methods:

- PatchCore-style memory-bank nearest-neighbor scoring for industrial defect
  detection.
- PaDiM-style patch distribution modeling with pretrained CNN features.
- FastFlow-style normalizing-flow density scoring on visual features.
- DRAEM-style synthetic anomaly reconstruction and segmentation.
- Reverse-distillation methods such as RD4AD.
- EfficientAD-style compact student-teacher detectors for practical industrial
  inspection.
- CLIP or VLM-based zero-shot visual anomaly detection, including WinCLIP-style
  prompt/image feature scoring.
- Multimodal detectors that combine image features with tabular metadata,
  time-series context, logs, or graph attributes.

Implementation plan:

- Introduce an `vision` or `multimodal` optional extra with explicit CPU/GPU
  guidance.
- Define an image detector input contract: image path, array or tensor, batch
  loader, optional segmentation mask, and optional product or category
  metadata.
- Record whether the method produces only one sample-level score or also emits
  patch-level and pixel-level anomaly maps.
- Add dataset metadata for image-level labels, pixel-level masks, product
  category, resolution, license, and download instructions.
- Record feature extractor model, checkpoint revision, image preprocessing,
  prompt templates, and patch/embedding parameters in benchmark manifests.
- Add a tiny synthetic image anomaly fixture for CI and document larger
  industrial datasets as externally fetched benchmark profiles.
- Keep zero-shot VLM methods experimental until prompts, checkpoints, and
  preprocessing are pinned.

Deliverables:

- Visual detector API guide.
- At least one classical feature-memory detector and one pretrained-feature
  baseline.
- Multimodal report fields for image metadata, mask availability, and feature
  extractor revision.
- Example visual benchmark config that runs on a tiny fixture.

Acceptance criteria:

- Default tests do not download pretrained weights.
- Visual smoke tests can run without a large GPU.
- Image-level AUROC and average precision are reported separately from
  pixel-level AUROC, pixel-level average precision, and AU-PRO or AUPRO.
- Reports include localisation latency, memory-bank size where applicable,
  inference throughput, preprocessing, and checkpoint fingerprints.
- Embedding-level multimodal scoring is not presented as equivalent to defect
  localisation.

## v0.13.0 - Diffusion Models And Anomaly Synthesis

Goal: evaluate diffusion-based detection and anomaly generation as experimental
methods for high-dimensional and low-defect regimes.

Candidate methods:

- Reconstruction-based diffusion anomaly scoring using denoising error.
- Density or likelihood-style scoring with diffusion or score-based models.
- Hybrid diffusion detectors that combine reconstruction, feature distance, and
  uncertainty.
- Diffusion-based synthetic anomaly generation for visual, time-series, tabular,
  and multimodal data.
- Conditional diffusion models for product category, operating state, or normal
  context.
- Lightweight adapters for externally maintained diffusion anomaly detection
  implementations when licensing and reproducibility are acceptable.

Implementation plan:

- Keep diffusion dependencies isolated behind a `diffusion` or `vision` extra.
- Record scheduler, denoising steps, noise schedule, checkpoint revision,
  conditioning inputs, sampling seed, and synthesis policy in manifests.
- Add a strict distinction between detection benchmarks and synthetic-data
  augmentation experiments.
- Add synthetic anomaly provenance fields so generated anomalies cannot be
  confused with real labels.
- Add runtime budgets because diffusion scoring can be expensive.

Deliverables:

- Experimental diffusion detector namespace.
- Anomaly synthesis utilities or documented plugin hooks.
- Benchmark config for a tiny image or time-series diffusion smoke fixture.
- Report fields for generated-anomaly provenance and diffusion configuration.

Acceptance criteria:

- Diffusion methods are never part of base install.
- Reports make generated versus real anomalies explicit.
- Smoke tests use tiny models or mocked adapters, not large checkpoint
  downloads.
- Runtime budget controls prevent accidental long benchmark runs.

## v0.14.0 - Causal Detection And Test-Time Adaptation

Goal: support changing normal behavior and causal/contextual explanations
without overstating causal guarantees.

Candidate methods:

- Causal graph residual scoring for tabular or time-series systems with known
  dependency structure.
- Environment-aware anomaly detection using invariance across operating
  regimes.
- Causal ordering or dependency-constrained tabular anomaly detection.
- Test-time adaptation for detectors facing new normal regimes.
- Drift-aware adaptation policies that separate retraining, calibration, and
  threshold updates.
- Conformal or uncertainty-aware adaptation layers for controlled false alarm
  behavior under drift.

Implementation plan:

- Add metadata for environment, regime, intervention, causal graph source, and
  adaptation policy.
- Keep causal graph discovery out of core scope unless a maintained dependency
  and clear validation plan exist.
- Separate explanation fields from detection scores so explanations do not
  silently affect benchmark comparability.
- Add drift/adaptation benchmark fixtures with pre-change, adaptation, and
  post-change segments.
- Record whether adaptation uses labels, normal-only samples, pseudo-labels, or
  unlabeled test data.

Deliverables:

- Causal/contextual detector guide.
- Test-time adaptation protocol for batch and time-series detectors.
- Report schema extension for regime and adaptation metadata.
- Synthetic drift benchmark config with deterministic seeds.

Acceptance criteria:

- Reports state the adaptation setting and data access assumptions.
- Causal methods expose assumptions and graph provenance.
- Benchmarks distinguish no-adaptation, calibration-only, and model-adaptation
  results.

## v1.0.0 - Stable Public Release

Goal: declare the public API and benchmark formats stable enough for downstream
users to depend on.

Requirements:

- Detector API is documented and covered by compatibility tests.
- Batch, streaming, graph, and time-series contracts are stable or clearly
  marked experimental.
- Visual, multimodal, diffusion, causal, log, and foundation-model tracks may
  remain optional and explicitly experimental; they are not required to block a
  stable core release.
- CLI configuration schema is documented and versioned.
- Leaderboard, manifest, and JSON report schemas are documented and versioned.
- Plugin contracts are documented and covered by tests.
- Supported Python versions are clearly declared and validated in CI.
- Release workflow is repeatable from the GitHub Actions dashboard.
- Zenodo citation metadata is current for the release.
- At least one benchmark config exists for tabular, time-series, graph, and
  streaming workflows.
- The upstream capability snapshot and implementation-provenance schema are
  documented and covered by compatibility tests.

Non-goals for `v1.0.0`:

- Supporting every anomaly detection algorithm.
- Bundling large datasets directly in the repository.
- Making all optional detector stacks part of the base install.
- Treating experimental foundation-model detectors as stable APIs.
- Requiring visual, multimodal, diffusion, causal, or hosted LLM dependencies
  for the base install.

## Research Watchlist

These references guide method selection. Inclusion here does not mean the
method is committed to a release; it means the project should track it and
evaluate dependency, license, and benchmark fit.

- Time-series transformers and representation learning:
  [Anomaly Transformer](https://arxiv.org/abs/2110.02642) as an upstream
  reference baseline,
  [TranAD](https://arxiv.org/abs/2201.07284),
  [TimesNet](https://arxiv.org/abs/2210.02186),
  [AnomalyBERT](https://arxiv.org/abs/2305.04468),
  [DCdetector](https://arxiv.org/abs/2306.10347), USAD, OmniAnomaly,
  Series2Graph, M2N2, and FITS.
- Time-series foundation models and benchmark concerns:
  [MOMENT](https://arxiv.org/abs/2402.03885),
  [TimeSeriesBench](https://arxiv.org/html/2402.10802v1), and
  [Towards A Reliable Time-Series Anomaly Detection Benchmark](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html).
- Tabular modern methods and tooling:
  DAGMM, DeepSAD, PReNet, FEAWAD, REPEN,
  [Retrieval Augmented Deep Anomaly Detection for Tabular Data](https://arxiv.org/abs/2401.17052),
  [Anomaly Detection of Tabular Data Using LLMs](https://arxiv.org/abs/2406.16308),
  [PyOD 2](https://arxiv.org/abs/2412.12154), and
  [DeepOD](https://github.com/xuhongzuo/deepod).
- Streaming and online detection: Half-Space Trees, xStream, Robust Random
  Cut Forest, KitNET, IForestASD, KNNCAD, incremental LODA, prequential
  evaluation, and delayed-label conformal calibration.
- Graph anomaly detection:
  [PyGOD](https://www.jmlr.org/papers/v25/23-0963.html),
  [GADBench](https://arxiv.org/abs/2306.12251),
  [GGAD](https://arxiv.org/abs/2402.11887), and
  [Deep Graph Anomaly Detection: A Survey and New Perspectives](https://arxiv.org/abs/2409.09957).
- Visual and industrial anomaly detection:
  [A Survey on Visual Anomaly Detection](https://arxiv.org/abs/2401.16402),
  [A survey of deep learning for industrial visual anomaly detection](https://link.springer.com/article/10.1007/s10462-025-11287-7),
  and [Critical Analysis and Best Practices for Visual Industrial Anomaly Detection](https://arxiv.org/html/2503.23451v1).
- Diffusion-based anomaly detection and anomaly synthesis:
  [A Survey on Diffusion Models for Anomaly Detection](https://arxiv.org/abs/2501.11430),
  [Anomaly Detection and Generation with Diffusion Models](https://arxiv.org/abs/2506.09368),
  and [A Survey on Industrial Anomalies Synthesis](https://arxiv.org/html/2502.16412v1).
- Log, event, and text anomaly detection:
  [LogLLM](https://arxiv.org/abs/2411.08561),
  [LLM-Enhanced Log Anomaly Detection](https://arxiv.org/abs/2604.12218),
  and [Large Language Models for Anomaly and Out-of-Distribution Detection](https://arxiv.org/abs/2409.01980).
- Conformal, causal, and adaptation-oriented anomaly detection:
  [Adaptive Conformal Anomaly Detection with Time Series Foundation Models](https://arxiv.org/abs/2604.20122),
  [CausalTAD](https://arxiv.org/html/2602.07798v1),
  and test-time adaptation work for changing normal regimes.

## Backlog

These items are useful but not yet assigned to a target version.

- Add documentation site generation.
- Add API docs from docstrings.
- Add benchmark result comparison across GitHub releases.
- Add model persistence helpers for detectors that support serialization.
- Evaluate Python `3.13` support once the dependency stack is stable.
- Add automated release note generation from merged pull requests.
- Add DOI badges once the preferred badge target is finalized.
- Add security guidance for benchmark datasets that may contain sensitive or
  operational telemetry.
- Add dependency compatibility matrix for optional extras.
- Automate a reviewed upstream capability diff without automatically changing
  detector classifications or benchmark defaults.
- Add benchmark budget controls for maximum runtime, memory, and GPU use.

## Contribution Guidance

When proposing roadmap work, include:

- Target version.
- User-facing behavior.
- Dependency impact.
- Testing plan.
- Documentation impact.
- Migration risk, if any.
- Citation or upstream implementation link for new research methods.
- Implementation classification: `native`, `adapter`, `baseline`,
  `experimental`, `benchmark-only`, or `watchlist`.
- Upstream provider and tested compatibility range for adapters and baselines.
- Benchmark configuration proving the method can run in the intended
  environment.

Small, well-tested pull requests are preferred over broad rewrites. Roadmap
items can move between versions when implementation risk, dependency support,
or research quality changes.
