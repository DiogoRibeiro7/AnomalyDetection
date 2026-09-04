# Contributing

## Setup

```bash
git clone https://github.com/DiogoRibeiro7/anomalybench
cd anomalybench
poetry install
poetry run pre-commit install
```

Python 3.12 only — 3.13 is not yet supported.

## Quality gates

CI runs the same three checks on every pull request, with pinned tool versions
so a suppression that is required locally is not flagged as unused in CI:

```bash
poetry run ruff check .
poetry run mypy anomalybench src tests
poetry run pytest -q --cov=anomalybench --cov-report=term-missing --cov-fail-under=71
```

!!! warning "Use `poetry run`, and make sure it resolves to the venv"
    `poetry run mypy` will silently fall through to a `mypy` on `PATH` if one is
    not installed in the virtualenv. That produces results from a different
    version than CI uses, which looks like unexplained drift. Install the pinned
    tools into the venv:

    ```bash
    poetry run pip install ruff==0.16.1 mypy==1.20.2 pytest-cov==7.1.0 \
      types-PyYAML==6.0.12.20260815
    ```

## Adding a detector

1. Subclass `BaseDetector` in the appropriate module under
   `anomalybench/analytics/detectors/`.
2. Implement `fit` and `score`, and **declare `score_orientation`**. Leaving it
   at the `estimator_defined` default means benchmark evaluation refuses the
   detector.
3. Register it in `anomalybench/analytics/detectors/__init__.py` with a dotted
   `module:Class` path — the registry is lazy, so a detector needing a heavy
   framework costs nothing until it is selected.
4. Add tests covering the behaviour and the edge cases.
5. Update the README and these docs when user-facing capability changes.

!!! note "Registrations are strings, not imports"
    The entries in the registry are dotted-path strings. A refactor that renames
    a module will not break them at import time, and the linter will not catch
    them either — the failure appears only when that detector is selected.

## Adding a dataset

1. Put compact, redistributable assets under `anomalybench/benchmarks/`.
2. Implement a loader in `load_datasets.py` returning
   `(dataframe, feature_columns, label_column, display_name)`.
3. Describe it in `datasets.yml` with tags (`tabular`, `graph`, `time_series`)
   and source metadata.
4. Cover it in `tests/test_benchmark_catalog.py`.

## Commit messages

The project uses [Conventional Commits] — release-please derives the version
bump and the changelog from them, so the prefix is functional, not cosmetic:

| Prefix | Effect |
| --- | --- |
| `feat:` | minor bump, "Added" section |
| `fix:` | patch bump, "Fixed" section |
| `perf:` | patch bump, "Performance" section |
| `chore:`, `ci:`, `docs:`, `test:`, `refactor:`, `style:`, `build:` | no release |

[Conventional Commits]: https://www.conventionalcommits.org/

Housekeeping prefixes are deliberately non-releasing. Marking them releasable
cuts versions for commits that changed nothing a user can observe.

## Documentation

```bash
poetry run pip install mkdocs==1.6.1 mkdocs-material==9.7.7 \n  mkdocstrings[python]==1.0.6
poetry run mkdocs serve          # live preview on localhost:8000
poetry run mkdocs build --strict # what CI runs
```

`--strict` turns broken links and unresolved mkdocstrings references into
failures. Run it before opening a PR.
