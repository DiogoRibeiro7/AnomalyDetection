"""Dataset loading helpers used throughout the benchmarking utilities."""

from __future__ import annotations

from benchmarks.catalog import get_dataset_functions, get_dataset_metadata


def load_all_datasets(names=None):
    """Load benchmark datasets by their canonical names.

    Parameters
    ----------
    names: list[str] | None
        Optional list of canonical dataset identifiers (e.g. ``"iris"``).
        ``None`` loads every dataset available in :mod:`benchmarks.load_datasets`.
    """

    functions = get_dataset_functions()
    if names is None:
        selected = list(functions.items())
    else:
        missing = [name for name in names if name not in functions]
        if missing:
            available = ", ".join(sorted(functions))
            missing_str = ", ".join(missing)
            raise KeyError(
                "Unknown dataset selector(s): "
                f"{missing_str}. Available datasets: {available}"
            )
        selected = [(name, functions[name]) for name in names]

    datasets = []
    for key, func in selected:
        df, feature_cols, label_col, display_name = func()
        datasets.append(
            {
                "dataframe": df,
                "feature_cols": feature_cols,
                "label_col": label_col,
                "name": display_name,
                "key": key,
                "metadata": get_dataset_metadata(key),
            }
        )

    return datasets
