"""
Uses python introspection to call all function in `data.load_datasets`

Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.
"""

from inspect import getmembers, isfunction
import benchmarks.load_datasets


def load_all_datasets(names=None):
    """Load benchmark datasets.

    Parameters
    ----------
    names: list[str] | None
        Optional list of dataset function suffixes (e.g. ``"iris"``) to
        restrict which datasets are loaded.
    """
    datasets = []
    for func_name, func in getmembers(benchmarks.load_datasets, isfunction):
        if names and func_name.replace("load_", "") not in names:
            continue
        df, feature_cols, label_col, name = func()
        datasets.append(
            {
                "dataframe": df,
                "feature_cols": feature_cols,
                "label_col": label_col,
                "name": name,
            }
        )

    return datasets
