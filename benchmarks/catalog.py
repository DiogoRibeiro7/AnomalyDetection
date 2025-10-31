"""Dataset catalog metadata and selection utilities."""

from __future__ import annotations

from functools import lru_cache
from inspect import getmembers, isfunction
from pathlib import Path
from typing import Iterable, List

import yaml

import benchmarks.load_datasets

_CATALOG_PATH = Path(__file__).with_name("datasets.yml")


@lru_cache(maxsize=1)
def get_dataset_functions() -> dict[str, callable]:
    """Return mapping of canonical dataset names to loader callables."""

    functions: dict[str, callable] = {}
    for func_name, func in getmembers(benchmarks.load_datasets, isfunction):
        if not func_name.startswith("load_"):
            continue
        key = func_name.replace("load_", "")
        functions[key] = func
    return functions


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, dict[str, object]]:
    """Load the dataset metadata catalog from disk."""

    if not _CATALOG_PATH.exists():
        return {}
    with _CATALOG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    raw_catalog = data.get("datasets", {})
    return {str(name): (details or {}) for name, details in raw_catalog.items()}


def get_dataset_metadata(name: str) -> dict[str, object]:
    """Return metadata for *name* from the catalog."""

    return load_catalog().get(name, {})


def list_available_datasets() -> List[str]:
    """Return the canonical dataset names discovered in the loaders."""

    return list(get_dataset_functions().keys())


def resolve_dataset_names(selectors) -> List[str] | None:
    """Resolve *selectors* into canonical dataset names.

    The *selectors* argument accepts strings (dataset names or ``tag:<name>``),
    dictionaries with ``include``/``exclude``/``limit`` keys, or iterables of
    those forms. ``None`` returns ``None`` to signal "all datasets".
    """

    available = list_available_datasets()
    if selectors is None:
        return None

    if isinstance(selectors, (list, tuple)):
        combined: List[str] = []
        for entry in selectors:
            names = resolve_dataset_names(entry)
            if names is None:
                names = available.copy()
            combined.extend(names)
        return _dedupe(combined)

    if isinstance(selectors, dict):
        keys = set(selectors)
        if {"include", "exclude", "limit"} & keys:
            include_spec = selectors.get("include")
            if include_spec is None:
                included = available.copy()
            else:
                included = resolve_dataset_names(include_spec) or []
            exclude_spec = selectors.get("exclude")
            excluded = set(resolve_dataset_names(exclude_spec) or [])
            resolved = [name for name in included if name not in excluded]
            limit = selectors.get("limit")
            if isinstance(limit, int) and limit > 0:
                resolved = resolved[:limit]
            return _dedupe(resolved)
        if "name" in selectors:
            return resolve_dataset_names(selectors["name"]) or []
        if "tag" in selectors:
            return _expand_tag(selectors["tag"], available)
        return []

    if isinstance(selectors, str):
        if selectors.startswith("tag:"):
            return _expand_tag(selectors.split(":", 1)[1], available)
        normalized = selectors.replace("load_", "")
        return [normalized]

    return []


def _expand_tag(tag: str, available: Iterable[str]) -> List[str]:
    matches = [
        name
        for name, meta in load_catalog().items()
        if tag in (meta.get("tags") or [])
    ]
    return [name for name in matches if name in available]


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered
