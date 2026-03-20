"""Dataset catalog metadata and selection utilities."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from inspect import getmembers, isfunction
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeAlias, TypedDict

import yaml

logger = logging.getLogger(__name__)

import benchmarks.load_datasets

__all__ = [
    "DatasetLoader",
    "DatasetRegistry",
    "DatasetSpec",
    "get_dataset_functions",
    "load_catalog",
    "get_dataset_metadata",
    "list_available_datasets",
    "resolve_dataset_names",
]

_CATALOG_PATH = Path(__file__).with_name("datasets.yml")

DatasetLoader: TypeAlias = Callable[[], tuple[Any, list[str] | None, str, str]]
DatasetRegistry: TypeAlias = dict[str, DatasetLoader]


class DatasetSpec(TypedDict):
    """Structured dataset entry returned by ``load_all_datasets``."""

    dataframe: Any
    feature_cols: list[str] | None
    label_col: str
    name: str
    key: str
    metadata: dict[str, object]


@lru_cache(maxsize=1)
def get_dataset_functions() -> DatasetRegistry:
    """Return mapping of canonical dataset names to loader callables."""

    functions: dict[str, DatasetLoader] = {}
    for func_name, func in getmembers(benchmarks.load_datasets, isfunction):
        if not func_name.startswith("load_"):
            continue
        key = func_name.replace("load_", "")
        functions[key] = func  # type: ignore[assignment]
    return functions


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, dict[str, object]]:
    """Load the dataset metadata catalog from disk."""

    if not _CATALOG_PATH.exists():
        return {}
    with _CATALOG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    raw_catalog = data.get("datasets", {})
    catalog: dict[str, dict[str, object]] = {}
    for name, details in raw_catalog.items():
        meta = dict(details or {})
        aliases = meta.get("aliases") or []
        if isinstance(aliases, (list, tuple)):
            meta["aliases"] = [str(alias) for alias in aliases if alias]
        else:
            meta["aliases"] = []
        catalog[str(name)] = meta
    return catalog


def get_dataset_metadata(name: str) -> dict[str, object]:
    """Return metadata for *name* from the catalog."""

    return load_catalog().get(name, {})


def list_available_datasets() -> List[str]:
    """Return the canonical dataset names discovered in the loaders."""

    return list(get_dataset_functions().keys())


def resolve_dataset_names(selectors: Any) -> List[str] | None:
    """Resolve *selectors* into canonical dataset names.

    The *selectors* argument accepts strings (dataset names or ``tag:<name>``),
    dictionaries with ``include``/``exclude``/``limit`` keys, or iterables of
    those forms. ``None`` returns ``None`` to signal "all datasets".
    """

    available = list_available_datasets()
    alias_map = _build_alias_map()
    if selectors is None:
        return None

    if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes)):
        combined: List[str] = []
        for entry in selectors:
            names = resolve_dataset_names(entry)
            if names is None:
                names = available.copy()
            combined.extend(names)
        return _dedupe(combined)

    if isinstance(selectors, Mapping):
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
        canonical = _lookup_alias(normalized, alias_map)
        if canonical:
            return [canonical]
        if normalized not in available:
            logger.warning("Unknown dataset selector '%s'", selectors)
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


@lru_cache(maxsize=1)
def _build_alias_map() -> Dict[str, str]:
    """Return mapping of known dataset aliases to canonical loader keys."""

    alias_map: Dict[str, str] = {}
    for canonical in list_available_datasets():
        alias_map[canonical] = canonical
        alias_map[canonical.lower()] = canonical

    catalog = load_catalog()
    for canonical, meta in catalog.items():
        display_name = meta.get("display_name")
        if isinstance(display_name, str) and display_name:
            alias_map.setdefault(display_name, canonical)
            alias_map.setdefault(display_name.lower(), canonical)
            alias_map.setdefault(display_name.replace(" ", "_").lower(), canonical)
        aliases = meta.get("aliases") or []
        if isinstance(aliases, Iterable):
            for alias in aliases:
                if not isinstance(alias, str) or not alias:
                    continue
                alias_map.setdefault(alias, canonical)
                alias_map.setdefault(alias.lower(), canonical)
    return alias_map


def _lookup_alias(name: str, alias_map: Dict[str, str]) -> str | None:
    if not name:
        return None
    return alias_map.get(name) or alias_map.get(name.lower())
