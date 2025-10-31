"""Tests for the dataset catalog and selection utilities."""

from __future__ import annotations

from benchmarks import catalog
from benchmarks.load_all_datasets import load_all_datasets


def test_catalog_includes_new_datasets() -> None:
    names = catalog.list_available_datasets()
    assert "thyroid" in names
    assert "kddcup_sample" in names
    thyroid_meta = catalog.get_dataset_metadata("thyroid")
    assert "tabular" in thyroid_meta.get("tags", [])


def test_resolve_dataset_names_by_tag() -> None:
    names = catalog.resolve_dataset_names("tag:graph")
    assert "karate_club_graph" in names
    assert "thyroid" not in names


def test_resolve_dataset_names_with_limit_and_exclude() -> None:
    selection = {
        "include": ["tag:tabular"],
        "exclude": ["iris"],
        "limit": 3,
    }
    names = catalog.resolve_dataset_names(selection)
    assert "iris" not in names
    assert len(names) == 3


def test_resolve_dataset_names_preserves_unknown() -> None:
    names = catalog.resolve_dataset_names("custom_dataset")
    assert names == ["custom_dataset"]


def test_load_all_datasets_includes_metadata() -> None:
    datasets = load_all_datasets(["thyroid"])
    assert datasets
    entry = datasets[0]
    assert entry["metadata"]["tags"]
    assert entry["key"] == "thyroid"
