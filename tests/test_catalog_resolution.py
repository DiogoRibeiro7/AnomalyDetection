"""Selector resolution and alias handling in the dataset catalog.

``test_benchmark_catalog`` covers the common selectors. The alias map, the
``tag:`` prefix, the recursive list form, and the fallbacks for unusable input
are pinned here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from anomalybench.benchmarks import catalog


def test_missing_catalog_file_yields_an_empty_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(catalog, "_CATALOG_PATH", tmp_path / "absent.yml")
    catalog.load_catalog.cache_clear()

    try:
        assert catalog.load_catalog() == {}
    finally:
        catalog.load_catalog.cache_clear()


def test_none_selector_means_every_dataset() -> None:
    """``None`` is the caller's way of saying 'do not filter'."""

    assert catalog.resolve_dataset_names(None) is None


@pytest.mark.parametrize("selector", [5, 1.5, object()])
def test_unusable_selector_types_resolve_to_nothing(selector: Any) -> None:
    assert catalog.resolve_dataset_names(selector) == []


def test_empty_mapping_resolves_to_nothing() -> None:
    assert catalog.resolve_dataset_names({}) == []


def test_name_key_resolves_a_single_dataset() -> None:
    assert catalog.resolve_dataset_names({"name": "iris"}) == ["iris"]


def test_tag_key_expands_to_the_tagged_datasets() -> None:
    assert catalog.resolve_dataset_names({"tag": "graph"}) == ["karate_club_graph"]


def test_tag_prefix_string_expands_the_same_way() -> None:
    assert catalog.resolve_dataset_names("tag:graph") == ["karate_club_graph"]


def test_display_name_alias_maps_to_the_canonical_key() -> None:
    assert catalog.resolve_dataset_names("wisconsinBreast") == [
        "wisconsin_breast_cancer"
    ]


def test_alias_lookup_is_case_insensitive() -> None:
    assert catalog.resolve_dataset_names("wisconsinbreast") == [
        "wisconsin_breast_cancer"
    ]


def test_load_prefix_is_stripped_before_lookup() -> None:
    assert catalog.resolve_dataset_names("load_iris") == ["iris"]


def test_unknown_selector_warns_and_is_passed_through(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The caller decides what to do; the catalog only reports the miss."""

    with caplog.at_level(logging.WARNING, logger=catalog.logger.name):
        resolved = catalog.resolve_dataset_names("not_a_dataset")

    assert resolved == ["not_a_dataset"]
    assert "not_a_dataset" in caplog.text


def test_a_list_selector_resolves_each_entry_and_deduplicates() -> None:
    resolved = catalog.resolve_dataset_names(["iris", "iris", "cardio"])
    assert resolved == ["iris", "cardio"]


def test_a_none_entry_inside_a_list_expands_to_every_dataset() -> None:
    available = catalog.list_available_datasets()
    resolved = catalog.resolve_dataset_names(["iris", None])

    assert resolved is not None
    assert set(resolved) == set(available)


def test_include_and_exclude_are_applied_with_a_limit() -> None:
    resolved = catalog.resolve_dataset_names(
        {"include": ["tag:tabular"], "exclude": ["iris"], "limit": 2}
    )

    assert resolved is not None
    assert "iris" not in resolved
    assert len(resolved) == 2


def test_metadata_for_an_unknown_dataset_is_empty() -> None:
    assert catalog.get_dataset_metadata("not_a_dataset") == {}
