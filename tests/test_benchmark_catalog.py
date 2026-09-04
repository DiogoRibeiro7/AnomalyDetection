"""Tests for the dataset catalog and selection utilities."""

from __future__ import annotations

from anomalybench.analytics.exceptions import UnknownDatasetError
from anomalybench.benchmarks import catalog
from anomalybench.benchmarks.load_all_datasets import load_all_datasets


def test_catalog_includes_new_datasets() -> None:
    names = catalog.list_available_datasets()
    assert "thyroid" in names
    assert "kddcup_sample" in names
    assert "fashion_mnist_sample" in names
    assert "nab_art_daily_small_noise" in names
    assert "nab_machine_temperature" in names
    thyroid_meta = catalog.get_dataset_metadata("thyroid")
    thyroid_tags = thyroid_meta.get("tags")
    assert isinstance(thyroid_tags, list)
    assert "tabular" in thyroid_tags


def test_resolve_dataset_names_by_tag() -> None:
    names = catalog.resolve_dataset_names("tag:graph")
    assert names is not None
    assert "karate_club_graph" in names
    assert "thyroid" not in names


def test_resolve_dataset_names_with_limit_and_exclude() -> None:
    selection = {
        "include": ["tag:tabular"],
        "exclude": ["iris"],
        "limit": 3,
    }
    names = catalog.resolve_dataset_names(selection)
    assert names is not None
    assert "iris" not in names
    assert len(names) == 3


def test_resolve_dataset_names_with_metadata_filters() -> None:
    tabular_names = catalog.resolve_dataset_names(
        {"modality": "tabular", "task": "classification", "limit": 5}
    )
    assert tabular_names
    assert "karate_club_graph" not in tabular_names
    assert all(
        catalog.get_dataset_metadata(name).get("modality") == "tabular"
        for name in tabular_names
    )


def test_resolve_dataset_names_preserves_unknown() -> None:
    names = catalog.resolve_dataset_names("custom_dataset")
    assert names == ["custom_dataset"]


def test_resolve_dataset_names_supports_display_aliases() -> None:
    names = catalog.resolve_dataset_names("wisconsinBreast")
    assert names == ["wisconsin_breast_cancer"]
    nab_names = catalog.resolve_dataset_names("nabArtDaily")
    assert nab_names == ["nab_art_daily_small_noise"]
    machine_names = catalog.resolve_dataset_names("machine_temperature")
    assert machine_names == ["nab_machine_temperature"]


def test_load_all_datasets_includes_metadata() -> None:
    datasets = load_all_datasets(["thyroid"])
    assert datasets
    entry = datasets[0]
    assert entry["metadata"]["tags"]
    assert entry["key"] == "thyroid"


def test_load_all_datasets_raises_on_unknown() -> None:
    try:
        load_all_datasets(["not_a_dataset"])
    except UnknownDatasetError as exc:
        assert "Unknown dataset" in str(exc)
        assert exc.identifier == "not_a_dataset"
    else:  # pragma: no cover - defensive fallback
        raise AssertionError("Expected KeyError for unknown dataset selector")


def test_build_alias_map_does_not_invoke_dataset_loaders(monkeypatch) -> None:
    def _loader_should_not_run():
        raise AssertionError("Loader must not be called while building aliases")

    monkeypatch.setattr(
        catalog, "get_dataset_functions", lambda: {"demo": _loader_should_not_run}
    )
    monkeypatch.setattr(
        catalog,
        "load_catalog",
        lambda: {
            "demo": {
                "display_name": "Demo Dataset",
                "aliases": ["demoAlias"],
            }
        },
    )
    catalog._build_alias_map.cache_clear()
    alias_map = catalog._build_alias_map()
    assert alias_map["demo"] == "demo"
    assert alias_map["demo dataset"] == "demo"
    assert alias_map["demo_dataset"] == "demo"
    assert alias_map["demoAlias"] == "demo"
    catalog._build_alias_map.cache_clear()
