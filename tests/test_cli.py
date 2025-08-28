import pytest

pytest.importorskip("networkx")

from cli import run_benchmarks, load_plugins, summarize_datasets


def test_run_benchmarks_subset():
    # Should run without raising and produce results for iris dataset
    run_benchmarks(datasets=["iris"], detectors=["isolation_forest"])


def test_summarize_outputs(capsys):
    summarize_datasets(["iris"])
    out = capsys.readouterr().out
    assert "Dataset: iris" in out


def test_plugin_restriction():
    with pytest.raises(ValueError):
        load_plugins(["os"])
