from pathlib import Path


def test_load_datasets_does_not_use_deprecated_iteritems() -> None:
    source = Path("anomalybench/benchmarks/load_datasets.py").read_text(
        encoding="utf-8"
    )
    assert "iteritems(" not in source
