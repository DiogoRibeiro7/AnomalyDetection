from cli import run_benchmarks


def test_run_benchmarks_subset():
    # Should run without raising and produce results for iris dataset
    run_benchmarks(datasets=["iris"], detectors=["isolation_forest"])
