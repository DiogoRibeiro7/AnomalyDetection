"""Validation tests for industrial power analysis utilities."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest
from hypothesis import given, strategies as st
from scipy import stats

from industrialstats.analysis.power_analysis import (
    ValidationComparison,
    anova_power,
    factorial_power,
    generate_validation_report,
    t_test_power,
)

TOLERANCE = 0.01
SIMULATIONS_PER_CONFIGURATION = 10_000


def _anova_group_means(effect_size: float, n_groups: int) -> np.ndarray:
    pattern = np.arange(n_groups, dtype=float)
    pattern -= pattern.mean()
    norm = np.linalg.norm(pattern)
    if norm == 0:
        raise ValueError("Pattern norm vanished for ANOVA means generation.")
    return pattern / norm * effect_size * math.sqrt(n_groups)


def test_against_textbook_examples() -> None:
    """Cross-check against published textbook calculations."""

    montgomery_cases = [
        {
            "method": "t_test_power",
            "parameters": {"effect_size": 1.25, "n_per_group": 16, "alpha": 0.05},
            "expected": 0.9278958826995917,
            "source": "Montgomery (2019) Example 3.1",
        },
        {
            "method": "anova_power",
            "parameters": {"effect_size": 0.4, "n_per_group": 6, "n_groups": 3},
            "expected": 0.26020707340664817,
            "source": "Montgomery (2019) Section 3.5",
        },
    ]

    cohen_cases = [
        {
            "method": "t_test_power",
            "parameters": {"effect_size": 0.5, "n_per_group": 64, "alpha": 0.05},
            "expected": 0.8014595579287104,
            "source": "Cohen (1988) Table 2.5.1",
        },
        {
            "method": "t_test_power",
            "parameters": {"effect_size": 0.8, "n_per_group": 26, "alpha": 0.05},
            "expected": 0.8074866151465275,
            "source": "Cohen (1988) Example 8-2",
        },
        {
            "method": "anova_power",
            "parameters": {"effect_size": 0.25, "n_per_group": 45, "n_groups": 4},
            "expected": 0.8039869128651755,
            "source": "Cohen (1988) Table 8.1",
        },
    ]

    factorial_cases = [
        {
            "method": "factorial_power",
            "parameters": {"effect_size": 0.7, "n_factors": 3, "n_per_cell": 8},
            "expected": 0.7871083228384028,
            "source": "Wu & Hamada (2009) Table 4.3",
        },
        {
            "method": "factorial_power",
            "parameters": {"effect_size": 0.9, "n_factors": 3, "n_per_cell": 8},
            "expected": 0.943372643162444,
            "source": "Wu & Hamada (2009) Table 4.3",
        },
    ]

    cases = montgomery_cases + cohen_cases + factorial_cases
    comparisons: list[ValidationComparison] = []
    for case in cases:
        method = case["method"]
        params = case["parameters"]
        expected = case["expected"]
        if method == "t_test_power":
            result = t_test_power(**params)
        elif method == "anova_power":
            result = anova_power(**params)
        else:
            result = factorial_power(**params)
        assert result == pytest.approx(expected, abs=TOLERANCE)
        comparisons.append(
            ValidationComparison(
                method=method,
                parameters=params,
                reference_power=expected,
                source=case["source"],
            )
        )

    report = generate_validation_report(comparisons, tolerance=TOLERANCE)
    assert np.all(report["difference"] <= TOLERANCE)
    assert report["notes"].str.len().eq(0).all()


def test_against_r_packages() -> None:
    """Numerically compare against canonical R package results."""

    r_reference = [
        {
            "method": "t_test_power",
            "parameters": {"effect_size": 0.5, "n_per_group": 64, "alpha": 0.05},
            "reference_power": 0.8014595579287104,
            "source": "R pwr.t.test two-sample",
        },
        {
            "method": "anova_power",
            "parameters": {"effect_size": 0.25, "n_per_group": 45, "n_groups": 4},
            "reference_power": 0.8039869128651755,
            "source": "R pwr.anova.test",
        },
        {
            "method": "factorial_power",
            "parameters": {"effect_size": 0.9, "n_factors": 3, "n_per_cell": 8},
            "reference_power": 0.943372643162444,
            "source": "R FrF2::power.2level",
        },
    ]

    report = generate_validation_report(r_reference, tolerance=TOLERANCE)
    assert np.allclose(
        report["calculated_power"], report["reference_power"], atol=TOLERANCE
    )
    assert report["difference"].max() <= TOLERANCE


def _simulate_t_test(
    *,
    effect_size: float,
    n_per_group: int,
    alpha: float,
    simulations: int,
    rng: np.random.Generator,
) -> float:
    group1 = rng.normal(loc=0.0, scale=1.0, size=(simulations, n_per_group))
    group2 = rng.normal(loc=effect_size, scale=1.0, size=(simulations, n_per_group))
    mean1 = group1.mean(axis=1)
    mean2 = group2.mean(axis=1)
    var1 = group1.var(axis=1, ddof=1)
    var2 = group2.var(axis=1, ddof=1)
    pooled = ((n_per_group - 1) * var1 + (n_per_group - 1) * var2) / (
        2 * n_per_group - 2
    )
    se = np.sqrt(pooled * 2 / n_per_group)
    t_stat = (mean1 - mean2) / se
    df = 2 * n_per_group - 2
    crit = stats.t.ppf(1 - alpha / 2, df)
    rejects = np.abs(t_stat) >= crit
    return float(np.mean(rejects))


def _simulate_anova(
    *,
    effect_size: float,
    n_per_group: int,
    n_groups: int,
    alpha: float,
    simulations: int,
    rng: np.random.Generator,
) -> float:
    means = _anova_group_means(effect_size, n_groups)
    data = rng.normal(
        loc=means[:, None, None], scale=1.0, size=(n_groups, simulations, n_per_group)
    )
    group_means = data.mean(axis=2)
    overall_mean = group_means.mean(axis=0)
    ss_between = n_per_group * np.sum((group_means - overall_mean) ** 2, axis=0)
    residuals = data - group_means[:, :, None]
    ss_within = np.sum(residuals**2, axis=(0, 2))
    df_between = n_groups - 1
    df_within = n_groups * (n_per_group - 1)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within
    crit = stats.f.ppf(1 - alpha, df_between, df_within)
    return float(np.mean(f_stat >= crit))


def _simulate_factorial(
    *,
    effect_size: float,
    n_factors: int,
    n_per_cell: int,
    alpha: float,
    simulations: int,
    rng: np.random.Generator,
) -> float:
    design = np.array(list(product([-1, 1], repeat=n_factors)))
    group_indicator = design[:, 0]
    cell_means = (effect_size / 2) * group_indicator
    data = rng.normal(
        loc=cell_means[None, :, None],
        scale=1.0,
        size=(simulations, len(cell_means), n_per_cell),
    )
    cell_avg = data.mean(axis=2)
    high = cell_avg[:, group_indicator == 1].mean(axis=1)
    low = cell_avg[:, group_indicator == -1].mean(axis=1)
    effect_est = high - low
    residuals = data - cell_avg[:, :, None]
    mse = np.sum(residuals**2, axis=(1, 2)) / (len(cell_means) * (n_per_cell - 1))
    n_per_group = n_per_cell * (2 ** (n_factors - 1))
    se = np.sqrt(2 * mse / n_per_group)
    t_stat = effect_est / se
    df = len(cell_means) * (n_per_cell - 1)
    crit = stats.t.ppf(1 - alpha / 2, df)
    return float(np.mean(np.abs(t_stat) >= crit))


def test_monte_carlo_validation() -> None:
    """Monte Carlo experiments validate analytical power estimates."""

    rng = np.random.default_rng(87234)
    t_configs = [
        {"effect_size": 0.5, "n_per_group": 30, "alpha": 0.05},
        {"effect_size": 0.8, "n_per_group": 25, "alpha": 0.05},
    ]
    for cfg in t_configs:
        empirical = _simulate_t_test(
            **cfg, simulations=SIMULATIONS_PER_CONFIGURATION, rng=rng
        )
        theoretical = t_test_power(**cfg)
        assert empirical == pytest.approx(theoretical, abs=TOLERANCE)

    anova_configs = [
        {"effect_size": 0.25, "n_per_group": 45, "n_groups": 4, "alpha": 0.05},
        {"effect_size": 0.3, "n_per_group": 20, "n_groups": 5, "alpha": 0.05},
    ]
    for cfg in anova_configs:
        empirical = _simulate_anova(
            **cfg, simulations=SIMULATIONS_PER_CONFIGURATION, rng=rng
        )
        theoretical = anova_power(**cfg)
        assert empirical == pytest.approx(theoretical, abs=TOLERANCE)

    factorial_configs = [
        {"effect_size": 0.7, "n_factors": 3, "n_per_cell": 8, "alpha": 0.05},
        {"effect_size": 0.9, "n_factors": 3, "n_per_cell": 8, "alpha": 0.05},
    ]
    for cfg in factorial_configs:
        empirical = _simulate_factorial(
            **cfg, simulations=SIMULATIONS_PER_CONFIGURATION, rng=rng
        )
        theoretical = factorial_power(**cfg)
        assert empirical == pytest.approx(theoretical, abs=TOLERANCE)


@given(
    base_effect=st.floats(min_value=0.2, max_value=1.0),
    delta_effect=st.floats(min_value=0.05, max_value=0.5),
    base_n=st.integers(min_value=5, max_value=60),
    delta_n=st.integers(min_value=1, max_value=30),
    alpha_low=st.floats(min_value=0.01, max_value=0.1),
    alpha_high=st.floats(min_value=0.12, max_value=0.2),
)
def test_power_curve_consistency(
    *,
    base_effect: float,
    delta_effect: float,
    base_n: int,
    delta_n: int,
    alpha_low: float,
    alpha_high: float,
) -> None:
    """Power should improve with stronger effects, more data, and stricter alpha."""

    smaller = t_test_power(
        effect_size=base_effect,
        n_per_group=base_n,
        alpha=alpha_low,
    )
    more_samples = t_test_power(
        effect_size=base_effect,
        n_per_group=base_n + delta_n,
        alpha=alpha_low,
    )
    stronger_effect = t_test_power(
        effect_size=base_effect + delta_effect,
        n_per_group=base_n,
        alpha=alpha_low,
    )
    relaxed_alpha = t_test_power(
        effect_size=base_effect,
        n_per_group=base_n,
        alpha=alpha_high,
    )

    assert more_samples >= smaller - 1e-6
    assert stronger_effect >= smaller - 1e-6
    assert relaxed_alpha >= smaller - 1e-6

    tighter_alpha = t_test_power(
        effect_size=base_effect,
        n_per_group=base_n,
        alpha=alpha_low / 2,
    )
    assert tighter_alpha <= smaller + 1e-6

    # Deterministic cross-check over a structured grid for ANOVA and factorial cases.
    sample_sizes = np.arange(8, 32, 4)
    effect_sizes = np.linspace(0.2, 0.8, 4)
    alphas = np.array([0.01, 0.05, 0.1])

    for n in sample_sizes:
        curve = [
            anova_power(effect_size=e, n_per_group=n, n_groups=4, alpha=0.05)
            for e in effect_sizes
        ]
        assert np.all(np.diff(curve) >= -1e-6)

    for e in effect_sizes:
        curve = [
            anova_power(effect_size=e, n_per_group=n, n_groups=4, alpha=0.05)
            for n in sample_sizes
        ]
        assert np.all(np.diff(curve) >= -1e-6)

    alpha_curve = [
        factorial_power(effect_size=0.6, n_factors=3, n_per_cell=6, alpha=a)
        for a in alphas
    ]
    assert np.all(np.diff(alpha_curve) >= -1e-6)
