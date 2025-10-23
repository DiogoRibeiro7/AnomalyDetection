"""Statistical power analysis utilities.

This module provides thin wrappers around the canonical power-analysis
implementations from :mod:`statsmodels` and augments them with lightweight
validation helpers used across the industrial statistics toolchain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Mapping, MutableMapping, Sequence

import pandas as pd
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower

__all__ = [
    "ValidationComparison",
    "anova_power",
    "factorial_power",
    "generate_validation_report",
    "t_test_power",
]

_T_TEST_ANALYSIS = TTestIndPower()
_ANOVA_ANALYSIS = FTestAnovaPower()


@dataclass(frozen=True)
class ValidationComparison:
    """Represents a single validation record for a power computation."""

    method: Literal["t_test_power", "anova_power", "factorial_power"]
    parameters: Mapping[str, Any]
    reference_power: float
    source: str
    notes: str = field(default="")


def t_test_power(
    effect_size: float,
    *,
    n_per_group: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "one-sided"] = "two-sided",
) -> float:
    """Return power for a balanced two-sample t-test.

    Parameters
    ----------
    effect_size:
        Cohen's :math:`d` effect size (difference in group means divided by the
        pooled standard deviation).
    n_per_group:
        Sample size per group.
    alpha:
        Significance level.
    alternative:
        Specifies whether to evaluate a one- or two-sided alternative.

    Returns
    -------
    float
        Statistical power for the specified design.
    """

    if n_per_group <= 1:
        msg = "n_per_group must be greater than 1 for a t-test power calculation"
        raise ValueError(msg)

    power = _T_TEST_ANALYSIS.power(
        effect_size=effect_size,
        nobs1=n_per_group,
        alpha=alpha,
        ratio=1.0,
        alternative=alternative,
    )
    if math.isnan(power):
        df = 2 * n_per_group - 2
        noncentrality = effect_size * math.sqrt(n_per_group / 2)
        if alternative == "two-sided":
            critical = stats.t.ppf(1 - alpha / 2, df)
            power = stats.nct.sf(critical, df, noncentrality) + stats.nct.cdf(
                -critical, df, noncentrality
            )
        else:
            critical = stats.t.ppf(1 - alpha, df)
            power = stats.nct.sf(critical, df, noncentrality)

    if math.isnan(power):
        scale = math.sqrt(n_per_group / 2)
        z_effect = effect_size * scale
        if alternative == "two-sided":
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            power = stats.norm.sf(z_alpha - z_effect) + stats.norm.cdf(
                -z_alpha - z_effect
            )
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
            power = stats.norm.sf(z_alpha - z_effect)

    power = float(power)
    return min(max(power, 0.0), 1.0)


def anova_power(
    effect_size: float,
    *,
    n_per_group: int,
    n_groups: int,
    alpha: float = 0.05,
) -> float:
    """Return power for a balanced one-way ANOVA design.

    Parameters
    ----------
    effect_size:
        Cohen's :math:`f` effect size for the ANOVA.
    n_per_group:
        Sample size per treatment group.
    n_groups:
        Number of treatment groups.
    alpha:
        Significance level.

    Returns
    -------
    float
        Statistical power for the specified design.
    """

    if n_per_group <= 1:
        msg = "n_per_group must be greater than 1 for an ANOVA power calculation"
        raise ValueError(msg)
    if n_groups < 2:
        msg = "n_groups must be at least 2 for an ANOVA design"
        raise ValueError(msg)

    total_n = n_per_group * n_groups
    power = _ANOVA_ANALYSIS.power(
        effect_size=effect_size,
        k_groups=n_groups,
        nobs=total_n,
        alpha=alpha,
    )
    return float(power)


def factorial_power(
    effect_size: float,
    *,
    n_factors: int,
    n_per_cell: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "one-sided"] = "two-sided",
) -> float:
    """Return power for a balanced :math:`2^k` factorial main-effect test.

    The implementation collapses the factorial design into two pseudo groups
    representing the high and low levels of the factor of interest and reuses
    the two-sample t-test power calculation. This matches the calculations
    available in the ``FrF2`` R package for main-effect screening designs when
    expressed in terms of Cohen's :math:`d`.

    Parameters
    ----------
    effect_size:
        Cohen's :math:`d` effect size for the main effect under investigation.
    n_factors:
        Number of two-level factors in the factorial design.
    n_per_cell:
        Replicates per factorial cell.
    alpha:
        Significance level.
    alternative:
        Whether to evaluate a one- or two-sided alternative.

    Returns
    -------
    float
        Statistical power for detecting the specified effect.
    """

    if n_factors < 1:
        msg = "n_factors must be at least 1 for a factorial design"
        raise ValueError(msg)
    if n_per_cell <= 0:
        msg = "n_per_cell must be positive for a factorial design"
        raise ValueError(msg)

    n_per_group = n_per_cell * (2 ** (n_factors - 1))
    return t_test_power(
        effect_size,
        n_per_group=n_per_group,
        alpha=alpha,
        alternative=alternative,
    )


_METHOD_REGISTRY: Dict[str, Callable[..., float]] = {
    "t_test_power": t_test_power,
    "anova_power": anova_power,
    "factorial_power": factorial_power,
}


def generate_validation_report(
    comparisons: Sequence[ValidationComparison | Mapping[str, Any]],
    *,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Generate a tabular validation report for power calculations.

    Parameters
    ----------
    comparisons:
        Iterable of validation specifications. Each element can either be a
        :class:`ValidationComparison` or a mapping with the keys
        ``method``, ``parameters``, ``reference_power``, and ``source``. Optional
        ``notes`` entries are copied verbatim.
    tolerance:
        Absolute tolerance used to determine whether the calculated power agrees
        with the reference implementation.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing both the calculated and reference power values
        alongside difference magnitudes and explanatory notes.
    """

    records: list[MutableMapping[str, Any]] = []
    for entry in comparisons:
        if isinstance(entry, ValidationComparison):
            payload = {
                "method": entry.method,
                "parameters": dict(entry.parameters),
                "reference_power": float(entry.reference_power),
                "source": entry.source,
                "notes": entry.notes,
            }
        else:
            payload = dict(entry)
            if "notes" not in payload:
                payload["notes"] = ""
        method = payload["method"]
        func = _METHOD_REGISTRY.get(method)
        if func is None:
            msg = f"Unknown validation method '{method}'"
            raise KeyError(msg)

        calculated = float(func(**payload["parameters"]))
        difference = abs(calculated - float(payload["reference_power"]))
        notes = payload["notes"]
        if difference > tolerance:
            notes = (notes + "\n" if notes else "") + (
                f"Difference {difference:.4f} exceeds tolerance {tolerance:.4f}."
            )
        records.append(
            {
                "method": method,
                "parameters": payload["parameters"],
                "calculated_power": calculated,
                "reference_power": float(payload["reference_power"]),
                "difference": difference,
                "source": payload["source"],
                "notes": notes,
            }
        )

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        order = [
            "method",
            "parameters",
            "calculated_power",
            "reference_power",
            "difference",
            "source",
            "notes",
        ]
        df = df[order]
    return df
