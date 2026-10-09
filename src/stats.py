"""Shared inference helpers — Newey-West HAC stats and significance stars.

Used by sections §3, §5–§11. Centralised here so the NW lag rule, mean
estimator, and significance-star convention can never drift between sections.
"""

import math

import numpy as np
import statsmodels.api as sm


def nw_lag(T: int) -> int:
    """Andrews (1991) primary lag: L = ceil(0.75 * T^(1/3))."""
    return math.ceil(0.75 * T ** (1 / 3))


def nw_stats(series) -> dict:
    """
    HAC Newey-West stats for H0: mean = 0 plus standard t-stat, annualised Sharpe.

    Returns a dict with: mean, t (NW HAC), p (NW HAC), se (NW HAC),
    nw_lag, T, std_t (plain OLS), sharpe (annualised, sqrt(12) * mean / std).
    Empty/short series (T < 5) return NaNs.
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    T   = len(arr)
    if T < 5:
        return {"mean": float("nan"), "t": float("nan"), "p": float("nan"),
                "se": float("nan"), "nw_lag": 0, "T": T,
                "std_t": float("nan"), "sharpe": float("nan")}

    L   = nw_lag(T)
    res = sm.OLS(arr, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1))
    return {
        "mean":   float(res.params[0]),
        "t":      float(res.tvalues[0]),
        "p":      float(res.pvalues[0]),
        "se":     float(res.bse[0]),
        "nw_lag": L,
        "T":      T,
        "std_t":  mean / (std / np.sqrt(T)) if std > 0 else float("nan"),
        "sharpe": float(np.sqrt(12) * mean / std) if std > 0 else float("nan"),
    }


def sig_stars(p: float) -> str:
    """*** / ** / * for p < .01 / .05 / .10. Empty string for NaN or larger p."""
    if math.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""
