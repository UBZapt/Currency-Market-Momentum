#!/usr/bin/env python3
"""
src/portfolios.py — §4 portfolio-construction machinery.

Functions (in call order):
  assign_portfolios       — sextile/quintile sort per formation month
  compute_cohort_returns  — frozen-composition holding returns over h months
  compute_mom_series      — overlapping long-short MOM time series
  compute_portfolio_series— per-bucket return time series
  build_mom_pipeline      — full stack: signal → sort → cohort → MOM
  build_portfolio_pipeline— full stack: signal → sort → cohort → per-bucket

No inference, no result tables. All outputs are DataFrames ready for Section 5.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
RETURNS_PATH = OUTPUT_DIR / "returns_panel.csv"

from src.signals import build_signal


def _get_log() -> logging.Logger:
    log = logging.getLogger("portfolios")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(levelname)-7s  %(message)s"))
        log.addHandler(h)
        log.propagate = False
    log.setLevel(logging.WARNING)
    return log


def assign_portfolios(signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    §4.1 sextile/quintile sort.

    Input: (date, currency_code, signal) — non-null signals only.
    Per formation month t:
      N >= 18  → 6 portfolios (sextiles)
      15–17    → 5 portfolios (quintiles)
      < 15     → month dropped

    Tie-break: equal signals ranked alphabetically by currency_code.
    Remainder: extra currencies go to the highest-indexed portfolios
    (floor/ceiling allocation).

    Output: (date, currency_code, portfolio, n_portfolios).
    """
    required = {"date", "currency_code", "signal"}
    missing  = required - set(signal_df.columns)
    if missing:
        raise ValueError(f"signal_df missing columns: {missing}")

    clean = signal_df.dropna(subset=["signal"]).copy()

    # Determine N and k per date, drop dates below threshold
    N_series = clean.groupby("date")["currency_code"].transform("count")
    clean = clean[N_series >= 15].copy()
    if clean.empty:
        raise ValueError("assign_portfolios: no months passed the N>=15 threshold")

    N_series = clean.groupby("date")["currency_code"].transform("count")
    clean["_N"] = N_series.astype(int)
    clean["_k"] = np.where(clean["_N"] >= 18, 6, 5)

    # 0-based rank within each date (tie-break: signal asc, then currency_code asc)
    clean = clean.sort_values(["date", "signal", "currency_code"]).reset_index(drop=True)
    clean["_rank"] = clean.groupby("date").cumcount()

    # Floor/ceiling assignment: last (N % k) portfolios each get one extra currency
    N  = clean["_N"]
    k  = clean["_k"]
    base   = N // k
    extras = N % k
    thresh = (k - extras) * base  # 0-based rank boundary between small and large buckets

    small = clean["_rank"] < thresh
    clean["portfolio"] = np.where(
        small,
        (clean["_rank"] // base + 1).values,
        ((k - extras) + (clean["_rank"] - thresh) // (base + 1) + 1).values,
    ).astype(int)
    clean["n_portfolios"] = k.astype(int)

    return (
        clean[["date", "currency_code", "portfolio", "n_portfolios"]]
        .sort_values(["date", "portfolio"])
        .reset_index(drop=True)
    )


def compute_cohort_returns(
    assignments: pd.DataFrame,
    returns_panel: pd.DataFrame,
    h: int,
) -> pd.DataFrame:
    """
    §4.2 / §5.2 frozen-composition holding returns.

    For each (formation_date τ, portfolio p): equal-weight excess_returns of the
    assigned currencies at realize_date = τ+1, …, τ+h.
    Period-merge ensures non-consecutive months never pair (gaps produce NaN).
    Currencies missing excess_return at a realize_date are excluded from that
    date's average but the cohort composition remains frozen.

    Output: (formation_date, realize_date, portfolio, n_portfolios, port_return).
    """
    required = {"date", "currency_code", "portfolio", "n_portfolios"}
    missing  = required - set(assignments.columns)
    if missing:
        raise ValueError(f"assignments missing columns: {missing}")

    rp = returns_panel[["date", "currency_code", "excess_return"]].copy()
    rp["ym"] = rp["date"].dt.to_period("M")

    # period_to_date is the same for every holding step — compute once
    period_to_date = (
        rp[["date", "ym"]].drop_duplicates()
        .rename(columns={"ym": "ym_realize", "date": "realize_date"})
    )
    rx_lookup = rp[["currency_code", "ym", "excess_return"]].rename(
        columns={"ym": "ym_realize"}
    )

    asgn = assignments[["date", "currency_code", "portfolio", "n_portfolios"]].copy()
    asgn["ym_form"] = asgn["date"].dt.to_period("M")

    records = []
    for k in range(1, h + 1):
        tmp = asgn.assign(ym_realize=asgn["ym_form"] + k)

        port_rets = (
            tmp.merge(rx_lookup, on=["currency_code", "ym_realize"], how="left")
            .groupby(["date", "ym_realize", "portfolio", "n_portfolios"])["excess_return"]
            .mean()
            .reset_index()
            .merge(period_to_date, on="ym_realize", how="left")
            .rename(columns={"date": "formation_date", "excess_return": "port_return"})
        )
        records.append(
            port_rets[["formation_date", "realize_date", "portfolio", "n_portfolios", "port_return"]]
        )

    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["formation_date", "realize_date", "portfolio"])
        .reset_index(drop=True)
    )


def compute_mom_series(cohort_rets: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    §5.2.3 overlapping MOM time series.

    MOM_t = (1/h) Σ_{j=0}^{h-1} [r_{P_high,t}^{τ=t-j} − r_{P_low,t}^{τ=t-j}]

    High = P(n_portfolios), Low = P1.
    mom_return is NaN for burn-in months (n_active_cohorts < h) to prevent
    downstream code from accidentally consuming partial-overlap months.

    Output: (date, mom_return, n_active_cohorts).
    """
    cr = cohort_rets.copy()
    cr["ym_form"]    = cr["formation_date"].dt.to_period("M")
    cr["ym_realize"] = cr["realize_date"].dt.to_period("M")

    records = []
    for realize_date, grp in cr.groupby("realize_date"):
        ym_t   = realize_date.to_period("M")
        active = grp[(grp["ym_form"] >= ym_t - h) & (grp["ym_form"] <= ym_t - 1)]
        n_active = active["ym_form"].nunique()

        spreads = []
        for _, cohort in active.groupby("ym_form"):
            k_val = int(cohort["n_portfolios"].iloc[0])
            high  = cohort.loc[cohort["portfolio"] == k_val, "port_return"]
            low   = cohort.loc[cohort["portfolio"] == 1,     "port_return"]
            if high.empty or low.empty:
                continue
            r_h, r_l = float(high.iloc[0]), float(low.iloc[0])
            if not (np.isnan(r_h) or np.isnan(r_l)):
                spreads.append(r_h - r_l)

        mom = float(np.mean(spreads)) if (spreads and n_active >= h) else float("nan")
        records.append({"date": realize_date, "mom_return": mom, "n_active_cohorts": n_active})

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def compute_portfolio_series(cohort_rets: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Per-portfolio return series averaged over active cohorts.

    burn-in months (n_active_cohorts < h) are included in output but callers
    should filter on n_active_cohorts == h before inference.
    P6 rows only appear for dates where at least one sextile cohort is active.

    Output: (date, portfolio, avg_return, n_active_cohorts,
             n_sextile_cohorts, n_quintile_cohorts).
    """
    cr = cohort_rets.copy()
    cr["ym_form"] = cr["formation_date"].dt.to_period("M")

    records = []
    for realize_date, grp in cr.groupby("realize_date"):
        ym_t   = realize_date.to_period("M")
        active = grp[(grp["ym_form"] >= ym_t - h) & (grp["ym_form"] <= ym_t - 1)]
        n_active = active["ym_form"].nunique()

        cohort_schemes = active.groupby("ym_form")["n_portfolios"].first()
        n_sextile  = int((cohort_schemes == 6).sum())
        n_quintile = int((cohort_schemes == 5).sum())

        for p, pgrp in active.groupby("portfolio"):
            records.append({
                "date":               realize_date,
                "portfolio":          p,
                "avg_return":         float(pgrp["port_return"].mean()),
                "n_active_cohorts":   n_active,
                "n_sextile_cohorts":  n_sextile,
                "n_quintile_cohorts": n_quintile,
            })

    return (
        pd.DataFrame(records)
        .sort_values(["date", "portfolio"])
        .reset_index(drop=True)
    )


def build_mom_pipeline(
    returns_panel: pd.DataFrame,
    f: int,
    h: int,
    signal_type: str = "A",
) -> pd.DataFrame:
    """
    Full pipeline: signal → assign → cohort returns → MOM series.
    Returns (date, mom_return, n_active_cohorts).
    The OLS rolling window is set inside signal_rolling_ols (methodology §5.4);
    it is not exposed here to keep Signal A/B free of any initial_window default.
    """
    signal_df   = build_signal(returns_panel, f, signal_type)
    assignments = assign_portfolios(signal_df)
    cohort_rets = compute_cohort_returns(assignments, returns_panel, h)
    return compute_mom_series(cohort_rets, h)


def build_portfolio_pipeline(
    returns_panel: pd.DataFrame,
    f: int,
    h: int,
    signal_type: str = "A",
) -> pd.DataFrame:
    """
    Full pipeline: signal → assign → cohort returns → per-portfolio return series.
    Returns (date, portfolio, avg_return, n_active_cohorts,
             n_sextile_cohorts, n_quintile_cohorts).
    """
    signal_df   = build_signal(returns_panel, f, signal_type)
    assignments = assign_portfolios(signal_df)
    cohort_rets = compute_cohort_returns(assignments, returns_panel, h)
    return compute_portfolio_series(cohort_rets, h)


if __name__ == "__main__":
    if not RETURNS_PATH.exists():
        raise FileNotFoundError(f"returns_panel not found: {RETURNS_PATH}")
    rp = pd.read_csv(RETURNS_PATH)
    rp["date"] = pd.to_datetime(rp["date"], dayfirst=True)

    mom = build_mom_pipeline(rp, f=6, h=1, signal_type="A")
    valid = mom.dropna(subset=["mom_return"])
    print(valid.head(5))
    print("First non-NaN MOM date:", valid["date"].min())
