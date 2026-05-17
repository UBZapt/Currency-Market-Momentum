#!/usr/bin/env python3
"""
src/signals.py — §4 signal construction machinery.

Three signal types (§5.1, §5.4):
  A  — equal-weighted excess return  (signal_excess_return)
  B  — equal-weighted spot change    (signal_spot_change)
  OLS— rolling OLS-weighted          (signal_rolling_ols)

All signals return (date, currency_code, signal) with date = formation month.

Signal window convention (§5.1, no skip-month — confirmed by §3 reversal test):
  S_{i,t} = col_t + col_{t-1} + ... + col_{t-f+1}   (f terms)
  = current formation-date value + (f-1) prior lags.

Signal B additionally requires excess_return to be non-null at the formation date
so that every currency in the signal can actually generate a portfolio return.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
RETURNS_PATH = OUTPUT_DIR / "returns_panel.csv"


def _get_log() -> logging.Logger:
    log = logging.getLogger("signals")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(levelname)-7s  %(message)s"))
        log.addHandler(h)
        log.propagate = False
    log.setLevel(logging.WARNING)
    return log


def _build_lag_panel(panel: pd.DataFrame, col: str, max_lag: int) -> pd.DataFrame:
    """
    Period-merge lag_1 … lag_{max_lag} for column `col`.
    Non-consecutive months produce NaN (never paired across gaps).
    Returns DataFrame with [date, currency_code, col, ym, lag_1, ..., lag_{max_lag}].
    """
    base = panel[["date", "currency_code", col]].copy()
    base["ym"] = base["date"].dt.to_period("M")
    result = base.copy()
    for k in range(1, max_lag + 1):
        src = base[["currency_code", "ym", col]].rename(columns={col: f"lag_{k}"}).copy()
        src["ym"] = src["ym"] + k
        result = result.merge(src, on=["currency_code", "ym"], how="left")
    return result


def _rolling_sum_signal(returns_panel: pd.DataFrame, col: str, f: int) -> pd.DataFrame:
    """
    Build f-month rolling sum signal for column `col`.
    Signal = col[t] + lag_1..lag_{f-1}[t]  (f terms, §5.1).
    NaN where any of the f terms is missing.
    """
    lp = _build_lag_panel(returns_panel, col, max(f - 1, 0))
    if f == 1:
        lp["signal"] = lp[col]
    else:
        lp["signal"] = lp[[col] + [f"lag_{k}" for k in range(1, f)]].sum(axis=1, min_count=f)
    return (
        lp[["date", "currency_code", "signal"]]
        .dropna(subset=["signal"])
        .reset_index(drop=True)
    )


def signal_excess_return(returns_panel: pd.DataFrame, f: int) -> pd.DataFrame:
    """Signal A (§5.1): S_{i,t} = rx_t + rx_{t-1} + ... + rx_{t-f+1}."""
    _validate_panel(returns_panel, ["excess_return"])
    return _rolling_sum_signal(returns_panel, "excess_return", f)


def signal_spot_change(returns_panel: pd.DataFrame, f: int) -> pd.DataFrame:
    """
    Signal B (§5.1): S_{i,t}^B = -Σ Δs_{i,t-j+1}.
    spot_change is already sign-flipped so the sum equals the negated cumulative spot change.
    Restricted to currencies with non-null excess_return at formation (required for portfolio returns).
    """
    _validate_panel(returns_panel, ["spot_change", "excess_return"])
    out = _rolling_sum_signal(returns_panel, "spot_change", f)
    rx_avail = (
        returns_panel[["date", "currency_code", "excess_return"]]
        .dropna(subset=["excess_return"])[["date", "currency_code"]]
    )
    return out.merge(rx_avail, on=["date", "currency_code"], how="inner").reset_index(drop=True)


def signal_rolling_ols(
    returns_panel: pd.DataFrame,
    f: int,
    initial_window: int = 36,
) -> pd.DataFrame:
    """
    Rolling OLS signal (§5.4).

    At each formation month t, estimates pooled OLS on all (i,τ) with τ < t:
      rx_{i,τ+1} = α + Σ_{j=1}^{f} β_j · rx_{i,τ-j+1} + ε
    then applies β̂ to the f lagged values at t:
      S_{i,t}^OLS = β̂ · [rx_t, rx_{t-1}, ..., rx_{t-f+1}]
    Fully out-of-sample: weights at t use only data through t-1.
    """
    _validate_panel(returns_panel, ["excess_return"])
    log = _get_log()

    lp         = _build_lag_panel(returns_panel, "excess_return", f)
    lag_cols   = [f"lag_{k}" for k in range(1, f + 1)]
    apply_cols = ["excess_return"] + [f"lag_{k}" for k in range(1, f)]

    pool = lp.dropna(subset=["excess_return"] + lag_cols).copy()

    first_rx_date = returns_panel.dropna(subset=["excess_return"])["date"].min()
    first_form_ym = first_rx_date.to_period("M") + initial_window
    form_months   = sorted(ym for ym in lp["ym"].unique() if ym >= first_form_ym)

    if not form_months:
        raise ValueError(
            f"No formation months after initial_window={initial_window} from {first_rx_date.date()}"
        )

    records = []
    for t_ym in form_months:
        train = pool[pool["ym"] < t_ym]
        if len(train) < f + 2:
            continue

        Y = train["excess_return"].values
        X = sm.add_constant(train[lag_cols].values, has_constant="add")

        if np.linalg.matrix_rank(X) < X.shape[1]:
            log.warning("OLS rank-deficient at %s — skipping", t_ym)
            continue

        coefs = sm.OLS(Y, X).fit().params[1:]  # drop intercept

        cur = lp[lp["ym"] == t_ym].dropna(subset=apply_cols).reset_index(drop=True)
        if cur.empty:
            continue
        batch = cur[["date", "currency_code"]].copy()
        batch["signal"] = cur[apply_cols].values @ coefs
        records.append(batch)

    if not records:
        raise ValueError("signal_rolling_ols produced no signals — check data coverage")

    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["date", "currency_code"])
        .reset_index(drop=True)
    )[["date", "currency_code", "signal"]]


def build_signal(
    returns_panel: pd.DataFrame,
    f: int,
    signal_type: str = "A",
    initial_window: int = 36,
) -> pd.DataFrame:
    """
    Dispatcher. signal_type: 'A' | 'B' | 'OLS'.
    Returns (date, currency_code, signal).
    """
    if signal_type == "A":
        return signal_excess_return(returns_panel, f)
    if signal_type == "B":
        return signal_spot_change(returns_panel, f)
    if signal_type == "OLS":
        return signal_rolling_ols(returns_panel, f, initial_window)
    raise ValueError(f"Unknown signal_type '{signal_type}'. Use 'A', 'B', or 'OLS'.")


def _validate_panel(panel: pd.DataFrame, required_cols: list) -> None:
    needed  = {"date", "currency_code"} | set(required_cols)
    missing = needed - set(panel.columns)
    if missing:
        raise ValueError(f"returns_panel missing required columns: {missing}")
    if panel.duplicated(subset=["date", "currency_code"]).any():
        raise ValueError("returns_panel has duplicate (date, currency_code) keys")


if __name__ == "__main__":
    if not RETURNS_PATH.exists():
        raise FileNotFoundError(f"returns_panel not found: {RETURNS_PATH}")
    rp = pd.read_csv(RETURNS_PATH)
    rp["date"] = pd.to_datetime(rp["date"], dayfirst=True)
    for f in [1, 6]:
        sa = signal_excess_return(rp, f)
        sb = signal_spot_change(rp, f)
        print(f"f={f}  Signal A: {len(sa)} rows  Signal B: {len(sb)} rows")
    sols = signal_rolling_ols(rp, f=6)
    print(f"OLS f=6: {len(sols)} rows, first date={sols['date'].min().date()}")
