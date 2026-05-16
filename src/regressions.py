#!/usr/bin/env python3
"""
src/regressions.py — §3 short-term reversal test.
"""

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
RETURNS_PATH = OUTPUT_DIR / "returns_panel.csv"

_F_GRID  = [1, 3, 6, 9, 12]
_MIN_OBS = 5   # minimum currencies per cross-sectional regression


def _get_log(verbose: bool = False) -> logging.Logger:
    log = logging.getLogger("regressions")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(levelname)-7s  %(message)s"))
        log.addHandler(h)
        log.propagate = False
    log.setLevel(logging.INFO if verbose else logging.WARNING)
    return log


def _build_lagged_panel(panel: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    """
    Period-merge lag_1 … lag_{max_lag} onto the panel.
    Ensures non-consecutive months are never paired (same approach as returns.py).
    """
    base = panel[["date", "currency_code", "excess_return"]].copy()
    base["ym"] = base["date"].dt.to_period("M")
    result = base.copy()
    for k in range(1, max_lag + 1):
        src = base[["currency_code", "ym", "excess_return"]].rename(
            columns={"excess_return": f"lag_{k}"}
        ).copy()
        src["ym"] = src["ym"] + k
        result = result.merge(src, on=["currency_code", "ym"], how="left")
    return result.drop(columns=["ym"])


def _ols_mean_tstat(series: np.ndarray) -> tuple:
    """Plain OLS t-test for H0: mean = 0 (no HAC). Returns (mean, se, t, p)."""
    T = len(series)
    res = sm.OLS(series, np.ones(T)).fit()
    return float(res.params[0]), float(res.bse[0]), float(res.tvalues[0]), float(res.pvalues[0])


def _nw_mean_tstat(series: np.ndarray, maxlags: int) -> tuple:
    """HAC t-test for H0: mean = 0. Returns (mean, se, t, p)."""
    T = len(series)
    res = sm.OLS(series, np.ones(T)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return float(res.params[0]), float(res.bse[0]), float(res.tvalues[0]), float(res.pvalues[0])


def _run_reversal_one_f(lagged: pd.DataFrame, f: int, log: logging.Logger) -> dict:
    """
    Cross-sectional FMB for one f. At realization date d (= t+1 in §3.1):

        Y  = excess_return_d           (rx^i_{t+1})
        X1 = lag_1                     (rx^i_t — most recent month)
        X2 = sum(lag_2 … lag_f)        (rx^i_{t-f:t-1}, cumulative control; omitted for f=1)

    Returns a row-dict for Table 1.
    """
    lag_cols = [f"lag_{k}" for k in range(1, f + 1)]
    df = lagged[["date", "excess_return"] + lag_cols].copy()

    if f > 1:
        # all f-1 lags must be non-NaN (min_count = f-1 enforces this)
        df["cum_ctrl"] = df[
            [f"lag_{k}" for k in range(2, f + 1)]
        ].sum(axis=1, min_count=f - 1)
        required = ["excess_return", "lag_1", "cum_ctrl"]
    else:
        required = ["excess_return", "lag_1"]

    n_regressors = 3 if f > 1 else 2   # including constant
    min_obs = max(_MIN_OBS, n_regressors + 1)

    beta1_ts = []

    for date, grp in df.groupby("date"):
        valid = grp.dropna(subset=required)
        if len(valid) < min_obs:
            continue

        Y = valid["excess_return"].values
        X_arr = (
            np.column_stack([valid["lag_1"].values, valid["cum_ctrl"].values])
            if f > 1
            else valid["lag_1"].values.reshape(-1, 1)
        )
        X = sm.add_constant(X_arr)

        if np.linalg.matrix_rank(X) < X.shape[1]:
            raise ValueError(f"Perfect multicollinearity: date={date}, f={f}")

        res = sm.OLS(Y, X).fit()
        beta1_ts.append(res.params[1])

    T_eff = len(beta1_ts)
    if T_eff < 20:
        raise ValueError(
            f"f={f}: only {T_eff} valid cross-sections — insufficient for NW inference"
        )

    beta1      = np.array(beta1_ts)
    beta1_mean = float(beta1.mean())
    L_andrews  = math.ceil(0.75 * T_eff ** (1 / 3))

    _, se_ols, t_ols, p_ols = _ols_mean_tstat(beta1)
    _, se_a,   t_a,   p_a   = _nw_mean_tstat(beta1, L_andrews)
    _, se_12,  t_12,  p_12  = _nw_mean_tstat(beta1, 12)

    log.info(
        "f=%2d  T=%d  L=%d  b1=%+.4f  t(OLS)=%+.2f  t(NW-A)=%+.2f  t(NW-12)=%+.2f",
        f, T_eff, L_andrews, beta1_mean, t_ols, t_a, t_12,
    )
    return {
        "f":              f,
        "n_months":       T_eff,
        "nw_lag_andrews": L_andrews,
        "beta1_mean":     beta1_mean,
        "se_ols":         float(se_ols),
        "t_ols":          float(t_ols),
        "p_ols":          float(p_ols),
        "se_nw_andrews":  float(se_a),
        "t_nw_andrews":   float(t_a),
        "p_nw_andrews":   float(p_a),
        "se_nw_12":       float(se_12),
        "t_nw_12":        float(t_12),
        "p_nw_12":        float(p_12),
    }


def _export_outputs(table1: pd.DataFrame, decision: dict, log: logging.Logger) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table1.to_csv(OUTPUT_DIR / "table1_reversal.csv", index=False)
    with open(OUTPUT_DIR / "skip_month_decision.json", "w") as fh:
        json.dump(decision, fh, indent=2)
    log.info(
        "Written: table1_reversal.csv (%d rows), skip_month_decision.json", len(table1)
    )


def short_term_reversal(
    returns_panel: pd.DataFrame | None = None,
    verbose: bool = False,
) -> dict:
    """
    §3 short-term reversal test.

    skip_month_flag is derived from f=1 only (univariate, Andrews NW lag, two-tailed
    5% threshold) — see Methodology_Detailed.md §3.3.

    Returns {"table1": pd.DataFrame, "skip_month_flag": bool, "decision": dict}.
    """
    log = _get_log(verbose)

    if returns_panel is None:
        if not RETURNS_PATH.exists():
            raise FileNotFoundError(
                f"returns_panel not found: {RETURNS_PATH} — run returns.py first"
            )
        returns_panel = pd.read_csv(RETURNS_PATH)
        returns_panel["date"] = pd.to_datetime(returns_panel["date"])
    else:
        returns_panel = returns_panel.copy()
        if not pd.api.types.is_datetime64_any_dtype(returns_panel["date"]):
            returns_panel["date"] = pd.to_datetime(returns_panel["date"])

    missing = {"date", "currency_code", "excess_return"} - set(returns_panel.columns)
    if missing:
        raise ValueError(f"returns_panel missing required columns: {missing}")

    if returns_panel.duplicated(subset=["date", "currency_code"]).any():
        raise ValueError("returns_panel has duplicate (date, currency_code) keys")

    inf_mask = np.isinf(returns_panel["excess_return"].dropna())
    if inf_mask.any():
        raise ValueError(f"{int(inf_mask.sum())} non-finite (Inf) excess_return values")

    lagged = _build_lagged_panel(returns_panel, max_lag=max(_F_GRID))
    rows = [_run_reversal_one_f(lagged, f, log) for f in _F_GRID]
    table1 = pd.DataFrame(rows)

    f1 = table1.loc[table1["f"] == 1].iloc[0]
    skip = bool(f1["beta1_mean"] < 0 and f1["p_nw_andrews"] < 0.05)

    decision = {
        "skip_month_flag":    skip,
        "basis_f":            1,
        "nw_lag_andrews":     int(f1["nw_lag_andrews"]),
        "beta1_mean":         float(f1["beta1_mean"]),
        "t_stat_nw_andrews":  float(f1["t_nw_andrews"]),
        "p_value_nw_andrews": float(f1["p_nw_andrews"]),
    }

    log.info("skip_month_flag=%s", skip)
    _export_outputs(table1, decision, log)

    return {"table1": table1, "skip_month_flag": skip, "decision": decision}


def print_reversal_table(result: dict) -> None:
    """Print the MATLAB-style Table 1 display for the short-term reversal result."""
    t1 = result["table1"]
    d  = result["decision"]
    W  = 96

    print()
    print("=" * W)
    print("Table 1 -- Short-Term Reversal Test (Fama-MacBeth cross-sectional OLS, §3)")
    print(f"{'Dep. var: rx(t+1) | Regressor 1: rx(t) | Regressor 2: sum(rx(t-2)...rx(t-f)) for f>1':^{W}}")
    print("=" * W)
    print(
        f"{'f':>4} {'T':>5} {'L':>3}  "
        f"{'beta1':>9} {'SE(OLS)':>9} {'t(OLS)':>8}  "
        f"{'SE(NW-A)':>9} {'t(NW-A)':>8}  "
        f"{'SE(NW-12)':>9} {'t(NW-12)':>8}"
    )
    print("-" * W)
    for _, row in t1.iterrows():
        print(
            f"{int(row['f']):>4} {int(row['n_months']):>5} {int(row['nw_lag_andrews']):>3}  "
            f"{row['beta1_mean']:>+9.4f} {row['se_ols']:>9.4f} {row['t_ols']:>+8.2f}  "
            f"{row['se_nw_andrews']:>9.4f} {row['t_nw_andrews']:>+8.2f}  "
            f"{row['se_nw_12']:>9.4f} {row['t_nw_12']:>+8.2f}"
        )
    print("-" * W)
    print(f"{'f':>4}  {'p(OLS)':>10}  {'p(NW-A)':>10}  {'p(NW-12)':>10}")
    print("-" * W)
    for _, row in t1.iterrows():
        print(
            f"{int(row['f']):>4}  {row['p_ols']:>10.3e}  "
            f"{row['p_nw_andrews']:>10.3e}  {row['p_nw_12']:>10.3e}"
        )
    print("=" * W)
    print(
        f"Decision (basis: f={d['basis_f']}, NW-Andrews L={d['nw_lag_andrews']}, two-tailed 5%): "
        f"beta1_mean={d['beta1_mean']:+.4f}, t={d['t_stat_nw_andrews']:+.4f}, "
        f"p={d['p_value_nw_andrews']:.3e}  =>  skip_month_flag={result['skip_month_flag']}"
    )
    print("=" * W)


if __name__ == "__main__":
    res = short_term_reversal(verbose=True)
    print_reversal_table(res)
