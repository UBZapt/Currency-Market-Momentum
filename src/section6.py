#!/usr/bin/env python3
"""
src/section6.py — §6 Transaction-cost-adjusted momentum returns.

Spread scenarios (alpha = fraction of quoted spread applied):
  1.00  100% quoted spread   — Panel A in Table 9
  0.75  75% effective spread — Panel B left
  0.50  50% effective spread — Panel B right

State logic (§6.x Menkhoff convention):
  stay (k < h): forward leg only incurs bid-ask cost; spot settled at mid (in excess_return)
  exit (k = h): both forward and spot legs incur bid-ask cost

Cost formulas (all in log-price space):
  fwd_half  = (fwd_ask_t  - fwd_bid_t)  / 2
  spot_half = (spot_ask_t1 - spot_bid_t1) / 2
  stay cost = alpha * fwd_half
  exit cost = alpha * (fwd_half + spot_half)

Mathematical identity enabling reuse of compute_mom_series (r_high - r_low):
  P6 (long)  port_return = excess_return - cost      [net long]
  P1 (short) port_return = excess_return + cost      [= -net_short]
  => r_P6 - r_P1 = MOM_gross - alpha*(cost_P6 + cost_P1) = MOM_net  (verified algebraically)
  Assertions #2-#4 hold because excess_return = fwd_mid_t - spot_mid_{t+1} (§2 validate_returns).
"""

import math
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"

from src.signals    import build_signal
from src.portfolios import assign_portfolios, compute_mom_series

F_GRID = [1, 3, 6, 9, 12]
H_GRID = [1, 3, 6, 9, 12]
ALPHAS = [1.00, 0.75, 0.50]


def _get_log() -> logging.Logger:
    log = logging.getLogger("section6")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(levelname)-7s  %(message)s"))
        log.addHandler(h)
        log.propagate = False
    log.setLevel(logging.INFO)
    return log


# ── HAC inference (replicated from section5 to avoid cross-module import) ──────

def _nw_lag(T: int) -> int:
    return math.ceil(0.75 * T ** (1 / 3))


def _nw_stats(series) -> dict:
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    T   = len(arr)
    if T < 5:
        return {"mean": float("nan"), "t": float("nan"), "se": float("nan"), "nw_lag": 0, "T": T}
    L   = _nw_lag(T)
    res = sm.OLS(arr, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    return {
        "mean":   float(res.params[0]),
        "t":      float(res.tvalues[0]),
        "se":     float(res.bse[0]),
        "nw_lag": L,
        "T":      T,
    }


# ── Position preparation — alpha-agnostic ──────────────────────────────────────

def _prepare_positions(
    assignments: pd.DataFrame,
    rp_lookup: pd.DataFrame,
    h: int,
) -> tuple:
    """
    Merge assignments with returns panel per k, drop missing bid/ask.
    Independent of alpha — call once per (f, h).

    Returns:
        frames:       list of (k, is_exit, cleaned_df)
        drop_records: list of dicts {k, is_exit, n_dropped, reason}
    """
    asgn = assignments.copy()
    asgn["ym_form"] = asgn["date"].dt.to_period("M")

    frames       = []
    drop_records = []

    for k in range(1, h + 1):
        is_exit = (k == h)
        tmp = asgn.assign(ym_realize=asgn["ym_form"] + k)
        tmp = tmp.merge(rp_lookup, on=["currency_code", "ym_realize"], how="left")

        # §6.x Assertion #1: one state per currency-position-month.
        # Post-merge check catches rp_lookup duplicates that a pre-merge check cannot.
        dups = tmp.duplicated(["date", "currency_code", "ym_realize"])
        assert not dups.any(), (
            f"Assertion #1 failed at k={k}: {int(dups.sum())} duplicate currency-position-months "
            f"(check rp_lookup uniqueness)"
        )

        # §6.x Assertion #5: drop missing bid/ask, do not replace with mid
        if is_exit:
            ba_mask = tmp[["fwd_bid_t", "fwd_ask_t", "spot_bid_t1", "spot_ask_t1"]].isna().any(axis=1)
            reason  = "exit_missing_fwd_or_spot_ba"
        else:
            ba_mask = tmp[["fwd_bid_t", "fwd_ask_t"]].isna().any(axis=1)
            reason  = "stay_missing_fwd_ba"

        n_dropped = int(ba_mask.sum())
        if n_dropped:
            drop_records.append({"k": k, "is_exit": is_exit, "n_dropped": n_dropped, "reason": reason})

        tmp = tmp[~ba_mask].reset_index(drop=True)
        if not tmp.empty:
            frames.append((k, is_exit, tmp))

    return frames, drop_records


# ── Cost application and MOM computation ───────────────────────────────────────

def _compute_mom_from_frames(
    frames: list,
    period_to_date: pd.DataFrame,
    h: int,
    alpha: float,
) -> pd.Series:
    """Apply bid-ask cost at given alpha to pre-prepared frames; return net MOM series."""
    records = []
    for k, is_exit, tmp in frames:
        fwd_half = (tmp["fwd_ask_t"].values - tmp["fwd_bid_t"].values) / 2.0
        if is_exit:
            spot_half = (tmp["spot_ask_t1"].values - tmp["spot_bid_t1"].values) / 2.0
            cost = alpha * (fwd_half + spot_half)
        else:
            cost = alpha * fwd_half

        k_vals = tmp["n_portfolios"].values
        is_p6  = (tmp["portfolio"].values == k_vals)
        is_p1  = (tmp["portfolio"].values == 1)

        port_return = tmp["excess_return"].values.copy()
        port_return[is_p6] -= cost[is_p6]
        port_return[is_p1] += cost[is_p1]

        port_rets = (
            tmp.assign(port_return=port_return)
            .groupby(["date", "ym_realize", "portfolio", "n_portfolios"])["port_return"]
            .mean()
            .reset_index()
            .merge(period_to_date, on="ym_realize", how="left")
            .rename(columns={"date": "formation_date"})
        )
        records.append(
            port_rets[["formation_date", "realize_date", "portfolio", "n_portfolios", "port_return"]]
        )

    if not records:
        return pd.Series(dtype=float)

    cohort_rets = (
        pd.concat(records, ignore_index=True)
        .sort_values(["formation_date", "realize_date", "portfolio"])
        .reset_index(drop=True)
    )
    mom_df  = compute_mom_series(cohort_rets, h)
    mom_val = mom_df[mom_df["n_active_cohorts"] == h].dropna(subset=["mom_return"])
    return mom_val.set_index("date")["mom_return"]


# ── Table 9 export ─────────────────────────────────────────────────────────────

def _panel_stats(grid: dict) -> dict:
    return {
        key: _nw_stats(series.values)
        for key, series in grid.items()
        if len(series) >= 5
    }


def _export_table9(
    gross_grid: dict,
    net_100: dict,
    net_75: dict,
    net_50: dict,
) -> tuple:
    """
    Produce output/table9_net_returns.csv (3-panel 5×5 grid, Menkhoff layout).
    Returns (s100, s75, s50) for reuse in _write_notes.
    """
    s100 = _panel_stats(net_100)
    s75  = _panel_stats(net_75)
    s50  = _panel_stats(net_50)

    panels = [
        ("Panel A — Net excess returns (100% quoted spread)",   s100),
        ("Panel B — Net excess returns (75% effective spread)", s75),
        ("Panel B — Net excess returns (50% effective spread)", s50),
    ]

    h_cols   = [f"h={h}" for h in H_GRID]
    row_col  = "Row"
    all_cols = [row_col] + h_cols
    rows_out = []
    spacer   = {c: "" for c in all_cols}

    for panel_label, stats in panels:
        rows_out.append({row_col: panel_label, **{f"h={h}": "" for h in H_GRID}})
        for f in F_GRID:
            mean_row  = {row_col: f"f={f}"}
            tstat_row = {row_col: ""}
            for h in H_GRID:
                s  = stats.get((f, h))
                hk = f"h={h}"
                if s is None or math.isnan(s.get("mean", float("nan"))):
                    mean_row[hk]  = ""
                    tstat_row[hk] = ""
                else:
                    mean_row[hk]  = round(s["mean"] * 1200, 2)
                    tstat_row[hk] = f"[{s['t']:.2f}]"
            rows_out.append(mean_row)
            rows_out.append(tstat_row)
        rows_out.append(spacer)

    pd.DataFrame(rows_out, columns=all_cols).to_csv(
        OUTPUT_DIR / "table9_net_returns.csv", index=False
    )
    log = _get_log()
    log.info("Written: table9_net_returns.csv")
    if log.isEnabledFor(logging.INFO):
        _print_table9(panels)

    return s100, s75, s50


def _print_table9(panels: list) -> None:
    col_w = 12
    W     = 10 + len(H_GRID) * col_w
    print()
    print("=" * W)
    print("  §6 Table 9 — Transaction-Cost-Adjusted Momentum Returns")
    print("  Annualised log excess return × 100 (%)  |  [NW t-stat]")
    print("=" * W)
    for panel_label, stats in panels:
        print(f"\n  {panel_label}")
        print(f"  {'f \\ h':<7}" + "".join(f"{'h='+str(h):>{col_w}}" for h in H_GRID))
        print("-" * W)
        for f in F_GRID:
            mean_line  = f"  {'f='+str(f):<7}"
            tstat_line = f"  {'':7}"
            for h in H_GRID:
                s = stats.get((f, h))
                if s is None or math.isnan(s.get("mean", float("nan"))):
                    mean_line  += f"{'N/A':>{col_w}}"
                    tstat_line += f"{'':>{col_w}}"
                else:
                    mean_line  += f"{s['mean'] * 1200:>{col_w}.2f}"
                    tstat_line += f"{'[' + str(round(s['t'], 2)) + ']':>{col_w}}"
            print(mean_line)
            print(tstat_line)
    print("=" * W)

# ── Main entry point ──────────────────────────────────────────────────────────

def run_section6(
    returns_panel: pd.DataFrame,
    section5_result: dict,
) -> dict:
    """
    §6 Transaction-cost-adjusted momentum returns.

    Args:
        returns_panel:   from compute_all_returns()["returns_panel"]
        section5_result: from run_section5()

    Returns:
        {"net_100": dict, "net_75": dict, "net_50": dict}
        Each dict: {(f, h): pd.Series(net_mom_return, index=date)}
    """
    log        = _get_log()
    gross_grid = section5_result["grid_A"]

    # Build lookup tables once — independent of f, h, alpha
    rp = returns_panel[["date", "currency_code", "excess_return",
                         "fwd_bid_t", "fwd_ask_t", "spot_bid_t1", "spot_ask_t1"]].copy()
    rp["ym"] = rp["date"].dt.to_period("M")
    rp_lookup = (
        rp[["currency_code", "ym", "excess_return",
            "fwd_bid_t", "fwd_ask_t", "spot_bid_t1", "spot_ask_t1"]]
        .rename(columns={"ym": "ym_realize"})
    )
    period_to_date = (
        rp[["date", "ym"]].drop_duplicates()
        .rename(columns={"ym": "ym_realize", "date": "realize_date"})
    )

    grids        = {a: {} for a in ALPHAS}
    drop_records = []

    for f in F_GRID:
        log.info("  Signal A f=%d: building assignments ...", f)
        signal_df   = build_signal(returns_panel, f, "A")
        assignments = assign_portfolios(signal_df)
        for h in H_GRID:
            # Merge and drop once per (f, h) — alpha-agnostic
            frames, drops = _prepare_positions(assignments, rp_lookup, h)
            drop_records.extend(drops)
            for alpha in ALPHAS:
                try:
                    grids[alpha][(f, h)] = _compute_mom_from_frames(frames, period_to_date, h, alpha)
                except AssertionError:
                    raise  # assertion failures are bugs — do not swallow
                except Exception as e:
                    log.warning("  SKIP f=%d h=%d alpha=%.2f: %s", f, h, alpha, e)

    s100, s75, s50 = _export_table9(gross_grid, grids[1.00], grids[0.75], grids[0.50])

    return {"net_100": grids[1.00], "net_75": grids[0.75], "net_50": grids[0.50]}
