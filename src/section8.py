#!/usr/bin/env python3
"""
src/section8.py — §8 Sharpe ratio grid.

Reports annualised Sharpe ratios for two strategy families:
  • Signal A momentum (5×5 f×h grid, excess returns)
  • Carry portfolio CT (HML-FD long-short from §7.1)

All input return series are MONTHLY log excess returns (methodology §0.1),
so the annualisation is:

    SR_annual = sqrt(12) · mean(r_monthly) / std(r_monthly, ddof=1)

Brackets show the NW HAC t-stat of H0: mean = 0, with Andrews (1991) lag
L = ceil(0.75 · T^(1/3)). This matches the inference convention used in §5/§7.
"""

import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"

from src.output_writer import add_sheet
from src.stats         import nw_stats, sig_stars

F_GRID = [1, 3, 6, 9, 12]
H_GRID = [1, 3, 6, 9, 12]


def _sharpe_stats(series) -> dict:
    """nw_stats projected to {sharpe, t, p, T, nw_lag} for §8 display."""
    s = nw_stats(series)
    return {"sharpe": s["sharpe"], "t": s["t"], "p": s["p"], "T": s["T"], "nw_lag": s["nw_lag"]}


# ── CSV export ────────────────────────────────────────────────────────────────

def _export_table12(grid_stats: dict, ct_stats: dict) -> None:
    """Wide-format CSV: 5×5 Signal A Sharpe grid + appended CT row."""
    h_cols   = [f"h={h}" for h in H_GRID]
    row_col  = "Row"
    all_cols = [row_col] + h_cols
    spacer   = {c: "" for c in all_cols}

    rows = []

    # Header banner
    rows.append({**spacer, row_col: "Panel — Sharpe ratios (annualised), Signal A: Excess Returns"})

    # 5×5 grid (Sharpe value, then [NW t-stat] underneath)
    for f in F_GRID:
        sr_row = {row_col: f"f={f}"}
        t_row  = {row_col: ""}
        for h in H_GRID:
            s  = grid_stats.get((f, h))
            hk = f"h={h}"
            if s is None or math.isnan(s["sharpe"]):
                sr_row[hk] = ""
                t_row[hk]  = ""
            else:
                sr_row[hk] = round(s["sharpe"], 3)
                t_row[hk]  = f"[{round(s['t'], 3)}]"
        rows.append(sr_row)
        rows.append(t_row)

    # CT (carry portfolio) — single-cell display under h=1 (CT has no f×h grid)
    rows.append(spacer)
    rows.append({**spacer, row_col: "Carry portfolio (CT = HML-FD long-short, §7.1)"})
    if not math.isnan(ct_stats["sharpe"]):
        ct_sr_row = {row_col: "CT", **{f"h={h}": "" for h in H_GRID}}
        ct_t_row  = {row_col: "",   **{f"h={h}": "" for h in H_GRID}}
        ct_sr_row["h=1"] = round(ct_stats["sharpe"], 3)
        ct_t_row["h=1"]  = f"[{round(ct_stats['t'], 3)}]"
        rows.append(ct_sr_row)
        rows.append(ct_t_row)

    add_sheet("table12_sharpe", pd.DataFrame(rows, columns=all_cols))


# ── Terminal display ──────────────────────────────────────────────────────────

def _print_table12(grid_stats: dict, ct_stats: dict) -> None:
    col_w = 12
    W     = 8 + len(H_GRID) * col_w
    print()
    print("=" * W)
    print("  §8 Table 12 — Sharpe Ratios (annualised)")
    print("  Signal A: Excess Returns   |   [NW HAC t-stat]   ***p<.01 **p<.05 *p<.10")
    print("=" * W)
    print(f"  {'f \\ h':<6}" + "".join(f"{'h='+str(h):>{col_w}}" for h in H_GRID))
    print("-" * W)
    for f in F_GRID:
        sr_line = f"  {'f='+str(f):<6}"
        t_line  = f"  {'':6}"
        for h in H_GRID:
            s = grid_stats.get((f, h))
            if s is None or math.isnan(s["sharpe"]):
                sr_line += f"{'N/A':>{col_w}}"
                t_line  += f"{'':>{col_w}}"
            else:
                t_str = f"[{s['t']:.2f}]{sig_stars(s['p'])}"
                sr_line += f"{s['sharpe']:>{col_w}.2f}"
                t_line  += f"{t_str:>{col_w}}"
        print(sr_line)
        print(t_line)
    print("=" * W)

    if not math.isnan(ct_stats["sharpe"]):
        stars = sig_stars(ct_stats["p"])
        print()
        print(
            f"  Carry portfolio (CT = HML-FD long-short, §7.1): "
            f"Sharpe = {ct_stats['sharpe']:.3f}   "
            f"[t = {ct_stats['t']:.2f}]{stars}   "
            f"T = {ct_stats['T']}   NW lag = {ct_stats['nw_lag']}"
        )
        print()


# ── Master ────────────────────────────────────────────────────────────────────

def run_section8(
    section5_result: dict,
    section7_result: dict,
) -> dict:
    """
    §8 master orchestrator.

    Parameters
    ----------
    section5_result : output of run_section5 (must contain 'grid_A')
    section7_result : output of run_section7 (must contain 'ct_series')

    Returns
    -------
    {"sharpe_grid": {(f, h): stats_dict}, "ct_sharpe": stats_dict}
    where each stats_dict carries: sharpe, t, p, T, nw_lag.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading: §8 Sharpe ratio grid (Table 12) ...")
    grid_A = section5_result["grid_A"]
    ct_df  = section7_result["ct_series"]

    # 5×5 Sharpe grid (one cell per (f, h))
    grid_stats = {
        (f, h): _sharpe_stats(series.values)
        for (f, h), series in grid_A.items()
    }

    # Carry CT long-short Sharpe
    ct_stats = _sharpe_stats(ct_df["mom_return"].values)

    _export_table12(grid_stats, ct_stats)
    _print_table12(grid_stats, ct_stats)

    return {"sharpe_grid": grid_stats, "ct_sharpe": ct_stats}
