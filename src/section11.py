#!/usr/bin/env python3
"""
src/section11.py — §11.2 Volatility-Scaled (TSMOM) f×h Grid (Appendix Table A3).

Single deliverable: a replica of the Signal A f×h grid (§5) under the
volatility-scaled signal from methodology §11.2 (Moskowitz, Ooi, Pedersen 2012):

    S_{i,t}^{TSMOM} = ( Σ_{j=0}^{f-1} rx_{i,t-j} ) / σ̂_{i,t}

where σ̂_{i,t} is the rolling 36-month standard deviation of rx_i ending at t.

Same dynamic sextile/quintile sort, same overlapping JT holding convention,
and same NW HAC inference (Andrews lag) as the Signal A grid.
"""

import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"

from src.portfolios    import build_mom_pipeline
from src.output_writer import add_sheet
from src.stats         import nw_stats, sig_stars

F_GRID = [1, 3, 6, 9, 12]
H_GRID = [1, 3, 6, 9, 12]


# ── CSV export (same shape as table_fxh_signal_A.csv) ────────────────────────

def _export_tableA3(stats_dict: dict) -> None:
    col_label = "Row"
    h_cols    = [f"h={h}" for h in H_GRID]
    all_cols  = [col_label] + h_cols

    mean_rows, nw_rows, std_rows, pval_rows = [], [], [], []
    for f in F_GRID:
        mr = {col_label: f"f={f}"}
        nr = {col_label: f"f={f}"}
        sr = {col_label: f"f={f}"}
        pr = {col_label: f"f={f}"}
        for h in H_GRID:
            s = stats_dict.get((f, h))
            if s is None or math.isnan(s["mean"]):
                mr[f"h={h}"] = nr[f"h={h}"] = sr[f"h={h}"] = pr[f"h={h}"] = ""
            else:
                mr[f"h={h}"] = round(s["mean"] * 1200, 3)
                nr[f"h={h}"] = round(s["t"], 3)
                sr[f"h={h}"] = round(s["std_t"], 3)
                pr[f"h={h}"] = round(s["p"], 3)
        mean_rows.append(mr)
        nw_rows.append(nr)
        std_rows.append(sr)
        pval_rows.append(pr)

    def hdr(text):
        return {c: "" for c in all_cols} | {col_label: text}

    spacer = {c: "" for c in all_cols}
    rows = (
        [hdr("Appendix Table A3 — TSMOM (vol-scaled Signal A): "
             "Annualised mean return (× 100)")]
        + mean_rows
        + [spacer, hdr("NW HAC t-statistic")]
        + nw_rows
        + [spacer, hdr("Standard t-statistic")]
        + std_rows
        + [spacer, hdr("p-value (3dp, NW HAC)")]
        + pval_rows
    )
    add_sheet("tableA3_tsmom_grid", pd.DataFrame(rows, columns=all_cols))


# ── Terminal display (mirrors §5 f×h printout) ───────────────────────────────

def _print_tableA3(stats_dict: dict) -> None:
    col_w = 14
    W     = 8 + len(H_GRID) * col_w
    print()
    print("=" * W)
    print("  §11.2 Appendix Table A3 — TSMOM (Volatility-Scaled) f×h Grid")
    print("  Signal: (Σ_{j=0}^{f-1} rx_{t-j}) / σ̂_{36m,t}")
    print("  Annualised log excess return × 100  |  [NW HAC t-stat]  (Std t-stat)")
    print("  ***p<.01 **p<.05 *p<.10")
    print("=" * W)
    print(f"  {'f \\ h':<6}" + "".join(f"{'h='+str(h):>{col_w}}" for h in H_GRID))
    print("-" * W)
    for f in F_GRID:
        mean_line = f"  {'f='+str(f):<6}"
        nw_line   = f"  {'':6}"
        std_line  = f"  {'':6}"
        for h in H_GRID:
            s = stats_dict.get((f, h))
            if s is None or math.isnan(s["mean"]):
                mean_line += f"{'N/A':>{col_w}}"
                nw_line   += f"{'':>{col_w}}"
                std_line  += f"{'':>{col_w}}"
            else:
                nw_str  = f"[{s['t']:.2f}]{sig_stars(s['p'])}"
                std_str = f"({s['std_t']:.2f})"
                mean_line += f"{s['mean']*1200:>{col_w}.2f}"
                nw_line   += f"{nw_str:>{col_w}}"
                std_line  += f"{std_str:>{col_w}}"
        print(mean_line)
        print(nw_line)
        print(std_line)
    print("=" * W)


# ── Master ────────────────────────────────────────────────────────────────────

def run_section11(returns_panel: pd.DataFrame) -> dict:
    """
    §11.2 only — TSMOM volatility-scaled f×h grid.

    Returns {(f, h): pd.Series(index=date, mom_return)}.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading: §11.2 TSMOM f×h grid (Table A3) ...")
    grid_dict: dict = {}
    records: list  = []

    for f in F_GRID:
        for h in H_GRID:
            try:
                mom = build_mom_pipeline(returns_panel, f, h, "TSMOM")
            except Exception:
                continue

            mom_valid = mom[mom["n_active_cohorts"] == h].dropna(subset=["mom_return"])
            series    = mom_valid.set_index("date")["mom_return"]
            grid_dict[(f, h)] = series
            for date, val in series.items():
                records.append({"date": date, "f": f, "h": h, "mom_return": val})

    stats_dict = {
        key: nw_stats(series.values)
        for key, series in grid_dict.items()
        if len(series) >= 5
    }

    add_sheet(
        "fxh_series_TSMOM",
        pd.DataFrame(records).sort_values(["f", "h", "date"]).reset_index(drop=True),
    )
    _export_tableA3(stats_dict)
    _print_tableA3(stats_dict)

    return grid_dict
