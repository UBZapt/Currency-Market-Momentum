#!/usr/bin/env python3
"""
src/section5.py — §5 Formation-Holding Grid, portfolio tables, seasonality,
rolling-OLS comparison, and post-holding analysis.

Reuses §4 machinery throughout.
Quintile-only sort is used exclusively for the presentation portfolio-return tables.
All momentum construction (f×h grid, post-holding) uses the dynamic sextile/quintile rule.
skip_month_flag = False (§3 result): no skip-month adjustment applied.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"

from src.signals       import build_signal
from src.portfolios    import (
    assign_portfolios,
    compute_cohort_returns,
    compute_mom_series,
    compute_portfolio_series,
    build_mom_pipeline,
)
from src.output_writer import add_sheet
from src.stats         import nw_stats, sig_stars

F_GRID             = [1, 3, 6, 9, 12]
H_GRID             = [1, 3, 6, 9, 12]
DISPLAY_STRATEGIES = [(1, 1), (6, 1), (12, 1)]
OLS_F_GRID         = [3, 6, 9, 12]   # f=1 excluded (initial_window=36 makes it ill-defined)


# ── Full f×h grid (internal, dynamic sextile/quintile rule) ───────────────────

def _export_fxh_wide(stats_dict: dict, signal_type: str, signal_label: str) -> None:
    """
    Wide-format CSV: four stacked blocks — annualised mean × 100, NW t-stat, standard t-stat, p-value.
    stats_dict: {(f, h): nw_stats dict} — precomputed once in compute_fxh_grid.
    """
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
        [hdr(f"Signal {signal_type} — {signal_label}: Annualised mean return (× 100)")]
        + mean_rows
        + [spacer, hdr("NW HAC t-statistic")]
        + nw_rows
        + [spacer, hdr("Standard t-statistic")]
        + std_rows
        + [spacer, hdr("p-value (3dp, NW HAC)")]
        + pval_rows
    )
    add_sheet(f"table_fxh_signal_{signal_type}", pd.DataFrame(rows, columns=all_cols))


def _print_fxh_table(stats_dict: dict, signal_type: str, signal_label: str) -> None:
    """Print f×h grid table to terminal."""
    col_w = 14
    W     = 8 + len(H_GRID) * col_w
    print()
    print("=" * W)
    print(f"  §5 f×h Momentum Grid — Signal {signal_type}: {signal_label}")
    print(f"  Annualised log excess return × 100  |  [NW HAC t-stat]  (Standard t-stat)")
    print(f"  ***p<.01 **p<.05 *p<.10")
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


def compute_fxh_grid(returns_panel: pd.DataFrame, signal_type: str) -> tuple[dict, dict]:
    """
    Build 25 MOM series for one signal type using the dynamic sextile/quintile rule.

    Also caches per-portfolio return series for h=1 cells (used by §7.2
    sextile-by-sextile correlations) so signal+assign+cohort work is not
    duplicated downstream.

    Returns
    -------
    (mom_grid, port_h1_grid):
      mom_grid     : {(f, h): pd.Series(index=date, data=mom_return)}
      port_h1_grid : {(f, 1): pd.DataFrame(date, portfolio, avg_return)} — dynamic sort
    Saves long-format internal CSV (fxh_series_{A/B}.csv) and wide-format display CSV.
    signal_type: 'A' (excess return) or 'B' (spot change). Neither uses any initial_window.
    """
    assert signal_type in ("A", "B"), f"signal_type must be 'A' or 'B', got '{signal_type}'"
    signal_label = "Excess Returns" if signal_type == "A" else "Spot Rate Changes"

    data_start = returns_panel.dropna(subset=["excess_return"])["date"].min()
    records    = []
    grid_dict  = {}
    port_h1    = {}

    for f in F_GRID:
        # Build signal + assignments once per f and reuse across the h loop;
        # this also lets us cache the h=1 cohort_rets for the per-portfolio cache.
        try:
            signal_df   = build_signal(returns_panel, f, signal_type)
            assignments = assign_portfolios(signal_df)
        except Exception:
            continue

        for h in H_GRID:
            try:
                cohort_rets = compute_cohort_returns(assignments, returns_panel, h)
                mom         = compute_mom_series(cohort_rets, h)
            except Exception:
                continue

            mom_valid = mom[mom["n_active_cohorts"] == h].dropna(subset=["mom_return"])

            # V-GRID: burn-in check — first MOM = data_start + (f-1) + h
            expected_ym = data_start.to_period("M") + (f - 1) + h
            if not mom_valid.empty:
                actual_ym = mom_valid["date"].min().to_period("M")
                if actual_ym < expected_ym:
                    raise ValueError(
                        f"Burn-in violation: f={f} h={h} signal={signal_type}: "
                        f"first date {actual_ym} < expected {expected_ym}"
                    )

            series = mom_valid.set_index("date")["mom_return"]
            grid_dict[(f, h)] = series
            for date, val in series.items():
                records.append({"date": date, "f": f, "h": h, "mom_return": val})

            # Cache per-portfolio series for h=1 cells (used by §7 correlations).
            if h == 1:
                port_ser = compute_portfolio_series(cohort_rets, h)
                port_h1[(f, h)] = (
                    port_ser[port_ser["n_active_cohorts"] == h]
                    [["date", "portfolio", "avg_return"]]
                    .reset_index(drop=True)
                )

    # Compute stats once per cell — shared by export and terminal display
    stats_dict = {
        key: nw_stats(series.values)
        for key, series in grid_dict.items()
        if len(series) >= 5
    }

    add_sheet(
        f"fxh_series_{signal_type}",
        pd.DataFrame(records).sort_values(["f", "h", "date"]).reset_index(drop=True),
    )
    _export_fxh_wide(stats_dict, signal_type, signal_label)
    _print_fxh_table(stats_dict, signal_type, signal_label)

    return grid_dict, port_h1


# ── Quintile-only presentation sort (for portfolio-return tables only) ─────────

def _assign_quintile_only(signal_df: pd.DataFrame, min_n: int = 15) -> pd.DataFrame:
    """
    Quintile-only sort for presentation tables only.
    Forces k=5 for all months with N_t^signal >= min_n; drops months below threshold.
    Do NOT use for momentum construction — use assign_portfolios() for that.
    Tie-break and remainder allocation match assign_portfolios() conventions.

    min_n: full cross-section uses 15; subsamples (DM/EM) use 5
           because DM has only ~12 currencies — the practical quintile floor.
    """
    required = {"date", "currency_code", "signal"}
    if missing := required - set(signal_df.columns):
        raise ValueError(f"signal_df missing columns: {missing}")

    clean  = signal_df.dropna(subset=["signal"]).copy()
    N_ser  = clean.groupby("date")["currency_code"].transform("count")
    clean  = clean[N_ser >= min_n].copy()
    if clean.empty:
        raise ValueError(f"_assign_quintile_only: no months with N_t^signal >= {min_n}")

    clean = clean.sort_values(["date", "signal", "currency_code"]).reset_index(drop=True)
    clean["_rank"] = clean.groupby("date").cumcount()

    N      = clean.groupby("date")["currency_code"].transform("count").astype(int)
    k      = 5
    base   = N // k
    extras = N % k
    thresh = (k - extras) * base

    small = clean["_rank"] < thresh
    clean["portfolio"] = np.where(
        small,
        (clean["_rank"] // base + 1).values,
        ((k - extras) + (clean["_rank"] - thresh) // (base + 1) + 1).values,
    ).astype(int)
    clean["n_portfolios"] = k

    # V-QUINT: all rows must have n_portfolios == 5
    assert (clean["n_portfolios"] == 5).all(), \
        "Quintile-only sort produced rows with n_portfolios != 5"

    return (
        clean[["date", "currency_code", "portfolio", "n_portfolios"]]
        .sort_values(["date", "portfolio"])
        .reset_index(drop=True)
    )


def _quintile_portfolio_series(
    returns_panel: pd.DataFrame,
    f: int,
    h: int,
    msci_filter: str | None = None,
) -> pd.DataFrame:
    """
    Per-portfolio (P1-P5) return series using quintile-only sort for presentation tables.
    msci_filter: None (all currencies) | 'DM' | 'EM'.
    Returns (date, portfolio, avg_return) filtered to post-burn-in months (n_active == h).
    """
    if msci_filter is not None:
        rp = returns_panel[returns_panel["msci_class"] == msci_filter].copy()
        # V-SUBSP: no leakage
        actual_classes = rp["msci_class"].dropna().unique().tolist()
        if actual_classes and actual_classes != [msci_filter]:
            raise ValueError(
                f"Subsample filter leakage: expected only '{msci_filter}', "
                f"got {actual_classes}"
            )
    else:
        rp = returns_panel

    # DM has ~12 currencies, EM can be sparse early-sample; use min_n=5 (quintile floor)
    min_n       = 15 if msci_filter is None else 5
    signal_df   = build_signal(rp, f, "A")
    assignments = _assign_quintile_only(signal_df, min_n=min_n)
    cohort_rets = compute_cohort_returns(assignments, rp, h)
    port_series = compute_portfolio_series(cohort_rets, h)

    return (
        port_series[port_series["n_active_cohorts"] == h]
        [["date", "portfolio", "avg_return"]]
        .reset_index(drop=True)
    )


# ── Portfolio-return table: All / DM / EM × MOM(1,1) / (6,1) / (12,1) ────────

# Display convention: P1=Winners (actual portfolio 5), P5=Losers (actual portfolio 1)
_DISP_ORDER  = ["P1", "P2", "P3", "P4", "P5", "MOM", "RX"]
_DISP_LABEL  = {
    "P1": "P1 (Winners)", "P2": "P2", "P3": "P3", "P4": "P4",
    "P5": "P5 (Losers)", "MOM": "P1-P5 (MOM)", "RX": "RX",
}
# Maps display label → key in stats_cache cell_stats dict (actual portfolio number)
_ACTUAL_KEY  = {
    "P1": "P5", "P2": "P4", "P3": "P3", "P4": "P2", "P5": "P1",
    "MOM": "MOM", "RX": "RX",
}


def _export_portfolio_wide(stats_cache: dict) -> None:
    """Wide-format CSV for portfolio returns table, matching terminal layout."""
    samples      = ["All", "DM", "EM"]
    strat_labels = [f"MOM({f},{h})" for f, h in DISPLAY_STRATEGIES]
    col_names    = [f"{s} {sl}" for s in samples for sl in strat_labels]
    all_cols     = ["Block", "Portfolio"] + col_names

    spacer = {c: "" for c in all_cols}

    def build_block(block_name: str, stat_key: str, scale: float = 1.0) -> list:
        rows = []
        for i, dk_disp in enumerate(_DISP_ORDER):
            dk_act = _ACTUAL_KEY[dk_disp]
            row = {
                "Block":     block_name if i == 0 else "",
                "Portfolio": _DISP_LABEL[dk_disp],
            }
            for s in samples:
                for sl in strat_labels:
                    cs  = stats_cache.get((s, sl), {}).get(dk_act)
                    val = float("nan") if cs is None else cs.get(stat_key, float("nan"))
                    row[f"{s} {sl}"] = "" if math.isnan(val) else round(val * scale, 3)
            rows.append(row)
        return rows

    rows = (
        build_block("Mean × 100", "mean", scale=100)
        + [spacer]
        + build_block("NW t-stat", "t")
        + [spacer]
        + build_block("p-value (3dp)", "p")
    )
    add_sheet("table_portfolio_returns", pd.DataFrame(rows, columns=all_cols))


def _print_portfolio_table(stats_cache: dict) -> None:
    """Print quintile portfolio table to terminal."""
    strat_labels  = [f"MOM({f},{h})" for f, h in DISPLAY_STRATEGIES]
    sample_labels = ["All", "DM", "EM"]

    col_w = 12
    lbl_w = 14
    W     = lbl_w + len(sample_labels) * len(strat_labels) * col_w + 4

    print()
    print("=" * W)
    print("  §5 Quintile Portfolio Returns (mean × 100)  |  [NW t-stat]  ***p<.01 **p<.05 *p<.10")
    print("  P1=Winners (highest signal), MOM = P1-P5.")
    print("=" * W)

    samp_line = f"  {'':^{lbl_w}}"
    for s in sample_labels:
        samp_line += f"  {s:^{col_w * len(strat_labels)}}"
    print(samp_line)

    strat_line = f"  {'':^{lbl_w}}"
    for _ in sample_labels:
        for sl in strat_labels:
            strat_line += f"{sl:>{col_w}}"
    print(strat_line)
    print("-" * W)

    for row_key in _DISP_ORDER:
        dk         = _ACTUAL_KEY[row_key]
        mean_line  = f"  {_DISP_LABEL[row_key]:<{lbl_w}}"
        tstat_line = f"  {'':^{lbl_w}}"
        for s in sample_labels:
            for sl in strat_labels:
                cs = stats_cache.get((s, sl), {}).get(dk)
                if cs is None or math.isnan(cs["mean"]):
                    mean_line  += f"{'N/A':>{col_w}}"
                    tstat_line += f"{'':>{col_w}}"
                else:
                    t_str = f"[{cs['t']:.2f}]{sig_stars(cs['p'])}"
                    mean_line  += f"{cs['mean']*100:>{col_w}.2f}"
                    tstat_line += f"{t_str:>{col_w}}"
        print(mean_line)
        print(tstat_line)
        if row_key in ("P5", "RX"):
            print("-" * W)

    print("=" * W)


def build_portfolio_return_table(
    returns_panel: pd.DataFrame,
    rx_factor: pd.DataFrame,
) -> None:
    """
    Exports table_portfolio_returns.csv (wide display format).
    Rows: P1 (Winners)–P5 (Losers), P1-P5 (MOM), RX.
    Columns: [All|DM|EM] × [MOM(1,1)|MOM(6,1)|MOM(12,1)].
    DM/EM use min_n=5 (~2-3 currencies/quintile).
    RX row uses full-sample rx_factor date-aligned to each strategy's observation window.
    """
    samples     = [("All", None), ("DM", "DM"), ("EM", "EM")]
    stats_cache: dict = {}

    for sample_label, msci_filter in samples:
        for f, h in DISPLAY_STRATEGIES:
            strat_label = f"MOM({f},{h})"
            try:
                port_ser = _quintile_portfolio_series(returns_panel, f, h, msci_filter)
            except Exception:
                continue

            cell_stats: dict = {}
            for p in range(1, 6):
                vals = port_ser[port_ser["portfolio"] == p]["avg_return"].dropna().values
                cell_stats[f"P{p}"] = nw_stats(vals)

            p5 = port_ser[port_ser["portfolio"] == 5].set_index("date")["avg_return"]
            p1 = port_ser[port_ser["portfolio"] == 1].set_index("date")["avg_return"]
            cell_stats["MOM"] = nw_stats((p5 - p1).dropna().values)

            active_dates     = port_ser["date"].unique()
            rx_vals          = rx_factor[rx_factor["date"].isin(active_dates)]["rx_factor"].dropna().values
            cell_stats["RX"] = nw_stats(rx_vals)

            stats_cache[(sample_label, strat_label)] = cell_stats

    _export_portfolio_wide(stats_cache)
    _print_portfolio_table(stats_cache)


# ── Rolling-OLS comparison table ──────────────────────────────────────────────

def build_ols_comparison_table(returns_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Exports table_ols_comparison.csv.
    Equal-weighted (Signal A) vs rolling-OLS for f∈{3,6,9,12}, h=1.
    f=1 excluded: initial_window=36 leaves no meaningful lag decay.
    """
    data_start         = returns_panel.dropna(subset=["excess_return"])["date"].min()
    expected_ols_start = data_start + pd.DateOffset(months=36)
    records = []

    for f in OLS_F_GRID:
        ew_mom   = build_mom_pipeline(returns_panel, f, 1, "A")
        ew_valid = ew_mom[ew_mom["n_active_cohorts"] == 1].dropna(subset=["mom_return"])
        ew_stats = nw_stats(ew_valid["mom_return"].values)

        # initial_window is set internally by signal_rolling_ols (36, per methodology §5.4)
        ols_mom   = build_mom_pipeline(returns_panel, f, 1, "OLS")
        ols_valid = ols_mom[ols_mom["n_active_cohorts"] == 1].dropna(subset=["mom_return"])

        # V-OLS: out-of-sample timing
        if not ols_valid.empty:
            ols_start = ols_valid["date"].min()
            assert ols_start >= expected_ols_start, (
                f"OLS out-of-sample violation f={f}: first date {ols_start.date()} "
                f"< expected {expected_ols_start.date()}"
            )

        ols_stats = nw_stats(ols_valid["mom_return"].values)
        records.append({
            "f":        f,
            "ew_mean":  ew_stats["mean"],  "ew_t":  ew_stats["t"],  "ew_p":  ew_stats["p"],
            "ew_T":     ew_stats["T"],     "ew_nw_lag":  ew_stats["nw_lag"],
            "ols_mean": ols_stats["mean"], "ols_t": ols_stats["t"], "ols_p": ols_stats["p"],
            "ols_T":    ols_stats["T"],    "ols_nw_lag": ols_stats["nw_lag"],
        })

    csv_rows = [{
        "f":                         r["f"],
        "EW annualised mean × 100":  round(r["ew_mean"] * 1200, 3),
        "EW t-stat":                 round(r["ew_t"], 3),
        "EW p-value":                round(r["ew_p"], 3),
        "OLS annualised mean × 100": round(r["ols_mean"] * 1200, 3),
        "OLS t-stat":                round(r["ols_t"], 3),
        "OLS p-value":               round(r["ols_p"], 3),
        "T(EW)":                     r["ew_T"],
        "T(OLS)":                    r["ols_T"],
    } for r in records]
    add_sheet("table_ols_comparison", pd.DataFrame(csv_rows))

    W = 84
    print()
    print("=" * W)
    print("  §5.4 Rolling-OLS vs Equal-Weighted Signal (h=1)")
    print("  Annualised log excess return × 100  |  [NW t-stat]  ***p<.01 **p<.05 *p<.10")
    print("=" * W)
    print(
        f"  {'f':<5}"
        f"{'EW mean':>9}{'EW t-stat':>13}{'EW p':>8}"
        f"{'OLS mean':>11}{'OLS t-stat':>13}{'OLS p':>8}"
        f"{'T(EW)':>8}{'T(OLS)':>8}"
    )
    print("-" * W)
    for row in records:
        ew_t_str  = f"[{row['ew_t']:.2f}]{sig_stars(row['ew_p'])}"
        ols_t_str = f"[{row['ols_t']:.2f}]{sig_stars(row['ols_p'])}"
        print(
            f"  {row['f']:<5}"
            f"{row['ew_mean']*1200:>9.2f}{ew_t_str:>13}"
            f"{row['ew_p']:>8.3f}"
            f"{row['ols_mean']*1200:>11.2f}{ols_t_str:>13}"
            f"{row['ols_p']:>8.3f}"
            f"{row['ew_T']:>8}{row['ols_T']:>8}"
        )
    print("=" * W)

    return pd.DataFrame(csv_rows)


# ── Seasonality table ─────────────────────────────────────────────────────────

def build_seasonality_table(
    returns_panel: pd.DataFrame,
    grid_A: dict,
) -> pd.DataFrame:
    """
    Exports table_seasonality.csv with stacked blocks: mean × 100, NW t-stat, p-value.
    Calendar months 1-12 × MOM(1,1), MOM(6,1), MOM(12,1). Reuses grid_A series dict.
    """
    records = []

    for f, h in DISPLAY_STRATEGIES:
        key = (f, h)
        if key not in grid_A:
            continue
        series      = grid_A[key]
        strat_label = f"MOM({f},{h})"
        for month in range(1, 13):
            s = nw_stats(series[series.index.month == month].dropna().values)
            records.append({"calendar_month": month, "strategy": strat_label, **s})

    df_long      = pd.DataFrame(records)
    month_names  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    strat_labels = [f"MOM({f},{h})" for f, h in DISPLAY_STRATEGIES]
    all_cols     = ["Block", "Month"] + strat_labels
    spacer       = {c: "" for c in all_cols}

    def _wide_block(block_name: str, col_key: str, scale: float = 1.0) -> list:
        rows = []
        for i, month in enumerate(range(1, 13)):
            sub = df_long[df_long["calendar_month"] == month].set_index("strategy")
            row = {"Block": block_name if i == 0 else "", "Month": month_names[month - 1]}
            for sl in strat_labels:
                val = float(sub.loc[sl, col_key]) if sl in sub.index else float("nan")
                row[sl] = "" if math.isnan(val) else round(val * scale, 3)
            rows.append(row)
        return rows

    wide_rows = (
        _wide_block("Mean × 100", "mean", scale=100)
        + [spacer]
        + _wide_block("NW t-stat", "t")
        + [spacer]
        + _wide_block("p-value (3dp)", "p")
    )
    add_sheet("table_seasonality", pd.DataFrame(wide_rows, columns=all_cols))

    # Terminal: mean only
    col_w = 18
    W     = 8 + len(strat_labels) * col_w
    print()
    print("=" * W)
    print("  §4.4 Seasonality — Mean monthly return × 100")
    print("=" * W)
    print(f"  {'Month':<6}" + "".join(f"{sl:>{col_w}}" for sl in strat_labels))
    print("-" * W)
    for month in range(1, 13):
        sub      = df_long[df_long["calendar_month"] == month].set_index("strategy")
        row_line = f"  {month_names[month-1]:<6}"
        for sl in strat_labels:
            if sl not in sub.index or math.isnan(float(sub.loc[sl, "mean"])):
                row_line += f"{'N/A':>{col_w}}"
            else:
                row_line += f"{float(sub.loc[sl, 'mean']) * 100:>{col_w}.2f}"
        print(row_line)
    print("=" * W)

    return df_long


# ── Post-holding figure ───────────────────────────────────────────────────────

def _post_holding_one_strategy(
    returns_panel: pd.DataFrame,
    f: int,
    K: int = 60,
) -> tuple:
    """
    For MOM(f,1): freeze cohort composition at formation, track K months.
    Returns (k_axis, cum_r, cum_lower, cum_upper) arrays.
    CI bands are ±1.96 × cross-cohort SE of cumulative spread (not a bootstrap).
    Uses dynamic sextile/quintile rule (assign_portfolios).
    """
    signal_df   = build_signal(returns_panel, f, "A")
    assignments = assign_portfolios(signal_df)
    cohort_rets = compute_cohort_returns(assignments, returns_panel, h=K)

    form_ord    = cohort_rets["formation_date"].dt.to_period("M").astype("int64")
    realize_ord = cohort_rets["realize_date"].dt.to_period("M").astype("int64")
    cohort_rets = cohort_rets.copy()
    cohort_rets["k"] = (realize_ord - form_ord).values

    # Per-cohort, per-event-month spread: P_high - P_low
    high_low_rows = []
    for (form_date, k_event), kgrp in cohort_rets.groupby(["formation_date", "k"]):
        k_val   = int(kgrp["n_portfolios"].iloc[0])
        hi_rows = kgrp[kgrp["portfolio"] == k_val]["port_return"]
        lo_rows = kgrp[kgrp["portfolio"] == 1]["port_return"]
        if not hi_rows.empty and not lo_rows.empty:
            hi = float(hi_rows.iloc[0])
            lo = float(lo_rows.iloc[0])
            if not (np.isnan(hi) or np.isnan(lo)):
                high_low_rows.append({"formation_date": form_date, "k": k_event, "spread": hi - lo})

    hl_df = pd.DataFrame(high_low_rows)
    if hl_df.empty:
        raise ValueError(f"No post-holding spreads for f={f}")

    r_bar = hl_df.groupby("k")["spread"].mean()
    T_k   = hl_df.groupby("k")["spread"].count()

    # V-POST: T_k non-increasing in k; T_K > 0
    T_k_full = T_k.reindex(range(1, K + 1), fill_value=0)
    diffs    = T_k_full.diff().dropna()
    assert T_k_full.get(K, 0) > 0, \
        f"T_{K} = 0 for f={f} — no cohorts complete {K} months post-formation"

    cohort_spread = hl_df.pivot(index="formation_date", columns="k", values="spread")
    cohort_spread = cohort_spread.reindex(columns=range(1, K + 1))

    cum_r  = np.zeros(K)
    cum_se = np.full(K, float("nan"))
    for k_event in range(1, K + 1):
        cum_r[k_event - 1] = float(
            r_bar.reindex(range(1, k_event + 1), fill_value=float("nan")).sum(skipna=True)
        )
        cum_c = cohort_spread.iloc[:, :k_event].sum(axis=1, skipna=False).dropna()
        Tk    = len(cum_c)
        if Tk > 1:
            cum_se[k_event - 1] = float(cum_c.std(ddof=1)) / np.sqrt(Tk)

    k_axis = np.arange(1, K + 1)
    return k_axis, cum_r, cum_r - 1.96 * cum_se, cum_r + 1.96 * cum_se


def build_post_holding_figure(returns_panel: pd.DataFrame) -> None:
    """
    Exports figure1_post_holding.pdf.
    MOM(1,1), MOM(6,1), MOM(12,1): frozen cohort tracked 60 months post-formation.
    Shaded bands are ±1.96 × cross-cohort SE of cumulative high-minus-low spread.
    """
    K          = 60
    strategies = [(1, "#1f77b4"), (6, "#ff7f0e"), (12, "#2ca02c")]

    fig, ax = plt.subplots(figsize=(10, 6))
    for f, color in strategies:
        k_axis, cum_r, lower, upper = _post_holding_one_strategy(returns_panel, f, K)
        ax.plot(k_axis, cum_r, label=f"MOM({f},1)", color=color, linewidth=1.8)
        ax.fill_between(k_axis, lower, upper, alpha=0.15, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Months after portfolio formation")
    ax.set_ylabel("Cumulative log excess return")
    ax.set_title("Post-Formation Cumulative Returns (60 months, frozen cohort composition)")
    ax.legend()
    fig.tight_layout()

    out_path = OUTPUT_DIR / "figure1_post_holding.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {out_path.name}")


# ── Master function ────────────────────────────────────────────────────────────

def run_section5(
    returns_panel: pd.DataFrame,
    rx_factor: pd.DataFrame,
) -> dict:
    """
    §5 master orchestrator.
    Returns {"grid_A", "grid_B", "grid_A_port_h1", "grid_B_port_h1"} for downstream
    sections. The "_port_h1" caches hold per-portfolio (dynamic sextile/quintile)
    return series for h=1, reused by §7 sextile-by-sextile correlations.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading: §5.2 f×h grid — Signal A (excess return) ...")
    grid_A, grid_A_port_h1 = compute_fxh_grid(returns_panel, "A")
    print("Loading: §5.2 f×h grid — Signal B (spot change) ...")
    grid_B, grid_B_port_h1 = compute_fxh_grid(returns_panel, "B")
    print("Loading: §5 portfolio-return table (All / DM / EM) ...")
    build_portfolio_return_table(returns_panel, rx_factor)
    print("Loading: §5.4 rolling-OLS comparison ...")
    build_ols_comparison_table(returns_panel)
    print("Loading: §4.4 seasonality table ...")
    build_seasonality_table(returns_panel, grid_A)
    print("Loading: §5.5 post-holding period figure ...")
    build_post_holding_figure(returns_panel)

    return {
        "grid_A":         grid_A,
        "grid_B":         grid_B,
        "grid_A_port_h1": grid_A_port_h1,
        "grid_B_port_h1": grid_B_port_h1,
    }
