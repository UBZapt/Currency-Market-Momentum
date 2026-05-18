#!/usr/bin/env python3
"""
src/section7.py — §7 Carry Trade Comparison and Double Sort.

§7.1  Carry portfolio: sorted by current forward discount (f_t − s_t), h=1.
§7.2  Pairwise correlations CT vs MOM(f, h*=1), sextile-by-sextile format.
§7.3  3×3 double sort: outer FD tertile, inner MOM tertile for f∈{1,6,12}.

Carry signal timing: forward_discount at returns_panel date=t+1 equals f_t − s_t
(carry known at formation date t). Shift date back by one month so the
formation_date convention matches the momentum signal convention used in §4/§5.
h* = 1 (from §6 determination). skip_month_flag = False (§3).
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
from src.portfolios import (
    assign_portfolios,
    compute_cohort_returns,
    compute_mom_series,
    compute_portfolio_series,
    build_portfolio_pipeline,
)

F_GRID_CORR  = [1, 3, 6, 9, 12]
F_GRID_DSORT = [1, 6, 12]
H_STAR       = 1


def _get_log() -> logging.Logger:
    log = logging.getLogger("section7")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(levelname)-7s  %(message)s"))
        log.addHandler(h)
        log.propagate = False
    log.setLevel(logging.INFO)
    return log


# ── HAC inference (local copy matching sections 5/6) ─────────────────────────

def _nw_lag(T: int) -> int:
    return math.ceil(0.75 * T ** (1 / 3))


def _nw_stats(series) -> dict:
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    T   = len(arr)
    if T < 5:
        return {"mean": np.nan, "t": np.nan, "p": np.nan,
                "se": np.nan, "nw_lag": 0, "T": T, "sharpe": np.nan}
    L      = _nw_lag(T)
    res    = sm.OLS(arr, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    std    = float(np.std(arr, ddof=1))
    sharpe = float(np.mean(arr) / std * np.sqrt(12)) if std > 0 else np.nan
    return {
        "mean":   float(res.params[0]),
        "t":      float(res.tvalues[0]),
        "p":      float(res.pvalues[0]),
        "se":     float(res.bse[0]),
        "nw_lag": L,
        "T":      T,
        "sharpe": sharpe,
    }


def _sig_stars(p: float) -> str:
    if math.isnan(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _fmt(val, decimals=3) -> str:
    return "" if (isinstance(val, float) and math.isnan(val)) else str(round(val, decimals))


# ── Carry signal ──────────────────────────────────────────────────────────────

def _make_carry_signal(returns_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Signal = forward_discount (f_t − s_t) at formation date t.

    In returns_panel, the row with date=t+1 contains forward_discount = f_t − s_t
    (it is the carry observable at end of t). Shift date back by one MonthEnd so
    that formation_date = t, matching the convention used in §4/§5 signal machinery.

    Filter to rows with both forward_discount and excess_return non-null so that
    every ranked currency is also eligible to contribute a portfolio return.
    """
    df = (
        returns_panel[["date", "currency_code", "forward_discount", "excess_return"]]
        .dropna(subset=["forward_discount", "excess_return"])
        .copy()
    )
    df["date"] = df["date"] + pd.offsets.MonthEnd(-1)
    return (
        df[["date", "currency_code", "forward_discount"]]
        .rename(columns={"forward_discount": "signal"})
        .reset_index(drop=True)
    )


# ── §7.1 Carry pipeline ───────────────────────────────────────────────────────

def _build_carry_pipeline(
    returns_panel: pd.DataFrame,
) -> tuple:
    """
    Full carry portfolio pipeline (always h=1).

    Returns
    -------
    ct_df       : (date, mom_return)           — CT long-short series, burn-in filtered
    carry_port_df: (date, portfolio, avg_return) — per-portfolio carry returns
    """
    carry_sig   = _make_carry_signal(returns_panel)
    assignments = assign_portfolios(carry_sig)
    cohort_rets = compute_cohort_returns(assignments, returns_panel, h=1)
    ct_raw      = compute_mom_series(cohort_rets, h=1)
    port_raw    = compute_portfolio_series(cohort_rets, h=1)

    ct_df = (
        ct_raw[ct_raw["n_active_cohorts"] == 1]
        .dropna(subset=["mom_return"])
        .reset_index(drop=True)
    )
    carry_port_df = (
        port_raw[port_raw["n_active_cohorts"] == 1]
        [["date", "portfolio", "avg_return"]]
        .reset_index(drop=True)
    )
    return ct_df, carry_port_df


# ── §7.2 MOM-carry correlation analysis ──────────────────────────────────────

def _build_table10(
    ct_df: pd.DataFrame,
    carry_port_df: pd.DataFrame,
    returns_panel: pd.DataFrame,
    section5_result: dict,
) -> pd.DataFrame:
    """
    Table 10: Part A — carry sextile stats; Part B — MOM-carry correlations.
    Exports table10_carry_correlations.csv.
    """
    log    = _get_log()
    grid_A = section5_result["grid_A"]

    # ── Part A: carry portfolio statistics ────────────────────────────────────
    max_p  = int(carry_port_df["portfolio"].max())
    rowsA  = []

    for p in range(1, max_p + 1):
        vals = carry_port_df[carry_port_df["portfolio"] == p]["avg_return"].dropna().values
        s    = _nw_stats(vals)
        if p == 1:
            lbl = "P1 (Low FD)"
        elif p == max_p:
            lbl = f"P{max_p} (High FD)"
        else:
            lbl = f"P{p}"
        rowsA.append({
            "part":     "A",
            "label":    lbl,
            "mean_ann": _fmt(s["mean"] * 1200),
            "t_stat":   _fmt(s["t"]),
            "p_val":    _fmt(s["p"]),
            "sharpe":   _fmt(s["sharpe"]),
            "T":        s["T"],
        })

    ct_s = _nw_stats(ct_df["mom_return"].values)
    rowsA.append({
        "part":     "A",
        "label":    "CT (HML carry)",
        "mean_ann": _fmt(ct_s["mean"] * 1200),
        "t_stat":   _fmt(ct_s["t"]),
        "p_val":    _fmt(ct_s["p"]),
        "sharpe":   _fmt(ct_s["sharpe"]),
        "T":        ct_s["T"],
    })

    # Terminal print — Part A
    W = 76
    print()
    print("=" * W)
    print("  §7.1 Carry Portfolio Statistics (annualised mean × 100 = mean_ann)")
    print("  ***p<.01 **p<.05 *p<.10")
    print("=" * W)
    print(f"  {'Portfolio':<20}{'Mean×1200':>12}{'NW t-stat':>12}{'p-value':>10}{'Sharpe':>10}{'T':>8}")
    print("-" * W)
    for r in rowsA:
        t_star = _sig_stars(float(r["p_val"])) if r["p_val"] else ""
        print(
            f"  {r['label']:<20}"
            f"{r['mean_ann']:>12}"
            f"{str(r['t_stat']) + t_star:>12}"
            f"{r['p_val']:>10}"
            f"{r['sharpe']:>10}"
            f"{r['T']:>8}"
        )
    print("=" * W)

    # ── Part B: correlations CT vs MOM(f, h*=1) ───────────────────────────────
    ct_indexed   = ct_df.set_index("date")["mom_return"]
    carry_pivot  = carry_port_df.pivot(index="date", columns="portfolio", values="avg_return")
    rowsB        = []

    for f in F_GRID_CORR:
        key = (f, H_STAR)
        if key not in grid_A:
            log.warning("  §7.2: grid_A missing (%d,%d) — skipping f=%d", f, H_STAR, f)
            continue

        mom_series = grid_A[key]
        aligned_ct = pd.DataFrame({"CT": ct_indexed, "MOM": mom_series}).dropna()
        overall    = aligned_ct["CT"].corr(aligned_ct["MOM"]) if len(aligned_ct) >= 5 else np.nan

        # Per-portfolio sextile correlations
        mom_port = build_portfolio_pipeline(returns_panel, f, H_STAR, "A")
        mom_port = mom_port[mom_port["n_active_cohorts"] == H_STAR][
            ["date", "portfolio", "avg_return"]
        ]
        mom_pivot = mom_port.pivot(index="date", columns="portfolio", values="avg_return")

        port_corrs = {}
        for p in range(1, 7):
            if p in carry_pivot.columns and p in mom_pivot.columns:
                aln = pd.DataFrame({"c": carry_pivot[p], "m": mom_pivot[p]}).dropna()
                port_corrs[f"P{p}"] = round(aln["c"].corr(aln["m"]), 3) if len(aln) >= 5 else np.nan
            else:
                port_corrs[f"P{p}"] = np.nan

        rowsB.append({
            "part":         "B",
            "label":        f"f={f}",
            "overall_corr": _fmt(overall),
            **{k: _fmt(v) for k, v in port_corrs.items()},
            "N_overlap":    len(aligned_ct),
        })
        log.info(
            "  §7.2 f=%d: corr(CT,MOM)=%.3f  N_overlap=%d",
            f, overall if not math.isnan(overall) else -99, len(aligned_ct)
        )

    # Terminal print — Part B
    p_hdrs = ["P1", "P2", "P3", "P4", "P5", "P6"]
    print()
    print("=" * W)
    print("  §7.2 MOM-Carry Correlations  (h*=1 for MOM)")
    print("=" * W)
    print(f"  {'f':<8}{'corr(CT,MOM)':>14}" + "".join(f"{ph:>8}" for ph in p_hdrs))
    print("-" * W)
    for r in rowsB:
        p_str = "".join(f"{str(r.get(ph, '')):>8}" for ph in p_hdrs)
        print(f"  {r['label']:<8}{r['overall_corr']:>14}{p_str}")
    print("=" * W)

    # ── Export ────────────────────────────────────────────────────────────────
    col_order = ["part", "label", "mean_ann", "t_stat", "p_val", "sharpe",
                 "T", "overall_corr", "P1", "P2", "P3", "P4", "P5", "P6", "N_overlap"]
    spacer    = {c: "" for c in col_order}
    combined  = (
        [{c: r.get(c, "") for c in col_order} for r in rowsA]
        + [spacer]
        + [{c: r.get(c, "") for c in col_order} for r in rowsB]
    )
    out = pd.DataFrame(combined, columns=col_order)
    out.to_csv(OUTPUT_DIR / "table10_carry_correlations.csv", index=False)
    log.info("Written: table10_carry_correlations.csv")
    return out


# ── §7.3 Double sort ──────────────────────────────────────────────────────────

def _tertile_labels(N: int) -> np.ndarray:
    """
    Assign tertile labels 1/2/3 to N pre-sorted items using linspace boundaries.
    Extras go to higher-indexed groups (consistent with assign_portfolios remainder rule).
    """
    ranks = np.arange(1, N + 1, dtype=float)
    bins  = np.linspace(0, N, 4)
    return pd.cut(ranks, bins=bins, labels=[1, 2, 3], include_lowest=True).astype(int)


def _double_sort_f(
    returns_panel: pd.DataFrame,
    carry_sig: pd.DataFrame,
    f: int,
) -> dict:
    """
    3×3 double sort for one formation horizon f, h=1.

    Outer sort: FD tertiles (1=low carry, 3=high carry).
    Inner sort: MOM tertiles within each FD group (1=low, 3=high).
    Drop month if N < 9 (< 3 per outer group) or any FD sub-group < 3.

    Returns dict with keys 'cell_series' and 'hml_fd_series':
      cell_series : {(fd, mom): pd.Series(date → avg excess_return)}
      hml_fd_series: {mom_col: pd.Series(date → spread)}  (fd=3 minus fd=1)
    """
    mom_sig = build_signal(returns_panel, f, "A")

    # Merge carry and MOM signals on formation date
    merged = mom_sig.merge(
        carry_sig.rename(columns={"signal": "signal_carry"}),
        on=["date", "currency_code"],
    ).rename(columns={"signal": "signal_mom"})

    # Return lookup: formation_date t → excess_return at t+1
    ret_lk = (
        returns_panel[["date", "currency_code", "excess_return"]]
        .dropna(subset=["excess_return"])
        .copy()
    )
    ret_lk["formation_date"] = ret_lk["date"] + pd.offsets.MonthEnd(-1)
    ret_lk = ret_lk[["formation_date", "currency_code", "excess_return"]]

    records = []
    for form_date, grp in merged.groupby("date"):
        grp = grp.dropna(subset=["signal_mom", "signal_carry"])
        N   = len(grp)
        if N < 9:
            continue

        # Outer FD tertile (sort ascending: low carry → P1)
        grp_fd = grp.sort_values(["signal_carry", "currency_code"]).reset_index(drop=True)
        grp_fd["fd_group"] = _tertile_labels(N)

        # Inner MOM tertile within each FD group
        for fd_g, fd_sub in grp_fd.groupby("fd_group"):
            Ng = len(fd_sub)
            if Ng < 3:
                continue
            fd_sub_s = fd_sub.sort_values(["signal_mom", "currency_code"]).reset_index(drop=True)
            fd_sub_s["mom_group"] = _tertile_labels(Ng)
            for _, row in fd_sub_s.iterrows():
                records.append({
                    "formation_date": form_date,
                    "fd_group":       int(fd_g),
                    "mom_group":      int(row["mom_group"]),
                    "currency_code":  row["currency_code"],
                })

    if not records:
        return {"cell_series": {}, "hml_fd_series": {}}

    cell_df  = pd.DataFrame(records)
    cell_df  = cell_df.merge(ret_lk, on=["formation_date", "currency_code"], how="left")
    cell_avg = (
        cell_df.groupby(["formation_date", "fd_group", "mom_group"])["excess_return"]
        .mean()
        .reset_index()
    )

    pivot = cell_avg.pivot_table(
        index=["formation_date", "fd_group"], columns="mom_group", values="excess_return"
    ).reset_index()

    # HML-MOM within each FD group (mom=3 minus mom=1)
    for col in [1, 2, 3]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["HML_MOM"] = pivot[3] - pivot[1]

    # Build cell series dict
    cell_series = {}
    for fd_g in [1, 2, 3]:
        sub = pivot[pivot["fd_group"] == fd_g].set_index("formation_date")
        for mom_g in [1, 2, 3, "HML_MOM"]:
            if mom_g in sub.columns:
                cell_series[(fd_g, mom_g)] = sub[mom_g].dropna()

    # HML-FD series: fd=3 minus fd=1 within each MOM group
    fd1 = pivot[pivot["fd_group"] == 1].set_index("formation_date")
    fd3 = pivot[pivot["fd_group"] == 3].set_index("formation_date")
    hml_fd_series = {}
    for mom_g in [1, 2, 3, "HML_MOM"]:
        if mom_g in fd1.columns and mom_g in fd3.columns:
            diff = (fd3[mom_g] - fd1[mom_g]).dropna()
            if len(diff) >= 5:
                hml_fd_series[mom_g] = diff

    return {"cell_series": cell_series, "hml_fd_series": hml_fd_series}


def _build_table11(
    returns_panel: pd.DataFrame,
    carry_sig: pd.DataFrame,
) -> pd.DataFrame:
    """
    Table 11: 3×3 double sort returns for f∈{1,6,12}.
    Rows: FD groups (Low/Mid/High) + HML-FD. Cols: MOM groups (Low/Mid/High) + HML-MOM.
    Mean × 1200 and [NW t-stat] reported. Stacked panels (one per f).
    Exports table11_double_sort.csv.
    """
    log      = _get_log()
    FD_LBLS  = {1: "Low FD", 2: "Mid FD", 3: "High FD", "HML": "HML-FD"}
    MOM_LBLS = {1: "Low MOM", 2: "Mid MOM", 3: "High MOM", "HML_MOM": "HML-MOM"}
    MOM_COLS = [1, 2, 3, "HML_MOM"]
    FD_ROWS  = [1, 2, 3, "HML"]

    col_keys = ["f", "fd_group", "mean_lowMOM", "t_lowMOM",
                "mean_midMOM", "t_midMOM", "mean_highMOM", "t_highMOM",
                "mean_HML_MOM", "t_HML_MOM", "T_HML_MOM"]
    all_rows = []

    W = 86
    print()
    print("=" * W)
    print("  §7.3 Double Sort — 3×3 FD × MOM  (h=1)")
    print("  Annualised mean × 100 = mean_ann  |  [NW t-stat]  ***p<.01 **p<.05 *p<.10")
    print("=" * W)

    for f in F_GRID_DSORT:
        result = _double_sort_f(returns_panel, carry_sig, f)
        cs     = result["cell_series"]
        hml_fd = result["hml_fd_series"]

        print(f"\n  f = {f}")
        hdr = f"  {'':18}" + "".join(f"{MOM_LBLS[m]:>16}" for m in MOM_COLS)
        print(hdr)
        print(f"  {'-' * (W - 2)}")

        for fd_row in FD_ROWS:
            mean_line = f"  {FD_LBLS[fd_row]:<18}"
            tstat_line= f"  {'':18}"
            for mom_g in MOM_COLS:
                if fd_row == "HML":
                    series = hml_fd.get(mom_g, pd.Series(dtype=float))
                else:
                    series = cs.get((fd_row, mom_g), pd.Series(dtype=float))

                s = _nw_stats(series.values)
                if math.isnan(s["mean"]):
                    mean_line  += f"{'N/A':>16}"
                    tstat_line += f"{'':>16}"
                else:
                    t_str = f"[{s['t']:.2f}]{_sig_stars(s['p'])}"
                    mean_line  += f"{s['mean']*1200:>16.2f}"
                    tstat_line += f"{t_str:>16}"

                # Store in CSV rows
                mom_key = MOM_LBLS[mom_g].replace(" ", "_")
                row_key = f"f={f}_fd={FD_LBLS[fd_row]}_mom={MOM_LBLS[mom_g]}"
                all_rows.append({
                    "f":        f,
                    "fd_group": FD_LBLS[fd_row],
                    "mom_group":MOM_LBLS[mom_g],
                    "mean_ann": round(s["mean"] * 1200, 3) if not math.isnan(s["mean"]) else "",
                    "t_stat":   round(s["t"],            3) if not math.isnan(s["t"])    else "",
                    "p_val":    round(s["p"],            3) if not math.isnan(s["p"])    else "",
                    "T":        s["T"],
                })

            print(mean_line)
            print(tstat_line)
            if fd_row in (3, "HML"):
                print(f"  {'-' * (W - 2)}")

    print("=" * W)

    out = pd.DataFrame(all_rows, columns=["f", "fd_group", "mom_group",
                                           "mean_ann", "t_stat", "p_val", "T"])
    out.to_csv(OUTPUT_DIR / "table11_double_sort.csv", index=False)
    log.info("Written: table11_double_sort.csv  (%d rows)", len(out))
    return out


# ── Notes export ──────────────────────────────────────────────────────────────

def _write_notes(ct_df: pd.DataFrame, carry_port_df: pd.DataFrame) -> None:
    ct_s   = _nw_stats(ct_df["mom_return"].values)
    max_p  = int(carry_port_df["portfolio"].max())
    lines  = [
        "Section 7 — Carry Trade Implementation Notes",
        "=" * 60,
        "",
        "1. CARRY SIGNAL TIMING",
        "-" * 40,
        "  forward_discount at returns_panel date=t+1 = f_t − s_t (known at end of t).",
        "  Shifted back by MonthEnd(-1) so formation_date = t, matching §4/§5 convention.",
        "  compute_cohort_returns looks up excess_return at formation_date+1 = t+1 ✓",
        "",
        "2. CARRY PORTFOLIO SUMMARY",
        "-" * 40,
        f"  T = {ct_s['T']} months  NW_lag = {ct_s['nw_lag']}",
        f"  CT mean (annualised) = {round(ct_s['mean']*1200, 3)}%",
        f"  CT NW t-stat         = {round(ct_s['t'], 3)}",
        f"  CT Sharpe            = {round(ct_s['sharpe'], 3)}",
        f"  n_portfolios (typical) = {max_p}",
        "",
        "3. DOUBLE SORT",
        "-" * 40,
        f"  f values: {F_GRID_DSORT}",
        "  Outer sort: FD tertiles (linspace boundaries, extras to higher groups).",
        "  Inner sort: MOM tertiles within each FD group, same convention.",
        "  Drop month if N_outer < 9 or any FD sub-group < 3.",
        "  h=1: every formation month is a single-period exit cohort.",
        "",
        "4. CORRELATION ANALYSIS",
        "-" * 40,
        f"  h* = {H_STAR} (from §6 determination).",
        f"  MOM series extracted from section5_result['grid_A'][(f, {H_STAR})] — no re-computation.",
        "  Sextile-by-sextile correlations use overlapping dates where both series have data.",
    ]
    (OUTPUT_DIR / "section7_notes.txt").write_text("\n".join(lines), encoding="utf-8")
    _get_log().info("Written: section7_notes.txt")


# ── Master function ────────────────────────────────────────────────────────────

def run_section7(
    returns_panel: pd.DataFrame,
    section5_result: dict,
) -> dict:
    """
    §7 master orchestrator.

    Parameters
    ----------
    returns_panel   : full returns panel from §2 (must include forward_discount)
    section5_result : {"grid_A": dict, "grid_B": dict} from run_section5

    Returns
    -------
    dict with keys: ct_series, carry_port_series, table10, table11
    """
    log = _get_log()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("§7.1 Building carry pipeline (h=1) ...")
    ct_df, carry_port_df = _build_carry_pipeline(returns_panel)
    log.info(
        "  Carry CT: T=%d months, mean_ann=%.2f%%, NW t=%.2f",
        len(ct_df),
        _nw_stats(ct_df["mom_return"].values)["mean"] * 1200,
        _nw_stats(ct_df["mom_return"].values)["t"],
    )

    log.info("§7.2 MOM-carry correlations ...")
    carry_sig = _make_carry_signal(returns_panel)
    table10   = _build_table10(ct_df, carry_port_df, returns_panel, section5_result)

    log.info("§7.3 Double sort (f in %s) ...", F_GRID_DSORT)
    table11   = _build_table11(returns_panel, carry_sig)

    _write_notes(ct_df, carry_port_df)

    return {
        "ct_series":        ct_df,
        "carry_port_series":carry_port_df,
        "table10":          table10,
        "table11":          table11,
    }
