#!/usr/bin/env python3
"""
src/section9.py — §9.1 Fama-MacBeth cross-sectional regressions (Table 13).

Replicates MSSS Table 6.

Three panels, ℓ ∈ {1, 6, 12}, with five specifications per panel:
  (1) rx_{t-ℓ+1:t}                    — cumulative excess return alone
  (2) (f_t − s_t)                      — forward discount alone
  (3) Δs_{t-ℓ+1:t}                     — cumulative spot change alone (sign-flipped)
  (4) rx_{t-ℓ+1:t} + (f_t − s_t)       — momentum + carry
  (5) (f_t − s_t) + Δs_{t-ℓ+1:t}       — carry + spot change

Each spec is run for both dependents:
  Left  half: rx_{t+1}      (excess return)
  Right half: Δs_{t+1}      (spot rate change, sign-flipped per §2.2)

Fama-MacBeth (1973) procedure:
  1. At each formation t, run cross-sectional OLS across N_t currencies.
  2. Save the slope vector b_t and R²_t.
  3. Time-series mean of b_t and R²_t.
  4. Standard t-stat: plain OLS t on the time series of b_t (no HAC).
  5. NW-Andrews HAC t-stat: L = ceil(0.75 · T^(1/3)).

Cumulative regressors use the no-skip convention (skip_month_flag = False from §3).
Δs is the sign-flipped spot_change column from §2.2 throughout.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"

from src.output_writer import add_sheet
from src.stats         import nw_lag, sig_stars

ELL_GRID     = [1, 6, 12]
PANEL_LABEL  = {1: "Panel A — One month (ℓ = 1)",
                6: "Panel B — Six months (ℓ = 6)",
                12: "Panel C — Twelve months (ℓ = 12)"}

# Specification id, list of regressor columns (constant always included).
SPECS = [
    ("(1)", ["cum_rx"]),
    ("(2)", ["fd"]),
    ("(3)", ["cum_ds"]),
    ("(4)", ["cum_rx", "fd"]),
    ("(5)", ["fd", "cum_ds"]),
]

# (display label, LHS column in fmb panel)
DEPS = [
    ("Excess Returns",   "lhs_rx"),
    ("Spot Rate Changes", "lhs_ds"),
]

# Display order across coefficient columns
COEF_ORDER  = ["const", "cum_rx", "fd", "cum_ds"]
COEF_LABEL  = {"const": "Const", "cum_rx": "rx", "fd": "f-s", "cum_ds": "Δs"}

MIN_OBS = 5   # min currencies per cross-section


# ── Panel construction ────────────────────────────────────────────────────────

def _cumulative(returns_panel: pd.DataFrame, col: str, ell: int, out_name: str) -> pd.DataFrame:
    """
    ℓ-month cumulative sum of `col` ending at formation date t (no-skip convention).
    Period-merge ensures non-consecutive months never pair across data gaps.
    Returns (date, currency_code, out_name); NaN rows are dropped.
    """
    base = returns_panel[["date", "currency_code", col]].copy()
    base["ym"] = base["date"].dt.to_period("M")
    result = base.copy()
    for k in range(1, ell):
        src = base[["currency_code", "ym", col]].rename(columns={col: f"_l{k}"}).copy()
        src["ym"] = src["ym"] + k
        result = result.merge(src, on=["currency_code", "ym"], how="left")

    if ell == 1:
        result[out_name] = result[col]
    else:
        lag_cols = [f"_l{k}" for k in range(1, ell)]
        result[out_name] = result[[col] + lag_cols].sum(axis=1, min_count=ell)

    return (
        result[["date", "currency_code", out_name]]
        .dropna(subset=[out_name])
        .reset_index(drop=True)
    )


def _build_fmb_panel(returns_panel: pd.DataFrame, ell: int) -> pd.DataFrame:
    """
    Construct the FMB regression panel for one ℓ.

    Columns:
      date          : formation date t
      currency_code
      cum_rx        : ℓ-month cumulative excess return ending at t
      fd            : forward discount at t  (f_t − s_t)
      cum_ds        : ℓ-month cumulative spot_change ending at t (sign-flipped)
      lhs_rx        : excess_return at t+1
      lhs_ds        : spot_change   at t+1 (sign-flipped)
    """
    cum_rx = _cumulative(returns_panel, "excess_return", ell, "cum_rx")
    cum_ds = _cumulative(returns_panel, "spot_change",   ell, "cum_ds")
    fd = (
        returns_panel[["date", "currency_code", "forward_discount"]]
        .dropna(subset=["forward_discount"])
        .rename(columns={"forward_discount": "fd"})
        .reset_index(drop=True)
    )

    reg = cum_rx.merge(fd, on=["date", "currency_code"], how="outer") \
                .merge(cum_ds, on=["date", "currency_code"], how="outer")

    # LHS at t+1, period-merged back to formation date t
    lhs = returns_panel[["date", "currency_code", "excess_return", "spot_change"]].copy()
    lhs["ym"] = lhs["date"].dt.to_period("M") - 1   # bring t+1 → t
    lhs = lhs.rename(columns={"excess_return": "lhs_rx", "spot_change": "lhs_ds"}) \
             [["currency_code", "ym", "lhs_rx", "lhs_ds"]]

    reg["ym"] = reg["date"].dt.to_period("M")
    panel = reg.merge(lhs, on=["currency_code", "ym"], how="left").drop(columns=["ym"])
    return panel


# ── Fama-MacBeth core ─────────────────────────────────────────────────────────

def _fmb_run(
    panel: pd.DataFrame,
    lhs_col: str,
    regs: list[str],
) -> dict | None:
    """
    Cross-sectional OLS at each t, then time-series stats on the slope time series.

    Returns dict with keys:
      coef[c]   : time-series mean of coefficient c (including 'const')
      t_ols[c]  : plain OLS t-stat for H0: mean = 0
      p_ols[c]  : associated p-value
      t_nw[c]   : NW-Andrews HAC t-stat
      p_nw[c]   : associated p-value
      r2_mean   : time-series mean of cross-sectional R²
      r2_se     : time-series SE of cross-sectional R² (i.e. std/sqrt(T))
      T         : effective months
      nw_lag    : Andrews L used
    """
    df = panel.dropna(subset=[lhs_col] + regs).copy()
    if df.empty:
        return None

    min_per_xs = max(MIN_OBS, len(regs) + 2)  # need at least k+1 obs for OLS + 1 dof
    slope_rows = []
    r2_list    = []

    for date, grp in df.groupby("date"):
        if len(grp) < min_per_xs:
            continue
        Y = grp[lhs_col].values
        X = sm.add_constant(grp[regs].values, has_constant="add")
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        res = sm.OLS(Y, X).fit()
        row = {"date": date, "const": float(res.params[0])}
        for i, c in enumerate(regs):
            row[c] = float(res.params[i + 1])
        slope_rows.append(row)
        r2_list.append(float(res.rsquared))

    if not slope_rows:
        return None

    slopes = pd.DataFrame(slope_rows).set_index("date").sort_index()
    T      = len(slopes)
    L_nw   = nw_lag(T)

    coef, t_ols, p_ols, t_nw, p_nw = {}, {}, {}, {}, {}
    for c in ["const"] + regs:
        arr = slopes[c].values
        coef[c] = float(arr.mean())

        res_ols    = sm.OLS(arr, np.ones(T)).fit()
        t_ols[c]   = float(res_ols.tvalues[0])
        p_ols[c]   = float(res_ols.pvalues[0])

        res_nw     = sm.OLS(arr, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": L_nw})
        t_nw[c]    = float(res_nw.tvalues[0])
        p_nw[c]    = float(res_nw.pvalues[0])

    r2_arr = np.asarray(r2_list, dtype=float)
    r2_mean = float(r2_arr.mean())
    r2_se   = float(r2_arr.std(ddof=1) / np.sqrt(T)) if T > 1 else float("nan")

    return {
        "coef":    coef,
        "t_ols":   t_ols, "p_ols": p_ols,
        "t_nw":    t_nw,  "p_nw":  p_nw,
        "r2_mean": r2_mean,
        "r2_se":   r2_se,
        "T":       T,
        "nw_lag":  L_nw,
    }


# ── Formatting helpers ────────────────────────────────────────────────────────

def _cell_value(result: dict, c: str) -> str:
    """Coefficient display."""
    if result is None or c not in result["coef"]:
        return ""
    return f"{result['coef'][c]:.3f}"


def _cell_tstat(result: dict, c: str, kind: str) -> str:
    """
    Bracketed t-stat display.
      kind = 'std' → standard OLS, [.]
      kind = 'nw'  → NW-Andrews HAC, [.]
    """
    if result is None or c not in result["t_ols"]:
        return ""
    if kind == "std":
        t, p = result["t_ols"][c], result["p_ols"][c]
    else:
        t, p = result["t_nw"][c],  result["p_nw"][c]
    return f"[{t:+.2f}]{sig_stars(p)}"


def _cell_r2(result: dict, row_kind: str) -> str:
    if result is None:
        return ""
    if row_kind == "value":
        return f"{result['r2_mean']:.3f}"
    if row_kind == "std":
        return f"({result['r2_se']:.3f})"
    return ""  # nw row leaves R² blank (no HAC-SE for cross-sectional R²)


# ── CSV export ────────────────────────────────────────────────────────────────

def _export_table13(results: dict) -> None:
    """
    Wide CSV mirroring the screenshot: side-by-side dependents per panel.

    results indexed by (ell, spec_id, dep_label).
    Each (panel, spec) emits 3 rows: coefficient value, [std-t], [NW-A-t].
    """
    coef_cols = ["Const", "rx", "f-s", "Δs", "R²"]
    headers   = ["Panel/Spec/Row"] + [f"{coef_cols[i]} | rx"   for i in range(5)] \
                                   + [f"{coef_cols[i]} | Δs"   for i in range(5)]
    spacer    = {h: "" for h in headers}

    def coef_label(c: str) -> str:
        return {"const": "Const", "cum_rx": "rx", "fd": "f-s", "cum_ds": "Δs"}[c]

    def build_row(label, results_by_dep, kind):
        """One CSV row across both dependents for a given (spec, statistic kind)."""
        row = {"Panel/Spec/Row": label}
        for dep_label, key in (("rx", "Excess Returns"), ("Δs", "Spot Rate Changes")):
            res = results_by_dep.get(key)
            # Const, rx, f-s, Δs, R² columns (in display order)
            for display_c, c_key in [("Const", "const"), ("rx", "cum_rx"),
                                      ("f-s", "fd"),     ("Δs", "cum_ds"),
                                      ("R²",  None)]:
                col_id = f"{display_c} | {dep_label}"
                if display_c == "R²":
                    row[col_id] = _cell_r2(res, kind)
                else:
                    if kind == "value":
                        row[col_id] = _cell_value(res, c_key)
                    elif kind == "std":
                        row[col_id] = _cell_tstat(res, c_key, "std")
                    else:
                        row[col_id] = _cell_tstat(res, c_key, "nw")
        return row

    rows = []
    for ell in ELL_GRID:
        rows.append({**spacer, "Panel/Spec/Row": PANEL_LABEL[ell]})

        # column-label row inside each panel
        hdr_row = {"Panel/Spec/Row": "Spec"}
        for dep_label in ("rx", "Δs"):
            for cl in coef_cols:
                hdr_row[f"{cl} | {dep_label}"] = cl
        rows.append(hdr_row)

        for spec_id, regs in SPECS:
            results_by_dep = {
                dep_label: results.get((ell, spec_id, dep_label))
                for dep_label, _ in DEPS
            }
            rows.append(build_row(f"{spec_id} value",     results_by_dep, "value"))
            rows.append(build_row(f"{spec_id} [std-t]",   results_by_dep, "std"))
            rows.append(build_row(f"{spec_id} [NW-A-t]",  results_by_dep, "nw"))
        rows.append(spacer)

    add_sheet("table13_fama_macbeth", pd.DataFrame(rows, columns=headers))


# ── Terminal display ──────────────────────────────────────────────────────────

def _print_table13(results: dict) -> None:
    coef_w = 12
    col_w  = 5 * coef_w
    W      = 12 + 2 * col_w + 3
    print()
    print("=" * W)
    print("  §9.1 Table 13 — Fama-MacBeth Cross-Sectional Regressions")
    print("  Coefficients (3dp)  |  [Std-OLS t-stat]  |  <NW-Andrews HAC t-stat>")
    print("  ***p<.01 **p<.05 *p<.10  (stars use NW-A p-values on the NW row)")
    print("=" * W)

    for ell in ELL_GRID:
        print()
        print(f"  {PANEL_LABEL[ell]}")
        # Header
        sub_hdr = f"  {'Spec':<8}"
        for dep_label, _ in DEPS:
            half = f"Dep: {dep_label}"
            sub_hdr += "  |  " + f"{half:^{col_w}}"
        print(sub_hdr)
        col_hdr = f"  {'':<8}"
        for _ in DEPS:
            for c in ["Const", "rx", "f-s", "Δs", "R²"]:
                col_hdr += f"{c:>{coef_w}}"
            col_hdr += "     "
        print(col_hdr)
        print("-" * W)

        for spec_id, regs in SPECS:
            for kind, label in (("value", spec_id), ("std", ""), ("nw", "")):
                line = f"  {label:<8}"
                for dep_label, _ in DEPS:
                    res = results.get((ell, spec_id, dep_label))
                    for c_key in ["const", "cum_rx", "fd", "cum_ds"]:
                        if kind == "value":
                            cell = _cell_value(res, c_key)
                        elif kind == "std":
                            cell = _cell_tstat(res, c_key, "std")
                        else:
                            # NW with stars
                            base = _cell_tstat(res, c_key, "nw")
                            cell = base.replace("[", "<").replace("]", ">") if base else ""
                        line += f"{cell:>{coef_w}}"
                    # R²
                    line += f"{_cell_r2(res, kind):>{coef_w}}"
                    line += "     "
                print(line)
        print("=" * W)


# ── Master ────────────────────────────────────────────────────────────────────

def run_section9(returns_panel: pd.DataFrame) -> dict:
    """
    Run §9.1 Fama-MacBeth regressions for all (ℓ, spec, dependent) combinations.

    Returns
    -------
    {(ell, spec_id, dep_label): result_dict, ...}
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading: §9.1 Fama-MacBeth regressions (Table 13) ...")
    results: dict = {}
    for ell in ELL_GRID:
        fmb_panel = _build_fmb_panel(returns_panel, ell)
        for spec_id, regs in SPECS:
            for dep_label, lhs_col in DEPS:
                results[(ell, spec_id, dep_label)] = _fmb_run(fmb_panel, lhs_col, regs)

    _export_table13(results)
    _print_table13(results)
    return results
