#!/usr/bin/env python3
"""
src/section10.py — §10 Factor Regressions (Table 15).

Panel A — Univariate:    MOM_t = α + β · F_t + ε
  F ∈ {MKTRF, SMB, HML, UMD, HML_FX, VOL_FX}
Panel B — Multivariate (Fama-French + Carhart):
  MOM_t = α + β1·MKTRF + β2·SMB + β3·HML + β4·UMD + ε

Strategies (columns): MOM(1,1), MOM(6,1), MOM(12,1) at the §6-determined h* = 1.

Factor sources:
  MKTRF, SMB, HML, UMD : Kenneth French Data Library (ff_factors_clean.csv)
  HML_FX               : carry trade long-short CT from §7.1
  VOL_FX               : monthly global FX volatility innovation factor (MSSS 2012a)

VOL_FX construction (monthly, no daily data):
  σ^FX_t  = (1/N_t) · Σ_i |Δs_{i,t}|       ← cross-section mean of monthly |log spot change|
  AR(1):  σ^FX_t = a + ρ · σ^FX_{t-1} + ε
  VOL^FX_t = ε_t                           ← innovation residual

Inference: NW HAC with Andrews lag L = ceil(0.75 · T^(1/3)).
α reported annualised (× 1200, log-return → annual percent).
"""

import math
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"

from src.output_writer import add_sheet
from src.stats         import nw_lag, sig_stars

STRATEGIES        = [(1, 1), (6, 1), (12, 1)]
UNIVARIATE_ORDER  = ["MKTRF", "SMB", "HML", "UMD", "HML_FX", "VOL_FX"]
MULTIVARIATE_COLS = ["MKTRF", "SMB", "HML", "UMD"]


# ── Factor loading and VOL_FX construction ────────────────────────────────────

def _load_ff_factors() -> pd.DataFrame:
    """Load Fama-French + Carhart monthly factors from output/ff_factors_clean.csv."""
    path = OUTPUT_DIR / "ff_factors_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run data.py first")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.columns = [c if c == "date" else c.lower() for c in df.columns]
    return df


def _build_vol_fx(returns_panel: pd.DataFrame) -> pd.Series:
    """
    Monthly VOL_FX innovation factor.

    σ_t = mean over currencies of |spot_change_t| (monthly log change, sign-flip
    immaterial under abs).  AR(1) residuals form VOL^FX_t.
    """
    sigma = (
        returns_panel.dropna(subset=["spot_change"])
        .assign(abs_ds=lambda d: d["spot_change"].abs())
        .groupby("date")["abs_ds"]
        .mean()
        .sort_index()
    )

    # AR(1): regress σ_t on σ_{t-1}
    y    = sigma.values[1:]
    xlag = sigma.values[:-1]
    X    = sm.add_constant(xlag, has_constant="add")
    fit = sm.OLS(y, X).fit()
    return pd.Series(fit.resid, index=sigma.index[1:], name="vol_fx")


# ── Regression core ───────────────────────────────────────────────────────────

def _factor_regression(y: pd.Series, X_df: pd.DataFrame) -> dict | None:
    """OLS with NW HAC (Andrews lag).  α reported annualised (× 1200)."""
    aligned = pd.concat([y.rename("y"), X_df], axis=1).dropna()
    if len(aligned) < 10:
        return None

    Y = aligned["y"].values
    X = sm.add_constant(aligned.drop(columns="y").values, has_constant="add")
    T = len(aligned)
    L = nw_lag(T)
    res = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})

    return {
        "alpha_monthly": float(res.params[0]),
        "alpha_ann":     float(res.params[0]) * 1200.0,
        "alpha_t":       float(res.tvalues[0]),
        "alpha_p":       float(res.pvalues[0]),
        "beta":          [float(p) for p in res.params[1:]],
        "beta_t":        [float(t) for t in res.tvalues[1:]],
        "beta_p":        [float(p) for p in res.pvalues[1:]],
        "r2":            float(res.rsquared),
        "T":             T,
        "nw_lag":        L,
    }


# ── Formatting helpers ────────────────────────────────────────────────────────

def _strat_label(f: int, h: int) -> str:
    return f"MOM({f},{h})"


def _val(x: float, dp: int = 3) -> str:
    return "" if (isinstance(x, float) and math.isnan(x)) else f"{x:.{dp}f}"


def _tstat(t: float, p: float) -> str:
    if math.isnan(t):
        return ""
    return f"[{t:+.2f}]{sig_stars(p)}"


# ── CSV export ────────────────────────────────────────────────────────────────

def _export_table15(panel_A: dict, panel_B: dict) -> None:
    """
    Wide CSV: rows = factors (with [t-stat] sub-row), columns = α/β/R² × 3 strategies.
    Panel B keeps α and R² on the MKTRF row only; β fills per factor row.
    """
    strat_labels = [_strat_label(f, h) for f, h in STRATEGIES]
    metric_cols  = []
    for sl in strat_labels:
        metric_cols += [f"α | {sl}", f"β | {sl}", f"R² | {sl}"]
    all_cols = ["Factor"] + metric_cols
    spacer   = {c: "" for c in all_cols}

    rows = []

    # ── Panel A: univariate ──
    rows.append({**spacer, "Factor": "Panel A — Univariate regressions"})
    rows.append({**spacer, "Factor": ""})

    for fac in UNIVARIATE_ORDER:
        val_row = {"Factor": fac}
        t_row   = {"Factor": ""}
        for sl in strat_labels:
            res = panel_A[sl][fac]
            if res is None:
                val_row[f"α | {sl}"] = val_row[f"β | {sl}"] = val_row[f"R² | {sl}"] = ""
                t_row[f"α | {sl}"]   = t_row[f"β | {sl}"]   = t_row[f"R² | {sl}"]   = ""
                continue
            val_row[f"α | {sl}"]  = round(res["alpha_ann"], 3)
            val_row[f"β | {sl}"]  = round(res["beta"][0], 3)
            val_row[f"R² | {sl}"] = round(res["r2"], 3)
            t_row[f"α | {sl}"]    = _tstat(res["alpha_t"], res["alpha_p"])
            t_row[f"β | {sl}"]    = _tstat(res["beta_t"][0], res["beta_p"][0])
            t_row[f"R² | {sl}"]   = f"(T={res['T']}, L={res['nw_lag']})"
        rows.append(val_row)
        rows.append(t_row)

    rows.append(spacer)

    # ── Panel B: multivariate (one regression per strategy) ──
    rows.append({**spacer, "Factor": "Panel B — Multivariate regressions (Fama-French + Carhart)"})
    rows.append({**spacer, "Factor": ""})

    for i, fac in enumerate(MULTIVARIATE_COLS):
        val_row = {"Factor": fac}
        t_row   = {"Factor": ""}
        for sl in strat_labels:
            res = panel_B[sl]
            if res is None:
                val_row[f"α | {sl}"] = val_row[f"β | {sl}"] = val_row[f"R² | {sl}"] = ""
                t_row[f"α | {sl}"]   = t_row[f"β | {sl}"]   = t_row[f"R² | {sl}"]   = ""
                continue
            if i == 0:
                val_row[f"α | {sl}"]  = round(res["alpha_ann"], 3)
                t_row[f"α | {sl}"]    = _tstat(res["alpha_t"], res["alpha_p"])
                val_row[f"R² | {sl}"] = round(res["r2"], 3)
                t_row[f"R² | {sl}"]   = f"(T={res['T']}, L={res['nw_lag']})"
            else:
                val_row[f"α | {sl}"]  = ""
                t_row[f"α | {sl}"]    = ""
                val_row[f"R² | {sl}"] = ""
                t_row[f"R² | {sl}"]   = ""
            val_row[f"β | {sl}"] = round(res["beta"][i], 3)
            t_row[f"β | {sl}"]   = _tstat(res["beta_t"][i], res["beta_p"][i])
        rows.append(val_row)
        rows.append(t_row)

    add_sheet("table15_factor_regressions", pd.DataFrame(rows, columns=all_cols))


# ── Terminal display ──────────────────────────────────────────────────────────

def _print_table15(panel_A: dict, panel_B: dict) -> None:
    cell_w = 10
    fac_w  = 10
    block  = 3 * cell_w + 2
    W      = fac_w + 3 * block

    strat_labels = [_strat_label(f, h) for f, h in STRATEGIES]

    print()
    print("=" * W)
    print("  §10 Table 15 — Factor Regressions")
    print("  α annualised × 1200 (%/yr from monthly log)  |  β monthly loading  |  R²")
    print("  [NW HAC Andrews t-stat]   ***p<.01 **p<.05 *p<.10")
    print("=" * W)

    def _hdr(title: str) -> None:
        print()
        print(f"  {title}")
        head1 = f"  {'':<{fac_w}}"
        for sl in strat_labels:
            head1 += f"{sl:^{block}}"
        print(head1)
        head2 = f"  {'Factor':<{fac_w}}"
        for _ in strat_labels:
            head2 += f"{'α':>{cell_w}}{'β':>{cell_w}}{'R²':>{cell_w}}  "
        print(head2)
        print("-" * W)

    # Panel A
    _hdr("Panel A — Univariate regressions")
    for fac in UNIVARIATE_ORDER:
        val_line = f"  {fac:<{fac_w}}"
        t_line   = f"  {'':<{fac_w}}"
        for sl in strat_labels:
            res = panel_A[sl][fac]
            if res is None:
                val_line += f"{'N/A':>{cell_w}}{'':>{cell_w}}{'':>{cell_w}}  "
                t_line   += f"{'':>{cell_w}}{'':>{cell_w}}{'':>{cell_w}}  "
                continue
            val_line += f"{res['alpha_ann']:>{cell_w}.2f}"
            val_line += f"{res['beta'][0]:>{cell_w}.3f}"
            val_line += f"{res['r2']:>{cell_w}.3f}  "
            t_line   += f"{_tstat(res['alpha_t'], res['alpha_p']):>{cell_w}}"
            t_line   += f"{_tstat(res['beta_t'][0], res['beta_p'][0]):>{cell_w}}"
            t_line   += f"{'':>{cell_w}}  "
        print(val_line)
        print(t_line)
    print("=" * W)

    # Panel B
    _hdr("Panel B — Multivariate regressions (Fama-French + Carhart 4-factor)")
    for i, fac in enumerate(MULTIVARIATE_COLS):
        val_line = f"  {fac:<{fac_w}}"
        t_line   = f"  {'':<{fac_w}}"
        for sl in strat_labels:
            res = panel_B[sl]
            if res is None:
                val_line += f"{'N/A':>{cell_w}}{'':>{cell_w}}{'':>{cell_w}}  "
                t_line   += f"{'':>{cell_w}}{'':>{cell_w}}{'':>{cell_w}}  "
                continue
            if i == 0:
                val_line += f"{res['alpha_ann']:>{cell_w}.2f}"
                t_line   += f"{_tstat(res['alpha_t'], res['alpha_p']):>{cell_w}}"
            else:
                val_line += f"{'':>{cell_w}}"
                t_line   += f"{'':>{cell_w}}"
            val_line += f"{res['beta'][i]:>{cell_w}.3f}"
            t_line   += f"{_tstat(res['beta_t'][i], res['beta_p'][i]):>{cell_w}}"
            if i == 0:
                val_line += f"{res['r2']:>{cell_w}.3f}  "
            else:
                val_line += f"{'':>{cell_w}}  "
            t_line += f"{'':>{cell_w}}  "
        print(val_line)
        print(t_line)
    print("=" * W)


# ── Master ────────────────────────────────────────────────────────────────────

def run_section10(
    returns_panel:   pd.DataFrame,
    factors:         pd.DataFrame | None,
    section5_result: dict,
    section7_result: dict,
) -> dict:
    """
    §10 Factor Regressions — Table 15.

    Parameters
    ----------
    returns_panel   : §2 returns panel (must contain spot_change for VOL_FX)
    factors         : ff_factors_clean DataFrame; if None, loaded from CSV
    section5_result : must contain 'grid_A' with (1,1)/(6,1)/(12,1) keys
    section7_result : must contain 'ct_series' (HML_FX = CT carry long-short)

    Returns
    -------
    {"panel_A": ..., "panel_B": ..., "vol_fx": pd.Series}
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and align factors ──
    if factors is None:
        factors = _load_ff_factors()
    else:
        factors = factors.copy()
        if not pd.api.types.is_datetime64_any_dtype(factors["date"]):
            factors["date"] = pd.to_datetime(factors["date"])
        factors.columns = [c if c == "date" else c.lower() for c in factors.columns]
    factors = factors.sort_values("date").set_index("date")

    print("Loading: §10 factor regressions (Table 15) ...")

    # ── Build VOL_FX (monthly innovations) ──
    vol_fx = _build_vol_fx(returns_panel)

    # ── HML_FX from §7.1 carry CT ──
    ct_df  = section7_result["ct_series"]
    hml_fx = ct_df.set_index("date")["mom_return"].sort_index().rename("hml_fx")

    factor_series = {
        "MKTRF":  factors["mktrf"],
        "SMB":    factors["smb"],
        "HML":    factors["hml"],
        "UMD":    factors["umd"],
        "HML_FX": hml_fx,
        "VOL_FX": vol_fx,
    }

    # ── MOM strategies ──
    grid_A = section5_result["grid_A"]
    mom_y  = {_strat_label(f, h): grid_A[(f, h)] for f, h in STRATEGIES}

    # ── Panel A: univariate ──
    panel_A: dict = {sl: {} for sl in mom_y}
    for sl, y in mom_y.items():
        for fac_name in UNIVARIATE_ORDER:
            X = pd.DataFrame({fac_name: factor_series[fac_name]})
            panel_A[sl][fac_name] = _factor_regression(y, X)

    # ── Panel B: multivariate 4-factor ──
    panel_B: dict = {}
    for sl, y in mom_y.items():
        X = pd.DataFrame({fac: factor_series[fac] for fac in MULTIVARIATE_COLS})
        panel_B[sl] = _factor_regression(y, X)

    _export_table15(panel_A, panel_B)
    _print_table15(panel_A, panel_B)

    return {"panel_A": panel_A, "panel_B": panel_B, "vol_fx": vol_fx}
