#!/usr/bin/env python3
"""
src/validate_section4.py — §4 machinery validation (V1–V13).

Run standalone:  python src/validate_section4.py
Or call:         run_section4_validation(returns_panel)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
RETURNS_PATH = OUTPUT_DIR / "returns_panel.csv"

from src.signals    import build_signal
from src.portfolios import (
    assign_portfolios,
    compute_cohort_returns,
    compute_mom_series,
)


def _get_log() -> logging.Logger:
    log = logging.getLogger("validate_s4")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(levelname)-7s  %(message)s"))
        log.addHandler(h)
        log.propagate = False
    log.setLevel(logging.INFO)
    return log


def _check(log, label, cond, detail=""):
    label  = label.encode("ascii", "replace").decode("ascii")
    detail = detail.encode("ascii", "replace").decode("ascii")
    if cond:
        log.info("PASS  %s  %s", label, detail)
    else:
        log.error("FAIL  %s  %s", label, detail)
    return cond


def _build_lag_check(
    returns_panel: pd.DataFrame,
    col: str,
    f: int,
) -> pd.DataFrame:
    """
    Build a DataFrame with columns [date, currency_code, _l0, _l1, ..., _l{f-1}]
    where _l{k} = col at (date - k months).
    Used by V3 without coupling to the private _build_lag_panel helper.
    """
    base = returns_panel[["date", "currency_code", col]].copy()
    base["ym"] = base["date"].dt.to_period("M")
    result = base[["date", "currency_code", "ym"]].copy()
    for lag in range(f):
        src = base[["currency_code", "ym", col]].copy()
        src["ym"] = src["ym"] + lag
        result = result.merge(
            src.rename(columns={col: f"_l{lag}"}),
            on=["currency_code", "ym"],
            how="left",
        )
    return result.drop(columns=["ym"])


# ─────────────────────────── Signal validators ────────────────────────────────

def validate_signals(
    signal_df: pd.DataFrame,
    returns_panel: pd.DataFrame,
    f: int,
    label: str = "",
    col: str = "excess_return",
) -> bool:
    log = _get_log()
    ok  = True

    # V1 — required columns present and non-empty
    has_cols = {"date", "currency_code", "signal"}.issubset(signal_df.columns)
    ok &= _check(log, f"V1 {label} signal columns", has_cols)
    if not has_cols:
        return False
    ok &= _check(log, f"V1 {label} signal non-empty", len(signal_df) > 0)

    # V2 — no signal before required lookback.
    # No-skip: first valid t = first_rx_date + (f-1) months.
    first_ret = returns_panel.dropna(subset=["excess_return"])["date"].min()
    expected_first_ym = first_ret.to_period("M") + (f - 1)
    actual_first_ym   = signal_df["date"].min().to_period("M")
    diff_months = (actual_first_ym - expected_first_ym).n
    ok &= _check(
        log, f"V2 {label} no early signal",
        diff_months >= 0,
        f"(expected first >= {expected_first_ym}, got {actual_first_ym}, diff={diff_months}m)",
    )

    # V3 — exhaustive eligibility: every signal row must have all f required inputs non-null.
    try:
        lag_check = _build_lag_check(returns_panel, col, f)
        merged = signal_df[["date", "currency_code"]].merge(
            lag_check, on=["date", "currency_code"], how="left"
        )
        lag_cols = [f"_l{i}" for i in range(f)]
        bad_rows = merged[lag_cols].isna().any(axis=1).sum()
        ok &= _check(
            log, f"V3 {label} lag eligibility (all rows)",
            bad_rows == 0,
            f"({bad_rows} rows missing required inputs out of {len(signal_df)})",
        )
    except Exception as e:
        ok &= _check(log, f"V3 {label} lag eligibility", False, str(e))

    return ok


# ──────────────────────── Assignment validators ───────────────────────────────

def validate_assignments(
    assignments: pd.DataFrame,
    signal_df: pd.DataFrame,
    label: str = "",
) -> bool:
    log = _get_log()
    ok  = True

    # V4 — N^signal >= 15 for all assigned months
    n_per_date = assignments.groupby("date")["currency_code"].count()
    ok &= _check(
        log, f"V4 {label} N^signal >= 15",
        (n_per_date >= 15).all(),
        f"(min={n_per_date.min()}, max={n_per_date.max()})",
    )

    # V5 — k in {5, 6}
    k_vals = assignments["n_portfolios"].unique()
    ok &= _check(log, f"V5 {label} k in {{5,6}}", set(k_vals).issubset({5, 6}), str(k_vals))

    # V5b — portfolio labels 1..k complete each month
    def labels_ok(grp):
        k = int(grp["n_portfolios"].iloc[0])
        return set(grp["portfolio"].unique()) == set(range(1, k + 1))
    ok &= _check(
        log, f"V5b {label} portfolio labels complete",
        assignments.groupby("date").apply(labels_ok).all(),
    )

    # V6 — bucket sizes balanced (max − min <= 1)
    bucket_sizes = assignments.groupby(["date", "portfolio"])["currency_code"].count()
    size_range   = bucket_sizes.groupby("date").agg(lambda x: x.max() - x.min())
    ok &= _check(
        log, f"V6 {label} bucket size balance (max-min<=1)",
        (size_range <= 1).all(),
        f"(max imbalance={size_range.max()})",
    )

    # V8 — months with N < 15 produce no assignments
    sig_counts  = signal_df.dropna(subset=["signal"]).groupby("date")["currency_code"].count()
    low_n_dates = sig_counts[sig_counts < 15].index
    if len(low_n_dates) > 0:
        leaked = assignments[assignments["date"].isin(low_n_dates)]
        ok &= _check(
            log, f"V8 {label} N<15 months dropped",
            leaked.empty,
            f"({len(low_n_dates)} low-N dates, {len(leaked)} leaked rows)",
        )

    # V8b — remainder allocation: extras go to the highest portfolios only
    def extras_to_highest(grp):
        k     = int(grp["n_portfolios"].iloc[0])
        sizes = grp.groupby("portfolio")["currency_code"].count()
        top   = sizes.get(k, 0)
        return all(top >= sizes.get(p, 0) for p in range(1, k))
    extras_ok = assignments.groupby("date").apply(extras_to_highest)
    ok &= _check(
        log, f"V8b {label} extras go to highest portfolios",
        extras_ok.all(),
        f"({(~extras_ok).sum()} months violating remainder rule)",
    )

    return ok


# ──────────────────────── Cohort return validators ────────────────────────────

def validate_cohort_returns(
    cohort_rets: pd.DataFrame,
    assignments: pd.DataFrame,
    returns_panel: pd.DataFrame,
    h: int,
    label: str = "",
) -> bool:
    log = _get_log()
    ok  = True

    # V9 — fully realizable cohorts have exactly h realize_dates each
    last_panel_date = returns_panel["date"].max()
    fully_realizable = (
        cohort_rets["formation_date"] + pd.DateOffset(months=h) <= last_panel_date
    )
    cr_full = cohort_rets[fully_realizable & cohort_rets["realize_date"].notna()]
    hold_counts = cr_full.groupby(["formation_date", "portfolio"])["realize_date"].nunique()
    ok &= _check(
        log, f"V9 {label} cohort holding depth == h={h} (fully realizable)",
        (hold_counts == h).all() if not hold_counts.empty else True,
        f"(min={hold_counts.min() if not hold_counts.empty else 'N/A'}, "
        f"max={hold_counts.max() if not hold_counts.empty else 'N/A'})",
    )

    # V9b — every (formation_date, portfolio) in cohort_rets has a matching entry in assignments
    cohort_keys  = set(zip(cohort_rets["formation_date"], cohort_rets["portfolio"]))
    asgn_keys    = set(zip(assignments["date"],           assignments["portfolio"]))
    missing_keys = cohort_keys - asgn_keys
    ok &= _check(
        log, f"V9b {label} all cohorts traceable to assignments",
        len(missing_keys) == 0,
        f"({len(missing_keys)} cohort keys missing from assignments)",
    )

    # V13 — equal-weight cross-check: 10 sampled cohort-date pairs
    sample_rows = cohort_rets.dropna(subset=["port_return"]).sample(
        min(10, len(cohort_rets.dropna(subset=["port_return"]))), random_state=7
    )
    mismatches = 0
    for _, row in sample_rows.iterrows():
        fd, rd, p = row["formation_date"], row["realize_date"], int(row["portfolio"])
        assigned_currencies = assignments[
            (assignments["date"] == fd) & (assignments["portfolio"] == p)
        ]["currency_code"].tolist()
        manual_mean = (
            returns_panel[
                (returns_panel["date"] == rd) &
                (returns_panel["currency_code"].isin(assigned_currencies))
            ]["excess_return"].mean()
        )
        stored = float(row["port_return"])
        if not np.isnan(manual_mean) and abs(stored - manual_mean) > 1e-10:
            mismatches += 1
    ok &= _check(
        log, f"V13 {label} equal-weight cross-check (10 sampled)",
        mismatches == 0,
        f"({mismatches} mismatches)",
    )

    return ok


# ──────────────────────── MOM series validators ───────────────────────────────

def validate_mom_series(
    mom_series: pd.DataFrame,
    cohort_rets: pd.DataFrame,
    h: int,
    label: str = "",
) -> bool:
    log = _get_log()
    ok  = True

    # V10 — burn-in months have NaN mom_return; post-burn-in have n_active == h
    post_burnin = mom_series.dropna(subset=["mom_return"])
    ok &= _check(
        log, f"V10 {label} burn-in months have NaN mom_return",
        mom_series[mom_series["n_active_cohorts"] < h]["mom_return"].isna().all(),
    )
    ok &= _check(
        log, f"V10b {label} n_active == h post burn-in",
        (post_burnin["n_active_cohorts"] == h).all() if not post_burnin.empty else True,
        f"({len(post_burnin)} post-burn-in rows)",
    )

    # V11 — first non-null MOM date >= first_formation + h months
    if not post_burnin.empty:
        first_full   = post_burnin["date"].min()
        first_form   = cohort_rets["formation_date"].min()
        expected_min = first_form + pd.DateOffset(months=h)
        ok &= _check(
            log, f"V11 {label} burn-in timing",
            first_full.to_period("M") >= expected_min.to_period("M"),
            f"(first_full={first_full.date()}, expected>={expected_min.date()})",
        )

    # V12 — MOM formula consistency: exhaustive check across all post-burn-in dates
    mismatches = 0
    for _, mom_row in post_burnin.iterrows():
        t       = mom_row["date"]
        row_mom = float(mom_row["mom_return"])
        ym_t    = t.to_period("M")
        active  = cohort_rets[
            (cohort_rets["realize_date"] == t) &
            (cohort_rets["formation_date"].dt.to_period("M") >= ym_t - h) &
            (cohort_rets["formation_date"].dt.to_period("M") <= ym_t - 1)
        ]
        spreads = []
        for _, coh in active.groupby("formation_date"):
            k_val = int(coh["n_portfolios"].iloc[0])
            hi    = coh[coh["portfolio"] == k_val]["port_return"]
            lo    = coh[coh["portfolio"] == 1]["port_return"]
            if hi.empty or lo.empty:
                continue
            h_val, l_val = float(hi.iloc[0]), float(lo.iloc[0])
            if not (np.isnan(h_val) or np.isnan(l_val)):
                spreads.append(h_val - l_val)
        manual = float(np.mean(spreads)) if spreads else float("nan")
        if not np.isnan(manual) and abs(row_mom - manual) > 1e-10:
            mismatches += 1
    ok &= _check(
        log, f"V12 {label} MOM formula consistency (all post-burn-in)",
        mismatches == 0,
        f"({mismatches} mismatches out of {len(post_burnin)} dates)",
    )

    return ok


# ─────────────────────────── Main entry point ─────────────────────────────────

def run_section4_validation(
    returns_panel: pd.DataFrame | None = None,
    f_test: int = 6,
    h_test: int = 1,
) -> bool:
    """
    Full §4 validation over one (f_test, h_test) pair for Signal A and Signal B.
    Also smoke-tests Signal OLS (construction only).
    Returns True iff all checks pass.
    """
    log = _get_log()

    if returns_panel is None:
        if not RETURNS_PATH.exists():
            raise FileNotFoundError(f"returns_panel not found: {RETURNS_PATH}")
        returns_panel = pd.read_csv(RETURNS_PATH)
        returns_panel["date"] = pd.to_datetime(returns_panel["date"], dayfirst=True)

    all_ok = True

    for sig_type, sig_label, col_hint in [
        ("A", "Signal-A", "excess_return"),
        ("B", "Signal-B", "spot_change"),
    ]:
        log.info("--- %s ---", sig_label)
        try:
            signal_df = build_signal(returns_panel, f_test, sig_type)
        except Exception as e:
            log.error("FAIL  signal construction for %s: %s", sig_label, e)
            all_ok = False
            continue

        all_ok &= validate_signals(signal_df, returns_panel, f_test, label=sig_label, col=col_hint)

        try:
            assignments = assign_portfolios(signal_df)
        except Exception as e:
            log.error("FAIL  assign_portfolios for %s: %s", sig_label, e)
            all_ok = False
            continue

        all_ok &= validate_assignments(assignments, signal_df, label=sig_label)

        try:
            cohort_rets = compute_cohort_returns(assignments, returns_panel, h_test)
        except Exception as e:
            log.error("FAIL  compute_cohort_returns for %s: %s", sig_label, e)
            all_ok = False
            continue

        all_ok &= validate_cohort_returns(cohort_rets, assignments, returns_panel, h_test, label=sig_label)

        try:
            mom_series = compute_mom_series(cohort_rets, h_test)
        except Exception as e:
            log.error("FAIL  compute_mom_series for %s: %s", sig_label, e)
            all_ok = False
            continue

        all_ok &= validate_mom_series(mom_series, cohort_rets, h_test, label=sig_label)

    # OLS: construction smoke test only (full grid deferred to §5)
    log.info("--- Signal-OLS (construction) ---")
    try:
        ols_sig   = build_signal(returns_panel, f_test, "OLS")
        first_d   = ols_sig["date"].min()
        first_ret = returns_panel.dropna(subset=["excess_return"])["date"].min()
        expected  = first_ret + pd.DateOffset(months=36)
        all_ok &= _check(log, "V-OLS non-empty", len(ols_sig) > 0)
        all_ok &= _check(
            log, "V-OLS first date >= initial_window",
            first_d.to_period("M") >= expected.to_period("M"),
            f"(first={first_d.date()}, expected>={expected.date()})",
        )
    except Exception as e:
        log.error("FAIL  Signal-OLS construction: %s", e)
        all_ok = False

    # f=1, h=1 edge case (single-lag path)
    if f_test != 1:
        log.info("--- Signal-A f=1 h=1 (edge case) ---")
        try:
            sig1  = build_signal(returns_panel, 1, "A")
            asgn1 = assign_portfolios(sig1)
            coh1  = compute_cohort_returns(asgn1, returns_panel, 1)
            mom1  = compute_mom_series(coh1, 1)
            all_ok &= validate_signals(sig1, returns_panel, 1, label="A-f1")
            all_ok &= validate_assignments(asgn1, sig1, label="A-f1")
            all_ok &= validate_cohort_returns(coh1, asgn1, returns_panel, 1, label="A-f1")
            all_ok &= validate_mom_series(mom1, coh1, 1, label="A-f1")
        except Exception as e:
            log.error("FAIL  f=1 edge case: %s", e)
            all_ok = False

    log.info("Section 4 validation: %s", "ALL PASS" if all_ok else "FAILURES DETECTED")
    return all_ok


if __name__ == "__main__":
    if not RETURNS_PATH.exists():
        raise FileNotFoundError(f"returns_panel not found: {RETURNS_PATH}")
    rp = pd.read_csv(RETURNS_PATH)
    rp["date"] = pd.to_datetime(rp["date"], dayfirst=True)
    ok = run_section4_validation(returns_panel=rp, f_test=6, h_test=1)
    sys.exit(0 if ok else 1)
