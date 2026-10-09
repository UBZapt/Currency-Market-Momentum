#!/usr/bin/env python3
"""
src/returns.py — §2 excess returns, spot changes, forward discounts, RX index.
All arithmetic in log-price space (§0.1).

Transaction-cost design: exports quote primitives for Step 6, not final net
returns.  State (entering / staying / exiting) is only known after portfolio
membership is constructed, so the formulas live in Step 6.  Primitives:
    fwd_bid_t, fwd_ask_t          — log forward bid/ask at t
    spot_mid_t1, spot_bid_t1, spot_ask_t1 — log spot quotes at t+1
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
PANEL_PATH   = OUTPUT_DIR / "fx_panel_clean.csv"

_log = logging.getLogger("returns")


def load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        _log.error("Panel not found: %s — run data.py first", PANEL_PATH)
        sys.exit(1)
    panel = pd.read_csv(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"], dayfirst=True)
    panel.sort_values(["currency_code", "date"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    return panel


def compute_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Period-merge (not shift) so non-consecutive months across data gaps are
    never paired.  Returns sit at realization date t+1; forward_discount at t.
    spot_change is sign-flipped: positive = foreign-currency appreciation (§2.2).
    """
    df = panel.copy()
    df["ym"] = df["date"].dt.to_period("M")

    fwd_cols = ["log_forward_mid_1m", "log_forward_bid_1m", "log_forward_ask_1m"]
    fwd_lag  = df[["currency_code", "ym"] + fwd_cols].copy()
    fwd_lag  = fwd_lag.rename(columns={c: f"lag_{c}" for c in fwd_cols})
    fwd_lag["ym"] = fwd_lag["ym"] + 1

    spot_lag = df[["currency_code", "ym", "log_spot_mid"]].copy()
    spot_lag = spot_lag.rename(columns={"log_spot_mid": "lag_log_spot_mid"})
    spot_lag["ym"] = spot_lag["ym"] + 1

    df = df.merge(fwd_lag,  on=["currency_code", "ym"], how="left")
    df = df.merge(spot_lag, on=["currency_code", "ym"], how="left")

    # §2.1 gross excess return: f_t − s_{t+1}
    df["excess_return"]    = df["lag_log_forward_mid_1m"] - df["log_spot_mid"]

    # §2.2 sign-flipped spot change: −(s_{t+1} − s_t); restricted to fwd universe
    df["spot_change"]      = (
        -(df["log_spot_mid"] - df["lag_log_spot_mid"])
    ).where(df["log_forward_mid_1m"].notna())

    # carry signal at t
    df["forward_discount"] = df["log_forward_mid_1m"] - df["log_spot_mid"]

    # Step 6 primitives: forward quotes at t, spot quotes at t+1
    df["fwd_bid_t"]   = df["lag_log_forward_bid_1m"]
    df["fwd_ask_t"]   = df["lag_log_forward_ask_1m"]
    df["spot_mid_t1"] = df["log_spot_mid"]
    df["spot_bid_t1"] = df["log_spot_bid"]
    df["spot_ask_t1"] = df["log_spot_ask"]

    df.drop(columns=[c for c in df.columns if c.startswith("lag_")] + ["ym"],
            inplace=True)

    return df[[
        "date", "currency_code", "msci_class",
        "excess_return", "spot_change", "forward_discount",
        "fwd_bid_t", "fwd_ask_t",
        "spot_mid_t1", "spot_bid_t1", "spot_ask_t1",
    ]].copy()


def validate_returns(returns_panel: pd.DataFrame, panel: pd.DataFrame) -> None:
    """§0.5: no NaN/Inf excess_return where both forward and spot inputs exist."""
    raw = panel.copy()
    raw["ym"] = raw["date"].dt.to_period("M")

    fwd_avail  = raw[raw["log_forward_mid_1m"].notna()][["currency_code", "ym"]].copy()
    fwd_avail["ym"] = fwd_avail["ym"] + 1
    spot_avail = raw[raw["log_spot_mid"].notna()][["currency_code", "ym", "date"]].copy()

    check = (
        fwd_avail.merge(spot_avail, on=["currency_code", "ym"], how="inner")
                 .merge(returns_panel[["date", "currency_code", "excess_return"]],
                        on=["date", "currency_code"], how="left")
    )
    bad = check["excess_return"].isna() | np.isinf(check["excess_return"])
    if bad.any():
        _log.error("SANITY FAIL — %d NaN/Inf excess_return despite both inputs available",
                   int(bad.sum()))
        sys.exit(1)

    # spot_change must be NaN wherever forward coverage is absent
    bad_sc = returns_panel[
        returns_panel["forward_discount"].isna() & returns_panel["spot_change"].notna()
    ]
    if not bad_sc.empty:
        _log.error("SANITY FAIL — %d rows have spot_change without forward coverage",
                   len(bad_sc))
        sys.exit(1)

    # Magnitude gate: a 1M log forward discount above ~0.20 implies spot/forward
    # are mis-aligned (different real dates, wrong units, etc.). EM crisis episodes
    # (TRY May 2023, RUB 2022Q1) legitimately reach ~0.10; 0.20 leaves slack while
    # still catching the date-parse bug class (>0.30).
    FD_THRESHOLD = 0.20
    fd_abs_max = float(returns_panel["forward_discount"].abs().max())
    if fd_abs_max > FD_THRESHOLD:
        bad = returns_panel.loc[
            returns_panel["forward_discount"].abs() > FD_THRESHOLD,
            ["date", "currency_code", "forward_discount"],
        ].sort_values("forward_discount", key=abs, ascending=False)
        _log.error(
            "SANITY FAIL — |forward_discount| max = %.4f exceeds %.2f; "
            "%d offending rows. First few:\n%s",
            fd_abs_max, FD_THRESHOLD, len(bad), bad.head(10).to_string(index=False),
        )
        sys.exit(1)


def compute_rx_factor(returns_panel: pd.DataFrame) -> pd.DataFrame:
    """RX_t = equal-weighted mean excess return across currencies at month t (§2.4)."""
    return (
        returns_panel.dropna(subset=["excess_return"])
        .groupby("date")
        .agg(rx_factor=("excess_return", "mean"), n_currencies=("excess_return", "count"))
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )


def cross_check_decomposition(returns_panel: pd.DataFrame) -> None:
    """Verify rx_{t+1} = fd_t + spot_change_{t+1} up to floating-point tolerance."""
    df  = returns_panel.copy()
    df["ym"] = df["date"].dt.to_period("M")

    fd_lag = df[["currency_code", "ym", "forward_discount"]].copy()
    fd_lag = fd_lag.rename(columns={"forward_discount": "fd_lag"})
    fd_lag["ym"] = fd_lag["ym"] + 1

    df  = df.merge(fd_lag, on=["currency_code", "ym"], how="left").drop(columns=["ym"])
    sub = df.dropna(subset=["excess_return", "fd_lag", "spot_change"])
    if sub.empty:
        return

    max_resid = float(
        (sub["excess_return"] - (sub["fd_lag"] + sub["spot_change"])).abs().max()
    )
    if max_resid > 1e-8:
        _log.error("DECOMP CHECK FAIL — residual %.2e exceeds 1e-8 tolerance", max_resid)
        sys.exit(1)


def export_outputs(returns_panel: pd.DataFrame, rx_factor: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    returns_panel.to_csv(OUTPUT_DIR / "returns_panel.csv", index=False)
    rx_factor.to_csv(OUTPUT_DIR / "rx_factor.csv", index=False)
    print("Written: returns_panel.csv")
    print("Written: rx_factor.csv")


def compute_all_returns(panel: pd.DataFrame | None = None) -> dict:
    """
    Run §2 returns pipeline.  Returns {'returns_panel', 'rx_factor'}.
    panel: cleaned FX panel from data.py; if None, reads fx_panel_clean.csv.
    """
    print("Loading: §2 returns panel + RX factor ...")
    if panel is None:
        panel = load_panel()
    else:
        panel = panel.copy()
        if not pd.api.types.is_datetime64_any_dtype(panel["date"]):
            panel["date"] = pd.to_datetime(panel["date"], dayfirst=True)
        panel.sort_values(["currency_code", "date"], inplace=True)
        panel.reset_index(drop=True, inplace=True)

    returns_panel = compute_returns(panel)
    validate_returns(returns_panel, panel)
    cross_check_decomposition(returns_panel)
    rx_factor = compute_rx_factor(returns_panel)
    export_outputs(returns_panel, rx_factor)

    return {"returns_panel": returns_panel, "rx_factor": rx_factor}
