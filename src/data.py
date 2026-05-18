#!/usr/bin/env python3
"""
src/data.py

Data cleaning and preparation for the Currency Momentum project.
Implements the data pipeline (§1).

Run standalone:  python src/data.py
Import in MAIN:  from src.data import clean_data; result = clean_data()

Inputs  (project root):
    FX_SPOT_Rates_USD.csv       — LSEG Datastream daily spot rates, USD-base, ~1.1M rows
    FX_1MFD_Rates_USD.csv       — 1-month forward rates, USD-base, ~173k rows
    FX_1MFD_Rates_AUD.csv       — 1-month forward rates, AUD-base (USD per 1 AUD), ~6.5k rows
    FX_1MFD_Rates_EUR.csv       — 1-month forward rates, EUR-base (USD per 1 EUR), ~6.5k rows
    FX_1MFD_Rates_GBP.csv       — 1-month forward rates, GBP-base (USD per 1 GBP), ~6.5k rows
    FX_1MFD_Rates_NZD.csv       — 1-month forward rates, NZD-base (USD per 1 NZD), ~6.5k rows
    FX_metadata_USD.csv         — instrument code mappings for all USD-base contracts
    FX_metadata_AUD.csv         — instrument code mappings for AUD-base contracts
    FX_metadata_EUR.csv         — instrument code mappings for EUR-base contracts
    FX_metadata_GBP.csv         — instrument code mappings for GBP-base contracts
    FX_metadata_NZD.csv         — instrument code mappings for NZD-base contracts
    ff_factors.csv              — Kenneth French / Carhart four-factor monthly data

Outputs (output/):
    fx_panel_clean.csv    — monthly end-of-month panel (date x currency)
    ff_factors_clean.csv  — cleaned Fama-French + Carhart factors
    excluded_pegged.csv   — rows removed for HKD / SAR / AED
    schema_mappings.json  — raw-to-standard column maps for all inputs
"""

import re
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"

INPUT_PATHS = {
    "fx_spot_usd" : PROJECT_ROOT / "FX_SPOT_Rates_USD.csv",
    "fx_fwd_usd"  : PROJECT_ROOT / "FX_1MFD_Rates_USD.csv",
    "fx_fwd_aud"  : PROJECT_ROOT / "FX_1MFD_Rates_AUD.csv",
    "fx_fwd_eur"  : PROJECT_ROOT / "FX_1MFD_Rates_EUR.csv",
    "fx_fwd_gbp"  : PROJECT_ROOT / "FX_1MFD_Rates_GBP.csv",
    "fx_fwd_nzd"  : PROJECT_ROOT / "FX_1MFD_Rates_NZD.csv",
    "meta_usd"    : PROJECT_ROOT / "FX_metadata_USD.csv",
    "meta_aud"    : PROJECT_ROOT / "FX_metadata_AUD.csv",
    "meta_eur"    : PROJECT_ROOT / "FX_metadata_EUR.csv",
    "meta_gbp"    : PROJECT_ROOT / "FX_metadata_GBP.csv",
    "meta_nzd"    : PROJECT_ROOT / "FX_metadata_NZD.csv",
    "ff_factors"  : PROJECT_ROOT / "ff_factors.csv",
}

# Non-USD forward currencies: rates are in CCY/USD (USD per 1 unit of foreign)
# and must be inverted to foreign/USD before joining the main panel.
NON_USD_FWD_CCYS = ["AUD", "EUR", "GBP", "NZD"]

SAMPLE_START = pd.Timestamp("2000-01-01")
SAMPLE_END   = pd.Timestamp("2024-12-31")

# Currencies with hard USD peg — excluded from main sample (methodology §1.1)
PEGGED_CURRENCIES = frozenset({"HKD", "SAR", "AED"})

# MSCI classification as of December 2024 (static, per methodology §0.2)
# Source: MSCI Market Classification Review (December 2024)
MSCI_DM = frozenset({
    # Americas  : Canada
    "CAD",
    # EMEA      : Denmark, Israel, Norway, Sweden, Switzerland, UK, and euro-area
    "DKK", "EUR", "GBP", "ILS", "NOK", "SEK", "CHF",
    # Asia-Pac  : Australia, Hong Kong (pegged but classified DM), Japan, NZ, Singapore
    "AUD", "HKD", "JPY", "NZD", "SGD",
})

MSCI_EM = frozenset({
    # Americas
    "BRL", "CLP", "COP", "MXN", "PEN",
    # EMEA
    "CZK", "EGP", "HUF", "KWD", "PLN", "QAR", "SAR", "ZAR", "TRY", "AED",
    # Asia-Pac
    "CNY", "CNH", "IDR", "INR", "KRW", "MYR", "PHP", "THB", "TWD",
})

# Rows per chunk when scanning large FX rate files
CHUNK_SIZE = 300_000

# Reference ranges for quote-orientation validation (foreign/USD convention)
# (currency_iso, min_plausible, max_plausible) — approximate over 2000-2024
_ORIENTATION_REFS = [
    ("JPY", 70.0,  165.0),
    ("AUD",  0.80,   2.20),
    ("GBP",  0.40,   0.90),
    ("EUR",  0.75,   1.35),
    ("CHF",  0.80,   1.90),
]


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.StreamHandler(sys.stdout)])
    return logging.getLogger("data")


# ──────────────────────────────────────────────────────────────────────────────
# Gate 1 — File availability
# ──────────────────────────────────────────────────────────────────────────────

def resolve_inputs(paths: dict, log: logging.Logger) -> None:
    missing = [name for name, p in paths.items() if not Path(p).exists()]
    if missing:
        log.error("Required files not found: %s", missing)
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Gate 2 — Schema inspection and column mapping
# ──────────────────────────────────────────────────────────────────────────────

def _short_name(raw: str) -> str:
    """Extract short identifier from Datastream's '(shortname) LongDesc' header."""
    m = re.match(r"\((\w+)\)", raw.strip())
    return m.group(1) if m else raw.strip().lower().replace(" ", "_").replace("-", "_")


def inspect_schemas(paths: dict, log: logging.Logger) -> dict:
    """Read one header row per input; return {file_key: {raw_columns, column_mapping}}."""
    schemas = {}
    for name, path in paths.items():
        sample  = pd.read_csv(path, nrows=1)
        raw     = list(sample.columns)
        col_map = {rc: _short_name(rc) for rc in raw}
        schemas[name] = {"raw_columns": raw, "column_mapping": col_map}
    return schemas


# ──────────────────────────────────────────────────────────────────────────────
# Gates 3 & 4 — Join-key and instrument identification
# ──────────────────────────────────────────────────────────────────────────────

def load_metadata(path: Path, col_map: dict, log: logging.Logger) -> pd.DataFrame:
    meta = pd.read_csv(path)
    meta.rename(columns=col_map, inplace=True)
    meta.columns = [_short_name(c) for c in meta.columns]
    return meta


def map_instruments(meta: pd.DataFrame,
                    log: logging.Logger) -> tuple[dict, dict]:
    """
    Return spot_map and fwd_usd_map ({currency_iso: exrateintcode}).
    When duplicates exist, keep the lowest exrateintcode (primary series).
    """
    usd = meta[meta["tocurrcode"] == "USD"].copy()

    def _dedupe(df: pd.DataFrame) -> dict:
        return (
            df.sort_values("exrateintcode")
              .drop_duplicates(subset="fromcurrcode", keep="first")
              .set_index("fromcurrcode")["exrateintcode"]
              .to_dict()
        )

    spot_map    = _dedupe(usd[usd["ratetypecode"] == "SPOT"])
    fwd_usd_map = _dedupe(usd[usd["ratetypecode"] == "1MFD"])

    for d in (spot_map, fwd_usd_map):
        d.pop("USD", None)

    if not spot_map:
        log.error("No SPOT instruments found in metadata")
        sys.exit(1)

    return spot_map, fwd_usd_map


# ──────────────────────────────────────────────────────────────────────────────
# Memory-efficient FX rate loading
# ──────────────────────────────────────────────────────────────────────────────

def load_fx_chunked(path: Path, col_map: dict, relevant_codes: set,
                    date_format: str, log: logging.Logger) -> pd.DataFrame:
    """
    Stream-load a large FX rate CSV and keep rows whose exrateintcode is in `relevant_codes`.
    date_format is the explicit pandas format string for exratedate (e.g. 'ISO8601',
    '%d/%m/%Y'). Each input file has a deterministic date format — never pass
    dayfirst=True, because that mis-parses ISO 'YYYY-MM-DD' as '%Y-%d-%m' and silently
    swaps month/day or drops rows.
    """
    codes = {int(c) for c in relevant_codes}

    chunks        = []
    total_scanned = 0
    total_kept    = 0

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        total_scanned += len(chunk)
        chunk.rename(columns=col_map, inplace=True)
        chunk.columns = [_short_name(c) for c in chunk.columns]

        chunk["exrateintcode"] = pd.to_numeric(chunk["exrateintcode"], errors="coerce")
        filtered = chunk[chunk["exrateintcode"].isin(codes)].copy()

        if not filtered.empty:
            filtered["exratedate"] = pd.to_datetime(
                filtered["exratedate"], format=date_format, errors="coerce"
            )
            total_kept += len(filtered)
            chunks.append(filtered)

    if not chunks:
        log.error(
            "No rows matched any instrument code in %s — check exrateintcode key space",
            path.name,
        )
        sys.exit(1)

    return pd.concat(chunks, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Non-USD forward loading and inversion
# ──────────────────────────────────────────────────────────────────────────────

def load_and_invert_non_usd_fwd(
    rates_path: Path,
    meta_path: Path,
    target_ccy: str,
    log: logging.Logger,
) -> tuple[pd.DataFrame, int]:
    """
    Load a non-USD-base 1MFD file and invert rates from CCY/USD convention
    (USD per 1 unit of foreign) to foreign/USD (units of foreign per 1 USD).

    Inversion uses the triangular no-arbitrage condition:
        new_mid = 1 / old_mid
        new_bid = 1 / old_ask   (lower bound when quoted as foreign/USD)
        new_ask = 1 / old_bid   (upper bound when quoted as foreign/USD)
    Bid/ask are swapped on inversion to preserve the effective spread.

    Returns (df, code) where df has columns [exrateintcode, exratedate, midrate,
    bidrate, offerrate] in foreign/USD convention, and code is the exrateintcode
    used (so the caller can build a code->currency mapping).
    """
    meta = pd.read_csv(meta_path)
    meta.columns = [_short_name(c) for c in meta.columns]

    # The relevant contract: fromcurrcode=USD, tocurrcode=target_ccy confirms
    # this is the USD-per-1-target_ccy series in the target_ccy-base file.
    mask = (
        (meta["fromcurrcode"] == "USD") &
        (meta["ratetypecode"] == "1MFD") &
        (meta["tocurrcode"] == target_ccy)
    )
    relevant = meta[mask].copy()

    if relevant.empty:
        log.error("No USD 1MFD entry in %s for %s", meta_path.name, target_ccy)
        sys.exit(1)

    # Lowest code = primary series if duplicates exist
    code = int(relevant.sort_values("exrateintcode")["exrateintcode"].iloc[0])

    df = pd.read_csv(rates_path, low_memory=False)
    df.columns = [_short_name(c) for c in df.columns]
    df["exrateintcode"] = pd.to_numeric(df["exrateintcode"], errors="coerce")
    df = df[df["exrateintcode"] == code].copy()

    if df.empty:
        log.error("No rows with code %d in %s", code, rates_path.name)
        sys.exit(1)

    # Non-USD-base 1MFD files are DMY-formatted ('03/01/2000'); never use dayfirst.
    df["exratedate"] = pd.to_datetime(
        df["exratedate"], format="%d/%m/%Y", errors="coerce"
    )

    # Invert: CCY/USD -> foreign/USD
    old_bid = df["bidrate"].copy()
    old_ask = df["offerrate"].copy()
    df["midrate"]   = 1.0 / df["midrate"]
    df["bidrate"]   = 1.0 / old_ask    # lower foreign/USD after inversion
    df["offerrate"] = 1.0 / old_bid    # higher foreign/USD after inversion

    return df[["exrateintcode", "exratedate", "midrate", "bidrate", "offerrate"]], code


# ──────────────────────────────────────────────────────────────────────────────
# Gate 5 — Quote orientation
# ──────────────────────────────────────────────────────────────────────────────

def check_quote_orientation(df: pd.DataFrame, spot_map: dict,
                             log: logging.Logger) -> None:
    """
    Validate that spot rates for reference currencies are consistent with
    the foreign-currency-per-USD convention (median within plausible range).
    """
    issues = []
    for ccy, lo, hi in _ORIENTATION_REFS:
        code = spot_map.get(ccy)
        if code is None:
            continue
        vals   = df.loc[df["exrateintcode"] == int(code), "midrate"].dropna()
        if vals.empty:
            continue
        median = float(vals.median())
        if not (lo <= median <= hi):
            issues.append(f"{ccy}: median={median:.4f}, expected [{lo}, {hi}]")

    if issues:
        log.error("GATE 5 FAIL — quote orientation anomaly: %s", issues)
        log.error("All rates must be in foreign/USD convention. Halting pipeline.")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Quote normalisation and mid derivation
# ──────────────────────────────────────────────────────────────────────────────

def normalize_quotes(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Rename 'offerrate' -> 'askrate' for clarity."""
    if "offerrate" in df.columns:
        df = df.rename(columns={"offerrate": "askrate"})
    return df


def derive_mid_quotes(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """
    Where 'midrate' is NaN but both 'bidrate' and 'askrate' are present,
    impute mid = (bid + ask) / 2.  A boolean 'mid_derived' column flags imputed rows.
    """
    df           = df.copy()
    missing      = df["midrate"].isna()
    both_present = df["bidrate"].notna() & df["askrate"].notna()
    mask         = missing & both_present

    df.loc[mask, "midrate"] = (df.loc[mask, "bidrate"] + df.loc[mask, "askrate"]) / 2.0
    df["mid_derived"] = mask
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Gate 6 — Month-end aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_to_month_end(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """
    Select the last observed trading day per (instrument, month).
    Output 'date' = last calendar day of month (aligns with FF factors).
    2-month pre-sample buffer retained for first-month return computation.

    Caller is responsible for ensuring `exratedate` is already datetime64. Each
    source file uses a deterministic format and is parsed by its loader; a single
    blanket `dayfirst=True` here mis-handles ISO dates by locking the column to
    '%Y-%d-%m', which swaps day/month for days 1-12 and drops days 13-31.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["exratedate"]):
        log.error("aggregate_to_month_end: exratedate not parsed by loader")
        sys.exit(1)
    df = df.dropna(subset=["exratedate"])

    buf_start = SAMPLE_START - pd.DateOffset(months=2)
    df = df[(df["exratedate"] >= buf_start) & (df["exratedate"] <= SAMPLE_END)]

    df["_ym"] = df["exratedate"].dt.to_period("M")
    df.sort_values(["exrateintcode", "exratedate"], inplace=True)

    # idxmax on exratedate gives the true last trading day per group — all
    # columns then come from that single row, preventing column mixing.
    last_idx = df.groupby(["exrateintcode", "_ym"])["exratedate"].idxmax()
    eom = df.loc[last_idx].reset_index(drop=True)

    eom["date"] = eom["_ym"].dt.to_timestamp("M")
    eom.drop(columns=["_ym"], inplace=True)

    return eom


# ──────────────────────────────────────────────────────────────────────────────
# Long-format instrument panel
# ──────────────────────────────────────────────────────────────────────────────

def build_instrument_long(eom: pd.DataFrame, spot_map: dict,
                          fwd_map: dict, log: logging.Logger) -> pd.DataFrame:
    code_to_info: dict[int, tuple] = {}
    for ccy, code in spot_map.items():
        code_to_info[int(code)] = (ccy, "spot")
    for ccy, code in fwd_map.items():
        icode = int(code)
        if icode not in code_to_info:
            code_to_info[icode] = (ccy, "fwd1m")

    eom = eom.copy()
    eom["exrateintcode"] = eom["exrateintcode"].astype(int)
    mask = eom["exrateintcode"].isin(code_to_info)
    df   = eom[mask].copy()

    df["currency_code"] = df["exrateintcode"].map(lambda c: code_to_info[c][0])
    df["rate_type"]     = df["exrateintcode"].map(lambda c: code_to_info[c][1])

    keep = ["date", "currency_code", "rate_type", "midrate", "bidrate", "askrate"]
    df   = df[[c for c in keep if c in df.columns]].copy()

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Wide panel (pivot)
# ──────────────────────────────────────────────────────────────────────────────

def pivot_to_panel(long_df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Pivot long (date, currency_code, rate_type) to wide with spot and forward columns."""
    spot = long_df[long_df["rate_type"] == "spot"].copy()
    fwd  = long_df[long_df["rate_type"] == "fwd1m"].copy()

    spot = spot.rename(columns={
        "midrate": "spot_mid",
        "bidrate": "spot_bid",
        "askrate": "spot_ask",
    })
    fwd = fwd.rename(columns={
        "midrate": "forward_mid_1m",
        "bidrate": "forward_bid_1m",
        "askrate": "forward_ask_1m",
    })

    spot_keep = ["date", "currency_code"] + [
        c for c in ["spot_mid", "spot_bid", "spot_ask"] if c in spot.columns
    ]
    fwd_keep  = ["date", "currency_code"] + [
        c for c in ["forward_mid_1m", "forward_bid_1m", "forward_ask_1m"] if c in fwd.columns
    ]

    if fwd.empty:
        panel = spot[spot_keep].copy()
        for col in ["forward_mid_1m", "forward_bid_1m", "forward_ask_1m"]:
            panel[col] = np.nan
    else:
        panel = pd.merge(
            spot[spot_keep],
            fwd[fwd_keep],
            on=["date", "currency_code"],
            how="outer",
        )

    panel.sort_values(["date", "currency_code"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    return panel


# ──────────────────────────────────────────────────────────────────────────────
# Gate 7 — Final-key uniqueness
# ──────────────────────────────────────────────────────────────────────────────

def check_key_uniqueness(panel: pd.DataFrame, log: logging.Logger) -> None:
    dupes = panel[panel.duplicated(subset=["date", "currency_code"], keep=False)]
    if not dupes.empty:
        log.error(
            "GATE 7 FAIL — %d duplicate (date, currency_code) rows found:\n%s",
            len(dupes),
            dupes[["date", "currency_code"]].drop_duplicates().to_string(),
        )
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# MSCI classification
# ──────────────────────────────────────────────────────────────────────────────

def apply_msci_classification(panel: pd.DataFrame,
                               log: logging.Logger) -> pd.DataFrame:
    """
    Assign static MSCI classification (DM / EM / NaN) per currency.
    Source: MSCI Market Classification as of December 2024.
    """
    panel = panel.copy()
    panel["msci_class"] = panel["currency_code"].map(
        lambda c: "DM" if c in MSCI_DM else ("EM" if c in MSCI_EM else None)
    )
    return panel


# ──────────────────────────────────────────────────────────────────────────────
# Pegged currency exclusion
# ──────────────────────────────────────────────────────────────────────────────

def flag_and_exclude_pegs(panel: pd.DataFrame,
                          log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask     = panel["currency_code"].isin(PEGGED_CURRENCIES)
    excluded = panel[mask].copy()
    main     = panel[~mask].copy()
    return main, excluded


# ──────────────────────────────────────────────────────────────────────────────
# Forward coverage filter
# ──────────────────────────────────────────────────────────────────────────────

def filter_forward_coverage(panel: pd.DataFrame,
                            log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop currencies with no forward_mid_1m data in the sample — they cannot
    compute excess returns and must not appear in fx_panel_clean.csv.
    Returns (retained_panel, spot_only_panel).
    """
    has_fwd  = panel.groupby("currency_code")["forward_mid_1m"].apply(
        lambda s: s.notna().any()
    )
    keep_ccys = has_fwd[has_fwd].index
    drop_ccys = has_fwd[~has_fwd].index

    retained  = panel[panel["currency_code"].isin(keep_ccys)].copy()
    spot_only = panel[panel["currency_code"].isin(drop_ccys)].copy()
    return retained, spot_only


def build_coverage_audit(
    retained  : pd.DataFrame,
    spot_only : pd.DataFrame,
    excluded  : pd.DataFrame,
    log       : logging.Logger,
) -> pd.DataFrame:
    """Per-currency coverage table: status, n_spot_months, n_forward_months."""
    rows = []

    for ccy, grp in retained.groupby("currency_code"):
        rows.append({
            "currency_code":    ccy,
            "status":           "kept",
            "n_spot_months":    int(grp["spot_mid"].notna().sum()),
            "n_forward_months": int(grp["forward_mid_1m"].notna().sum()),
        })

    for ccy, grp in spot_only.groupby("currency_code"):
        rows.append({
            "currency_code":    ccy,
            "status":           "spot_only_dropped",
            "n_spot_months":    int(grp["spot_mid"].notna().sum()),
            "n_forward_months": 0,
        })

    for ccy, grp in excluded.groupby("currency_code"):
        rows.append({
            "currency_code":    ccy,
            "status":           "pegged_excluded",
            "n_spot_months":    int(grp["spot_mid"].notna().sum()),
            "n_forward_months": int(grp["forward_mid_1m"].notna().sum()),
        })

    return (
        pd.DataFrame(rows)
          .sort_values(["status", "currency_code"])
          .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Log-price fields
# ──────────────────────────────────────────────────────────────────────────────

def add_log_fields(panel: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """
    Add log-transformed versions of all level rate columns.
    Per methodology §0.1: all spot/forward arithmetic uses log prices.
    Non-positive values (hyperinflation era artifacts) are replaced with NaN.
    """
    rate_cols = [
        "spot_mid", "spot_bid", "spot_ask",
        "forward_mid_1m", "forward_bid_1m", "forward_ask_1m",
    ]
    for col in rate_cols:
        if col in panel.columns:
            vals = panel[col].astype(float)
            bad  = (vals <= 0)
            if bad.any():
                vals = vals.where(~bad, other=np.nan)
            panel[f"log_{col}"] = np.log(vals)
    return panel


# ──────────────────────────────────────────────────────────────────────────────
# Gates 8 & 9 — Sample-window and final schema validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_panel(panel: pd.DataFrame, log: logging.Logger) -> None:
    if panel.empty:
        log.error("Panel is empty after all filters")
        sys.exit(1)

    required_cols = [
        "date", "currency_code",
        "spot_mid", "forward_mid_1m",
        "log_spot_mid", "log_forward_mid_1m",
        "msci_class",
    ]
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        log.error("Output missing required columns: %s", missing)
        sys.exit(1)

    inf_count = int(np.isinf(panel.select_dtypes("number")).sum().sum())
    if inf_count:
        log.error(
            "GATE 9 FAIL — %d infinite values in numeric columns after log transforms; "
            "check for zero or negative source rates",
            inf_count,
        )
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Factor file cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_factors(path: Path, col_map: dict, log: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.rename(columns=col_map, inplace=True)
    df.columns = [_short_name(c) for c in df.columns]

    date_candidates = [c for c in df.columns if "date" in c.lower()]
    if not date_candidates:
        log.error("FACTOR GATE FAIL — no date column found in ff_factors.csv")
        sys.exit(1)
    date_col = date_candidates[0]

    # ff_factors.csv stores dates as DD/MM/YYYY ('31/01/2000').
    df["date"] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"])

    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

    desired    = ["mktrf", "smb", "hml", "umd", "rf"]
    avail_map  = {c.lower(): c for c in df.columns}
    keep_cols  = ["date"] + [avail_map[d] for d in desired if d in avail_map]

    df = df[keep_cols].copy()
    df = df[(df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)].copy()
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df["date"].duplicated().any():
        log.error("Duplicate dates in ff_factors.csv after cleaning")
        sys.exit(1)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────────

def export_outputs(
    panel     : pd.DataFrame,
    spot_only : pd.DataFrame,
    excluded  : pd.DataFrame,
    factors   : pd.DataFrame,
    audit     : pd.DataFrame,
    schemas   : dict,
    spot_map  : dict,
    fwd_map   : dict,
    log       : logging.Logger,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel.to_csv(OUTPUT_DIR / "fx_panel_clean.csv", index=False)
    print("Written: fx_panel_clean.csv")

    excluded.to_csv(OUTPUT_DIR / "excluded_pegged.csv", index=False)
    print("Written: excluded_pegged.csv")

    factors.to_csv(OUTPUT_DIR / "ff_factors_clean.csv", index=False)
    print("Written: ff_factors_clean.csv")

    audit.to_csv(OUTPUT_DIR / "coverage_audit.csv", index=False)
    print("Written: coverage_audit.csv")

    schemas_serial = {}
    for name, s in schemas.items():
        schemas_serial[name] = {
            "raw_columns"   : s["raw_columns"],
            "column_mapping": {k: v for k, v in s["column_mapping"].items()},
        }
    with open(OUTPUT_DIR / "schema_mappings.json", "w", encoding="utf-8") as f:
        json.dump(schemas_serial, f, indent=2)
    print("Written: schema_mappings.json")



# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def clean_data() -> dict:
    """Run the full data cleaning pipeline; return {'panel', 'excluded', 'factors'}."""
    log = setup_logging()
    print("Loading: §1 data cleaning (FX panel + factors) ...")

    # ── Gate 1: file availability ──
    resolve_inputs(INPUT_PATHS, log)

    # ── Gate 2: schema inspection ──
    schemas  = inspect_schemas(INPUT_PATHS, log)
    col_maps = {name: s["column_mapping"] for name, s in schemas.items()}

    # ── Gates 3/4: metadata and instrument identification (USD-base) ──
    meta_usd                = load_metadata(INPUT_PATHS["meta_usd"], col_maps["meta_usd"], log)
    spot_map, fwd_usd_map   = map_instruments(meta_usd, log)

    # ── Spot rates (chunked, USD-base) ──
    # FX_SPOT_Rates_USD.csv stores dates as ISO YYYY-MM-DD.
    spot_codes = {int(c) for c in spot_map.values()}
    raw_spot = load_fx_chunked(
        INPUT_PATHS["fx_spot_usd"], col_maps["fx_spot_usd"], spot_codes,
        "ISO8601", log,
    )

    # ── USD-base 1M forward rates (chunked) ──
    # FX_1MFD_Rates_USD.csv stores dates as DD/MM/YYYY.
    fwd_usd_codes = {int(c) for c in fwd_usd_map.values()}
    raw_fwd_usd = load_fx_chunked(
        INPUT_PATHS["fx_fwd_usd"], col_maps["fx_fwd_usd"], fwd_usd_codes,
        "%d/%m/%Y", log,
    )

    # ── Non-USD 1M forward rates (invert CCY/USD -> foreign/USD) ──
    non_usd_dfs:      list[pd.DataFrame] = []
    non_usd_code_map: dict[int, str]     = {}  # {exrateintcode: currency_iso}

    for ccy in NON_USD_FWD_CCYS:
        key_rates = f"fx_fwd_{ccy.lower()}"
        key_meta  = f"meta_{ccy.lower()}"
        df, code  = load_and_invert_non_usd_fwd(
            INPUT_PATHS[key_rates], INPUT_PATHS[key_meta], ccy, log,
        )
        non_usd_dfs.append(df)
        non_usd_code_map[code] = ccy

    raw_fwd_non_usd = pd.concat(non_usd_dfs, ignore_index=True)

    # ── Gate 5: quote orientation (spot rates only) ──
    check_quote_orientation(raw_spot, spot_map, log)

    # ── Quote normalisation and mid derivation ──
    raw_spot        = normalize_quotes(raw_spot,        log)
    raw_spot        = derive_mid_quotes(raw_spot,       log)

    raw_fwd_usd     = normalize_quotes(raw_fwd_usd,     log)
    raw_fwd_usd     = derive_mid_quotes(raw_fwd_usd,    log)

    raw_fwd_non_usd = normalize_quotes(raw_fwd_non_usd, log)
    raw_fwd_non_usd = derive_mid_quotes(raw_fwd_non_usd, log)

    # ── Gate 6: month-end aggregation (three separate calls) ──
    spot_eom        = aggregate_to_month_end(raw_spot,        log)
    fwd_usd_eom     = aggregate_to_month_end(raw_fwd_usd,     log)
    fwd_non_usd_eom = aggregate_to_month_end(raw_fwd_non_usd, log)

    # ── Instrument labelling -> long format ──
    spot_long    = build_instrument_long(spot_eom,    spot_map,    {},          log)
    fwd_usd_long = build_instrument_long(fwd_usd_eom, {},          fwd_usd_map, log)

    # Label non-USD forward rows directly
    fwd_non_usd_eom = fwd_non_usd_eom.copy()
    fwd_non_usd_eom["exrateintcode"] = fwd_non_usd_eom["exrateintcode"].astype(int)
    fwd_non_usd_eom["currency_code"] = fwd_non_usd_eom["exrateintcode"].map(non_usd_code_map)
    fwd_non_usd_eom["rate_type"]     = "fwd1m"
    non_usd_long = fwd_non_usd_eom[
        ["date", "currency_code", "rate_type", "midrate", "bidrate", "askrate"]
    ].copy()

    # ── Combined long format ──
    long_df = pd.concat([spot_long, fwd_usd_long, non_usd_long], ignore_index=True)

    # ── Pivot to wide panel ──
    panel = pivot_to_panel(long_df, log)

    # ── Gate 7: key uniqueness ──
    check_key_uniqueness(panel, log)

    # ── MSCI classification ──
    panel = apply_msci_classification(panel, log)

    # ── Pegged currency exclusion ──
    panel, excluded = flag_and_exclude_pegs(panel, log)

    # ── Forward coverage filter (drop spot-only currencies) ──
    panel, spot_only = filter_forward_coverage(panel, log)

    # ── Coverage audit ──
    audit = build_coverage_audit(panel, spot_only, excluded, log)

    # ── Log-price fields ──
    panel = add_log_fields(panel, log)

    # ── Gates 8/9: sample window and final schema ──
    validate_panel(panel, log)

    # ── Factor file ──
    factors = clean_factors(INPUT_PATHS["ff_factors"], col_maps["ff_factors"], log)

    # ── Combined forward map (USD + non-USD) for metadata export ──
    fwd_all_map = {**fwd_usd_map, **{v: k for k, v in non_usd_code_map.items()}}

    # ── Export all outputs ──
    export_outputs(panel, spot_only, excluded, factors, audit, schemas, spot_map, fwd_all_map, log)

    return {"panel": panel, "excluded": excluded, "factors": factors}


# ──────────────────────────────────────────────────────────────────────────────

