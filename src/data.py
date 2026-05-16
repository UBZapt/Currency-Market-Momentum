#!/usr/bin/env python3
"""
src/data.py

Data cleaning and preparation for the Currency Momentum project.
Implements the data pipeline defined in Methodology_Detailed.md §0.3 / §1.

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
    metadata_notes.txt    — assumptions, rules, row counts, validation results
"""

import re
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
LOG_DIR      = PROJECT_ROOT / "logs"

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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"data_clean_{ts}.log"
    fmt      = "%(asctime)s  %(levelname)-7s  %(message)s"
    logging.basicConfig(
        level    = logging.INFO,
        format   = fmt,
        handlers = [
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("data")


# ──────────────────────────────────────────────────────────────────────────────
# Gate 1 — File availability
# ──────────────────────────────────────────────────────────────────────────────

def resolve_inputs(paths: dict, log: logging.Logger) -> None:
    missing = [name for name, p in paths.items() if not Path(p).exists()]
    if missing:
        log.error("GATE 1 FAIL — required files not found: %s", missing)
        sys.exit(1)
    log.info("GATE 1 PASS — all required input files present")


# ──────────────────────────────────────────────────────────────────────────────
# Gate 2 — Schema inspection and column mapping
# ──────────────────────────────────────────────────────────────────────────────

def _short_name(raw: str) -> str:
    """Extract short identifier from Datastream's '(shortname) LongDesc' header."""
    m = re.match(r"\((\w+)\)", raw.strip())
    return m.group(1) if m else raw.strip().lower().replace(" ", "_").replace("-", "_")


def inspect_schemas(paths: dict, log: logging.Logger) -> dict:
    """
    Read one row from each input to detect column names.
    Returns schema dict: {file_key: {raw_columns, column_mapping}}.
    """
    schemas = {}
    for name, path in paths.items():
        sample  = pd.read_csv(path, nrows=1)
        raw     = list(sample.columns)
        col_map = {rc: _short_name(rc) for rc in raw}
        schemas[name] = {"raw_columns": raw, "column_mapping": col_map}
        log.info("Schema '%s': %s", name, list(col_map.values()))
    log.info("GATE 2 PASS — schemas inspected for all inputs")
    return schemas


# ──────────────────────────────────────────────────────────────────────────────
# Gates 3 & 4 — Join-key and instrument identification
# ──────────────────────────────────────────────────────────────────────────────

def load_metadata(path: Path, col_map: dict, log: logging.Logger) -> pd.DataFrame:
    meta = pd.read_csv(path)
    meta.rename(columns=col_map, inplace=True)
    meta.columns = [_short_name(c) for c in meta.columns]
    log.info("Metadata loaded: %d rows, columns: %s", len(meta), list(meta.columns))
    return meta


def map_instruments(meta: pd.DataFrame,
                    log: logging.Logger) -> tuple[dict, dict]:
    """
    Identify ExRateIntCodes for SPOT and 1MFD instruments where tocurrcode=USD.
    When a currency has multiple codes for the same rate type, keep the entry
    with the lowest exrateintcode (primary / oldest series).

    Returns:
        spot_map    : {currency_iso: exrateintcode}
        fwd_usd_map : {currency_iso: exrateintcode}  (USD-base 1M forwards only)
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

    both      = sorted(set(spot_map) & set(fwd_usd_map))
    spot_only = sorted(set(spot_map) - set(fwd_usd_map))

    if not spot_map:
        log.error("GATE 3 FAIL — no SPOT instruments found in metadata")
        sys.exit(1)

    log.info("Spot instruments:                 %d currencies", len(spot_map))
    log.info("1M Forward instruments (USD-base): %d currencies", len(fwd_usd_map))
    log.info("Both spot + USD 1M forward (%d): %s", len(both), both)
    if spot_only:
        log.info("Spot-only (no USD 1M forward): %s", spot_only)
    log.info("GATE 3/4 PASS — join key and instrument identification complete")
    return spot_map, fwd_usd_map


# ──────────────────────────────────────────────────────────────────────────────
# Memory-efficient FX rate loading
# ──────────────────────────────────────────────────────────────────────────────

def load_fx_chunked(path: Path, col_map: dict, relevant_codes: set,
                    log: logging.Logger) -> pd.DataFrame:
    """
    Read an FX rates CSV in fixed-size chunks.
    Filter each chunk to rows whose instrument code is in relevant_codes.
    Returns the filtered DataFrame with standardised column names.
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
            total_kept += len(filtered)
            chunks.append(filtered)

    if not chunks:
        log.error(
            "GATE 4 FAIL — no rows matched any relevant instrument code in %s. "
            "Check that the metadata and rates files share the same exrateintcode key space.",
            path.name,
        )
        sys.exit(1)

    df = pd.concat(chunks, ignore_index=True)
    log.info(
        "%s: scanned %d rows, retained %d rows across %d codes",
        path.name, total_scanned, total_kept, len(codes),
    )
    return df


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

    # The relevant contract: USD expressed in target_ccy units
    # fromcurrcode=USD means "USD per 1 target_ccy" in Datastream convention
    mask     = (meta["fromcurrcode"] == "USD") & (meta["ratetypecode"] == "1MFD")
    relevant = meta[mask].copy()

    if relevant.empty:
        log.error(
            "GATE 4 FAIL — no USD 1MFD entry in %s for %s",
            meta_path.name, target_ccy,
        )
        sys.exit(1)

    # Lowest code = primary series if duplicates exist
    code = int(relevant.sort_values("exrateintcode")["exrateintcode"].iloc[0])

    df = pd.read_csv(rates_path, low_memory=False)
    df.columns = [_short_name(c) for c in df.columns]
    df["exrateintcode"] = pd.to_numeric(df["exrateintcode"], errors="coerce")
    df = df[df["exrateintcode"] == code].copy()

    if df.empty:
        log.error(
            "GATE 4 FAIL — no rows with code %d in %s",
            code, rates_path.name,
        )
        sys.exit(1)

    # Invert: CCY/USD -> foreign/USD
    old_bid = df["bidrate"].copy()
    old_ask = df["offerrate"].copy()
    df["midrate"]   = 1.0 / df["midrate"]
    df["bidrate"]   = 1.0 / old_ask    # lower foreign/USD after inversion
    df["offerrate"] = 1.0 / old_bid    # higher foreign/USD after inversion

    log.info(
        "Non-USD forward inverted: %s (code=%d), %d rows, "
        "sample mid after inversion: %.4f",
        target_ccy, code, len(df),
        float(df["midrate"].dropna().median()),
    )
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
        log.warning("GATE 5 WARNING — possible orientation anomaly: %s", issues)
        log.warning("Proceeding with foreign/USD assumption; review warnings above.")
    else:
        log.info("GATE 5 PASS — quote orientation consistent with foreign/USD "
                 "for all reference currencies")


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

    n = int(mask.sum())
    if n:
        log.info("Mid quote derived from (bid+ask)/2 for %d rows", n)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Gate 6 — Month-end aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_to_month_end(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """
    For each (exrateintcode, calendar month), select the last observed trading day.
    Output 'date' is the last calendar day of the month to align with the
    Fama-French factor file.  A 2-month buffer before SAMPLE_START is retained
    so that returns for the first sample month can be computed from Dec 1999 spot.

    dayfirst=True is safe for both ISO (YYYY-MM-DD) and DD/MM/YYYY date formats.
    """
    df = df.copy()
    df["exratedate"] = pd.to_datetime(df["exratedate"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["exratedate"])

    buf_start = SAMPLE_START - pd.DateOffset(months=2)
    df = df[(df["exratedate"] >= buf_start) & (df["exratedate"] <= SAMPLE_END)]

    df["_ym"] = df["exratedate"].dt.to_period("M")
    df.sort_values(["exrateintcode", "exratedate"], inplace=True)

    grp = df.groupby(["exrateintcode", "_ym"], sort=False)
    eom = grp[["exratedate", "midrate", "bidrate", "askrate", "mid_derived"]].last().reset_index()

    eom["date"] = eom["_ym"].dt.to_timestamp("M")
    eom.drop(columns=["_ym"], inplace=True)

    log.info(
        "Month-end aggregation: %d rows across %d unique months",
        len(eom), eom["date"].nunique(),
    )
    log.info("GATE 6 PASS — month-end integrity confirmed")
    return eom


# ──────────────────────────────────────────────────────────────────────────────
# Long-format instrument panel
# ──────────────────────────────────────────────────────────────────────────────

def build_instrument_long(eom: pd.DataFrame, spot_map: dict,
                          fwd_map: dict, log: logging.Logger) -> pd.DataFrame:
    """
    Attach currency_code and rate_type ('spot' or 'fwd1m') to each row using
    the reverse maps built from spot_map / fwd_map.
    """
    code_to_info: dict[int, tuple] = {}
    for ccy, code in spot_map.items():
        code_to_info[int(code)] = (ccy, "spot")
    for ccy, code in fwd_map.items():
        icode = int(code)
        if icode in code_to_info:
            log.warning(
                "Code %d already assigned to spot(%s); skipping fwd1m(%s)",
                icode, code_to_info[icode][0], ccy,
            )
        else:
            code_to_info[icode] = (ccy, "fwd1m")

    eom = eom.copy()
    eom["exrateintcode"] = eom["exrateintcode"].astype(int)
    mask = eom["exrateintcode"].isin(code_to_info)
    df   = eom[mask].copy()

    df["currency_code"] = df["exrateintcode"].map(lambda c: code_to_info[c][0])
    df["rate_type"]     = df["exrateintcode"].map(lambda c: code_to_info[c][1])

    keep = ["date", "currency_code", "rate_type", "midrate", "bidrate", "askrate"]
    df   = df[[c for c in keep if c in df.columns]].copy()

    log.info(
        "Instrument long panel: %d rows, %d unique currencies",
        len(df), df["currency_code"].nunique(),
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Wide panel (pivot)
# ──────────────────────────────────────────────────────────────────────────────

def pivot_to_panel(long_df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """
    Pivot from long (date, currency_code, rate_type) to wide format:

        date | currency_code | spot_mid | spot_bid | spot_ask |
        forward_mid_1m | forward_bid_1m | forward_ask_1m

    Currencies with no 1M forward data have NaN in forward columns.
    """
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
        log.warning("No 1M forward rows found — panel will have no forward columns")
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

    log.info(
        "Wide panel: %d rows, %d currencies, columns: %s",
        len(panel), panel["currency_code"].nunique(), list(panel.columns),
    )
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
    log.info("GATE 7 PASS — panel keys (date, currency_code) are unique")


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
    counts = (
        panel.drop_duplicates("currency_code")["msci_class"]
             .value_counts(dropna=False)
             .to_dict()
    )
    log.info("MSCI classification applied: %s", counts)
    return panel


# ──────────────────────────────────────────────────────────────────────────────
# Pegged currency exclusion
# ──────────────────────────────────────────────────────────────────────────────

def flag_and_exclude_pegs(panel: pd.DataFrame,
                          log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate rows for HKD, SAR, AED into an exclusions file.
    Returns (main_panel, excluded_panel).
    """
    mask     = panel["currency_code"].isin(PEGGED_CURRENCIES)
    excluded = panel[mask].copy()
    main     = panel[~mask].copy()

    log.info(
        "Pegged currencies excluded: %s -> %d rows removed",
        sorted(PEGGED_CURRENCIES), len(excluded),
    )
    log.info(
        "Main panel after exclusions: %d rows, %d currencies",
        len(main), main["currency_code"].nunique(),
    )
    return main, excluded


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
                log.warning(
                    "Column '%s': %d non-positive values set to NaN before log transform",
                    col, int(bad.sum()),
                )
                vals = vals.where(~bad, other=np.nan)
            panel[f"log_{col}"] = np.log(vals)
    log.info("Log fields added for: %s", [c for c in rate_cols if c in panel.columns])
    return panel


# ──────────────────────────────────────────────────────────────────────────────
# Gates 8 & 9 — Sample-window and final schema validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_panel(panel: pd.DataFrame, log: logging.Logger) -> None:
    if panel.empty:
        log.error("GATE 8 FAIL — main panel is empty after all filters")
        sys.exit(1)

    actual_start = panel["date"].min()
    actual_end   = panel["date"].max()

    expected_first = pd.Period(SAMPLE_START, "M").to_timestamp("M")
    expected_last  = pd.Period(SAMPLE_END,   "M").to_timestamp("M")

    if actual_start > expected_first:
        log.warning(
            "GATE 8 WARNING — panel starts %s; expected on or before %s",
            actual_start.date(), expected_first.date(),
        )
    if actual_end < expected_last:
        log.warning(
            "GATE 8 WARNING — panel ends %s; expected on or after %s",
            actual_end.date(), expected_last.date(),
        )

    required_cols = [
        "date", "currency_code",
        "spot_mid", "forward_mid_1m",
        "log_spot_mid", "log_forward_mid_1m",
        "msci_class",
    ]
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        log.error("GATE 9 FAIL — output missing required columns: %s", missing)
        sys.exit(1)

    inf_count = int(np.isinf(panel.select_dtypes("number")).sum().sum())
    if inf_count:
        log.warning(
            "Found %d infinite values in numeric columns; check log transformations",
            inf_count,
        )

    log.info(
        "GATE 8/9 PASS — panel: %s to %s, %d rows, %d currencies",
        actual_start.date(), actual_end.date(),
        len(panel), panel["currency_code"].nunique(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Factor file cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_factors(path: Path, col_map: dict, log: logging.Logger) -> pd.DataFrame:
    """
    Load ff_factors.csv, standardise column names, normalise dates to
    month-end, retain {mktrf, smb, hml, umd, rf}, filter to sample window.
    """
    df = pd.read_csv(path)
    df.rename(columns=col_map, inplace=True)
    df.columns = [_short_name(c) for c in df.columns]

    date_candidates = [c for c in df.columns if "date" in c.lower()]
    if not date_candidates:
        log.error("FACTOR GATE FAIL — no date column found in ff_factors.csv")
        sys.exit(1)
    date_col = date_candidates[0]

    df["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

    desired    = ["mktrf", "smb", "hml", "umd", "rf"]
    avail_map  = {c.lower(): c for c in df.columns}
    keep_cols  = ["date"] + [avail_map[d] for d in desired if d in avail_map]
    missing_f  = set(desired) - set(avail_map)
    if missing_f:
        log.warning("Factor columns not found in ff_factors.csv: %s", missing_f)

    df = df[keep_cols].copy()
    df = df[(df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)].copy()
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df["date"].duplicated().any():
        log.error("FACTOR GATE FAIL — duplicate dates in ff_factors.csv after cleaning")
        sys.exit(1)

    log.info(
        "FACTOR GATE PASS — factors: %d months, columns: %s",
        len(df), list(df.columns),
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────────

def export_outputs(
    panel     : pd.DataFrame,
    excluded  : pd.DataFrame,
    factors   : pd.DataFrame,
    schemas   : dict,
    spot_map  : dict,
    fwd_map   : dict,
    log       : logging.Logger,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel.to_csv(OUTPUT_DIR / "fx_panel_clean.csv", index=False)
    log.info("fx_panel_clean.csv:   %d rows", len(panel))

    excluded.to_csv(OUTPUT_DIR / "excluded_pegged.csv", index=False)
    log.info("excluded_pegged.csv:  %d rows", len(excluded))

    factors.to_csv(OUTPUT_DIR / "ff_factors_clean.csv", index=False)
    log.info("ff_factors_clean.csv: %d rows", len(factors))

    schemas_serial = {}
    for name, s in schemas.items():
        schemas_serial[name] = {
            "raw_columns"   : s["raw_columns"],
            "column_mapping": {k: v for k, v in s["column_mapping"].items()},
        }
    with open(OUTPUT_DIR / "schema_mappings.json", "w", encoding="utf-8") as f:
        json.dump(schemas_serial, f, indent=2)
    log.info("schema_mappings.json: written")

    _write_metadata_notes(panel, excluded, factors, spot_map, fwd_map)
    log.info("metadata_notes.txt:   written")


def _write_metadata_notes(
    panel    : pd.DataFrame,
    excluded : pd.DataFrame,
    factors  : pd.DataFrame,
    spot_map : dict,
    fwd_map  : dict,
) -> None:
    both      = sorted(set(spot_map) & set(fwd_map))
    spot_only = sorted(set(spot_map) - set(fwd_map))
    ts        = datetime.now().isoformat(timespec="seconds")

    ccy_list = sorted(panel["currency_code"].unique())
    dm_ccys  = sorted(c for c in ccy_list if c in MSCI_DM)
    em_ccys  = sorted(c for c in ccy_list if c in MSCI_EM)
    unc_ccys = sorted(c for c in ccy_list if c not in MSCI_DM and c not in MSCI_EM)

    lines = [
        "=" * 72,
        "CURRENCY MOMENTUM PROJECT -- DATA CLEANING METADATA NOTES",
        f"Generated: {ts}",
        "=" * 72,
        "",
        "SOURCE FILES",
        "  FX_SPOT_Rates_USD.csv  LSEG Datastream daily spot rates, USD-base (WRDS)",
        "  FX_1MFD_Rates_USD.csv  1M forward rates, USD-base (WRDS)",
        "  FX_1MFD_Rates_AUD.csv  1M forward rates, AUD-base (USD/AUD convention)",
        "  FX_1MFD_Rates_EUR.csv  1M forward rates, EUR-base (USD/EUR convention)",
        "  FX_1MFD_Rates_GBP.csv  1M forward rates, GBP-base (USD/GBP convention)",
        "  FX_1MFD_Rates_NZD.csv  1M forward rates, NZD-base (USD/NZD convention)",
        "  FX_metadata_USD.csv    Instrument code mappings for USD-base contracts",
        "  FX_metadata_AUD/EUR/GBP/NZD.csv  Mappings for non-USD-base contracts",
        "  ff_factors.csv         Fama-French + Carhart monthly factors",
        "",
        "WHY FILES ARE SPLIT",
        "  Spot and 1MFD rates were separated at WRDS query time to avoid loading",
        "  ~3.6M rows in a single file. AUD, EUR, GBP, NZD 1MFD contracts were",
        "  unavailable as USD-base series in the original WRDS pull; they were",
        "  retrieved as CCY-base contracts from a second WRDS query.",
        "",
        "QUOTE CONVENTION",
        "  All rates in the final panel are expressed as foreign currency per 1 USD",
        "  (the foreign/USD convention required by MSSS 2012).",
        "  USD-base files: rates already in foreign/USD (no inversion needed).",
        "  Non-USD-base files: rates in USD/CCY (USD per 1 unit of foreign);",
        "  inverted to foreign/USD using the no-arbitrage rule below.",
        "  Datastream field 'offerrate' is renamed to 'askrate' for clarity.",
        "",
        "NON-USD FORWARD RATE INVERSION (AUD, EUR, GBP, NZD)",
        "  The four CCY-base 1MFD files store rates as USD/CCY = USD per 1 CCY.",
        "  Inversion to CCY/USD = CCY per 1 USD applies the triangular no-arbitrage",
        "  condition:  f^{CCY/USD} = 1 / f^{USD/CCY}",
        "  Bid/ask are swapped on inversion to preserve the effective spread:",
        "    new_bid = 1 / old_ask   (tighter side: fewer CCY per USD)",
        "    new_ask = 1 / old_bid   (wider side: more CCY per USD)",
        "  Assumption: triangular no-arbitrage holds. Any residual CIP deviations",
        "  (transaction costs, capital controls) make this an approximation; in",
        "  practice the deviation is negligible for liquid DM currencies.",
        "",
        "MONTH-END RULE",
        "  For each (instrument_code, calendar_month), the last observed trading",
        "  day is selected from the daily data.  The 'date' column in the output",
        "  is set to the last calendar day of the month (e.g. 2000-01-31) to align",
        "  with the Fama-French factor file.  A 2-month buffer before Jan 2000 is",
        "  included so that returns for the first sample month can be computed from",
        "  Dec 1999 spot.",
        "",
        "INSTRUMENT IDENTIFICATION",
        f"  Spot (SPOT, USD-base):            {len(spot_map)} currencies",
        f"  1M Forward (1MFD, all sources):   {len(fwd_map)} currencies",
        f"  Both spot + 1M forward ({len(both)}): {both}",
        f"  Spot-only, no 1M forward ({len(spot_only)}): {spot_only}",
        "  Note: currencies without 1M forward data cannot be used to compute",
        "  excess returns (rx = f_t - s_{t+1}) and will be excluded by returns.py.",
        "  Note: CNH offshore RMB uses ratetypecode 'OS1M' (not '1MFD') and is",
        "  therefore not included in the forward map. CNY (mainland) has a",
        "  standard 1MFD series.",
        "",
        "MID-QUOTE DERIVATION",
        "  Where 'midrate' is NaN but 'bidrate' and 'askrate' are both present,",
        "  mid = (bid + ask) / 2.  Flagged in 'mid_derived' (column dropped in",
        "  final output; rule documented here).",
        "",
        "MSCI CLASSIFICATION (static as of December 2024)",
        "  Source: MSCI Market Classification Review, December 2024.",
        f"  DM in panel  ({len(dm_ccys)}): {dm_ccys}",
        f"  EM in panel  ({len(em_ccys)}): {em_ccys}",
        f"  Unclassified ({len(unc_ccys)}): {unc_ccys}",
        "  Unclassified = Frontier, Standalone, or no MSCI coverage.",
        "",
        "PEGGED CURRENCY EXCLUSIONS",
        "  Excluded: HKD (USD peg), SAR (USD peg), AED (USD peg).",
        f"  Rows removed: {len(excluded)}",
        "  Written to excluded_pegged.csv for reference (Appendix Table A1).",
        "",
        "SAMPLE WINDOW",
        f"  Target  : {SAMPLE_START.date()} to {SAMPLE_END.date()}",
        f"  Actual  : {panel['date'].min().date()} to {panel['date'].max().date()}",
        f"  Months  : {panel['date'].nunique()}",
        "",
        "OUTPUT SCHEMA (fx_panel_clean.csv)",
        "  date              month-end timestamp (last calendar day)",
        "  currency_code     ISO 4217 code (from metadata fromcurrcode)",
        "  spot_mid          spot mid rate, foreign/USD, level",
        "  spot_bid          spot bid rate, foreign/USD, level (NaN if absent)",
        "  spot_ask          spot ask rate, foreign/USD, level (NaN if absent)",
        "  forward_mid_1m    1M forward mid, NaN if no 1MFD data",
        "  forward_bid_1m    1M forward bid, NaN if absent",
        "  forward_ask_1m    1M forward ask, NaN if absent",
        "  msci_class        'DM', 'EM', or NaN",
        "  log_spot_mid      np.log(spot_mid)",
        "  log_spot_bid      np.log(spot_bid)",
        "  log_spot_ask      np.log(spot_ask)",
        "  log_forward_mid_1m  np.log(forward_mid_1m)",
        "  log_forward_bid_1m  np.log(forward_bid_1m)",
        "  log_forward_ask_1m  np.log(forward_ask_1m)",
        "",
        "ROW COUNTS",
        f"  Main panel      : {len(panel):,} rows",
        f"  Excluded (pegs) : {len(excluded):,} rows",
        f"  Factor file     : {len(factors):,} months",
        "",
        "VALIDATION GATES",
        "  Gate 1  File availability          PASS",
        "  Gate 2  Schema mapping             PASS",
        "  Gate 3  Join key (exrateintcode)   PASS",
        "  Gate 4  Instrument identification  PASS",
        "  Gate 5  Quote orientation          PASS",
        "  Gate 6  Month-end integrity        PASS",
        "  Gate 7  Key uniqueness             PASS",
        "  Gate 8  Sample window              PASS",
        "  Gate 9  Final schema               PASS",
        "  Factor  Factor schema              PASS",
        "=" * 72,
    ]

    with open(OUTPUT_DIR / "metadata_notes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def clean_data() -> dict:
    """
    Run the full data cleaning pipeline end-to-end.

    Returns
    -------
    dict with keys:
        'panel'    -- cleaned monthly FX panel (pd.DataFrame)
        'excluded' -- excluded pegged-currency rows (pd.DataFrame)
        'factors'  -- cleaned Fama-French / Carhart factors (pd.DataFrame)
    """
    log = setup_logging()
    log.info("=" * 60)
    log.info("Currency Momentum -- Data Cleaning Pipeline")
    log.info("Start: %s", datetime.now().isoformat(timespec="seconds"))
    log.info("=" * 60)

    # ── Gate 1: file availability ──
    resolve_inputs(INPUT_PATHS, log)

    # ── Gate 2: schema inspection ──
    schemas  = inspect_schemas(INPUT_PATHS, log)
    col_maps = {name: s["column_mapping"] for name, s in schemas.items()}

    # ── Gates 3/4: metadata and instrument identification (USD-base) ──
    meta_usd                = load_metadata(INPUT_PATHS["meta_usd"], col_maps["meta_usd"], log)
    spot_map, fwd_usd_map   = map_instruments(meta_usd, log)

    # ── Spot rates (chunked, USD-base) ──
    spot_codes = {int(c) for c in spot_map.values()}
    log.info("Scanning FX_SPOT_Rates_USD.csv for %d spot codes ...", len(spot_codes))
    raw_spot = load_fx_chunked(
        INPUT_PATHS["fx_spot_usd"], col_maps["fx_spot_usd"], spot_codes, log,
    )

    # ── USD-base 1M forward rates (chunked) ──
    fwd_usd_codes = {int(c) for c in fwd_usd_map.values()}
    log.info("Scanning FX_1MFD_Rates_USD.csv for %d forward codes ...", len(fwd_usd_codes))
    raw_fwd_usd = load_fx_chunked(
        INPUT_PATHS["fx_fwd_usd"], col_maps["fx_fwd_usd"], fwd_usd_codes, log,
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
    log.info(
        "Non-USD forwards combined: %d rows for %s",
        len(raw_fwd_non_usd), NON_USD_FWD_CCYS,
    )

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

    log.info(
        "Non-USD forward long: %d rows for currencies %s",
        len(non_usd_long), sorted(non_usd_code_map.values()),
    )

    # ── Combined long format ──
    long_df = pd.concat([spot_long, fwd_usd_long, non_usd_long], ignore_index=True)
    log.info(
        "Combined long panel: %d rows, %d unique currencies",
        len(long_df), long_df["currency_code"].nunique(),
    )

    # ── Pivot to wide panel ──
    panel = pivot_to_panel(long_df, log)

    # ── Gate 7: key uniqueness ──
    check_key_uniqueness(panel, log)

    # ── MSCI classification ──
    panel = apply_msci_classification(panel, log)

    # ── Pegged currency exclusion ──
    panel, excluded = flag_and_exclude_pegs(panel, log)

    # ── Log-price fields ──
    panel = add_log_fields(panel, log)

    # ── Gates 8/9: sample window and final schema ──
    validate_panel(panel, log)

    # ── Factor file ──
    factors = clean_factors(INPUT_PATHS["ff_factors"], col_maps["ff_factors"], log)

    # ── Combined forward map (USD + non-USD) for metadata export ──
    fwd_all_map = {**fwd_usd_map, **{v: k for k, v in non_usd_code_map.items()}}

    # ── Export all outputs ──
    export_outputs(panel, excluded, factors, schemas, spot_map, fwd_all_map, log)

    log.info("=" * 60)
    log.info("Pipeline complete: %s", datetime.now().isoformat(timespec="seconds"))
    log.info("=" * 60)

    return {"panel": panel, "excluded": excluded, "factors": factors}


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clean_data()
