#!/usr/bin/env python3
"""
MAIN.py — Currency Momentum Project master script.

Orchestrates the full pipeline from data cleaning through all empirical steps.
Each section maps to Methodology_Detailed.md.

Usage:
    python MAIN.py               # run full pipeline
    python MAIN.py --skip-data   # skip data cleaning (use cached panel)
"""

import sys
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
LOG_DIR      = PROJECT_ROOT / "logs"

PANEL_PATH   = OUTPUT_DIR / "fx_panel_clean.csv"


def get_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    logging.basicConfig(
        level    = logging.INFO,
        format   = fmt,
        handlers = [
            logging.FileHandler(LOG_DIR / f"main_{ts}.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("main")


def main() -> None:
    parser = argparse.ArgumentParser(description="Currency Momentum Pipeline")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data cleaning and use cached fx_panel_clean.csv",
    )
    args = parser.parse_args()

    log = get_logger()
    log.info("=" * 60)
    log.info("Currency Momentum -- Master Pipeline")
    log.info("=" * 60)

    # ── §1: Data cleaning ──────────────────────────────────────────────────────
    if args.skip_data:
        if not PANEL_PATH.exists():
            log.error("--skip-data requested but %s not found. Run without flag.", PANEL_PATH)
            sys.exit(1)
        log.info("§1 Data cleaning: skipped (cached panel found at %s)", PANEL_PATH.name)
        from src.data import clean_data
        data_result = {"panel": None, "excluded": None, "factors": None}
    else:
        log.info("§1 Data cleaning: starting ...")
        from src.data import clean_data
        data_result = clean_data()
        log.info("§1 Data cleaning: complete")

    # ── §2: Returns and RX benchmark ──────────────────────────────────────────
    log.info("§2 Compute returns + RX index: starting ...")
    from src.returns import compute_all_returns
    returns_result = compute_all_returns(panel=data_result.get("panel"))
    log.info("§2 Returns: complete")

    # ── §3: Short-term reversal test ──────────────────────────────────────────
    log.info("§3 Short-term reversal test: starting ...")
    from src.regressions import short_term_reversal, print_reversal_table
    reversal_result = short_term_reversal(
        returns_panel=returns_result["returns_panel"], verbose=True
    )
    print_reversal_table(reversal_result)
    skip_month_flag = reversal_result["skip_month_flag"]
    log.info("§3 Short-term reversal: complete — skip_month_flag=%s", skip_month_flag)

    # ── §4: Portfolio machinery validation ───────────────────────────────────
    log.info("§4 Portfolio machinery validation: starting ...")
    from src.validate_section4 import run_section4_validation
    section4_ok = run_section4_validation(
        returns_panel=returns_result["returns_panel"],
    )
    if not section4_ok:
        log.error("§4 validation failed — aborting")
        sys.exit(1)
    log.info("§4 Portfolio machinery: validated")

    # ── §5: Formation-Holding Grid + Section 5 outputs ───────────────────────
    log.info("§5 Formation-Holding Grid + outputs: starting ...")
    from src.section5 import run_section5
    section5_result = run_section5(
        returns_panel=returns_result["returns_panel"],
        rx_factor=returns_result["rx_factor"],
    )
    log.info("§5 complete.")

    # ── §6: Transaction-cost-adjusted momentum returns ────────────────────────
    log.info("§6 Transaction costs: starting ...")
    from src.section6 import run_section6
    section6_result = run_section6(
        returns_panel=returns_result["returns_panel"],
        section5_result=section5_result,
    )
    _ = section6_result  # available for §7 onward
    log.info("§6 complete.")

    # ── §7: Carry trade comparison + double sort ──────────────────────────────
    log.info("§7 Carry trade + double sort: starting ...")
    from src.section7 import run_section7
    section7_result = run_section7(
        returns_panel=returns_result["returns_panel"],
        section5_result=section5_result,
    )
    log.info("§7 complete.")

    # ── §8: Sharpe ratio grid (Signal A + CT carry) ──────────────────────────
    log.info("§8 Sharpe ratio grid: starting ...")
    from src.section8 import run_section8
    section8_result = run_section8(
        section5_result=section5_result,
        section7_result=section7_result,
    )
    _ = section8_result
    log.info("§8 complete.")

    # ── §9.1: Fama-MacBeth cross-sectional regressions (Table 13) ────────────
    # §9.2 (lag decomposition / Table 14) intentionally omitted — rolling OLS
    # results in §5.4 were inconclusive, so the lag decomposition is not reported.
    log.info("§9.1 Fama-MacBeth regressions: starting ...")
    from src.section9 import run_section9
    section9_result = run_section9(
        returns_panel=returns_result["returns_panel"],
    )
    _ = section9_result
    log.info("§9.1 complete.")

    # ── §10: Factor regressions (Table 15) ────────────────────────────────────
    log.info("§10 Factor regressions: starting ...")
    from src.section10 import run_section10
    section10_result = run_section10(
        returns_panel=returns_result["returns_panel"],
        factors=data_result.get("factors"),
        section5_result=section5_result,
        section7_result=section7_result,
    )
    _ = section10_result
    log.info("§10 complete.")

    # ── §11.2: Volatility-scaled (TSMOM) f×h grid (Appendix Table A3) ────────
    # §11.1, §11.3, §11.4 intentionally omitted — only §11.2 implemented.
    log.info("§11.2 TSMOM f×h grid: starting ...")
    from src.section11 import run_section11
    section11_result = run_section11(
        returns_panel=returns_result["returns_panel"],
    )
    _ = section11_result
    log.info("§11.2 complete.")

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
