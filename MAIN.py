#!/usr/bin/env python3
"""
MAIN.py — Currency Momentum Project master script.

Orchestrates the full pipeline from data cleaning through all empirical steps.
Section numbers (§) follow the pipeline map in README.md.

Usage:
    python MAIN.py               # run full pipeline
    python MAIN.py --skip-data   # skip data cleaning (use cached panel)
"""

import sys
import argparse
from pathlib import Path

# Output tables contain Unicode (ℓ, ², ×, etc.); force utf-8 on the console.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
PANEL_PATH   = OUTPUT_DIR / "fx_panel_clean.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Currency Momentum Pipeline")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data cleaning and use cached fx_panel_clean.csv",
    )
    args = parser.parse_args()

    # ── §1: Data cleaning ──────────────────────────────────────────────────────
    if args.skip_data:
        if not PANEL_PATH.exists():
            print(f"--skip-data requested but {PANEL_PATH} not found. Run without flag.",
                  file=sys.stderr)
            sys.exit(1)
        data_result = {"panel": None, "excluded": None, "factors": None}
    else:
        from src.data import clean_data
        data_result = clean_data()

    # ── §2: Returns and RX benchmark ──────────────────────────────────────────
    from src.returns import compute_all_returns
    returns_result = compute_all_returns(panel=data_result.get("panel"))

    # ── §3: Short-term reversal test ──────────────────────────────────────────
    from src.regressions import short_term_reversal, print_reversal_table
    reversal_result = short_term_reversal(returns_panel=returns_result["returns_panel"])
    print_reversal_table(reversal_result)

    # ── §4: Portfolio machinery validation ───────────────────────────────────
    from src.validate_section4 import run_section4_validation
    if not run_section4_validation(returns_panel=returns_result["returns_panel"]):
        sys.exit(1)

    # ── §5: Formation-Holding Grid + Section 5 outputs ───────────────────────
    from src.section5 import run_section5
    section5_result = run_section5(
        returns_panel=returns_result["returns_panel"],
        rx_factor=returns_result["rx_factor"],
    )

    # ── §6: Transaction-cost-adjusted momentum returns ────────────────────────
    from src.section6 import run_section6
    run_section6(
        returns_panel=returns_result["returns_panel"],
        section5_result=section5_result,
    )

    # ── §7: Carry trade comparison + double sort ──────────────────────────────
    from src.section7 import run_section7
    section7_result = run_section7(
        returns_panel=returns_result["returns_panel"],
        section5_result=section5_result,
    )

    # ── §8: Sharpe ratio grid (Signal A + CT carry) ──────────────────────────
    from src.section8 import run_section8
    run_section8(
        section5_result=section5_result,
        section7_result=section7_result,
    )

    # ── §9.1: Fama-MacBeth cross-sectional regressions (Table 13) ────────────
    from src.section9 import run_section9
    run_section9(returns_panel=returns_result["returns_panel"])

    # ── §10: Factor regressions (Table 15) ────────────────────────────────────
    from src.section10 import run_section10
    run_section10(
        returns_panel=returns_result["returns_panel"],
        factors=data_result.get("factors"),
        section5_result=section5_result,
        section7_result=section7_result,
    )

    # ── §11.2: Volatility-scaled (TSMOM) f×h grid (Appendix Table A3) ────────
    from src.section11 import run_section11
    run_section11(returns_panel=returns_result["returns_panel"])

    # ── Flush consolidated results workbook ──────────────────────────────────
    from src.output_writer import write_workbook
    write_workbook()


if __name__ == "__main__":
    main()
