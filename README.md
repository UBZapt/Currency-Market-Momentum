# Currency-Market-Momentum

Replication of Menkhoff, Sarno, Schmeling & Schrimpf (2012) "Carry Trades and Global Foreign Exchange Volatility" using LSEG Datastream data via WRDS, covering January 2000 – December 2024.

---

## Dependencies

- Python 3.10+
- pandas, numpy

Install: `pip install pandas numpy`

---

## Environment setup

No live WRDS credentials are required for routine runs once the raw data files are present locally. The pipeline reads CSV files directly.

---

## Data Sources and Preparation

### Primary source

All FX rate data is sourced from LSEG Datastream via WRDS. Fama-French and Carhart four-factor data are from the Kenneth French Data Library.

### Why files are split

Spot and 1-month forward (1MFD) rates are stored in separate files because they were queried separately at WRDS to avoid loading a single ~3.6 million-row file. The split also makes the instrument scope explicit at a glance.

### Non-USD forward contracts (AUD, EUR, GBP, NZD)

AUD, EUR, GBP, and NZD 1MFD contracts were unavailable as USD-base series in the original WRDS pull. They were retrieved as CCY-base contracts from a second WRDS query, where rates are expressed as:

    CCY/USD  =  USD per 1 unit of foreign currency  (e.g. AUD/USD ≈ 0.658)

The methodology requires all rates in foreign/USD convention (units of foreign per 1 USD). The pipeline inverts these four series using the triangular no-arbitrage condition:

    f^{foreign/USD} = 1 / f^{USD/foreign}

Bid and ask are swapped on inversion to preserve the effective spread:

    new_bid = 1 / old_ask   (lower bound: fewest units of foreign per USD)
    new_ask = 1 / old_bid   (upper bound: most units of foreign per USD)

**Assumption:** triangular no-arbitrage holds exactly. Any residual covered-interest-parity deviations driven by transaction costs or capital controls mean this inversion is an approximation. In practice the deviation is negligible for these liquid developed-market currencies.

---

## Required input files (place in project root)

| File | Description |
|---|---|
| `FX_SPOT_Rates_USD.csv` | Daily spot rates, USD-base, ~1.1M rows |
| `FX_1MFD_Rates_USD.csv` | Daily 1M forward rates, USD-base, ~173k rows |
| `FX_1MFD_Rates_AUD.csv` | Daily 1M forward rates, AUD-base (USD/AUD), ~6.5k rows |
| `FX_1MFD_Rates_EUR.csv` | Daily 1M forward rates, EUR-base (USD/EUR), ~6.5k rows |
| `FX_1MFD_Rates_GBP.csv` | Daily 1M forward rates, GBP-base (USD/GBP), ~6.5k rows |
| `FX_1MFD_Rates_NZD.csv` | Daily 1M forward rates, NZD-base (USD/NZD), ~6.5k rows |
| `FX_metadata_USD.csv` | Instrument code mappings for USD-base contracts |
| `FX_metadata_AUD.csv` | Instrument code mappings for AUD-base contracts |
| `FX_metadata_EUR.csv` | Instrument code mappings for EUR-base contracts |
| `FX_metadata_GBP.csv` | Instrument code mappings for GBP-base contracts |
| `FX_metadata_NZD.csv` | Instrument code mappings for NZD-base contracts |
| `ff_factors.csv` | Fama-French + Carhart monthly factors |

All rates files have columns: `exrateintcode, exratedate, midrate, bidrate, offerrate, licflag`

Date format: YYYY-MM-DD in `FX_SPOT_Rates_USD.csv`; DD/MM/YYYY in all forward files.

---

## Running the pipeline

```
python src/data.py
```

This runs the full cleaning pipeline and writes outputs to `output/`.

---

## Outputs

| File | Description |
|---|---|
| `output/fx_panel_clean.csv` | Monthly end-of-month FX panel (date × currency) |
| `output/ff_factors_clean.csv` | Cleaned Fama-French + Carhart factors |
| `output/excluded_pegged.csv` | Rows removed for HKD, SAR, AED (USD pegs) |
| `output/schema_mappings.json` | Raw-to-standard column maps for all input files |
| `output/metadata_notes.txt` | Full documentation of assumptions, rules, and counts |

---

## Assumptions and implementation notes

- **Quote convention:** all rates in `fx_panel_clean.csv` are in foreign/USD (units of foreign per 1 USD).
- **Month-end rule:** for each instrument and calendar month, the last observed trading day is used. The `date` column is set to the last calendar day of the month to align with the Fama-French file.
- **Pegged currencies excluded:** HKD, SAR, AED are removed from the main panel (methodology §1.1).
- **MSCI classification:** static as of December 2024. Unclassified currencies are Frontier, Standalone, or not covered by MSCI.
- **Non-positive spot values:** zero and negative rate values (Zimbabwe dollar hyperinflation era artifacts) are replaced with NaN before log transformation.
- **Forward coverage:** 31 currencies have 1M forward data (27 USD-base + AUD, EUR, GBP, NZD via inversion). Only these 31 can be used to compute excess returns in `src/returns.py`.

---

## Deviations from Methodology_Detailed.md

| Date | Deviation | Reason |
|---|---|---|
| 2026-05-16 | AUD/EUR/GBP/NZD 1MFD rates retrieved as CCY-base and inverted | WRDS did not carry these pairs as USD-base 1MFD contracts; retrieved from second query and inverted under no-arbitrage assumption |

---

## Attribution

- Menkhoff, L., Sarno, L., Schmeling, M., & Schrimpf, A. (2012). Carry trades and global foreign exchange volatility. *Journal of Finance*, 67(2), 681–718.
- FX data: LSEG Datastream via WRDS.
- Factor data: Kenneth R. French Data Library.
