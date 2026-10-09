# Currency-Market-Momentum

Replication and extension of Menkhoff, Sarno, Schmeling & Schrimpf (2012), *"Currency Momentum Strategies"* (*Journal of Financial Economics*), for January 2000 – December 2024. Returns are measured from the perspective of a U.S. investor (USD base) across the 31 currencies with one-month forward quotes in LSEG Datastream. USD-pegged currencies (HKD, SAR, AED) are excluded.

The pipeline builds cross-sectional momentum portfolios over a 5 × 5 formation/holding grid, then tests them against transaction costs, the carry trade, Fama-MacBeth regressions, equity and FX risk factors, and a volatility-scaled signal.

---

## Requirements

- **Python 3.12 or later.** Several table-printing f-strings contain a backslash inside the expression, which is a syntax error before Python 3.12.
- Packages (versions the pipeline was run with):

| Package | Version | Used for |
|---|---|---|
| pandas | 2.2.3 | Panel data |
| numpy | 2.4.5 | Arithmetic |
| statsmodels | 0.14.6 | OLS and Newey-West HAC inference |
| matplotlib | 3.10.9 | Figure 1 (PDF) |
| openpyxl | 3.1.5 | Writing `output/results.xlsx` |

```
pip install pandas==2.2.3 numpy==2.4.5 statsmodels==0.14.6 matplotlib==3.10.9 openpyxl==3.1.5
```

---

## Data access and licensing

**No data is included in this repository.** The FX data is licensed from LSEG Datastream through WRDS and may not be redistributed. This applies to the raw extracts and to the derived files written to `output/`. To reproduce the results you need your own WRDS subscription with access to Datastream FX rates.

The code reads local CSV files only. It does not connect to WRDS, read credentials, or make network calls.

### Required input files (project root)

| File | Rows (incl. header) | Contents |
|---|---|---|
| `FX_SPOT_Rates_USD.csv` | 1,100,953 | Daily spot rates, USD-base contracts |
| `FX_1MFD_Rates_USD.csv` | 173,277 | Daily 1-month forward rates, USD-base contracts |
| `FX_1MFD_Rates_AUD.csv` | 6,523 | Daily 1-month forward rates, AUD-base (USD per 1 AUD) |
| `FX_1MFD_Rates_EUR.csv` | 6,523 | Daily 1-month forward rates, EUR-base (USD per 1 EUR) |
| `FX_1MFD_Rates_GBP.csv` | 6,523 | Daily 1-month forward rates, GBP-base (USD per 1 GBP) |
| `FX_1MFD_Rates_NZD.csv` | 6,510 | Daily 1-month forward rates, NZD-base (USD per 1 NZD) |
| `FX_metadata_USD.csv` | 689 | Instrument definitions for USD-base contracts |
| `FX_metadata_AUD.csv`, `_EUR`, `_GBP`, `_NZD` | 42–220 | Instrument definitions for the non-USD-base contracts |
| `ff_factors.csv` | 301 | Monthly Fama-French three factors + Carhart momentum, decimal form |

Source columns expected by `src/data.py`:

| File type | Columns | Date format |
|---|---|---|
| Rate files | `exrateintcode, exratedate, midrate, bidrate, offerrate, licflag` | `YYYY-MM-DD` (spot); `DD/MM/YYYY` (all forward files) |
| Metadata files | `tocurrcode, exrateintcode, exratecode, fromcurrcode, sourcecode, ratetypecode, exratedesc` | — |
| Factor file | `mktrf, smb, hml, rf, umd, dateff` | `DD/MM/YYYY` |

### Field mapping

- `ratetypecode` = `SPOT` or `1MFD` selects spot or one-month forward instruments. For USD-base contracts, `tocurrcode = USD` and `fromcurrcode` gives the ISO currency code (`currency_code`).
- Where a currency has more than one series, the lowest `exrateintcode` is kept as the primary series.
- `midrate`, `bidrate` and `offerrate` map to `spot_mid` / `spot_bid` / `spot_ask` and `forward_mid_1m` / `forward_bid_1m` / `forward_ask_1m`. A missing mid is filled with (bid + ask) / 2 when both are present.
- **AUD, EUR, GBP and NZD forwards** were only available as CCY-base contracts (USD per 1 unit of foreign currency). The pipeline inverts them to foreign per USD using triangular no-arbitrage, swapping bid and ask so the spread is preserved:

      mid' = 1 / mid      bid' = 1 / ask      ask' = 1 / bid

---

## How to run

From the project root, with the input files in place:

```
python MAIN.py
```

This runs every step end-to-end and writes all outputs to `output/`. To reuse an already-cleaned panel (`output/fx_panel_clean.csv` and `output/ff_factors_clean.csv`) and skip the raw-file load:

```
python MAIN.py --skip-data
```

---

## Pipeline map

`MAIN.py` calls each step in order. Section numbers (§) used in code comments refer to these steps.

| § | Module | Step | Output |
|---|---|---|---|
| 1 | `src/data.py` | Load, month-end sample, invert non-USD forwards, classify, exclude pegs, log prices | `fx_panel_clean.csv`, `ff_factors_clean.csv`, `excluded_pegged.csv`, `coverage_audit.csv`, `schema_mappings.json` |
| 2 | `src/returns.py` | Excess returns, spot changes, forward discounts, bid/ask inputs, RX index | `returns_panel.csv`, `rx_factor.csv` |
| 3 | `src/regressions.py` | Short-term reversal test (decides whether to skip the most recent month) | `table1_reversal`, `skip_month_decision` |
| 4 | `src/signals.py`, `src/portfolios.py`, `src/validate_section4.py` | Signal construction, portfolio sorts, frozen-cohort holding returns, machinery checks | — |
| 5 | `src/section5.py` | f × h grids for both signals, portfolio returns (All / DM / EM), rolling-OLS signal, seasonality, post-formation returns | `table_fxh_signal_A/B`, `fxh_series_A/B`, `table_portfolio_returns`, `table_ols_comparison`, `table_seasonality`, `figure1_post_holding.pdf` |
| 6 | `src/section6.py` | Transaction-cost-adjusted grid (100 / 75 / 50 % of quoted spread) | `table9_net_returns` |
| 7 | `src/section7.py` | Carry portfolio, carry–momentum correlations, 3 × 3 double sort | `table10_carry_correlations`, `table11_double_sort` |
| 8 | `src/section8.py` | Sharpe ratio grid | `table12_sharpe` |
| 9 | `src/section9.py` | Fama-MacBeth cross-sectional regressions | `table13_fama_macbeth` |
| 10 | `src/section10.py` | Factor regressions (univariate and four-factor) | `table15_factor_regressions` |
| 11 | `src/section11.py` | Volatility-scaled (TSMOM) signal grid | `tableA3_tsmom_grid`, `fxh_series_TSMOM` |

Shared helpers: `src/stats.py` (Newey-West statistics, significance stars) and `src/output_writer.py` (collects every table into one workbook).

---

## Outputs

All outputs are written to `output/`, which is not version-controlled.

| File | Contents |
|---|---|
| `fx_panel_clean.csv` | Monthly panel keyed on (`date`, `currency_code`): spot and forward mid/bid/ask levels, their logs, and `msci_class` |
| `ff_factors_clean.csv` | `date, mktrf, smb, hml, umd, rf`, 300 months |
| `excluded_pegged.csv` | Panel rows removed for HKD, SAR and AED |
| `coverage_audit.csv` | Per-currency status (`kept`, `spot_only_dropped`, `pegged_excluded`) with spot and forward month counts |
| `schema_mappings.json` | Raw-to-standard column mapping for every input file |
| `returns_panel.csv` | Per currency-month: excess return, spot change, forward discount, bid/ask inputs for net returns |
| `rx_factor.csv` | Equal-weighted dollar factor RX and number of currencies per month |
| `results.xlsx` | All result tables, one sheet each (sheet names in the pipeline map above) |
| `figure1_post_holding.pdf` | Cumulative 60-month post-formation returns for MOM(1,1), MOM(6,1), MOM(12,1) with 95 % bands |

### Reading the tables

- All returns are **log** returns. The f × h grids, net-return grid, OLS comparison, carry, double-sort and factor-regression α are **annualised × 100** (monthly mean × 1200). `table_portfolio_returns` and `table_seasonality` report **monthly** means × 100.
- t-statistics are Newey-West HAC with lag L = ⌈0.75 · T^(1/3)⌉ (Andrews, 1991), computed from each series' own length T. Table 1 also reports L = 12 and plain OLS t-statistics. Stars: \*\*\* p < .01, \*\* p < .05, \* p < .10.
- Sharpe ratios are annualised: √12 × mean / standard deviation of monthly returns.
- In the grids, MOM is the highest-signal portfolio minus the lowest-signal portfolio. `table_portfolio_returns` lists quintiles with P1 = winners and P5 = losers, so MOM there is P1 − P5.

---

## Implementation conventions

- **Quotes:** all rates are foreign currency per USD, and all arithmetic is in log prices. Excess return rx(t+1) = f(t) − s(t+1); spot change is sign-flipped so a positive value means the foreign currency appreciated.
- **Month-end sampling:** the last observed trading day of each month is used, dated to the last calendar day so it joins with the factor file.
- **Sample:** currencies with no forward quotes at all are dropped (31 remain). Zero or negative source rates are set to missing before taking logs. Lags are matched by calendar month, so returns are never computed across a gap in a currency's data.
- **Classification:** MSCI Developed / Emerging as of December 2024, applied to the whole sample. Currencies outside both groups (BHD, ISK, RON, RUB) appear only in the full-sample results.
- **Signals:** Signal A is the sum of the last f monthly excess returns, including the formation month. Signal B is the same sum of spot changes. The rolling-OLS signal uses lag weights estimated on an expanding window, with a 36-month initial window and only data before the formation month. The TSMOM signal divides Signal A by each currency's rolling 36-month return volatility.
- **Portfolio sorts:** each month, currencies with a complete signal are ranked. With 18 or more currencies they form six portfolios; with 15–17, five; with fewer, no portfolios are formed. Ties are broken alphabetically and leftover currencies go to the highest portfolios. Membership is frozen for the h-month holding period, and overlapping cohorts are equally weighted (Jegadeesh-Titman). The first h − 1 months of each strategy are discarded as burn-in.
- **Transaction costs:** half the quoted bid-ask spread is charged in log terms. The forward leg is charged every month a position is held, and the spot leg when the position is closed. The 75 % and 50 % scenarios scale the spread. Positions without the required bid/ask quotes are dropped rather than priced at mid.
- **Double sort:** currencies are split into forward-discount tertiles, then into momentum tertiles within each group.
- **VOL_FX:** the cross-sectional mean of absolute monthly log spot changes, with the AR(1) residual used as the volatility innovation.

---

## References

Methods:

- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817–854.
- Carhart, M. M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57–82.
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.
- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607–636.
- Goyal, A., & Saretto, A. (2009). Cross-section of option returns and volatility. *Journal of Financial Economics*, 94(2), 310–326.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.
- Jegadeesh, N., & Titman, S. (2001). Profitability of momentum strategies: An evaluation of alternative explanations. *Journal of Finance*, 56(2), 699–720.
- Lustig, H., Roussanov, N., & Verdelhan, A. (2011). Common risk factors in currency markets. *Review of Financial Studies*, 24(11), 3731–3777.
- Menkhoff, L., Sarno, L., Schmeling, M., & Schrimpf, A. (2012). Currency momentum strategies. *Journal of Financial Economics*, 106(3), 660–684. **(Replicated study.)**
- Menkhoff, L., Sarno, L., Schmeling, M., & Schrimpf, A. (2012a). Carry trades and global foreign exchange volatility. *Journal of Finance*, 67(2), 681–718. (VOL_FX factor.)
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228–250.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Novy-Marx, R. (2012). Is momentum really momentum? *Journal of Financial Economics*, 103(3), 429–453.

Data:

- FX spot and forward rates: LSEG Datastream, accessed through Wharton Research Data Services (WRDS).
- Equity factors (MKTRF, SMB, HML, UMD, RF): Kenneth R. French Data Library.
- Market classification: MSCI Market Classification (December 2024).
