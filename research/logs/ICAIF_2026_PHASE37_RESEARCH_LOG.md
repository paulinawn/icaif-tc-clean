# ICAIF 2026 — Phase 37 Risk/Exposure Tournament

**Status:** Research-only. No Official submission, Hetzner deployment, scheduled task, market-data API or LLM call.

## Context and hypothesis

Phase 33 2026 research indicated that its 80%-equity core outperformed most active stock tilts. The accepted adaptive trades did not add robust alpha, and the Sep–Oct diagnostic period is already exposed to us. We therefore hypothesize that **controlling broad equity exposure and constructing a lower-volatility diversified core** may improve the four-metric outcome with fewer decisions and lower fees than repeated stock-return forecasts.

This is an economic hypothesis, not a claimed edge. Holding 60–80% equity may sacrifice bull-market gains. Market volatility/drawdown signals may react late, whipsaw and trigger fees. Raw IEX prices omit dividends and corporate actions. Competition ranks depend on *other teams*, which our research tournament cannot observe.

## Exact code and data

New script: `icaif_phase37_risk_regime_tournament.py` — depends only on `icaif_phase30_2026_walkforward_agent_lab_v2.py` beside it (normally both in Downloads).

Default read-only input: `artifacts/phase29_alpaca_refresh/alpaca_iex_15m_through_2026-10-09.csv` in the starter-kit working directory. Its exact SHA256 is recorded in output. No original input is modified.

Outputs, written to the starter-kit working directory:

- `artifacts/phase37_risk_regime_tournament/phase37_four_metric_tournament.csv`
- `artifacts/phase37_risk_regime_tournament/phase37_all_seven_round_decisions.csv`
- `artifacts/phase37_risk_regime_tournament/phase37_manifest.json`

## Predeclared ablations

Baselines: Equal60 HOLD, Equal80 HOLD, Industry80 HOLD, and Phase33's 50/50 Equal80+Industry80 blended HOLD (same definitions). Additional HOLD alternatives: 80% six-sector equal allocation with 40% inverse-20-day-volatility allocation within each sector, and a 50/50 blend of that low-volatility core with Phase33's blended core.

Adaptive alternatives: monthly low-volatility refresh; weekly stress guards (20-day/60-day aggregate realized volatility ratio > 1.25; weak 20-day trend < −2.5% and weak 60-day trend); their OR combination; sticky two-week confirmation; riskmix+joint guard; and a symmetric risk-on/off experiment including 90% equity under a strongly positive trend. Defensive exposure is 60%, baseline 80%. The remaining 40%, 20% or 10% is cash. Exposure changes are constrained by a minimum 10%-NAV gross planned turnover. Monthly stock composition changes require minimum 4%-NAV gross deviation. Weekly decisions are spaced five retained trading sessions apart. No new stock-return ML model is used here; Laya and CaMeL remain an independent *forward-only event experiment* without backdated classifications.

Features are calculated from close prices of trading sessions **completed before** the simulated decision. The only active rebalance window in this ablation is R1; R2–R7 are explicit HOLD decisions and are still included in the NAV and Sharpe calculations. Decisions cannot read the execution bar before trade; executed portfolio weights are applied at the following allowed-bar opening proxy. Long-only, max 30% per stock, sum weights ≤1, buy/sell notional 0.1% fees.

## Design and evaluation

- July–August 2026: development, reset $1m portfolio capital.
- September–October 9 2026: **already examined**, exposed diagnostic, separately reset $1m.
- July–October 9 2026: overlapping continuous no-reset sensitivity, **not an additional independent holdout**.
- Outputs show return, annualized seven-period Sharpe proxy, peak-to-trough drawdown at each period endpoint, turnover (mean per round as fraction of pre-execution NAV), fees and accepted trade count, plus mean-equity-exposure and risk-off session counts.
- Per-metric percentiles are among the **13 research policies**, not actual contestant standings. Favor pairwise absolute metric comparisons over minor percentile fluctuations.
- Keep a defensive or low-volatility variant only if incremental net benefit is persistent across segment boundaries, including stable or improved drawdown/Sharpe and acceptable turnover. Reject if profits depend only on the exposed diagnostic, if it fails to trade as designed, or if fee costs overwhelm the risk benefit.

## Tests executed before delivery

1. `--self-test`: PASS; zero-future-data invariance, exposure/weight constraints, seven-round accounting, fee total reconciliation, artificial volatility-shock de-risk and subsequent recovery.
2. Full synthetically generated 289-business-day 15-minute dataset, 30 stocks, 26 bars each: PASS, writing tournament and decision outputs through all three windows (synthetic market is **not evidence of real alpha**).
3. Independent Phase33 baseline replay against the exact same synthetic dataset: PASS, matching the **Equal60, Equal80, Industry80, blended-core** return/Sharpe/drawdown/turnover/fees to numerical precision (max return difference 0, max Sharpe difference ~8.3e-14).
4. No real Alpaca dataset in this analysis runtime; user must run on the previously refreshed local Windows data and share results. No prediction of resulting performance is made.

## Open limitations

- Properly account for holiday/half-day sessions and 36 incomplete historical sessions rather than assuming every day contains 26 bars. No future price filling.
- Adjust or independently source historical stock data for dividends, splits and spinoffs before making long-horizon total-return claims.
- Confirm organizer fill-price calibration, particularly opening execution gap, with externally verifiable source.
- Diagnose regime switch frequency and event timing; high-turnover exposure timing may underperform an unchanging core.
- Backtests sample few independent market regimes; statistical confidence cannot be supplied by pooling all correlated stock/round observations as if independent.
- LLM research can extract upcoming catalysts and actual surprises, but only timestamped **forward** event data can establish return predictive skill. Phase36's CaMeL preview correctly abstained from claiming a direct new Nvidia financial fact; this is an event-screening test, not realized alpha.
- **Explicit user directive:** No Official deployment, activation, scheduler changes or organizer submission until user authorizes it. This research phase makes no such calls or changes.

## Decision state

Phase37 software tests complete. Real 2026 result and keep/reject decision **pending user's Windows run**. Retain the Phase33 80%-blended core as a benchmark; no strategy has been promoted for Official.
