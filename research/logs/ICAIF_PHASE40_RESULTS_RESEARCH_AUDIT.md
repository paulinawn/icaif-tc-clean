# ICAIF 2026 — Stage 40 Results and Independent Raw-Data Audit

Research-only review, 11 October 2026. No Official submission, deployment, live LLM inference or paid API calls.

## Inputs and provenance

- `phase40_episode_metrics.csv`, `phase40_policy_summary.csv`, `phase40_paired_summary.csv`, `phase40_round_activity.csv`, `phase40_feature_ic.csv` as supplied by the user.
- `icaif_phase40_floating_cash_stock_selection.py` SHA256 `3be484d0efb638790aa5fcdd0bc106ba8079177ee69b5fcd2b514c20817de0cd`.
- Independently audited `alpaca_iex_15m_400d.csv` SHA256 `77e70af43c515db342cf13ecfde213078a4fd4e7dcd77a41857e421e6ea5fcf6`, 214,457 observations; 30 symbols; Sep 2, 2025–Oct 6, 2026. This is **not** the refreshed Oct 9 input used by the Stage 40 simulator. The raw audit is therefore limited to Oct 6.
- 276 distinct source dates; 240 dates have the complete 30-stock × 26 regular-session 15-minute price grid; 36 excluded by complete-day filter.
- Stage 40 results: six non-overlapping 15-session episodes, 105 rounds per episode and strategy. Three early episodes, two July–August episodes, one September episode. Sept–Oct is previously exposed research data, not an untouched test.

## Six-episode mean performance, per 15 sessions

| Policy | Net return | Sharpe proxy | Maximum drawdown | Fees, USD | Trade rounds | Average equity exposure | Average holdings |
|---|---:|---:|---:|---:|---:|---:|---:|
| Equal60 HOLD | 1.131% | 3.222 | 1.196% | 600 | 1.0 | 60.24% | 30.0 |
| Equal80 HOLD | 1.508% | 3.229 | 1.589% | 799 | 1.0 | 80.16% | 30.0 |
| Blend80 HOLD | 1.564% | 3.338 | 1.664% | 799 | 1.0 | 80.16% | 30.0 |
| Legacy lowvol80 HOLD | 1.425% | 3.130 | 1.532% | 799 | 1.0 | 80.16% | 30.0 |
| Momentum floating cash-first | 1.063% | 1.329 | 2.462% | 2,555 | 13.7 | 88.62% | 22.46 |
| Reversal floating cash-first | 1.426% | 2.269 | 1.917% | 4,836 | 15.0 | 88.83% | 24.34 |
| Hybrid floating cash-first | 1.229% | 1.748 | 2.339% | 2,367 | 14.0 | 88.67% | 22.64 |
| Hybrid floating full-rebalance | 1.278% | 1.907 | 2.069% | 3,813 | 15.0 | 84.71% | 22.57 |
| Hybrid floating intraday cash-first | 1.044% | 1.472 | 2.464% | 3,274 | 43.2 | 88.94% | 22.61 |

All results use the provided Stage 40 simulator's modeled 0.1% fees and noisy short-horizon annualized Sharpe estimates. Internal policy ranks should never be interpreted as actual organizer ranks.

- No tested dynamic strategy beats Equal80 HOLD across all six episodes; reversal outperforms July–August but deteriorates in September.
- Momentum floating beats Equal80 in 2/6 episodes; reversal in 3/6; hybrid floating in 1/6; hybrid full rebalancing in 2/6; intraday hybrid in 1/6.
- Cash-first vs full hybrid saves approximately $1,446 fees/episode, but the two executions have **different realized exposures** and therefore are not a controlled execution-only ablation.
- CaMeL twin produces identical historical results; zero contemporaneously eligible CaMeL classifications were present. This says nothing about potential forward CaMeL alpha.

## Implementation defects and model-design concerns

1. **Exposure control is not isolated.** The supposed `hybrid_cashfirst_fixed80` averages 86.97% equity exposure and exceeds 90% on 31/630 rounds. `hybrid_cashfirst_floating` averages 88.67% and exceeds the 90% nominal band on 140/630 rounds, maximum observed 90.24%. This must be fixed before comparing cash-first execution against a fixed or a true floating target.
2. **Relative eligibility is not absolute investment merit.** Each cross-section is z-scored to mean 0. A stock may pass the threshold purely because peers score worse. Eligibility thresholds -0.50 on new names and -0.85 on held names ensure no absolute forecast of net return versus cash. Floating holdings are real, but they do not yet represent independently calibrated positive opportunities.
3. **Daily churn conflicts with multi-day predictive horizon.** The most promising IC is at five sessions, yet reversal rebalances on every R1 and incurs approximately $4,836 of fees per 15-session episode. Test cost-aware 3–5-day holds, minimum confidence margin, and individual incremental trade gates rather than indiscriminate daily reselection.
4. **Intraday activity is economically detrimental as implemented.** Intraday hybrid generates 259 executed round decisions across six episodes, versus 84 for daily cash-first hybrid, but worse return, Sharpe, drawdown and fees.
5. **Risk constraints on requested weights are not identical to realized-portfolio constraints.** Verify realized equity band, sector caps, individual caps and price-gap behavior after every fill; existing runtime only tolerates modest overages on name and global equity cap and does not recheck realized sector concentration.
6. **The separate September episode has already been used repeatedly in research.** Do not optimize to win it, and do not claim statistical confidence from six episodes.

## Independent Alpaca IEX exploratory diagnostics

Constructed daily and intraday price matrices on the 240 complete sessions through October 6. Every signal is formed exclusively using prices available at the preceding session close (for R1) or 09:30–10:15 ET completed data for R2. All statistics below are diagnostic Spearman cross-sectional information coefficients (IC), not executable backtest returns. Five-session forward labels overlap and cannot be counted as independent trials.

### R1 price factors versus next five-session stock return

| Factor | Jan–Jun mean IC, 120 days | Jul–Oct 6 mean IC, 62 days | Jan–Jun sector-demeaned IC | Jul–Oct 6 sector-demeaned IC |
|---|---:|---:|---:|---:|
| Stage 40 reversal score, `z(-.70 * prior 1d return - .30 * prior 5d return)` | +0.0255 | +0.0602 | +0.0293 | +0.0425 |
| Stage 40 momentum score, `z(.70 * prior 5d return + .30 * prior 20d return)` | -0.0343 | -0.0027 | -0.0351 | -0.0065 |
| Pure 5-session reversal | +0.0392 | +0.0161 | +0.0430 | -0.0058 |

For the exact Stage 40 reversal score, the top 20% scored names exceed the **same-day all-stock mean forward five-session return**, on average, by +0.092 percentage points (Jan–Jun) and +0.150 percentage points (Jul–Oct 6), *gross of trading fees and portfolio constraints*. Those are weak edges relative to a possible 0.2% round-trip fee on traded notional. Outcomes vary by month; no stable profitable strategy is established.

### R2 signals: R2 decision before 10:25 ET, simulated fill at 10:30 ET

| Signal | Target | Jan–Jun IC | Jul–Oct 6 IC |
|---|---|---:|---:|
| Overnight-gap fade (prior close to 09:30 open) | 10:30–close stock return | +0.0200 | +0.0289 |
| Fade first 45 minutes of regular session (known by 10:15) | 10:30–close stock return | -0.0127 | +0.0188 |
| Surprising first 45-min IEX volume (vs previous 20 complete sessions) | Absolute next-hour stock return | +0.0795 | +0.0810 |
| Same early-volume shock | Absolute 10:30–close stock return | +0.0569 | +0.0974 |
| Same early-volume shock | Signed 10:30–close stock return | +0.0212 | -0.0155 |

The volume result is potentially useful as a **volatility/uncertainty indicator, not a directional alpha predictor**. Gap fade is small relative to turnover costs. The 2026 July–October data is not an independent holdout; these are discovery hypotheses only.

## Decisions

**Retain:** Equal60, Equal80 and Blended80 as reproducible fee-efficient controls, with legacy low-vol80 as a regime-sensitivity check. Retain variable holding-count and cash-first architecture only as engineering research.

**Reject current implementation:** generic daily momentum, daily hybrid, and all-hour intraday hybrid as candidate competition agents. Do not promote the existing reversal policy, although its lagged-price feature deserves further testing at an appropriate five-session horizon.

**Next controlled experiment:** (1) enforce and audit realized 70–90%/80% equity exposure; (2) classify individual investment eligibility using estimated return above cash **minus expected fee and uncertainty**, not rank z-scores alone; (3) compare identical desired holdings under cash-first vs full rebalance with matched exposure; (4) test 3–5-session minimum hold and trade gates for reversal; (5) examine opening relative-volume as a sizing/risk variable, not automatically as a buy signal; (6) keep CaMeL event signals strictly forward-timestamped. Predeclare thresholds, use chronology, test small candidate set and avoid optimizing to September.

No Official deployment/submission was made or authorized.
