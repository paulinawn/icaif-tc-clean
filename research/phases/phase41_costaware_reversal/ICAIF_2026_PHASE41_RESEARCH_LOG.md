# ICAIF 2026 — Phase 41 Cost-Aware Reversal Research

**Prepared 11 October 2026. Research only.** No Official submission, server changes, Finnhub/SEC queries, CaMeL/Laya provider calls, or trades.

## Goal and provenance

- Implement 15 consecutive complete-session episodes, exactly seven decision deadlines per day, all 105 rounds and realistic 0.1% buy/sell notional fee accounting.
- Preserve existing Phase 39 static baselines, and compare an individual-stock, low-turnover reversal selector against those baselines and against cash-first/full rebalancing.
- Correct Stage 40's realized risk-cap deficiency. Do not force investment in stocks failing an absolute economic eligibility gate.
- Model last-five-session reversal using only information at the **prior completed session close**, and fit a **shrunk, historical five-session return predictor** on labels whose five-day outcomes have completed by that cutoff.
- Input market data: Alpaca IEX 15-minute raw stock price and volume; no corporate-action total-return normalization. False or missing 15-minute bars cause a complete session to be excluded by existing Phase 30 loader.
- Future verified CaMeL events can be optionally ingested as *shadow classification records only*. No model-generated returns are fabricated from those classifications, and no historic events are retrofitted.

**Implementation:** `icaif_phase41_costaware_reversal.py` SHA256 `d6296a19fe33889493d93f47b0c9acdfb1102b4c62a5dddedc46e13112446ef0`.

## Strategies

Static reference group: Equal60 HOLD, Equal80 HOLD, Blend80 HOLD, legacy low-vol80 HOLD.

Active group: three-session cash-first fixed-80, five-session cash-first fixed-80, five-session full fixed-80, five-session cash-first floating 70–90 target, five-session full floating, five-session floating with preceding-session opening-volume shock as risk gate, five-session floating with strictly optional R2 buy-only gap filter, and identical five-session floating with an event-observation-only CaMeL shadow branch.

**Holding count is emergent**, not fixed at 15/20/25/30. Names pass independent return-vs-cash gates; not all 30 are forced eligible. Target bands can be missed when the expected return is insufficient or the position limits make full investment impossible; every decision reports the shortfall. Planned single-name and manually grouped sector weights have buffers below 10% and 30% respectively to reduce execution-price breaches.

## Model construction, exact data timing

At R1, the latest observable stock price is the previous completed US session close. Predictor input: standardized `-0.70 * previous-one-day-log-return - 0.30 * previous-five-day-log-return`. The model estimates a cross-sectional coefficient using up to 90 *historical daily dates*, with five-day labels fully matured before or at the R1 prior-close cutoff; shrinks the coefficient by 35%. A similarly shrunk prior mean market return supplies a conditional intercept. This is a simple **research forecasting heuristic**, not a proven expected-return calibration.

An unheld stock only qualifies if estimated five-day return exceeds a modeled 0.2% **round-trip** commission allowance plus a fixed 0.06% and 3% of the stock's five-day realized volatility as additional uncertainty budget. Existing holdings may persist with a weaker hysteresis threshold, but normally cannot be voluntarily liquidated before three or five trading sessions. All trades are evaluated against the same historical actual timestamps. At R2 (10:25 ET decision, 10:30 ET execution) a specific optional branch can make **buy-only** additions on stocks with a sufficiently negative opening gap, positive net forecast, adequate cash and eligible risk capacity. Early volume used at R2 comes only from bars closing by 10:15 ET. At R1, the volume-based branch uses *previous session's* first-45-minute volume shock. R3–R7 are still logged, normally HOLD.

There are no direct historical Finnhub, SEC, CaMeL or Laya predictive inputs. The separate provider pipelines remain available to produce timestamped prospective signals for a subsequent research design.

## Audited software tests

- Python syntax compile: PASS.
- `--self-test`: PASS, including label maturity and poisoning data after forecast cutoff, 15 sessions × seven rounds, buy/sell fee conservation, R2 only using known early bars, R2 buy-only, variable stock count and broker-independent research-only execution.
- Direct end-to-end **real Alpaca Oct 6 file** research check: PASS; six episodes × 12 policies × 105 rounds = 7,560 logged decisions.
- Static Equal60, Equal80, Blend80 and legacy Lowvol80 results match uploaded Phase 40 episode metrics to floating point tolerance on net return, Sharpe, drawdown, turnover and fees.
- No recorded over-90%-equity, above-10%-name, or above-30%-sector breaches in this specific Oct 6 research check. Realized exposure can still drift between rounds; out-of-band events are logged and are not automatically declared legal trading decisions.

## **Preliminary negative result** — ONLY through October 6

Data file: `alpaca_iex_15m_400d.csv` SHA256 `77e70af43c515db342cf13ecfde213078a4fd4e7dcd77a41857e421e6ea5fcf6`.

The six episodes (three Apr–Jun, two Jul–Aug, one September) are already examined in this project; **not independent** research holdout. Results below are *mean of six per-episode metrics, after modeled fees*. Not official competition performance.

| Policy | Mean net 15-session return | Mean 15-session drawdown | Mean equity exposure | Mean holdings | Mean fees/episode |
|---|---:|---:|---:|---:|---:|
| Equal60 HOLD | +1.131% | 1.196% | 60.2% | 30.0 | $600 |
| Equal80 HOLD | +1.508% | 1.589% | 80.2% | 30.0 | $799 |
| Blended80 HOLD | +1.564% | 1.664% | 80.2% | 30.0 | $799 |
| Five-day reversal cash-first fixed80 target | **−0.321%** | 0.882% | **16.5%** | **2.0** | $458 |
| Five-day reversal cash-first floating70–90 target | **−0.317%** | 0.877% | **16.5%** | **1.9** | $472 |
| Five-day reversal plus prior volume risk gate | −0.304% | 0.901% | 15.0% | 1.7 | $463 |

Why this matters: a nominal “fixed 80%” *target* is not achieved when too few stocks clear the absolute gate. Such results must not be presented as a fair, matched 80%-versus-flexible-exposure performance comparison. The primary Stage 41 test is a **stress test of our signal's economic strength**. On the observed data, the strict return-after-fee/uncertainty criterion is so restrictive that it cannot construct a properly diversified 70–90% portfolio. The isolated stock bets lose money despite lower total fees.

Do **not** loosen thresholds to make these same episodes look good. The result argues against a *standalone* reversal-only selective portfolio. A later paired experiment should maintain a diversified baseline core and evaluate whether a **small, capped, evidence-gated signal sleeve** adds return after fees. It would explicitly test a new hypothesis, not retroactively re-label this candidate as successful.

## Known caveats

- Only six non-overlapping episodes; their month regimes have been repeatedly inspected during model development. No untouched holdout.
- All seven rounds are simulated and reported, but Stage 41's causal five-day reversal model is a **daily** forecast. R2 has one tightly gated optional add-on; R3–R7 correctly choose HOLD instead of inventing speculative intraday alpha.
- Prices are Alpaca IEX 15m proxies, not organizer quotes/fills; bid–ask spreads, slippage and corporate-action adjustments not fully modeled.
- Training labels span overlapping five-day future returns and are not independent; regression coefficients and Sharpe estimates are noisy.
- The selected sector grouping is manually defined in Phase 38 and may not reflect official industry classifications.
- Cash-first vs full rebalancing still incurs different real exposures when independent issuer eligibility is sparse. Match on exposure and portfolio concentration before attributing performance differences entirely to execution method.
- Exposure below 70% is expected when independent stock gates reject most candidates; this is explicitly **not** a successful 70–90% fully invested portfolio.
- No LLM alpha: the `__camel` shadow branch only records verified forward events and contains no fabricated numeric alpha; in the Oct 6 evaluation it is identical to its control.

## Windows PowerShell (next to your Phase 39/38/37/30 files)

```powershell
Set-Location "$HOME\Documents\ICAIF2026\2026ICAIF_Trading_Agent_Competition\starter-kit"
.\.venv\Scripts\Activate.ps1
python "$HOME\Downloads\icaif_phase41_costaware_reversal.py" --self-test
$LASTEXITCODE
python "$HOME\Downloads\icaif_phase41_costaware_reversal.py" --run --stride 15 2>&1 |
    Tee-Object "artifacts\handover_logs\phase41_console.txt"
$LASTEXITCODE
```

Expected success marker `PHASE41_RESEARCH_COMPLETE_NO_LLM_NO_OFFICIAL_SUBMISSION`, exit `0`.

Result folder: `artifacts/phase41_feeaware_reversal`. Analyze `phase41_episode_metrics.csv`, `phase41_policy_summary.csv`, `phase41_paired_summary.csv`, `phase41_all_105_round_decisions.csv`. The manifest contains the input file SHA256 and limitations.

**Experiment decision as of Oct 6 provisional data: REJECT standalone reversal-only stock selector**, keep the causal validation/risk system and test a small diversified-core overlay later. Do not infer any result for Oct 7–9 before the actual Windows replay.
