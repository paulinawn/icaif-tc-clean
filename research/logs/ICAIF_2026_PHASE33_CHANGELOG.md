# ICAIF 2026 — Phase 33 experiment record (2026-10-11)

## Purpose
Test whether a **bounded, calibrated, fee-aware active overlay** improves a diversified 80% equity core against static 60% / 80% / industry-momentum / stock-momentum baselines. This is a deliberately specified, standalone research test; not a live trading strategy.

## Files
- New: `icaif_phase33_core80_selective_overlay.py`
- Read-only dependencies: `icaif_phase30_2026_walkforward_agent_lab_v2.py`, `icaif_phase31_integrated_alpha_ablation_v3.py`, and optionally `icaif_phase19_sec_ml_research.py` plus its 30 cached SEC CompanyFacts files.
- Input default: `artifacts/phase29_alpaca_refresh/alpaca_iex_15m_through_2026-10-09.csv`.
- Nothing is submitted, no network/LLM credentials loaded, and previous scripts/data are not overwritten.

## Hypotheses and structure
1. Core: invest 80% at the first simulated decision in a 50/50 blend of equal-weight 80% and industry-weighted 80%, then hold the core baseline. Compare also with standalone Equal60, Equal80, Industry80, Momentum80, core-blend HOLD.
2. Forecasts: reuse Phase31's monthly walk-forward, 1-day and 5-day relative-return forecasts and calibration, with point-in-time SEC approximation optional. Evaluate separate price, industry, price-industry consensus, and price-industry-SEC consensus ablations.
3. Overlay: cash-neutral stock weight shifts, at most 1.2% of portfolio per stock from initial core, aggregate L1 deviation at most 16% NAV. Total target invested exposure remains nominally 80% at proposal time (actual holdings drift if there is no trade).
4. Execution: last observable mark (prior close or completed preceding bar), fixed 0.1% buy/sell notional fees, incrementally priced risk penalty, forecast uncertainty penalty, 0.8%-NAV minimum proposed change, up to 8%-NAV nominal one-round reallocation, and expected incremental utility requirement above 0.0025% NAV. HOLD if not worthwhile. Daily/weekly/partial and exploratory intraday variants.
5. Observability: log **every** one of seven decision slots, including rejected proposals, accepted trades and reasons. Previous Phase31 diagnostics covered only accepted trades.

All numeric overlay constraints are **unvalidated design hypotheses**, explicitly not optimized or tuned against the exposed September–October segment. No promise of improving returns. Core exposure is fixed at nominal 80%, not a validated market-timing rule.

## Prior diagnostic findings motivating Phase33
- `phase30_incomplete_days.csv`: 36 excluded sessions: 32 in September–December 2025 and four in 2026. November 28 and December 24, 2025 have hundreds of missing standard 15m-bar slots, consistent with shortened market sessions. Real exchange calendar handling remains open.
- `phase31_trade_gate_diagnostics.csv`: 50 accepted simulated gate decisions across five strategy records (16 distinct trading dates), all at Round 1. Median expected gain ~0.000058 of NAV, fees ~0.000028, net utility ~0.000029. This audit file omits rejected proposals. Future intraday value is not demonstrated.

## Validation completed locally
- Syntax and Phase33 `--self-test`: PASS.
- End-to-end, market-shaped synthetic price data, all 13 no-SEC profiles across two windows, 7 rounds: PASS (26 results, 6,643 logged decisions).
- End-to-end synthetic price + 30-issuer synthetic SEC CompanyFacts data: PASS (30 results, 7,665 logged decisions).
- **No real Alpaca 2026 replay run in this environment.** Win/loss unknown until Windows `--run` results.

## Output and promotion conditions
Reports: `phase33_four_metric_comparison.csv`, `phase33_all_round_decisions.csv`, `phase33_decision_counts.csv`, `phase33_monthly_model_provenance.json`, `phase33_manifest.json`.

Retain/promote overlay only if it adds reproducible fee-net advantage over the same 80% core in multiple development regimes, with acceptable drawdown, turnover and robust data handling. Do not equate research peer-set ranks to Official competition ranks. September–October 2026 has already been examined and is not a clean holdout.

## Pending improvements, separate experiments
- Fix 15-minute market calendar awareness, preserve shortened sessions and selectively missing IEX quotes without manufacturing bars.
- Confirm corporate action adjustments and organizer fill vs IEX price differences.
- Expand genuinely time-stamped, permissible public SEC/news event archive. Use CaMeL/Laya for forward-only structured business-surprise extraction, directional agreement, contradiction detection and issuer relevance; **never backfill classifications to their publication timestamp**. No current validated LLM alpha or live model invocation in Phase33.
- If adaptive exposure is desired, validate a separate **market-level** expected return/regime forecaster rather than using cross-sectional scores to infer cash exposure.
