# ICAIF 2026 — Phase 40 research log (2026-10-11)

## Hard operating boundary
Research only. No Hetzner modifications, Official API calls, uploads, trades, Laya calls or CaMeL calls. User authorization required for any Official deployment. All historic LLM events must pass publication, first-observation, model-completion and manual-review as-of checks before their first eligible decision.

## Phase 39 findings used BEFORE building Phase 40
Uploaded v2 outputs: `phase39_paired_episode_deltas.csv`, `phase39_paired_summary.csv`, `phase39_policy_summary.csv`, `phase39_round_activity.csv`. The reports comprise **six non-overlapping 15-session episodes** (3 April–June, 2 July–August, 1 September), across all seven daily decision slots. Excluded sessions reduce coverage. Already examined samples, not pristine OOS.

Immediate-entry HOLD cores, average net per-episode return: 

| Core | Apr–Jun, 3 | Jul–Aug, 2 | September, 1 |
|---|---:|---:|---:|
| equal60_hold | +0.758% | +1.896% | +0.723% |
| equal80_hold | +1.010% | +2.528% | +0.964% |
| blend80_hold | +1.296% | +2.327% | +0.842% |
| lowvol80_legacy_hold | +0.795% | +2.764% | +0.635% |

Fixed 7-round staging versus identical immediate-entry core: Equal80 -0.04675 percentage points on average, positive in only 2/6 episodes; adaptive contrarian Equal80 -0.02923 pp and positive 2/6. For blend80 contrarian: -0.05724 pp and positive 1/6; lowvol80 contrarian: -0.02428 pp, positive 3/6. Not reliable. No historical CaMeL event trades; paired LLM branches duplicate controls. Strongest conclusion: do not force portfolio-wide first-day 30-to-80% staging. Explore separate issuer decisions and selective new deployment of cash instead.

## Phase 40 hypotheses, predeclared before checking real outcomes
1. Flexible equity budget `[0.70,0.90]` derived from *prior observed* 20-session equal-basket return and risk, rather than fixed 0.80, improves the 4-metric trade-off.
2. Each issuer earns its place from an independent lagged feature threshold; there is **no chosen target number of names**. It may be 13, 21, 27, 30 or other permissible counts. Turnover hysteresis differs for new buys vs existing holdings. A 10%-of-NAV single-name cap and 30%-of-NAV sector cap limit extreme concentration; these do indirectly limit the feasible minimum number of names, but never fix a target count.
3. Using cash to buy newly qualifying stocks, instead of automatically selling other qualifying stocks, improves net performance/turnover versus full rebalancing using the *identical* eligibility scores.
4. Predeclared signal families (5-/20-day relative momentum, 1-/5-day reversal, volatility-scaled momentum, blended relative strength/short-horizon pullback) and *completed-bar-only* intraday hybrid distinguish predictors at next-hour, same-day-close, five-session horizons. Measure Spearman rank IC descriptively across the 30-issuer universe.
5. Verified real CaMeL events can later be overlaid on the intraday hybrid as a strict forward-only paper event. In our pre-Oct11 price sample the paired branch MUST agree exactly with no-CaMeL variant.

## Program
File: `icaif_phase40_floating_cash_stock_selection.py`.
Dependencies stored alongside it: `icaif_phase39_105round_walkforward_v2.py`, `icaif_phase38_105slot_camel_staging.py`, `icaif_phase37_risk_regime_tournament.py`, `icaif_phase30_2026_walkforward_agent_lab_v2.py`.

Benchmark policies: equal60, equal80, blend80, legacy lowvol80 HOLD.
Dynamic policies: 5-day momentum cash-first floating; 1-day reversal cash-first floating; vol-adjusted momentum cash-first floating; hybrid cash-first fixed80; hybrid cash-first floating; hybrid full rebalance floating; hybrid intraday cash-first floating; identical hybrid intraday + causal CaMeL paper event.

Every policy evaluates 15 sessions × seven scheduled rounds = 105 decision slots; some policies abstain at R2–R7. Initial positions are purchased immediately, NO automatic first-day staging. Both buys and sells incur modeled 0.1% notional fees. Per-stock gates and hysteresis decide which shares are held or excluded, not a specified top 15/20/25/30. Cash-first orders use existing cash; sells are permitted for eligibility failure, cap excess or broad risk reduction. With too few eligible stocks the script reports the exposure shortfall instead of secretly buying rejected securities. **Actual exposure** can drift outside 70–90% due to prices, delays, liquidity or eligibility shortage and is audited. The range is a planning band, not a guaranteed hard bound at every timestamp.

Critical unverified assumptions: private IEX open prices approximate organizer fills; raw prices are not fully corporate-action-adjusted; missing sessions skipped; manual sector grouping; Sharpe is noisily annualized from 105 returns; the early 2026 results have also now been examined; signal thresholds were selected as engineering hypotheses, not statistically calibrated; fixed turnover gates can reject beneficial trades or churn.

## Test results performed here
- Python syntax compile: PASS.
- Offline self-test: PASS, including 105 round accounting, single-issuer variable-count eligibility check, as-of intraday safeguard, fee/cash conservation, and CaMeL/no-CaMeL parity when events absent.
- Full synthetic dataset rehearsal: PASS: 9 non-overlapping synthetic 15-day episodes × 12 policies × 105 slots = 11,340 round records; 1,476 exploratory factor-IC rows.
- The synthetic hybrid intraday variant held a **variable** 18–27 names across episodes with a changing actual equity fraction; this verifies design responsiveness, NOT historical alpha or any live forecast.
- Some synthetic decisions drifted about 0.02 percentage points above 90% due to fill/price movements. Exposure-band diagnostics explicitly record any occurrence beyond the audit tolerance; the candidate is not certified as hard-bounded for deployment.
- Real 2026 Alpaca dataset **not present in this container**, so NO real Phase 40 returns or new real-data signal IC have been calculated. The user can run the script on their local full 2026 raw IEX file, or upload it for direct inspection here.

## Reproducibility commands (Windows PowerShell)

```powershell
Set-Location "$HOME\Documents\ICAIF2026\2026ICAIF_Trading_Agent_Competition\starter-kit"
.\.venv\Scripts\Activate.ps1
python "$HOME\Downloads\icaif_phase40_floating_cash_stock_selection.py" --self-test
New-Item -ItemType Directory -Force "artifacts\handover_logs" | Out-Null
python "$HOME\Downloads\icaif_phase40_floating_cash_stock_selection.py" --run --stride 15 2>&1 | Tee-Object "artifacts\handover_logs\phase40_console.txt"
$LASTEXITCODE
```

Expected final success marker: `PHASE40_RESEARCH_COMPLETE_NO_LLM_NO_SUBMISSION`, code `0`.
Output folder `artifacts\phase40_floating_portfolio` includes per-episode four-metric outcomes, paired comparisons, per-round activity, 105-round action audit, feature IC, reproducibility manifest and key exposure/holdings statistics. Upload `phase40_policy_summary.csv`, `phase40_paired_summary.csv`, `phase40_feature_ic.csv`, `phase40_round_activity.csv`, plus `phase40_episode_metrics.csv` for paired robustness review.

## Promotion/rejection rules
Retain a feature only if it shows a consistent, economically plausible sign of **timestamp-clean** IC at the relevant horizon and the net portfolio version improves paired returns/Sharpe/drawdown/turnover without unacceptable exposure or concentration breaches. No optimizing thresholds on the same six previously examined episodes. A historical LLM branch with no eligible event is **unmeasured alpha**, not a 0% alpha estimate. Reject a dynamic variant if its after-fee four-metric trade-off is worse than the equal80 and matched fixed80 controls, or if it requires illegal/risky exposures. Require independent forward evidence before Official usage.
