# ICAIF 2026: Phase 44 final four-metric research decision

Date: October 11, 2026. **Research-only; NOT a live model or an Official submission.**

## Question

Does a flexible 5-day reversal model, a small reversal sleeve, or the available historical CaMeL earnings classifications materially improve the four organizer metrics over Equal80/Blended80?

## Design and evidence

- Historical raw Alpaca IEX 15-minute dataset ending October 6, 2026; 30 symbols; 240 complete days and 36 incomplete recorded sessions; individual IEX 15-minute open prices proxy organizer hourly execution prices.
- Exactly six nonoverlapping 15-session episodes (3 April-June, 2 July-August, 1 September); already examined multiple times during Phases 38-43. Not independent or blinded validation.
- 7 prespecified policies, 105 round returns each (4,410 round decisions). Results after 0.1% fee on total buy-plus-sell notional.
- M1 terminal NAV / initial NAV - 1; M2 sqrt(252*7) x mean/std of 105 returns; M3 high-water-mark decline including intermediate 16:00 closes (14 extra per episode); M4 mean over 105 of traded notional divided by preexecution NAV, initial purchases included.
- For each R7 except final, period ends immediately before the next day 09:30 rebalance. No extra Sharpe period at daily 16:00.
- The pre-existing dynamic 5-day reversal policy reproduces Phase 42 M1 and M4 *exactly*, while M2/M3 are re-evaluated using corrected overnight/close convention.

## Four metric results — mean across episodes

| Research candidate | M1 return % ↑ | M2 Sharpe ↑ | M3 MDD % ↓ | M4 avg per-round turnover % ↓ | CaMeL entry trades |
|---|---:|---:|---:|---:|---:|
| Equal80 HOLD | +1.508 | 3.333 | 1.596 | 0.761 | 0 |
| Blended80 HOLD | +1.564 | 3.453 | 1.664 | 0.761 | 0 |
| Dynamic reversal (full portfolio) | +1.780 | 3.103 | 1.585 | 1.263 | 0 |
| Blended80 + 15% reversal sleeve | +1.655 | 3.568 | 1.632 | 0.952 | 0 |
| Blended80 + GS CaMeL (illustrative) | +1.563 | 3.449 | 1.667 | 0.762 | 1 |
| Blended80 + reversal sleeve + GS CaMeL | +1.655 | 3.566 | 1.633 | 0.953 | 1 |
| Blended80 + GS/JPM repeat sensitivity | +1.563 | 3.448 | 1.668 | 0.763 | 2 |

These are means of six **separate** 15-session simulation scores, not a combined 90-day account result, official standings, or tournament ranks.

## Paired results, controlling initial date and cash reset

- **Blended80 + 15% reversal sleeve versus Blended80**: M1 wins 5/6; mean ΔM1 = +0.0908 percentage points; ΔM2 = +0.1150; ΔM3 = -0.0325 pp; ΔM4 = +0.1905 pp.
- **Dynamic reversal (full portfolio) versus Blended80**: M1 wins 3/6; mean ΔM1 = +0.2159 percentage points; ΔM2 = -0.3495; ΔM3 = -0.0788 pp; ΔM4 = +0.5020 pp.
- **Blended80 + GS CaMeL (illustrative) versus Blended80**: M1 wins 0/6; mean ΔM1 = -0.0006 percentage points; ΔM2 = -0.0034; ΔM3 = +0.0026 pp; ΔM4 = +0.0009 pp.

## Historical CaMeL provenance and timing

- Inputs are **actual SEC source excerpts** but **model classifications were made October 11, 2026**. This retrospective inference is vulnerable to model-training memory/hindsight; it is not an observation of a July live LLM.
- GS earnings source accepted 2026-07-14 **07:30:40 ET** (not 12:30 ET); 60 minutes hypothetical processing lag; first eligible simulated R1 at 09:10 ET deadline and 09:30 execution. Classification was positive/strong, analyst surprise unknown.
- JPM later earnings **presentation** source accepted July 14 **09:52:21 ET**; early release accepted **06:30:38 ET**. Main analysis excludes JPM as a likely duplicate disclosure. A separate sensitivity includes the 09:52 event at R3, clearly labeled repeat/unverified novelty.
- Positive classifications cause **cash-funded, long-only 0.3% of NAV buys** with 5-session holding period and subsequent sale if within the episode; result is a direct conditional trade ablation with one qualifying GS event. No shorting or forced final liquidation.
- The CaMeL policy is NOT driven by verified price-relative post-announcement upside or a genuine analyst-consensus earnings surprise. That missing input materially limits interpretation.
- No Finnhub or SEC fresh network calls, no new CaMeL calls, and no organizer requests were made by the Phase44 backtester.

## Decision and scope

1. **Production anchor: Blended80 HOLD.** A defensible 4-metric baseline with one initial rebalance and no dependency on Finnhub/SEC/LLM uptime. This is a research choice, not a claim it will beat other entrants.
2. **One active challenger: Blended80 + 15% 5-day reversal sleeve.** Attractive mean M1/M2/M3 but M4 is about 25% higher than Blended80; still vulnerable to overfitting, and holding count remains 30 in this particular core–satellite design. The full dynamic reversal alternative genuinely changes holdings and exposure but is less robust.
3. **Do not let unconditional CaMeL positive earnings summaries trade in Official.** The studied event worsened its own core; it lacked earnings-surprise calibration or an independent novelty check. Use the Finnhub–SEC–CaMeL pipeline for *shadow logging* only unless a future rule with stronger evidence passes a forward gate. Keeping CaMeL for extraction without any downstream decision does not create alpha or competition score.
4. **Fallback Equal80 HOLD** if the legacy risk/weighting data needed for Blended80 cannot be reproduced.
5. **End strategy expansion**. Remaining work is engineering: clean data ingestion, 30-weight validation, no-leakage guard, dry run, checkpoints, restart behavior, safe read-only schedule, credentials isolation and explicit authorization before anything Official.

## Files

- `icaif_phase44_final_four_metric_comparison.py` — reproducible offline replay (six local phase dependencies).
- `phase44_episode_M1_M4.csv` — all per-episode results.
- `phase44_policy_mean_M1_M4.csv` — four metric summary.
- `phase44_paired_M1_M4.csv` — paired episode deltas to Equal80 and Blended80.
- `phase44_all_105_rounds.csv` — each simulated round with NAV/return/traded notional and audit trail.
- `phase44_event_provenance.csv` — corrected SEC event times and novelty flags.
- `phase44_manifest.json` — hashes and limitations.

## Important known limitations

- The raw dataset stops October 6; users can run the same program against the refreshed October 9 CSV. This may not change the 15-day windows due segment boundaries.
- This price-only test includes only two historical CaMeL inferences, one counted under a permissive hypothetical event rule. An event-evidence extraction pipeline is not an integrated production news decision engine.
- Market liquidity, IEX quotes, slippage, delayed receipt, trading halts and early-close schedule changes are not fully simulated.
- The six episodes are not a genuinely unseen final holdout; performance can regress materially on the Official dates.
- Official ranking is mean of **ranks among competitors** for the four metrics, not a known weighted function of these metric values; no winning probability is estimated.
