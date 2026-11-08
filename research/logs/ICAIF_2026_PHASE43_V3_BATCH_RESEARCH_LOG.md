# ICAIF 2026 — Phase 43 v3: multi-batch retrospective CaMeL research

**Date:** 2026-10-11. **Status:** code complete; no real v3 CaMeL API calls executed. Official deployment: expressly prohibited until the user authorizes it.

## Question
Can structured CaMeL interpretation of historical, publicly filed issuer news improve stock selection or the four organizer metrics (M1 return, M2 Sharpe, M3 maximum drawdown, M4 turnover) **over a paired identical benchmark**?

## Evidence examined
User-provided `historical_camel_packet.json` includes two original SEC 8-K earnings exhibits filed/accepted July 14, 2026. JPM earnings presentation reports 2Q26 financial highlights, e.g. net income $21.2B and EPS $7.70 within its truncated excerpt. GS earnings-results exhibit has net revenues $20.338B, diluted EPS $20.98 and additional business data; the excerpt includes a large amount of style/color table clutter. Neither source contains verified contemporaneous sell-side analyst consensus. Filed historical XBRL values are filtered by filing date, but are **not** substitutes for historical analyst expectations. The original source availability time is modeled conservatively; observed data were downloaded in October 2026.

## Change from v2
- Added `icaif_phase43_camel_research_batch_v3.py`, leaving Phase 43 v2 intact.
- Import existing v2 SEC evidence packet without losing context.
- Query further **historical** SEC filings and Finnhub news using Phase 43 v2 data-retrieval safeguards, archive each distinct historical snapshot, maintain an append-only event library and flag conflicting later extracts instead of silently rewriting earlier evidence.
- Deterministic issuer-rotation and SEC-first selection, at most one case per ticker/day; no evaluation-period returns are consulted during case selection.
- Prepare up to eight research batches, with two cases by default and three maximum per batch. Previously attempted event IDs are excluded from any later prepared batch, even after importing additional archives. By default **only original SEC evidence** is eligible. Optional `--include-news` includes current-retrieved historical Finnhub summaries, flagged as possibly revised.
- Each batch requires a separate explicit `--probe-camel --batch-index N --ack-provider-credits` command. Maximum eight reserved requests, at most 17 KB request bytes and 1,700 output tokens each, deduplication by case-evidence fingerprint, no retries, and separate ledger preserving historical Phase 43 v2 reservations. No automated paid loops.
- Retain source excerpts, as-of SEC facts, model, prompt fingerprint, actual retrospective inference time, provider token usage (if returned), and CaMeL classification. Ground source quotes as exact original excerpt substrings.
- Offline event-outcome scorer: assign first permissible 7-round execution after simulated data access and one-hour hypothetical processing, and report 1- and 5-session next-close performance versus the contemporaneous equal-30 benchmark. Bearish signals are **not** interpreted as permission to short. Scores are diagnostics, not M1–M4. Full 105-round paired portfolio evaluation still necessary before adoption.

## Research-only limitations
- **Retrospective model contamination:** even without future facts in prompt, today's CaMeL model may have memorized historical events or outcomes. Historical performance cannot substitute for genuine forward inference.
- SEC original accession files are the strongest retrospective evidence, but historical market availability may predate/lag the modeled acceptance-plus-buffer timestamp; test conservatively.
- Finnhub headline/summary retrieved now may have been revised later; use original source archives where possible, and otherwise treat as lower-confidence.
- No verified historical analyst consensus is present in current two cases. Do not invent surprise-versus-consensus or forecast a numerical stock return from an adjective.
- Provider reports may round tiny costs to zero. **The 8-call cap is on requests, not dollars; actual billing balance must be checked independently.** At most eight additional requests with a fresh v3 ledger, versus older v2 ledgers.
- Trading fees (0.1% buy/sell notional) affect M1/M2/M3 via NAV, **but fee dollars are not the fourth metric; M4 is official average traded-notional/pre-execution-NAV across 105 rounds**.
- Event scoring uses Alpaca IEX proxy prices and an equal-30 stock comparator, not the official execution engine or complete 105-round portfolio valuation convention.

## Offline validation done
1. `python -m py_compile` PASS.
2. `--self-test` PASS: issuer/source eligibility, Finnhub-risk default exclusion, batch selection, prompt restrictions and evidence hashing.
3. Imported actual user-provided JPM/GS packet: `NEW 2 TOTAL 2` PASS.
4. Prepared actual packet: one two-case SEC batch, with distinct evidence fingerprint PASS.
5. Mocked, zero-credit provider request: two parsed `unknown` responses correctly stored, one reservation logged with fake token usage, and repeated-case prevention structural check PASS.
6. Offline scorer run over actual uploaded Alpaca IEX historical dataset through October 6: GS July 14 Round 6 and JPM July 15 Round 1; source cases aligned and one-day stock/market-relative returns produced PASS.

**No real v3 CaMeL call, Finnhub/SEC network request, organizer API call, trade or live server change was executed here.**

## Exact Windows workflow

From the starter-kit working directory with its virtual environment activated and with Phase 43 v3, Phase 43 v2, and Phase 23 all in Downloads:

```powershell
python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --self-test
python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --import-existing
python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --prepare --batch-size 2
python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --budget-status

# First ONE paid request analyzes the already-inspected July JPM+GS SEC exhibits.
python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --probe-camel --batch-index 1 --ack-provider-credits
python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --inspect
python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --budget-status

# Expand SEC-based evidence across issuers/dates; these are read-only public-source calls.
python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --collect --symbols "MSFT GOOGL META" --start 2026-07-15 --end 2026-07-31 --ack-readonly-api-requests
python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --collect --symbols "NVDA AMZN WMT" --start 2026-08-01 --end 2026-08-31 --ack-readonly-api-requests
python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --collect --symbols "LLY UNH JNJ" --start 2026-07-01 --end 2026-07-31 --ack-readonly-api-requests

# Regenerate manifest: the previously attempted JPM+GS events are automatically excluded.
python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --prepare --batch-size 2
python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --budget-status

# Each further batch requires its own explicit one-request command after inspecting the manifest.
# python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --probe-camel --batch-index 1 --ack-provider-credits

python "$HOME\Downloads\icaif_phase43_camel_research_batch_v3.py" --score --bars "artifacts\phase29_alpaca_refresh\alpaca_iex_15m_through_2026-10-09.csv"
```

**Send back:** `artifacts\phase43_camel_research_batch_v3\retrospective_predictions.json`, `retrospective_event_outcomes.csv`, `batch_manifest.json`, and optionally the console `--budget-status` output. Do not upload `.env`, token values or private API keys.

## Decision gate
After the historical call batch, inspect (a) source-grounding and abstention rate, (b) signs versus relative stock returns, (c) causal timing, and (d) sample size/confounding. If feasible, create one bounded **paired 105-round portfolio ablation** (core versus identical core plus 0.3pp CaMeL paper tilts), with fees in NAV and explicit M1–M4. Then select/freeze; do not prolong research indefinitely.
