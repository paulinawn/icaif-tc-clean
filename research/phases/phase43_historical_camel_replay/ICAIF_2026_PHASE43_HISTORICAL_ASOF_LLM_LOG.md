# ICAIF 2026 Phase 43 — retrospective point-in-time Finnhub + SEC + CaMeL

Status: research-only implementation and offline tests passed on 2026-10-11. No live Finnhub/SEC or paid CaMeL calls made in this development container. No official trading, upload, credentials, or Hetzner changes.

## Why this experiment exists

Phase 39/42 restricted CaMeL to events freshly collected in the preceding eight hours. That was an appropriate constraint for the forward observation experiment but incorrectly prevented retrospective research. Historical issuer news and original filings can be backfilled as *data inputs*. A model call made today on historic data is a **counterfactual as-of analysis**, not an actual historical inference timestamp. Its predictive accuracy is provisional because model training may encode knowledge after the historical date, and news text retrieved today may have been edited since publication.

## Available components

New research-only file: `icaif_phase43_historical_camel_replay.py`.

1. `--collect`: GET Finnhub `company-news` with historical `from`/`to` dates (free-account access is not guaranteed), SEC original submission records, SEC reported XBRL facts and at most two original EDGAR accession documents per symbol. Limited to 5 symbols and a 32-day lookback window per collection. All SEC details retain filing accession and dates, news retains publisher/article date and actual retrieval time. Current SEC CompanyFacts are **filtered by each fact's filed date before** including them in a past as-of packet; do not use a fact merely because its financial-statement period-end predates the decision. If an SEC acceptance datetime lacks an explicit UTC offset, conservatively delay eligibility by two calendar days. If an SEC original accession fetch fails, do not fabricate the filing text.
2. `--prepare`: Select a maximum of three historical events with material source excerpts and supply only facts available before their modeled evidence cutoff. SEC primary documents are prioritized above retrospective Finnhub snippets. No network or provider calls.
3. `--probe-camel --ack-provider-credits`: One CaMeL `gemini-2.5-flash` batch request for at most three historical cases. Maximum two reserved requests across this Phase43 lab. No retries, source quotes are validated verbatim, results are appended to an audit JSON with ACTUAL computation time. Input is limited to 13,800 bytes. Laya unused. Keys remain in the preexisting private Phase23 key file. This is experimental as-of interpretation; **do not use results as historically observed model calls**.
4. `--score`: Map each model interpretation to the first organizer round whose *deadline* follows the modeled news delay plus a separate modeled 60-minute inference delay; price using Alpaca's next 15-minute bar opening. Report one- and five-session forward stock returns. Negative predictions are evaluated as directional correctness and avoided returns, never fictitious short-sale profits. Event returns are exploratory and are NOT official M1–M4; a complete portfolio replay must follow to decide whether the event classifier adds value.

## Explicit nonclaims and caveats

- Historical Finnhub availability on the user's plan has not been tested. Historical news summaries may have changed between original publication and retrieval. SEC original accession content provides stronger point-in-time evidence, but indexing/availability delays remain approximated. An as-of SEC filing or older reported XBRL fact does not establish analyst consensus or the difference from expectations. A news headline cannot justify a numeric return forecast without quantitative calibration.
- A present-day CaMeL model can have internal training knowledge of post-event news; prompt restrictions reduce but cannot eliminate this hindsight contamination.
- Calling CaMeL now does not establish that equivalent provider model, latency, or pricing was available on historical dates. Live use still requires current organizer compliance and explicit user approval.
- Test dates are already repeatedly used in development; historical results are exploratory only. A 2–6-event sample is a pipeline proof, not statistically sufficient evidence of alpha.
- M4 is actual average portfolio traded notional / pre-round NAV across 105 decisions; event prediction accuracy or research model costs are NOT M4. A final official evaluator-parity simulation must also count Round 7 through next day's R1, recording the intervening 16:00 close as an MDD-only observation.

## Completed offline checks

- Python compilation and `--self-test` PASS.
- As-of SEC XBRL restatement exclusion PASS.
- Finnhub modeled publication delay and competition deadline/round eligibility PASS.
- CaMeL typed-JSON and verbatim source-quote guard PASS.
- Synthetic historical SEC accession event through `--prepare` PASS.
- Historical stock return scoring using the uploaded original Alpaca CSV through October 6 PASS (test classification synthetic only; **no genuine LLM prediction was evaluated**).
- Bearish event scoring rejects fictitious short-trading-profit calculation PASS.

## Recommended Windows experiment

From starter-kit, with the existing venv activated and Phase23 script in Downloads:

```powershell
python "$HOME\Downloads\icaif_phase43_historical_camel_replay.py" --self-test
python "$HOME\Downloads\icaif_phase43_historical_camel_replay.py" --collect --symbols "JPM GS" --start 2026-07-01 --end 2026-07-15 --ack-readonly-api-requests
python "$HOME\Downloads\icaif_phase43_historical_camel_replay.py" --prepare --max-cases 2
python "$HOME\Downloads\icaif_phase43_historical_camel_replay.py" --budget-status
```

Inspect `artifacts\phase43_historical_camel_replay\historical_camel_packet.json` before choosing to spend CaMeL credits. If source excerpts contain genuine financial news from those dates, the optional single paid batch is:

```powershell
python "$HOME\Downloads\icaif_phase43_historical_camel_replay.py" --probe-camel --ack-provider-credits
python "$HOME\Downloads\icaif_phase43_historical_camel_replay.py" --score --bars "artifacts\phase29_alpaca_refresh\alpaca_iex_15m_through_2026-10-09.csv"
```

Upload `historical_camel_interpretations.json` (it holds PUBLIC source text and model outputs only, no credentials), `retrospective_event_outcomes.csv`, and `historical_camel_packet.json`; exclude all private credentials. If the provider denies past news access, `source_status` records the error and SEC original filings are still attempted. No automatic paid provider retry. These outputs will decide whether to include a limited CaMeL sleeve in the single remaining full 105-round M1–M4 finalist evaluation before model freeze.
