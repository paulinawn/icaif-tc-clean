# ICAIF 2026 — Phase 43 v2 evidence-quality correction (2026-10-11)

**Scope:** Historical research only. No Official submission, server operation, live order or paid model call was made by this correction.

## Observed from user's Phase 43 run

- SEC: JPM 2 eligible filings, GS 1; three original documents downloaded.
- Finnhub: 245 returned items per issuer (490 items before keyword/date filters), historical snapshot with 217 distinct events.
- Historical packet selected the JPM and GS 2026-07-14 8-K filings.
- One CaMeL request was reserved and completed: both `direction=unknown`, `event_type=other`, `strength=weak`.
- Event outcome CSV showed JPM -0.9056% 1 session, +0.6299% 5 sessions; GS +2.0502% 1 session, -3.7995% 5 sessions, measured from distinct hypothetical first eligible execution points. Neither classification generated a trade signal, so these are **not LLM strategy returns**.
- SEC XBRL CompanyFacts returned `RESPONSE_TOO_LARGE` because the previous client imposed a 6 MB response-size limit; this is a client retrieval limit, not evidence that SEC data is absent.

## Root-cause hypothesis

Prior `filing_excerpt()` blindly took the first 1,700 characters of the 8-K's primary filing. Earnings disclosures are often in Item 2.02 or an Exhibit 99.1 press release, which may not appear in that first excerpt. We **cannot verify whether that specifically happened for JPM/GS** without their uploaded source packet or fetched accession contents. Abstention is correct given inadequate evidence; do not treat the unknown as an inaccurate directional call.

## Changes in `icaif_phase43_historical_camel_replay_v2.py`

1. Prefer original SEC earnings-release exhibits with a filename indicating Exhibit 99.1, first via same-accession HTML links then via the SEC accession `index.json` listing. Prevent cross-domain and cross-accession links. Falls back to the filing's Item 2.02/7.01 passage only if it actually contains substantive financial evidence. If neither source has substantive information, skip the filing rather than send the cover sheet to CaMeL.
2. Use three individually fetched SEC `companyconcept` financial facts (net income and two revenue concepts), with original filed dates and as-of filters, rather than attempting a full huge current CompanyFacts JSON. An issuer may not publish all tags. This reduces response-size problems; it does not guarantee complete revenue coverage or consensus-expectation data.
3. Fix the context label `prior_filings` to exclude the very same accession being assessed.
4. New `--audit-evidence` (no API/model call) prints source type, evidence length, financial-evidence score, original source URL and an excerpt preview. Use it before authorizing the final permitted CaMeL request.
5. New scoring guard: if a newly prepared case contains different evidence under the same historical event ID, **old CaMeL classification cannot be silently scored against that new evidence**. You must first obtain a classification based on the new packet.
6. Original request ledger remains in the same output directory and its **two-request cap is unchanged**. The prior successful request counts as one used; the v2 script does not reset or refund it. No retries, no Laya.

## Tests actually executed

- Python syntax compilation PASS.
- Existing Phase 43 as-of/timestamp, quote and slot unit tests PASS.
- New SEC Exhibit 99.1 identification, cross-domain rejection, substantive financial excerpt detection and cover-sheet rejection unit tests PASS.
- Mocked end-to-end SEC submission + earnings exhibit + targeted SEC CompanyConcept + prepare + offline packet-audit PASS, including rejection of a later-filed 2026 fact from the July as-of context.
- Real SEC/Finnhub network, actual JPM/GS Exhibit 99.1 presence and CaMeL results on v2 were **not** tested here; those require the user's configured private providers and careful source inspection.

## Windows research-only commands

Run from activated starter-kit virtual environment:

```powershell
python "$HOME\Downloads\icaif_phase43_historical_camel_replay_v2.py" --self-test
python "$HOME\Downloads\icaif_phase43_historical_camel_replay_v2.py" --collect --symbols "JPM GS" --start 2026-07-01 --end 2026-07-15 --ack-readonly-api-requests
python "$HOME\Downloads\icaif_phase43_historical_camel_replay_v2.py" --prepare --max-cases 2
python "$HOME\Downloads\icaif_phase43_historical_camel_replay_v2.py" --audit-evidence
python "$HOME\Downloads\icaif_phase43_historical_camel_replay_v2.py" --budget-status
```

Do not call `--probe-camel` unless the packet actually contains substantive **historical original earnings-release text**, not filing boilerplate; verify expectations separately. Upload `historical_camel_packet.json` and the audit console output. This is research-only; no Official actions.
