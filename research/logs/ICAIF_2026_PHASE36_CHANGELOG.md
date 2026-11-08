# ICAIF 2026 — Phase 36: One-call CaMeL catalyst watch

Date: 2026-10-11. Research only; not deployed to Hetzner or organizer.

## Hypothesis

A low-cost LLM can transform an earnings-preview article into a structured issuer/catalyst watchlist without hallucinating an earnings result, a company-specific Nvidia surprise, an event timestamp, or a trading signal. This is a necessary quality control, not evidence of positive trading alpha.

## Script

`icaif_phase36_camel_earnings_watch_lab.py` (new; depends on the existing Phase 23 provider client). It consumes the existing Phase 35 `phase35_phase34_candidate_event.json` and a short public-source summary of the October 11 Investors.com preview. The script preserves the original first-observed timestamp and the separate time of inference.

`--self-test`, `--prepare`, `--budget-status`: offline and free. `--probe-camel --ack-provider-credits`: one CaMeL request, up to 550 response tokens, no retries and no Laya calls. It records a one-call reservation before network access and never reads competitor credentials or submits trades. No direct return forecast or portfolio weights are produced. A provider error may consume credits even if no report is created.

The user's verbally reported remaining budgets are 9,000 Laya input tokens/month and $50 CaMeL. The script cannot verify balances. It spends no Laya and caps its research CaMeL calls at one, with actual cost unknown until the provider dashboard shows it.

## Validation

- Python compile: PASS.
- `--self-test`: PASS, including schema enforcement, bogus past-report rejection, event identity check, and offline atomic report writing.
- Synthetic starter-kit directory `--prepare`: PASS, `--budget-status`: PASS.
- Live CaMeL call: **NOT PERFORMED**; needs user's private Windows key and explicit script invocation.
- Strategy alpha, Official trading, or production compatibility: **NOT TESTED**.

## Next gates

1. Read-only Official schedule and existing Hetzner service/timer inventory. No change to the demonstrated Validation automation.
2. If desired, run the one-call CaMeL issuer-attribution test and review its output. No Laya on a mere earnings preview.
3. For a *new* official earnings release or filed guidance update, construct a sourced event with pre-announcement expectations; request Laya only after issuer/materiality screening, and paper-track subsequent price changes. Prevent historical backfills.
