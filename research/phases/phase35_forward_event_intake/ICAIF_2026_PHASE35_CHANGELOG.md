# ICAIF 2026 — Phase 35, 2026-10-11

## Scope and rationale
- Phase 33: static blended Core80 outperformed the tested adaptive overlays in July–August; every overlay held after initial allocation in September–October, so no validated event alpha.
- Goal: establish a genuinely forward-only, economical event intake for CaMeL/Laya; no retrospective LLM decisions, no uncalibrated trading weights.
- Reported research balances on Oct 11: Laya 9,000 remaining monthly input tokens; CaMeL USD 50. These are user-reported, not verified by provider API. Keep Official reserve.
- Official R1 deadline Oct 12 at 09:10 America/New_York (=15:10 Europe/Berlin CEST), contingent on current live schedule confirmation.

## Files
1. `icaif_phase35_forward_event_intake.py` (new; stdlib only). CLI: `--self-test`, `--status`, `--collect [--limit 30] [--news-days 2]`, `--prepare`.
2. `icaif_phase34_budgeted_event_overlay_shadow_v2.py` (reversible derivative of Phase34); enforces BOTH publication age and first-observed age <=120 minutes before any provider calls, adds `--validate-event FILE` read-only mode.

## Provenance and safety
- Finnhub **read-only collection** only on `--collect`, once per requested ticker; no automatic retries, stops additional calls after HTTP 429.
- Existing private FINNHUB_API_KEY loaded from `~/.icaif/research_sources.env` (or environment), never printed or persisted.
- Creates immutable per-run captures plus de-duplicated event index. The first observation timestamp never advances when the article is fetched again. Archives never claimed as historical feed.
- Event screening needs genuine issuer alias, material-event term, and BOTH publication and observation freshness; still requires human fact-checking.
- Model calls are **not part of Phase35**. Optional, explicitly credit-acknowledged Phase34 request is a separate command and research-only; do not run merely to consume credits.
- This does NOT implement Official trading, an autonomous live LLM integration, or alpha-validated portfolio tilts.

## Tests
- Phase35 offline unit tests PASS.
- Phase34 v2 offline unit tests PASS.
- Python syntax compilation PASS.
- End-to-end no-network synthetic test: fresh source => candidate JSON => Phase34 `--validate-event` PASS.
- End-to-end stale publication despite recent first observation => Phase35 abstention PASS.
- No real Finnhub/LLM API calls in this container. No actual trading or Official submissions.

## Next decision
1. Collect once for a fresh public event and inspect; if no fresh eligible news on Sunday, abstain.
2. Verify Hetzner Official watch and schedule before Mon Oct12 09:10 ET (15:10 CEST) decision deadline; do not promote any research-only module directly to Official.
3. Obtain actual forward event/price labels from new post-observation sessions before approving LLM tilts. Preserve budgets in the meantime.
