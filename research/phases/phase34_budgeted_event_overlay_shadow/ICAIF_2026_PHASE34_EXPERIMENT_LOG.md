# ICAIF 2026 — Phase 34 event-conditioned LLM shadow overlay

Created 2026-10-11. Research-only, no competition API calls or trading.

## Evidence from Phase 33 (actual user-provided CSV reports)

- 2026 development July–August: 42 complete sessions. `core80_blend_hold` returned +4.5206%, versus +4.2185% for `core80_consensus_partial` and +4.0003% for `core80_consensus_daily`.
- Exposed September–October 9 diagnostic: 28 sessions, `core80_blend_hold` +0.2187%; every active core80 overlay policy returned identically to core80 HOLD because the trade gate accepted no additional trades.
- All 7,350 decision records audited: 23 accepted overlays across multiple candidate policies in development, zero in the exposed diagnostic, and zero after the first round. There is **no positive incremental alpha evidence** for the current active overlay.
- The artificial rank depends on the candidate set. Seven 15-minute valuation marks are not necessarily organizer fills. July–August and September–October each restart cash/NAV and do not constitute one continuous Official horizon.

## Phase 34 hypothesis

Targeted, newly observed company events extracted with CaMeL and, conditionally, Laya might add distinct information unavailable in lagged price features. This hypothesis is **unproven**. Do not retroactively apply a classification generated today to past trade decisions.

## Design and safety

1. Read a time-stamped source event (automatically prepared from a Phase18 snapshot, or a user-verified public event JSON).
2. Reject invalid timestamps and events first observed more than 120 minutes ago unless explicitly flagged as a stale negative-control experiment.
3. Call CaMeL at most once, **only after explicit credit-use acknowledgement**. Use Phase23's existing credential storage, allowlisted endpoint and validated JSON parser. No retries.
4. Only if CaMeL detects material company-specific news that is directional, not a recycled stock-price reaction, and does not require missing full text, request one Laya classification. Laya is scarce; otherwise save the call.
5. Independently validate issuer relevance and direction. Disagreement or weak evidence produces ABSTAIN.
6. Reconstruct the Phase33 80% blended price/industry core from the user's frozen October 9 Alpaca dataset. Generate a cash-neutral, capped 0.3-percentage-point **paper-only** issuer tilt, and the hypothetical 0.1%-fee cost.
7. Save the exact feature availability time. The resulting records have no calibrated return forecast and cannot be used in Official without another explicit promotion and compliance decision.
8. Persist a local one-call-per-provider reservation ledger **before** each provider request. This is not synchronized with the provider's real billing service, and previous Phase23/32 requests are unobserved.

## User-reported quotas, not provider-verifiable from these files

- Laya: **10,000 input tokens per month**. Phase34 allows at most one exploratory Laya request, and caps its request payload to 1,800 UTF-8 bytes. A byte count is a conservative research proxy, **not** actual token accounting or a guaranteed provider-token bound.
- CaMeL: approximately **$50 API credits** available originally. Actual remaining balance and per-request costs **unknown**. Phase34 allows at most one exploratory CaMeL request; no estimate substitutes for the provider's actual billing panel.
- Reserve the overwhelming majority for Official. In a 105-round competition, Laya should be triggered by unusual high-materiality events, not invoked seven times a day by default. Always ensure there is a no-LLM fallback, an error/timeout path, and a timestamped output ledger.

## Files and dependencies

- `icaif_phase34_budgeted_event_overlay_shadow.py` — standalone new research driver; requires `icaif_phase23_financial_llm_event_lab.py` in the same Downloads folder, and `icaif_phase30_2026_walkforward_agent_lab_v2.py` for shadow-core reconstruction.
- Existing `artifacts/phase29_alpaca_refresh/alpaca_iex_15m_through_2026-10-09.csv` is read-only.
- Existing `artifacts/public_data_sources/phase18_universe/phase18_snapshot_*.json` is read-only.
- New outputs in `artifacts/phase34_budgeted_event_overlay_shadow`. Provider keys remain outside the repository in `~/.icaif/llm_event_lab_keys.env` and are never recorded here.

## Validation status

- Python compile: PASS.
- Offline self-test: PASS (timestamp/age, abstention, risk limits, de-duplication, call-budget reservation).
- Synthetic full 2026-history core reconstruction and hypothetical cash-neutral tilt: PASS.
- Actual provider request: NOT RUN.
- Real-world predictive alpha: NOT ESTABLISHED.
- Official deployment integration or compliance verification: NOT DONE.

## Decision gate

Keep the feature in shadow mode pending actual forward event/return evidence, reliable provider usage information and testing against core80 blended HOLD on identical decisions. No market trading should depend on this script as delivered.
