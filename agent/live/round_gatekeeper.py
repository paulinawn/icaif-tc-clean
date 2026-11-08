#!/usr/bin/env python3
# Role: GATEKEEPER - decides whether the current round may act, and asks the strategy what to do.
"""ICAIF V37: review-driven organizer adapter. UNARMED unless a separate approved deploy.

No ``watch``. HOLD is a local no-upload action. Authoritative executed evidence
is necessary to move the strategy state. All real uploads are disabled in this
candidate; --dry-run wraps every organizer HTTP method except GET in a blocker.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone, timedelta

APP = Path('/opt/icaif2026')
V23 = APP/'official_staging_v23_unarmed'/'code'
V28 = APP/'official_v3932_release'/'code'  # versioned repaired refresher only
DATA = APP/'official_clean_bootstrap_v28_unarmed'/'data'  # original, verified cache
UTC = timezone.utc


class Stop(Exception):
    """Fail-closed sentinel: carries a machine-readable reason string."""


def require(test, reason):
    """Fail closed unless ``test`` holds.

    Args:
        test: condition that must evaluate to truthy.
        reason: failure reason string carried by the raised exception.

    Raises:
        Stop: when ``test`` is falsy.
    """
    if not test:
        raise Stop(reason)


def paths():
    """Return the data paths used by this adapter.

    Returns:
        tuple of three ``Path`` objects: the cache provenance manifest,
        the latest read-only snapshot JSON, and the Alpaca IEX raw 15m CSV.
    """
    return DATA/'cache_provenance.json', DATA/'latest_readonly_snapshot.json', DATA/'alpaca_iex_raw_15m.csv'



def verified_own_no_upload(own, rid, *, allow_missing_record):
    """GET-only round response: either no decisions or ONE synthetic MISSING row.

    Never infer an upload from the synthetic decision_id/submitted_at alone.
    True receipt/upload artifacts and nonmissing status still fail closed.
    Caller separately requires no execution, complete history and clean ledger.

    Args:
        own: organizer round payload as returned by ``session.round(rid)``.
        rid: round id the payload must describe.
        allow_missing_record: keyword-only flag permitting exactly one
            synthetic MISSING decision row.

    Returns:
        ``True`` when the round has no decisions or one verified MISSING row.

    Raises:
        Stop: on any execution, decision-list, or artifact inconsistency.
    """
    require(isinstance(own, dict) and own.get('execution') is None,
            'EARLIER_OR_CURRENT_EXECUTION_PRESENT')
    decisions = own.get('decisions')
    require(isinstance(decisions, list), 'OWN_DECISION_LIST_UNVERIFIED')
    if decisions == []:
        return True
    require(allow_missing_record and len(decisions) == 1,
            'OWN_ROUND_HAS_UNCERTAIN_DECISION')
    row = decisions[0]
    require(isinstance(row, dict) and row.get('round_id') == rid and
            row.get('status') == 'MISSING' and
            row.get('selection_status') in (None, 'UNKNOWN') and
            row.get('execution_status') in (None, 'HELD'),
            'OWN_MISSING_RECORD_NOT_CONFIRMED')
    require(all(row.get(key) in (None, '') for key in (
            'submission_id', 'submission_url', 'submission_key', 'uploaded_at',
            'receipt_id', 'receipt_status', 'upload_id', 'file_name', 'file_sha256',
            'asset_key')), 'OWN_MISSING_HAS_UPLOAD_ARTIFACT')
    require(row.get('validation_status') in (None, '', 'MISSING'),
            'OWN_MISSING_HAS_VALIDATION_RECEIPT')
    return True

def verified_completed_missing_hold(own, rid, ledger):
    """Validate a completed organizer MISSING round's zero-trade evidence.

    A completed organizer MISSING round may have a synthetic zero-trade
    execution and ledger record. Require all fields; never count it as a POST.
    No fallback for partial, inconsistent, unknown, or fee-bearing evidence.

    Args:
        own: organizer round payload as returned by ``session.round(rid)``.
        rid: round id the payload must describe.
        ledger: official ledger payload containing ``rounds`` and
            ``transactions`` lists.

    Returns:
        ``True`` when every check passes.

    Raises:
        Stop: on the first failed check, with a ``MISSING_HOLD_*`` reason.
    """
    from decimal import Decimal, InvalidOperation

    def exactly_zero(v):
        """Return True only for finite numeric zero values."""
        if type(v) not in (str, int, float, Decimal):
            return False
        try:
            amount = Decimal(str(v))
        except (InvalidOperation, ValueError):
            return False
        return amount.is_finite() and amount == 0

    require(isinstance(own, dict), 'MISSING_HOLD_OWN_INVALID')
    decisions = own.get('decisions')
    require(isinstance(decisions, list) and len(decisions) == 1,
            'MISSING_HOLD_DECISION_COUNT')
    decision = decisions[0]
    require(isinstance(decision, dict) and decision.get('round_id') == rid
            and decision.get('status') == 'MISSING'
            and decision.get('selection_status') in (None, 'UNKNOWN')
            and decision.get('execution_status') in (None, 'HELD'),
            'MISSING_HOLD_DECISION_STATUS')
    require(all(decision.get(k) in (None, '') for k in (
            'submission_id', 'submission_url', 'submission_key', 'uploaded_at',
            'receipt_id', 'receipt_status', 'upload_id', 'file_name', 'file_sha256',
            'asset_key')) and decision.get('validation_status') in (None, '', 'MISSING'),
            'MISSING_HOLD_UPLOAD_ARTIFACT')
    require(own.get('trades') == [], 'MISSING_HOLD_HAS_TRADES')
    execution = own.get('execution')
    require(isinstance(execution, dict) and execution.get('decision_status') == 'MISSING'
            and execution.get('round_id') == rid, 'MISSING_HOLD_EXECUTION_MISMATCH')
    require(exactly_zero(execution.get('fee')) and exactly_zero(execution.get('traded_notional')),
            'MISSING_HOLD_EXECUTION_NOT_ZERO')
    require(execution.get('cash_before') is not None and
            execution.get('cash_before') == execution.get('cash_after'),
            'MISSING_HOLD_EXECUTION_CASH_MOVED')
    require(isinstance(ledger, dict) and isinstance(ledger.get('rounds'), list),
            'MISSING_HOLD_LEDGER_INVALID')
    matches = [x for x in ledger['rounds']
               if isinstance(x, dict) and x.get('round_id') == rid]
    require(len(matches) == 1, 'MISSING_HOLD_LEDGER_COUNT')
    ledger_round = matches[0]
    require(ledger_round.get('decision_status') == 'MISSING'
            and exactly_zero(ledger_round.get('fee')) and exactly_zero(ledger_round.get('traded_notional')),
            'MISSING_HOLD_LEDGER_NOT_ZERO')
    require(ledger_round.get('cash_before') is not None and
            ledger_round.get('cash_before') == ledger_round.get('cash_after'),
            'MISSING_HOLD_LEDGER_CASH_MOVED')
    # When transaction history exists, none of the entries may be from this round.
    transactions = ledger.get('transactions')
    require(isinstance(transactions, list), 'MISSING_HOLD_TRANSACTIONS_UNVERIFIED')
    require(not any(not isinstance(t, dict) or t.get('round_id') == rid
                    for t in transactions), 'MISSING_HOLD_TRANSACTION_CONFLICT')
    return True


def live_evidence(session, machine, schedule, history, ledger, portfolio, row, now, state):
    """Mint a catch-up proof for the current round from authenticated GETs.

    Only this authenticated-GET collector may mint a catch-up proof.

    Every earlier round must be COMPLETED and match a strict, zero-trade
    MISSING/HELD accounting record, with no attributable upload. This intentionally
    excludes any ambiguous pending trade. Caller holds exclusive native lock.

    Args:
        session: authenticated organizer session used for per-round GETs.
        machine: state machine module (unused here; kept for signature parity).
        schedule: organizer schedule payload with a ``rounds`` list.
        history: paginated decision history with ``decisions`` and
            ``has_more``.
        ledger: official ledger payload.
        portfolio: organizer portfolio payload for the official phase.
        row: current round row with ``id``, ``day`` and ``number``.
        now: current server time for the proof timestamp.
        state: journal state dict with ``mode`` and
            ``last_executed_round_id``.

    Returns:
        dict proof payload keyed by mode/date.

    Raises:
        Stop: on the first failed verification, or when no mode/date rule
            permits catch-up (``CATCHUP_NOT_PERMITTED_FOR_MODE_DATE``).
    """
    day = row['day']
    round_number = row['number']
    require(round_number >= 2, 'CATCHUP_NOT_R2_OR_LATER')
    require(history['has_more'] is False, 'INCOMPLETE_DECISION_HISTORY')
    require(isinstance(ledger.get('rounds'), list), 'LEDGER_UNVERIFIED')
    expected = [r for r in schedule['rounds']
                if r.get('phase') == 'official' and r.get('day') == day
                and type(r.get('number')) is int and r['number'] < round_number]
    require(len(expected) == round_number - 1, 'MISSING_EARLIER_ROUNDS')
    for prior in expected:
        require(prior['status'] == 'COMPLETED', 'EARLIER_ROUND_NOT_FINALIZED')
        own = session.round(prior['id'])
        verified_completed_missing_hold(own, prior['id'], ledger)
        require(all(d.get('round_id') != prior['id'] for d in history['decisions']),
                'EARLIER_ROUND_HISTORY_CONFLICT')
    own_current = session.round(row['id'])
    verified_own_no_upload(own_current, row['id'], allow_missing_record=False)
    require(all(d.get('round_id') != row['id'] for d in history['decisions']),
            'CURRENT_ROUND_IN_HISTORY')
    require(not str(state.get('last_executed_round_id') or '').startswith(f'official-{day}-r'),
            'SAME_DAY_EXECUTION_ALREADY_RECORDED')
    base = {'source': 'AUTHENTICATED_ORGANIZER_GET',
            'round_id': row['id'], 'checked_utc': now.isoformat()}
    if state['mode'] == 'PRE_ENTRY' and day == '2026-10-12':
        import organizer_bridge as bridge
        from sleeve_engine import SYMBOLS
        bridge.cash_only_portfolio(portfolio, SYMBOLS)
        return {'preentry_reconciliation': dict(base, earlier_rounds_finalized=True,
            earlier_rounds_no_executed_trade=True, current_round_no_attributable_attempt=True,
            verified_cash_portfolio=True)}
    if state['mode'] == 'PRIMARY' and day in ('2026-10-19', '2026-10-26'):
        require(state['first_execution_confirmed'] is True, 'NO_CONFIRMED_PRIMARY_HOLDINGS')
        import organizer_bridge as bridge
        from sleeve_engine import SYMBOLS
        bridge.normalize_portfolio(portfolio, SYMBOLS, state)
        require(bool(state.get('last_executed_round_id')) and
                not state['last_executed_round_id'].startswith(f'official-{day}-r'),
                'SAME_DAY_PRIMARY_ALREADY_EXECUTED')
        return {'rebalance_reconciliation': dict(base, prior_rounds_finalized=True,
            no_earlier_executed_primary_trade=True, current_round_no_attributable_attempt=True,
            portfolio_reconciled=True, single_writer_lock_held=True)}
    raise Stop('CATCHUP_NOT_PERMITTED_FOR_MODE_DATE')


def organizer_held_placeholder(row, completed_round_ids):
    """Classify an organizer-generated MISSING/HELD row as an ignorable placeholder.

    Do not confuse organizer-generated MISSING/HELD rows with submissions.

    A candidate is ignored ONLY when all three exact status fields describe a
    missing trade, the organizer schedule marks that round completed, and there
    is no submission/upload/receipt artifact. Any unexpected shape fails closed.
    The caller has already exhaustively retrieved all authenticated pages.

    Args:
        row: one decision-history row.
        completed_round_ids: set of round ids the organizer marks COMPLETED.

    Returns:
        ``True`` when the row is a confirmed organizer-held placeholder,
        ``False`` when it is material evidence.

    Raises:
        Stop: when the row is partially but not fully consistent with a
            placeholder shape (fail-closed behavior).
    """
    if row.get('status') != 'MISSING' or row.get('execution_status') not in (None, 'HELD'):
        return False
    if row.get('selection_status') not in (None, 'UNKNOWN'):
        return False
    require(row.get('round_id') in completed_round_ids,
            'MISSING_HELD_ROW_NOT_COMPLETED')
    # Organizer-generated MISSING/HELD summaries populate `submitted_at`
    # even when no team uploaded anything (observed in all 34 completed
    # no-submission rows via authenticated GET on 2026-10-19). It is NOT
    # sufficient evidence of an upload. The separately checked actual
    # receipt/upload fields below remain mandatory evidence gates.
    # A row with any inconsistent status remains material and fails closed.
    artifact_fields = (
        'submission_id', 'submission_url', 'submission_key',
        'uploaded_at', 'receipt_id', 'receipt_status',
        'upload_id', 'file_name', 'file_sha256', 'asset_key',
    )
    require(all(row.get(key) in (None, '') for key in artifact_fields),
            'MISSING_HELD_ROW_HAS_SUBMISSION_ARTIFACT')
    # Catch-up separately verifies zero-trade synthetic execution and ledger.
    # Inconsistent or positive accounting evidence continues to fail closed.
    require(row.get('validation_status') in (None, '', 'MISSING'),
            'MISSING_ROW_HAS_VALIDATION_RECEIPT')
    return True


def read_pages(session, machine, schedule):
    """Fetch the full official decision history and drop held placeholders.

    Args:
        session: authenticated organizer session with ``decisions``.
        machine: state machine module providing
            ``complete_history_pages``.
        schedule: organizer schedule payload used to identify completed
            official rounds.

    Returns:
        dict with ``decisions`` (material rows only) and ``has_more`` False.

    Raises:
        Stop: on malformed or incomplete history pages or an unverifiable
            schedule.
    """
    pages = []
    offset = 0
    for _ in range(100):
        part = session.decisions('official', offset=offset, limit=200)
        require(isinstance(part, dict) and isinstance(part.get('decisions'), list),
                'BAD_HISTORY_PAGE')
        pages.append(part)
        offset += len(part['decisions'])
        if part.get('has_more') is False:
            break
        require(part.get('has_more') is True and len(part['decisions']) == 200,
                'INCOMPLETE_HISTORY_PAGE')
    records = machine.complete_history_pages(pages)
    require(isinstance(schedule, dict) and isinstance(schedule.get('rounds'), list),
            'MISSING_AUTHENTICATED_SCHEDULE_FOR_HISTORY')
    completed = {r.get('id') for r in schedule['rounds']
                 if isinstance(r, dict) and r.get('phase') == 'official'
                 and r.get('status') == 'COMPLETED'}
    material = []
    ignored = 0
    for row in records:
        if organizer_held_placeholder(row, completed):
            ignored += 1
        else:
            material.append(row)
    # Keep this observable without ever printing payloads or credentials.
    print('V392_HISTORY_RECORDS', len(records), 'MATERIAL', len(material),
          'HELD_PLACEHOLDERS', ignored, flush=True)
    return {'decisions': material, 'has_more': False}


def collect(session, machine):
    """Gather schedule, portfolio, decision history, and ledger in one pass.

    Args:
        session: authenticated organizer session.
        machine: state machine module (unused here; kept for signature parity).

    Returns:
        tuple ``(schedule, portfolio, history, ledger)``.

    Raises:
        Stop: when the ledger is not a valid official ledger.
    """
    schedule = session.schedule()
    portfolio = session.portfolio('official')
    history = read_pages(session, machine, schedule)
    ledger = session.ledger('official')
    require(isinstance(ledger, dict) and ledger.get('phase') == 'official' and
            isinstance(ledger.get('rounds'), list), 'BAD_LEDGER')
    return schedule, portfolio, history, ledger


def load_snapshot(prior):
    """Load and validate the cached read-only snapshot.

    Args:
        prior: rehearsal module providing ``validate_snapshot_meta``.

    Returns:
        parsed snapshot dict.

    Raises:
        Stop: propagated from ``prior.validate_snapshot_meta`` when the
            snapshot fails verification; file/parsing errors propagate as
            their native exceptions.
    """
    meta_path, snap_path, csv_path = paths()
    manifest = json.loads(meta_path.read_text(encoding='utf-8'))
    snap = json.loads(snap_path.read_text(encoding='utf-8'))
    prior.validate_snapshot_meta(snap, manifest, csv_path)
    return snap


def inspect_pending(journal, history):
    """List intent round ids still awaiting confirmed execution.

    No subsequent allocation until all existing intents are fully executed.

    Args:
        journal: journal dict with ``intents`` and
            ``execution_confirmations``.
        history: decision history payload (accepted for signature parity;
            unused here).

    Returns:
        sorted list of pending round ids, or ``[]`` when none remain.
    """
    pending = {rid for rid, rec in journal['intents'].items()
               if rec.get('status') != 'ABANDONED_NO_POST'} - set(journal['execution_confirmations'])
    if pending:
        return sorted(pending)
    return []


def reconcile(session, journal, schedule, portfolio, history, ledger, machine, engine):
    """Reconcile journaled intents against authenticated organizer evidence.

    Args:
        session: authenticated organizer session used for per-round GETs.
        journal: journal dict mutated via returned updates.
        schedule: organizer schedule payload (unused here; kept for signature parity).
        portfolio: organizer portfolio payload forwarded to execution apply.
        history: decision history payload.
        ledger: official ledger payload.
        machine: state machine module providing journal helpers.
        engine: sleeve engine module providing ``SYMBOLS``.

    Returns:
        updated journal dict.

    Raises:
        Stop: on unjournaled decisions or late-appearing abandoned posts.
    """
    machine.enforce_no_unjournaled_decisions(journal, history['decisions'])
    for item in history['decisions']:
        rec = journal['intents'].get(item['round_id'])
        require(rec is not None and rec.get('status') != 'ABANDONED_NO_POST',
                'LATE_APPEARING_ABANDONED_POST_CONFLICT')
    for rid in list(journal['intents']):
        if journal['intents'][rid].get('status') == 'ABANDONED_NO_POST':
            continue
        own = session.round(rid)
        updated, changed = machine.apply_confirmed_execution(
            journal, rid, own, ledger, history['decisions'], portfolio, engine.SYMBOLS, engine)
        journal = updated
        if changed:
            print('V37_CONFIRMED_EXECUTION_ONCE', rid, flush=True)
    return journal


def choose(schedule, portfolio, history, ledger, own, snap, journal,
           bridge, engine, machine, now, *, session=None):
    """Ask the strategy what to do for the current round.

    Pure selection except authenticated catch-up GETs via injected session.

    Args:
        schedule: organizer schedule payload.
        portfolio: organizer portfolio payload.
        history: decision history payload.
        ledger: official ledger payload.
        own: current round payload from the organizer.
        snap: validated snapshot used by the engine.
        journal: journal dict with ``state``, ``intents`` and
            ``execution_confirmations``.
        bridge: organizer bridge module.
        engine: sleeve engine module.
        machine: state machine module.
        now: local time used for round authentication.
        session: optional authenticated session required for catch-up
            rounds (number > 1).

    Returns:
        tuple ``(row, result)`` of the authenticated round row and the
        engine's evaluation result.

    Raises:
        Stop: on pending executions, missing catch-up collector, bad
            engine output, or unexpected engine actions.
    """
    row, server_time = bridge.authenticated_round(schedule, own, engine.SYMBOLS, local_time=now)
    machine.enforce_no_unjournaled_decisions(journal, history['decisions'])
    require(not inspect_pending(journal, history), 'PENDING_EXECUTION_BLOCKS_NEW_DECISION')
    state = journal['state']
    normalized = bridge.normalize_portfolio(portfolio, engine.SYMBOLS, state)
    observation = {'phase': 'official', 'round': row, 'symbols': list(engine.SYMBOLS),
                   'as_of': server_time.isoformat(), 'portfolio': normalized, 'round_state': own}
    if row['number'] > 1:
        require(session is not None, 'NO_AUTHENTICATED_CATCHUP_COLLECTOR')
        observation.update(live_evidence(session, machine, schedule, history, ledger, portfolio, row, server_time, state))
    result = engine.evaluate(observation, snap, state, now=server_time)
    require(isinstance(result, dict) and isinstance(result.get('next_state'), dict), 'BAD_ENGINE_OUTPUT')
    engine.state_valid(result['next_state'])
    if result['action'] == 'HOLD':
        require(result.get('target') is None, 'HOLD_WITH_TARGET')
        return row, result
    require(result['action'] in ('PRIMARY_TARGET', 'EQUAL80_RECOVERY_TARGET'), 'UNEXPECTED_ACTION')
    machine.canonical_target(result['target'], engine.SYMBOLS)
    return row, result


def candidate_round(schedule, engine):
    """Return the current official round row when it is actionable.

    Args:
        schedule: organizer schedule payload with ``current_round``.
        engine: sleeve engine module providing ``OFFICIAL_DAYS``.

    Returns:
        the current round dict, or ``None`` when no valid official round
        of number 1-7 is active.
    """
    row = schedule.get('current_round')
    if not isinstance(row, dict) or row.get('phase') != 'official' or row.get('day') not in engine.OFFICIAL_DAYS:
        return None
    if type(row.get('number')) is not int or row['number'] not in range(1, 8):
        return None
    return row


def eligible_for_market_refresh(row, state):
    """Decide whether the current round may trigger an Alpaca refresh.

    No wasted Alpaca request when this round can only HOLD.

    Args:
        row: current round dict with ``day`` and ``number``, or ``None``.
        state: journal state dict with ``mode`` and
            ``last_executed_round_id``.

    Returns:
        ``True`` when a market refresh could change the outcome.
    """
    if row is None:
        return False
    # The authenticated cycle has already reconciled every material execution
    # into its durable, validated journal. Once a primary rebalance has actually
    # EXECUTED today, later rounds are not catch-up candidates; HOLD locally.
    # This deliberately does not modify R1, or a later round after failed R1.
    if row['number'] > 1 and state.get('mode') == 'PRIMARY':
        executed = state.get('last_executed_round_id')
        if isinstance(executed, str) and executed.startswith(f"official-{row['day']}-r"):
            return False
    if row['number'] == 1:
        return state['mode'] not in ('EQUAL80_LOCKED',)
    if row['number'] > 1 and row['day'] == '2026-10-12' and state['mode'] == 'PRE_ENTRY':
        return True
    if row['number'] > 1 and row['day'] in ('2026-10-19', '2026-10-26') and state['mode'] == 'PRIMARY':
        return True
    return False


def decision_payload(session, weights, row, engine):
    """Build the official decision submission payload (never uploaded here).

    Args:
        session: authenticated organizer session used for team credentials.
        weights: target weights dict keyed by symbol.
        row: current round dict providing ``id``.
        engine: sleeve engine module providing ``SYMBOLS``.

    Returns:
        payload dict with submission type, team credentials, phase, round
        id, and float weights for every engine symbol.
    """
    from kit.contracts import validate_payload
    values = {s: float(weights[s]) for s in engine.SYMBOLS}
    payload = {'submission_type': 'decision', 'team_id': session._team()['team_id'],
               'team_token': session._team()['team_token'], 'phase': 'official',
               'round_id': row['id'], 'weights': values}
    # The signed live organizer schedule is passed to native _validated later.
    return payload


def run_readonly(*, force_now=None):
    """Run the unarmed read-only rehearsal cycle.

    Collects organizer evidence with a GET-only transport, evaluates the
    current official round, and prints the would-be action. No organizer
    HTTP method other than GET is ever attempted and no submission occurs.

    Args:
        force_now: accepted for signature parity; currently unused.

    Raises:
        Stop: on any failed verification; the exception propagates to
            ``main`` which exits with code 2.
    """
    import httpx
    sys.path[:0] = [str(APP), str(V23), str(V28)]
    import organizer_bridge as bridge
    import readonly_rehearsal as prior
    import state_machine as machine
    import sleeve_engine as engine
    import market_data_fetcher as refresh
    from kit.config import load_environment
    from kit.original_client import OriginalSession, load_profile

    require(V23.is_dir() and V28.is_dir(), 'STAGES_MISSING')
    load_environment(APP/'.env')
    profile = Path(os.environ.get('ICAIF_PROFILE', ''))
    if not profile.is_absolute():
        profile = APP/profile
    token = os.environ.get('CODABENCH_TOKEN', '')
    require(bool(token), 'NO_CODABENCH_TOKEN')
    cfg = load_profile(profile)
    # A disposable journal and checkpoint ensure this rehearsal cannot alter
    # any future persistent Official state. GET-only transport protects both
    # the organizer API and the signed upload storage endpoint.
    with tempfile.TemporaryDirectory(prefix='icaif_v37_readonly_') as temp_root:
        scratch = Path(temp_root)
        get_only = prior.GetOnlyTransport(httpx.HTTPTransport())
        with OriginalSession(profile=cfg, token=token, checkpoint=scratch/'checkpoint.json',
             credentials=APP/'.icaif'/'credentials.json', transport=get_only, result_timeout=10,
             timeout=18) as session:
            schedule, portfolio, history, ledger = collect(session, machine)
            require(not history['decisions'] and not ledger['rounds'] and
                    not portfolio.get('positions'), 'READONLY_INIT_NO_CASH_ACCOUNT')
            journal = machine.blank_journal(engine.sample_state())
            row = candidate_round(schedule, engine)
            if row is None:
                print('V37_DRYRUN_HOLD NO_ACTIVE_OFFICIAL_ROUND', flush=True)
                return
            bridge.authenticated_round(schedule, session.round(row['id']), engine.SYMBOLS,
                local_time=datetime.now(UTC))
            require(eligible_for_market_refresh(row, journal['state']), 'NO_ELIGIBLE_MARKET_REFRESH')
            refresh.run('refresh', now=datetime.now(UTC))
            schedule, portfolio, history, ledger = collect(session, machine)
            row2 = candidate_round(schedule, engine)
            require(row2 is not None and row2['id'] == row['id'], 'ROUND_CHANGED_DURING_REFRESH')
            require(not history['decisions'] and not ledger['rounds'] and not portfolio.get('positions'),
                    'OFFICIAL_ACCOUNT_CHANGED_DURING_REFRESH')
            snapshot = load_snapshot(prior)
            own = session.round(row['id'])
            row, result = choose(schedule, portfolio, history, ledger, own, snapshot, journal,
                              bridge, engine, machine, datetime.now(UTC), session=session)
            print('V37_DRYRUN_ACTION', result['action'], flush=True)
            print('V37_DRYRUN_REASON', result['note'], flush=True)
            if result['action'] != 'HOLD':
                candidate = decision_payload(session, result['target'], row, engine)
                # Same kit validator used by native uploader, but NO submission.
                from kit.contracts import validate_payload
                with tempfile.TemporaryDirectory() as payload_dir:
                    schedule_path = Path(payload_dir)/'schedule.json'
                    schedule_path.write_text(json.dumps(schedule), encoding='utf-8')
                    validate_payload(candidate, 'decision.json', schedule_path=schedule_path)
                kind = 'initial' if journal['state']['mode'] == 'PRE_ENTRY' else (
                    'recovery' if journal['state']['mode'] == 'RECOVERY_PENDING' else 'primary_rebalance')
                journal = machine.record_intent(journal, row['id'], kind, result['target'], engine.SYMBOLS, engine)
                print('V37_DRYRUN_INTENT_PRECOMMIT_AND_ORGANIZER_SCHEMA_PASS', flush=True)
                print('V37_DRYRUN_TARGET_SUM', f"{sum(result['target'].values()):.12f}", flush=True)
            require(bool(get_only.attempts) and set(get_only.attempts) == {'GET'}, 'UNEXPECTED_NON_GET_ATTEMPT')
            print('V37_ORGANIZER_NETWORK_GET_ONLY_TRUE', flush=True)
    print('V37_UNARMED_RUN_COMPLETE_NO_OFFICIAL_SUBMISSIONS', flush=True)


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--dry-run-get-only',action='store_true')
    a=p.parse_args()
    require(a.self_test != a.dry_run_get_only,'SELECT_ONE_OPERATION')
    if a.self_test:
        self_test()
    else:
        try:run_readonly()
        except Exception as e:
            print('V37_FAIL_CLOSED',type(e).__name__,str(e)[:120],flush=True)
            raise SystemExit(2)


def self_test():
    import importlib.util
    for name in ('organizer_bridge','state_machine'):
        require(importlib.util.find_spec(name) is not None,'MISSING_DEPENDENCY_'+name)
    import state_machine as m
    v=m.blank_journal({'schema':2,'mode':'PRE_ENTRY','first_execution_confirmed':False,
        'first_execution_day':None,'recovery_execution_confirmed':False,
        'last_executed_round_id':None,'last_health_day':None,'consecutive_primary_fail_days':0})
    assert not inspect_pending(v,{})
    assert eligible_for_market_refresh({'day':'2026-10-12','number':1},v['state'])
    assert not eligible_for_market_refresh({'day':'2026-10-13','number':2},v['state'])
    print('V37_LOCAL_SELF_TEST_PASS_NO_SSH_NO_API_NO_TRADING')


if __name__=='__main__':main()
