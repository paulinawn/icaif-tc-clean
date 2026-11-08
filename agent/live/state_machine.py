#!/usr/bin/env python3
# Role: JOURNAL - records each submission intent before sending and confirms execution afterwards (exactly-once).
"""JOURNAL offline execution-reconciliation state machine.

No organizer network code, no submission code, no timer actions.
All arguments are data supplied by a caller; never infer authentication from them.
Only a future credential-bound collector is allowed to call reconcile_round_live.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

RELEASE = '8347a897ec6a'
SCHEMA = 33
ROUND_RE = re.compile(r'^official-(2026-10-(?:12|13|14|15|16|19|20|21|22|23|26|27|28|29|30))-r([1-7])$')

class UnsafeTransition(ValueError):
    pass


def require(ok, reason):
    """Raise UnsafeTransition with the given reason code unless ok is truthy.

    Inputs: ok -- a truthy/falsy condition; reason -- a string reason code.
    Returns: None. Raises: UnsafeTransition(reason) when ok is falsy.
    """
    if not ok:
        raise UnsafeTransition(reason)


def canonical_target(weights, symbols):
    """Validate the 30-stock target weights and return them as floats.

    Inputs: weights -- dict of symbol -> weight; symbols -- the expected
    30-symbol list. Returns: a new dict of symbol -> float weight.
    Raises: UnsafeTransition with INVALID_30_STOCK_TARGET,
    INVALID_TARGET_WEIGHT, or LEVERAGED_TARGET.
    """
    require(type(weights) is dict and set(weights) == set(symbols) and len(symbols) == 30,
            'INVALID_30_STOCK_TARGET')
    target = {}
    for symbol in symbols:
        weight_value = weights[symbol]
        require(type(weight_value) in (int, float) and math.isfinite(weight_value)
                and 0 <= weight_value <= 0.30,
                'INVALID_TARGET_WEIGHT')
        target[symbol] = float(weight_value)
    require(sum(target.values()) <= 1.0 + 1e-10, 'LEVERAGED_TARGET')
    return target


def target_hash(weights, symbols):
    """Return the SHA-256 hex digest of the canonical target weights.

    Inputs: weights -- dict of symbol -> weight; symbols -- the expected
    30-symbol list. Returns: a 64-character lowercase hex digest string.
    Raises: UnsafeTransition propagated from canonical_target.
    """
    target = canonical_target(weights, symbols)
    raw = json.dumps(target, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('ascii')).hexdigest()


def blank_journal(strategy_state):
    """Create a fresh empty journal for a PRE_ENTRY strategy state.

    Inputs: strategy_state -- dict with mode == 'PRE_ENTRY'. Returns: a new
    journal dict with empty intents and execution_confirmations.
    Raises: UnsafeTransition with INVALID_FIRST_STATE.
    """
    require(isinstance(strategy_state, dict) and strategy_state.get('mode') == 'PRE_ENTRY',
            'INVALID_FIRST_STATE')
    return {'schema': SCHEMA, 'strategy_git': RELEASE,
            'state': copy.deepcopy(strategy_state), 'intents': {}, 'execution_confirmations': {}}


def check_journal(journal, engine):
    """Validate a journal's schema, state, and record shapes in place.

    Inputs: journal -- the journal dict to validate; engine -- object with
    a state_valid method. Returns: the same journal dict.
    Raises: UnsafeTransition with JOURNAL_SCOPE_OR_SCHEMA_BAD,
    JOURNAL_RECORDS_MALFORMED, UNKNOWN_JOURNAL_ROUND,
    JOURNAL_INTENT_MALFORMED, JOURNAL_HASH_MALFORMED,
    JOURNAL_ROUND_MISMATCH, or ORPHAN_EXECUTION_CONFIRMATION.
    """
    require(isinstance(journal, dict) and journal.get('schema') == SCHEMA and
            journal.get('strategy_git') == RELEASE, 'JOURNAL_SCOPE_OR_SCHEMA_BAD')
    engine.state_valid(journal.get('state'))
    for collection_name in ('intents', 'execution_confirmations'):
        require(isinstance(journal.get(collection_name), dict), 'JOURNAL_RECORDS_MALFORMED')
    for round_id, record in journal['intents'].items():
        require(ROUND_RE.fullmatch(round_id) is not None, 'UNKNOWN_JOURNAL_ROUND')
        require(isinstance(record, dict)
                and record.get('kind') in ('initial', 'primary_rebalance', 'recovery'),
                'JOURNAL_INTENT_MALFORMED')
        require(re.fullmatch(r'[a-f0-9]{64}', record.get('target_sha256', '')) is not None,
                'JOURNAL_HASH_MALFORMED')
        require(record.get('round_id') == round_id, 'JOURNAL_ROUND_MISMATCH')
    require(set(journal['execution_confirmations']).issubset(set(journal['intents'])),
            'ORPHAN_EXECUTION_CONFIRMATION')
    return journal


def record_intent(journal, round_id, kind, weights, symbols, engine):
    """Return a new journal with this submission intent recorded for round_id.

    This persists the intent BEFORE the real client's immutable _decision
    method is invoked; it does NOT initiate any upload. Re-recording an
    identical intent is idempotent; a differing intent for the same round
    fails.

    Inputs: journal -- validated journal dict; round_id -- official round id;
    kind -- one of 'initial', 'primary_rebalance', 'recovery'; weights --
    dict of symbol -> weight; symbols -- expected 30-symbol list; engine --
    object with state_valid. Returns: a deep-copied journal with the intent
    added. Raises: UnsafeTransition with JOURNAL_SCOPE_OR_SCHEMA_BAD,
    INVALID_OFFICIAL_ROUND, INVALID_ACTION_KIND, INVALID_30_STOCK_TARGET,
    INVALID_TARGET_WEIGHT, LEVERAGED_TARGET, or IMMUTABLE_INTENT_CONFLICT.
    """
    check_journal(journal, engine)
    require(ROUND_RE.fullmatch(round_id) is not None, 'INVALID_OFFICIAL_ROUND')
    require(kind in ('initial', 'primary_rebalance', 'recovery'), 'INVALID_ACTION_KIND')
    digest = target_hash(weights, symbols)
    previous = journal['intents'].get(round_id)
    record = {'round_id': round_id, 'kind': kind, 'target_sha256': digest}
    require(previous is None or previous == record, 'IMMUTABLE_INTENT_CONFLICT')
    new_journal = copy.deepcopy(journal)
    new_journal['intents'][round_id] = record
    return new_journal


def complete_history_pages(pages):
    """Flatten paginated organizer history responses into one row list.

    Inputs: pages -- list of page dicts collected exhaustively from
    organizer GETs; each page must agree with request order on offset and
    limit. Returns: the concatenated list of decision row dicts.
    Raises: UnsafeTransition with BAD_HISTORY_PAGES, INVALID_HISTORY_PAGE,
    HISTORY_OFFSET_OR_LIMIT, BAD_HISTORY_ROWS, HISTORY_NOT_EXHAUSTED,
    EMPTY_CONTINUATION_PAGE, HISTORY_EXCESSIVE, NON_OFFICIAL_HISTORY_ROW,
    or DUPLICATE_DECISION_ID.
    """
    require(isinstance(pages, list) and 1 <= len(pages) <= 100, 'BAD_HISTORY_PAGES')
    rows_all = []
    next_offset = 0
    for page_index, page in enumerate(pages):
        require(isinstance(page, dict) and isinstance(page.get('decisions'), list) and
                type(page.get('has_more')) is bool, 'INVALID_HISTORY_PAGE')
        require(page.get('offset') == next_offset and page.get('limit') == 200,
                'HISTORY_OFFSET_OR_LIMIT')
        rows = page['decisions']
        require(len(rows) <= 200 and all(isinstance(row, dict) for row in rows),
                'BAD_HISTORY_ROWS')
        rows_all.extend(rows)
        next_offset += len(rows)
        require(page['has_more'] == (page_index < len(pages) - 1), 'HISTORY_NOT_EXHAUSTED')
        require(len(rows) > 0 or not page['has_more'], 'EMPTY_CONTINUATION_PAGE')
    require(len(rows_all) < 20000, 'HISTORY_EXCESSIVE')
    seen_decision_ids = set()
    for row in rows_all:
        round_id = row.get('round_id')
        require(isinstance(round_id, str) and ROUND_RE.fullmatch(round_id),
                'NON_OFFICIAL_HISTORY_ROW')
        decision_id = row.get('decision_id')
        require(decision_id not in (None, '') and decision_id not in seen_decision_ids,
                'DUPLICATE_DECISION_ID')
        seen_decision_ids.add(decision_id)
    return rows_all


def enforce_no_unjournaled_decisions(journal, decisions):
    """Verify every organizer decision corresponds to a durable local intent.

    Inputs: journal -- journal dict with an 'intents' mapping; decisions --
    list of organizer decision dicts. Returns: True.
    Raises: UnsafeTransition with UNRECOGNIZED_OFFICIAL_SUBMISSION for any
    decision whose round_id has no recorded intent.
    """
    intent_round_ids = set(journal['intents'])
    for decision in decisions:
        require(decision['round_id'] in intent_round_ids, 'UNRECOGNIZED_OFFICIAL_SUBMISSION')
    return True


def verify_execution_evidence(journal, round_id, own_round, ledger, decisions,
                              portfolio, symbols):
    """Conservatively validate organizer-authenticated GET outputs.

    Returns True only when a single VALID+SELECTED+EXECUTED decision agrees
    with the detailed round execution and ledger, and the official account
    has an invested portfolio. NEVER use a scoring receipt by itself.

    Inputs: journal -- journal dict; round_id -- the round to verify;
    own_round -- the detailed round response; ledger -- official ledger;
    decisions -- organizer decision list; portfolio -- official portfolio;
    symbols -- expected symbol list. Returns: True on verified execution,
    False when evidence is absent or still pending. Raises: UnsafeTransition
    with UNJOURNALED_EXECUTION, MULTIPLE_DECISIONS_CONFLICT,
    INVALID_OR_UNSELECTED_RECEIPT, UNKNOWN_OR_FAILED_EXECUTION_STATUS,
    ROUND_RESPONSE_NOT_VERIFIED, OWN_ROUND_MISMATCH,
    MISSING_AUTHORITATIVE_EXECUTION, MISSING_OFFICIAL_LEDGER,
    MISSING_OR_DUPLICATE_LEDGER_ROUND, LEDGER_FINANCIAL_FIELDS_MISSING,
    EXECUTION_LEDGER_FINANCIAL_MISMATCH,
    NO_AUTHORITATIVE_INVESTED_PORTFOLIO,
    INVESTED_PORTFOLIO_INCOMPLETE_WEIGHTS,
    INVESTED_PORTFOLIO_NONNUMERIC, or INVESTED_PORTFOLIO_ILLEGAL.
    """
    require(round_id in journal['intents'], 'UNJOURNALED_EXECUTION')
    matching_decisions = [d for d in decisions if d.get('round_id') == round_id]
    require(len(matching_decisions) <= 1, 'MULTIPLE_DECISIONS_CONFLICT')
    if not matching_decisions:
        return False
    decision = matching_decisions[0]
    require(decision.get('status') == 'VALID'
            and decision.get('selection_status') == 'SELECTED',
            'INVALID_OR_UNSELECTED_RECEIPT')
    if decision.get('execution_status') != 'EXECUTED':
        require(decision.get('execution_status') in (None, 'PENDING', 'NOT_EXECUTED', 'SCHEDULED'),
                'UNKNOWN_OR_FAILED_EXECUTION_STATUS')
        return False
    require(isinstance(own_round, dict) and isinstance(own_round.get('decisions'), list),
            'ROUND_RESPONSE_NOT_VERIFIED')
    own_decisions = own_round['decisions']
    require(len(own_decisions) == 1
            and own_decisions[0].get('decision_id') == decision.get('decision_id')
            and own_decisions[0].get('execution_status') == 'EXECUTED',
            'OWN_ROUND_MISMATCH')
    execution = own_round.get('execution')
    require(isinstance(execution, dict) and execution.get('round_id') == round_id,
            'MISSING_AUTHORITATIVE_EXECUTION')
    require(isinstance(ledger, dict) and ledger.get('phase') == 'official'
            and isinstance(ledger.get('rounds'), list), 'MISSING_OFFICIAL_LEDGER')
    ledger_rows = [r for r in ledger['rounds']
                   if isinstance(r, dict) and r.get('round_id') == round_id]
    require(len(ledger_rows) == 1, 'MISSING_OR_DUPLICATE_LEDGER_ROUND')
    ledger_row = ledger_rows[0]
    # Preserve financial identity without assuming timestamp serialization.
    for field_name in ('cash_after', 'nav_after'):
        try:
            execution_value = float(execution[field_name])
            ledger_value = float(ledger_row[field_name])
        except (KeyError, TypeError, ValueError):
            raise UnsafeTransition('LEDGER_FINANCIAL_FIELDS_MISSING') from None
        require(math.isfinite(execution_value) and math.isfinite(ledger_value)
                and abs(execution_value - ledger_value) < 0.02,
                'EXECUTION_LEDGER_FINANCIAL_MISMATCH')
    require(portfolio.get('phase') == 'official'
            and isinstance(portfolio.get('positions'), list)
            and len(portfolio['positions']) > 0,
            'NO_AUTHORITATIVE_INVESTED_PORTFOLIO')
    current_weights = portfolio.get('current_weights')
    require(isinstance(current_weights, dict) and set(current_weights) == set(symbols),
            'INVESTED_PORTFOLIO_INCOMPLETE_WEIGHTS')
    try:
        invested_total = sum(float(v) for v in current_weights.values())
    except (TypeError, ValueError):
        raise UnsafeTransition('INVESTED_PORTFOLIO_NONNUMERIC') from None
    require(0.01 < invested_total <= 1.000001, 'INVESTED_PORTFOLIO_ILLEGAL')
    return True


def apply_confirmed_execution(journal, round_id, own_round, ledger, decisions,
                              portfolio, symbols, engine):
    """Return (new_journal, confirmed) after reconciling one round's execution.

    If round_id was already confirmed, re-verify the evidence and confirm
    nothing changed; otherwise attempt verification and, on success, advance
    the engine state and record the confirmation.

    Inputs: journal -- journal dict; round_id -- the round to reconcile;
    own_round, ledger, decisions, portfolio, symbols -- evidence inputs as
    in verify_execution_evidence; engine -- object with state_valid and
    confirm_execution. Returns: a (journal, bool) tuple where the bool is
    True only when a new confirmation was recorded. Raises: UnsafeTransition
    propagated from check_journal, verify_execution_evidence,
    PREVIOUSLY_CONFIRMED_EXECUTION_DISAPPEARED, or
    PREVIOUSLY_CONFIRMED_RECEIPT_CHANGED.
    """
    check_journal(journal, engine)
    if round_id in journal['execution_confirmations']:
        require(verify_execution_evidence(journal, round_id, own_round, ledger, decisions,
                                          portfolio, symbols),
                'PREVIOUSLY_CONFIRMED_EXECUTION_DISAPPEARED')
        matching_decisions = [d for d in decisions if d.get('round_id') == round_id]
        require(len(matching_decisions) == 1
                and journal['execution_confirmations'][round_id].get('decision_id')
                == matching_decisions[0]['decision_id'],
                'PREVIOUSLY_CONFIRMED_RECEIPT_CHANGED')
        return copy.deepcopy(journal), False
    verified = verify_execution_evidence(journal, round_id, own_round, ledger, decisions,
                                         portfolio, symbols)
    if not verified:
        return copy.deepcopy(journal), False
    intent_record = journal['intents'][round_id]
    round_day = ROUND_RE.fullmatch(round_id).group(1)
    new_state = engine.confirm_execution(journal['state'], intent_record['kind'],
                                         round_day, round_id, verified=True)
    new_journal = copy.deepcopy(journal)
    new_journal['state'] = new_state
    new_journal['execution_confirmations'][round_id] = {
        'decision_id': next(decision['decision_id'] for decision in decisions
                            if decision['round_id'] == round_id),
        'confirmed': True}
    check_journal(new_journal, engine)
    return new_journal, True


def atomic_private_json(path, value):
    """Atomically replace the journal file at path with the given value.

    Writes to a temp file in the same directory with mode 0600, flushes and
    fsyncs it, then os.replace()s it over the target, and on POSIX fsyncs
    the directory for durability. The caller MUST hold an independent OS
    exclusive lock.

    Inputs: path -- destination file path; value -- JSON-serializable
    journal dict. Returns: None. Raises: UnsafeTransition with
    UNSAFE_JOURNAL_PATH, plus OSError from filesystem operations.
    """
    target_path = Path(path)
    require(target_path.parent.is_dir() and not target_path.is_symlink(),
            'UNSAFE_JOURNAL_PATH')
    temp_fd, temp_name = tempfile.mkstemp(prefix='.journal-', dir=target_path.parent)
    try:
        os.fchmod(temp_fd, 0o600)
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target_path)
        # POSIX supports durability barriers on directory descriptors; Windows
        # generally denies opening directories with os.open(..., O_RDONLY).
        # Keep the POSIX barrier for the future Linux production deployment.
        if os.name == 'posix':
            directory_fd = os.open(target_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_journal(path, engine):
    """Load and validate the persistent journal file at path.

    Inputs: path -- journal file path; engine -- object with state_valid.
    Returns: the validated journal dict. Raises: UnsafeTransition with
    MISSING_PERSISTENT_JOURNAL, JOURNAL_PERMISSIONS_INSECURE (POSIX only),
    JOURNAL_UNREADABLE_OR_CORRUPT, or errors from check_journal.
    """
    journal_path = Path(path)
    require(journal_path.is_file() and not journal_path.is_symlink(),
            'MISSING_PERSISTENT_JOURNAL')
    if os.name == 'posix':
        require(journal_path.stat().st_mode & 0o077 == 0, 'JOURNAL_PERMISSIONS_INSECURE')
    try:
        value = json.loads(journal_path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError, OSError):
        raise UnsafeTransition('JOURNAL_UNREADABLE_OR_CORRUPT') from None
    return check_journal(value, engine)
