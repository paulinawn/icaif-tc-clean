#!/usr/bin/env python3
# Role: JOURNAL - records each submission intent before sending and confirms execution afterwards (exactly-once).
"""ICAIF V33 offline execution-reconciliation state machine.

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
    if not ok:
        raise UnsafeTransition(reason)


def canonical_target(weights, symbols):
    require(type(weights) is dict and set(weights) == set(symbols) and len(symbols)==30,
            'INVALID_30_STOCK_TARGET')
    target = {}
    for name in symbols:
        x = weights[name]
        require(type(x) in (int,float) and math.isfinite(x) and 0 <= x <= 0.30,
                'INVALID_TARGET_WEIGHT')
        target[name] = float(x)
    require(sum(target.values()) <= 1.0 + 1e-10, 'LEVERAGED_TARGET')
    return target


def target_hash(weights, symbols):
    target = canonical_target(weights, symbols)
    raw = json.dumps(target, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('ascii')).hexdigest()


def blank_journal(strategy_state):
    require(isinstance(strategy_state, dict) and strategy_state.get('mode')=='PRE_ENTRY',
            'INVALID_FIRST_STATE')
    return {'schema':SCHEMA, 'strategy_git':RELEASE,
            'state':copy.deepcopy(strategy_state), 'intents':{}, 'execution_confirmations':{}}


def check_journal(journal, engine):
    require(isinstance(journal, dict) and journal.get('schema')==SCHEMA and
            journal.get('strategy_git')==RELEASE, 'JOURNAL_SCOPE_OR_SCHEMA_BAD')
    engine.state_valid(journal.get('state'))
    for key in ('intents','execution_confirmations'):
        require(isinstance(journal.get(key),dict),'JOURNAL_RECORDS_MALFORMED')
    for rid,rec in journal['intents'].items():
        require(ROUND_RE.fullmatch(rid) is not None, 'UNKNOWN_JOURNAL_ROUND')
        require(isinstance(rec,dict) and rec.get('kind') in ('initial','primary_rebalance','recovery'),
                'JOURNAL_INTENT_MALFORMED')
        require(re.fullmatch(r'[a-f0-9]{64}',rec.get('target_sha256','')) is not None,
                'JOURNAL_HASH_MALFORMED')
        require(rec.get('round_id')==rid,'JOURNAL_ROUND_MISMATCH')
    require(set(journal['execution_confirmations']).issubset(set(journal['intents'])),
            'ORPHAN_EXECUTION_CONFIRMATION')
    return journal


def record_intent(journal, round_id, kind, weights, symbols, engine):
    """Persist intent BEFORE invoking the real client's immutable _decision method.
    This function does NOT initiate any upload. Duplicate differing intents fail.
    """
    check_journal(journal,engine)
    require(ROUND_RE.fullmatch(round_id) is not None, 'INVALID_OFFICIAL_ROUND')
    require(kind in ('initial','primary_rebalance','recovery'), 'INVALID_ACTION_KIND')
    digest=target_hash(weights,symbols)
    old=journal['intents'].get(round_id)
    record={'round_id':round_id,'kind':kind,'target_sha256':digest}
    require(old is None or old==record,'IMMUTABLE_INTENT_CONFLICT')
    new=copy.deepcopy(journal)
    new['intents'][round_id]=record
    return new


def complete_history_pages(pages):
    """Accept only paginated responses collected exhaustively from organizer GETs.
    Every page's returned offset and limit must agree with request order.
    """
    require(isinstance(pages,list) and 1<=len(pages)<=100,'BAD_HISTORY_PAGES')
    result=[]
    next_offset=0
    for i,p in enumerate(pages):
        require(isinstance(p,dict) and isinstance(p.get('decisions'),list) and
                type(p.get('has_more')) is bool,'INVALID_HISTORY_PAGE')
        require(p.get('offset') == next_offset and p.get('limit')==200,'HISTORY_OFFSET_OR_LIMIT')
        rows=p['decisions']
        require(len(rows)<=200 and all(isinstance(x,dict) for x in rows),'BAD_HISTORY_ROWS')
        result.extend(rows)
        next_offset+=len(rows)
        require(p['has_more']==(i<len(pages)-1),'HISTORY_NOT_EXHAUSTED')
        require(len(rows)>0 or not p['has_more'],'EMPTY_CONTINUATION_PAGE')
    require(len(result)<20000,'HISTORY_EXCESSIVE')
    seen=set()
    for row in result:
        rid=row.get('round_id')
        require(isinstance(rid,str) and ROUND_RE.fullmatch(rid),'NON_OFFICIAL_HISTORY_ROW')
        identity=row.get('decision_id')
        require(identity not in (None,'') and identity not in seen,'DUPLICATE_DECISION_ID')
        seen.add(identity)
    return result


def enforce_no_unjournaled_decisions(journal, decisions):
    """Existing Official uploads must correspond to a durable local intent.
    An unknown submission means HOLD; never adopt human/other agents' trades.
    """
    intents=set(journal['intents'])
    for decision in decisions:
        require(decision['round_id'] in intents,'UNRECOGNIZED_OFFICIAL_SUBMISSION')
    return True


def verify_execution_evidence(journal, round_id, own_round, ledger, decisions,
                              portfolio, symbols):
    """Conservative validation of organizer-authenticated GET outputs.

    Returns True only when a single VALID+SELECTED+EXECUTED decision agrees
    with the detailed round execution and ledger, and the official account
    has an invested portfolio. NEVER use a scoring receipt by itself.
    """
    require(round_id in journal['intents'],'UNJOURNALED_EXECUTION')
    matching=[d for d in decisions if d.get('round_id')==round_id]
    require(len(matching)<=1,'MULTIPLE_DECISIONS_CONFLICT')
    if not matching:
        return False
    d=matching[0]
    require(d.get('status')=='VALID' and d.get('selection_status')=='SELECTED',
            'INVALID_OR_UNSELECTED_RECEIPT')
    if d.get('execution_status')!='EXECUTED':
        require(d.get('execution_status') in (None,'PENDING','NOT_EXECUTED','SCHEDULED'),
                'UNKNOWN_OR_FAILED_EXECUTION_STATUS')
        return False
    require(isinstance(own_round,dict) and isinstance(own_round.get('decisions'),list),
            'ROUND_RESPONSE_NOT_VERIFIED')
    own=own_round['decisions']
    require(len(own)==1 and own[0].get('decision_id')==d.get('decision_id') and
            own[0].get('execution_status')=='EXECUTED', 'OWN_ROUND_MISMATCH')
    execution=own_round.get('execution')
    require(isinstance(execution,dict) and execution.get('round_id')==round_id,
            'MISSING_AUTHORITATIVE_EXECUTION')
    require(isinstance(ledger,dict) and ledger.get('phase')=='official' and
            isinstance(ledger.get('rounds'),list),'MISSING_OFFICIAL_LEDGER')
    rows=[r for r in ledger['rounds'] if isinstance(r,dict) and r.get('round_id')==round_id]
    require(len(rows)==1,'MISSING_OR_DUPLICATE_LEDGER_ROUND')
    line=rows[0]
    # Preserve financial identity without assuming timestamp serialization.
    for name in ('cash_after','nav_after'):
        try:
            a=float(execution[name]);b=float(line[name])
        except (KeyError,TypeError,ValueError):
            raise UnsafeTransition('LEDGER_FINANCIAL_FIELDS_MISSING') from None
        require(math.isfinite(a) and math.isfinite(b) and abs(a-b)<0.02,
                'EXECUTION_LEDGER_FINANCIAL_MISMATCH')
    require(portfolio.get('phase')=='official' and isinstance(portfolio.get('positions'),list)
            and len(portfolio['positions'])>0, 'NO_AUTHORITATIVE_INVESTED_PORTFOLIO')
    current=portfolio.get('current_weights')
    require(isinstance(current,dict) and set(current)==set(symbols),
            'INVESTED_PORTFOLIO_INCOMPLETE_WEIGHTS')
    try:
        total=sum(float(v) for v in current.values())
    except (TypeError,ValueError):
        raise UnsafeTransition('INVESTED_PORTFOLIO_NONNUMERIC') from None
    require(0.01<total<=1.000001, 'INVESTED_PORTFOLIO_ILLEGAL')
    return True


def apply_confirmed_execution(journal, round_id, own_round, ledger, decisions,
                              portfolio, symbols, engine):
    check_journal(journal,engine)
    if round_id in journal['execution_confirmations']:
        require(verify_execution_evidence(journal,round_id,own_round,ledger,decisions,portfolio,symbols),
                'PREVIOUSLY_CONFIRMED_EXECUTION_DISAPPEARED')
        matching=[d for d in decisions if d.get('round_id')==round_id]
        require(len(matching)==1 and
                journal['execution_confirmations'][round_id].get('decision_id')==matching[0]['decision_id'],
                'PREVIOUSLY_CONFIRMED_RECEIPT_CHANGED')
        return copy.deepcopy(journal), False
    ok=verify_execution_evidence(journal,round_id,own_round,ledger,decisions,portfolio,symbols)
    if not ok:
        return copy.deepcopy(journal), False
    record=journal['intents'][round_id]
    day=ROUND_RE.fullmatch(round_id).group(1)
    state=engine.confirm_execution(journal['state'],record['kind'],day,round_id,verified=True)
    new=copy.deepcopy(journal)
    new['state']=state
    new['execution_confirmations'][round_id]={'decision_id': next(d['decision_id'] for d in decisions if d['round_id']==round_id),
                                               'confirmed':True}
    check_journal(new,engine)
    return new,True


def atomic_private_json(path, value):
    """Atomic replacement; caller MUST hold an independent OS exclusive lock."""
    target=Path(path)
    require(target.parent.is_dir() and not target.is_symlink(),'UNSAFE_JOURNAL_PATH')
    fd,tmp=tempfile.mkstemp(prefix='.journal-',dir=target.parent)
    try:
        os.fchmod(fd,0o600)
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
            stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,target)
        # POSIX supports durability barriers on directory descriptors; Windows
        # generally denies opening directories with os.open(..., O_RDONLY).
        # Keep the POSIX barrier for the future Linux production deployment.
        if os.name == 'posix':
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def read_journal(path,engine):
    p=Path(path)
    require(p.is_file() and not p.is_symlink(),'MISSING_PERSISTENT_JOURNAL')
    if os.name == 'posix':
        require(p.stat().st_mode & 0o077 == 0,'JOURNAL_PERMISSIONS_INSECURE')
    try:
        value=json.loads(p.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError, OSError):
        raise UnsafeTransition('JOURNAL_UNREADABLE_OR_CORRUPT') from None
    return check_journal(value,engine)
