#!/usr/bin/env python3
# Role: AUDIT TOOL - read-only, one-shot printout of the official schedule, cash and holdings.
"""One-shot authenticated GET-only ICAIF Official schedule/cash/holdings audit.

Execution: run in memory as Unix user 'icaif' on the existing Hetzner VPS.
Never call submission endpoints, change trading services, or persist account files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

APP = Path('/opt/icaif2026')
ET = ZoneInfo('America/New_York')
BERLIN = ZoneInfo('Europe/Berlin')
ROUND_RE = re.compile(r'^official-\d{4}-\d{2}-\d{2}-r[1-7]$')
VALID_STATUS = re.compile(r'^[A-Z][A-Z0-9_]{0,35}$')


class GetOnlyTransport:
    def __init__(self, inner):
        self.inner = inner
        self.methods = []

    def handle_request(self, request):
        self.methods.append(request.method)
        if request.method != 'GET':
            raise RuntimeError('V32_NON_GET_BLOCKED')
        return self.inner.handle_request(request)

    def close(self):
        self.inner.close()


def iso(value):
    if not isinstance(value, str):
        raise ValueError('TIMESTAMP_NOT_STRING')
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('TIMESTAMP_NO_TIMEZONE')
    return dt.astimezone(timezone.utc)


def money(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError('MONEY_WRONG_TYPE')
    if isinstance(value, str) and not re.fullmatch(r'\d+(?:\.\d{1,12})?', value):
        raise ValueError('MONEY_BAD_DECIMAL_STRING')
    try:
        d = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('MONEY_BAD_DECIMAL') from exc
    if not d.is_finite() or d < 0:
        raise ValueError('MONEY_OUT_OF_RANGE')
    return d


def status(value):
    if not isinstance(value, str):
        return 'UNKNOWN'
    s = value.strip().upper()
    return s if VALID_STATUS.fullmatch(s) else 'OTHER'


def safe_history(client):
    collected = []
    offset = 0
    for _ in range(20):
        page = client.decisions('official', offset=offset, limit=200)
        if not isinstance(page, dict) or type(page.get('has_more')) is not bool:
            raise ValueError('DECISIONS_INVALID_PAGINATION')
        rows = page.get('decisions')
        if not isinstance(rows, list) or len(rows) > 200:
            raise ValueError('DECISIONS_BAD_ROWS')
        if any(not isinstance(x, dict) for x in rows):
            raise ValueError('DECISIONS_BAD_RECORD')
        collected.extend(rows)
        if not page['has_more']:
            return collected
        if not rows:
            raise ValueError('DECISIONS_STUCK_PAGINATION')
        offset += len(rows)
    raise ValueError('DECISIONS_TOO_MANY_PAGES')


def one_round(row):
    if not isinstance(row, dict):
        raise ValueError('ROUND_NOT_DICT')
    rid = row.get('id')
    if not isinstance(rid, str) or not ROUND_RE.fullmatch(rid):
        raise ValueError('ROUND_UNEXPECTED_ID')
    deadline = iso(row['deadline'])
    return rid, deadline, status(row.get('status'))


def print_summary(schedule, portfolio, history, ledger, own=None):
    if not isinstance(schedule, dict) or schedule.get('universe_confirmed') is not True:
        raise ValueError('SCHEDULE_UNVERIFIED')
    if not isinstance(portfolio, dict) or portfolio.get('phase') != 'official':
        raise ValueError('PORTFOLIO_UNVERIFIED')
    if not isinstance(ledger, dict) or ledger.get('phase') != 'official':
        raise ValueError('LEDGER_UNVERIFIED')
    if not isinstance(history, list):
        raise ValueError('HISTORY_UNVERIFIED')

    server_now = iso(schedule['current_time'])
    nav = money(portfolio['nav'])
    cash = money(portfolio['cash'])
    if nav <= 0 or cash > nav + Decimal('0.02'):
        raise ValueError('ACCOUNT_INCONSISTENT')
    positions = portfolio.get('positions')
    weights = portfolio.get('current_weights')
    if not isinstance(positions, list) or not isinstance(weights, dict):
        raise ValueError('PORTFOLIO_STRUCTURE_INVALID')
    if any(not isinstance(s, str) or not re.fullmatch('[A-Z]{1,5}', s) for s in weights):
        raise ValueError('WEIGHT_SYMBOL_INVALID')
    exposure = sum((money(w) for w in weights.values()), Decimal(0))
    if exposure > Decimal('1.00000001'):
        raise ValueError('EXPOSURE_OVER_100PCT')
    ledger_rounds = ledger.get('rounds')
    if not isinstance(ledger_rounds, list):
        raise ValueError('LEDGER_ROUNDS_NOT_LIST')

    print('V32_SERVER_TIME_ET', server_now.astimezone(ET).isoformat(), flush=True)
    print('V32_SERVER_TIME_BERLIN', server_now.astimezone(BERLIN).isoformat(), flush=True)
    print('V32_ACCOUNT_AS_OF', portfolio.get('as_of') if isinstance(portfolio.get('as_of'), str) else 'UNKNOWN', flush=True)
    print('V32_OFFICIAL_CASH_USD', format(cash, '.2f'), flush=True)
    print('V32_OFFICIAL_NAV_USD', format(nav, '.2f'), flush=True)
    print('V32_OFFICIAL_INVESTED_APPROX_USD', format(nav-cash, '.2f'), flush=True)
    print('V32_OFFICIAL_POSITIONS_COUNT', len(positions), flush=True)
    print('V32_OFFICIAL_WEIGHT_SYMBOLS_COUNT', len(weights), flush=True)
    print('V32_OFFICIAL_WEIGHT_SUM', format(exposure, '.8f'), flush=True)
    print('V32_OFFICIAL_DECISIONS_COUNT', len(history), flush=True)
    print('V32_OFFICIAL_LEDGER_ROUNDS_COUNT', len(ledger_rounds), flush=True)
    print('V32_LAST_DECISION_STATUS', status(portfolio.get('last_decision_status')), flush=True)
    print('V32_LAST_COMPLETED_ROUND_PRESENT', bool(portfolio.get('last_completed_round')), flush=True)

    current = schedule.get('current_round')
    if isinstance(current, dict) and current.get('phase') == 'official':
        rid, deadline, st = one_round(current)
        print('V32_CURRENT_OFFICIAL_ROUND', rid, flush=True)
        print('V32_CURRENT_ROUND_STATUS', st, flush=True)
        print('V32_CURRENT_ROUND_DEADLINE_ET', deadline.astimezone(ET).isoformat(), flush=True)
        print('V32_CURRENT_ROUND_DEADLINE_BERLIN', deadline.astimezone(BERLIN).isoformat(), flush=True)
        if not isinstance(own, dict) or not isinstance(own.get('decisions'), list):
            raise ValueError('CURRENT_OWN_ROUND_INVALID')
        print('V32_CURRENT_ROUND_OWN_DECISIONS', len(own['decisions']), flush=True)
        if isinstance(own.get('execution'), dict):
            print('V32_CURRENT_ROUND_EXECUTION_RECORD', 'PRESENT', flush=True)
        else:
            print('V32_CURRENT_ROUND_EXECUTION_RECORD', 'ABSENT', flush=True)
    else:
        print('V32_CURRENT_OFFICIAL_ROUND NONE', flush=True)

    rows = schedule.get('rounds')
    if not isinstance(rows, list):
        raise ValueError('SCHEDULE_ROUNDS_INVALID')
    official = []
    for row in rows:
        if isinstance(row, dict) and row.get('phase') == 'official':
            official.append(one_round(row))
    if len(official) != 105:
        print('V32_OFFICIAL_SCHEDULE_ROUND_COUNT', len(official), flush=True)
        raise ValueError('OFFICIAL_SCHEDULE_ROUNDS_NOT_105')
    official.sort(key=lambda item:item[1])
    today = server_now.astimezone(ET).date()
    print('V32_SCHEDULE_OFFICIAL_TOTAL', len(official), flush=True)
    same_day = [r for r in official if r[1].astimezone(ET).date() == today]
    print('V32_TODAY_ET', today.isoformat(), flush=True)
    print('V32_TODAY_OFFICIAL_ROUNDS', len(same_day), flush=True)
    for rid, deadline, st in same_day:
        print('V32_TODAY_ROUND', rid, 'DEADLINE_ET', deadline.astimezone(ET).strftime('%H:%M'),
              'BERLIN', deadline.astimezone(BERLIN).strftime('%H:%M'), 'STATUS', st, flush=True)
    future = [r for r in official if r[1] > server_now]
    print('V32_REMAINING_OFFICIAL_DEADLINES', len(future), flush=True)
    for rid, deadline, st in future[:3]:
        print('V32_NEXT_ROUND', rid, 'DEADLINE_ET', deadline.astimezone(ET).isoformat(),
              'DEADLINE_BERLIN', deadline.astimezone(BERLIN).isoformat(), 'STATUS', st, flush=True)

    if history:
        latest = history[-1]
        for label, field in (('VALIDATION','status'),('SELECTION','selection_status'),('EXECUTION','execution_status')):
            print('V32_LATEST_DECISION_'+label, status(latest.get(field)), flush=True)
    print('V32_OFFICIAL_ACCOUNT_GET_ONLY_PASS', flush=True)


def run():
    import httpx
    from kit.config import load_environment
    from kit.original_client import OriginalSession, load_profile
    load_environment(APP/'.env')
    token = os.environ.get('CODABENCH_TOKEN')
    prof = os.environ.get('ICAIF_PROFILE')
    if not token or not prof:
        raise ValueError('PRIVATE_CONFIG_NOT_AVAILABLE')
    pp = Path(prof)
    if not pp.is_absolute():
        pp = APP / pp
    cfg = load_profile(pp)
    with tempfile.TemporaryDirectory(prefix='icaif_v32_getonly_') as tmp:
        gate = GetOnlyTransport(httpx.HTTPTransport())
        with OriginalSession(profile=cfg, token=token,
                             checkpoint=Path(tmp)/'checkpoint.json',
                             credentials=APP/'.icaif'/'credentials.json',
                             transport=gate, timeout=18, result_timeout=10) as client:
            schedule = client.schedule()
            portfolio = client.portfolio('official')
            history = safe_history(client)
            ledger = client.ledger('official')
            current = schedule.get('current_round')
            own = None
            if isinstance(current, dict) and current.get('phase') == 'official':
                rid = current.get('id')
                if not isinstance(rid, str) or not ROUND_RE.fullmatch(rid):
                    raise ValueError('CURRENT_ROUND_ID_INVALID')
                own = client.round(rid)
            if not gate.methods or any(x != 'GET' for x in gate.methods):
                raise ValueError('GET_ONLY_TRANSPORT_FAILED')
            print_summary(schedule, portfolio, history, ledger, own)
            print('V32_HTTP_METHODS_GET_ONLY TRUE', flush=True)
    print('V32_NO_SUBMISSION_NO_TIMERS_NO_STRATEGY_CHANGES', flush=True)


def self_test():
    class Post:method='POST'
    class Never:
        def handle_request(self, request):
            raise AssertionError('NON_GET_ESCAPED')
        def close(self):pass
    gate=GetOnlyTransport(Never())
    try:gate.handle_request(Post())
    except RuntimeError as e:assert str(e)=='V32_NON_GET_BLOCKED'
    else:raise AssertionError('POST not blocked')
    for bad in ('NaN','+1','-1','1,000',' 1','Infinity','1e4',None,True,{}):
        try:money(bad)
        except ValueError:pass
        else:raise AssertionError(f'Invalid balance accepted: {type(bad)}')
    assert money('1000000.00') == Decimal('1000000.00')
    assert iso('2026-10-12T13:10:00+00:00').astimezone(BERLIN).strftime('%H:%M')=='15:10'
    class Hist:
        def decisions(self, phase, *,offset,limit):
            assert phase=='official' and offset==0
            return {'decisions':[],'has_more':False}
    assert safe_history(Hist()) == []
    print('V32_SELF_TEST_PASS_GET_ONLY_DECIMALS_TIMEZONE_PAGINATION')
    print('V32_SELF_TEST_NO_NETWORK_NO_REMOTE_WRITES')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    choices=p.add_mutually_exclusive_group(required=True)
    choices.add_argument('--self-test',action='store_true')
    choices.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.self_test:
        self_test()
        return 0
    try:
        run()
        return 0
    except Exception as exc:
        print('V32_FAIL_CLOSED',type(exc).__name__,flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
