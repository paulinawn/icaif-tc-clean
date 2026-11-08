#!/usr/bin/env python3
"""ICAIF V29: pure read-only integration bridge and adversarial offline tests.

No credentials. No HTTP. No SSH. No filesystem writes except temporary self-test data.
Never calls session.watch(), _decision(), decision(), or _submit().
No deployment, schedulers, uploads, or trade decisions can be made by this module.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

UTC = timezone.utc
ET = ZoneInfo('America/New_York')

class ReadOnlyGate(Exception):
    """Safe-to-print reason the unarmed adapter refuses an actionable proposal."""


# Organizer account balances may be represented as JSON decimal strings.
# Accept finite non-negative plain decimals with up to 64 fractional digits.
# Organizer returned 34-41 fractional digits on 2026-10-19; no rounding here.
# Keep strict sign/exponent/length rejection and all downstream legality checks.
# This does not relax the cash==NAV, portfolio, or strategy safety gates.
_CANONICAL_MONEY = re.compile(r'(?:0|[1-9][0-9]{0,31})(?:\.[0-9]{1,64})?\Z')


def as_decimal(value):
    if type(value) is str:
        if len(value) > 100 or _CANONICAL_MONEY.fullmatch(value) is None:
            raise ReadOnlyGate('INVALID_NUMERIC_STRING')
    elif type(value) not in (int, float, Decimal):
        raise ReadOnlyGate('UNEXPECTED_NUMERIC_TYPE')
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ReadOnlyGate('INVALID_NUMBER') from None
    if not result.is_finite():
        raise ReadOnlyGate('NONFINITE_NUMBER')
    return result


def parse_time(value):
    if not isinstance(value, str):
        raise ReadOnlyGate('NO_AUTHENTICATED_SERVER_TIMESTAMP')
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            raise ValueError()
        return stamp.astimezone(UTC)
    except (ValueError, TypeError):
        raise ReadOnlyGate('BAD_SERVER_TIMESTAMP') from None


def cash_only_portfolio(portfolio, symbols):
    """Normalize {} to 30 zeros ONLY with strong authenticated cash evidence.

    Returned map is not proof of execution or permission to upload.
    A cash portfolio with unexplained positions/weights fails closed.
    """
    if not isinstance(portfolio, dict) or portfolio.get('phase') != 'official':
        raise ReadOnlyGate('NOT_OFFICIAL_PORTFOLIO')
    current, positions = portfolio.get('current_weights'), portfolio.get('positions')
    if not isinstance(current, dict) or not isinstance(positions, list):
        raise ReadOnlyGate('MISSING_PORTFOLIO_FIELDS')
    if current != {} or positions:
        raise ReadOnlyGate('NOT_VERIFIABLY_ALL_CASH')
    nav, cash = as_decimal(portfolio.get('nav')), as_decimal(portfolio.get('cash'))
    if nav <= 0 or cash < 0 or abs(cash/nav - 1) > Decimal('0.000001'):
        raise ReadOnlyGate('PORTFOLIO_CASH_NAV_INCONSISTENT')
    return {sym:0.0 for sym in symbols}


def normalize_portfolio(portfolio, symbols, state):
    if not isinstance(portfolio, dict) or portfolio.get('phase') != 'official':
        raise ReadOnlyGate('NOT_OFFICIAL_PORTFOLIO')
    current=portfolio.get('current_weights')
    if not isinstance(current, dict):
        raise ReadOnlyGate('WEIGHTS_MISSING')
    if state.get('mode')=='PRE_ENTRY':
        return dict(portfolio, current_weights=cash_only_portfolio(portfolio, symbols))
    # Not enough knowledge of the organizer's sparse invested-position contract.
    # Require a full 30-symbol authenticated weights mapping until verified.
    if set(current)!=set(symbols):
        raise ReadOnlyGate('POSTENTRY_WEIGHTS_NOT_COMPLETE_NEED_SCHEMA')
    values={s:float(as_decimal(current[s])) for s in symbols}
    if any(v<0 or v>.3 for v in values.values()) or sum(values.values())>1.0000000001:
        raise ReadOnlyGate('POSTENTRY_WEIGHTS_ILLEGAL')
    return dict(portfolio, current_weights=values)


def authenticated_round(schedule, own_round, symbol_order, *, local_time=None):
    """Only an already authenticated organizer GET may supply these values.

    This function cannot authenticate a JSON fixture. It is not a writer gate.
    """
    if not isinstance(schedule, dict) or schedule.get('universe_confirmed') is not True:
        raise ReadOnlyGate('UNCONFIRMED_SCHEDULE')
    if set(schedule.get('symbols', []))!=set(symbol_order) or len(schedule['symbols'])!=30:
        raise ReadOnlyGate('WRONG_UNIVERSE')
    server_time=parse_time(schedule.get('current_time'))
    if local_time is not None:
        if local_time.tzinfo is None or abs((server_time-local_time.astimezone(UTC)).total_seconds())>120:
            raise ReadOnlyGate('SERVER_CLOCK_SKEW')
    row = schedule.get('current_round')
    if not isinstance(row,dict):
        raise ReadOnlyGate('CURRENT_ROUND_NOT_OBJECT')
    if row.get('phase')!='official' or row.get('status')=='CANCELLED':
        raise ReadOnlyGate('CURRENT_ROUND_NOT_OFFICIAL')
    day, number, round_id=row.get('day'),row.get('number'),row.get('id')
    if not isinstance(day,str) or type(number)!=int or not isinstance(round_id,str):
        raise ReadOnlyGate('ROUND_IDENTITY_MALFORMED')
    if round_id!=f'official-{day}-r{number}':
        raise ReadOnlyGate('ROUND_ID_MISMATCH')
    matches=[r for r in schedule.get('rounds',[]) if isinstance(r,dict) and r.get('id')==round_id]
    if len(matches)!=1 or any(matches[0].get(k)!=row.get(k) for k in ('day','number','deadline','opens_at','phase')):
        raise ReadOnlyGate('ROUND_SCHEDULE_DISAGREES')
    opens,deadline=parse_time(row.get('opens_at')),parse_time(row.get('deadline'))
    if not opens<=server_time<deadline-timedelta(minutes=3):
        raise ReadOnlyGate('ROUND_NOT_OPEN_OR_TOO_LATE')
    if server_time.astimezone(ET).date().isoformat()!=day:
        raise ReadOnlyGate('WRONG_ET_DAY')
    if not isinstance(own_round,dict) or not isinstance(own_round.get('decisions'),list):
        raise ReadOnlyGate('OWN_ROUND_NOT_VERIFIED')
    # Conservative: any returned decision may have consumed the slot, even INVALID.
    if own_round['decisions']:
        raise ReadOnlyGate('OWN_ROUND_POSSIBLY_CONSUMED')
    if own_round.get('round_id') not in (None,round_id) or own_round.get('id') not in (None,round_id):
        raise ReadOnlyGate('OWN_ROUND_ID_MISMATCH')
    return row,server_time


def guard_history(history, state):
    """Limit unarmed PRE_ENTRY proposals to verified no-decision history.

    For any postentry situation, a future authenticated execution journal is
    required; a receipt or callback alone does not constitute execution.
    """
    if not isinstance(history,dict) or not isinstance(history.get('decisions'),list):
        raise ReadOnlyGate('DECISION_HISTORY_NOT_VERIFIED')
    if history.get('has_more') is not False:
        raise ReadOnlyGate('DECISION_HISTORY_NOT_EXHAUSTED')
    if state.get('mode')=='PRE_ENTRY' and history['decisions']:
        raise ReadOnlyGate('UNRECONCILED_EARLIER_DECISIONS')


def offline_proposal(schedule, portfolio, history, own_round, snapshot, state, engine, local_time=None):
    """Compute a proposal ONLY. Never write, submit, or return a submit callable.

    Data supplied must be authenticated by a separate read-only collector.
    This function never claims its input JSON was authenticated.
    """
    hold={'action':'HOLD','note':'UNSPECIFIED_SAFETY_STOP','target':None,'would_submit':False,
          'state_mutation':False}
    try:
        engine.state_valid(state)
        rd,now=authenticated_round(schedule,own_round,engine.SYMBOLS,local_time=local_time)
        guard_history(history,state)
        pf=normalize_portfolio(portfolio,engine.SYMBOLS,state)
        # Catch-up evidence MUST come from a dedicated authenticated reconciler,
        # so this unarmed bridge deliberately provides no purported booleans.
        obs={'phase':'official','round':rd,'symbols':list(engine.SYMBOLS),
             'as_of':now.isoformat(),'portfolio':pf,'round_state':own_round}
        candidate=engine.evaluate(obs,snapshot,copy.deepcopy(state),now=now)
        result=dict(hold,action=candidate['action'],note=candidate['note'])
        if candidate['action']!='HOLD':
            target=candidate['target']; engine.validate(target)
            # The target is usable for an offline diagnostic only: NEVER submission.
            result['target']=target
            result['target_sum']=float(sum(target.values()))
        return result
    except (ReadOnlyGate,ValueError,TypeError,KeyError,OverflowError) as e:
        reason=str(e) if isinstance(e,ReadOnlyGate) else 'ENGINE_OR_SCHEMA_REJECTED'
        return dict(hold,note=reason)


def load_engine(path):
    if not Path(path).is_dir():
        raise ReadOnlyGate('V23_DIRECTORY_NOT_FOUND')
    sys.path.insert(0,str(Path(path).resolve()))
    return importlib.import_module('icaif_official_sleeve_engine_v23')


def offline_tests(engine, kit_dir):
    # Exact organizer source is required: this self-test does not make HTTP requests.
    sys.path.insert(0,str(Path(kit_dir).resolve().parent))
    from kit.original_client import OriginalSession
    engine.regression_self_test()
    now=datetime(2026,10,12,8,55,tzinfo=ET)
    utc=now.astimezone(UTC)
    opens=(now-timedelta(minutes=20)).isoformat()
    deadline=datetime(2026,10,12,9,10,tzinfo=ET).isoformat()
    rd={'phase':'official','day':'2026-10-12','number':1,'id':'official-2026-10-12-r1',
        'opens_at':opens,'deadline':deadline,
        'close_time':(now+timedelta(hours=8)).isoformat(),
        'execution_time':(now+timedelta(minutes=35)).isoformat(),
        'status':'OPEN'}
    schedule={'universe_confirmed':True,'current_time':utc.isoformat(),
              'symbols':list(engine.SYMBOLS),'current_round':rd,'rounds':[dict(rd)]}
    pf={'phase':'official','current_weights':{},'positions':[],
        'cash':1000000,'nav':1000000,'last_completed_round':None}
    history={'decisions':[],'has_more':False,'limit':200,'offset':0}
    own={'round_id':rd['id'],'decisions':[]}
    raw = [[80.0+j for j in range(30)] for i in range(121)]
    for i in range(121):
        for j in range(30):raw[i][j]*=math.exp(.001*i+.002*math.sin(i/8+j))
    target,top,core=engine.sleeve_target_from_daily(raw)
    snapshot={'schema':2,'purpose':'LIVE_READONLY_VERIFIED','strategy':engine.SNAP_STRATEGY,
         'created_utc':(utc-timedelta(minutes=1)).isoformat(),
         'last_complete_day_et':'2026-10-09','complete_days':170,
         'lookback_sha256':'a'*64,'target_weights':target}
    st=engine.sample_state()
    before=json.dumps(st,sort_keys=True)
    res=offline_proposal(schedule,pf,history,own,snapshot,st,engine,local_time=utc)
    assert res['action']=='PRIMARY_TARGET',res
    assert res['would_submit'] is False and res['state_mutation'] is False
    assert json.dumps(st,sort_keys=True)==before
    assert abs(res['target_sum']-.8)<1e-9
    print('V29_PURE_DAY1_CASH_NORMALIZATION_PROPOSAL_PASS')
    def rejects(change, why):
        a,b,c,d,e,f=copy.deepcopy((schedule,pf,history,own,snapshot,st))
        change(a,b,c,d,e,f)
        out=offline_proposal(a,b,c,d,e,f,engine,local_time=utc)
        assert out['action']=='HOLD' and out['target'] is None,(why,out)
    rejects(lambda a,b,c,d,e,f:b['positions'].append({'symbol':'AAPL'}),'noncash-positions')
    rejects(lambda a,b,c,d,e,f:b.update(cash=500000),'cash-nav-mismatch')
    rejects(lambda a,b,c,d,e,f:c.update(has_more=True),'paginated-history')
    rejects(lambda a,b,c,d,e,f:c['decisions'].append({'selection_status':'INVALID'}),'historical-attempt')
    rejects(lambda a,b,c,d,e,f:d['decisions'].append({'selection_status':'INVALID'}),'occupied-round')
    rejects(lambda a,b,c,d,e,f:a['current_round'].update(status='CANCELLED'),'cancelled')
    rejects(lambda a,b,c,d,e,f:a.update(current_time=(utc+timedelta(minutes=18)).isoformat()),'stale')
    rejects(lambda a,b,c,d,e,f:a['current_round'].update(number=2),'fake-catchup')
    rejects(lambda a,b,c,d,e,f:e.update(created_utc=(utc-timedelta(minutes=35)).isoformat()),'stale-snapshot')
    rejects(lambda a,b,c,d,e,f:e.update(purpose='OFFLINE_DIAGNOSTIC'),'unverified-snapshot')
    rejects(lambda a,b,c,d,e,f:a['current_round'].update(id='official-2026-10-12-r2'),'id-mismatch')
    rejects(lambda a,b,c,d,e,f:a.update(universe_confirmed=False),'universe-unverified')
    rejects(lambda a,b,c,d,e,f:b.update(current_weights={'AAPL':0.2}),'noncash-sparse')
    rejects(lambda a,b,c,d,e,f:a.update(current_round='official-2026-10-12-r1'),'round-string')
    print('V29_FOURTEEN_ADVERSARIAL_FAIL_CLOSED_CASES_PASS')
    # Valid *open* R2 schedule is still HOLD: catch-up requires independently
    # authenticated proof, never caller-supplied flags.
    for review_day,prev in [('2026-10-12','2026-10-09'),
                            ('2026-10-19','2026-10-16'),
                            ('2026-10-26','2026-10-23')]:
        r2_now=datetime.fromisoformat(review_day+'T10:10:00').replace(tzinfo=ET)
        rr=dict(rd,day=review_day,number=2,id=f'official-{review_day}-r2',
                opens_at=(r2_now-timedelta(minutes=5)).isoformat(),
                deadline=(r2_now+timedelta(minutes=15)).isoformat(),
                close_time=(r2_now+timedelta(hours=5)).isoformat())
        sch=dict(schedule,current_round=rr,rounds=[rr],current_time=r2_now.astimezone(UTC).isoformat())
        own2={'round_id':rr['id'],'decisions':[]}
        snap2=dict(snapshot,last_complete_day_et=prev,
                   created_utc=(r2_now-timedelta(minutes=1)).astimezone(UTC).isoformat())
        if review_day=='2026-10-12':
            tst_state=engine.sample_state()
            tst_portfolio=pf
            tst_history=history
        else:
            tst_state=engine.confirm_execution(engine.sample_state(),'initial',
                '2026-10-12','official-2026-10-12-r1',verified=True)
            tst_portfolio=dict(pf,current_weights={s:0.8/30 for s in engine.SYMBOLS},
                               positions=[{'symbol':'AAPL'}],cash=200000,nav=1000000)
            tst_history={'decisions':[{'status':'EXECUTED'}],'has_more':False}
        ans=offline_proposal(sch,tst_portfolio,tst_history,own2,snap2,tst_state,engine,
                             local_time=r2_now.astimezone(UTC))
        assert ans['action']=='HOLD' and ans['would_submit'] is False,(review_day,ans)
    print('V29_DAY1_DAY6_DAY11_R2_UNPROVEN_CATCHUP_HOLD_PASS')

    # Reproduce the problem using the organizer's ACTUAL watch() source, with
    # all methods that could upload stubbed out before invocation.
    sess=object.__new__(OriginalSession)
    sess.state={'operations':{}}
    sess.schedule=lambda:schedule
    sess._now=lambda s:utc
    sess.round=lambda key:own
    sess._occupied=lambda r:False
    sess.portfolio=lambda phase:pf
    sess._team=lambda:{'team_id':'FAKE_ID','team_token':'FAKE_TOKEN'}
    attempts=[]
    sess._decision=lambda raw,payload:attempts.append(payload) or {'status':'FAKE_BLOCKED'}
    held=sess.watch(lambda observation:'HOLD',phase='official',once=True)
    assert held['status']=='FAKE_BLOCKED'
    assert len(attempts)==1 and attempts[0]['weights']=='HOLD'
    print('V29_NATIVE_WATCH_HOLD_WOULD_UPLOAD_PROVEN_WITH_STUB_NO_HTTP')
    print('V29_ORGANIZER_SOURCE_INTEGRATION_OFFLINE_PASS')
    print('NO_KEYS_NO_NETWORK_NO_SSH_NO_GIT_NO_OFFICIAL_POST_NO_STAGE_WRITES')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true',required=True)
    p.add_argument('--v23-dir',type=Path,required=True)
    p.add_argument('--kit-dir',type=Path,required=True)
    a=p.parse_args()
    if not (a.kit_dir/'original_client.py').is_file():
        p.error('--kit-dir must contain original_client.py')
    engine=load_engine(a.v23_dir)
    offline_tests(engine,a.kit_dir)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
