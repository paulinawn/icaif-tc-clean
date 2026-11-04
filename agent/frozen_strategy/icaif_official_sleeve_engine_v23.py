#!/usr/bin/env python3
"""ICAIF 2026 v2.3 CANDIDATE engine: OFFLINE/READ-ONLY; NO organizer writer.

Exact Phase44 Blended80 + 15%-of-EQUITY five-day reversal sleeve:
* Initial Day 1 R1, or Day 1 R2-R7 with fresh, reconciled read-only evidence;
  scheduled five-day reviews occur Day6/11 R1, with same-day R2-R7
  catch-up only if authoritative evidence shows no earlier executed rebalance.
* Threshold: 3 percentage points L1; cap: 10pp L1 per scheduled rebalance.
* Temporary failure = HOLD; two consecutive R1 daily data/computation failures
  AFTER a verified first execution = request one-way Equal80 RECOVERY.
* Recovery is not considered COMPLETED until authenticated organizer evidence of
  executed allocation is reconciled by an as-yet-unbuilt official writer.
No orders, submissions, model calls, credentials, SSH, or service changes.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import icaif_official_market_support_v2 as base

ET=ZoneInfo('America/New_York'); UTC=timezone.utc
SYMBOLS=base.SYMBOLS
OFFICIAL_DAYS=tuple(date(2026,10,d).isoformat() for d in
                    (12,13,14,15,16,19,20,21,22,23,26,27,28,29,30))
SCHEMA=2
SNAP_STRATEGY='blend80_15pct_equity_fiveday_reversal_phase44_exact'
STATUS=('PRE_ENTRY','PRIMARY','RECOVERY_PENDING','EQUAL80_LOCKED')


def expected_previous_day(day):
    d=date.fromisoformat(day)-timedelta(days=1)
    while d.weekday()>=5: d-=timedelta(days=1)
    return d.isoformat()


def z(x):
    a=np.asarray(x,float)
    if not np.isfinite(a).all():raise ValueError('Non-finite signal')
    return np.clip((a-a.mean())/max(float(a.std()),1e-8),-3.,3.)


def sleeve_target_from_daily(daily):
    """Exact Phase44 sleeve formula, including stable tie ranking, using 121 closes."""
    arr=np.asarray(daily[-121:],dtype=float)
    if arr.shape!=(121,30) or not np.isfinite(arr).all() or np.any(arr<=0):
        raise ValueError('Require 121 completed 30-stock daily close vectors')
    core=base.blend_weights(arr.tolist())
    log=np.log(arr)
    r1=log[-1]-log[-2];r5=log[-1]-log[-6]
    reversal=z(-.75*z(r5)-.25*z(r1))
    chosen=np.argsort(-reversal,kind='stable')[:5]
    concentrated=np.zeros(30,dtype=float)
    concentrated[chosen]=.8/5.
    blend=np.array([core[s] for s in SYMBOLS])
    final=.85*blend+.15*concentrated
    result={sym:float(final[i]) for i,sym in enumerate(SYMBOLS)}
    validate(result,expect_equity=.8)
    return result, [SYMBOLS[i] for i in chosen], core


def validate(weights, expect_equity=None):
    if not isinstance(weights,dict) or set(weights)!=set(SYMBOLS):
        raise ValueError('Must include exactly the 30 official stocks')
    if any(type(weights[s]) not in (float,int) or not math.isfinite(weights[s])
           or weights[s]<0 or weights[s]>.30 for s in SYMBOLS):
        raise ValueError('Invalid numeric long-only weights')
    total=sum(float(weights[s]) for s in SYMBOLS)
    if total>1+1e-10:raise ValueError('Portfolio leverage prohibited')
    if expect_equity is not None and abs(total-expect_equity)>1e-9:
        raise ValueError('Unexpected target investment exposure')
    return total


def snapshot_from_bars(rows, now_utc, purpose='OFFLINE_DIAGNOSTIC'):
    """Use the SAME complete 26-bar-session gate as release V1."""
    dates,daily,missing=base.history_closes(rows,now_utc)
    target,selected,core=sleeve_target_from_daily(daily)
    digest=hashlib.sha256(json.dumps({'dates':dates[-121:],
          'daily':daily[-121:]},sort_keys=True).encode()).hexdigest()
    return {'schema':SCHEMA,'strategy':SNAP_STRATEGY,'purpose':purpose,
       'created_utc':now_utc.isoformat(),'last_complete_day_et':dates[-1],
       'complete_days':len(dates),'incomplete_days':len(missing),
       'lookback_sha256':digest,'selected_five':selected,
       'target_weights':target,'base_weights':core,
       'feed':'Alpaca IEX 15Min raw; completed 15:45ET closes'}


def sample_state():
    return {'schema':SCHEMA,'mode':'PRE_ENTRY','first_execution_confirmed':False,
       'first_execution_day':None,'consecutive_primary_fail_days':0,
       'last_health_day':None,'last_executed_round_id':None,
       'recovery_execution_confirmed':False}


def state_valid(state):
    if not isinstance(state,dict) or state.get('schema')!=SCHEMA or state.get('mode') not in STATUS:
        raise ValueError('Unknown or uninitialized state; HOLD')
    if type(state.get('first_execution_confirmed'))!=bool:
        raise ValueError('Unverified first execution')
    n=state.get('consecutive_primary_fail_days')
    if type(n)!=int or n<0 or n>15:raise ValueError('Bad failure counter')
    if state['mode']!='PRE_ENTRY' and not state['first_execution_confirmed']:
        raise ValueError('Postentry state without confirmed first execution')
    if state['mode']=='RECOVERY_PENDING' and n<2:
        raise ValueError('Recovery mode without persistent failures')
    if state['mode']=='EQUAL80_LOCKED' and state.get('recovery_execution_confirmed') is not True:
        raise ValueError('Recovery lock without executed evidence')
    return state


def live_round(observation, now):
    """Require actual organizer-provided phase/round/portfolio fields; fail closed."""
    if not isinstance(observation,dict):raise ValueError('Observation not mapping')
    rd=observation.get('round');pf=observation.get('portfolio')
    if not isinstance(rd,dict) or not isinstance(pf,dict):
        raise ValueError('Missing organizer round or portfolio')
    if rd.get('phase')!='official' or pf.get('phase')!='official':
        raise ValueError('Wrong competition phase')
    day=rd.get('day');num=rd.get('number')
    if day not in OFFICIAL_DAYS or type(num)!=int or num not in range(1,8):
        raise ValueError('Unexpected Official day/round')
    assert now.tzinfo is not None
    deadline=base.utc_stamp(rd['deadline'] if 'deadline' in rd else rd['deadline_at'])
    if now.astimezone(ET).date().isoformat()!=day:
        raise ValueError('Not current ET trading day')
    if now>=deadline-timedelta(minutes=3):raise ValueError('Insufficient submission buffer')
    if 'opens_at' in rd and now<base.utc_stamp(rd['opens_at']):
        raise ValueError('Round not open')
    hhmm=base.DEADLINES[num]
    if deadline.astimezone(ET).strftime('%H:%M')!=hhmm:
        raise ValueError('Deadline differs from standard schedule; need review')
    cur=pf.get('current_weights')
    validate(cur)
    return day,num,cur


def good_snapshot(snapshot,now,day):
    if not isinstance(snapshot,dict) or snapshot.get('schema')!=SCHEMA:
        raise ValueError('No verified v2 snapshot')
    if snapshot.get('strategy')!=SNAP_STRATEGY or snapshot.get('purpose')!='LIVE_READONLY_VERIFIED':
        raise ValueError('Wrong snapshot strategy/purpose')
    stamp=base.utc_stamp(snapshot['created_utc'])
    if stamp>now or now-stamp>timedelta(minutes=30):
        raise ValueError('Missing or stale authenticated data snapshot')
    if snapshot.get('last_complete_day_et')!=expected_previous_day(day):
        raise ValueError('Previous full session not confirmed')
    if type(snapshot.get('complete_days'))!=int or snapshot['complete_days']<121:
        raise ValueError('Insufficient complete history')
    if not isinstance(snapshot.get('lookback_sha256'),str) or len(snapshot['lookback_sha256'])!=64:
        raise ValueError('Missing provenance digest')
    validate(snapshot.get('target_weights'),expect_equity=.8)
    return snapshot['target_weights']


def trim_target(current,target,initial=False,band=.03,max_l1=.10):
    """Phase44 L1 turnover band/cap using organizer's pre-execution CURRENT weights."""
    validate(current);validate(target,expect_equity=.8)
    if initial:return target
    old=np.array([float(current[s]) for s in SYMBOLS]);new=np.array([float(target[s]) for s in SYMBOLS]);
    distance=float(np.abs(new-old).sum())
    if distance<band:return None
    if distance>max_l1:new=old+(new-old)*(max_l1/distance)
    res={s:float(new[j]) for j,s in enumerate(SYMBOLS)}
    validate(res)
    return res


def verified_late_entry_clearance(observation, day, r, now):
    """Fail-closed contract for the future authenticated single-writer adapter.

    WARNING: Evidence here is a DATA CONTRACT, not independently authenticated.
    A future production runner MUST generate these fields itself from live
    organizer GETs and MUST NOT trust input files/user-supplied booleans.
    This module has NO organizer GET/POST capability and cannot trade.
    """
    if day!=OFFICIAL_DAYS[0] or r<=1:
        return False
    ev=observation.get('preentry_reconciliation') if isinstance(observation,dict) else None
    if not isinstance(ev,dict) or ev.get('source')!='AUTHENTICATED_ORGANIZER_GET':
        return False
    if ev.get('round_id')!=f'official-{day}-r{r}':
        return False
    for key in ('earlier_rounds_finalized','earlier_rounds_no_executed_trade',
                'current_round_no_attributable_attempt','verified_cash_portfolio'):
        if ev.get(key) is not True:
            return False
    try:
        checked=base.utc_stamp(ev['checked_utc'])
    except (ValueError,KeyError,TypeError):
        return False
    age=(now-checked).total_seconds()
    return -5<=age<=120



def verified_rebalance_catchup(observation, day, r, now, state):
    """Fail-closed, future runner-supplied history/position reconciliation contract.

    This checks ONLY the fields and consistency of a purported reconciliation.
    It CANNOT authenticate those fields. The production organizer adapter MUST
    build them directly from real GET responses with a complete history, plus
    an independently enforced single-writer lock. User-provided input must never
    be accepted as proof for a live upload.
    """
    if day not in (OFFICIAL_DAYS[5], OFFICIAL_DAYS[10]) or r<=1:
        return False
    if str(state.get('last_executed_round_id') or '').startswith(f'official-{day}-r'):
        return False
    ev=observation.get('rebalance_reconciliation') if isinstance(observation,dict) else None
    if not isinstance(ev,dict) or ev.get('source')!='AUTHENTICATED_ORGANIZER_GET':
        return False
    if ev.get('round_id')!=f'official-{day}-r{r}':
        return False
    for flag in ('prior_rounds_finalized', 'no_earlier_executed_primary_trade',
                 'current_round_no_attributable_attempt', 'portfolio_reconciled',
                 'single_writer_lock_held'):
        if ev.get(flag) is not True:
            return False
    try:checked=base.utc_stamp(ev['checked_utc'])
    except (KeyError,TypeError,ValueError):return False
    age=(now-checked).total_seconds()
    return -5<=age<=120


def evaluate(observation,snapshot,state,now=None):
    """PURE proposal. State must be committed by external reconciled runner.

    Returns {action: HOLD|PRIMARY_TARGET|EQUAL80_RECOVERY_TARGET,
             target: dict|None, next_state: dict, note: str}.
    Crucially, an intent to trade is NOT proof that any organizer trade executed.
    """
    now=now or datetime.now(UTC)
    if now.tzinfo is None:raise ValueError('Naive clock')
    st=copy.deepcopy(state)
    try:state_valid(st)
    except Exception as e:
        return {'action':'HOLD','target':None,'next_state':copy.deepcopy(state),'note':'INVALID_STATE_'+type(e).__name__}
    def result(action='HOLD',target=None,note=''):
        return {'action':action,'target':target,'next_state':st,'note':note}
    try:
        day,r,cur=live_round(observation,now)
    except Exception as e:
        return result(note='INVALID_OR_UNAVAILABLE_OBSERVATION_'+type(e).__name__)
    idx=OFFICIAL_DAYS.index(day)
    # Fail-closed if state and verified current holdings conflict.
    actual_equity=sum(cur.values())
    if st['mode']=='PRE_ENTRY':
        if st['first_execution_confirmed'] or actual_equity>.0001:
            return result(note='PRE_ENTRY_STATE_PORTFOLIO_CONFLICT')
        if day!=OFFICIAL_DAYS[0]:
            return result(note='NO_NEXT_DAY_INITIAL_ENTRY_POLICY')
        if r>1 and not verified_late_entry_clearance(observation,day,r,now):
            return result(note='LATE_ENTRY_REQUIRES_AUTHENTICATED_ROUND_RECONCILIATION')
        try:target=good_snapshot(snapshot,now,day)
        except Exception as e:return result(note='PREENTRY_WAIT_FOR_PRIMARY_'+type(e).__name__)
        return result('PRIMARY_TARGET',target,'INITIAL_SLEEVE_ALLOCATION_PENDING_EXECUTION')
    if actual_equity<.01:
        return result(note='POSTENTRY_PORTFOLIO_IS_CASH_OR_UNCONFIRMED')
    if st['mode']=='EQUAL80_LOCKED':return result(note='RECOVERY_COMPLETED_HOLD_REMAINDER')
    # A recovery target may be issued by the external runner only once it has
    # confirmed that no prior attempt already consumed this decision slot.
    if st['mode']=='RECOVERY_PENDING':
        if r!=1:return result(note='RECOVERY_PENDING_WAIT_R1')
        target={s:float(.8/30) for s in SYMBOLS}
        if sum(abs(cur[s]-target[s]) for s in SYMBOLS)<.002:
            return result(note='EQUAL80_ALREADY_NEAR_TARGET_REQUIRE_EXECUTION_RECONCILIATION')
        return result('EQUAL80_RECOVERY_TARGET',target,'ONE_WAY_RECOVERY_REQUIRES_CONFIRMED_EXECUTION')
    # PRIMARY: R2-R7 catch-up exists ONLY on the two scheduled rebalance days.
    # Recompute the *same prior-close target*, never an intraday alpha signal.
    # The production writer must authenticate the reconciliation itself.
    if r!=1:
        if idx not in (5,10):return result(note='SCHEDULED_R2_R7_HOLD')
        if not verified_rebalance_catchup(observation,day,r,now,st):
            return result(note='REBALANCE_CATCHUP_REQUIRES_AUTHENTICATED_RECONCILIATION')
        try:target=good_snapshot(snapshot,now,day)
        except Exception as e:return result(note='REBALANCE_CATCHUP_DATA_NOT_READY_'+type(e).__name__)
        desired=trim_target(cur,target,initial=False)
        st['last_health_day']=day
        st['consecutive_primary_fail_days']=0
        if desired is None:return result(note='REBALANCE_CATCHUP_BELOW_3PCT_BAND_HOLD')
        return result('PRIMARY_TARGET',desired,'SLEEVE_FIVEDAY_SAME_DAY_CATCHUP_PENDING_EXECUTION')
    # Daily R1 telemetry counts at most once. A failed scheduled review gets
    # its same-day R2-R7 recovery opportunity before terminal-mode escalation.
    if st.get('last_health_day')==day:
        return result(note='R1_ALREADY_PROCESSED_WAIT_FOR_RECONCILIATION')
    try:target=good_snapshot(snapshot,now,day)
    except Exception as e:
        st['last_health_day']=day
        st['consecutive_primary_fail_days']+=1
        if st['consecutive_primary_fail_days']>=2:
            if idx in (5,10):
                # Preserve same-day opportunity to repair data and execute the
                # planned rebalance at R2-R7. Recheck persistent failure at
                # the next day's R1 if it never recovers.
                return result(note='REVIEW_DAY_FAILURE_DEFER_RECOVERY_ALLOW_CATCHUP_'+type(e).__name__)
            st['mode']='RECOVERY_PENDING'
            # Safety: HOLD in the round that flips state. A separate authenticated
            # next-round decision must reconcile the prior outcome first.
            return result(note='PERSISTENT_FAILURE_ENTER_RECOVERY_PENDING_'+type(e).__name__)
        return result(note='TRANSIENT_FAILURE_HOLD_'+type(e).__name__)
    st['consecutive_primary_fail_days']=0
    st['last_health_day']=day
    if idx not in (5,10):return result(note='HEALTHY_NON_REBALANCE_DAY_HOLD')
    desired=trim_target(cur,target,initial=False)
    if desired is None:return result(note='BELOW_3PCT_L1_TRADE_BAND_HOLD')
    return result('PRIMARY_TARGET',desired,'SLEEVE_FIVEDAY_R1_REBALANCE_PENDING_EXECUTION')


def confirm_execution(state,kind,day,round_id,verified):
    """Test helper. In production, ONLY reconciled organizer EXECUTED evidence
    may supply verified=True, never a local callback return or upload receipt.
    """
    st=copy.deepcopy(state);state_valid(st)
    if verified is not True or day not in OFFICIAL_DAYS or not round_id:
        raise ValueError('EXECUTED organizer evidence required')
    if kind=='initial' and st['mode']=='PRE_ENTRY':
        st.update(mode='PRIMARY',first_execution_confirmed=True,
                  first_execution_day=day,last_executed_round_id=round_id)
    elif kind=='recovery' and st['mode']=='RECOVERY_PENDING':
        st.update(mode='EQUAL80_LOCKED',recovery_execution_confirmed=True,
                  last_executed_round_id=round_id)
    elif kind=='primary_rebalance' and st['mode']=='PRIMARY':
        st['last_executed_round_id']=round_id
    else:raise ValueError('Invalid mode transition')
    return st


def demo_obs(day='2026-10-12',r=1,amount=0.,now=None):
    h,m=map(int,base.DEADLINES[r].split(':'))
    deadline=datetime.fromisoformat(day).replace(hour=h,minute=m,tzinfo=ET)
    return {'round':{'phase':'official','day':day,'number':r,
       'deadline':deadline.isoformat()},'portfolio':{'phase':'official',
       'current_weights':{s:float(amount/30) for s in SYMBOLS}}}


def regression_self_test():
    import numpy as np
    x=np.array([[80.+j for j in range(30)] for i in range(121)],float)
    for i in range(121):x[i]*=np.exp(.001*i+.005*np.sin(i/8.+np.arange(30)))
    target,selected,core=sleeve_target_from_daily(x)
    assert len(selected)==5 and len(set(selected))==5
    assert abs(sum(target.values())-.8)<1e-9
    a=np.array([core[s] for s in SYMBOLS]);res=np.array([target[s] for s in SYMBOLS])
    assert np.max(np.abs(res-a))>.005
    now=datetime(2026,10,12,8,58,tzinfo=ET)
    snap={'schema':2,'purpose':'LIVE_READONLY_VERIFIED','strategy':SNAP_STRATEGY,
          'created_utc':now.astimezone(UTC).isoformat(),'last_complete_day_et':'2026-10-09',
          'complete_days':243,'lookback_sha256':'a'*64,'target_weights':target}
    st=sample_state()
    obs=demo_obs()
    q=evaluate(obs,snap,st,now);assert q['action']=='PRIMARY_TARGET'
    badsnap=dict(snap);badsnap['created_utc']=(now-timedelta(hours=3)).astimezone(UTC).isoformat()
    assert evaluate(obs,badsnap,st,now)['action']=='HOLD' # no pretrade Equal80
    assert evaluate(demo_obs(r=2),snap,st,now)['action']=='HOLD'
    # Same-day initial catch-up: requires a live, authoritative reconciliation
    # that Day1 R1 has finished without a trade and R2 is untouched.
    r2_time=datetime(2026,10,12,10,10,tzinfo=ET)
    r2=demo_obs('2026-10-12',2)
    good_ev={'source':'AUTHENTICATED_ORGANIZER_GET',
        'round_id':'official-2026-10-12-r2',
        'earlier_rounds_finalized':True,
        'earlier_rounds_no_executed_trade':True,
        'current_round_no_attributable_attempt':True,
        'verified_cash_portfolio':True,
        'checked_utc':(r2_time-timedelta(seconds=30)).astimezone(UTC).isoformat()}
    fresh=dict(snap,created_utc=(r2_time-timedelta(minutes=1)).astimezone(UTC).isoformat())
    assert evaluate(r2,fresh,st,r2_time)['action']=='HOLD'
    r2['preentry_reconciliation']=good_ev
    assert evaluate(r2,fresh,st,r2_time)['action']=='PRIMARY_TARGET'
    for field in ('earlier_rounds_finalized','earlier_rounds_no_executed_trade',
                  'current_round_no_attributable_attempt','verified_cash_portfolio'):
        e=copy.deepcopy(r2);e['preentry_reconciliation'][field]=False
        assert evaluate(e,fresh,st,r2_time)['action']=='HOLD'
    e=copy.deepcopy(r2);e['preentry_reconciliation']['checked_utc']=(r2_time-timedelta(minutes=4)).astimezone(UTC).isoformat()
    assert evaluate(e,fresh,st,r2_time)['action']=='HOLD'
    e=copy.deepcopy(r2);e['portfolio']['current_weights']={sym:float(.8/30) for sym in SYMBOLS}
    assert evaluate(e,fresh,st,r2_time)['action']=='HOLD'
    # Remaining Day 1 windows are also valid as catch-up candidates only if
    # each has its OWN new organizer reconciliation; never reuse an R2 proof.
    for rr in range(3,8):
        deadline_h,deadline_m=map(int,base.DEADLINES[rr].split(':'))
        t=datetime(2026,10,12,deadline_h,deadline_m-10,tzinfo=ET)
        later=demo_obs('2026-10-12',rr)
        snap_later=dict(snap,created_utc=(t-timedelta(minutes=1)).astimezone(UTC).isoformat())
        ev_later=dict(good_ev,round_id=f'official-2026-10-12-r{rr}',
                      checked_utc=(t-timedelta(seconds=30)).astimezone(UTC).isoformat())
        later['preentry_reconciliation']=ev_later
        assert evaluate(later,snap_later,st,t)['action']=='PRIMARY_TARGET'
        later['preentry_reconciliation']=dict(ev_later,round_id='official-2026-10-12-r2')
        assert evaluate(later,snap_later,st,t)['action']=='HOLD'
    # An R2 first execution is recognized without ever using Equal80.
    confirmed_r2=confirm_execution(sample_state(),'initial','2026-10-12',
                                   'official-2026-10-12-r2',verified=True)
    assert confirmed_r2['mode']=='PRIMARY'
    # No retroactive attempt after the 10:25 deadline buffer.
    assert evaluate(r2,fresh,st,datetime(2026,10,12,10,23,tzinfo=ET))['action']=='HOLD'
    # Still no blanket permission to initiate on Day 2.
    assert evaluate(demo_obs('2026-10-13',1),snap,st,datetime(2026,10,13,8,55,tzinfo=ET))['action']=='HOLD'
    st=confirm_execution(st,'initial','2026-10-12','official-2026-10-12-r1',verified=True)
    assert st['mode']=='PRIMARY'
    after=demo_obs('2026-10-13',1,.8);tomorrow=datetime(2026,10,13,8,55,tzinfo=ET)
    fail1=evaluate(after,None,st,tomorrow)
    assert fail1['action']=='HOLD' and fail1['next_state']['consecutive_primary_fail_days']==1
    # One recovered data check breaks the failure streak.
    recovered=dict(snap,created_utc=datetime(2026,10,14,8,54,tzinfo=ET).astimezone(UTC).isoformat(),last_complete_day_et='2026-10-13')
    reset=evaluate(demo_obs('2026-10-14',1,.8),recovered,fail1['next_state'],datetime(2026,10,14,8,55,tzinfo=ET))
    assert reset['next_state']['consecutive_primary_fail_days']==0 and reset['next_state']['mode']=='PRIMARY'
    day3=demo_obs('2026-10-14',1,.8);third=datetime(2026,10,14,8,55,tzinfo=ET)
    fail2=evaluate(day3,None,fail1['next_state'],third)
    assert fail2['action']=='HOLD' and fail2['next_state']['mode']=='RECOVERY_PENDING'
    assert evaluate(demo_obs('2026-10-14',2,.8),snap,fail2['next_state'],third)['action']=='HOLD'
    day4=demo_obs('2026-10-15',1,.8)
    day4['portfolio']['current_weights'][SYMBOLS[0]]+=.04
    day4['portfolio']['current_weights'][SYMBOLS[1]]-=.02
    day4['portfolio']['current_weights'][SYMBOLS[2]]-=.02
    fourth=datetime(2026,10,15,8,55,tzinfo=ET)
    equal=evaluate(day4,None,fail2['next_state'],fourth)
    assert equal['action']=='EQUAL80_RECOVERY_TARGET'
    assert abs(sum(equal['target'].values())-.8)<1e-9
    assert evaluate(day4,None,fail2['next_state'],fourth)['action']=='EQUAL80_RECOVERY_TARGET' # pure repeat proposal; writer must fence
    locked=confirm_execution(fail2['next_state'],'recovery','2026-10-15','official-2026-10-15-r1',verified=True)
    assert evaluate(demo_obs('2026-10-16',1,.8),snap,locked,datetime(2026,10,16,8,55,tzinfo=ET))['target'] is None
    for d,pre in [('2026-10-19','2026-10-16'),('2026-10-26','2026-10-23')]:
        assert expected_previous_day(d)==pre
    now6=datetime(2026,10,19,8,55,tzinfo=ET)
    cur={s:float(.8/30) for s in SYMBOLS}
    tgt={s:float((.8/30)+(.07 if s==SYMBOLS[0] else -.07/29)) for s in SYMBOLS}
    v=trim_target(cur,tgt);assert v is not None
    assert abs(sum(abs(v[s]-cur[s]) for s in SYMBOLS)-.10)<1e-9
    assert trim_target(cur,cur) is None
    try:
        confirm_execution(sample_state(),'initial','2026-10-12','test',False)
    except ValueError:pass
    else:raise AssertionError('Unverified execution accepted')
    wrong=demo_obs();wrong['round']['phase']='validation'
    assert evaluate(wrong,snap,sample_state(),now)['action']=='HOLD'
    print('RELEASE_V23_SELF_TEST_PASS')
    print('SAME_DAY_R2_ENTRY_AND_REBALANCE_RECONCILIATION_GATES_PASS')
    print('PHASE44_SLEEVE_15PCT_EQUITY_STABLE_TOP5_R1_DAY1_6_11_BAND_CAP_PASS')
    print('PREENTRY_NO_EQUAL80_TWO_R1_FAILURES_EQUAL80_ONEWAY_CONFIRMED_EXECUTION_PASS')
    print('NO_ORGANIZER_NO_LLM_NO_SSH_NO_NETWORK_NO_TRADES')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['self-test','inspect-csv','offline-dry-run'])
    p.add_argument('--csv',type=Path)
    args=p.parse_args()
    if args.action=='self-test':regression_self_test();return
    if not args.csv or not args.csv.is_file():p.error('Must provide --csv original Alpaca 15-minute bars')
    rows=list(base.load_csv(args.csv))
    snap=snapshot_from_bars(rows,datetime.now(UTC))
    print('COMPLETED_DAYS',snap['complete_days'],'LAST_COMPLETE',snap['last_complete_day_et'])
    print('SLEEVE_SELECTED_FIVE',','.join(snap['selected_five']))
    print('SLEEVE_TOTAL_EQUITY',sum(snap['target_weights'].values()))
    print('TOP_TARGET_WEIGHTS',json.dumps(dict(sorted(snap['target_weights'].items(),key=lambda x:-x[1])[:8]),sort_keys=True))
    if args.action=='offline-dry-run':
        # Deliberately mark offline packet as NOT authenticated live data.
        now=datetime(2026,10,12,8,55,tzinfo=ET)
        o=evaluate(demo_obs(),snap,sample_state(),now)
        print('OFFLINE_PACKET_CANNOT_TRADE',o['action'],o['note'])
    print('OFFLINE_ONLY_NO_ORGANIZER_NO_SUBMISSION')

if __name__=='__main__': main()
