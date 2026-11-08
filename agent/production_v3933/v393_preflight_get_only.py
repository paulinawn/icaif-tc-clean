#!/usr/bin/env python3
# Role: PREFLIGHT CHECK - read-only verification used before the round-3 catch-up repair.
"""Strict October 19 R2 readiness review: organizer/Alpaca GET only, no POST/write."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import os,sys,tempfile
APP=Path('/opt/icaif2026');RELEASE=Path(__file__).resolve().parent
sys.path[:0]=[str(APP),str(RELEASE),str(APP/'official_staging_v23_unarmed'/'code'),str(RELEASE/'code')]

def verify_pre_r3_window(schedule):
    """Accept only the window after October 19 R2 completed, before R3 timer.
    This is a deployment audit, not permission to upload a trading decision.
    """
    stamp=schedule.get('current_time')
    if not isinstance(stamp,str):raise AssertionError('INVALID_ORGANIZER_CLOCK')
    try:
        now=datetime.fromisoformat(stamp.replace('Z','+00:00'))
        if now.tzinfo is None:raise ValueError('no timezone')
        now=now.astimezone(timezone.utc)
    except (TypeError,ValueError):raise AssertionError('INVALID_ORGANIZER_CLOCK')
    lo=datetime(2026,10,19,14,30,tzinfo=timezone.utc)
    hi=datetime(2026,10,19,15,2,tzinfo=timezone.utc)
    if not (lo<=now<hi):raise AssertionError('NOT_PRE_R3_DEPLOYMENT_WINDOW')
    rd=schedule.get('current_round')
    cid=rd.get('id') if isinstance(rd,dict) else None
    if cid not in (None,'official-2026-10-19-r2','official-2026-10-19-r3'):
        raise AssertionError('UNEXPECTED_CURRENT_ROUND')
    rounds=schedule.get('rounds')
    if not isinstance(rounds,list):raise AssertionError('ROUND_SCHEDULE_MISSING')
    for n in (1,2):
        rid=f'official-2026-10-19-r{n}'
        rows=[r for r in rounds if isinstance(r,dict) and r.get('id')==rid]
        if len(rows)!=1 or rows[0].get('status')!='COMPLETED':
            raise AssertionError('PRIOR_ROUND_NOT_COMPLETED')
    rid='official-2026-10-19-r3'
    r3=[r for r in rounds if isinstance(r,dict) and r.get('id')==rid]
    if (len(r3)!=1 or r3[0].get('phase')!='official' or
            r3[0].get('day')!='2026-10-19' or r3[0].get('number')!=3 or
            r3[0].get('status') not in ('SCHEDULED','OPEN')):
        raise AssertionError('R3_SCHEDULE_NOT_VERIFIED')
    return cid or 'NONE'

def check():
 import httpx
 import icaif_v29_bridge_unarmed as bridge
 import icaif_v37_production_candidate as adapter
 import icaif_v33_state_machine as machine
 import icaif_official_sleeve_engine_v23 as engine
 import icaif_v28_clean_alpaca_bootstrap as refresh
 assert Path(refresh.__file__).resolve()==(RELEASE/'code'/'icaif_v28_clean_alpaca_bootstrap.py').resolve(), 'WRONG_REFRESH_MODULE_LOADED'
 from icaif_v30_remote_get_only import GetOnlyTransport
 from kit.config import load_environment
 from kit.original_client import OriginalSession,load_profile
 private=APP/'.icaif'/'official_v39'
 journal=machine.read_journal(private/'execution_journal.json',engine)
 assert journal['state']['mode']=='PRIMARY' and journal['state']['first_execution_confirmed'] is True,'STATE_NOT_CONFIRMED_PRIMARY'
 assert journal['state']['last_executed_round_id']=='official-2026-10-12-r1','DIFFERENT_LAST_EXECUTED_TRADE'
 assert 'official-2026-10-12-r1' in journal['execution_confirmations'],'FIRST_TRADE_NOT_CONFIRMED'
 assert not (set(journal['intents'])-set(journal['execution_confirmations'])),'PENDING_UPLOAD_INTENT'
 load_environment(APP/'.env')
 token=os.environ.get('CODABENCH_TOKEN','')
 profile=Path(os.environ.get('ICAIF_PROFILE',''))
 if not profile.is_absolute():profile=APP/profile
 assert token and profile.is_file(),'PRIVATE_AUTH_MISSING'
 gate=GetOnlyTransport(httpx.HTTPTransport())
 with tempfile.TemporaryDirectory(prefix='v393_getonly_') as scratch:
  with OriginalSession(profile=load_profile(profile),token=token,
       checkpoint=Path(scratch)/'probe_checkpoint.json',credentials=APP/'.icaif'/'credentials.json',
       transport=gate,timeout=18,result_timeout=15) as session:
   schedule,portfolio,history,ledger=adapter.collect(session,machine)
   current=schedule.get('current_round')
   current_id=current.get('id') if isinstance(current,dict) else None
   print('V3932_CURRENT_ROUND',current_id or 'NONE',flush=True)
   verified_current=verify_pre_r3_window(schedule)
   print('V3932_PRE_R3_SCHEDULE_ACCEPTED',verified_current,flush=True)
   for n in (1,2):
    rid=f'official-2026-10-19-r{n}'
    own=session.round(rid)
    adapter.verified_completed_missing_hold(own,rid,ledger)
    assert all(d.get('round_id')!=rid for d in history['decisions']), 'UNFILTERED_PRIOR_MATERIAL_HISTORY'
   actual=history['decisions']
   assert len(actual)==1 and actual[0].get('round_id')=='official-2026-10-12-r1','UNEXPECTED_MATERIAL_SUBMISSION'
   machine.enforce_no_unjournaled_decisions(journal,actual)
   _=adapter.reconcile(session,journal,schedule,portfolio,history,ledger,machine,engine)
   third=[r for r in schedule.get('rounds',[]) if r.get('id')=='official-2026-10-19-r3']
   assert len(third)==1,'R3_SCHEDULE_MISSING'
   adapter.live_evidence(session,machine,schedule,history,ledger,portfolio,third[0],datetime.now(timezone.utc),journal['state'])
   print('V3932_R3_AUTHENTICATED_CATCHUP_EVIDENCE_PASS',flush=True)
   _=bridge.normalize_portfolio(portfolio,engine.SYMBOLS,journal['state'])
   print('V393_HIGH_PRECISION_PORTFOLIO_NORMALIZATION_PASS',flush=True)
   assert len(portfolio['positions'])==30, 'POSITIONS_NOT_30'
   # Confirm verified prior data cache and the CURRENT session can be prepared
   # using the FIXED refresher WITHOUT overwriting the production cache.
   _rows,snap,rec=refresh.prepare(datetime.now(timezone.utc),mode='refresh')
   assert snap['last_complete_day_et']=='2026-10-16','UNEXPECTED_PREVIOUS_MARKET_DAY'
   assert abs(sum(snap['target_weights'].values())-.8)<1e-9,'MARKET_TARGET_ILLEGAL'
   print('V393_ALPACA_CURRENT_PRIOR_SESSION',rec['last_complete_day_et'],flush=True)
   print('V393_ALPACA_GET_PREPARE_PASS_NO_CACHE_WRITE',flush=True)
 assert gate.attempts and set(gate.attempts)=={'GET'},'NON_GET_ORGANIZER_OPERATION'
 print('V3932_R3_AUTHENTICATED_READONLY_PREFLIGHT_PASS_NO_POST',flush=True)

if __name__=='__main__':
 try:check()
 except Exception as e:
  msg=str(e)
  allowed=('STATE_NOT_CONFIRMED_PRIMARY','DIFFERENT_LAST_EXECUTED_TRADE',
           'FIRST_TRADE_NOT_CONFIRMED','PENDING_UPLOAD_INTENT','PRIVATE_AUTH_MISSING',
           'INVALID_ORGANIZER_CLOCK','NOT_PRE_R3_DEPLOYMENT_WINDOW',
           'UNEXPECTED_CURRENT_ROUND','ROUND_SCHEDULE_MISSING','PRIOR_ROUND_NOT_COMPLETED',
           'R3_SCHEDULE_NOT_VERIFIED','R3_SCHEDULE_MISSING',
           'UNFILTERED_PRIOR_MATERIAL_HISTORY','UNEXPECTED_MATERIAL_SUBMISSION',
           'POSITIONS_NOT_30','UNEXPECTED_PREVIOUS_MARKET_DAY','MARKET_TARGET_ILLEGAL',
           'NON_GET_ORGANIZER_OPERATION','WRONG_REFRESH_MODULE_LOADED')
  import re
  code=(msg if isinstance(msg,str) and re.fullmatch(r'[A-Z][A-Z0-9_]{3,90}',msg) else 'CHECK_FAILED')
  print('V3932_PREFLIGHT_FAIL_CLOSED',type(e).__name__,code,flush=True)
  raise SystemExit(2)
