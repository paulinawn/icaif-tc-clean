#!/usr/bin/env python3
"""Offline: no network, no secrets, no persistent state, no HTTP POST."""
import sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import icaif_v37_production_candidate as a
import icaif_v39_live_writer as writer

def row(day,num):return {'id':f'official-{day}-r{num}','day':day,'number':num,'status':'OPEN','phase':'official'}
base={'mode':'PRIMARY','first_execution_confirmed':True,'last_executed_round_id':'official-2026-10-12-r1'}
after_r3=dict(base,last_executed_round_id='official-2026-10-19-r3')
assert a.eligible_for_market_refresh(row('2026-10-19',3),base)
assert not a.eligible_for_market_refresh(row('2026-10-19',4),after_r3)
for n in (2,3,4,5,6,7):
    assert not a.eligible_for_market_refresh(row('2026-10-19',n),after_r3)
assert a.eligible_for_market_refresh(row('2026-10-26',1),after_r3)
assert a.eligible_for_market_refresh(row('2026-10-26',2),after_r3)
after_oct26_r1=dict(after_r3,last_executed_round_id='official-2026-10-26-r1')
for n in range(2,8):assert not a.eligible_for_market_refresh(row('2026-10-26',n),after_oct26_r1)
assert a.eligible_for_market_refresh(row('2026-10-26',1),after_oct26_r1)
assert a.eligible_for_market_refresh(row('2026-10-19',3),dict(base,last_executed_round_id='official-2026-10-12-r1'))
assert not a.eligible_for_market_refresh(row('2026-10-21',4),after_r3)
assert not a.eligible_for_market_refresh(row('2026-10-12',4),after_r3)
assert a.eligible_for_market_refresh(row('2026-10-12',4),{'mode':'PRE_ENTRY','last_executed_round_id':None})
assert not a.eligible_for_market_refresh(None,after_r3)
assert not a.eligible_for_market_refresh(row('2026-10-19',4),dict(after_r3,mode='EQUAL80_LOCKED'))
print('V3933_PURE_POSTEXEC_HOLD_AND_OCT26_R1_ELIGIBILITY_PASS',flush=True)

# Verify that the real V39 cycle does not fetch Alpaca data, enter the target
# decision path, or POST after locally confirmed same-day execution. Mock ONLY
# the I/O/reconciliation boundary, never replace the tested eligibility code.
original_collect=a.collect; original_reconcile=a.reconcile
original_initial=writer.initial_only;original_fence=writer.native_operations_fenced
try:
  now=datetime(2026,10,19,16,5,tzinfo=timezone.utc)
  sched={'current_round':row('2026-10-19',4),'rounds':[]}
  port={'phase':'official','positions':[]}
  hist={'decisions':[],'has_more':False}
  ledger={'phase':'official','rounds':[],'transactions':[]}
  a.collect=lambda *args:(sched,port,hist,ledger)
  a.reconcile=lambda s,j,*args:j
  writer.initial_only=lambda j,*args:j
  writer.native_operations_fenced=lambda *args:{}
  class Session:
    def round(self,rid):return {'decisions':[],'execution':None}
    def _decision(self,*args):raise AssertionError('FORBIDDEN_UPLOAD')
  class Original:
    def authenticated_round(self,s,o,symbols,local_time):return s['current_round'],now
  class Refresh:
    def run(self,*args,**kwargs):raise AssertionError('FORBIDDEN_PRICE_REFRESH')
  class Engine:
    OFFICIAL_DAYS=('2026-10-19','2026-10-26')
    SYMBOLS=tuple('AAPL AMZN BA BAC CAT CRM CVX DIS GE GOOGL GS INTC JNJ JPM KO LLY META MSFT NEE NKE NVDA PFE PYPL T TMO TSLA UNH V WMT XOM'.split())
  j={'state':after_r3,'intents':{},'execution_confirmations':{}}
  output,action=writer.cycle(Session(),j,live=True,engine=Engine(),machine=object(),original=Original(),prior=object(),refresh=Refresh(),save=lambda x:None)
  assert output == j and action=='HOLD'
finally:
  a.collect=original_collect;a.reconcile=original_reconcile
  writer.initial_only=original_initial;writer.native_operations_fenced=original_fence
print('V3933_REAL_V39_CYCLE_POSTEXEC_NO_ALPACA_NO_POST_PASS',flush=True)
print('V3933_OFFLINE_TEST_PASS_NO_NETWORK_NO_PRIVATE_STATE_CHANGE',flush=True)
