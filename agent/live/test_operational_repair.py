#!/usr/bin/env python3
"""Offline tests, no network, no credentials and no writes to trading state."""
from pathlib import Path
import hashlib
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import organizer_bridge as bridge
import round_gatekeeper as adapter

for n in ('0.'+'0'*40+'1','999999.1234567890123456789012345678901234',
          '0.12345678901234567890123456789012345678901'):
    assert bridge.as_decimal(n).is_finite()
for n in ('nan','Infinity','-1.0','+2','2e3','2,000.00',' 2.00','2.00 ',
          '0.'+'1'*65,'0'+'0'*33+'.1','not a number'):
    try:bridge.as_decimal(n)
    except bridge.ReadOnlyGate:pass
    else:raise AssertionError('UNSAFE_INPUT_PASSED')
print('V393_HIGH_PRECISION_NUMERIC_STRICT_FINITE_DECIMAL_PASS',flush=True)

R1='official-2026-10-19-r1'
synthetic={'round_id':R1,'decision_id':'synthetic-planned-round',
           'status':'MISSING','selection_status':None,'execution_status':None,
           'submitted_at':'2026-10-19T13:02:00Z'}

adapter.verified_own_no_upload({'decisions':[],'execution':None},R1,allow_missing_record=True)
adapter.verified_own_no_upload({'decisions':[synthetic],'execution':None},R1,allow_missing_record=True)
s=synthetic.copy();s['selection_status']='UNKNOWN';s['execution_status']='HELD'
adapter.verified_own_no_upload({'decisions':[s],'execution':None},R1,allow_missing_record=True)

def rejects(own,code,*,allow=True):
    try:adapter.verified_own_no_upload(own,R1,allow_missing_record=allow)
    except adapter.Stop as exc:
        assert str(exc)==code,(code,str(exc));return
    raise AssertionError('DID_NOT_FAIL_'+code)
rejects({'decisions':[synthetic],'execution':None},'OWN_ROUND_HAS_UNCERTAIN_DECISION',allow=False)
rejects({'decisions':[dict(synthetic,uploaded_at='2026-10-19T13:01Z')],'execution':None},'OWN_MISSING_HAS_UPLOAD_ARTIFACT')
rejects({'decisions':[dict(synthetic,receipt_status='INVALID')],'execution':None},'OWN_MISSING_HAS_UPLOAD_ARTIFACT')
rejects({'decisions':[dict(synthetic,status='INVALID')],'execution':None},'OWN_MISSING_RECORD_NOT_CONFIRMED')
rejects({'decisions':[dict(synthetic,selection_status='SELECTED')],'execution':None},'OWN_MISSING_RECORD_NOT_CONFIRMED')
rejects({'decisions':[dict(synthetic,execution_status='EXECUTED')],'execution':None},'OWN_MISSING_RECORD_NOT_CONFIRMED')
rejects({'decisions':[synthetic],'execution':{'id':'real'}},'EARLIER_OR_CURRENT_EXECUTION_PRESENT')
rejects({'decisions':[synthetic,synthetic],'execution':None},'OWN_ROUND_HAS_UNCERTAIN_DECISION')
rejects({'decisions':[dict(synthetic,round_id='wrong')],'execution':None},'OWN_MISSING_RECORD_NOT_CONFIRMED')
rejects({'decisions':[dict(synthetic,validation_status='VALID')],'execution':None},'OWN_MISSING_HAS_VALIDATION_RECEIPT')
print('V393_SYNTHETIC_OWN_ROUND_ONLY_REAL_UPLOAD_STILL_FAILS_CLOSED_PASS',flush=True)

old=Path('/opt/icaif2026/official_clean_bootstrap_v28_unarmed/data')
assert adapter.DATA==old
assert adapter.V28==Path('/opt/icaif2026/official_v3932_release/code')
source=(Path(__file__).resolve().parent/'market_data_fetcher.py').read_text()
assert 'Unarmed synthetic decision does not match market snapshot' not in source
assert 'engine.snapshot_from_bars(rows,now' in source
assert 'engine.validate(snap[' in source
assert 'Cache hash mismatch: refusing refresh' in source
assert 'stale' not in '' # no effect
print('V393_ONLY_OBSOLETE_OCT12_SYNTHETIC_GATE_REMOVED_ORIGINAL_DATA_PATH_RETAINED_PASS',flush=True)
print('V393_OFFLINE_TEST_NO_HTTP_NO_POST_NO_PRIVATE_STATE_CHANGE',flush=True)

# V3.9.3.1: the finalized R1 may be MISSING/None/None rather than MISSING/UNKNOWN/HELD.
# Only a COMPLETED schedule row with no receipt or upload artifact is omitted.
import state_machine as machine
completed={R1}
assert adapter.organizer_held_placeholder(synthetic, completed)
assert not adapter.organizer_held_placeholder(dict(synthetic,status='VALID'),completed)
assert not adapter.organizer_held_placeholder(dict(synthetic,execution_status='EXECUTED'),completed)
for field,val in (('uploaded_at','2026-10-19T12:57:00Z'),('submission_id',1234),('validation_status','VALID')):
    try:adapter.organizer_held_placeholder(dict(synthetic,**{field:val}),completed)
    except adapter.Stop:pass
    else:raise AssertionError('UNSAFE_R1_SYNTHETIC_ROW_PASSED_'+field)
class SyntheticSession:
    def decisions(self,phase,offset,limit):
        assert phase=='official' and offset==0 and limit==200
        return {'decisions':[{'round_id':'official-2026-10-12-r1','decision_id':'real-1581','status':'VALID','selection_status':'SELECTED','execution_status':'EXECUTED'},synthetic], 'offset':0, 'limit':200, 'has_more':False}
filtered=adapter.read_pages(SyntheticSession(),machine,{'rounds':[{'id':'official-2026-10-12-r1','status':'COMPLETED','phase':'official'},{'id':R1,'status':'COMPLETED','phase':'official'}]})
assert len(filtered['decisions'])==1
print('V3931_COMPLETED_MISSING_NONE_NONE_HISTORY_FILTER_PASS',flush=True)

from catchup_preflight import verify_pre_r3_window
from datetime import datetime,timezone
r1={'id':R1,'phase':'official','day':'2026-10-19','number':1,'status':'COMPLETED'}
r2={'id':'official-2026-10-19-r2','phase':'official','day':'2026-10-19','number':2,'status':'COMPLETED'}
r3={'id':'official-2026-10-19-r3','phase':'official','day':'2026-10-19','number':3,'status':'SCHEDULED'}
base={'current_time':'2026-10-19T14:40:00+00:00','current_round':None,'rounds':[r1,r2,r3]}
assert verify_pre_r3_window(base)=='NONE'
for fixture,err in [({'current_time':'2026-10-19T14:05:00+00:00'},'NOT_PRE_R3_DEPLOYMENT_WINDOW'),
                    ({'current_time':'2026-10-19T15:02:00+00:00'},'NOT_PRE_R3_DEPLOYMENT_WINDOW'),
                    ({'current_round':{'id':'official-2026-10-19-r4'}},'UNEXPECTED_CURRENT_ROUND'),
                    ({'rounds':[r1,dict(r2,status='OPEN'),r3]},'PRIOR_ROUND_NOT_COMPLETED'),
                    ({'rounds':[r1,r2,dict(r3,status='COMPLETED')]},'R3_SCHEDULE_NOT_VERIFIED')]:
    try:verify_pre_r3_window(dict(base,**fixture))
    except AssertionError as exc:assert str(exc)==err,(str(exc),err)
    else:raise AssertionError('UNSAFE_R3_GATE_'+err)

# Validate synthetic zero-trade accounting and rejection of any real execution.
from decimal import Decimal
RID='official-2026-10-19-r1'
ledgerrow={'round_id':RID,'decision_status':'MISSING','fee':'0.00',
           'traded_notional':'0.0000','cash_before':'199840.13','cash_after':'199840.13'}
execution={'round_id':RID,'decision_status':'MISSING','fee':'0.00000',
           'traded_notional':'0','cash_before':'199840.13','cash_after':'199840.13'}
held={'decisions':[synthetic],'execution':execution,'trades':[]}
ledger={'rounds':[ledgerrow],'transactions':[]}
assert adapter.verified_completed_missing_hold(held,RID,ledger)
for field,value in [('fee','0.01'),('traded_notional','1.00'),('cash_after','199839.13'),('decision_status','VALID'),('round_id','other')]:
    bad=dict(held,execution=dict(execution,**{field:value}))
    try:adapter.verified_completed_missing_hold(bad,RID,ledger)
    except adapter.Stop:pass
    else:raise AssertionError('UNSAFE_SYNTHETIC_EXECUTION_ACCEPTED_'+field)
for field,value in [('fee','0.001'),('traded_notional','2.00'),('cash_after','199830.13'),('decision_status','INVALID')]:
    bad=dict(ledger,rounds=[dict(ledgerrow,**{field:value})])
    try:adapter.verified_completed_missing_hold(held,RID,bad)
    except adapter.Stop:pass
    else:raise AssertionError('UNSAFE_SYNTHETIC_LEDGER_ACCEPTED_'+field)
for bad in [dict(held,trades=[{'symbol':'AAPL'}]), dict(held,decisions=[]),
            dict(held,decisions=[dict(synthetic,uploaded_at='2026-10-19T13:00Z')]),
            dict(held,execution=None)]:
    try:adapter.verified_completed_missing_hold(bad,RID,ledger)
    except adapter.Stop:pass
    else:raise AssertionError('UNSAFE_SYNTHETIC_OWN_RECORD_ACCEPTED')
for bad in [dict(ledger,transactions=[{'round_id':RID,'symbol':'AAPL'}]),
            dict(ledger,rounds=[]),dict(ledger,rounds=[ledgerrow,ledgerrow]),
            dict(ledger,transactions=None)]:
    try:adapter.verified_completed_missing_hold(held,RID,bad)
    except adapter.Stop:pass
    else:raise AssertionError('UNSAFE_SYNTHETIC_LEDGER_RECORD_ACCEPTED')
print('V3932_SYNTHETIC_ZERO_ACCOUNTING_ONLY_AND_REJECT_NONZERO_PASS',flush=True)
print('V3932_R3_TIME_WINDOW_AND_PREVIOUS_ROUNDS_PASS',flush=True)

# Test actual two-round R3 catch-up GET-to-evidence path, not only helper predicates.
# On server, the engine lives under official_staging_v23_unarmed/code;
# the regression uses an isolated import via the normal installed source when present.
import types
_engine=types.ModuleType('sleeve_engine')
_engine.SYMBOLS=tuple('AAPL AMZN BA BAC CAT CRM CVX DIS GE GOOGL GS INTC JNJ JPM KO LLY META MSFT NEE NKE NVDA PFE PYPL T TMO TSLA UNH V WMT XOM'.split())
assert len(_engine.SYMBOLS)==30 and len(set(_engine.SYMBOLS))==30
sys.modules['sleeve_engine']=_engine
class FakeRoundSession:
    def __init__(self,objects):self.objects=objects
    def round(self,rid):return self.objects[rid]
    def round_calls(self):return []
def mkhold(rid):
    d=dict(synthetic,round_id=rid)
    ex=dict(execution,round_id=rid)
    return {'decisions':[d],'execution':ex,'trades':[]}
def mkledger(rid):return dict(ledgerrow,round_id=rid)
r1id='official-2026-10-19-r1';r2id='official-2026-10-19-r2';r3id='official-2026-10-19-r3'
rd1=dict(r1,id=r1id);rd2=dict(r2,id=r2id)
rd3=dict(r3,id=r3id,status='OPEN')
rd3['day']='2026-10-19';rd3['number']=3
session=FakeRoundSession({r1id:mkhold(r1id),r2id:mkhold(r2id),r3id:{'decisions':[],'execution':None,'trades':[]}})
ledger3={'rounds':[mkledger(r1id),mkledger(r2id)],'transactions':[]}
sched3={'rounds':[rd1,rd2,rd3]}
portfolio3={'phase':'official','current_weights':{k:.8/30 for k in _engine.SYMBOLS},'positions':[{'symbol':k,'shares':'1.0'} for k in _engine.SYMBOLS]}
state3={'mode':'PRIMARY','first_execution_confirmed':True,'last_executed_round_id':'official-2026-10-12-r1'}
proof=adapter.live_evidence(session,None,sched3,{'decisions':[],'has_more':False},ledger3,
                            portfolio3,rd3,datetime(2026,10,19,15,5,tzinfo=timezone.utc),state3)
assert proof['rebalance_reconciliation']['prior_rounds_finalized'] is True
assert proof['rebalance_reconciliation']['no_earlier_executed_primary_trade'] is True
print('V3932_R3_TWO_PRIOR_ZERO_TRADES_CATCHUP_EVIDENCE_PASS',flush=True)
for badsession,badledger,badstatus,reason in [
    (FakeRoundSession({**session.objects,r1id:dict(session.objects[r1id],trades=[{'symbol':'AAPL'}])}),ledger3,'COMPLETED','MISSING_HOLD_HAS_TRADES'),
    (session,{'rounds':[dict(mkledger(r1id),fee='0.1'),mkledger(r2id)],'transactions':[]},'COMPLETED','MISSING_HOLD_LEDGER_NOT_ZERO'),
    (session,ledger3,'OPEN','EARLIER_ROUND_NOT_FINALIZED'),
]:
    sched=dict(sched3,rounds=[dict(rd1,status=badstatus),rd2,rd3])
    try:adapter.live_evidence(badsession,None,sched,{'decisions':[],'has_more':False},badledger,
                              portfolio3,rd3,datetime(2026,10,19,15,5,tzinfo=timezone.utc),state3)
    except adapter.Stop as exc:assert str(exc)==reason,(str(exc),reason)
    else:raise AssertionError('R3_UNSAFE_CATCHUP_ACCEPTED_'+reason)
print('V3932_R3_CATCHUP_NEGATIVE_PATHS_FAIL_CLOSED_PASS',flush=True)
