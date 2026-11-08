#!/usr/bin/env python3
"""ICAIF V39 hardened single-writer candidate.

Dry-run is organizer GET-only with an Alpaca IEX GET refresh. Real submission is
HARD DISABLED unless an owner-only arm marker is installed by separate user
approval. Neither this file nor the staging package creates that marker.
"""
from __future__ import annotations

import base64
import copy
import hashlib
try:
    import fcntl
except ImportError:  # Windows offline tests only; actual writer requires Linux.
    fcntl=None
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import icaif_v37_production_candidate as bridge37

APP=bridge37.APP
PRIVATE=APP/'.icaif'/'official_v39'
ARM_MARKER=PRIVATE/'OFFICIAL_SUBMISSIONS_ARMED'
ARM_BYTES=b'ICAIF_V39_OFFICIAL_USER_AUTHORIZED\n'
UTC=timezone.utc


class Halt(Exception):pass


def must(condition, code):
    if not condition:raise Halt(code)


def check_live_arming():
    """Never create this marker from the application, installer or tests."""
    must(ARM_MARKER.is_file() and not ARM_MARKER.is_symlink(), 'NOT_ARMED')
    meta=ARM_MARKER.stat()
    must(meta.st_mode & 0o077 == 0 and meta.st_uid==os.geteuid(), 'ARM_FILE_NOT_PRIVATE_OWNER')
    must(ARM_MARKER.read_bytes()==ARM_BYTES,'ARM_MARKER_NOT_EXACT')


def setup_private_dir():
    must(PRIVATE.is_dir() and not PRIVATE.is_symlink(),'PRIVATE_DIRECTORY_NOT_PREPARED')
    meta=PRIVATE.stat()
    must(meta.st_uid==os.geteuid() and meta.st_mode & 0o077 == 0, 'PRIVATE_DIRECTORY_PERMISSIONS')


class Exclusive:
    def __init__(self,path):self.path=Path(path);self.file=None
    def __enter__(self):
        must(fcntl is not None,'EXCLUSIVE_WRITER_REQUIRES_LINUX')
        must(not self.path.is_symlink(),'LOCK_SYMLINK')
        self.file=self.path.open('a+b')
        os.chmod(self.path,0o600)
        try:fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close();self.file=None
            raise Halt('ANOTHER_OFFICIAL_WRITER_ACTIVE')
        return self
    def __exit__(self,*exc):
        if self.file:
            fcntl.flock(self.file.fileno(),fcntl.LOCK_UN);self.file.close()


def initial_only(journal,history,ledger,portfolio,engine,machine):
    if journal is not None:return journal
    must(history['decisions']==[] and ledger.get('rounds')==[],'NO_JOURNAL_BUT_HISTORY_EXISTS')
    import icaif_v29_bridge_unarmed as original
    original.cash_only_portfolio(portfolio,engine.SYMBOLS)
    return machine.blank_journal(engine.sample_state())


def native_operations_fenced(session,journal):
    operations=session.state.get('operations')
    must(isinstance(operations,dict),'NATIVE_CHECKPOINT_INVALID')
    for key in operations:
        if key.startswith('decision:official:'):
            rid=key[len('decision:official:'):]
            must(rid in journal['intents'],'UNJOURNALED_NATIVE_OPERATION')
    return operations


def durable_intent(journal, round_id, kind, target, raw, *, machine, engine):
    """One atomic journal holds intent AND original upload bytes before native uploader.

    The raw bytes contain private team credentials. The journal stays mode 0600 in
    an owner-only directory, and must NEVER be logged, exported or committed.
    """
    must(type(raw) is bytes and 0 < len(raw) <= 120000, 'BAD_IMMUTABLE_RAW')
    changed = machine.record_intent(journal, round_id, kind, target, engine.SYMBOLS, engine)
    record = changed['intents'][round_id]
    record['raw_b64'] = base64.b64encode(raw).decode('ascii')
    record['raw_sha256'] = hashlib.sha256(raw).hexdigest()
    return changed


def exact_recorded_raw(journal, rid, machine, engine):
    record=journal['intents'][rid]
    must(isinstance(record.get('raw_b64'),str) and
         isinstance(record.get('raw_sha256'),str), 'PENDING_INTENT_BYTES_NOT_DURABLE')
    try:
        raw=base64.b64decode(record['raw_b64'],validate=True)
        from kit.original_client import strict_json
        payload=strict_json(raw)
    except Exception:
        raise Halt('PENDING_INTENT_PAYLOAD_CORRUPT') from None
    must(hashlib.sha256(raw).hexdigest()==record['raw_sha256'], 'INTENT_RAW_SHA_MISMATCH')
    must(isinstance(payload,dict) and payload.get('round_id')==rid and
         payload.get('phase')=='official' and
         payload.get('submission_type')=='decision','PENDING_PHASE_WRONG')
    weights=payload.get('weights')
    must(isinstance(weights,dict) and set(weights)==set(engine.SYMBOLS),
         'INTENT_SYMBOL_SET_MISMATCH')
    # Native strict_json parses fractional JSON numbers as Decimal; the frozen
    # state-machine's canonical hash intentionally accepts Python float only.
    must(machine.target_hash({key:float(value) if type(value) is Decimal else value
                              for key,value in weights.items()}, engine.SYMBOLS)==
         record['target_sha256'],'INTENT_TARGET_MISMATCH')
    return raw,payload


def recover_existing(session,journal,current_round,machine,engine,history=None,ledger=None):
    """Only reuse an exact durable intent, never recompute target on restart.

    A missing native operation means no POST could have occurred through this
    process, because OriginalSession durably checkpoints BEFORE its first POST.
    Still require fresh empty organizer state and the original eligible window.
    If any evidence is ambiguous, halt instead of attempting an upload.
    """
    operations=native_operations_fenced(session,journal)
    for rid,record in journal['intents'].items():
        if rid in journal['execution_confirmations']:continue
        if record.get('status')=='ABANDONED_NO_POST':continue
        op=operations.get('decision:official:'+rid)
        if op and op.get('state')=='finished':continue
        raw,payload=exact_recorded_raw(journal,rid,machine,engine)
        must(payload.get('team_id')==session._team()['team_id'] and
             payload.get('team_token')==session._team()['team_token'],
             'PENDING_TEAM_CREDENTIALS_MISMATCH')
        if op is not None:
            must(op.get('sha256')==hashlib.sha256(raw).hexdigest() and
                 op.get('file_name')=='decision.json', 'NATIVE_RAW_SHA_MISMATCH')
            if isinstance(op.get('raw_b64'),str):
                must(op['raw_b64']==record['raw_b64'],'NATIVE_RAW_CONTENT_MISMATCH')
            # OriginalSession holds its own immutable operation checkpoint.
            result=session._decision(raw,payload)
            print('V39_RESUME_ORIGINAL_NATIVE_OPERATION',rid,flush=True)
            return result
        # The journal exists but first-party uploader had not checkpointed yet.
        # We can retry only inside the SAME round, with EXACTLY the saved bytes.
        must(isinstance(current_round,dict) and current_round.get('id')==rid and
             current_round.get('phase')=='official' and
             current_round.get('status')=='OPEN','ORPHAN_INTENT_NOT_IN_CURRENT_OPEN_ROUND')
        deadline=datetime.fromisoformat(current_round['deadline'].replace('Z','+00:00'))
        must(deadline.tzinfo is not None and
             deadline.astimezone(UTC)-datetime.now(UTC) > timedelta(minutes=3),
             'ORPHAN_INTENT_TOO_LATE_TO_RECOVER')
        must(isinstance(history,dict) and isinstance(history.get('decisions'),list) and
             all(x.get('round_id')!=rid for x in history['decisions']),
             'ORPHAN_INTENT_HISTORY_NOT_EMPTY')
        must(isinstance(ledger,dict) and isinstance(ledger.get('rounds'),list) and
             all(x.get('round_id')!=rid for x in ledger['rounds']),
             'ORPHAN_INTENT_LEDGER_NOT_EMPTY')
        own=session.round(rid)
        must(isinstance(own,dict) and own.get('decisions')==[] and
             own.get('execution') is None,'ORPHAN_INTENT_ROUND_NOT_EMPTY')
        # Native _decision additionally performs live schedule and slot checks.
        result=session._decision(raw,payload)
        print('V39_RECOVER_PRE_CHECKPOINT_SAME_BYTES',rid,flush=True)
        return result
    return None


def abandon_orphaned_before_post(session,journal,schedule,history,ledger,machine,engine,save):
    """After missed round finishes, prove NO native checkpoint/attempt/trade.

    Only this narrow case can release a never-posted intent for R2 catch-up.
    A native operation in ANY state is never abandoned here.
    """
    operations=native_operations_fenced(session,journal)
    changed=copy.deepcopy(journal)
    dirty=False
    for rid,record in changed['intents'].items():
        if rid in changed['execution_confirmations'] or record.get('status')=='ABANDONED_NO_POST':
            continue
        if ('decision:official:'+rid) in operations:continue
        matching=[x for x in schedule.get('rounds',[]) if x.get('id')==rid]
        must(len(matching)==1,'ORPHAN_SCHEDULE_MISSING')
        if matching[0].get('status')!='COMPLETED':continue
        must(not any(x.get('round_id')==rid for x in history['decisions']),
             'ORPHAN_COMPLETED_HISTORY_CHANGED')
        must(not any(x.get('round_id')==rid for x in ledger['rounds']),
             'ORPHAN_COMPLETED_LEDGER_CHANGED')
        own=session.round(rid)
        must(own.get('decisions')==[] and own.get('execution') is None,
             'ORPHAN_COMPLETED_OWN_ROUND_CHANGED')
        exact_recorded_raw(changed,rid,machine,engine)
        record['status']='ABANDONED_NO_POST'
        dirty=True
        print('V39_CONFIRMED_ABANDONED_NEVER_POSTED',rid,flush=True)
    if dirty:save(changed)
    return changed

def persist(journal,engine,machine,path):
    machine.check_journal(journal,engine)
    machine.atomic_private_json(path,journal)


def same_executed_holdings(before, after, original):
    """Compare actual held shares, not drifting intraday mark-to-market weights."""
    must(before.get('phase')=='official' and after.get('phase')=='official',
         'OFFICIAL_PORTFOLIO_PHASE_CHANGED')
    must(before.get('last_completed_round')==after.get('last_completed_round'),
         'COMPLETED_ROUND_CHANGED_DURING_REFRESH')
    first,second=before.get('positions'),after.get('positions')
    must(isinstance(first,list) and isinstance(second,list), 'PORTFOLIO_POSITIONS_SCHEMA')
    def lots(positions):
        output={}
        for p in positions:
            must(isinstance(p,dict) and isinstance(p.get('symbol'),str),
                 'BAD_POSITION_SYMBOL')
            sym=p['symbol'];must(sym not in output,'DUPLICATE_POSITION')
            must('shares' in p,'MISSING_POSITION_SHARES')
            output[sym]=original.as_decimal(p['shares'])
        return output
    must(lots(first)==lots(second),'HOLDING_SHARES_CHANGED_DURING_REFRESH')
    must(abs(original.as_decimal(before.get('cash')) -
             original.as_decimal(after.get('cash')))<Decimal('0.02'),
         'CASH_CHANGED_DURING_REFRESH')
    return True


def cycle(session,journal,*, live,engine,machine,original,prior,refresh,save=None):
    # All GETs originate from authenticated session; complete pagination checked.
    schedule,portfolio,history,ledger=bridge37.collect(session,machine)
    journal=initial_only(journal,history,ledger,portfolio,engine,machine)
    native_operations_fenced(session,journal)
    journal=bridge37.reconcile(session,journal,schedule,portfolio,history,ledger,machine,engine)
    if live:
        save(journal)
        journal=abandon_orphaned_before_post(session,journal,schedule,history,ledger,machine,engine,save)
    pending=bridge37.inspect_pending(journal,history)
    if pending:
        if live:
            # Recovery is permitted ONLY through native immutable checkpoint.
            recover_existing(session,journal,schedule.get('current_round'),machine,engine,history,ledger)
        print('V37_HOLD_PENDING_EXECUTION',len(pending),flush=True)
        return journal,'HOLD'
    row=bridge37.candidate_round(schedule,engine)
    if row is not None and row.get('status')!='OPEN':
        print('V37_HOLD_CURRENT_ROUND_NOT_MARKED_OPEN',flush=True)
        return journal,'HOLD'
    if row is None:
        print('V37_HOLD_NO_ACTIVE_OFFICIAL_ROUND',flush=True)
        return journal,'HOLD'
    own=session.round(row['id'])
    # The 3-minute deadline buffer is checked once before refresh and again
    # afterwards, as the refresh might cross the deadline.
    rd,now=original.authenticated_round(schedule,own,engine.SYMBOLS,local_time=datetime.now(UTC))
    if not bridge37.eligible_for_market_refresh(rd,journal['state']):
        print('V37_HOLD_UNSCHEDULED_REVIEW',flush=True)
        return journal,'HOLD'
    snapshot=None
    try:
        refresh.run('refresh',now=datetime.now(UTC))
        snapshot=bridge37.load_snapshot(prior)
    except (ValueError,RuntimeError,FileNotFoundError,OSError,TimeoutError,KeyError) as error:
        print('V37_MARKET_REFRESH_UNAVAILABLE',type(error).__name__,flush=True)
    schedule2,portfolio2,history2,ledger2=bridge37.collect(session,machine)
    row2=bridge37.candidate_round(schedule2,engine)
    must(row2 is not None and row2['id']==row['id'],'ROUND_CHANGED_DURING_REFRESH')
    own2=session.round(row['id'])
    original.authenticated_round(schedule2,own2,engine.SYMBOLS,local_time=datetime.now(UTC))
    # No new/unknown submission appears during refresh or before upload.
    must(history2['decisions']==history['decisions'],'HISTORY_CHANGED_DURING_REFRESH')
    same_executed_holdings(portfolio,portfolio2,original)
    must(ledger2.get('rounds')==ledger.get('rounds'),'LEDGER_CHANGED_DURING_REFRESH')
    row2,result=bridge37.choose(schedule2,portfolio2,history2,ledger2,own2,
                         snapshot,journal,original,engine,machine,datetime.now(UTC),session=session)
    if result['action']=='HOLD':
        new=copy.deepcopy(journal)
        new['state']=result['next_state']
        if live:save(new)
        print('V37_HOLD_NO_UPLOAD',result['note'],flush=True)
        return new,'HOLD'
    target=result['target']
    payload=bridge37.decision_payload(session,target,row2,engine)
    # Same native organizer validation method used by actual first-party uploader.
    from kit.contracts import validate_payload
    import tempfile
    with tempfile.TemporaryDirectory(prefix='icaif_v37_schedule_') as td:
        path=Path(td)/'schedule.json'
        path.write_text(json.dumps(schedule2),encoding='utf-8')
        validate_payload(payload,'decision.json',schedule_path=path)
    if not live:
        print('V37_DRYRUN_VALIDATED_TARGET',row2['id'],flush=True)
        print('V37_DRYRUN_EQUITY',f'{sum(target.values()):.12f}',flush=True)
        return journal,'TARGET_UNARMED'
    kind='initial' if journal['state']['mode']=='PRE_ENTRY' else (
        'recovery' if journal['state']['mode']=='RECOVERY_PENDING' else 'primary_rebalance')
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    changed=durable_intent(journal,row2['id'],kind,target,raw,machine=machine,engine=engine)
    changed['state']=result['next_state']
    # CRITICAL: one atomic durability barrier for BOTH intent and exact payload,
    # before ANY call to the native uploader (which checkpoints before POST).
    save(changed)
    # _decision rechecks schedule and owns exactly-once immutable POST checkpoint.
    receipt=session._decision(raw,payload)
    print('V37_NATIVE_RESULT_RECORDED_PENDING_EXECUTION',row2['id'],flush=True)
    # No execution state advance here, even if receipt says VALID.
    return changed,'SUBMITTED_RECEIPT_NOT_EXECUTION'


def run(*,live=False):
    import httpx
    sys.path[:0]=[str(APP),str(bridge37.V23),str(bridge37.V28)]
    import icaif_v33_state_machine as machine
    import icaif_v29_bridge_unarmed as original
    import icaif_v30_remote_get_only as prior
    import icaif_official_sleeve_engine_v23 as engine
    import icaif_v28_clean_alpaca_bootstrap as refresh
    from kit.config import load_environment
    from kit.original_client import OriginalSession,load_profile
    if live:
        check_live_arming();setup_private_dir()
    load_environment(APP/'.env')
    env_profile=Path(os.environ.get('ICAIF_PROFILE',''))
    if not env_profile.is_absolute():env_profile=APP/env_profile
    profile=load_profile(env_profile)
    token=os.environ.get('CODABENCH_TOKEN','')
    must(bool(token),'NO_PLATFORM_TOKEN')
    with tempfile.TemporaryDirectory(prefix='icaif_v37_') as temp:
        if live:
            checkpoint=PRIVATE/'native_checkpoint.json'
            journal_path=PRIVATE/'execution_journal.json'
            lock_file=PRIVATE/'single_writer.lock'
        else:
            checkpoint=Path(temp)/'checkpoint.json'
            journal_path=Path(temp)/'journal.json'
            lock_file=Path(temp)/'single_writer.lock'
        with Exclusive(lock_file):
            gate=prior.GetOnlyTransport(httpx.HTTPTransport()) if not live else None
            with OriginalSession(profile=profile,token=token,checkpoint=checkpoint,
                   credentials=APP/'.icaif'/'credentials.json',transport=gate,
                   timeout=18,result_timeout=35,poll_interval=3) as session:
                j=machine.read_journal(journal_path,engine) if journal_path.exists() else None
                saved=(lambda x:persist(x,engine,machine,journal_path))
                state,kind=cycle(session,j,live=live,engine=engine,machine=machine,
                                 original=original,prior=prior,refresh=refresh,save=saved)
                if not live:
                    must(bool(gate.attempts) and set(gate.attempts)=={'GET'},'NON_GET_REQUEST_ATTEMPTED')
                    print('V37_OFFICIAL_TRANSPORT_GET_ONLY',flush=True)
    print('V37_CYCLE_COMPLETE',kind,'LIVE' if live else 'UNARMED',flush=True)


def main():
    import argparse
    ap=argparse.ArgumentParser()
    modes=ap.add_mutually_exclusive_group(required=True)
    modes.add_argument('--get-only',action='store_true')
    modes.add_argument('--official-submit-armed',action='store_true')
    a=ap.parse_args()
    try:
        run(live=a.official_submit_armed)
    except Exception as e:
        print('V37_FAIL_CLOSED',type(e).__name__,str(e)[:95],flush=True)
        raise SystemExit(2)

if __name__=='__main__':main()
