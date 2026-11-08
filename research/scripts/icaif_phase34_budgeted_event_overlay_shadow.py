#!/usr/bin/env python3
"""ICAIF Phase34: budget-gated CaMeL-first, Laya-second financial event shadow overlay.

Offline: --self-test, --prepare, --budget-status. No trading or organizer calls.
Paid provider calls ONLY with --probe-event PATH --ack-provider-credits.
Uses existing Phase23 credentials, provider endpoints and parsers. Requires Phase23
beside this script in Downloads. No provider secrets are logged or saved here.

A paper overlay is NOT a return forecast, backtest, trade, or Official decision.
No historical 2026 decision gets retrospectively modified by LLM output.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT_OUT = Path('artifacts/phase34_budgeted_event_overlay_shadow')
SYMS = tuple('AAPL MSFT NVDA INTC CRM JPM BAC GS V PYPL LLY JNJ UNH PFE TMO AMZN TSLA WMT NKE KO CAT GE BA XOM CVX GOOGL META DIS T NEE'.split())
REACTION_TERMS = ('stock surged','stock plunged','shares surged','shares tumbled','stock gained','stock fell',
                  'stock jumps','stock drops','shares soar','shares fell','market reacts','stock rallied')
# A conservative UTF-8-byte *proxy* for Laya input tokens, NOT a tokenizer or provider bill.
# The user's stated quota is 10k Laya input tokens/month. Prior requests made by
# other scripts are NOT visible here, so this does NOT guarantee quota remaining.
RESEARCH_LAYA_MAX_CALLS = 1
RESEARCH_LAYA_MAX_REQUEST_BYTES = 1800
RESEARCH_CAMEL_MAX_CALLS = 1
MAX_HEADLINE_AGE_MINUTES = 120
MAX_PAPER_WEIGHT_TILT = .003   # +/- 0.3 percentage points of NAV, not executed
FEE = .001


def now_utc(): return datetime.now(timezone.utc)
def iso(dt): return dt.astimezone(timezone.utc).isoformat().replace('+00:00','Z')
def parse_utc(s):
    if not isinstance(s,str): raise ValueError('BAD_TIMESTAMP')
    try: d=datetime.fromisoformat(s.replace('Z','+00:00'))
    except ValueError: raise ValueError('BAD_TIMESTAMP') from None
    if d.tzinfo is None:raise ValueError('TIMESTAMP_NEEDS_TIMEZONE')
    return d.astimezone(timezone.utc)


def load_phase23():
    source=Path(__file__).resolve().parent/'icaif_phase23_financial_llm_event_lab.py'
    if not source.is_file():raise FileNotFoundError('Phase23 Python file must be in the same Downloads folder.')
    spec=importlib.util.spec_from_file_location('phase34_dep23',source)
    m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m
    assert spec.loader
    spec.loader.exec_module(m)
    return m


def phase33_core_from_local_prices(root):
    """Read the latest frozen 2026 IEX file and reproduce Phase33's 80% blend.

    This is a paper research core calculated from data through 2026-10-09;
    it is NOT a live-round portfolio or executable weight recommendation.
    """
    dependency=Path(__file__).resolve().parent/'icaif_phase30_2026_walkforward_agent_lab_v2.py'
    if not dependency.exists():raise FileNotFoundError('Phase30 v2 must be beside Phase34 in Downloads.')
    source=root/'artifacts'/'phase29_alpaca_refresh'/'alpaca_iex_15m_through_2026-10-09.csv'
    if not source.exists():raise FileNotFoundError('Phase29 refreshed Alpaca CSV is needed for blended-core paper weights.')
    spec=importlib.util.spec_from_file_location('phase34_dep30',dependency)
    p30=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p30
    assert spec.loader
    spec.loader.exec_module(p30)
    market=p30.read_market(source,min_last_day='2026-10-09')
    t=len(market.dates)-1
    industry=p30.desired('industry80_hold',market,t,{})
    blended=[.5*float(x)+.5*(.8/30.) for x in industry]
    if len(blended)!=30 or abs(sum(blended)-.8)>1e-9 or min(blended)<0 or max(blended)>.30:
        raise ValueError('INVALID_RESEARCH_CORE')
    return blended,str(market.dates[t].date())


def event_from_snapshot(root,p23):
    data,sha=p23.load_snapshot(root)
    events=[]
    for e in data.get('events',[]):
        if e.get('source')!='FINNHUB' or e.get('symbol') not in SYMS:continue
        try:
            published=parse_utc(e.get('event_time_utc'))
            observed=parse_utc(e.get('available_at_utc'))
        except ValueError:continue
        headline=str(e.get('headline','')).strip()[:300]
        if not headline or observed<published or observed>now_utc():continue
        ev={'event_id':str(e.get('event_id',''))[:100], 'symbol':e['symbol'],
            'headline':headline,'published_utc':iso(published),
            'first_observed_utc':iso(observed),'source':'FINNHUB',
            'source_snapshot_sha256':sha,'source_verification':'UNVERIFIED_RESEARCH'}
        if ev['event_id']:events.append(ev)
    if not events: raise RuntimeError('NO_VALID_SNAPSHOT_EVENTS')
    events.sort(key=lambda e:e['first_observed_utc'], reverse=True)
    return events[0]


def check_event(ev,at=None,allow_stale=False):
    if not isinstance(ev,dict):raise ValueError('BAD_EVENT_JSON')
    if ev.get('symbol') not in SYMS:raise ValueError('UNIVERSE_SYMBOL_MISMATCH')
    if not isinstance(ev.get('event_id'),str) or not (2<=len(ev['event_id'])<=100):raise ValueError('BAD_EVENT_ID')
    h=ev.get('headline')
    if not isinstance(h,str) or not (15<=len(h)<=300):raise ValueError('BAD_HEADLINE')
    for field,maxchars in [('evidence_excerpt',220),('prior_expectation',120)]:
        value=ev.get(field,'')
        if not isinstance(value,str) or len(value)>maxchars:raise ValueError('BAD_OPTIONAL_EVIDENCE_LENGTH')
    published=parse_utc(ev.get('published_utc'));observed=parse_utc(ev.get('first_observed_utc'))
    at=at or now_utc()
    if published>observed or observed>at+timedelta(seconds=5):raise ValueError('INVALID_EVENT_AVAILABILITY')
    age=(at-observed).total_seconds()/60
    stale=age>MAX_HEADLINE_AGE_MINUTES
    if stale and not allow_stale:raise ValueError('STALE_EVENT_ABORT_NO_PROVIDER_CALL')
    return {'age_minutes':round(max(age,0),1), 'is_stale':stale,
            'paper_feature_available_utc':iso(at)}


def event_sha(ev):
    fields=['event_id','symbol','headline','published_utc','first_observed_utc']
    return hashlib.sha256('\0'.join(str(ev[k]) for k in fields).encode()).hexdigest()


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.phase34-',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(obj,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def ledger_path(root):return root/ROOT_OUT/'phase34_credit_reservations.json'
def load_ledger(root):
    path=ledger_path(root)
    if path.exists():
        obj=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(obj,dict) or not isinstance(obj.get('reservations'),list):raise ValueError('BAD_LEDGER')
        return obj
    return {'note':'Local Phase34 only; prior provider use not captured. Charges unverified.', 'reservations':[]}


def reserve(root,provider,body,evsha):
    if provider not in ('camel','laya'):raise ValueError('BAD_PROVIDER')
    raw=json.dumps(body,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    if provider=='laya' and len(raw)>RESEARCH_LAYA_MAX_REQUEST_BYTES:raise ValueError('LAYA_REQUEST_BYTE_PROXY_BUDGET_EXCEEDED')
    ledger=load_ledger(root); entries=ledger['reservations']
    if any(r['event_sha']==evsha and r['provider']==provider for r in entries):
        raise ValueError('EVENT_PROVIDER_ALREADY_RESERVED_NO_RETRY')
    limit=RESEARCH_LAYA_MAX_CALLS if provider=='laya' else RESEARCH_CAMEL_MAX_CALLS
    if sum(r['provider']==provider for r in entries)>=limit:raise ValueError('RESEARCH_PROVIDER_CALL_CAP_REACHED')
    row={'provider':provider,'event_sha':evsha,'reserved_at_utc':iso(now_utc()),
         'request_utf8_bytes_proxy':len(raw),'actual_provider_input_tokens':'UNKNOWN',
         'actual_usd_cost':'UNKNOWN','charge_or_timeout_may_consume_credit':True}
    entries.append(row);atomic_json(ledger_path(root),ledger)
    return row


def pre_screen(ev,camel):
    headline=(ev['headline']+' '+ev.get('evidence_excerpt','')).lower()
    return (camel.get('event_type') in ('earnings','guidance','revenue','regulatory','legal','merger','product')
            and camel.get('direction') in ('positive','negative')
            and camel.get('materiality') in ('medium','high')
            and camel.get('is_company_specific') is True
            and not camel.get('needs_full_text')
            and not any(t in headline for t in REACTION_TERMS))


def paper_overlay(ev,camel,laya,at,core_weights=None):
    # Demonstrate how a timestamped consensus COULD be represented in a bounded
    # 80% cash-neutral stock tilt. This is NOT calibrated and MUST NOT be submitted.
    if not laya:
        vote='abstain'
    else:
        a=camel.get('direction');b=laya.get('direction')
        ok=(pre_screen(ev,camel) and a==b and a in ('positive','negative')
            and float(laya.get('company_relevance_model_probability_not_calibrated',0))>=.75)
        vote=a if ok else 'abstain'
    w=[float(x) for x in (core_weights if core_weights is not None else [.8/30]*30)]
    if len(w)!=30 or min(w)<0 or max(w)>.30 or abs(sum(w)-.8)>1e-9:raise ValueError('BAD_CORE_WEIGHTS')
    tilt=0.
    if vote in ('positive','negative'):
        tilt=MAX_PAPER_WEIGHT_TILT*(1 if vote=='positive' else -1)
        j=SYMS.index(ev['symbol']);w[j]+=tilt
        for k in range(30):
            if k!=j:w[k]-=tilt/29
    if not (all(0<=x<=.3 for x in w) and abs(sum(w)-.8)<1e-9):raise AssertionError('BAD_PAPER_WEIGHTS')
    gross=2*abs(tilt)
    return {'paper_vote':vote,'paper_weight_tilt':tilt,'paper_weights_30':dict(zip(SYMS,w)),
            'equity_exposure':sum(w),'one_way_equity_shift':abs(tilt),
            'hypothetical_roundtrip_notional_fraction_before_drift':gross,
            'hypothetical_fee_fraction':FEE*gross,
            'incremental_return_forecast':None,'expected_alpha_validated':False,
            'usable_for_historical_backtest':False,'feature_available_after_utc':iso(at),
            'authorized_for_official_submission':False}


def do_prepare(root):
    p23=load_phase23();ev=event_from_snapshot(root,p23)
    meta=check_event(ev,allow_stale=True)
    path=root/ROOT_OUT/'phase34_candidate_event.json'
    atomic_json(path,ev)
    print('CANDIDATE_SNAPSHOT_EVENT:',ev['symbol'])
    print('SOURCE_FIRST_OBSERVED_UTC:',ev['first_observed_utc'])
    print('AGE_MINUTES:',meta['age_minutes'],'STALE_FOR_ACTION:',meta['is_stale'])
    print('CANDIDATE_FILE:',path)
    print('NO_PROVIDER_REQUEST_NO_CREDENTIALS_NO_TRADES')


def do_budget(root):
    r=load_ledger(root)['reservations']
    print('PHASE34_LOCAL_RESEARCH_RESERVATIONS_CAMEL:',sum(x['provider']=='camel' for x in r),'OF',RESEARCH_CAMEL_MAX_CALLS)
    print('PHASE34_LOCAL_RESEARCH_RESERVATIONS_LAYA:',sum(x['provider']=='laya' for x in r),'OF',RESEARCH_LAYA_MAX_CALLS)
    print('REPORTED_LAYA_MONTHLY_INPUT_TOKENS: 10000 (user-provided, not verified with provider)')
    print('REPORTED_CAMEL_INITIAL_CREDIT_USD: 50 (user-provided, current balance unknown)')
    print('ACTUAL_PROVIDER_BALANCES_UNKNOWN_CHECK_DASHBOARDS_BEFORE_OFFICIAL')
    print('NO_PROVIDER_REQUEST_NO_TRADES')


def provider_event_text(ev):
    # Supports guidance/filing comparisons as well as news. The fields MUST
    # contain only source text/known expectations already available at observe time.
    combined=ev['headline']
    if ev.get('evidence_excerpt'):
        combined+=' | PUBLIC SOURCE EXCERPT: '+ev['evidence_excerpt']
    if ev.get('prior_expectation'):
        combined+=' | PRIOR PUBLISHED EXPECTATION: '+ev['prior_expectation']
    return combined


def do_probe(root,event_file,allow_stale):
    p23=load_phase23()
    event_path=Path(event_file).expanduser().resolve()
    ev=json.loads(event_path.read_text(encoding='utf-8'))
    validity=check_event(ev,allow_stale=allow_stale)
    digest=event_sha(ev)
    # IMPORTANT: check ledger BEFORE reading keys or sending a network request.
    old=load_ledger(root)['reservations']
    if any(r.get('event_sha')==digest for r in old):raise ValueError('EVENT_ALREADY_PROBED_NO_RETRY')
    camel_key=p23.load_key('camel')
    inputs=[{'event_id':ev['event_id'],'symbol':ev['symbol'],'headline':provider_event_text(ev)}]
    camel_body={'model':p23.CAMEL_MODEL,'messages':[{'role':'user','content':p23.prompt_events(inputs)}],
                'temperature':0,'max_tokens':450}
    reserve(root,'camel',camel_body,digest)
    print('CAMEL_SINGLE_REQUEST_STARTED_RESERVATION_PERSISTED',flush=True)
    response=p23.transport(p23.CAMEL_URL,camel_key,camel_body)
    camel=p23.parse_camel_response(response,inputs)[0]
    laya=None; laya_was_called=False
    if pre_screen(ev,camel):
        laya_body={'state':{'ticker':ev['symbol'],'headline':provider_event_text(ev)},'questions':p23.laya_questions()}
        # Verify Laya key locally before reserving credits.
        laya_key=p23.load_key('laya')
        reserve(root,'laya',laya_body,digest)
        print('LAYA_SINGLE_REQUEST_STARTED_AFTER_CAMEL_SCREEN',flush=True)
        laya_resp=p23.transport(p23.LAYA_URL,laya_key,laya_body)
        laya=p23.parse_laya_response(laya_resp,inputs[0])
        laya_was_called=True
    complete=now_utc()
    core,core_asof=phase33_core_from_local_prices(root)
    paper=paper_overlay(ev,camel,laya,complete,core_weights=core)
    report={'research_only':True,'event':ev,'event_sha256':digest,
            'source_availability_gate':validity,'camel':camel,'laya':laya,
            'laya_called_only_after_camel_screen':laya_was_called,
            'feature_available_after_utc':iso(complete),
            'paper_overlay':paper,'paper_core_specification':'Phase33 50% industry80 + 50% equal80',
            'paper_core_market_data_asof_et':core_asof,'no_historical_backfill':True,
            'realized_forward_return':None,'official_submission_authorized':False,
            'provider_billing_amounts_unknown':True,
            'limitations':['Single event cannot prove predictive alpha',
                           'Laya input tokens estimated by UTF-8 bytes, not provider tokenizer',
                           'Remaining provider balance unknown; earlier Phase23/32 requests not in this ledger',
                           'Paper tilt is intentionally NOT approved for real trade execution']}
    path=root/ROOT_OUT/f'phase34_shadow_{complete.strftime("%Y%m%dT%H%M%SZ")}_{digest[:10]}.json'
    atomic_json(path,report)
    print('CAMEL_SCREEN:',camel['event_type'],camel['direction'],camel['materiality'])
    print('LAYA_CALLED:',laya_was_called,'PAPER_VOTE:',paper['paper_vote'])
    print('REPORT:',path)
    print('FORWARD_ONLY_UNCALIBRATED_NO_TRADES_NO_ORGANIZER_SUBMISSION')


def self_test():
    stamp=now_utc();ev={'event_id':'test123','symbol':'NVDA','headline':'NVIDIA raises revenue guidance above estimates',
                          'published_utc':iso(stamp-timedelta(minutes=20)),
                          'first_observed_utc':iso(stamp-timedelta(minutes=10))}
    assert check_event(ev,at=stamp)['is_stale'] is False
    old=dict(ev,first_observed_utc=iso(stamp-timedelta(days=3)))
    try:check_event(old,at=stamp);raise AssertionError('STALE_EVENT_ACCEPTED')
    except ValueError as e:assert str(e)=='INVALID_EVENT_AVAILABILITY' # published later than observed
    old=dict(ev,published_utc=iso(stamp-timedelta(days=4)),first_observed_utc=iso(stamp-timedelta(days=3)))
    try:check_event(old,at=stamp);raise AssertionError('STALE_EVENT_ACCEPTED')
    except ValueError as e:assert str(e)=='STALE_EVENT_ABORT_NO_PROVIDER_CALL'
    camel={'event_type':'guidance','direction':'positive','materiality':'high',
           'is_company_specific':True,'needs_full_text':False}
    laya={'direction':'positive','company_relevance_model_probability_not_calibrated':.90}
    yes=paper_overlay(ev,camel,laya,stamp)
    assert yes['paper_vote']=='positive' and abs(yes['equity_exposure']-.8)<1e-10
    assert yes['paper_weight_tilt']==.003 and len(yes['paper_weights_30'])==30
    assert paper_overlay(ev,camel,None,stamp)['paper_vote']=='abstain'
    assert paper_overlay(ev,camel,dict(laya,direction='negative'),stamp)['paper_vote']=='abstain'
    assert pre_screen(dict(ev,headline='NVIDIA stock surged after earnings'),camel) is False
    assert 'PRIOR PUBLISHED EXPECTATION' in provider_event_text(dict(ev, prior_expectation='Earlier revenue guidance: $100m'))
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)
        atomic_json(root/'one.json',{'x':1})
        assert json.loads((root/'one.json').read_text())['x']==1
        obj={'state':{'ticker':'NVDA','headline':ev['headline']},'questions':{'a':'test'}}
        reserve(root,'laya',obj,'a')
        try:reserve(root,'laya',obj,'a');raise AssertionError('DUPLICATE_ALLOWED')
        except ValueError as e:assert 'ALREADY_RESERVED' in str(e)
        try:reserve(root,'laya',obj,'b');raise AssertionError('BUDGET_ALLOWED')
        except ValueError as e:assert 'CALL_CAP' in str(e)
    print('PHASE34_SELF_TEST_PASS')
    print('TIMESTAMP_FRESHNESS_SAME_EVENT_DEDUP_PROVIDER_RESERVATION_PASS')
    print('CAMEL_FIRST_LAYA_SECOND_BUDGET_POLICY_AND_80_PCT_SHADOW_TILT_PASS')
    print('NO_NETWORK_NO_CREDENTIALS_NO_ORGANIZER_SUBMISSION')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    commands=ap.add_mutually_exclusive_group(required=True)
    commands.add_argument('--self-test',action='store_true')
    commands.add_argument('--prepare',action='store_true')
    commands.add_argument('--budget-status',action='store_true')
    commands.add_argument('--probe-event',metavar='JSON_FILE')
    ap.add_argument('--ack-provider-credits',action='store_true',help='Required to incur one CaMeL and potentially one Laya call.')
    ap.add_argument('--allow-stale-research',action='store_true',help='For explicit historical negative control only. Not actionable.')
    a=ap.parse_args()
    try:
        if a.self_test:self_test();return
        root=Path.cwd()
        if a.prepare:do_prepare(root)
        elif a.budget_status:do_budget(root)
        else:
            if not a.ack_provider_credits:raise ValueError('PAID_CALL_ACKNOWLEDGEMENT_REQUIRED')
            do_probe(root,a.probe_event,a.allow_stale_research)
    except (FileNotFoundError,RuntimeError,ValueError,KeyError,TypeError) as e:
        # Never display server/provider exception details which may contain secrets.
        msg=str(e)
        if msg in ('BAD_TIMESTAMP','TIMESTAMP_NEEDS_TIMEZONE','BAD_EVENT_JSON','UNIVERSE_SYMBOL_MISMATCH',
                   'BAD_EVENT_ID','BAD_HEADLINE','BAD_OPTIONAL_EVIDENCE_LENGTH','INVALID_EVENT_AVAILABILITY','STALE_EVENT_ABORT_NO_PROVIDER_CALL',
                   'BAD_LEDGER','BAD_PROVIDER','LAYA_REQUEST_BYTE_PROXY_BUDGET_EXCEEDED',
                   'EVENT_PROVIDER_ALREADY_RESERVED_NO_RETRY','RESEARCH_PROVIDER_CALL_CAP_REACHED',
                   'EVENT_ALREADY_PROBED_NO_RETRY','PAID_CALL_ACKNOWLEDGEMENT_REQUIRED','NO_VALID_SNAPSHOT_EVENTS'):
            print('PHASE34_STOP:',msg,file=sys.stderr)
        else:print('PHASE34_STOP:',type(e).__name__,'— see local dependency/file readiness',file=sys.stderr)
        raise SystemExit(2)

if __name__=='__main__':main()
