#!/usr/bin/env python3
r"""ICAIF Phase43 v3: retrospective multi-batch CaMeL event research.

RESEARCH ONLY. No official submissions, no orders, no server changes.
Requires Phase43 v2 and Phase23 in same folder. Uses the user's existing
FINNHUB_API_KEY and SEC_CONTACT_EMAIL via Phase43 v2, and private CaMeL key
via Phase23. One provider request requires one explicit --ack-provider-credits.

Work from the competition starter-kit directory in activated venv.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, re, sys, tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

UTC=timezone.utc
OUT=Path('artifacts/phase43_camel_research_batch_v3')
BASE=Path('artifacts/phase43_historical_camel_replay')
CAP=8
MAX_BATCH=3
MAX_BYTES=17000
MAX_TOKENS=1700


def now(): return datetime.now(UTC).isoformat().replace('+00:00','Z')
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.safe_',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(obj,f,indent=2,ensure_ascii=False)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def helper():
    path=Path(__file__).resolve().parent/'icaif_phase43_historical_camel_replay_v2.py'
    if not path.is_file(): raise ValueError('Place Phase43 v2 alongside Phase43 batch v3 in Downloads')
    spec=importlib.util.spec_from_file_location('icaif_p43v2_internal',path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


def case_hash(e):
    obj={k:e.get(k) for k in ('event_id','symbol','published_utc','modeled_available_utc','evidence_text','sec_context_asof','source_quality','url')}
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def is_usable(e, include_news=False):
    if not isinstance(e,dict):return False
    required=('event_id','symbol','published_utc','modeled_available_utc','evidence_text','source_quality','url')
    if any(not isinstance(e.get(k),str) or not e[k].strip() for k in required):return False
    if len(e['evidence_text'])<130:return False
    if not e['url'].startswith('https://'):return False
    quality=e['source_quality']
    if quality.startswith('SEC_ORIGINAL_ACCESSION_'):
        return e.get('issuer_verified') is True and e['url'].startswith('https://www.sec.gov/Archives/edgar/data/')
    return include_news and quality=='FINNHUB_RETRIEVED_NOW_SUMMARY_MAY_HAVE_BEEN_EDITED'


def enrich(snapshot, h):
    out=[]
    for e in snapshot.get('events',[]):
        if not is_usable(e,include_news=True):continue
        x=dict(e)
        if not isinstance(x.get('sec_context_asof'),dict):
            x['sec_context_asof']=h.prior_packet(x,snapshot.get('sec',{}).get(x['symbol'],{}))
        x['evidence_sha256']=case_hash(x)
        out.append(x)
    return out


def dataset():
    p=OUT/'event_library.json'
    return load(p) if p.exists() else {'version':3,'cases':[], 'imports':[]}


def merge(rows, label):
    old=dataset()
    index={r['event_id']:r for r in old['cases']}
    collisions=[];added=0;changed=0
    for x in rows:
        eid=x['event_id']
        if eid in index:
            if case_hash(index[eid])!=case_hash(x):
                # Do not silently modify text after inference; keep prior version, flag conflict.
                collisions.append(eid);changed+=1
        else:
            index[eid]=x;added+=1
    old['cases']=sorted(index.values(),key=lambda x:(x['published_utc'],x['symbol'],x['event_id']))
    old['imports'].append({'at_utc':now(),'label':label,'new_cases':added,'changed_conflicts':changed})
    atomic(OUT/'event_library.json',old)
    print('P43V3_IMPORT',label,'NEW',added,'TOTAL',len(old['cases']),'CONFLICTS',changed)
    if collisions:print('P43V3_CONFLICT_IDS',','.join(collisions[:8]),'OLD_EVIDENCE_PRESERVED')


def import_existing(a):
    h=helper()
    p=Path(a.path) if a.path else BASE/'historical_camel_packet.json'
    if not p.exists():raise ValueError('Historical packet not found: '+str(p))
    obj=load(p)
    if 'cases' not in obj:raise ValueError('Expected historical_camel_packet.json containing cases')
    rows=[]
    for e in obj['cases']:
        if not is_usable(e,True):continue
        x=dict(e);x['evidence_sha256']=case_hash(x);rows.append(x)
    merge(rows,'PACKET:'+p.name)


def collect(a):
    if not a.ack_readonly_api_requests:raise ValueError('Use --ack-readonly-api-requests to collect data')
    h=helper()
    # Same historical-source guardrails and private-key handling as vetted Phase43 v2.
    h.hist_collect(SimpleNamespace(symbols=a.symbols,start=a.start,end=a.end,ack_readonly_api_requests=True))
    p=BASE/'historical_source_snapshot.json'
    snap=load(p)
    rows=enrich(snap,h)
    merge(rows,'COLLECT:'+a.start+'_'+a.end+'_'+a.symbols)
    # Snapshots stored for reproducibility even when collecting later date windows.
    digest=hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    atomic(OUT/'snapshots'/('sec_finnhub_'+a.start+'_'+digest+'.json'),snap)
    print('P43V3_SOURCE_ARCHIVED',digest,'MODEL_REQUESTS_ZERO')


def reviewed_cases(allow_news=False):
    return [x for x in dataset()['cases'] if is_usable(x,allow_news)]


def build_batches(cases, batch_size):
    """Predetermined round-robin across issuers; never use future returns for selection."""
    # One event per symbol/month first; then additional dates from same symbol.
    bins=defaultdict(list)
    for e in cases:
        bins[e['symbol']].append(e)
    # SEC originals first within each issuer, then chronological order.
    def order(e):
        return (0 if e['source_quality']=='SEC_ORIGINAL_ACCESSION_EXHIBIT' else
                1 if e['source_quality']=='SEC_ORIGINAL_ACCESSION_DOCUMENT' else 2,
                e['published_utc'], e['event_id'])
    for sym in bins:bins[sym].sort(key=order)
    picked=[];seen_dates=set()
    # Rotate issuers, reject multiple same-issuer same-day; keep different issuer dates.
    while any(bins.values()):
        advanced=False
        for sym in sorted(bins):
            while bins[sym]:
                e=bins[sym].pop(0)
                key=(sym,e['published_utc'][:10])
                if key in seen_dates:continue
                seen_dates.add(key)
                picked.append(e);advanced=True;break
        if not advanced:break
    result=[]
    for i in range(0,len(picked),batch_size):
        block=picked[i:i+batch_size]
        if not block:continue
        digest=hashlib.sha256('|'.join(case_hash(x) for x in block).encode()).hexdigest()[:16]
        result.append({'batch_id':digest,'cases':block})
    return result


def prep(a):
    if not 1<=a.batch_size<=MAX_BATCH:raise ValueError('Batch size must be 1-3')
    attempted={eid for attempt in ledger()['attempts'] for eid in attempt.get('case_ids',[])}
    pool=[case for case in reviewed_cases(a.include_news) if case['event_id'] not in attempted]
    batches=build_batches(pool,a.batch_size)
    manifest={'created_utc':now(), 'selection':'SEC evidence first within each issuer; round-robin, one event per issuer/day; no price outcomes',
              'including_finnhub_snippets_with_edit_risk':bool(a.include_news),
              'batches':batches[:CAP]}
    atomic(OUT/'batch_manifest.json',manifest)
    print('P43V3_PREPARED',len(manifest['batches']),'BATCHES',sum(len(b['cases']) for b in manifest['batches']),'CASES','PREVIOUSLY_ATTEMPTED_EXCLUDED',len(attempted),'MAX_REQUESTS',CAP)
    for i,b in enumerate(manifest['batches'],1):
        print('P43V3_BATCH',i,b['batch_id'],' '.join(e['symbol']+':'+e['published_utc'][:10]+':'+e['source_quality'] for e in b['cases']))
    if not manifest['batches']:print('P43V3_NO_ELIGIBLE_CASES_COLLECT_MORE_SOURCE_WINDOWS')
    print('P43V3_NO_PROVIDER_CALL_NO_TRADES')


def ledger():
    p=OUT/'llm_request_ledger.json'
    return load(p) if p.exists() else {'version':3,'limit':CAP,'attempts':[]}


def budget():
    ll=ledger(); attempts=ll['attempts']
    print('P43V3_REQUESTS_RESERVED',len(attempts),'OF',CAP)
    print('P43V3_SUCCESS',sum(v.get('status')=='ok' for v in attempts),'FAILED_OR_UNKNOWN',sum(v.get('status')!='ok' for v in attempts))
    print('P43V3_TOKEN_USAGE_REPORTED',sum(int((v.get('usage') or {}).get('prompt_tokens',0)) for v in attempts),
        'input,',sum(int((v.get('usage') or {}).get('completion_tokens',0)) for v in attempts),'output')
    print('P43V3_ACTUAL_BILLING_NOT_READ_VERIFY_PROVIDER_BALANCE')
    if (BASE/'camel_historical_ledger.json').exists():
        old=load(BASE/'camel_historical_ledger.json')
        print('P43V2_OLDER_CALLS_RESERVED',len(old.get('reserved_calls',[])))


def predictions():
    p=OUT/'retrospective_predictions.json'
    return load(p) if p.exists() else {'version':3,'retrospective_only':True,'events':[]}


def probe(a):
    if not a.ack_provider_credits:raise ValueError('Add --ack-provider-credits for ONE provider request')
    manifest=load(OUT/'batch_manifest.json')
    index=a.batch_index-1
    if not (0<=index<len(manifest['batches'])):raise ValueError('Batch index outside prepared manifest')
    block=manifest['batches'][index]
    old=ledger()
    if len(old['attempts'])>=CAP:raise ValueError('Maximum 8 calls used; stop and review results')
    attempted={eid for v in old['attempts'] for eid in v.get('case_ids',[])}
    if any(e['event_id'] in attempted for e in block['cases']):
        raise ValueError('One or more cases already attempted under an earlier batch; prepare again')
    if any(v['batch_id']==block['batch_id'] for v in old['attempts']):
        raise ValueError('That exact batch has already been attempted, no duplicate retries')
    h=helper()
    prompt=h.llm_prompt(block['cases'])
    provider=h.provider()
    body={'model':provider.CAMEL_MODEL,'messages':[{'role':'user','content':prompt}],
          'temperature':0,'max_tokens':MAX_TOKENS}
    size=len(json.dumps(body,ensure_ascii=False).encode('utf-8'))
    if size>MAX_BYTES:raise ValueError('Batch too large for bounded CaMeL call; use batch-size 1 or 2')
    fingerprint=hashlib.sha256(prompt.encode()).hexdigest()
    # Resolve local secrets before reservation. Never print the key.
    key=provider.load_key('camel')
    entry={'batch_id':block['batch_id'],'batch_index':a.batch_index,'prompt_sha256':fingerprint,
           'case_ids':[e['event_id'] for e in block['cases']],'reserved_at_utc':now(),
           'model':provider.CAMEL_MODEL,'prompt_bytes':size,'status':'reserved'}
    old['attempts'].append(entry)
    atomic(OUT/'llm_request_ledger.json',old)
    print('P43V3_ONE_PROVIDER_CALL_RESERVED',a.batch_index,'OF_MAX',CAP,'NO_AUTORETRY',flush=True)
    try:
        raw=provider.transport(provider.CAMEL_URL,key,body,timeout=45)
        result=h.parse_llm(raw,block['cases'])
    except Exception as ex:
        old['attempts'][-1]['status']='failed_or_invalid'
        old['attempts'][-1]['failure_type']=type(ex).__name__
        atomic(OUT/'llm_request_ledger.json',old)
        print('P43V3_PROVIDER_FAILURE_NO_RETRY',type(ex).__name__,file=sys.stderr)
        return 2
    inferred=now()
    usage=raw.get('usage') or {}
    # Persist classification and case evidence together for revision safety.
    stored=predictions();bykey={e['event_id']:e for e in stored['events']}
    indexed={e['event_id']:e for e in result}
    for e in block['cases']:
        bykey[e['event_id']]={'case':e,'classification':indexed[e['event_id']],
             'case_evidence_sha256':case_hash(e),'prompt_sha256':fingerprint,
             'inferred_at_utc':inferred,'retrospective_research_only':True}
    stored['events']=list(bykey.values())
    atomic(OUT/'retrospective_predictions.json',stored)
    old['attempts'][-1].update({'status':'ok','finished_at_utc':inferred,
                                'usage':{k:v for k,v in usage.items() if isinstance(v,(int,float,str))}})
    atomic(OUT/'llm_request_ledger.json',old)
    print('P43V3_BATCH_COMPLETE',a.batch_index,'EVENTS',len(result),'MODEL',provider.CAMEL_MODEL)
    for case in block['cases']:
        pred=indexed[case['event_id']]
        print('P43V3_CASE',case['symbol'],case['published_utc'][:10],pred['direction'],pred['strength'],pred['event_type'])
    print('P43V3_OUTPUT',OUT/'retrospective_predictions.json')
    print('P43V3_RESEARCH_ONLY_NO_LAYA_NO_TRADES_NO_OFFICIAL')
    return 0


def inspect():
    df=predictions()['events']
    print('P43V3_PREDICTIONS',len(df),'DIRECTION_COUNTS',dict(Counter(e['classification']['direction'] for e in df)))
    for e in df:
        case=e['case'];x=e['classification']
        print('P43V3_OBSERVATION',case['symbol'],case['published_utc'][:10],x['direction'],x['strength'],
              'NEW',x['material_new_fact'],'EXPECTATION_KNOWN',x['expectation_known'])
    print('P43V3_NO_NETWORK_NO_MODEL')


def score(a):
    import pandas as pd
    h=helper()
    events=predictions()['events']
    if not events:raise ValueError('No predictions yet')
    bars=pd.read_csv(a.bars,usecols=['symbol','timestamp','open','close'])
    bars['timestamp']=pd.to_datetime(bars['timestamp'],utc=True)
    bars['d']=bars['timestamp'].dt.tz_convert('America/New_York').dt.date
    days=sorted(bars['d'].unique());ix={d:j for j,d in enumerate(days)}
    # Market-return comparison for same execution and exit dates (equal-weight universe).
    stockgroups={sym:g.sort_values('timestamp') for sym,g in bars.groupby('symbol')}
    rows=[]
    for entry in events:
        ev=entry['case'];p=entry['classification']
        slot=h.next_slot(h.dt(ev['modeled_available_utc']),days)
        row={'event_id':ev['event_id'],'symbol':ev['symbol'],'original_publication_utc':ev['published_utc'],
            'hypothetical_evidence_available_utc':ev['modeled_available_utc'],
            'actual_camel_inference_utc':entry['inferred_at_utc'],
            'direction':p['direction'],'strength':p['strength'],'material_new_fact':p['material_new_fact'],
            'retrospective_model_training_contamination_possible':True,'fee_rate_one_side':0.001}
        if slot:
            d,k,deadline,exe=slot
            row.update({'decision_date':str(d),'decision_round':k,'deadline_utc':h.stamp(deadline),'execution_utc':h.stamp(exe)})
            sym=stockgroups.get(ev['symbol'])
            if sym is not None:
                sub=sym[sym['timestamp']>=pd.Timestamp(exe)]
                if len(sub):
                    fill=float(sub.iloc[0]['open'])
                    for horizon in (1,5):
                        n=ix[d]+horizon
                        if n>=len(days):continue
                        date_end=days[n]
                        closing=sym[sym['d']==date_end]
                        if closing.empty:continue
                        stockret=float(closing.iloc[-1]['close'])/fill-1
                        # Use first available bar at/after execution for all 30 equal-weight marks.
                        market=[]
                        for other,g in stockgroups.items():
                            ge=g[g['timestamp']>=pd.Timestamp(exe)]
                            gd=g[g['d']==date_end]
                            if len(ge) and len(gd):
                                market.append(float(gd.iloc[-1]['close'])/float(ge.iloc[0]['open'])-1)
                        benchmark=sum(market)/len(market) if market else float('nan')
                        row['stock_return_'+str(horizon)+'d']=stockret
                        row['equal30_return_'+str(horizon)+'d']=benchmark
                        row['relative_return_'+str(horizon)+'d']=stockret-benchmark
                        if p['direction']=='positive':
                            row['hypothetical_long_vs_cash_after_buy_and_sell_fee_'+str(horizon)+'d']=stockret-0.002
                            row['hypothetical_long_vs_equal30_after_fee_'+str(horizon)+'d']=stockret-benchmark-0.002
                        elif p['direction']=='negative':
                            row['hypothetical_return_avoided_vs_long_'+str(horizon)+'d']=-stockret
        rows.append(row)
    dest=OUT/'retrospective_event_outcomes.csv';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8') as f:
        cols=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    print('P43V3_SCORED',len(rows),'PATH',dest)
    print('P43V3_OUTCOMES_DIAGNOSTIC_ONLY_NOT_OFFICIAL_M1_M2_M3_M4')


def self_test():
    # Offline tests run on real schema patterns, no network, no private keys, no trades.
    h=helper()
    example={'event_id':'x','symbol':'GS','headline':'GS earnings announcement','published_utc':'2026-07-14T16:30:40Z',
        'modeled_available_utc':'2026-07-14T16:30:40Z','source_quality':'SEC_ORIGINAL_ACCESSION_EXHIBIT',
        'evidence_text':'Net revenues were $20 billion; diluted earnings per share were $21.00. '*4,
        'url':'https://www.sec.gov/Archives/edgar/data/886982/000088698226000294/ex99.htm',
        'issuer_verified':True,'sec_context_asof':{'prior_asof_companyfacts':[]}}
    assert is_usable(example)
    bad=dict(example,source_quality='FINNHUB_RETRIEVED_NOW_SUMMARY_MAY_HAVE_BEEN_EDITED')
    assert not is_usable(bad) and is_usable(bad,True)
    x=dict(example,symbol='JPM',event_id='y',published_utc='2026-08-01T16:30:40Z')
    batches=build_batches([example,x],2)
    assert len(batches)==1 and len(batches[0]['cases'])==2
    assert case_hash(example)!=case_hash(dict(example,evidence_text='changed'))
    assert case_hash(example)==case_hash(dict(example,evidence_sha256='irrelevant'))
    prompt=h.llm_prompt([example,x]);assert 'memorized facts' in prompt
    assert 'expected numerical return' in prompt
    print('P43V3_SELF_TEST_PASS')
    print('SEC_HISTORICAL_SOURCE_INTEGRITY_NO_HINDSIGHT_LLM_QUOTE_GATE_BATCH_DEDUP_PASS')
    print('ZERO_NETWORK_ZERO_CREDITS_ZERO_ORGANIZER_ZERO_TRADES')


def main():
    pa=argparse.ArgumentParser()
    g=pa.add_mutually_exclusive_group(required=True)
    for a in ('self-test','import-existing','collect','prepare','inspect','budget-status','probe-camel','score'):
        g.add_argument('--'+a,action='store_true')
    pa.add_argument('--path',default=None)
    pa.add_argument('--symbols',default='JPM GS')
    pa.add_argument('--start',default='2026-07-01')
    pa.add_argument('--end',default='2026-07-15')
    pa.add_argument('--batch-size',type=int,default=2)
    pa.add_argument('--batch-index',type=int,default=1)
    pa.add_argument('--include-news',action='store_true',help='Include historical Finnhub summaries whose edits cannot be audited')
    pa.add_argument('--ack-readonly-api-requests',action='store_true')
    pa.add_argument('--ack-provider-credits',action='store_true')
    pa.add_argument('--bars',default='artifacts/phase29_alpaca_refresh/alpaca_iex_15m_through_2026-10-09.csv')
    a=pa.parse_args()
    if a.self_test:self_test()
    elif a.import_existing:import_existing(a)
    elif a.collect:collect(a)
    elif a.prepare:prep(a)
    elif a.inspect:inspect()
    elif a.budget_status:budget()
    elif a.probe_camel:sys.exit(probe(a))
    elif a.score:score(a)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('P43V3_STOP',type(e).__name__,str(e)[:220],file=sys.stderr)
        sys.exit(2)
