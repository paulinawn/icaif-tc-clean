#!/usr/bin/env python3
"""ICAIF 2026 Phase35: low-cost, first-observation-anchored Finnhub event intake.

Research only; no LLM inference, no portfolio decisions, no trading.
Calls to Finnhub occur ONLY with --collect (max one request per symbol).
--prepare outputs one verified-timestamp, *unverified-relevance* candidate JSON
compatible with Phase34 --probe-event. Never implies historical backtest access.

Place this file in Downloads; run from the official starter-kit directory.
Keys are loaded from ~/.icaif/research_sources.env or environment.
Secrets are neither printed nor persisted in output. No retry on failures.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

UTC = timezone.utc
STOCKS = tuple('AAPL MSFT NVDA INTC CRM JPM BAC GS V PYPL LLY JNJ UNH PFE TMO AMZN TSLA WMT NKE KO CAT GE BA XOM CVX GOOGL META DIS T NEE'.split())
OUT = Path('artifacts/phase35_forward_event_intake')
FRESH_MINUTES = 120
RELEVANT = ('earnings','revenue','guidance','outlook','forecast','profit','loss','quarter','fda',
            'approval','recall','investigation','settlement','acquisition','merger','antitrust',
            'dividend','buyback','contract','raises','cuts','lowers','misses','beats','files',
            'sec filing','8-k','10-q','10-k','tariff','guides','restrictions','lawsuit')
REACTION = ('stock surged','stock plunged','shares surged','shares tumbled','stock gained',
            'stock fell','stock jumps','stock drops','shares soar','shares fell',
            'market reacts','stock rallied','shares rise','shares slid')
# First-pass issuer check, not an issuer verification. False negatives acceptable:
# ambiguous news should not incur scarce model credits.
ALIASES = {
 'AAPL':('apple',), 'MSFT':('microsoft',), 'NVDA':('nvidia','nvda'), 'INTC':('intel',),
 'CRM':('salesforce',), 'JPM':('jpmorgan','jp morgan'), 'BAC':('bank of america',),
 'GS':('goldman sachs',), 'V':('visa ',), 'PYPL':('paypal',), 'LLY':('eli lilly','lilly'),
 'JNJ':('johnson & johnson','johnson and johnson','jnj'), 'UNH':('unitedhealth',),
 'PFE':('pfizer',), 'TMO':('thermo fisher',), 'AMZN':('amazon',), 'TSLA':('tesla',),
 'WMT':('walmart',), 'NKE':('nike',), 'KO':('coca-cola','coca cola'),
 'CAT':('caterpillar',), 'GE':('general electric','ge aerospace'),
 'BA':('boeing',), 'XOM':('exxon','mobil'), 'CVX':('chevron',),
 'GOOGL':('alphabet','google'), 'META':('meta platforms','facebook','meta '),
 'DIS':('disney',), 'T':('at&t',), 'NEE':('nextera energy','next era energy'),
}


def now(): return datetime.now(UTC)
def iso(d):return d.astimezone(UTC).isoformat().replace('+00:00','Z')
def parse_time(s):
    try:
        d=datetime.fromisoformat(str(s).replace('Z','+00:00'))
        if d.tzinfo is None:raise ValueError('naive')
        return d.astimezone(UTC)
    except (ValueError,TypeError):raise ValueError('INVALID_TIMESTAMP') from None


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.phase35_')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as h:
            json.dump(obj,h,ensure_ascii=False,indent=2);h.flush();os.fsync(h.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def config_token():
    p=Path.home()/'.icaif'/'research_sources.env'
    v=os.environ.get('FINNHUB_API_KEY','').strip()
    if v:return v
    if p.exists():
        for line in p.read_text(encoding='utf-8-sig').splitlines():
            k,sep,val=line.strip().partition('=')
            if sep and k.strip()=='FINNHUB_API_KEY':
                v=val.strip().strip('"\'')
    if not v:raise ValueError('FINNHUB_KEY_NOT_CONFIGURED_PRIVATELY')
    return v


def get_news(sym,key,days):
    query=urlencode({'symbol':sym,'from':(now().date()-timedelta(days=days)).isoformat(),
                     'to':now().date().isoformat()})
    req=Request('https://finnhub.io/api/v1/company-news?'+query,
                headers={'X-Finnhub-Token':key,'Accept':'application/json',
                         'User-Agent':'ICAIF2026-ResearchReadOnly/1.0'})
    try:
        with urlopen(req,timeout=18) as response:
            if response.status!=200:raise ValueError('FINNHUB_NON_200')
            payload=response.read(1_500_001)
    except HTTPError as ex:raise ValueError('FINNHUB_HTTP_'+str(ex.code)) from None
    except (URLError,OSError,TimeoutError):raise ValueError('FINNHUB_NETWORK_ERROR') from None
    if len(payload)>1_500_000:raise ValueError('FINNHUB_RESPONSE_TOO_LARGE')
    try:r=json.loads(payload)
    except ValueError:raise ValueError('FINNHUB_BAD_JSON') from None
    if not isinstance(r,list):raise ValueError('FINNHUB_NOT_NEWS_LIST')
    return r,now()


def to_records(sym,payload,observed):
    out=[];seen=set()
    for raw in payload[:200]:
        if not isinstance(raw,dict):continue
        headline=str(raw.get('headline') or '').strip()[:300]
        if len(headline)<15:continue
        try:
            published=datetime.fromtimestamp(float(raw['datetime']),UTC)
        except (ValueError,KeyError,TypeError,OverflowError,OSError):continue
        if published>observed+timedelta(minutes=2):continue
        identity=str(raw.get('id') or raw.get('url') or headline)
        digest=hashlib.sha256((sym+':'+identity).encode()).hexdigest()[:32]
        if digest in seen:continue
        seen.add(digest)
        out.append({'event_id':digest,'symbol':sym,'headline':headline,
                    'published_utc':iso(published),'first_observed_utc':iso(observed),
                    'source':'FINNHUB','publisher':str(raw.get('source') or '')[:100],
                    'source_url':str(raw.get('url') or '')[:500],
                    'company_relevance_verified':False,'historical_backtest_eligible':False})
    return out


def read_index(root):
    path=root/OUT/'event_index.json'
    if not path.exists():return {'version':1,'events':{},'collections':[]}
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or not isinstance(data.get('events'),dict) or not isinstance(data.get('collections'),list):
        raise ValueError('INDEX_INVALID')
    return data


def identity(row):return row['symbol']+':'+row['event_id']


def integrate(index,rows,received,stats):
    # Immutable first_observed: refreshing an old headline must not reset its event clock.
    newer=0
    for ev in rows:
        k=identity(ev)
        if k not in index['events']:
            index['events'][k]=ev;newer+=1
        else:
            existing=index['events'][k]
            if parse_time(ev['first_observed_utc'])<parse_time(existing['first_observed_utc']):
                raise ValueError('FIRST_OBSERVATION_WOULD_MOVE_BACKWARD')
    index['collections'].append({'collected_at_utc':iso(received),'new_events':newer,
                                  'provider_status':stats})
    # finite disk bound for each run; no historical rewrites/truncation.
    return newer


def event_assessment(ev,asof=None):
    asof=asof or now(); p=parse_time(ev['published_utc']); o=parse_time(ev['first_observed_utc'])
    if p>o or o>asof+timedelta(minutes=2):return 'BAD_TIMESTAMP'
    if (asof-p)>timedelta(minutes=FRESH_MINUTES):return 'STALE_PUBLICATION'
    if (asof-o)>timedelta(minutes=FRESH_MINUTES):return 'STALE_OBSERVATION'
    text=ev['headline'].casefold()
    if any(w in text for w in REACTION):return 'PRICE_REACTION_HEADLINE'
    if not any(w in text for w in RELEVANT):return 'LOW_EVENT_SPECIFICITY'
    if not any(name in text for name in ALIASES[ev['symbol']]):return 'ISSUER_UNVERIFIED'
    return 'CANDIDATE_NEEDS_EVIDENCE_CHECK'


def do_collect(root,limit,days):
    if not (root/'tools'/'auto_submit.py').is_file():raise ValueError('STARTER_KIT_DIRECTORY_REQUIRED')
    key=config_token();index=read_index(root)
    ts=now();all_rows=[];stats={};rate_limited=False
    print('PHASE35_FINNHUB_READ_ONLY_START SYMBOLS:',limit,'NO_LLM_NO_TRADES',flush=True)
    for k,sym in enumerate(STOCKS[:limit],1):
        if rate_limited:
            stats[sym]='NOT_ATTEMPTED_AFTER_429';continue
        try:
            raw,at=get_news(sym,key,days)
            rows=to_records(sym,raw,at);all_rows.extend(rows)
            stats[sym]='OK_'+str(len(rows))
        except ValueError as exc:
            msg=str(exc)
            if msg not in ('FINNHUB_NON_200','FINNHUB_RESPONSE_TOO_LARGE','FINNHUB_BAD_JSON',
                           'FINNHUB_NOT_NEWS_LIST','FINNHUB_NETWORK_ERROR') and not re.fullmatch(r'FINNHUB_HTTP_[0-9]{3}',msg):
                msg='SOURCE_ERROR'
            stats[sym]=msg
            if msg=='FINNHUB_HTTP_429':rate_limited=True
        print(f'{k:02d}/{limit:02d} {sym} {stats[sym]}',flush=True)
        if k<limit and not rate_limited:time.sleep(1.1)
    # Never save credentials in output; archive raw FINNHUB summaries with time proof.
    got=integrate(index,all_rows,now(),stats)
    identifier=ts.strftime('%Y%m%dT%H%M%S%fZ')
    snap=root/OUT/'snapshots'/f'phase35_capture_{identifier}.json'
    snap.parent.mkdir(parents=True,exist_ok=True)
    with snap.open('x',encoding='utf-8') as h:
        json.dump({'read_only':True,'observed_not_historically_backtestable':True,
                   'collected_at_utc':iso(ts),'symbols_requested':list(STOCKS[:limit]),
                   'statuses':stats,'records':all_rows},h,indent=2,ensure_ascii=False)
    atomic_json(root/OUT/'event_index.json',index)
    print('PHASE35_CAPTURE_COMPLETE_NEW_EVENTS:',got,'TOTAL_DISTINCT:',len(index['events']))
    print('FRESH_MATERIAL_ISSUER_CANDIDATES:',sum(event_assessment(x)=='CANDIDATE_NEEDS_EVIDENCE_CHECK' for x in index['events'].values()))
    print('SNAPSHOT_PATH:',snap)
    print('NO_LLM_CREDITS_USED_NO_ORGANIZER_SUBMISSION')


def do_status(root,prepare):
    index=read_index(root)
    rows=list(index['events'].values())
    statuses={}
    for x in rows:
        status=event_assessment(x);statuses[status]=statuses.get(status,0)+1
    print('PHASE35_DISTINCT_EVENTS:',len(rows),'COLLECTIONS:',len(index['collections']))
    print('PHASE35_EVENT_STATUS_COUNTS:',json.dumps(statuses,sort_keys=True))
    eligible=sorted((x for x in rows if event_assessment(x)=='CANDIDATE_NEEDS_EVIDENCE_CHECK'),
                    key=lambda x:parse_time(x['published_utc']),reverse=True)
    for x in eligible[:5]:
        print('FRESH_CANDIDATE:',x['symbol'],x['published_utc'],x['headline'][:190])
    if prepare:
        if not eligible:
            print('PHASE35_NO_FRESH_ISSUER_SPECIFIC_EVENT_NO_PROVIDER_CALL')
            return
        ev=eligible[0].copy()
        # Phase34 schema; explicitly not company-verified or historically tradable.
        ev['source_verification']='FINNHUB_HEADER_AUTH_READ_ONLY_NO_ISSUER_FACT_CHECK'
        target=root/OUT/'phase35_phase34_candidate_event.json'
        atomic_json(target,ev)
        print('PHASE35_EVENT_PREPARED_FOR_MANUAL_REVIEW:',target)
        print('CHECK_SOURCE_TEXT_AND_EXPECTATIONS_BEFORE_OPTIONAL_PROVIDER_PROBE')
    print('NO_FINNHUB_CALL_NO_LLM_CALL_NO_TRADES')


def self_test():
    at=datetime(2026,10,11,12,0,tzinfo=UTC)
    row={'id':100,'datetime':int((at-timedelta(minutes=12)).timestamp()),
         'headline':'Apple raises full-year revenue guidance above expectations',
         'source':'Test','url':'https://example.com/a'}
    a=to_records('AAPL',[row,row],at)
    assert len(a)==1 and event_assessment(a[0],asof=at)=='CANDIDATE_NEEDS_EVIDENCE_CHECK'
    assert event_assessment(a[0],asof=at+timedelta(days=1))=='STALE_PUBLICATION'
    i={'version':1,'events':{},'collections':[]}
    integrate(i,a,at,{'AAPL':'OK_1'})
    newer=dict(a[0],first_observed_utc=iso(at+timedelta(minutes=20)))
    integrate(i,[newer],at+timedelta(minutes=20),{'AAPL':'OK_1'})
    assert i['events'][identity(a[0])]['first_observed_utc']==iso(at)
    assert event_assessment(dict(a[0],symbol='NVDA'),asof=at)=='ISSUER_UNVERIFIED'
    assert event_assessment(dict(a[0],headline='Apple shares surged after guidance'),asof=at)=='PRICE_REACTION_HEADLINE'
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'ledger.json';atomic_json(p,i)
        assert read_index(Path(t))['events']=={} and json.loads(p.read_text())['version']==1
    assert len(STOCKS)==30 and len(ALIASES)==30
    print('PHASE35_SELF_TEST_PASS')
    print('PUBLICATION_AND_FIRST_OBSERVED_FRESHNESS_GATES_PASS')
    print('IMMUTABLE_FIRST_SEEN_DEDUP_ISSUER_SCREEN_NO_HISTORICAL_BACKFILL_PASS')
    print('NO_NETWORK_NO_LLM_NO_TRADES_NO_CREDENTIALS_READ')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-test',action='store_true');g.add_argument('--status',action='store_true')
    g.add_argument('--collect',action='store_true');g.add_argument('--prepare',action='store_true')
    ap.add_argument('--limit',type=int,default=30);ap.add_argument('--news-days',type=int,default=2)
    a=ap.parse_args()
    if not 1<=a.limit<=30 or not 1<=a.news_days<=7:ap.error('limit 1..30, news-days 1..7')
    try:
        if a.self_test:self_test();return
        root=Path.cwd().resolve()
        if a.collect:do_collect(root,a.limit,a.news_days)
        else:do_status(root,a.prepare)
    except (ValueError,FileNotFoundError,RuntimeError) as exc:
        safe=str(exc)
        if safe not in ('FINNHUB_KEY_NOT_CONFIGURED_PRIVATELY','STARTER_KIT_DIRECTORY_REQUIRED',
                        'INDEX_INVALID','FIRST_OBSERVATION_WOULD_MOVE_BACKWARD'):
            safe=type(exc).__name__
        print('PHASE35_STOP:',safe,file=sys.stderr)
        raise SystemExit(2)

if __name__=='__main__':main()
