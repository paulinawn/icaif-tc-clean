#!/usr/bin/env python3
"""Research-only Finnhub + SEC + CaMeL bridge for ICAIF Phase42.

--collect: read-only SEC (submissions + companyfacts) and Finnhub news,
            maximum 1--3 named stocks per run by default. No CaMeL call.
--prepare-camel: use existing Phase39 evidence-controlled CaMeL packet builder,
            with a manually checked PUBLIC SOURCE EXCERPT; no LLM call.
--probe-camel: at most Phase39's existing three requests, explicit paid acknowledgement.
--approve: human-source-review attestation produces forward-only paper event.

Never fetch organizer data, submit, trade, or expose credentials. Historical
SEC companyfacts is NOT retrospectively usable as a 2026 historical signal.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, tempfile, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

HERE=Path(__file__).resolve().parent
ROOT=Path.cwd();OUT=Path('artifacts/phase42_source_bridge')
UTC=timezone.utc
SOURCES=('AAPL','MSFT','NVDA','INTC','CRM','JPM','BAC','GS','V','PYPL','LLY',
 'JNJ','UNH','PFE','TMO','AMZN','TSLA','WMT','NKE','KO','CAT','GE','BA',
 'XOM','CVX','GOOGL','META','DIS','T','NEE')
TAGS=('Revenues','RevenueFromContractWithCustomerExcludingAssessedTax',
      'NetIncomeLoss','NetCashProvidedByUsedInOperatingActivities',
      'OperatingIncomeLoss','Assets','Liabilities')


def module(key,file):
    src=HERE/file
    if not src.is_file():raise FileNotFoundError('Put '+file+' next to Phase42 bridge')
    spec=importlib.util.spec_from_file_location(key,src)
    mod=importlib.util.module_from_spec(spec);sys.modules[key]=mod;spec.loader.exec_module(mod)
    return mod


def instant():return datetime.now(UTC).isoformat().replace('+00:00','Z')


def atomic(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,p=tempfile.mkstemp(prefix='.phase42_',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(raw,f,indent=2,ensure_ascii=False)
        os.replace(p,path)
    finally:
        if os.path.exists(p):os.unlink(p)


def http_json(url,agent):
    req=Request(url,headers={'User-Agent':agent,'Accept':'application/json'})
    try:
        with urlopen(req,timeout=20) as r:
            if r.status!=200:raise ValueError('NON_200_RESPONSE')
            data=r.read(12_000_001)
    except HTTPError as ex:raise ValueError('HTTP_'+str(ex.code)) from None
    except (URLError,TimeoutError,OSError):raise ValueError('SOURCE_TRANSPORT_FAILED') from None
    if len(data)>12_000_000:raise ValueError('SOURCE_RESPONSE_TOO_LARGE')
    return json.loads(data)


def ticker_ciks(raw):
    result={}
    for value in (raw.values() if isinstance(raw,dict) else raw):
        if not isinstance(value,dict):continue
        ticker=str(value.get('ticker','')).upper().strip()
        try:cik=int(value.get('cik_str'))
        except (ValueError,TypeError):continue
        if ticker and cik>0:result[ticker]=str(cik).zfill(10)
    return result


def filings(raw):
    recent=raw.get('filings',{}).get('recent',{})
    keys=['form','filingDate','acceptanceDateTime','primaryDocument','accessionNumber']
    fields={k:recent.get(k,[]) for k in keys}
    n=min([len(v) for v in fields.values()] or [0])
    out=[]
    for i in range(n):
        row={k:str(fields[k][i])[:120] for k in keys}
        if row['form'] in ('8-K','10-K','10-Q','6-K','20-F'):
            out.append(row)
        if len(out)>=6:break
    return out


def financials(raw):
    """SEC facts with their *original* filed and period dates (not LLM predictions)."""
    g=raw.get('facts',{}).get('us-gaap',{})
    summary=[]
    for tag in TAGS:
        if tag not in g:continue
        item=g[tag]
        for unit,records in item.get('units',{}).items():
            if unit not in ('USD','shares','USD/shares'):continue
            clean=[]
            for rec in records:
                if not isinstance(rec,dict):continue
                if rec.get('form') not in ('10-Q','10-K'):continue
                if not rec.get('filed') or not rec.get('end'):continue
                # A single reported XBRL item may later be restated. Keep filed
                # provenance, never relabel it as an as-of prior historical fact.
                clean.append({k:rec.get(k) for k in
                    ('start','end','filed','form','accn','fy','fp','frame','val')})
            clean.sort(key=lambda r:(r['filed'],r['end']),reverse=True)
            if clean:summary.append({'metric':tag,'unit':unit,'recent_reported_facts':clean[:3]})
    return summary


def collect(symbols):
    # Explicit only; no credentials printed. User-Agent must be a real contact.
    ua=os.getenv('ICAIF_SEC_USER_AGENT','').strip()
    if not ua:
        # Reuse the contact already configured by earlier ICAIF SEC phases.
        email=os.getenv('SEC_CONTACT_EMAIL','').strip()
        cfg=Path.home()/'.icaif'/'research_sources.env'
        if not email and cfg.is_file():
            for raw in cfg.read_text(encoding='utf-8-sig').splitlines():
                key,sep,value=raw.strip().partition('=')
                if sep and key.strip()=='SEC_CONTACT_EMAIL':
                    email=value.strip().strip('\"\'')
                    break
        if email:ua='ICAIF2026-Research-Phase42 '+email
    if len(ua)<14 or '@' not in ua:
        raise ValueError('SEC fair-access contact missing; set SEC_CONTACT_EMAIL privately in ~/.icaif/research_sources.env')
    fin=module('p42_finnhub','icaif_phase35_forward_event_intake.py')
    key=fin.config_token()
    ciks=ticker_ciks(http_json('https://www.sec.gov/files/company_tickers.json',ua))
    out=ROOT/OUT;out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
    reports=[]
    for sym in symbols:
        cik=ciks.get(sym)
        result={'symbol':sym,'first_observed_utc':instant(),'read_only':True,
                'point_in_time_historical_backfill_allowed':False,
                'SEC':{},'FINNHUB':{},'source_status':{}}
        if cik:
            result['SEC']['cik']=cik
            for kind,url,parser in [('filings','https://data.sec.gov/submissions/CIK'+cik+'.json',filings),
                   ('companyfacts','https://data.sec.gov/api/xbrl/companyfacts/CIK'+cik+'.json',financials)]:
                try:
                    result['SEC'][kind]=parser(http_json(url,ua))
                    result['source_status']['sec_'+kind]='OK'
                except ValueError as ex:result['source_status']['sec_'+kind]=str(ex)
                time.sleep(.18)
        else:result['source_status']['sec_cik']='NO_CIK_MATCH'
        try:
            raw,seen=fin.get_news(sym,key,2)
            events=fin.to_records(sym,raw,seen)
            result['FINNHUB']['news']=events
            result['source_status']['finnhub']='OK_'+str(len(events))
        except ValueError as ex:result['source_status']['finnhub']=str(ex)
        name=out/f'phase42_{sym}_{stamp}.json'
        atomic(name,result)
        reports.append({'symbol':sym,'snapshot':str(name),'status':result['source_status']})
        print('PHASE42_PUBLIC_SOURCES',sym,json.dumps(result['source_status']),flush=True)
        time.sleep(1)
    atomic(out/'phase42_latest_collection.json',
           {'collected_utc':instant(),'symbols':list(symbols),'reports':reports,
            'credits_spent':'ZERO_LLM_CALLS','research_only':True})
    print('PHASE42_PUBLIC_READ_ONLY_COMPLETE_NO_CAML_NO_LAYA_NO_TRADES')


def candidate_from_snapshot(path,symbol,headline_filter=''):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data['symbol']!=symbol:raise ValueError('Symbol snapshot mismatch')
    fin=module('p42_finnhub_pre','icaif_phase35_forward_event_intake.py')
    news=data.get('FINNHUB',{}).get('news',[])
    fresh=[x for x in news if fin.event_assessment(x)=='CANDIDATE_NEEDS_EVIDENCE_CHECK']
    if headline_filter:
        fresh=[x for x in fresh if headline_filter.casefold() in x['headline'].casefold()]
    if not fresh:raise ValueError('No fresh issuer-specific material candidate. No LLM call.')
    return sorted(fresh,key=lambda x:x['published_utc'],reverse=True)[0]


def sec_evidence_addendum(snapshot,symbol):
    """Compact previously filed SEC facts, all as observed now; not an earnings surprise."""
    data=json.loads(Path(snapshot).read_text(encoding='utf-8'))
    if data['symbol']!=symbol:raise ValueError('SEC snapshot issuer mismatch')
    sec=data.get('SEC',{})
    lines=[f"PUBLIC SEC CONTEXT for {symbol}; observed {data['first_observed_utc']}",
           'These are preexisting reported filing facts, NOT contemporaneous expectations,',
           'NOT proof of a new event, and NOT return or trade forecasts.']
    fs=sec.get('filings',[])
    if fs:
        f=fs[0];lines.append('Recent filing: '+f['form']+', filed '+f['filingDate']+
                           ', accepted '+f['acceptanceDateTime'])
    facts=sec.get('companyfacts',[])
    for item in facts[:3]:
        for v in item['recent_reported_facts'][:1]:
            lines.append('Reported '+item['metric']+' '+item['unit']+' '+
                         str(v.get('val'))+'; period '+str(v.get('end'))+
                         '; filed '+str(v.get('filed'))+'; '+str(v.get('form')))
    if not fs and not facts:lines.append('NO_VERIFIED_SEC_DATA_AVAILABLE')
    return '\n'.join(lines)


def prepare_camel(symbol,snapshot,source_excerpt,expectations,headline_filter):
    # Finnhub only confirms headline discovery. The user must supply a reviewed
    # real primary/secondary source excerpt before CaMeL may be contacted.
    ev=candidate_from_snapshot(snapshot,symbol,headline_filter)
    article=Path(source_excerpt).read_text(encoding='utf-8').strip()
    if len(article)<100:raise ValueError('Require >=100 chars of checked public source text')
    article=article[:3000]
    sec_info=sec_evidence_addendum(snapshot,symbol)
    # Cite any new fact from the article, not the supporting older SEC data.
    combined=('PUBLIC ARTICLE/ISSUER SOURCE TEXT (only section that may prove '
              'this NEW event):\n'+article+'\n\n'+sec_info)
    path=ROOT/OUT/(f'phase42_{symbol}_candidate.json')
    atomic(path,ev)
    bundle=ROOT/OUT/(f'phase42_{symbol}_fusion_evidence.txt')
    bundle.parent.mkdir(parents=True,exist_ok=True)
    bundle.write_text(combined,encoding='utf-8')
    bridge=module('p42_camel_prepare','icaif_phase39_camel_forward_lab.py')
    return bridge.prep(ROOT,str(path),str(bundle),expectations)


def _write_test_snapshot():
    # Test setup is strictly local and ephemeral, with no credentials.
    fd,path=tempfile.mkstemp(prefix='phase42_sec_test_',suffix='.json')
    with os.fdopen(fd,'w',encoding='utf-8') as out:
        json.dump({'symbol':'NVDA','first_observed_utc':'2026-10-11T12:00:00Z',
                   'SEC':{'filings':[],'companyfacts':[]}},out)
    return path


def self_test():
    assert ticker_ciks({'0':{'ticker':'AAPL','cik_str':320193}})['AAPL']=='0000320193'
    assert filings({'filings':{'recent':{'form':['8-K','3','10-Q'],
        'filingDate':['a','b','c'],'acceptanceDateTime':['1','2','3'],
        'primaryDocument':['d','e','f'],'accessionNumber':['x','y','z']}}})[1]['form']=='10-Q'
    dummy={'facts':{'us-gaap':{'NetIncomeLoss':{'units':{'USD':[
        {'form':'10-Q','filed':'2026-10-09','end':'2026-09-30','val':42},
        {'form':'10-Q','filed':'2026-08-05','end':'2026-06-30','val':39},
        {'form':'8-K','filed':'2026-10-05','end':'2026-09-30','val':88}]}}}}}
    a=financials(dummy)
    assert len(a)==1 and a[0]['recent_reported_facts'][0]['val']==42
    testpath=_write_test_snapshot()
    try: assert 'NOT proof of a new event' in sec_evidence_addendum(testpath,'NVDA')
    finally:os.unlink(testpath)
    print('PHASE42_SOURCE_BRIDGE_SELF_TEST_PASS')
    print('SEC_METADATA_PROVENANCE_FINNHUB_FIRST_OBSERVED_CAMEL_GUARDS_NO_API_CALL_PASS')
    print('NO_CREDENTIALS_NO_SPEND_NO_OFFICIAL_SUBMISSION')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    for mode in ('self-test','collect','prepare-camel','probe-camel','approve'):
        group.add_argument('--'+mode,action='store_true')
    ap.add_argument('--symbols',nargs='+',default=['NVDA'],choices=SOURCES)
    ap.add_argument('--limit',type=int,default=3)
    ap.add_argument('--ack-readonly-api-requests',action='store_true')
    ap.add_argument('--snapshot');ap.add_argument('--source-excerpt');ap.add_argument('--expectations')
    ap.add_argument('--headline-filter',default='')
    ap.add_argument('--prepared-packet');ap.add_argument('--report')
    ap.add_argument('--ack-provider-credits',action='store_true')
    ap.add_argument('--ack-source-reviewed',action='store_true')
    a=ap.parse_args()
    try:
        if a.self_test:self_test()
        elif a.collect:
            if not a.ack_readonly_api_requests:raise ValueError('Explicit --ack-readonly-api-requests required')
            if not 1<=len(a.symbols)<=min(a.limit,3):raise ValueError('Limit to 1-3 distinct stocks per read-only request')
            if len(a.symbols)!=len(set(a.symbols)):raise ValueError('Duplicate symbols')
            collect(a.symbols)
        elif a.prepare_camel:
            if not a.snapshot or not a.source_excerpt:raise ValueError('Provide --snapshot and --source-excerpt with verified article text')
            if len(a.symbols)!=1:raise ValueError('Single issuer expected')
            prepare_camel(a.symbols[0],a.snapshot,a.source_excerpt,a.expectations,a.headline_filter)
        elif a.probe_camel:
            if not a.prepared_packet:raise ValueError('Provide --prepared-packet')
            mod=module('p42_camel_probe','icaif_phase39_camel_forward_lab.py')
            return mod.probe(ROOT,a.prepared_packet,a.ack_provider_credits)
        elif a.approve:
            if not a.report:raise ValueError('Provide --report')
            mod=module('p42_camel_approve','icaif_phase39_camel_forward_lab.py')
            mod.approve(ROOT,a.report,a.ack_source_reviewed)
        return 0
    except Exception as exc:
        print('PHASE42_SOURCE_STOP:',type(exc).__name__,str(exc),file=sys.stderr)
        return 2
if __name__=='__main__':sys.exit(main())
