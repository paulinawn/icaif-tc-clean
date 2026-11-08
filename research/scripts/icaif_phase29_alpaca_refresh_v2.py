#!/usr/bin/env python3
"""ICAIF Phase 29 v2: read-only authenticated Alpaca IEX historical refresh + safe local merge.

No organizer endpoints, orders, trades, or writes to existing source datasets.
Run from the competition starter-kit directory:
  python "$HOME\\Downloads\\icaif_phase29_alpaca_refresh.py" --self-test
  python "$HOME\\Downloads\\icaif_phase29_alpaca_refresh.py" --refresh

Credentials: APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables;
alternatively --env-file PATH to a private KEY=VALUE file outside Git.

Hard gates: exact universe, feed=iex, adjustment=raw, overlap Oct 6 equality,
complete Oct 7,8,9 regular trading sessions, price/volume/candle sanity,
no future records, no duplicates and unchanged original CSV.
"""
from __future__ import annotations
import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from urllib import request, parse, error

import numpy as np
import pandas as pd

SYMBOLS=('AAPL','MSFT','NVDA','INTC','CRM','JPM','BAC','GS','V','PYPL',
 'LLY','JNJ','UNH','PFE','TMO','AMZN','TSLA','WMT','NKE','KO','CAT','GE',
 'BA','XOM','CVX','GOOGL','META','DIS','T','NEE')
COLS=('symbol','timestamp','open','high','low','close','volume','trade_count','vwap')
SOURCE=Path('data/cache/alpaca_iex_15m_400d.csv')
OUT=Path('artifacts/phase29_alpaca_refresh')
DAY_START='2026-10-06'  # mandatory one-day overlap to validate adjustments/feed
DAYS=('2026-10-07','2026-10-08','2026-10-09')
FEED='iex'
ADJUSTMENT='raw'
API='https://data.alpaca.markets/v2/stocks/bars'
ET='America/New_York'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_creds(envfile=None):
    env={}
    if envfile:
        path=Path(envfile).expanduser()
        if not path.is_file():raise FileNotFoundError('Credential file does not exist: '+str(path))
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            s=line.strip()
            if not s or s.startswith('#'):continue
            if s.startswith('export '):s=s[7:]
            if '=' in s:
                k,v=s.split('=',1)
                env[k.strip()]=v.strip().strip('"').strip("'")
    # When an explicit private --env-file is supplied, prefer its keys over
    # potentially stale inherited shell environment variables.
    env = {**dict(os.environ), **env}
    key=env.get('APCA_API_KEY_ID') or env.get('ALPACA_API_KEY') or env.get('ALPACA_API_KEY_ID')
    secret=env.get('APCA_API_SECRET_KEY') or env.get('ALPACA_SECRET_KEY') or env.get('ALPACA_API_SECRET_KEY')
    if not key or not secret:
        raise RuntimeError('Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the current PowerShell process, or pass --env-file to a private credentials file. Do NOT paste keys into chat.')
    return {'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret,'Accept':'application/json'}


def fetch(headers,transport=None):
    """Download ALL pages (API may group bars by ticker instead of by timestamp)."""
    if transport is None:transport=api_get
    args={'symbols':','.join(SYMBOLS),'timeframe':'15Min',
          'start':'2026-10-06T00:00:00-04:00',
          'end':'2026-10-10T00:00:00-04:00',
          'limit':'10000','feed':FEED,'adjustment':ADJUSTMENT,'sort':'asc'}
    pages=[]; tokens=set()
    for page_i in range(30):
        obj=transport(args,headers)
        if not isinstance(obj,dict) or not isinstance(obj.get('bars'),dict):
            raise ValueError('Unexpected Alpaca response schema: expected bars by symbol')
        for sym,items in obj['bars'].items():
            if sym not in SYMBOLS:raise ValueError('Unexpected API symbol '+sym)
            for v in items:
                pages.append({'symbol':sym,'timestamp':v['t'],'open':v['o'],'high':v['h'],
                              'low':v['l'],'close':v['c'],'volume':v['v'],
                              'trade_count':v.get('n',0),'vwap':v.get('vw',v['c'])})
        token=obj.get('next_page_token')
        if not token:break
        if token in tokens:raise RuntimeError('Alpaca pagination token loop')
        tokens.add(token);args['page_token']=token
    else:raise RuntimeError('Exceeded Alpaca pagination safety limit')
    if not pages:raise RuntimeError('Alpaca returned no bars')
    return pd.DataFrame(pages,columns=COLS),len(tokens)+1


def api_get(params,headers):
    url=API+'?'+parse.urlencode(params)
    for attempt in range(5):
        req=request.Request(url,headers=headers)
        try:
            with request.urlopen(req,timeout=30) as resp:
                return json.loads(resp.read(5_000_000).decode('utf-8'))
        except error.HTTPError as exc:
            if exc.code in (401,403):raise RuntimeError(f'Alpaca HTTP {exc.code}: check permissions/credentials/feed; keys redacted') from None
            if exc.code not in (429,500,502,503,504):
                raise RuntimeError(f'Alpaca HTTP {exc.code}: request rejected; no dataset modified') from None
            if attempt==4:raise RuntimeError(f'Alpaca HTTP {exc.code} after retries') from None
            time.sleep(min(8,2**attempt))
        except (error.URLError,TimeoutError) as exc:
            if attempt==4:raise RuntimeError('Alpaca connection/timeout after retries') from None
            time.sleep(min(8,2**attempt))
    raise RuntimeError('Alpaca request failed')


def normalize(raw):
    required={'symbol','timestamp','open','high','low','close'}
    if not required.issubset(raw.columns):raise ValueError('CSV/API missing columns: '+str(sorted(required-set(raw.columns))))
    x=raw.copy()
    x['symbol']=x['symbol'].astype(str).str.upper().str.strip()
    if (~x.symbol.isin(SYMBOLS)).any():raise ValueError('Unexpected symbol in dataset')
    utc=pd.to_datetime(x.timestamp,utc=True,errors='coerce')
    if utc.isna().any():raise ValueError('Invalid timestamp values')
    x['ts_utc']=utc
    local=utc.dt.tz_convert(ET)
    x['day_et']=local.dt.strftime('%Y-%m-%d')
    x['hm_et']=local.dt.strftime('%H:%M')
    x=x[(x.hm_et>='09:30')&(x.hm_et<='15:45')&(
        local.dt.minute.mod(15)==0)].copy()
    x['timestamp']=x.ts_utc.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    for c in ('open','high','low','close','volume','trade_count','vwap'):
        if c not in x.columns:raise ValueError('Missing '+c)
        x[c]=pd.to_numeric(x[c],errors='coerce')
    for c in ('open','high','low','close','vwap'):
        if (~np.isfinite(x[c]) | (x[c]<=0)).any():raise ValueError('Nonpositive/missing price: '+c)
    if (x.high+1e-8<x[['open','close','low']].max(axis=1)).any() or (x.low-1e-8>x[['open','close','high']].min(axis=1)).any():
        raise ValueError('Invalid OHLC bar with high below or low above open/close')
    if (x.volume<0).any() or (x.trade_count<0).any():raise ValueError('Negative volume/trades')
    if x.duplicated(['symbol','timestamp']).any():raise ValueError('Duplicate (symbol,timestamp) records')
    return x.reset_index(drop=True)


def regular_slots():
    return [(datetime(2026,10,7,9,30)+timedelta(minutes=15*i)).strftime('%H:%M') for i in range(26)]


def audit_merge(old,new):
    old=normalize(old);new=normalize(new)
    overlap_old=old[old.day_et==DAY_START]
    overlap_new=new[new.day_et==DAY_START]
    if set(overlap_old.symbol)!=set(SYMBOLS) or set(overlap_new.symbol)!=set(SYMBOLS):
        raise ValueError('Overlapping October 6 data must contain all 30 symbols')
    match=overlap_old.merge(overlap_new,on=['symbol','timestamp'],suffixes=('_old','_api'),how='inner')
    if len(match)<0.90*len(overlap_old):
        raise ValueError(f'Overlapping October 6 coverage too low: {len(match)}/{len(overlap_old)}')
    deviations={}
    for c in ('open','high','low','close'):
        worst=float(np.max(np.abs(match[f'{c}_old']/match[f'{c}_api']-1)))
        deviations[c]=worst
    if max(deviations.values())>0.0025:
        raise ValueError(f'PRICE_BASIS_MISMATCH_ABORT: Oct 6 IEX/raw prices differ; max relative deviation {max(deviations.values()):.6f}. Existing source not changed. Confirm original feed/adjustment.')
    fresh=new[new.day_et.isin(DAYS)].copy()
    if not set(DAYS).issubset(set(fresh.day_et)):
        raise ValueError('At least one requested October 7–9 day has no downloaded data')
    complete=[];missing=[]
    slots=set(regular_slots())
    for day in DAYS:
        for sym in SYMBOLS:
            observed=set(fresh.loc[(fresh.day_et==day)&(fresh.symbol==sym),'hm_et'])
            if not observed:raise ValueError(f'No bars for {sym} on {day}')
            for tm in sorted(slots-observed):missing.append({'day':day,'symbol':sym,'bar_start_et':tm})
            complete.append({'day':day,'symbol':sym,'bars':len(observed)})
    if len(missing)>0.05*(len(DAYS)*len(SYMBOLS)*26):
        raise ValueError(f'More than 5% expected intraday IEX bars missing ({len(missing)}); abort merge.')
    conflict=old.merge(fresh,on=['symbol','timestamp'],suffixes=('_old','_new'),how='inner')
    if len(conflict):
        max_diff=float(np.max(np.abs(conflict.close_old/conflict.close_new-1)))
        if max_diff>0.0025:raise ValueError('Conflicting existing and newly fetched October bars')
    merged=pd.concat([old[list(COLS)],fresh[list(COLS)]],ignore_index=True)
    merged=merged.drop_duplicates(['symbol','timestamp'],keep='first').sort_values(['symbol','timestamp']).reset_index(drop=True)
    if merged.duplicated(['symbol','timestamp']).any():raise AssertionError('Unexpected duplicates')
    tmp=normalize(merged)
    if set(tmp.day_et[tmp.day_et>'2026-10-09']):raise ValueError('Future/post-cutoff records in merged data')
    # Whole-file gap report: flag, but do not mistake high market volatility for corrupted data.
    z=tmp.sort_values(['symbol','ts_utc'])
    ratio=z.groupby('symbol').close.shift(1)
    jump=z.open/ratio-1
    big=z.loc[jump.abs()>.15,['symbol','timestamp','open','close']].copy()
    big['prior_bar_gap_fraction']=jump.loc[big.index].values
    meta={'status':'PROVISIONAL_MERGE_PASS','source_feed':FEED,'requested_adjustment':ADJUSTMENT,
          'original_rows':len(old),'fetched_oct07_09_rows':len(fresh),'merged_rows':len(merged),
          'overlap_oct06_compared':len(match),'overlap_max_ohlc_deviation':deviations,
          'missing_bar_count':len(missing),'missing_expected_bars_pct':len(missing)/(len(DAYS)*30*26),
          'fetched_by_date':{d:int((fresh.day_et==d).sum()) for d in DAYS},
          'merged_first_day':str(tmp.day_et.min()),'merged_last_day':str(tmp.day_et.max()),
          'notes':['Original source never overwritten','IEX is exchange-specific, not SIP',
                   'Requested raw is crosschecked only via Oct 6 price overlap',
                   'Intraday IEX may omit bars with no eligible trades',
                   '2026 rows are historical only; no API data beyond Oct 9 is requested',
                   'Unadjusted total returns require corporate-action treatment']}
    return merged,meta,pd.DataFrame(missing),pd.DataFrame(complete),big


def save_results(root,source,merged,meta,miss,cov,gaps,pages):
    out=root/OUT
    out.mkdir(parents=True,exist_ok=True)
    outcsv=out/'alpaca_iex_15m_through_2026-10-09.csv'
    merged.to_csv(outcsv,index=False,float_format='%.10g')
    meta.update({'original_sha256':sha256(source),'merged_sha256':sha256(outcsv),
                 'api_pages':pages,'merged_file':str(outcsv)})
    (out/'phase29_refresh_report.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    miss.to_csv(out/'phase29_missing_bars.csv',index=False)
    cov.to_csv(out/'phase29_coverage.csv',index=False)
    gaps.to_csv(out/'phase29_large_price_gaps.csv',index=False)
    return out


def self_test():
    import tempfile
    slots=regular_slots()
    assert len(slots)==26 and slots[0]=='09:30' and slots[-1]=='15:45'
    days=(DAY_START,)+DAYS
    rows=[]
    for day_i,d in enumerate(days):
        for sym_i,sym in enumerate(SYMBOLS):
            for t in slots:
                ts=pd.Timestamp(f'{d} {t}',tz=ET).tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
                price=100.+sym_i+day_i/10.
                rows.append(dict(symbol=sym,timestamp=ts,open=price,high=price+.1,low=price-.1,close=price,
                                 volume=100,trade_count=2,vwap=price))
    source=pd.DataFrame([r for r in rows if r['timestamp']<pd.Timestamp('2026-10-07',tz=ET).tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')])
    fetched=pd.DataFrame(rows)
    merged,meta,miss,cov,gaps=audit_merge(source,fetched)
    assert meta['fetched_oct07_09_rows']==30*26*3 and len(merged)==len(rows)
    assert len(miss)==0 and meta['overlap_oct06_compared']==30*26
    def transport(args,headers):
        if 'page_token' not in args:
            return {'bars':{'AAPL':[{'t':rows[0]['timestamp'],'o':100.,'h':100.1,'l':99.9,'c':100.,'v':100,'n':2,'vw':100.}]},'next_page_token':'p2'}
        return {'bars':{'MSFT':[]},'next_page_token':None}
    z,n=fetch({},transport)
    assert len(z)==1 and n==2
    tampered=fetched.copy(); mask=(tampered.symbol=='AAPL')&(normalize(tampered).day_et==DAY_START)
    for field in ('open','high','low','close','vwap'):tampered.loc[mask,field]*=1.1
    try:audit_merge(source,tampered)
    except ValueError as exc:assert 'MISMATCH' in str(exc)
    else:raise AssertionError('Overlapping price mismatch was not blocked')
    print('PHASE29_SELF_TEST_PASS')
    print('PAGINATION_TIMESTAMP_MERGE_AND_OVERLAP_GATES_PASS')
    print('NO_TRADES_NO_ORGANIZER_CALLS')


def run_auth_check(args):
    """One read-only AAPL historical bars request; never prints credentials."""
    headers=load_creds(args.env_file)
    print('ALPACA_AUTH_CHECK_START_READ_ONLY',flush=True)
    params={'symbols':'AAPL','timeframe':'15Min',
            'start':'2026-10-06T09:30:00-04:00',
            'end':'2026-10-06T11:00:00-04:00',
            'limit':'100','feed':FEED,'adjustment':ADJUSTMENT}
    obj=api_get(params,headers)
    bars=obj.get('bars',{}).get('AAPL',[]) if isinstance(obj,dict) else []
    print('ALPACA_AUTHENTICATED_READ_ONLY_OK',flush=True)
    print('AAPL_BAR_COUNT:',len(bars),flush=True)
    print('NO_KEYS_PRINTED_NO_TRADES_NO_COMPETITION_SUBMISSION',flush=True)


def run_refresh(args):
    root=Path.cwd()
    source=root/SOURCE
    if not source.is_file():raise FileNotFoundError('Expected source file not found: '+str(source))
    headers=load_creds(args.env_file)
    print('FETCHING_ALPACA_IEX_RAW_OCT06_OVERLAP_AND_OCT07_TO_09',flush=True)
    fetched,pages=fetch(headers)
    print('ALPACA_PAGES:',pages,'RAW_BARS:',len(fetched),flush=True)
    raw=pd.read_csv(source)
    merged,meta,miss,cov,gaps=audit_merge(raw,fetched)
    out=save_results(root,source,merged,meta,miss,cov,gaps,pages)
    print('OVERLAP_OCT06_COMPARED:',meta['overlap_oct06_compared'])
    print('FRESH_DAYS:',meta['fetched_by_date'])
    print('MISSING_IEX_BAR_SLOTS:',meta['missing_bar_count'])
    print('MERGED_ROWS:',meta['merged_rows'])
    print('OUTPUT_DIR:',out)
    print('PHASE29_REFRESH_COMPLETE_NO_SUBMISSION')


def main():
    p=argparse.ArgumentParser()
    x=p.add_mutually_exclusive_group(required=True)
    x.add_argument('--self-test',action='store_true');x.add_argument('--refresh',action='store_true');x.add_argument('--check-auth',action='store_true')
    p.add_argument('--env-file',default=None,help='Optional PRIVATE Alpaca KEY=VALUE file, outside Git')
    args=p.parse_args()
    try:
        if args.self_test:self_test()
        elif args.check_auth:run_auth_check(args)
        else:run_refresh(args)
    except Exception as exc:
        print('PHASE29_FAILED:',str(exc),file=sys.stderr)
        sys.exit(2)

if __name__=='__main__':main()
