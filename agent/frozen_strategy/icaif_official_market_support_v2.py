#!/usr/bin/env python3
"""ICAIF 2026 release V2 support library; read-only market data and pure blending math.

Subcommands:
  self-test: synthetic tests incl. HOLD, malformed portfolio, out-of-time snapshot
  inspect-csv: audit local legacy Alpaca 15m history / calculate initial Blend80
  probe-alpaca: authenticated, read-only Alpaca IEX bars health test
  prepare-snapshot: read-only Alpaca refresh + seed CSV, atomic snapshot for standalone strategy.

No strategy function, no organizer writer, and no initial Equal80 fallback exist in this module.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import parse, request, error
from zoneinfo import ZoneInfo
from collections import defaultdict

ET = ZoneInfo('America/New_York')
UTC = timezone.utc
SYMBOLS = ('AAPL','MSFT','NVDA','INTC','CRM','JPM','BAC','GS','V','PYPL',
 'LLY','JNJ','UNH','PFE','TMO','AMZN','TSLA','WMT','NKE','KO','CAT','GE',
 'BA','XOM','CVX','GOOGL','META','DIS','T','NEE')
GROUPS = tuple(tuple(range(i,i+5)) for i in range(0,30,5))
DEADLINES = {1:'09:10',2:'10:25',3:'11:25',4:'12:25',5:'13:25',6:'14:25',7:'15:25'}
API = 'https://data.alpaca.markets/v2/stocks/bars'
EXPOSURE = .80


def utc_stamp(s):
    d=datetime.fromisoformat(s.replace('Z','+00:00'))
    if d.tzinfo is None: raise ValueError('Expected timezone-aware UTC datetime')
    return d.astimezone(UTC)


def standardize(v):
    vals=[float(x) for x in v]
    if any(not math.isfinite(x) for x in vals):raise ValueError('Non-finite weight or feature')
    avg=sum(vals)/len(vals)
    std=math.sqrt(sum((x-avg)**2 for x in vals)/len(vals))
    return [min(3.,max(-3.,(x-avg)/max(std,1e-8))) for x in vals]


def blend_weights(daily):
    """Exact Phase30 v2/Phase37 formula, lagged historical CLOSES only."""
    if len(daily)<121:raise ValueError('Need >=121 complete daily closes')
    import numpy as np
    arr=np.array(daily[-121:],dtype=float)
    if arr.shape!=(121,30) or not np.isfinite(arr).all() or (arr<=0).any():
        raise ValueError('Bad historical market shape or prices')
    log=np.log(arr)
    r20=log[-1]-log[-21];r60=log[-1]-log[-61];r120=log[-1]-log[-121]
    returns=log[-20:]-log[-21:-1]
    sig=np.std(returns,axis=0,ddof=1)
    relative=r20-np.repeat(r20.reshape(6,5).mean(axis=1),5)
    features=np.stack((standardize(r20),standardize(r60),standardize(r120),
        standardize(-sig),standardize(relative)),axis=1)
    momentum=.25*features[:,0]+.50*features[:,1]+.25*features[:,2]
    group=np.repeat(momentum.reshape(6,5).mean(axis=1),5)
    score=.65*np.array(standardize(group))+.35*np.array(standardize(momentum))
    x=np.array(standardize(score))
    prop=np.clip(1.+.35*x,.01,None)
    industry=.8*prop/prop.sum()
    result=.5*industry+.5*np.full(30,.8/30)
    validate_targets(dict(zip(SYMBOLS,map(float,result))))
    if abs(result.sum()-.8)>1e-9:raise AssertionError('Blend total equity wrong')
    return dict(zip(SYMBOLS,map(float,result)))


def validate_targets(w):
    if not isinstance(w,dict) or set(w)!=set(SYMBOLS):raise ValueError('Expected exact 30 official symbols')
    if any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=.30 for x in w.values()):
        raise ValueError('Illegal portfolio number or stock weight')
    if sum(w.values())>1.+1e-9:raise ValueError('Total weight exceeds 1')
    if sum(w.values())<.01:raise ValueError('Accidental all-zero liquidation blocked')


def parse_bars(rows,until):
    """Only fully completed regular-session IEX bars; no future record accepted."""
    day_rows=defaultdict(dict)
    for rec in rows:
        s=rec.get('symbol')
        if s not in SYMBOLS:continue
        dt=utc_stamp(rec['timestamp']); local=dt.astimezone(ET)
        minute=local.hour*60+local.minute
        if minute<9*60+30 or minute>15*60+45 or minute%15:continue
        if dt+timedelta(minutes=15)>until:continue
        key=(local.date().isoformat(),local.strftime('%H:%M'),s)
        price=float(rec['close'])
        if not math.isfinite(price) or price<=0:raise ValueError('Invalid price')
        if key in day_rows[local.date().isoformat()]:raise ValueError('Duplicate date-time symbol bar')
        day_rows[local.date().isoformat()][key]=price
    return day_rows


def history_closes(rows,until):
    day_rows=parse_bars(rows,until)
    dates=[];values=[];incomplete=[]
    expected=[f'{h:02d}:{m:02d}' for h in range(9,16) for m in (0,15,30,45) if (h>9 or m>=30)]
    # Add 15:00..15:45; stop at 15:45; 26 bars in total.
    if len(expected)!=26:raise AssertionError('Incorrect 15-minute session grid')
    for date in sorted(day_rows):
        record=day_rows[date]
        if any((date,hm,s) not in record for hm in expected for s in SYMBOLS):
            incomplete.append(date);continue
        vals=[record[(date,'15:45',s)] for s in SYMBOLS]
        dates.append(date);values.append(vals)
    return dates,values,incomplete


def load_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        for x in csv.DictReader(f):
            yield {k: x.get(k) for k in ('symbol','timestamp','close')}


def infer_snapshot(rows,now_utc,kind='OFFLINE_DIAGNOSTIC'):
    dates,daily,missing=history_closes(rows,now_utc)
    if len(dates)<121:raise ValueError(f'Only {len(dates)} complete 30-stock daily closes; require >=121')
    p=blend_weights(daily)
    digest=hashlib.sha256(json.dumps({'dates':dates[-121:],'daily':daily[-121:]},sort_keys=True).encode()).hexdigest()
    return {'schema':1,'purpose':kind,'created_utc':now_utc.isoformat(),
        'last_complete_day_et':dates[-1],'complete_days':len(dates),
        'missing_endofday_sessions':len(missing),'lookback_sha256':digest,
        'symbols':list(SYMBOLS),'target_weights':p,'lookback_days':121,
        'data_feed':'Alpaca IEX 15Min raw daily 15:45 ET closes',
        'strategy':'exact-phase37-core80-blend-hold-initial-only'}


def load_alpaca_headers(env_file=None):
    env=dict(os.environ)
    if env_file is not None:
        path=Path(env_file).expanduser()
        if not path.is_file():raise FileNotFoundError('Private Alpaca environment file missing')
        for raw in path.read_text(encoding='utf-8-sig').splitlines():
            line=raw.strip()
            if not line or line.startswith('#'):continue
            if line.startswith('export '):line=line[7:]
            if '=' in line:
                k,v=line.split('=',1)
                if k.strip() in ('APCA_API_KEY_ID','ALPACA_API_KEY','APCA_API_SECRET_KEY','ALPACA_SECRET_KEY'):
                    env[k.strip()]=v.strip().strip(chr(34)).strip(chr(39))
    key=env.get('APCA_API_KEY_ID') or env.get('ALPACA_API_KEY')
    secret=env.get('APCA_API_SECRET_KEY') or env.get('ALPACA_SECRET_KEY')
    if not key or not secret: raise ValueError('Alpaca credentials not in process environment; DO NOT share them')
    return {'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret,'Accept':'application/json'}


def fetch_iex(start_utc,end_utc,transport=None,env_file=None):
    headers=load_alpaca_headers(env_file) if transport is None else {}
    params={'symbols':','.join(SYMBOLS),'timeframe':'15Min',
     'start':start_utc.isoformat(),'end':end_utc.isoformat(),
     'feed':'iex','adjustment':'raw','sort':'asc','limit':'10000'}
    seen=set();out=[]
    for page in range(50):
        if transport is not None: obj=transport(params)
        else:
            uri=API+'?'+parse.urlencode(params)
            try:
                with request.urlopen(request.Request(uri,headers=headers),timeout=22) as resp:
                    obj=json.loads(resp.read(8_000_000))
            except error.HTTPError as exc:
                raise RuntimeError('Alpaca HTTP '+str(exc.code)+'; no trading action') from None
            except (error.URLError,TimeoutError) as exc:
                raise RuntimeError('Alpaca IEX network unavailable; HOLD') from None
        if not isinstance(obj,dict) or not isinstance(obj.get('bars'),dict):
            raise ValueError('Alpaca bars response schema invalid')
        for symbol,items in obj['bars'].items():
            if symbol not in SYMBOLS:raise ValueError('Unexpected source symbol')
            for x in items:
                out.append({'symbol':symbol,'timestamp':x['t'],'close':x['c']})
        token=obj.get('next_page_token')
        if not token:break
        if token in seen:raise ValueError('Alpaca duplicate pagination token')
        seen.add(token);params['page_token']=token
    else:raise ValueError('Alpaca pagination overflow')
    return out


def safe_atomic_json(target,obj):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=str(target.parent),
         delete=False,prefix='._icaif_',suffix='.json') as file:
        tmp=Path(file.name)
        json.dump(obj,file,indent=2,sort_keys=True);file.flush();os.fsync(file.fileno())
    try:os.replace(tmp,target)
    finally:
        if tmp.exists():tmp.unlink()
