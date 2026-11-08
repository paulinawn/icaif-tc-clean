#!/usr/bin/env python3
# Role: DATA UPDATER - read-only Alpaca history updater and snapshot builder, used by the data fetcher.
"""Read-only Alpaca IEX history updater and v2 candidate snapshot builder.

No order API, organizer API, LLM, SSH, timers, services, or remote writes.
Only local CSV cache and JSON snapshot are written, atomically.
"""
from __future__ import annotations
import argparse,csv,os,tempfile,math
from datetime import datetime,timedelta,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import market_support as v1
import sleeve_engine as engine
UTC=timezone.utc
ET=ZoneInfo('America/New_York')


def consolidate(cache_rows, incoming):
    """Persist full history beyond the 9-day Alpaca API fetch window."""
    store={}
    for x in cache_rows:
        if x.get('symbol') not in engine.SYMBOLS:continue
        t=v1.utc_stamp(x['timestamp']).isoformat()
        key=(x['symbol'],t)
        c=float(x['close'])
        if not math.isfinite(c) or c<=0 or key in store:
            raise ValueError('Invalid/duplicate historical seed bars; no update')
        store[key]=c
    fresh=set()
    for x in incoming:
        if x.get('symbol') not in engine.SYMBOLS:raise ValueError('Unexpected Alpaca symbol')
        t=v1.utc_stamp(x['timestamp']).isoformat()
        key=(x['symbol'],t)
        if key in fresh:raise ValueError('Duplicate bar within API result')
        fresh.add(key)
        c=float(x['close'])
        if not math.isfinite(c) or c<=0:raise ValueError('Bad recent Alpaca close')
        if key in store and abs(store[key]/c-1)>.0025:
            raise ValueError('Price disagrees with stored history: review corporate actions/feed')
        store[key]=c
    return [{'symbol':sym,'timestamp':t,'close':float(c)} for (sym,t),c in sorted(store.items(),key=lambda x:(x[0][1],x[0][0]))]


def atomic_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',newline='',encoding='utf-8',dir=str(path.parent),
                                     prefix='._icaif_',suffix='.csv',delete=False) as out:
        tmp=Path(out.name)
        writer=csv.DictWriter(out,fieldnames=['symbol','timestamp','close'])
        writer.writeheader();writer.writerows(rows);out.flush();os.fsync(out.fileno())
    try:os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def api_has_expected_full_session(api_rows,expected):
    # Authenticated latest session source, not seed-only/unchanged cached prices.
    start=expected+'T09:30';seen=set()
    for r in api_rows:
        if r.get('symbol') not in engine.SYMBOLS:continue
        d=v1.utc_stamp(r['timestamp']).astimezone(ET)
        if d.date().isoformat()==expected and d.strftime('%H:%M')=='15:45':
            seen.add(r['symbol'])
    return len(seen)==30


def refresh(seed,cache,out,env_file=None,now=None,fetch=None):
    now=now or datetime.now(UTC)
    etday=now.astimezone(ET).date().isoformat()
    if etday not in engine.OFFICIAL_DAYS:
        raise ValueError('Read-only prep is restricted to an Official competition date')
    previous=engine.expected_previous_day(etday)
    path=Path(cache)
    if path.exists():older=list(v1.load_csv(path))
    else:
        if not Path(seed).is_file():raise FileNotFoundError('Seed CSV missing')
        older=list(v1.load_csv(seed))
    bars=fetch(now-timedelta(days=9),now) if fetch is not None else v1.fetch_iex(now-timedelta(days=9),now,env_file=env_file)
    if not bars or not api_has_expected_full_session(bars,previous):
        raise ValueError('Fresh Alpaca IEX did not authenticate full prior-session closes (all 30)')
    merged=consolidate(older,bars)
    snap=engine.snapshot_from_bars(merged,now,purpose='LIVE_READONLY_VERIFIED')
    if snap['last_complete_day_et']!=previous:
        raise ValueError('Last complete day does not match expected official prior session')
    # Both local files are write-only on success. No organizer strategy invoked.
    atomic_csv(cache,merged)
    v1.safe_atomic_json(out,snap)
    return snap,len(bars),len(merged)


def selftest():
    # End-to-end: mocked read-only API + persistent cache, no network.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)
        ts=lambda s:datetime.fromisoformat(s).replace(tzinfo=ET).astimezone(UTC).isoformat()
        # Synthetic 121 *complete* daily sessions, with final prior session 2026-10-09.
        days=[];start=__import__('datetime').date(2026,4,24)
        for k in range((__import__('datetime').date(2026,10,9)-start).days+1):
            day=start+timedelta(days=k)
            if day.weekday()<5:days.append(day.isoformat())
        assert days[-1]=='2026-10-09' and len(days)>=121
        hm=[f'{h:02d}:{m:02d}' for h in range(9,16) for m in (0,15,30,45) if (h>9 or m>=30)]
        seed=[]
        for di,day in enumerate(days):
            for hh in hm:
                for j,s in enumerate(engine.SYMBOLS):
                    seed.append(dict(symbol=s,timestamp=ts(day+'T'+hh+':00'),
                                     close=50+j+.005*di+math.sin((di+j)/7.)*.1))
        seedpath=d/'seed.csv';atomic_csv(seedpath,seed)
        recent=[x for x in seed if x['timestamp']>=ts('2026-10-05T00:00:00')]
        now=datetime(2026,10,12,8,55,tzinfo=ET)
        snap,n,new=refresh(seedpath,d/'rolling.csv',d/'snap.json',now=now,fetch=lambda a,b:recent)
        assert n>0 and new==len(seed)
        assert snap['last_complete_day_et']=='2026-10-09' and snap['complete_days']==len(days)
        assert len(snap['selected_five'])==5
        assert sum(snap['target_weights'].values())==.8 or abs(sum(snap['target_weights'].values())-.8)<1e-10
        q=engine.evaluate(engine.demo_obs(),snap,engine.sample_state(),now)
        assert q['action']=='PRIMARY_TARGET'
        # Missing latest authenticated close, even with historical seed available => failure.
        incomplete=[x for x in recent if not (v1.utc_stamp(x['timestamp']).astimezone(ET).strftime('%Y-%m-%d %H:%M')=='2026-10-09 15:45' and x['symbol']=='AAPL')]
        try:refresh(seedpath,d/'rolling.csv',d/'snap.json',now=now,fetch=lambda a,b:incomplete)
        except ValueError:pass
        else:raise AssertionError('Accepted incomplete authenticated market closes')
    print('RELEASE_V2_REFRESHER_SELF_TEST_PASS')
    print('HISTORICAL_CACHE_PERSISTS_FRESH_30_STOCK_PRIOR_SESSION_AUTHENTICATED_NO_TRADES')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['self-test','refresh-readonly'])
    p.add_argument('--seed-csv',type=Path)
    p.add_argument('--bars-cache',type=Path)
    p.add_argument('--snapshot-out',type=Path)
    p.add_argument('--env-file',type=Path)
    p.add_argument('--ack-readonly-api-requests',action='store_true')
    a=p.parse_args()
    if a.action=='self-test':return selftest()
    if not a.ack_readonly_api_requests:p.error('Explicit read-only Alpaca request acknowledgement required')
    if not a.seed_csv or not a.bars_cache or not a.snapshot_out:
        p.error('Provide --seed-csv, --bars-cache and --snapshot-out')
    snap,got,total=refresh(a.seed_csv,a.bars_cache,a.snapshot_out,a.env_file)
    print('RELEASE_V2_ALPACA_READONLY_REFRESH_OK','RECORDS',got,'CACHE_BARS',total)
    print('PRIOR_SESSION',snap['last_complete_day_et'],'COMPLETE_DAYS',snap['complete_days'])
    print('SLEEVE_SELECTED_FIVE',','.join(snap['selected_five']))
    print('TARGET_EQUITY',sum(snap['target_weights'].values()))
    print('LOCAL_SNAPSHOT',a.snapshot_out,'LOCAL_ROLLING_HISTORY',a.bars_cache)
    print('NO_ORGANIZER_NO_SUBMISSION_NO_TRADES')

if __name__=='__main__':main()
