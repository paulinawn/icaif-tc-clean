#!/usr/bin/env python3
# Role: DATA FETCHER - downloads and checks Alpaca 15-minute bars and builds the 121-day price history.
"""ICAIF V28 unarmed, Alpaca-only IEX raw 15m history bootstrap and refresh.

Only network calls: Alpaca market-data GET. Only writes: V28 staging data.
NO organizer, auto_submit, broker orders, services, timers or Git operations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time as time_module
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
UTC = timezone.utc
BASE = Path(__file__).resolve().parent
DATA = Path('/opt/icaif2026/official_clean_bootstrap_v28_unarmed/data')  # preserve production cache
LEGACY_CODE = Path('/opt/icaif2026/official_staging_v23_unarmed/code')
# Windows standalone self-test may use the extracted 2026 V23 directory.
if not LEGACY_CODE.exists():
    LEGACY_CODE = Path(os.environ.get('ICAIF_V28_V23_CODE', str(BASE)))
sys.path.insert(0, str(LEGACY_CODE))

import icaif_official_market_support_v2 as market
import icaif_official_sleeve_engine_v23 as engine
import icaif_official_sleeve_refresher_v23 as refresher

PRICE_COLUMNS = ('symbol', 'timestamp', 'close')
CACHE = DATA / 'alpaca_iex_raw_15m.csv'
SNAP = DATA / 'latest_readonly_snapshot.json'
MANIFEST = DATA / 'cache_provenance.json'
EXPECTED_FEED = 'iex'
EXPECTED_ADJUSTMENT = 'raw'
BOOTSTRAP_CALENDAR_DAYS = 250
CHUNK_CALENDAR_DAYS = 14
CHUNK_PAUSE_SECONDS = 6
MIN_COMPLETE_DAYS = 121
MAX_RECORDS = 260_000


def require_isolated_data_path() -> None:
    # Prevent accidentally writing into old, root-owned V27 seed or project .env.
    if BASE.parent.name != 'code' and BASE.name != 'code':
        raise ValueError('V28 must be run from its isolated code directory')
    if DATA.resolve().is_relative_to(Path('/opt/icaif2026/official_staging_v23_unarmed').resolve()):
        raise ValueError('Cannot write in the legacy V27 staging area')
    if DATA.is_symlink() or BASE.is_symlink():
        raise ValueError('Staging directory symlink prohibited')


def closing_date(now: datetime) -> str:
    if now.tzinfo is None:
        raise ValueError('Clock must include timezone')
    today = now.astimezone(ET).date().isoformat()
    if today not in engine.OFFICIAL_DAYS:
        raise ValueError('Official-day data check not applicable: ' + today)
    return engine.expected_previous_day(today)


def fetch_window(now: datetime, bootstrap: bool) -> tuple[datetime, datetime]:
    last = date.fromisoformat(closing_date(now))
    start_day = last - timedelta(days=BOOTSTRAP_CALENDAR_DAYS if bootstrap else 9)
    # Stop at midnight ET immediately after *last complete session*. Do not
    # include the current trading day even when fetched after market opening.
    start = datetime.combine(start_day, time.min, ET).astimezone(UTC)
    end = datetime.combine(last + timedelta(days=1), time.min, ET).astimezone(UTC)
    assert start < end <= now.astimezone(UTC)
    return start, end


def authenticated_fetch_chunks(start: datetime,end: datetime,fetch=None,pause=None):
    """Bound Alpaca pagination and rate consumption for a 250-day backfill.

    The underlying V2.3 adapter has a HARD 50-page bound per call; the 14-day
    requests are independently paginated. ET-midnight half-open windows avoid
    duplicated regular-session bars even across US daylight saving changes.
    """
    if not start < end:raise ValueError('Invalid Alpaca history window')
    get=fetch or (lambda a,b:market.fetch_iex(a,b,env_file=Path('/opt/icaif2026/.env')))
    sleep=pause or time_module.sleep
    cursor=start
    combined=[]
    chunks=0
    while cursor<end:
        boundary_local=cursor.astimezone(ET).date()+timedelta(days=CHUNK_CALENDAR_DAYS)
        next_at=min(datetime.combine(boundary_local,time.min,ET).astimezone(UTC),end)
        if next_at<=cursor:raise ValueError('No forward progress in API chunking')
        if chunks and fetch is None:sleep(CHUNK_PAUSE_SECONDS)
        page=get(cursor,next_at)
        if not isinstance(page,list):raise ValueError('Market GET did not return list')
        combined.extend(page)
        if len(combined)>MAX_RECORDS:raise ValueError('Too many Alpaca historical records')
        chunks+=1
        cursor=next_at
    return combined,chunks


def normalize_bars(raw):
    result = {}
    for rec in raw:
        sym = rec.get('symbol')
        if sym not in engine.SYMBOLS:
            raise ValueError('Unexpected symbol in Alpaca result')
        stamp = market.utc_stamp(rec['timestamp']).isoformat()
        price = float(rec['close'])
        if not (math.isfinite(price) and price > 0):
            raise ValueError('Nonpositive/nonfinite Alpaca price')
        key = (sym, stamp)
        if key in result:
            raise ValueError('Duplicate Alpaca symbol/timestamp bar')
        result[key] = price
        if len(result) > MAX_RECORDS:
            raise ValueError('Excessive historical record count')
    return result


def rows_from_map(m):
    return [{'symbol':s,'timestamp':t,'close':v} for (s,t),v in sorted(m.items(),key=lambda x:(x[0][1],x[0][0]))]


def coverage(rows,now):
    valid_dates, daily, incomplete = market.history_closes(rows,now)
    if len(valid_dates)<MIN_COMPLETE_DAYS:
        raise ValueError(f'Insufficient complete 30-stock sessions {len(valid_dates)}/{MIN_COMPLETE_DAYS}')
    previous=closing_date(now)
    if valid_dates[-1] != previous:
        raise ValueError('Latest completed market session not present: ' + previous)
    d = date.fromisoformat(previous)
    today_dates={s for s in valid_dates if s==previous}
    if len(today_dates)!=1 or len(daily[-1])!=30:
        raise ValueError('Missing prior session prices')
    return valid_dates, incomplete


def hash_rows(rows):
    dig=hashlib.sha256()
    for item in rows:
        dig.update(f"{item['symbol']},{item['timestamp']},{float(item['close']):.15g}\n".encode('ascii'))
    return dig.hexdigest()


def make_snapshot(rows,now):
    valid_dates,incomplete=coverage(rows,now)
    snap=engine.snapshot_from_bars(rows,now,purpose='LIVE_READONLY_VERIFIED')
    engine.validate(snap['target_weights'],expect_equity=.80)
    if snap.get('last_complete_day_et')!=closing_date(now):
        raise ValueError('Snapshot day differs from API session')
    if snap.get('complete_days')!=len(valid_dates):
        raise ValueError('Engine coverage disagrees with validator')
    # The prior Oct-12 synthetic R1 assertion is invalid for a genuine Oct-19
    # or Oct-26 snapshot. This function is only a data/snapshot validator.
    # engine.snapshot_from_bars and engine.validate above remain unchanged.
    return snap,valid_dates,incomplete


def verified_old_cache():
    if not CACHE.is_file() or not MANIFEST.is_file():
        raise ValueError('V28 bootstrap missing; use bootstrap once')
    previous=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if previous.get('schema')!=28 or previous.get('feed')!='iex' or previous.get('adjustment')!='raw':
        raise ValueError('Wrong cache schema / feed / adjustment')
    rows=list(market.load_csv(CACHE))
    normalized=normalize_bars(rows)
    canonical=rows_from_map(normalized)
    if previous.get('rows_sha256')!=hash_rows(canonical):
        raise ValueError('Cache hash mismatch: refusing refresh')
    if previous.get('row_count')!=len(canonical):
        raise ValueError('Cache size mismatch')
    return canonical


def prepare(now, mode, fetch=None):
    if now.tzinfo is None:raise ValueError('Timezone-aware clock required')
    start,end=fetch_window(now,mode=='bootstrap')
    raw,chunks=authenticated_fetch_chunks(start,end,fetch=fetch)
    incoming=normalize_bars(raw)
    if not incoming:raise ValueError('Alpaca returned no IEX bars')
    incoming_rows=rows_from_map(incoming)
    if not refresher.api_has_expected_full_session(incoming_rows,closing_date(now)):
        raise ValueError('Authenticated Alpaca GET lacks 30 prior-session closing prices')
    if mode=='bootstrap':
        if any(p.exists() for p in (CACHE,SNAP,MANIFEST)):
            raise ValueError('V28 cache already exists: refusing bootstrap overwrite')
        combined=incoming
        legacy_used=False
    elif mode=='refresh':
        old=verified_old_cache()
        combined=normalize_bars(refresher.consolidate(old,incoming_rows))
        legacy_used=False
    else:raise ValueError('Unsupported operation')
    rows=rows_from_map(combined)
    snap,days,incomplete=make_snapshot(rows,now)
    record={
        'schema':28,'mode':mode,'feed':EXPECTED_FEED,'adjustment':EXPECTED_ADJUSTMENT,
        'timeframe':'15Min','data_origin':('ALPACA_API_ONLY_NO_IMPORTED_SEED' if fetch is None else 'OFFLINE_INJECTED_TEST'),
        'last_complete_day_et':closing_date(now),
        'fetched_from_utc':start.isoformat(),'fetched_to_utc':end.isoformat(),
        'fetched_at_utc':now.astimezone(UTC).isoformat(),
        'incoming_bars':len(incoming),'get_windows':chunks,'row_count':len(rows),
        'complete_days':len(days),'incomplete_days':len(incomplete),
        'rows_sha256':hash_rows(rows),'snapshot_lookback_sha256':snap['lookback_sha256'],
        'v23_strategy':snap['strategy'],
    }
    return rows,snap,record


def persist(rows,snapshot,meta):
    require_isolated_data_path()
    DATA.mkdir(mode=0o700,parents=True,exist_ok=True)
    # Individual atomic writes; provenance is published LAST, and verifies data.
    refresher.atomic_csv(CACHE,rows)
    market.safe_atomic_json(SNAP,snapshot)
    market.safe_atomic_json(MANIFEST,meta)


def run(mode, now=None,fetch=None,write=True):
    now=now or datetime.now(UTC)
    require_isolated_data_path()
    rows,snap,record=prepare(now,mode,fetch=fetch)
    if write:persist(rows,snap,record)
    print('V28_CLEAN_ALPACA_'+mode.upper()+'_PASS')
    print('V28_SOURCE', 'ALPACA_AUTHENTICATED_IEX_15MIN_RAW_NO_SEED' if fetch is None else 'SYNTHETIC_OFFLINE_IEX_RAW_TEST')
    print('V28_PRIOR_COMPLETE_SESSION',record['last_complete_day_et'])
    print('V28_FETCHED_BARS',record['incoming_bars'])
    print('V28_BOUNDED_GET_WINDOWS',record['get_windows'])
    print('V28_COMPLETE_DAYS',record['complete_days'])
    print('V28_STORED_BARS',record['row_count'])
    print('V28_INCOMPLETE_DAYS',record['incomplete_days'])
    print('V28_DATA_SHA256',record['rows_sha256'])
    print('V28_LOOKBACK_SHA256',record['snapshot_lookback_sha256'])
    print('V28_SELECTED_FIVE',','.join(snap['selected_five']))
    print('V28_TARGET_EQUITY',format(sum(snap['target_weights'].values()),'.12f'))
    print('V393_SNAPSHOT_TARGET_SHAPE_VALIDATED_NO_SYNTHETIC_R1')
    print('V28_DATA_LOCATION',DATA)
    print('V28_NO_ORGANIZER_NO_POST_NO_SCHEDULER_NO_TRADES')
    return record


def test():
    """Exercise V2.3 portfolio computation AND on-disk bootstrap/refresh paths."""
    import tempfile
    now=datetime(2026,10,12,8,55,tzinfo=ET)
    end=date(2026,10,9)
    days=[]
    for i in range(205):
        d=end-timedelta(days=i)
        if d.weekday()<5:days.append(d)
    days.sort()
    assert len(days)>=121
    bar_times=[f'{h:02d}:{m:02d}' for h in range(9,16) for m in (0,15,30,45) if h>9 or m>=30]
    rows=[]
    for di,d in enumerate(days):
        for hhmm in bar_times:
            stamp=datetime.fromisoformat(d.isoformat()+'T'+hhmm+':00').replace(tzinfo=ET).astimezone(UTC).isoformat()
            for j,s in enumerate(engine.SYMBOLS):
                rows.append({'symbol':s,'timestamp':stamp,'close':float(60+j+di*.027+j*.0003*di)})
    assert len(rows)==len(days)*26*30
    last_day=days[-1].isoformat()
    latest=[r for r in rows if market.utc_stamp(r['timestamp']).astimezone(ET).date().isoformat()==last_day]
    assert len(latest)==780
    with tempfile.TemporaryDirectory() as temp:
        fake=Path(temp);(fake/'code').mkdir();(fake/'data').mkdir()
        global DATA,BASE,CACHE,SNAP,MANIFEST
        prior=(DATA,BASE,CACHE,SNAP,MANIFEST)
        try:
            DATA=fake/'data';BASE=fake/'code'
            CACHE=DATA/'alpaca_iex_raw_15m.csv'
            SNAP=DATA/'latest_readonly_snapshot.json'
            MANIFEST=DATA/'cache_provenance.json'
            # Incomplete authenticated last session must fail without any writes.
            missing=rows[:-1]
            def windowed(data):
                return lambda start,end:[r for r in data if start<=market.utc_stamp(r['timestamp'])<end]
            try:run('bootstrap',now=now,fetch=windowed(missing))
            except ValueError:pass
            else:raise AssertionError('Incomplete bootstrap accepted')
            assert not CACHE.exists() and not SNAP.exists()
            a=run('bootstrap',now=now,fetch=windowed(rows))
            assert a['complete_days']>=121 and a['last_complete_day_et']=='2026-10-09'
            assert a['get_windows']>=15
            cache_bytes=CACHE.read_bytes()
            # Existing clean cache is never overwritten by bootstrap.
            try:run('bootstrap',now=now,fetch=windowed(rows))
            except ValueError:pass
            else:raise AssertionError('Repeated bootstrap overwrote cache')
            b=run('refresh',now=now,fetch=windowed(latest))
            assert b['row_count']==a['row_count'] and b['complete_days']==a['complete_days']
            original_manifest=MANIFEST.read_bytes()
            # A real price revision is investigated, not silently inserted.
            modified=[dict(r) for r in latest]
            for r in modified:
                if r['symbol']=='JPM':r['close']+=1.65
            try:run('refresh',now=now,fetch=windowed(modified))
            except ValueError as e:
                assert 'Price disagrees' in str(e)
            else:raise AssertionError('Accepted material historical revision')
            assert MANIFEST.read_bytes()==original_manifest
            assert CACHE.read_bytes()==cache_bytes
            # Hash-detected file corruption must stop refresh without changing it.
            with CACHE.open('a',encoding='utf-8') as f:f.write('AAPL,2026-10-09T19:45:00+00:00,123\n')
            try:run('refresh',now=now,fetch=windowed(latest))
            except ValueError:pass
            else:raise AssertionError('Corrupt cache passed integrity check')
        finally:
            DATA,BASE,CACHE,SNAP,MANIFEST=prior
    print('V28_OFFLINE_SELF_TEST_PASS')
    print('BOOTSTRAP_REFRESH_INCOMPLETE_LAST_DAY_REVISION_CACHE_HASH_GATES_PASS')
    print('NO_NETWORK_NO_ORGANIZER_NO_SUBMISSION_NO_PERSISTENT_WRITES')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('self-test','bootstrap','refresh'))
    p.add_argument('--ack-readonly-alpaca-get',action='store_true')
    opt=p.parse_args()
    if opt.action=='self-test':return test()
    if not opt.ack_readonly_alpaca_get:p.error('Require --ack-readonly-alpaca-get')
    try:run(opt.action)
    except (ValueError,KeyError,FileNotFoundError,RuntimeError,TimeoutError,OSError) as e:
        print('V28_FAIL_CLOSED',type(e).__name__,str(e)[:190])
        raise SystemExit(2)

if __name__=='__main__':main()
