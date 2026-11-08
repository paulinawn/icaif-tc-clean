#!/usr/bin/env python3
# Role: DATA UPDATER - read-only Alpaca history updater and snapshot builder, used by the data fetcher.
"""Read-only Alpaca IEX history updater and v2 candidate snapshot builder.

No order API, organizer API, LLM, SSH, timers, services, or remote writes.
Only local CSV cache and JSON snapshot are written, atomically.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import market_support as v1
import sleeve_engine as engine

UTC = timezone.utc
ET = ZoneInfo('America/New_York')


def consolidate(cache_rows, incoming):
    """Merge cached and freshly fetched bars into canonical history.

    Args:
        cache_rows: iterable of cached bar mappings with 'symbol',
            'timestamp', and 'close' fields; symbols outside the official
            universe are ignored.
        incoming: iterable of Alpaca bar mappings in the same shape.

    Returns:
        list[dict]: canonical bars sorted by timestamp, then symbol, each
            containing 'symbol', 'timestamp', and float 'close' fields.

    Raises:
        ValueError: for invalid or duplicate cached bars, unexpected or
            duplicate incoming symbols/bars, invalid incoming closes, or a
            material disagreement with cached history.
    """
    store = {}
    for row in cache_rows:
        symbol = row.get('symbol')
        if symbol not in engine.SYMBOLS:
            continue
        timestamp = v1.utc_stamp(row['timestamp']).isoformat()
        key = (row['symbol'], timestamp)
        close = float(row['close'])
        if not math.isfinite(close) or close <= 0 or key in store:
            raise ValueError('Invalid/duplicate historical seed bars; no update')
        store[key] = close

    fresh = set()
    for row in incoming:
        symbol = row.get('symbol')
        if symbol not in engine.SYMBOLS:
            raise ValueError('Unexpected Alpaca symbol')
        timestamp = v1.utc_stamp(row['timestamp']).isoformat()
        key = (row['symbol'], timestamp)
        if key in fresh:
            raise ValueError('Duplicate bar within API result')
        fresh.add(key)
        close = float(row['close'])
        if not math.isfinite(close) or close <= 0:
            raise ValueError('Bad recent Alpaca close')
        if key in store and abs(store[key] / close - 1) > .0025:
            raise ValueError(
                'Price disagrees with stored history: '
                'review corporate actions/feed'
            )
        store[key] = close

    return [
        {'symbol': symbol, 'timestamp': timestamp, 'close': float(close)}
        for (symbol, timestamp), close in sorted(
            store.items(), key=lambda item: (item[0][1], item[0][0])
        )
    ]


def atomic_csv(path, rows):
    """Write bar rows to a CSV file through a same-directory temporary file.

    Args:
        path: destination filesystem path.
        rows: iterable of mappings containing 'symbol', 'timestamp', and
            'close' fields.

    Returns:
        None.

    Raises:
        OSError: if directory creation, temporary-file handling, or the
            atomic replacement fails.
        csv.Error: if CSV writing fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        'w', newline='', encoding='utf-8', dir=str(path.parent),
        prefix='._icaif_', suffix='.csv', delete=False
    ) as out:
        tmp = Path(out.name)
        writer = csv.DictWriter(out, fieldnames=['symbol', 'timestamp', 'close'])
        writer.writeheader()
        writer.writerows(rows)
        out.flush()
        os.fsync(out.fileno())
    try:
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def api_has_expected_full_session(api_rows, expected):
    """Check for all 30 authenticated 15:45 closes on the prior session.

    Args:
        api_rows: iterable of Alpaca bar mappings with 'symbol',
            'timestamp', and related fields.
        expected: ISO date string for the expected ET trading session.

    Returns:
        bool: True only when every official symbol has a 15:45 ET bar on
            the expected date.

    Raises:
        ValueError: if a bar timestamp is invalid or lacks timezone data.
        KeyError: if a candidate bar lacks a 'timestamp' field.
    """
    # Authenticated latest session source, not seed-only/unchanged cached prices.
    session_start_marker = expected + 'T09:30'
    seen_symbols = set()
    for row in api_rows:
        symbol = row.get('symbol')
        if symbol not in engine.SYMBOLS:
            continue
        local_time = v1.utc_stamp(row['timestamp']).astimezone(ET)
        if local_time.date().isoformat() != expected:
            continue
        if local_time.strftime('%H:%M') == '15:45':
            seen_symbols.add(row['symbol'])
    return len(seen_symbols) == 30


def refresh(seed, cache, out, env_file=None, now=None, fetch=None):
    """Refresh the rolling bar cache and write a verified price snapshot.

    Args:
        seed: fallback CSV path used only when the rolling cache is absent.
        cache: rolling CSV cache path; existing contents are loaded and
            merged with fresh Alpaca bars.
        out: destination path for the JSON snapshot.
        env_file: optional credential environment file for the live Alpaca
            GET when fetch is not supplied.
        now: optional timezone-aware current instant; defaults to now in UTC.
        fetch: optional callable(start_utc, end_utc) returning Alpaca bars,
            used instead of the live GET.

    Returns:
        tuple: (snapshot dict, fetched bar count, merged cache bar count).

    Raises:
        ValueError: if the current ET day is not official, fresh API bars do
            not authenticate the complete prior session, or snapshot coverage
            disagrees with the expected prior session.
        FileNotFoundError: if the cache is absent and the seed CSV is missing.
        OSError: if local cache or snapshot writes fail.
    """
    current_time = now or datetime.now(UTC)
    current_et_day = current_time.astimezone(ET).date().isoformat()
    if current_et_day not in engine.OFFICIAL_DAYS:
        raise ValueError('Read-only prep is restricted to an Official competition date')
    previous_day = engine.expected_previous_day(current_et_day)
    cache_path = Path(cache)
    if cache_path.exists():
        cached_rows = list(v1.load_csv(cache_path))
    else:
        if not Path(seed).is_file():
            raise FileNotFoundError('Seed CSV missing')
        cached_rows = list(v1.load_csv(seed))
    if fetch is not None:
        bars = fetch(current_time - timedelta(days=9), current_time)
    else:
        bars = v1.fetch_iex(
            current_time - timedelta(days=9), current_time, env_file=env_file
        )
    if not bars or not api_has_expected_full_session(bars, previous_day):
        raise ValueError(
            'Fresh Alpaca IEX did not authenticate full prior-session '
            'closes (all 30)'
        )
    merged_rows = consolidate(cached_rows, bars)
    snapshot = engine.snapshot_from_bars(
        merged_rows, current_time, purpose='LIVE_READONLY_VERIFIED'
    )
    if snapshot['last_complete_day_et'] != previous_day:
        raise ValueError(
            'Last complete day does not match expected official prior session'
        )
    # Both local files are write-only on success. No organizer strategy invoked.
    atomic_csv(cache, merged_rows)
    v1.safe_atomic_json(out, snapshot)
    return snapshot, len(bars), len(merged_rows)


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
    """Run the built-in self-test or a confirmed read-only history refresh.

    Returns:
        The self-test return value for 'self-test'; otherwise None after a
        successful refresh or an argparse error exit.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['self-test', 'refresh-readonly'])
    parser.add_argument('--seed-csv', type=Path)
    parser.add_argument('--bars-cache', type=Path)
    parser.add_argument('--snapshot-out', type=Path)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--ack-readonly-api-requests', action='store_true')
    args = parser.parse_args()

    if args.action == 'self-test':
        return selftest()
    if not args.ack_readonly_api_requests:
        parser.error('Explicit read-only Alpaca request acknowledgement required')
    if not args.seed_csv or not args.bars_cache or not args.snapshot_out:
        parser.error('Provide --seed-csv, --bars-cache and --snapshot-out')

    snapshot, fetched_count, merged_count = refresh(
        args.seed_csv, args.bars_cache, args.snapshot_out, args.env_file
    )
    print(
        'RELEASE_V2_ALPACA_READONLY_REFRESH_OK',
        'RECORDS',
        fetched_count,
        'CACHE_BARS',
        merged_count,
    )
    print(
        'PRIOR_SESSION',
        snapshot['last_complete_day_et'],
        'COMPLETE_DAYS',
        snapshot['complete_days'],
    )
    print('SLEEVE_SELECTED_FIVE', ','.join(snapshot['selected_five']))
    print('TARGET_EQUITY', sum(snapshot['target_weights'].values()))
    print('LOCAL_SNAPSHOT', args.snapshot_out, 'LOCAL_ROLLING_HISTORY', args.bars_cache)
    print('NO_ORGANIZER_NO_SUBMISSION_NO_TRADES')


if __name__ == '__main__':
    main()
