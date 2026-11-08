#!/usr/bin/env python3
# Role: DATA FETCHER - downloads and checks Alpaca 15-minute bars and builds the 121-day price history.
"""ICAIF V28 unarmed, Alpaca-only IEX raw 15m history bootstrap and refresh.

Only network calls: Alpaca market-data GET. Only writes: V28 staging data.
NO organizer, auto_submit, broker orders, services, timers or Git operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time as time_module
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
UTC = timezone.utc
BASE = Path(__file__).resolve().parent
# Preserve the production cache path.
DATA = Path('/opt/icaif2026/official_clean_bootstrap_v28_unarmed/data')
LEGACY_CODE = Path('/opt/icaif2026/official_staging_v23_unarmed/code')
# Windows standalone self-test may use the extracted 2026 V23 directory.
if not LEGACY_CODE.exists():
    LEGACY_CODE = Path(os.environ.get('ICAIF_V28_V23_CODE', str(BASE)))
sys.path.insert(0, str(LEGACY_CODE))

import market_support as market
import sleeve_engine as engine
import history_refresher as refresher

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
    """Return the expected prior completed ET session for an official day.

    Args:
        now: timezone-aware current instant.

    Returns:
        str: ISO date of the previous official weekday session.

    Raises:
        ValueError: if now has no timezone or its ET date is not an official
            competition day.
    """
    if now.tzinfo is None:
        raise ValueError('Clock must include timezone')
    today = now.astimezone(ET).date().isoformat()
    if today not in engine.OFFICIAL_DAYS:
        raise ValueError('Official-day data check not applicable: ' + today)
    return engine.expected_previous_day(today)


def fetch_window(now: datetime, bootstrap: bool) -> tuple[datetime, datetime]:
    """Compute the half-open UTC window for a bootstrap or refresh fetch.

    Args:
        now: timezone-aware current instant.
        bootstrap: True for the 250-calendar-day bootstrap window, False for
            the 9-calendar-day refresh window.

    Returns:
        tuple[datetime, datetime]: UTC start and exclusive end instants,
            ending at ET midnight after the last complete session.

    Raises:
        ValueError: propagated from closing_date for invalid clocks or
            non-official ET dates.
        AssertionError: if the computed window is empty or extends past now.
    """
    last = date.fromisoformat(closing_date(now))
    start_day = last - timedelta(days=BOOTSTRAP_CALENDAR_DAYS if bootstrap else 9)
    # Stop at midnight ET immediately after *last complete session*. Do not
    # include the current trading day even when fetched after market opening.
    start = datetime.combine(start_day, time.min, ET).astimezone(UTC)
    end = datetime.combine(last + timedelta(days=1), time.min, ET).astimezone(UTC)
    assert start < end <= now.astimezone(UTC)
    return start, end


def authenticated_fetch_chunks(start: datetime, end: datetime, fetch=None, pause=None):
    """Fetch a long history window through bounded, paced GET chunks.

    The underlying V2.3 adapter has a HARD 50-page bound per call; the 14-day
    requests are independently paginated. ET-midnight half-open windows avoid
    duplicated regular-session bars even across US daylight saving changes.

    Args:
        start: inclusive UTC window start.
        end: exclusive UTC window end.
        fetch: optional callable(start_utc, end_utc) returning bar rows;
            when omitted, market.fetch_iex reads the production .env file.
        pause: optional callable(seconds) used between live GET chunks.

    Returns:
        tuple: (combined bar rows, number of GET chunks attempted).

    Raises:
        ValueError: for an invalid window, non-list GET result, stalled
            chunking, or excessive record count.
        OSError: propagated from the credential file or HTTP layer.
        RuntimeError: propagated from the market data adapter.
    """
    if not start < end:
        raise ValueError('Invalid Alpaca history window')
    fetch_bars = fetch or (
        lambda window_start, window_end: market.fetch_iex(
            window_start, window_end, env_file=Path('/opt/icaif2026/.env')
        )
    )
    sleep = pause or time_module.sleep
    cursor = start
    combined = []
    chunks = 0
    while cursor < end:
        boundary_date = (
            cursor.astimezone(ET).date()
            + timedelta(days=CHUNK_CALENDAR_DAYS)
        )
        next_at = min(
            datetime.combine(boundary_date, time.min, ET).astimezone(UTC), end
        )
        if next_at <= cursor:
            raise ValueError('No forward progress in API chunking')
        if chunks and fetch is None:
            sleep(CHUNK_PAUSE_SECONDS)
        page = fetch_bars(cursor, next_at)
        if not isinstance(page, list):
            raise ValueError('Market GET did not return list')
        combined.extend(page)
        if len(combined) > MAX_RECORDS:
            raise ValueError('Too many Alpaca historical records')
        chunks += 1
        cursor = next_at
    return combined, chunks


def normalize_bars(raw):
    """Validate raw Alpaca bars and index them by symbol and timestamp.

    Args:
        raw: iterable of bar mappings with 'symbol', 'timestamp', and 'close'.

    Returns:
        dict: {(symbol, ISO timestamp): positive finite float close}.

    Raises:
        ValueError: for an unexpected symbol, invalid timestamp/close,
            duplicate symbol-timestamp pair, or excessive record count.
        KeyError: if a bar lacks 'timestamp' or 'close'.
    """
    result = {}
    for record in raw:
        symbol = record.get('symbol')
        if symbol not in engine.SYMBOLS:
            raise ValueError('Unexpected symbol in Alpaca result')
        stamp = market.utc_stamp(record['timestamp']).isoformat()
        price = float(record['close'])
        if not (math.isfinite(price) and price > 0):
            raise ValueError('Nonpositive/nonfinite Alpaca price')
        key = (symbol, stamp)
        if key in result:
            raise ValueError('Duplicate Alpaca symbol/timestamp bar')
        result[key] = price
        if len(result) > MAX_RECORDS:
            raise ValueError('Excessive historical record count')
    return result


def rows_from_map(m):
    """Convert a normalized bar map to canonically sorted row dicts.

    Args:
        m: mapping of (symbol, timestamp) keys to close prices.

    Returns:
        list[dict]: rows sorted by timestamp then symbol, with fields in
            'symbol', 'timestamp', 'close' order.
    """
    return [
        {'symbol': symbol, 'timestamp': timestamp, 'close': price}
        for (symbol, timestamp), price in sorted(
            m.items(), key=lambda item: (item[0][1], item[0][0])
        )
    ]


def coverage(rows, now):
    """Validate that rows cover enough complete sessions through prior day.

    Args:
        rows: iterable of raw 15-minute bar mappings.
        now: timezone-aware current instant.

    Returns:
        tuple: (sorted complete ET session dates, incomplete session dates).

    Raises:
        ValueError: if fewer than MIN_COMPLETE_DAYS sessions are complete,
            the latest complete session is not the expected prior session,
            or the prior session is incomplete.
    """
    valid_dates, daily_closes, incomplete_dates = market.history_closes(rows, now)
    if len(valid_dates) < MIN_COMPLETE_DAYS:
        raise ValueError(
            'Insufficient complete 30-stock sessions '
            f'{len(valid_dates)}/{MIN_COMPLETE_DAYS}'
        )
    previous = closing_date(now)
    if valid_dates[-1] != previous:
        raise ValueError('Latest completed market session not present: ' + previous)
    previous_date = date.fromisoformat(previous)
    matching_dates = {
        session_date for session_date in valid_dates if session_date == previous
    }
    if len(matching_dates) != 1 or len(daily_closes[-1]) != 30:
        raise ValueError('Missing prior session prices')
    return valid_dates, incomplete_dates


def hash_rows(rows):
    """Hash canonical bar rows with the V28 provenance format.

    Args:
        rows: iterable of mappings with 'symbol', 'timestamp', and 'close'.

    Returns:
        str: lowercase SHA-256 hex digest of the formatted rows.

    Raises:
        KeyError: if a row lacks a required field.
        ValueError: if a close value cannot be converted to float.
        UnicodeEncodeError: if a field cannot be ASCII-encoded.
    """
    digest = hashlib.sha256()
    for row in rows:
        line = (
            f"{row['symbol']},{row['timestamp']},"
            f"{float(row['close']):.15g}\n"
        )
        digest.update(line.encode('ascii'))
    return digest.hexdigest()


def make_snapshot(rows, now):
    """Build and cross-check the engine snapshot for validated rows.

    Args:
        rows: canonical bar rows accepted by coverage and the engine.
        now: timezone-aware current instant.

    Returns:
        tuple: (snapshot dict, complete session dates, incomplete dates).

    Raises:
        ValueError: if coverage, snapshot day, engine weights, or engine
            coverage checks fail.
    """
    valid_dates, incomplete_dates = coverage(rows, now)
    snap = engine.snapshot_from_bars(rows,now,purpose='LIVE_READONLY_VERIFIED')
    engine.validate(snap['target_weights'], expect_equity=.80)
    if snap.get('last_complete_day_et') != closing_date(now):
        raise ValueError('Snapshot day differs from API session')
    if snap.get('complete_days') != len(valid_dates):
        raise ValueError('Engine coverage disagrees with validator')
    # The prior Oct-12 synthetic R1 assertion is invalid for a genuine Oct-19
    # or Oct-26 snapshot. This function is only a data/snapshot validator.
    # engine.snapshot_from_bars and engine.validate above remain unchanged.
    return snap, valid_dates, incomplete_dates


def verified_old_cache():
    """Load the existing V28 cache only after provenance verification.

    Returns:
        list[dict]: canonical cached bar rows when the manifest schema,
            feed, adjustment, row hash, and row count all match.

    Raises:
        ValueError: if cache or manifest is missing, provenance metadata is
            wrong, the cache hash differs, or the row count differs.
        OSError: if the cache or manifest cannot be read.
        json.JSONDecodeError: if the manifest is not valid JSON.
    """
    if not CACHE.is_file() or not MANIFEST.is_file():
        raise ValueError('V28 bootstrap missing; use bootstrap once')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if (
        manifest.get('schema') != 28
        or manifest.get('feed') != 'iex'
        or manifest.get('adjustment') != 'raw'
    ):
        raise ValueError('Wrong cache schema / feed / adjustment')
    rows = list(market.load_csv(CACHE))
    normalized_bars = normalize_bars(rows)
    canonical_rows = rows_from_map(normalized_bars)
    if manifest.get('rows_sha256') != hash_rows(canonical_rows):
        raise ValueError('Cache hash mismatch: refusing refresh')
    if manifest.get('row_count') != len(canonical_rows):
        raise ValueError('Cache size mismatch')
    return canonical_rows


def prepare(now, mode, fetch=None):
    """Prepare canonical rows, a verified snapshot, and provenance metadata.

    This function performs read-only Alpaca GETs and in-memory validation only;
    it does not write cache, snapshot, or manifest files.

    Args:
        now: timezone-aware current instant.
        mode: 'bootstrap' for a fresh 250-day backfill or 'refresh' to merge
            new bars into the verified existing cache.
        fetch: optional callable(start_utc, end_utc) returning bar rows;
            when omitted, bounded live Alpaca GET chunks are used.

    Returns:
        tuple: (canonical row list, snapshot dict, provenance record dict).

    Raises:
        ValueError: for a naive clock, invalid fetch window, empty or invalid
            Alpaca result, incomplete prior session, existing bootstrap cache,
            unverified old cache, unsupported mode, or snapshot mismatch.
        OSError: propagated from credential, cache, or manifest reads.
        RuntimeError: propagated from the market data adapter.
    """
    if now.tzinfo is None:
        raise ValueError('Timezone-aware clock required')
    start, end = fetch_window(now, mode == 'bootstrap')
    raw, chunks = authenticated_fetch_chunks(start, end, fetch=fetch)
    incoming = normalize_bars(raw)
    if not incoming:
        raise ValueError('Alpaca returned no IEX bars')
    incoming_rows = rows_from_map(incoming)
    has_full_session = refresher.api_has_expected_full_session(
        incoming_rows, closing_date(now)
    )
    if not has_full_session:
        raise ValueError(
            'Authenticated Alpaca GET lacks 30 prior-session closing prices'
        )
    if mode == 'bootstrap':
        if any(path.exists() for path in (CACHE, SNAP, MANIFEST)):
            raise ValueError('V28 cache already exists: refusing bootstrap overwrite')
        combined = incoming
        legacy_used = False
    elif mode == 'refresh':
        old_rows = verified_old_cache()
        combined = normalize_bars(refresher.consolidate(old_rows, incoming_rows))
        legacy_used = False
    else:
        raise ValueError('Unsupported operation')
    rows = rows_from_map(combined)
    snap, complete_days, incomplete_days = make_snapshot(rows, now)
    record = {
        'schema': 28,
        'mode': mode,
        'feed': EXPECTED_FEED,
        'adjustment': EXPECTED_ADJUSTMENT,
        'timeframe': '15Min',
        'data_origin': (
            'ALPACA_API_ONLY_NO_IMPORTED_SEED'
            if fetch is None
            else 'OFFLINE_INJECTED_TEST'
        ),
        'last_complete_day_et': closing_date(now),
        'fetched_from_utc': start.isoformat(),
        'fetched_to_utc': end.isoformat(),
        'fetched_at_utc': now.astimezone(UTC).isoformat(),
        'incoming_bars': len(incoming),
        'get_windows': chunks,
        'row_count': len(rows),
        'complete_days': len(complete_days),
        'incomplete_days': len(incomplete_days),
        'rows_sha256': hash_rows(rows),
        'snapshot_lookback_sha256': snap['lookback_sha256'],
        'v23_strategy': snap['strategy'],
    }
    return rows, snap, record


def persist(rows, snapshot, meta):
    """Write verified bars, snapshot, and provenance manifest in order.

    Args:
        rows: canonical bar rows for the CSV cache.
        snapshot: verified snapshot dict for the JSON snapshot file.
        meta: provenance record dict for the JSON manifest file.

    Returns:
        None.

    Raises:
        ValueError: if the data path fails isolation checks.
        OSError: if directory creation or atomic file replacement fails.
    """
    require_isolated_data_path()
    DATA.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Individual atomic writes; provenance is published LAST, and verifies data.
    refresher.atomic_csv(CACHE, rows)
    market.safe_atomic_json(SNAP, snapshot)
    market.safe_atomic_json(MANIFEST, meta)


def run(mode, now=None, fetch=None, write=True):
    """Prepare V28 data, optionally persist it, and print status markers.

    Args:
        mode: 'bootstrap' or 'refresh', passed through to prepare.
        now: optional timezone-aware current instant; defaults to UTC now.
        fetch: optional bar-fetch callable passed through to prepare.
        write: truthy value enables persist; falsy value is dry-run only.

    Returns:
        dict: the provenance record produced by prepare.

    Raises:
        ValueError: propagated from clock, fetch, validation, or cache
            verification failures.
        OSError: propagated from reads or atomic writes.
        RuntimeError: propagated from the market data adapter.
    """
    current_time = now or datetime.now(UTC)
    require_isolated_data_path()
    rows, snap, record = prepare(current_time, mode, fetch=fetch)
    if write:
        persist(rows, snap, record)
    print('V28_CLEAN_ALPACA_' + mode.upper() + '_PASS')
    print(
        'V28_SOURCE',
        'ALPACA_AUTHENTICATED_IEX_15MIN_RAW_NO_SEED'
        if fetch is None
        else 'SYNTHETIC_OFFLINE_IEX_RAW_TEST'
    )
    print('V28_PRIOR_COMPLETE_SESSION', record['last_complete_day_et'])
    print('V28_FETCHED_BARS', record['incoming_bars'])
    print('V28_BOUNDED_GET_WINDOWS', record['get_windows'])
    print('V28_COMPLETE_DAYS', record['complete_days'])
    print('V28_STORED_BARS', record['row_count'])
    print('V28_INCOMPLETE_DAYS', record['incomplete_days'])
    print('V28_DATA_SHA256', record['rows_sha256'])
    print('V28_LOOKBACK_SHA256', record['snapshot_lookback_sha256'])
    print('V28_SELECTED_FIVE', ','.join(snap['selected_five']))
    print('V28_TARGET_EQUITY', format(sum(snap['target_weights'].values()), '.12f'))
    print('V393_SNAPSHOT_TARGET_SHAPE_VALIDATED_NO_SYNTHETIC_R1')
    print('V28_DATA_LOCATION', DATA)
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
    """Run the built-in self-test or an acknowledged read-only refresh mode.

    Returns:
        The self-test return value for 'self-test'; otherwise None after a
        successful run, or SystemExit(2) for a handled operational failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('self-test', 'bootstrap', 'refresh'))
    parser.add_argument('--ack-readonly-alpaca-get', action='store_true')
    args = parser.parse_args()

    if args.action == 'self-test':
        return test()
    if not args.ack_readonly_alpaca_get:
        parser.error('Require --ack-readonly-alpaca-get')
    try:
        run(args.action)
    except (
        ValueError,
        KeyError,
        FileNotFoundError,
        RuntimeError,
        TimeoutError,
        OSError,
    ) as exc:
        print('V28_FAIL_CLOSED', type(exc).__name__, str(exc)[:190])
        raise SystemExit(2)


if __name__ == '__main__':
    main()
