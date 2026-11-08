#!/usr/bin/env python3
# Role: STRATEGY BASE - price-data validation, standardization and the 80% core allocation.
"""ICAIF 2026 release V2 support library; read-only market data and pure blending math.

Subcommands:
  self-test: synthetic tests incl. HOLD, malformed portfolio, out-of-time snapshot
  inspect-csv: audit local legacy Alpaca 15m history / calculate initial Blend80
  probe-alpaca: authenticated, read-only Alpaca IEX bars health test
  prepare-snapshot: read-only Alpaca refresh + seed CSV, atomic snapshot for standalone strategy.

No strategy function, no organizer writer, and no initial Equal80 fallback exist in this module.
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
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import error, parse, request
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
UTC = timezone.utc
SYMBOLS = ('AAPL', 'MSFT', 'NVDA', 'INTC', 'CRM', 'JPM', 'BAC', 'GS', 'V', 'PYPL',
           'LLY', 'JNJ', 'UNH', 'PFE', 'TMO', 'AMZN', 'TSLA', 'WMT', 'NKE', 'KO',
           'CAT', 'GE', 'BA', 'XOM', 'CVX', 'GOOGL', 'META', 'DIS', 'T', 'NEE') # Never change the order, ordered by industry
GROUPS = tuple(tuple(range(i, i + 5)) for i in range(0, 30, 5)) # Devin found: Never used, dead end
DEADLINES = {1: '09:10', 2: '10:25', 3: '11:25', 4: '12:25', 5: '13:25',
             6: '14:25', 7: '15:25'}
API = 'https://data.alpaca.markets/v2/stocks/bars'
EXPOSURE = .80


def utc_stamp(iso_text):
    """Parse an ISO-8601 timestamp into a timezone-aware UTC datetime.

    Args:
        iso_text: ISO-8601 timestamp string; a trailing 'Z' is accepted.

    Returns:
        datetime: the same instant expressed in UTC.

    Raises:
        ValueError: if the input has no timezone information.
    """
    parsed = datetime.fromisoformat(iso_text.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Expected timezone-aware UTC datetime')
    return parsed.astimezone(UTC)


def standardize(values): # Fairly compare x-day returns later
    """Z-score each value, clip to [-3, 3], and guard against zero variance.

    Args:
        values: iterable of numeric weights or feature values.

    Returns:
        list[float]: standardized values clipped to the range [-3, 3].

    Raises:
        ValueError: if any value is non-finite.
    """
    numbers = [float(x) for x in values]
    if any(not math.isfinite(x) for x in numbers): # To refuse to calculate with broken data
        raise ValueError('Non-finite weight or feature') # Appears when something upstream went wrong, e.g., missing piece or log(0) = minus infinity
    mean = sum(numbers) / len(numbers)
    std_dev = math.sqrt(sum((x - mean) ** 2 for x in numbers) / len(numbers))
    return [min(3., max(-3., (x - mean) / max(std_dev, 1e-8))) for x in numbers] # max/min value of z-score capped to 3 and -3; 1e-8 is to avoid division by 0


# Calculate 34% diversified core and 34% momentum-based weights
def blend_weights(daily_closes):
    """Compute the exact Phase30 v2/Phase37 blended target weights.

    Uses lagged historical CLOSE prices only (121 complete daily closes,
    30 symbols ordered as SYMBOLS).

    Args:
        daily_closes: sequence of 121+ rows; each row is a sequence of 30
            positive closing prices.

    Returns:
        dict: symbol -> target weight summing to EXPOSURE (0.80).

    Raises:
        ValueError: if fewer than 121 days are given or the price matrix
            is malformed.
        AssertionError: if the blended total deviates from 0.80.
    """

    # Fail-closed input checks
    if len(daily_closes) < 121:
        raise ValueError('Need >=121 complete daily closes')
    import numpy as np
    prices = np.array(daily_closes[-121:], dtype=float)
    if prices.shape != (121, 30) or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Bad historical market shape or prices')
    
    # Converting into log returns (for accurate addition over time)
    log_prices = np.log(prices)
    return_20d = log_prices[-1] - log_prices[-21] # Negative indexing, -1 is most recent day
    return_60d = log_prices[-1] - log_prices[-61]
    return_120d = log_prices[-1] - log_prices[-121]
    daily_log_returns = log_prices[-20:] - log_prices[-21:-1] # 20 day-to-day log returns per stock, spread = sd = volatility
    volatility = np.std(daily_log_returns, axis=0, ddof=1)
    
    # Building momentum features: momentum score, industry momentum, and combined score
    relative_momentum = return_20d - np.repeat(return_20d.reshape(6, 5).mean(axis=1), 5) # Unused; Reshape: build groups; mean: 1 momentum per industry; repeat: each stock receives industry avg
    features = np.stack((standardize(return_20d), standardize(return_60d),
                         standardize(return_120d), standardize(-volatility),
                         standardize(relative_momentum)), axis=1) # Last two features unused in most recent code
    momentum = .25 * features[:, 0] + .50 * features[:, 1] + .25 * features[:, 2] # Weights medium-term momentum most
    group_momentum = np.repeat(momentum.reshape(6, 5).mean(axis=1), 5)
    score = .65 * np.array(standardize(group_momentum)) + .35 * np.array(standardize(momentum)) # Combine industry and stock momentum
    standardized_score = np.array(standardize(score))

    # Convert scores to weights
    proportional_weights = np.clip(1. + .35 * standardized_score, .01, None) # Multiplier around 1, ensures weights are positive (long-only)
    industry_weights = .8 * proportional_weights / proportional_weights.sum() # Normalize to sum to 0.8
    blended = .5 * industry_weights + .5 * np.full(30, .8 / 30) # 50% industry weights, 50% equal weights

    # Final checks
    validate_targets(dict(zip(SYMBOLS, map(float, blended))))
    if abs(blended.sum() - .8) > 1e-9:
        raise AssertionError('Blend total equity wrong') # Sanity check for total weight of 0.8
    return dict(zip(SYMBOLS, map(float, blended)))


def validate_targets(weights):
    """Validate a candidate target-weight dict against portfolio rules.

    Args:
        weights: dict mapping each official symbol to a numeric weight.

    Raises:
        ValueError: if the dict is not exactly the 30 SYMBOLS, any weight
            is non-finite or outside [0, 0.30], the total exceeds 1, or
            the total is below 0.01.
    """
    if not isinstance(weights, dict) or set(weights) != set(SYMBOLS):
        raise ValueError('Expected exact 30 official symbols')
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= .30
           for x in weights.values()):
        raise ValueError('Illegal portfolio number or stock weight')
    if sum(weights.values()) > 1. + 1e-9:
        raise ValueError('Total weight exceeds 1')
    if sum(weights.values()) < .01:
        raise ValueError('Accidental all-zero liquidation blocked')


# Read Alpaca's raw records and group them 
def parse_bars(rows, until_utc):
    """Collect fully completed regular-session IEX 15-minute bars.

    Args:
        rows: iterable of bar records with 'symbol', 'timestamp', 'close'.
        until_utc: timezone-aware UTC cutoff; bars completing after this
            instant are rejected so no future record is accepted.

    Returns:
        dict: ET date string -> {(date, 'HH:MM', symbol): close price}.

    Raises:
        ValueError: on invalid prices, duplicate bars, or bad timestamps.
    """

    bars_by_date = defaultdict(dict)
    for record in rows:
        symbol = record.get('symbol')
        
        # Ignore stocks outside 30-stock universe
        if symbol not in SYMBOLS:
            continue
        
        # Convert time
        bar_start_utc = utc_stamp(record['timestamp'])
        local_time = bar_start_utc.astimezone(ET)

        # Keep only stock market hour bars: 9.30 - 16.00 -> 26 bars per day
        minute_of_day = local_time.hour * 60 + local_time.minute
        if minute_of_day < 9 * 60 + 30 or minute_of_day > 15 * 60 + 45 or minute_of_day % 15:
            continue
        if bar_start_utc + timedelta(minutes=15) > until_utc: # Only finished bars
            continue
        
        # Create bar key with date, time, stock
        bar_key = (local_time.date().isoformat(), local_time.strftime('%H:%M'), symbol)
        close_price = float(record['close'])

        # Validate prices
        if not math.isfinite(close_price) or close_price <= 0:
            raise ValueError('Invalid price')
        if bar_key in bars_by_date[local_time.date().isoformat()]:
            raise ValueError('Duplicate date-time symbol bar')
        
        # Add price to dictionary
        bars_by_date[local_time.date().isoformat()][bar_key] = close_price
    return bars_by_date


def history_closes(rows, until_utc):
    """Reduce 15-minute bars to one complete daily close vector per session.

    Args:
        rows: iterable of raw bar records (see parse_bars).
        until_utc: UTC cutoff for accepting bars.

    Returns:
        tuple: (complete_dates, daily_closes, incomplete_dates) where
            complete_dates is a sorted list of ET date strings,
            daily_closes is a parallel list of 30-price lists taken at the
            15:45 ET bar, and incomplete_dates lists sessions missing any
            expected bar.
    """

    # Group bars by date, create 3 lists to collect results
    bars_by_date = parse_bars(rows, until_utc)
    complete_dates = []
    daily_closes = []
    incomplete_dates = []

    # Keep only 15-minute bars during stock market hours (9:30-16:00)
    session_times = [f'{hour:02d}:{minute:02d}'
                     for hour in range(9, 16) for minute in (0, 15, 30, 45)
                     if (hour > 9 or minute >= 30)]
    
    # Check grid size
    if len(session_times) != 26:
        raise AssertionError('Incorrect 15-minute session grid')

    # Sort by date, keep only complete sessions, keep closing price
    for date in sorted(bars_by_date):
        day_bars = bars_by_date[date]
        if any((date, hm, symbol) not in day_bars
               for hm in session_times for symbol in SYMBOLS):
            incomplete_dates.append(date)
            continue
        closing_prices = [day_bars[(date, '15:45', symbol)] for symbol in SYMBOLS]
        complete_dates.append(date)
        daily_closes.append(closing_prices)
    return complete_dates, daily_closes, incomplete_dates


def load_csv(path):
    """Yield the relevant columns of a legacy Alpaca CSV export.

    Args:
        path: filesystem path to a UTF-8 CSV with 'symbol', 'timestamp',
            and 'close' columns (a UTF-8 BOM is tolerated).

    Yields:
        dict: {'symbol': ..., 'timestamp': ..., 'close': ...} per row.
    """
    with open(path, encoding='utf-8-sig', newline='') as csv_file:
        for row in csv.DictReader(csv_file):
            yield {key: row.get(key) for key in ('symbol', 'timestamp', 'close')}


def infer_snapshot(rows, now_utc, kind='OFFLINE_DIAGNOSTIC'):
    """Build a standalone-strategy snapshot dict from raw bar rows.

    Args:
        rows: iterable of raw bar records (see parse_bars).
        now_utc: timezone-aware UTC instant used as the freshness cutoff.
        kind: snapshot 'purpose' label stored in the result.

    Returns:
        dict: snapshot metadata including lookback_sha256 and
            target_weights from blend_weights.

    Raises:
        ValueError: if fewer than 121 complete daily closes exist.
    """
    dates, daily_closes, missing_dates = history_closes(rows, now_utc)
    if len(dates) < 121:
        raise ValueError(
            f'Only {len(dates)} complete 30-stock daily closes; require >=121')
    target_weights = blend_weights(daily_closes)
    digest = hashlib.sha256(json.dumps(
        {'dates': dates[-121:], 'daily': daily_closes[-121:]},
        sort_keys=True).encode()).hexdigest()
    return {'schema': 1, 'purpose': kind, 'created_utc': now_utc.isoformat(),
            'last_complete_day_et': dates[-1], 'complete_days': len(dates),
            'missing_endofday_sessions': len(missing_dates),
            'lookback_sha256': digest,
            'symbols': list(SYMBOLS), 'target_weights': target_weights,
            'lookback_days': 121,
            'data_feed': 'Alpaca IEX 15Min raw daily 15:45 ET closes',
            'strategy': 'exact-phase37-core80-blend-hold-initial-only'}


def load_alpaca_headers(env_file=None):
    """Resolve Alpaca API credentials into request headers.

    Args:
        env_file: optional path to a private env file; recognized keys are
            APCA_API_KEY_ID/ALPACA_API_KEY and
            APCA_API_SECRET_KEY/ALPACA_SECRET_KEY. File entries overwrite
            the copied process environment for those keys.

    Returns:
        dict: Alpaca request headers including key, secret, and Accept.

    Raises:
        FileNotFoundError: if env_file is given but does not exist.
        ValueError: if either credential is missing.
    """
    environment = dict(os.environ)
    if env_file is not None:
        env_path = Path(env_file).expanduser()
        if not env_path.is_file():
            raise FileNotFoundError('Private Alpaca environment file missing')
        for raw_line in env_path.read_text(encoding='utf-8-sig').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:]
            if '=' in line:
                key, value = line.split('=', 1)
                if key.strip() in ('APCA_API_KEY_ID', 'ALPACA_API_KEY',
                                   'APCA_API_SECRET_KEY', 'ALPACA_SECRET_KEY'):
                    environment[key.strip()] = value.strip().strip(chr(34)).strip(chr(39))
    api_key = environment.get('APCA_API_KEY_ID') or environment.get('ALPACA_API_KEY')
    api_secret = (environment.get('APCA_API_SECRET_KEY')
                  or environment.get('ALPACA_SECRET_KEY'))
    if not api_key or not api_secret:
        raise ValueError('Alpaca credentials not in process environment; DO NOT share them')
    return {'APCA-API-KEY-ID': api_key, 'APCA-API-SECRET-KEY': api_secret,
            'Accept': 'application/json'}


def fetch_iex(start_utc, end_utc, transport=None, env_file=None):
    """Fetch Alpaca IEX 15-minute bars for all SYMBOLS over a UTC window.

    Args:
        start_utc: timezone-aware UTC start instant.
        end_utc: timezone-aware UTC end instant.
        transport: optional callable(params)->response dict used instead of
            the real HTTP request (for tests/offline use); when None,
            credentials are loaded via load_alpaca_headers.
        env_file: optional private env file for credentials (HTTP mode only).

    Returns:
        list[dict]: flattened bars {'symbol', 'timestamp', 'close'} across
            all paginated responses.

    Raises:
        RuntimeError: on Alpaca HTTP or network failure.
        ValueError: on response schema, symbol, or pagination errors.
    """
    headers = load_alpaca_headers(env_file) if transport is None else {}
    query_params = {'symbols': ','.join(SYMBOLS), 'timeframe': '15Min',
                    'start': start_utc.isoformat(), 'end': end_utc.isoformat(),
                    'feed': 'iex', 'adjustment': 'raw', 'sort': 'asc',
                    'limit': '10000'}
    seen_tokens = set()
    collected_bars = []
    for _page_number in range(50):
        if transport is not None:
            response = transport(query_params)
        else:
            url = API + '?' + parse.urlencode(query_params)
            try:
                with request.urlopen(request.Request(url, headers=headers), timeout=22) as resp:
                    response = json.loads(resp.read(8_000_000))
            except error.HTTPError as exc:
                raise RuntimeError('Alpaca HTTP ' + str(exc.code) + '; no trading action') from None
            except (error.URLError, TimeoutError):
                raise RuntimeError('Alpaca IEX network unavailable; HOLD') from None
        if not isinstance(response, dict) or not isinstance(response.get('bars'), dict):
            raise ValueError('Alpaca bars response schema invalid')
        for symbol, items in response['bars'].items():
            if symbol not in SYMBOLS:
                raise ValueError('Unexpected source symbol')
            for bar in items:
                collected_bars.append({'symbol': symbol, 'timestamp': bar['t'],
                                       'close': bar['c']})
        next_token = response.get('next_page_token')
        if not next_token:
            break
        if next_token in seen_tokens:
            raise ValueError('Alpaca duplicate pagination token')
        seen_tokens.add(next_token)
        query_params['page_token'] = next_token
    else:
        raise ValueError('Alpaca pagination overflow')
    return collected_bars


def safe_atomic_json(target, obj):
    """Write obj as JSON to target via a flushed, fsynced temp file.

    Args:
        target: destination file path (parent dirs are created).
        obj: JSON-serializable object written with indent=2, sort_keys=True.
    """
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8',
                                     dir=str(target_path.parent), delete=False,
                                     prefix='._icaif_', suffix='.json') as temp_file:
        temp_path = Path(temp_file.name)
        json.dump(obj, temp_file, indent=2, sort_keys=True)
        temp_file.flush()
        os.fsync(temp_file.fileno())
    try:
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
