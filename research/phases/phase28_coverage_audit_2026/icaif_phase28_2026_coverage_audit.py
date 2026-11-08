#!/usr/bin/env python3
r"""ICAIF Phase 28: read-only coverage audit for local 2026 Alpaca IEX CSVs.

Run from existing starter-kit root (Windows PowerShell, venv active):
  python "$HOME\Downloads\icaif_phase28_2026_coverage_audit.py" --self-test
  python "$HOME\Downloads\icaif_phase28_2026_coverage_audit.py" --audit

Does NOT train, fetch market data, access credentials, change strategy, or submit.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

SYMBOLS = tuple("AAPL MSFT NVDA INTC CRM JPM BAC GS V PYPL LLY JNJ UNH PFE TMO "
                "AMZN TSLA WMT NKE KO CAT GE BA XOM CVX GOOGL META DIS T NEE".split())
FILENAMES = ('alpaca_iex_15m_400d.csv', 'alpaca_iex_15min_recent.csv')
TIME_CANDIDATES = ('timestamp_et', 'timestamp', 'datetime', 'time', 't', 'date', 'Date', 'Timestamp')
SYMBOL_CANDIDATES = ('symbol', 'ticker', 'Symbol', 'Ticker', 'S')
ET = ZoneInfo('America/New_York')
CUTOFF = '2026-10-09'


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def choose_field(fieldnames, candidates):
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def parse_timestamp(value: str, column: str):
    """Return (ET datetime, timestamp classification). Reject ambiguous numbers."""
    value = value.strip()
    if not value:
        raise ValueError('Empty timestamp')
    if value.isdigit() and len(value) in (10, 13):
        timestamp = int(value) / (1000 if len(value) == 13 else 1)
        return datetime.fromtimestamp(timestamp, timezone.utc).astimezone(ET), 'epoch_utc'
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        try:
            dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        except ValueError as exc:
            raise ValueError('Unrecognized timestamp format') from exc
    if dt.tzinfo is None:
        # Do not silently assume a time zone unless the field explicitly says ET.
        if column == 'timestamp_et':
            return dt.replace(tzinfo=ET), 'naive_explicit_et'
        return dt.replace(tzinfo=ET), 'naive_unknown_tz_ASSUMED_ET_FOR_COVERAGE_ONLY'
    return dt.astimezone(ET), 'tz_aware'


def audit_csv(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Missing CSV: {path.name}')
    all_symbols = collections.Counter()
    y2026_symbols = collections.Counter()
    date_2026 = collections.Counter()
    session_days = collections.defaultdict(set)
    minute_2026 = collections.Counter()
    time_types = collections.Counter()
    invalid_time = 0
    duplicate_count = 0
    observed_keys = set()
    min_date = max_date = min_2026 = max_2026 = None
    future_rows = 0
    rows = 0
    examples_invalid = []
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        sym_field = choose_field(columns, SYMBOL_CANDIDATES)
        time_field = choose_field(columns, TIME_CANDIDATES)
        if sym_field is None or time_field is None:
            return {'filename':path.name, 'status':'UNSUPPORTED_SCHEMA',
                    'columns':columns, 'symbol_column':sym_field,
                    'timestamp_column':time_field, 'sha256':fingerprint(path)}
        for row in reader:
            rows += 1
            symbol = str(row.get(sym_field) or '').strip().upper()
            all_symbols[symbol] += 1
            try:
                dt,kind = parse_timestamp(str(row.get(time_field) or ''),time_field)
            except (ValueError,OverflowError,OSError):
                invalid_time += 1
                if len(examples_invalid) < 3:
                    examples_invalid.append(rows)
                continue
            time_types[kind] += 1
            ds = dt.date().isoformat()
            min_date = min(min_date, ds) if min_date is not None else ds
            max_date = max(max_date, ds) if max_date is not None else ds
            if ds > CUTOFF:
                future_rows += 1
            if dt.year == 2026:
                min_2026 = min(min_2026, ds) if min_2026 is not None else ds
                max_2026 = max(max_2026, ds) if max_2026 is not None else ds
                y2026_symbols[symbol] += 1
                date_2026[ds] += 1
                minute_2026[dt.strftime('%H:%M')] += 1
                if symbol in SYMBOLS and ds <= CUTOFF:
                    session_days[symbol].add(ds)
                key = (symbol,dt.isoformat())
                if key in observed_keys:
                    duplicate_count += 1
                observed_keys.add(key)
    valid_2026_symbols = sorted(s for s in y2026_symbols if s in SYMBOLS)
    complete_days = sum(1 for d,c in date_2026.items() if d <= CUTOFF and c >= 30)
    return {
        'filename':path.name,'status':'AUDITED','bytes':path.stat().st_size,
        'sha256':fingerprint(path),'columns':columns,
        'symbol_column':sym_field,'timestamp_column':time_field,
        'rows':rows,'invalid_timestamps':invalid_time,
        'invalid_timestamp_row_numbers':examples_invalid,
        'timestamp_interpretations':dict(time_types),
        'first_date_et':min_date,'last_date_et':max_date,
        'rows_after_preofficial_cutoff_oct09':future_rows,
        'first_date_2026_et':min_2026,'last_date_2026_et':max_2026,
        'records_2026':sum(y2026_symbols.values()),
        'symbols_2026':valid_2026_symbols,
        'missing_universe_symbols_2026':sorted(set(SYMBOLS)-set(valid_2026_symbols)),
        'records_2026_per_symbol':{s:y2026_symbols[s] for s in SYMBOLS},
        'preofficial_2026_distinct_days_per_symbol':{s:len(session_days[s]) for s in SYMBOLS},
        'days_2026_with_at_least_30_csv_rows_preofficial':complete_days,
        'duplicate_symbol_timestamp_records_2026':duplicate_count,
        'most_common_bar_times_et_2026':minute_2026.most_common(16),
        'first_ten_2026_dates':sorted(date_2026)[:10],
        'last_ten_2026_dates':sorted(date_2026)[-10:],
    }


def self_test():
    assert parse_timestamp('2026-10-09T13:30:00Z','timestamp')[0].strftime('%H:%M') == '09:30'
    assert parse_timestamp('2026-10-09 09:30:00','timestamp_et')[1]=='naive_explicit_et'
    assert parse_timestamp('1791552600','t')[1]=='epoch_utc'
    with TemporaryDirectory() as temp:
        p = Path(temp)/'alpaca_iex_15m_400d.csv'
        p.write_text('symbol,timestamp,open,close\nAAPL,2026-10-09T13:30:00Z,100,101\n'
                     'MSFT,2025-12-31T14:30:00Z,200,202\n',encoding='utf-8')
        a = audit_csv(p)
        assert a['status']=='AUDITED' and a['records_2026']==1
        assert a['most_common_bar_times_et_2026'][0][0]=='09:30'
        assert 'MSFT' in a['missing_universe_symbols_2026']
        assert a['rows_after_preofficial_cutoff_oct09']==0
    print('PHASE28_SELF_TEST_PASS')
    print('UTC_TO_ET_AND_YEAR_COVERAGE_PASS')
    print('NO_NETWORK_NO_CREDENTIALS_NO_ORGANIZER_SUBMISSION')


def run():
    root = Path.cwd().resolve()
    if not (root/'tools'/'auto_submit.py').is_file():
        raise RuntimeError('Open the starter-kit directory first')
    folder = root/'data'/'cache'
    reports = []
    for name in FILENAMES:
        report = audit_csv(folder/name)
        reports.append(report)
        print('FILE:',name)
        print('  STATUS:',report['status'])
        print('  COLUMNS:',','.join(report.get('columns',[])))
        if report['status']!='AUDITED':continue
        print('  TOTAL_ROWS:',report['rows'],'INVALID_TIMESTAMPS:',report['invalid_timestamps'])
        print('  DATE_RANGE_ET:',report['first_date_et'],'TO',report['last_date_et'])
        print('  2026_RANGE_ET:',report['first_date_2026_et'],'TO',report['last_date_2026_et'])
        print('  2026_ROWS:',report['records_2026'],'SYMBOLS:',len(report['symbols_2026']),'/30')
        print('  MISSING_2026_SYMBOLS:',','.join(report['missing_universe_symbols_2026']) or 'NONE')
        print('  ROWS_AFTER_OCT09:',report['rows_after_preofficial_cutoff_oct09'])
        print('  TIMESTAMP_INTERPRETATIONS:',report['timestamp_interpretations'])
        print('  COMMON_ET_BAR_TIMES:',report['most_common_bar_times_et_2026'][:8])
        print('  DUPLICATES_2026:',report['duplicate_symbol_timestamp_records_2026'])
    out = root/'artifacts'/'phase28_2026_coverage'
    out.mkdir(parents=True,exist_ok=True)
    path = out/'phase28_2026_coverage.json'
    path.write_text(json.dumps({'asof_cutoff_et':CUTOFF,'files':reports,
                  'note':'Coverage audit only. Not evidence of adjustment, feed provenance, price accuracy, or future label maturity.'},
                  indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('REPORT:',str(path))
    print('PHASE28_COVERAGE_AUDIT_COMPLETE_NO_SUBMISSION')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--self-test',action='store_true')
    action.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:run()

if __name__=='__main__':
    main()
