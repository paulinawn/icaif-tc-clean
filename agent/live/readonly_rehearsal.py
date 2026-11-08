#!/usr/bin/env python3
"""V30 one-shot unarmed server reality check.

Runs only organizer GET via a transport that hard-rejects every non-GET.
Reads (does not mutate) the isolated V28 data, invokes the frozen V23 pure
strategy via the pure V29 bridge, and prints safe status codes only.
The *only* filesystem writes are inside a disposable organizer checkpoint
under /tmp; it is deleted before process exit. No state journal is advanced.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

APP = Path('/opt/icaif2026')
DATA = APP / 'official_clean_bootstrap_v28_unarmed' / 'data'
V23 = APP / 'official_staging_v23_unarmed' / 'code'


def safe_history(session, *, limit=200, max_pages=5):
    """Fully paginate, disallow odd offsets, fail closed on incomplete history.

    Args:
        session: organizer client exposing ``decisions(phase, offset=, limit=)``.
        limit: page size requested from the server.
        max_pages: hard cap on pagination loops before failing closed.

    Returns:
        dict with keys ``decisions`` (all rows), ``has_more`` (False),
        ``limit`` and ``offset`` (0).

    Raises:
        ValueError: on schema violations, ambiguous or incomplete pagination.
    """
    out = []
    offset = 0
    for page in range(max_pages):
        response = session.decisions('official', offset=offset, limit=limit)
        if not isinstance(response, dict) or not isinstance(response.get('decisions'), list):
            raise ValueError('BAD_DECISIONS_SCHEMA')
        rows = response['decisions']
        if any(not isinstance(x, dict) for x in rows):
            raise ValueError('BAD_DECISION_RECORD')
        out.extend(rows)
        if response.get('has_more') is False:
            return {'decisions': out, 'has_more': False, 'limit': limit, 'offset': 0}
        if response.get('has_more') is not True or not rows:
            raise ValueError('HISTORY_PAGINATION_AMBIGUOUS')
        if len(rows) != limit:
            raise ValueError('HISTORY_PAGE_LENGTH_DISAGREES')
        offset += len(rows)
    raise ValueError('HISTORY_EXCEEDS_MAX_PAGES')


def validate_snapshot_meta(snapshot, manifest, csvpath):
    """Read V28 provenance and ensure it matches the present V23 snapshot.

    A full CSV hash is validated by independent V28 verifier, not guessed from
    this abbreviated metadata gate. Any mismatch must HOLD.

    Args:
        snapshot: parsed ``latest_readonly_snapshot.json`` contents.
        manifest: parsed ``cache_provenance.json`` contents.
        csvpath: path to the cached Alpaca IEX 15m CSV.

    Returns:
        int: number of canonical (symbol, timestamp) rows hashed.

    Raises:
        ValueError: on any provenance, schema, or hash mismatch.
    """
    if manifest.get('schema') != 28 or manifest.get('data_origin') != 'ALPACA_API_ONLY_NO_IMPORTED_SEED':
        raise ValueError('BAD_ALPACA_PROVENANCE')
    if manifest.get('feed') != 'iex' or manifest.get('adjustment') != 'raw' or manifest.get('timeframe') != '15Min':
        raise ValueError('BAD_MARKET_FEED')
    if manifest.get('row_count', 0) < 121 * 26 * 30 or manifest.get('complete_days', 0) < 121:
        raise ValueError('INSUFFICIENT_MARKET_HISTORY')
    if not isinstance(snapshot, dict) or snapshot.get('purpose') != 'LIVE_READONLY_VERIFIED':
        raise ValueError('NOT_VERIFIED_SNAPSHOT')
    if snapshot.get('strategy') != manifest.get('v23_strategy'):
        raise ValueError('STRATEGY_PROVENANCE_MISMATCH')
    if snapshot.get('lookback_sha256') != manifest.get('snapshot_lookback_sha256'):
        raise ValueError('LOOKBACK_PROVENANCE_MISMATCH')
    if snapshot.get('last_complete_day_et') != manifest.get('last_complete_day_et'):
        raise ValueError('ASOF_PROVENANCE_MISMATCH')
    if not csvpath.is_file() or csvpath.is_symlink() or csvpath.stat().st_size < 100000:
        raise ValueError('ALPACA_CACHE_MISSING')
    # Import the same canonical hash function and parser as V28, then check
    # the complete stored data. Does not fetch or modify files.
    import market_support as market
    from collections import defaultdict
    canonical = {}
    for r in market.load_csv(csvpath):
        key = (r['symbol'], market.utc_stamp(r['timestamp']).isoformat())
        if key in canonical:
            raise ValueError('DUPLICATED_CACHED_BAR')
        canonical[key] = float(r['close'])
    if len(canonical) != manifest['row_count']:
        raise ValueError('ROW_COUNT_PROVENANCE_MISMATCH')
    ordered = sorted(canonical.items(), key=lambda e: (e[0][1], e[0][0]))
    h = hashlib.sha256()
    for (sym, stamp), value in ordered:
        h.update(f'{sym},{stamp},{value:.15g}\n'.encode('ascii'))
    if h.hexdigest() != manifest.get('rows_sha256'):
        raise ValueError('CSV_SHA256_MISMATCH')
    return len(canonical)


class GetOnlyTransport:
    """HTTP transport denies all network methods except GET, at IO boundary."""
    def __init__(self, delegate):
        """Wrap a real HTTP transport delegate.

        Args:
            delegate: underlying transport whose ``handle_request`` performs GETs.
        """
        self.delegate = delegate
        self.attempts = []

    def handle_request(self, request):
        """Record the attempt and forward GET requests only.

        Args:
            request: httpx request object.

        Returns:
            The delegate's response for GET requests.

        Raises:
            RuntimeError: 'V30_NETWORK_WRITE_BLOCKED' for any non-GET method.
        """
        self.attempts.append(request.method)
        if request.method != 'GET':
            raise RuntimeError('V30_NETWORK_WRITE_BLOCKED')
        return self.delegate.handle_request(request)

    def close(self):
        """Close the underlying delegate transport."""
        self.delegate.close()


def main():
    """Run the one-shot GET-only reality check against the live organizer.

    Reads V28 cache provenance, opens a disposable checkpoint session,
    collects schedule/portfolio/history/ledger via GET only, and prints
    V30_* status lines. Raises on any verification failure (fail closed).

    Returns:
        None.

    Raises:
        ValueError: on missing stage dirs, missing credentials, or bad ledger.
        RuntimeError: on non-GET transport attempts or broken bridge contract.
    """
    # Local package modules injected by the Windows launcher entirely in memory.
    import organizer_bridge as bridge
    sys.path.insert(0, str(APP))
    sys.path.insert(0, str(V23))
    import httpx
    from kit.config import load_environment
    from kit.original_client import OriginalSession, load_profile

    if not APP.is_dir() or not V23.is_dir() or not DATA.is_dir():
        raise ValueError('EXPECTED_HETZNER_STAGE_NOT_FOUND')
    load_environment(APP / '.env')
    token = os.environ.get('CODABENCH_TOKEN', '')
    profile = os.environ.get('ICAIF_PROFILE', '')
    if not token or not profile:
        raise ValueError('PRIVATE_CONFIG_NOT_AVAILABLE')
    p = Path(profile)
    if not p.is_absolute():
        p = APP / p
    prof = load_profile(p)
    import sleeve_engine as engine
    snap = json.loads((DATA / 'latest_readonly_snapshot.json').read_text(encoding='utf-8'))
    manifest = json.loads((DATA / 'cache_provenance.json').read_text(encoding='utf-8'))
    n = validate_snapshot_meta(snap, manifest, DATA / 'alpaca_iex_raw_15m.csv')
    print('V30_CLEAN_ALPACA_CACHE_HASH_PASS', flush=True)
    print('V30_CACHE_COMPLETE_DAYS', manifest['complete_days'], flush=True)
    print('V30_CACHE_ROWS', n, flush=True)
    print('V30_SNAPSHOT_DAY', snap['last_complete_day_et'], flush=True)

    # Only GET requests are network-reachable. The private checkpoint and lock
    # are disposable and cannot collide with the future Official writer.
    with tempfile.TemporaryDirectory(prefix='icaif_v30_get_only_') as temp:
        gate = GetOnlyTransport(httpx.HTTPTransport())
        with OriginalSession(profile=prof, token=token,
                             checkpoint=Path(temp) / 'checkpoint.json',
                             credentials=APP / '.icaif' / 'credentials.json',
                             transport=gate, timeout=18, result_timeout=10) as session:
            schedule = session.schedule()
            portfolio = session.portfolio('official')
            history = safe_history(session)
            ledger = session.ledger('official')
            if not isinstance(ledger, dict) or ledger.get('phase') != 'official':
                raise ValueError('UNVERIFIED_OFFICIAL_LEDGER')
            row = schedule.get('current_round')
            if not isinstance(row, dict) or row.get('phase') != 'official':
                print('V30_UNARMED_ACTION HOLD', flush=True)
                print('V30_UNARMED_REASON NO_OPEN_OFFICIAL_ROUND_OBJECT', flush=True)
            else:
                own = session.round(row['id'])
                if not isinstance(own, dict) or not isinstance(own.get('decisions'), list):
                    raise ValueError('UNVERIFIED_OWN_ROUND')
                print('V30_ALL_REAL_GET_ENDPOINTS_PASS', flush=True)
                print('V30_OFFICIAL_TOTAL_DECISIONS', len(history['decisions']), flush=True)
                print('V30_OFFICIAL_CURRENT_ROUND_DECISIONS', len(own['decisions']), flush=True)
                print('V30_LEDGER_ROUNDS_COUNT', len(ledger['rounds']) if isinstance(ledger.get('rounds'), list) else 'UNKNOWN', flush=True)
                # Nonpersistent PRE_ENTRY is valid for a *one-shot diagnostic only*
                # while no Official decisions or positions exist. If the account
                # has changed, default to HOLD; never invent trading state.
                if history['decisions'] or portfolio.get('positions') or portfolio.get('current_weights'):
                    print('V30_UNARMED_ACTION HOLD', flush=True)
                    print('V30_UNARMED_REASON NO_VERIFIED_PERSISTENT_STATE_REQUIRES_RECONCILIATION', flush=True)
                else:
                    # Types only, never balances or credential values.
                    print('V30_1_NAV_TYPE', type(portfolio.get('nav')).__name__, flush=True)
                    print('V30_1_CASH_TYPE', type(portfolio.get('cash')).__name__, flush=True)
                    result = bridge.offline_proposal(schedule, portfolio, history, own, snap,
                                                     engine.sample_state(), engine, local_time=datetime.now(timezone.utc))
                    print('V30_UNARMED_ACTION', result['action'], flush=True)
                    print('V30_UNARMED_REASON', result['note'], flush=True)
                    print('V30_WOULD_SUBMIT', str(result['would_submit']).upper(), flush=True)
                    if result['action'] == 'PRIMARY_TARGET':
                        print('V30_TARGET_SUM', format(result['target_sum'], '.12f'), flush=True)
                    # Defensive: the bridge MUST never claim real execution.
                    if result['would_submit'] or result['state_mutation']:
                        raise RuntimeError('V30_PURE_BRIDGE_CONTRACT_BROKEN')
            is_get_only = bool(gate.attempts) and all(x == 'GET' for x in gate.attempts)
            print('V30_HTTP_METHODS_GET_ONLY', str(is_get_only).upper(), flush=True)
            if not is_get_only:
                raise RuntimeError('NETWORK_METHOD_AUDIT_FAILED')
    print('V30_GET_ONLY_REALITY_CHECK_PASS_NO_POST_NO_SCHEDULER', flush=True)


def self_test():
    import organizer_bridge as bridge
    from decimal import Decimal
    assert bridge.as_decimal('1000000.00') == Decimal('1000000.00')
    assert bridge.as_decimal('0') == Decimal('0')
    assert bridge.as_decimal(Decimal('1000000.0')) == Decimal('1000000.0')
    for bad in ('nan','Infinity','+100',' 100','01','-1','1e400','1,000','1_000','',
                True,None,[],{}):
        try:bridge.as_decimal(bad)
        except bridge.ReadOnlyGate:pass
        else:raise AssertionError('Invalid numeric accepted: '+type(bad).__name__)
    print('V30_1_STRICT_NUMERIC_STRING_SELFTEST_PASS')
    class Req:
        method='POST'
    class Fake:
        def handle_request(self,req):raise AssertionError('POST escaped')
        def close(self):pass
    gate=GetOnlyTransport(Fake())
    try:gate.handle_request(Req())
    except RuntimeError as e:assert str(e)=='V30_NETWORK_WRITE_BLOCKED'
    else:raise AssertionError('Write transport not blocked')
    class Page:
        def __init__(self,pages):self.pages=pages
        def decisions(self, phase, *, offset,limit):return self.pages[offset]
    assert len(safe_history(Page({0:{'decisions':[],'has_more':False}}))['decisions'])==0
    try:safe_history(Page({0:{'decisions':[{}],'has_more':True}}))
    except ValueError:pass
    else:raise AssertionError('Ambiguous pagination accepted')
    for field in ('data_origin','schema','feed','adjustment','timeframe'):
        fake={'data_origin':'ALPACA_API_ONLY_NO_IMPORTED_SEED','schema':28,
           'feed':'iex','adjustment':'raw','timeframe':'15Min',
           'row_count':150000,'complete_days':170,'v23_strategy':'x',
           'last_complete_day_et':'2026-10-09','snapshot_lookback_sha256':'a'*64}
        bad=dict(fake);bad[field]='INVALID'
        try:validate_snapshot_meta({'purpose':'LIVE_READONLY_VERIFIED','strategy':'x','lookback_sha256':'a'*64,'last_complete_day_et':'2026-10-09'}, bad,Path('/nonexistent'))
        except ValueError:pass
        else:raise AssertionError('Bad provenance accepted '+field)
    print('V30_TRANSPORT_DENIES_POST_OFFLINE_PASS')
    print('V30_HISTORY_INCOMPLETE_PAGINATION_FAIL_CLOSED_PASS')
    print('V30_ALPACA_PROVENANCE_GATES_OFFLINE_PASS')
    print('V30_SELF_TEST_PASS_NO_NETWORK_NO_FILESYSTEM_WRITES')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--run-get-only', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.run_get_only:
        try:
            main()
        except Exception as exc:
            print('V30_FAIL_CLOSED', type(exc).__name__, flush=True)
            raise SystemExit(2)
    else:
        parser.error('Use --self-test or --run-get-only')
