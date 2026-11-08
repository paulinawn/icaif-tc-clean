#!/usr/bin/env python3
"""ICAIF Phase 18: bounded, auditable all-30 SEC + Finnhub research data collection.

Run from starter-kit directory in your existing Windows virtual environment:
  python "$HOME\\Downloads\\icaif_phase18_collect_30_sources.py" --self-test
  python "$HOME\\Downloads\\icaif_phase18_collect_30_sources.py" --collect

No organizer/API submissions, no strategy changes, no scheduling, no cloud changes.
Provider keys loaded locally from ~/.icaif/research_sources.env (not printed).
Research snapshots fetched today are NOT automatically valid for historic backtests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
SYMBOLS = ("AAPL MSFT NVDA INTC CRM JPM BAC GS V PYPL LLY JNJ UNH PFE TMO "
           "AMZN TSLA WMT NKE KO CAT GE BA XOM CVX GOOGL META DIS T NEE").split()
TAGS = {
    "quarter_revenue_usd": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
    "quarter_net_income_usd": ("NetIncomeLoss",),
    "assets_usd": ("Assets",),
    "liabilities_usd": ("Liabilities",),
    "quarter_operating_cashflow_usd": ("NetCashProvidedByUsedInOperatingActivities",),
}
DURATION = {"quarter_revenue_usd", "quarter_net_income_usd", "quarter_operating_cashflow_usd"}
CONFIG = Path.home() / ".icaif" / "research_sources.env"
OUT_REL = Path("artifacts") / "public_data_sources" / "phase18_universe"
UA_BASE = "ICAIF2026-universe-research"


def utcnow():
    return datetime.now(UTC)


def iso(d):
    return d.astimezone(UTC).isoformat()


def parse_utc(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, UTC)
        except (OverflowError, ValueError, OSError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        d = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return d.astimezone(UTC) if d.tzinfo else None
    except ValueError:
        return None


def accession_time_utc(s):
    """SEC acceptance timestamps have exchange-local wall times (US Eastern)."""
    if not isinstance(s, str):
        return None
    try:
        d = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
        return d.astimezone(UTC) if d.tzinfo else d.replace(tzinfo=NY).astimezone(UTC)
    except ValueError:
        return None


def load_private_config(root):
    allow = {"SEC_CONTACT_EMAIL", "FINNHUB_API_KEY"}
    config = {}
    for path in (root / ".env", CONFIG):
        if path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip().strip("\"'")
                if key in allow and val and "\r" not in val and "\n" not in val:
                    config[key] = val
    for key in allow:
        if os.environ.get(key):
            config[key] = os.environ[key]
    return config


class SourceError(Exception):
    def __init__(self, code):
        assert re.fullmatch(r"[A-Z0-9_]+", code)
        self.code = code
        super().__init__(code)


def get_json(url, headers, *, max_bytes=20_000_000, timeout=22):
    """Caller supplies hardcoded provider URL, never print URL or HTTP response."""
    req = Request(url, headers={"User-Agent": UA_BASE, "Accept": "application/json", **headers})
    try:
        with urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise SourceError("HTTP_NOT_200")
            data = resp.read(max_bytes + 1)
    except HTTPError as e:
        raise SourceError("HTTP_" + str(e.code)) from None
    except (URLError, TimeoutError, OSError):
        raise SourceError("NETWORK_OR_TIMEOUT") from None
    if len(data) > max_bytes:
        raise SourceError("RESPONSE_SIZE_EXCEEDED")
    try:
        return json.loads(data), utcnow()
    except (UnicodeDecodeError, ValueError):
        raise SourceError("INVALID_JSON") from None


def cik_map(payload):
    vals = payload.values() if isinstance(payload, dict) else payload
    if not isinstance(vals, (list, tuple)) and not isinstance(payload, dict):
        raise SourceError("SEC_MAPPING_UNEXPECTED_SCHEMA")
    result = {}
    for r in vals:
        if not isinstance(r, dict):
            continue
        try:
            sym = str(r["ticker"]).upper().strip()
            cik = int(r["cik_str"])
        except (KeyError, TypeError, ValueError):
            continue
        if sym in SYMBOLS and sym not in result and cik > 0:
            result[sym] = cik
    return result


def extract_facts(document, retrieved):
    """Latest quarter duration or balance sheet instant, as observed today.

    Filed is NOT a precise publication timestamp; all values are tagged with
    the actual retrieval time and excluded from historical backtesting.
    """
    facts = document.get("facts", {}).get("us-gaap", {})
    out = {}
    for feature, candidates in TAGS.items():
        matches = []
        for priority, tag in enumerate(candidates):
            items = facts.get(tag, {}).get("units", {}).get("USD", [])
            for row in items:
                if not isinstance(row, dict) or row.get("form") not in ("10-Q", "10-K"):
                    continue
                try:
                    filed = datetime.strptime(row["filed"], "%Y-%m-%d").date()
                    end = datetime.strptime(row["end"], "%Y-%m-%d").date()
                    value = float(row["val"])
                except (TypeError, ValueError, KeyError, OverflowError):
                    continue
                if not math.isfinite(value) or end > filed or filed > retrieved.date():
                    continue
                if feature in DURATION:
                    try:
                        start = datetime.strptime(row["start"], "%Y-%m-%d").date()
                    except (TypeError, ValueError, KeyError):
                        continue
                    if not 65 <= (end - start).days <= 115:
                        continue
                elif row.get("start"):
                    continue
                matches.append((end, filed, -priority, value, tag, row.get("accn"), row.get("form")))
        if not matches:
            out[feature] = None
            continue
        end, filed, _, val, tag, accn, form = max(matches, key=lambda t: t[:3])
        out[feature] = {"value": val, "period_end": end.isoformat(),
                        "filing_date": filed.isoformat(), "taxonomy_tag": tag,
                        "form": form, "accession": accn,
                        "available_at_utc": iso(retrieved),
                        "historical_backtest_eligible": False}
    return out


def extract_filings(document, symbol, cik, retrieved, limit=30):
    recent = document.get("filings", {}).get("recent", {})
    rows = []
    for form, accn, t in zip(recent.get("form", []), recent.get("accessionNumber", []),
                             recent.get("acceptanceDateTime", [])):
        if form not in ("8-K", "8-K/A", "10-Q", "10-K", "6-K"):
            continue
        timestamp = accession_time_utc(t)
        if timestamp is None or timestamp > retrieved:
            continue
        rows.append({
            "source": "SEC", "symbol": symbol, "kind": form,
            "event_id": hashlib.sha256((str(cik) + ":" + str(accn)).encode()).hexdigest()[:32],
            "event_time_utc": iso(timestamp),
            "event_time_type": "filing_acceptance_eastern",
            "retrieved_utc": iso(retrieved), "available_at_utc": iso(retrieved),
            "historical_backtest_eligible": False,
        })
        if len(rows) >= limit:
            break
    return rows


def extract_news(document, symbol, retrieved, cap=100):
    if not isinstance(document, list):
        raise SourceError("FINNHUB_UNEXPECTED_SCHEMA")
    rows, seen = [], set()
    for r in document[: min(250, 3 * cap)]:
        if not isinstance(r, dict):
            continue
        timestamp = parse_utc(r.get("datetime"))
        if timestamp is None or timestamp > retrieved:
            continue
        headline = str(r.get("headline") or "").strip()[:300]
        if not headline:
            continue
        basis = str(r.get("id") or r.get("url") or (headline + iso(timestamp)))
        hashed = hashlib.sha256((symbol + ":" + basis).encode()).hexdigest()[:32]
        if hashed in seen:
            continue
        seen.add(hashed)
        rows.append({
            "source": "FINNHUB", "symbol": symbol, "kind": "headline",
            "event_id": hashed, "event_time_utc": iso(timestamp),
            "event_time_type": "provider_article_datetime",
            "retrieved_utc": iso(retrieved), "available_at_utc": iso(retrieved),
            "headline": headline, "publisher": str(r.get("source") or "")[:90],
            "company_relevance_verified": False,
            "historical_backtest_eligible": False,
        })
        if len(rows) >= cap:
            break
    return rows


def valid_snapshot(events):
    keys = set()
    errors = []
    for ev in events:
        stamp = parse_utc(ev.get("available_at_utc"))
        retrieval = parse_utc(ev.get("retrieved_utc"))
        published = parse_utc(ev.get("event_time_utc"))
        identity = (ev.get("source"), ev.get("symbol"), ev.get("event_id"))
        if not stamp or not retrieval or stamp < retrieval or (published and published > retrieval):
            errors.append("BAD_TIMESTAMP")
        if identity in keys:
            errors.append("DUPLICATE_EVENT_ID")
        keys.add(identity)
        if ev.get("historical_backtest_eligible") is not False:
            errors.append("INVALID_HISTORICAL_ELIGIBILITY")
    return errors


def self_test():
    assert len(SYMBOLS) == 30 and len(set(SYMBOLS)) == 30
    ref = datetime(2026, 10, 9, 12, tzinfo=UTC)
    mapping = cik_map({"0": {"ticker": "AAPL", "cik_str": 320193},
                       "1": {"ticker": "MSFT", "cik_str": 789019}})
    assert mapping["AAPL"] == 320193
    assert accession_time_utc("2026-10-09T08:30:00") == datetime(2026, 10, 9, 12, 30, tzinfo=UTC)
    mock = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"form": "10-Q", "filed": "2026-08-02", "start": "2026-04-01",
             "end": "2026-06-30", "val": 1000000},
            {"form": "10-K", "filed": "2026-02-02", "start": "2025-01-01",
             "end": "2025-12-31", "val": 5000000},
        ]}}
    }}}
    facts = extract_facts(mock, ref)
    assert facts["quarter_revenue_usd"]["value"] == 1000000
    assert facts["quarter_revenue_usd"]["historical_backtest_eligible"] is False
    filings = extract_filings({"filings": {"recent": {
        "form": ["8-K", "10-Q"], "accessionNumber": ["aa", "bb"],
        "acceptanceDateTime": ["2026-10-08T16:45:00", "2026-10-11T16:45:00"]
    }}}, "AAPL", 320193, ref)
    assert len(filings) == 1
    news = extract_news([
        {"id": 1, "headline": "Company beats estimates", "datetime": 1791545400, "source": "Test"},
        {"id": 1, "headline": "Duplicate", "datetime": 1791545400, "source": "Test"},
        {"id": 2, "headline": "Future", "datetime": 1891545400, "source": "Test"},
    ], "AAPL", ref)
    assert len(news) == 1 and not valid_snapshot(news)
    assert valid_snapshot(news + news) == ["DUPLICATE_EVENT_ID"]
    # Ensure only explicitly allowlisted research keys are loaded.
    assert len(set(["SEC_CONTACT_EMAIL", "FINNHUB_API_KEY"])) == 2
    print("PHASE18_SELF_TEST_PASS")
    print("SEC_CIK_XBRL_TIME_GUARDS_PASS")
    print("FINNHUB_DEDUP_AND_FUTURE_GUARDS_PASS")
    print("NO_NETWORK_NO_TRADES")


def provider_fetch_sec(symbol, cik, ua):
    hdr = {"User-Agent": ua}
    company, cf_at = get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", hdr)
    time.sleep(0.55)
    filings, f_at = get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", hdr,
                             max_bytes=8_000_000)
    return extract_facts(company, cf_at), extract_filings(filings, symbol, cik, f_at)


def collect(args):
    root = Path.cwd().resolve()
    if not (root / "tools" / "auto_submit.py").is_file():
        raise SystemExit("ERROR: Run from your starter-kit folder (VS Code terminal).")
    selected = SYMBOLS[:args.limit]
    conf = load_private_config(root)
    sec_email = conf.get("SEC_CONTACT_EMAIL", "")
    finnhub_key = conf.get("FINNHUB_API_KEY", "")
    if not ("@" in sec_email and not any(c.isspace() for c in sec_email)):
        raise SystemExit("ERROR: SEC contact email missing from your private research config.")
    if not finnhub_key:
        raise SystemExit("ERROR: Finnhub key missing from your private research config.")

    folder = root / OUT_REL
    folder.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    ident = started.strftime("%Y%m%dT%H%M%SZ")
    email_safe = sec_email[:180]
    sec_ua = f"{UA_BASE} contact={email_safe}"

    print("=== PHASE18 30-STOCK RESEARCH INGESTION ===", flush=True)
    print(f"STOCKS_REQUESTED: {len(selected)}", flush=True)
    print(f"MAX_REQUESTS: SEC={1 + 2 * len(selected)}, FINNHUB={len(selected)}", flush=True)
    print("NETWORK_ACCESS: SEC + Finnhub only; no organizer calls", flush=True)
    print("HISTORICAL_BACKTEST_ELIGIBLE: NO", flush=True)
    print("Keys/contact values: not printed or stored in output", flush=True)

    status = {sym: {"symbol": sym, "sec": "NOT_ATTEMPTED", "sec_filings": 0,
                    "sec_facts": 0, "finnhub": "NOT_ATTEMPTED", "finnhub_news": 0}
              for sym in selected}
    events, financials = [], []
    sec_paused, finnhub_paused = False, False

    try:
        mapping_payload, _ = get_json("https://www.sec.gov/files/company_tickers.json",
                                      {"User-Agent": sec_ua}, max_bytes=2_000_000)
        mapping = cik_map(mapping_payload)
    except SourceError as e:
        mapping = {}
        sec_paused = True
        print("SEC_MAPPING_ERROR:", e.code, flush=True)
    time.sleep(0.6)

    for idx, sym in enumerate(selected, 1):
        if sec_paused:
            status[sym]["sec"] = "SKIPPED_SOURCE_UNAVAILABLE"
        elif sym not in mapping:
            status[sym]["sec"] = "UNMAPPED_CIK"
        else:
            try:
                facts, filings = provider_fetch_sec(sym, mapping[sym], sec_ua)
                financials.append({"symbol": sym, "source": "SEC_XBRL", "cik": mapping[sym],
                                   "metrics": facts, "retrieval_is_historical_backtest_eligible": False})
                events.extend(filings)
                count = sum(x is not None for x in facts.values())
                status[sym].update(sec="OK", sec_filings=len(filings), sec_facts=count)
            except SourceError as e:
                status[sym]["sec"] = e.code
                if e.code == "HTTP_429":
                    sec_paused = True
            except (ValueError, TypeError, KeyError):
                status[sym]["sec"] = "SEC_UNEXPECTED_FORMAT"
            time.sleep(0.65)

        if finnhub_paused:
            status[sym]["finnhub"] = "SKIPPED_RATE_LIMITED"
        else:
            dt = utcnow()
            since = (dt.date() - timedelta(days=args.news_days)).isoformat()
            to = dt.date().isoformat()
            # Header authorization, never add API keys to URLs or logs.
            from urllib.parse import urlencode
            url = "https://finnhub.io/api/v1/company-news?" + urlencode({
                "symbol": sym, "from": since, "to": to,
            })
            try:
                article_payload, received = get_json(url, {"X-Finnhub-Token": finnhub_key},
                                                     max_bytes=3_000_000)
                news = extract_news(article_payload, sym, received, cap=100)
                events.extend(news)
                status[sym]["finnhub"] = "OK" if news else "OK_EMPTY"
                status[sym]["finnhub_news"] = len(news)
            except SourceError as e:
                status[sym]["finnhub"] = e.code
                if e.code == "HTTP_429":
                    finnhub_paused = True
            time.sleep(1.30)
        print(f"{idx:2d}/{len(selected):2d} {sym:5s} SEC={status[sym]['sec']:26s} "
              f"facts={status[sym]['sec_facts']} filings={status[sym]['sec_filings']:2d} "
              f"FINNHUB={status[sym]['finnhub']:22s} news={status[sym]['finnhub_news']:3d}",
              flush=True)

    integrity = valid_snapshot(events)
    # Do not save invalid point-in-time snapshots.
    if integrity:
        print("FATAL_INTEGRITY_FAILURE", ",".join(sorted(set(integrity))))
        raise SystemExit(2)

    report = {
        "schema_version": 1,
        "purpose": "research_snapshot_only_no_trading",
        "collected_from_utc": iso(started),
        "collected_until_utc": iso(utcnow()),
        "selected_universe": selected,
        "providers": ["SEC", "FINNHUB"],
        "historical_backtest_eligible": False,
        "history_limitation": "Publication timestamp does not prove historical feed availability. All data first observed during this run.",
        "symbol_status": [status[sym] for sym in selected],
        "events": events,
        "financials": financials,
    }
    outfile = folder / ("phase18_snapshot_" + ident + ".json")
    with outfile.open("x", encoding="utf-8") as h:
        json.dump(report, h, indent=2, ensure_ascii=False)
    try:
        os.chmod(outfile, 0o600)
    except OSError:
        pass
    table = folder / ("phase18_coverage_" + ident + ".csv")
    with table.open("x", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=["symbol", "sec", "sec_filings", "sec_facts",
                                               "finnhub", "finnhub_news"])
        writer.writeheader()
        writer.writerows(status[sym] for sym in selected)

    print("=== PHASE18 DATA QUALITY SUMMARY ===")
    print("SEC_COVERAGE:", sum(x["sec"] == "OK" for x in status.values()), "/", len(selected))
    print("FINNHUB_COVERAGE:", sum(x["finnhub"] in ("OK", "OK_EMPTY") for x in status.values()), "/", len(selected))
    print("NEWS_EVENTS:", sum(x["source"] == "FINNHUB" for x in events))
    print("SEC_FILINGS:", sum(x["source"] == "SEC" for x in events))
    print("INTEGRITY_ERRORS:", len(integrity))
    print("POINT_IN_TIME_BACKTEST_ELIGIBLE: NO")
    print("RESEARCH_SNAPSHOT:", outfile)
    print("COVERAGE_CSV:", table)
    print("TRADING_IMPACT: NONE")
    print("PHASE18_COLLECT_COMPLETE_NO_SUBMISSION")


def main():
    a = argparse.ArgumentParser(description=__doc__)
    commands = a.add_mutually_exclusive_group(required=True)
    commands.add_argument("--self-test", action="store_true")
    commands.add_argument("--collect", action="store_true")
    a.add_argument("--limit", type=int, default=30, help="1-30 symbols in announced order")
    a.add_argument("--news-days", type=int, default=4, help="1-7 calendar days of Finnhub news")
    args = a.parse_args()
    if not 1 <= args.limit <= 30 or not 1 <= args.news_days <= 7:
        a.error("--limit must be 1..30 and --news-days must be 1..7")
    if args.self_test:
        self_test()
    else:
        collect(args)


if __name__ == "__main__":
    main()
