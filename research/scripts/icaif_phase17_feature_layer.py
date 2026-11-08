#!/usr/bin/env python3
"""ICAIF Phase 17: deterministic research-only feature extraction.

Use from starter-kit working directory:
  python "$HOME\\Downloads\\icaif_phase17_feature_layer.py" --self-test
  python "$HOME\\Downloads\\icaif_phase17_feature_layer.py" --build
  python "$HOME\\Downloads\\icaif_phase17_feature_layer.py" --build --sec-facts

Reads existing Phase 16/connector snapshots; optional three SEC CompanyFacts
GETs. No organizer access, submission capability, trading service changes,
model training or claim of historical point-in-time reconstructability.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

UTC = timezone.utc
SYMBOLS = (
    "AAPL MSFT NVDA INTC CRM JPM BAC GS V PYPL LLY JNJ UNH PFE TMO "
    "AMZN TSLA WMT NKE KO CAT GE BA XOM CVX GOOGL META DIS T NEE"
).split()
NEWS_SOURCES = {"FINNHUB", "GDELT", "ALPHAVANTAGE"}
NEWS_POS = (
    "raises guidance", "raised guidance", "beats estimates", "beat estimates",
    "better than expected", "record revenue", "strong demand", "upgrades",
    "higher outlook", "profit jumps", "revenue grows", "dividend increase",
    "buyback", "share repurchase", "contract win",
)
NEWS_NEG = (
    "cuts guidance", "cut guidance", "misses estimates", "missed estimates",
    "worse than expected", "weak demand", "downgrades", "lower outlook",
    "profit warning", "revenue falls", "cuts outlook", "investigation", "recall",
    "data breach", "regulatory fine", "layoffs", "bankruptcy",
)
TAG_CHOICES = {
    "revenues_usd": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
    "net_income_usd": ("NetIncomeLoss",),
    "assets_usd": ("Assets",),
    "liabilities_usd": ("Liabilities",),
    "operating_cashflow_usd": ("NetCashProvidedByUsedInOperatingActivities",),
}
DURATION = {"revenues_usd", "net_income_usd", "operating_cashflow_usd"}


def dt(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.astimezone(UTC) if result.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def numeric(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def words(title):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(title).lower())).strip()


def classify(title):
    text = words(title)
    pos = sorted(k for k in NEWS_POS if k in text)
    neg = sorted(k for k in NEWS_NEG if k in text)
    direction = 1 if pos and not neg else -1 if neg and not pos else 0
    return direction, pos + neg


def valid_event(row, asof):
    if not isinstance(row, dict) or not row.get("event_id"):
        return False
    available = dt(row.get("available_at_utc"))
    fetched = dt(row.get("retrieved_utc"))
    published = dt(row.get("event_time_utc"))
    return bool(
        available is not None and fetched is not None
        and available >= fetched and available <= asof
        and (published is None or published <= available)
        and row.get("historical_backtest_eligible") is False
    )


def extract_features(report, asof):
    if report.get("schema_version") != 2:
        raise ValueError("UNSUPPORTED_PROBE_SCHEMA")
    if report.get("historical_backtest_permission", "").startswith("NO:") is False:
        raise ValueError("PROBE_NOT_MARKED_RESEARCH_ONLY")
    raw = report.get("events")
    if not isinstance(raw, list):
        raise ValueError("MISSING_EVENTS_LIST")

    events = [r for r in raw if valid_event(r, asof)]
    rejected = len(raw) - len(events)
    oldest = asof - timedelta(hours=72)
    statuses = report.get("sources") or []
    per_source = {}
    for s in statuses:
        if isinstance(s, dict):
            per_source[(s.get("provider"), s.get("symbol"))] = s.get("status")

    fred = {}
    for r in events:
        if r.get("source") != "FRED":
            continue
        d = r.get("data") or {}
        key = d.get("series")
        try:
            day = datetime.strptime(d.get("observation_date", ""), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        value = numeric(d.get("value"))
        if key in ("DGS10", "DFF") and value is not None and day <= asof.date():
            fred.setdefault(key, {})[day] = value
    macro = {}
    for key in ("DGS10", "DFF"):
        pairs = sorted(fred.get(key, {}).items(), reverse=True)
        macro[key + "_pct"] = pairs[0][1] if pairs else None
        macro[key + "_change_pct_points"] = pairs[0][1] - pairs[1][1] if len(pairs) >= 2 else None
        macro[key + "_observation_age_days"] = (asof.date() - pairs[0][0]).days if pairs else None
        # Critical: observation date does NOT establish intraday release time.
    macro["macro_release_time_verified"] = False

    rows = []
    for sym in SYMBOLS:
        sec7 = sec30 = k8 = k10q = k10k = 0
        news24 = news72 = pos24 = neg24 = neutral24 = 0
        sentiment_scores = []
        seen_titles = set()
        for r in events:
            if r.get("symbol") != sym:
                continue
            src = r.get("source")
            if src == "SEC":
                accepted = dt(r.get("event_time_utc"))
                if accepted is None:
                    continue
                age = asof - accepted
                if timedelta(0) <= age <= timedelta(days=30):
                    sec30 += 1
                    if age <= timedelta(days=7):
                        sec7 += 1
                        frm = (r.get("data") or {}).get("form", r.get("kind"))
                        if frm in ("8-K", "8-K/A"):
                            k8 += 1
                        elif frm == "10-Q":
                            k10q += 1
                        elif frm == "10-K":
                            k10k += 1
            elif src in NEWS_SOURCES:
                # Live proxy: require actual retrieval within 72 hours. An old
                # publication retrieved now is still not backtest eligible.
                observed = dt(r.get("event_time_utc")) or dt(r.get("retrieved_utc"))
                if observed is None or observed < oldest or observed > asof:
                    continue
                title = words((r.get("data") or {}).get("title"))
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                news72 += 1
                if asof - observed <= timedelta(hours=24):
                    news24 += 1
                    score, hits = classify(title)
                    pos24 += score > 0
                    neg24 += score < 0
                    neutral24 += score == 0
                    s = numeric((r.get("data") or {}).get("sentiment"))
                    if s is not None and -1 <= s <= 1 and src == "ALPHAVANTAGE":
                        sentiment_scores.append(s)
        have_news = any(
            status and str(status).startswith("CONNECTED")
            for (source, symbol), status in per_source.items()
            if source in NEWS_SOURCES and symbol == sym
        )
        have_sec = str(per_source.get(("SEC", sym), "")).startswith("CONNECTED")
        rows.append({
            "symbol": sym,
            "sec_connected": have_sec,
            "sec_accepted_7d": sec7 if have_sec else None,
            "sec_accepted_30d": sec30 if have_sec else None,
            "sec_8k_7d": k8 if have_sec else None,
            "sec_10q_7d": k10q if have_sec else None,
            "sec_10k_7d": k10k if have_sec else None,
            "news_connected": have_news,
            "news_unique_24h": news24 if have_news else None,
            "news_unique_72h": news72 if have_news else None,
            "news_positive_keyword_24h": pos24 if have_news else None,
            "news_negative_keyword_24h": neg24 if have_news else None,
            "news_neutral_keyword_24h": neutral24 if have_news else None,
            "alpha_sentiment_mean_24h": sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None,
            "keyword_rule_not_calibrated": True,
            "xbrl_metrics": None,
        })
    return rows, macro, len(events), rejected


def sec_user_agent(root):
    # Only read SEC contact, never load/print provider or organizer tokens.
    for path in (Path.home() / ".icaif" / "research_sources.env", root / ".env"):
        if path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.startswith("SEC_CONTACT_EMAIL="):
                    val = line.partition("=")[2].strip().strip('"\'')
                    if "@" in val and not any(c.isspace() for c in val) and len(val) < 180:
                        return "ICAIF2026-Research XBRL contact=" + val
    return None


def xbrl_facts(doc, fetched):
    result = {}
    facts = doc.get("facts", {}).get("us-gaap", {})
    if not isinstance(facts, dict):
        return result
    for metric, tags in TAG_CHOICES.items():
        observations = []
        for tag_priority, tag in enumerate(tags):
            items = facts.get(tag, {}).get("units", {}).get("USD", [])
            for item in items:
                if not isinstance(item, dict) or item.get("form") not in ("10-K", "10-Q"):
                    continue
                try:
                    filed = datetime.strptime(item["filed"], "%Y-%m-%d").date()
                    end = datetime.strptime(item["end"], "%Y-%m-%d").date()
                except (KeyError, TypeError, ValueError):
                    continue
                if filed > fetched.date() or end > filed:
                    continue
                val = numeric(item.get("val"))
                if val is None:
                    continue
                start = item.get("start")
                if metric in DURATION:
                    try:
                        interval_days = (end - datetime.strptime(start, "%Y-%m-%d").date()).days
                    except (TypeError, ValueError):
                        continue
                    # Quarterly *duration* only; mixing annual with quarterly
                    # revenues would create invalid cross-company comparisons.
                    if not 65 <= interval_days <= 115:
                        continue
                elif start:
                    continue
                observations.append((end, filed, -tag_priority, val, tag, item))
        if not observations:
            result[metric] = None
            continue
        end, filed, _, val, tag, item = max(observations, key=lambda x: x[:3])
        result[metric] = {
            "value": val,
            "period_end": end.isoformat(),
            "filing_date": filed.isoformat(),
            "taxonomy_tag": tag,
            "form": item.get("form"),
            "filing_accession": item.get("accn"),
            "available_at_utc": fetched.isoformat(),
            "historical_backtest_eligible": False,
        }
    return result


def fetch_company_facts(cik, user_agent):
    path = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    req = Request(path, headers={"User-Agent": user_agent, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=22) as resp:
            payload = resp.read(18_000_001)
    except HTTPError as exc:
        raise RuntimeError("SEC_HTTP_" + str(exc.code)) from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("SEC_NETWORK_ERROR") from None
    if len(payload) > 18_000_000:
        raise RuntimeError("SEC_XBRL_RESPONSE_TOO_LARGE")
    try:
        return json.loads(payload), datetime.now(UTC)
    except (ValueError, UnicodeDecodeError):
        raise RuntimeError("SEC_XBRL_BAD_JSON") from None


def add_xbrl(report, rows, root):
    user_agent = sec_user_agent(root)
    if not user_agent:
        print("SEC_FACTS: SKIPPED - SEC contact configuration missing")
        return
    for index, row in enumerate(rows):
        if row["symbol"] not in ("AAPL", "MSFT", "NVDA"):
            continue
        ciks = [int(r["data"]["cik"]) for r in report["events"]
                if r.get("source") == "SEC" and r.get("symbol") == row["symbol"]
                and isinstance(r.get("data"), dict) and r["data"].get("cik")]
        if not ciks:
            print("SEC_FACTS:", row["symbol"], "SKIPPED CIK_NOT_AVAILABLE")
            continue
        try:
            body, downloaded = fetch_company_facts(ciks[0], user_agent)
            row["xbrl_metrics"] = xbrl_facts(body, downloaded)
        except RuntimeError as exc:
            print("SEC_FACTS:", row["symbol"], "ERROR", str(exc))
        else:
            count = sum(v is not None for v in row["xbrl_metrics"].values())
            print("SEC_FACTS:", row["symbol"], "CONNECTED", f"metrics={count}")
        time.sleep(0.45)  # conservative SEC access rate


def selftest():
    when = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    def make(source, symbol, name, title=None, age_hours=1):
        return {
            "source": source, "symbol": symbol, "event_id": name,
            "event_time_utc": (when-timedelta(hours=age_hours)).isoformat(),
            "retrieved_utc": when.isoformat(),
            "available_at_utc": when.isoformat(),
            "historical_backtest_eligible": False,
            "data": {"title": title} if title is not None else {},
        }
    events = [make("FINNHUB", "AAPL", "n1", "Apple raises guidance"),
              make("GDELT", "AAPL", "n2", "Apple raises guidance"),
              make("FINNHUB", "AAPL", "n3", "Apple cuts guidance")]
    sec = make("SEC", "AAPL", "f1", age_hours=2)
    sec["data"] = {"form":"8-K", "cik": 320193}
    events.append(sec)
    report = {"schema_version":2, "historical_backtest_permission":"NO: current snapshots",
              "events":events,
              "sources":[{"provider":"FINNHUB","symbol":"AAPL","status":"CONNECTED_WITH_DATA"},
                         {"provider":"SEC","symbol":"AAPL","status":"CONNECTED_WITH_DATA"}]}
    rows,macro,n,rej = extract_features(report, when)
    a = rows[0]
    assert len(rows) == 30 and n == 4 and rej == 0
    assert a["news_unique_24h"] == 2 and a["news_positive_keyword_24h"] == 1
    assert a["news_negative_keyword_24h"] == 1 and a["sec_8k_7d"] == 1
    assert rows[1]["news_unique_24h"] is None  # no probe != no news
    future_rows, _, count, rejected = extract_features(report, when - timedelta(minutes=1))
    assert count == 0 and rejected == 4
    assert future_rows[0]["news_unique_24h"] is None or future_rows[0]["news_unique_24h"] == 0
    invalid = make("FINNHUB", "AAPL", "bad", "profit warning")
    invalid["available_at_utc"] = (when-timedelta(seconds=1)).isoformat()
    assert not valid_event(invalid, when)
    assert classify("Company raises guidance")[0] == 1
    assert classify("Company cuts guidance")[0] == -1
    assert classify("Company raises guidance but cuts outlook")[0] == 0
    doc = {"facts":{"us-gaap":{"NetIncomeLoss":{"units":{"USD":[
        {"form":"10-Q","filed":"2026-10-01","end":"2026-09-30","start":"2026-07-01","val":500},
        {"form":"10-K","filed":"2026-10-02","end":"2026-09-30","start":"2025-10-01","val":8000},
        {"form":"10-Q","filed":"2026-10-15","end":"2026-09-30","start":"2026-07-01","val":1000}
    ]}}}}}
    parsed = xbrl_facts(doc, when)
    assert parsed["net_income_usd"]["value"] == 500
    assert parsed["net_income_usd"]["historical_backtest_eligible"] is False
    print("PHASE17_SELF_TEST_PASS")
    print("NO_LOOKAHEAD_PASS")
    print("NEWS_DEDUPLICATION_PASS")
    print("MISSING_DATA_FLAGS_PASS")
    print("SEC_XBRL_QUARTERLY_FILTER_PASS")
    print("NO_COMPETITION_SUBMISSIONS")


def build(args):
    root = Path.cwd().resolve()
    if not (root / "tools" / "auto_submit.py").is_file():
        raise SystemExit("ERROR_OPEN_STARTER_KIT_AS_WORKING_FOLDER")
    storage = root / "artifacts" / "public_data_sources"
    available = sorted(storage.glob("probe_*.json"), key=lambda p:p.stat().st_mtime, reverse=True)
    if args.probe:
        source = Path(args.probe).resolve()
    elif available:
        source = available[0]
    else:
        raise SystemExit("ERROR_NO_PROVIDER_PROBE: run existing connector --connect first")
    if not source.is_file() or source.parent != storage.resolve():
        raise SystemExit("ERROR_PROBE_MUST_BE_IN_PROJECT_ARTIFACT_FOLDER")
    report = json.loads(source.read_text(encoding="utf-8"))
    asof = datetime.now(UTC)
    rows,macro,count,rejected = extract_features(report, asof)
    if args.sec_facts:
        add_xbrl(report, rows, root)
    produced_at = datetime.now(UTC)  # after any XBRL network fetch
    batch = {
        "schema_version":1, "research_only": True, "live_trading_enabled": False,
        "historical_backtest_eligible":False,
        "feature_available_at_utc":produced_at.isoformat(),
        "input_probe_name": source.name,
        "input_probe_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "valid_events":count,"rejected_events":rejected,
        "macro_data":macro,
        "macro_vintage_validated":False,
        "columns_semantics":"Counts and keyword labels are observational, NOT predictions.",
        "rows":rows,
    }
    stamp = produced_at.strftime("%Y%m%dT%H%M%SZ")
    out = storage / f"phase17_features_{stamp}.json"
    csv_out = storage / f"phase17_features_{stamp}.csv"
    out.write_text(json.dumps(batch, indent=2, allow_nan=False), encoding="utf-8")
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        cols = [key for key in rows[0] if key != "xbrl_metrics"]
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in cols})
    for f in (out, csv_out):
        try: os.chmod(f, 0o600)
        except OSError: pass
    print("=== ICAIF PHASE17 FEATURE LAYER ===")
    print("INPUT_PROBE:",source.name)
    print("VALID_SOURCE_EVENTS:",count)
    print("REJECTED_SOURCE_EVENTS:",rejected)
    print("STOCKS_WITH_NEWS_COVERAGE:",sum(r["news_connected"] for r in rows),"of 30")
    print("STOCKS_WITH_SEC_COVERAGE:",sum(r["sec_connected"] for r in rows),"of 30")
    print("MACRO_SNAPSHOT_AVAILABLE:",any(v is not None for k,v in macro.items() if k.endswith("_pct")))
    print("HISTORICAL_BACKTEST_ELIGIBLE: NO")
    print("MODEL_ALPHA_VALIDATED: NO")
    print("REPORT_JSON:",out)
    print("FEATURE_CSV:",csv_out)
    print("COMPETITION_SUBMISSION: NONE")
    print("PHASE17_COMPLETE_NO_TRADING")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--self-test", action="store_true")
    grp.add_argument("--build", action="store_true")
    ap.add_argument("--sec-facts", action="store_true", help="Optional 1 SEC CompanyFacts request each for AAPL/MSFT/NVDA only")
    ap.add_argument("--probe", help="Optional saved probe_*.json path inside artifacts/public_data_sources")
    args = ap.parse_args()
    if args.self_test:
        selftest()
    else:
        build(args)


if __name__ == "__main__":
    main()
