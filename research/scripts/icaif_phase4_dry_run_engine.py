from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()


def run(*args: str) -> None:
    print(">", " ".join(args))
    cp = subprocess.run(args, cwd=ROOT, text=True)
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)


def output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def fail(message: str) -> None:
    raise SystemExit(f"\nERROR: {message}\n")


required = [
    ROOT / "universe.json",
    ROOT / "our_agent" / "data" / "alpaca.py",
    ROOT / "tests",
    ROOT / ".gitignore",
]

missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    fail(
        "Run this from the ICAIF starter-kit folder. "
        f"Missing: {missing}"
    )

if output("git", "branch", "--show-current") != "our-agent":
    fail("Expected Git branch 'our-agent'.")

if subprocess.run(
    ["git", "check-ignore", "-q", ".env"],
    cwd=ROOT,
).returncode != 0:
    fail(".env must remain Git-ignored.")

print("=== ICAIF PHASE 4: DATA + FEATURE + DRY-RUN ENGINE ===")
print("Competition submission remains DISABLED.")
print()



files = {'our_agent/data/canonical.py': '"""Canonicalize and validate Alpaca OHLCV bars for research/live signals."""\n\nfrom __future__ import annotations\n\nfrom datetime import time\nfrom zoneinfo import ZoneInfo\n\nimport pandas as pd\n\nfrom our_agent.config import OFFICIAL_SYMBOLS\n\nNY = ZoneInfo("America/New_York")\n\n\nclass MarketDataValidationError(ValueError):\n    """Raised when market-data input is incomplete or malformed."""\n\n\ndef bars_payload_to_frame(payload: dict) -> pd.DataFrame:\n    bars = payload.get("bars")\n    if not isinstance(bars, dict):\n        raise MarketDataValidationError("Payload has no bars mapping.")\n\n    records: list[dict] = []\n\n    for symbol in OFFICIAL_SYMBOLS:\n        rows = bars.get(symbol)\n        if not isinstance(rows, list):\n            raise MarketDataValidationError(\n                f"{symbol}: missing list of bars."\n            )\n\n        for row in rows:\n            if not isinstance(row, dict):\n                raise MarketDataValidationError(\n                    f"{symbol}: malformed bar row."\n                )\n\n            records.append(\n                {\n                    "symbol": symbol,\n                    "timestamp": row.get("t"),\n                    "open": row.get("o"),\n                    "high": row.get("h"),\n                    "low": row.get("l"),\n                    "close": row.get("c"),\n                    "volume": row.get("v"),\n                    "trade_count": row.get("n"),\n                    "vwap": row.get("vw"),\n                }\n            )\n\n    if not records:\n        raise MarketDataValidationError("No bars were returned.")\n\n    frame = pd.DataFrame.from_records(records)\n\n    frame["timestamp"] = pd.to_datetime(\n        frame["timestamp"],\n        utc=True,\n        errors="raise",\n    )\n\n    numeric_cols = [\n        "open",\n        "high",\n        "low",\n        "close",\n        "volume",\n        "trade_count",\n        "vwap",\n    ]\n\n    for column in numeric_cols:\n        frame[column] = pd.to_numeric(\n            frame[column],\n            errors="coerce",\n        )\n\n    required_price_cols = ["open", "high", "low", "close"]\n\n    if frame[required_price_cols].isna().any().any():\n        raise MarketDataValidationError(\n            "OHLC contains non-numeric or missing values."\n        )\n\n    if (frame[required_price_cols] <= 0).any().any():\n        raise MarketDataValidationError(\n            "OHLC prices must be positive."\n        )\n\n    if frame["volume"].isna().any() or (frame["volume"] < 0).any():\n        raise MarketDataValidationError(\n            "Volume must be finite and non-negative."\n        )\n\n    duplicates = frame.duplicated(\n        subset=["symbol", "timestamp"],\n        keep=False,\n    )\n\n    if duplicates.any():\n        sample = frame.loc[\n            duplicates,\n            ["symbol", "timestamp"],\n        ].head(10)\n        raise MarketDataValidationError(\n            "Duplicate symbol/timestamp bars found: "\n            + sample.to_dict(orient="records").__repr__()\n        )\n\n    actual_symbols = set(frame["symbol"])\n    expected_symbols = set(OFFICIAL_SYMBOLS)\n\n    if actual_symbols != expected_symbols:\n        raise MarketDataValidationError(\n            f"Symbol mismatch. Missing="\n            f"{sorted(expected_symbols - actual_symbols)}"\n        )\n\n    return frame.sort_values(\n        ["timestamp", "symbol"],\n        kind="stable",\n    ).reset_index(drop=True)\n\n\ndef regular_session_only(frame: pd.DataFrame) -> pd.DataFrame:\n    """Keep 09:30 <= timestamp < 16:00 America/New_York on weekdays."""\n\n    local = frame["timestamp"].dt.tz_convert(NY)\n\n    mask = (\n        (local.dt.weekday < 5)\n        & (\n            (local.dt.time >= time(9, 30))\n            & (local.dt.time < time(16, 0))\n        )\n    )\n\n    result = frame.loc[mask].copy()\n\n    if result.empty:\n        raise MarketDataValidationError(\n            "No regular-session bars remain after filtering."\n        )\n\n    return result.reset_index(drop=True)\n\n\ndef common_information_cutoff(frame: pd.DataFrame) -> pd.Timestamp:\n    """Latest timestamp available for every competition symbol."""\n\n    latest_by_symbol = frame.groupby("symbol")["timestamp"].max()\n\n    missing = sorted(\n        set(OFFICIAL_SYMBOLS) - set(latest_by_symbol.index)\n    )\n    if missing:\n        raise MarketDataValidationError(\n            f"Missing symbols before cutoff calculation: {missing}"\n        )\n\n    return latest_by_symbol.min()\n\n\ndef trim_to_common_cutoff(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:\n    cutoff = common_information_cutoff(frame)\n    trimmed = frame.loc[frame["timestamp"] <= cutoff].copy()\n\n    if set(trimmed["symbol"]) != set(OFFICIAL_SYMBOLS):\n        raise MarketDataValidationError(\n            "Common-cutoff trimming removed one or more symbols."\n        )\n\n    return trimmed.reset_index(drop=True), cutoff\n\n\ndef coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:\n    summary = (\n        frame.groupby("symbol")\n        .agg(\n            bars=("timestamp", "size"),\n            first_timestamp=("timestamp", "min"),\n            last_timestamp=("timestamp", "max"),\n            last_close=("close", "last"),\n        )\n        .reindex(OFFICIAL_SYMBOLS)\n    )\n\n    return summary\n', 'our_agent/features/core.py': '"""Transparent candidate feature set for ICAIF plumbing tests.\n\nThis is an initial hypothesis, not a selected competition model.\nThe organizer historical dataset will be used for actual model selection.\n"""\n\nfrom __future__ import annotations\n\nimport numpy as np\nimport pandas as pd\n\nfrom our_agent.config import OFFICIAL_SYMBOLS, SECTORS\nfrom our_agent.data.canonical import (\n    MarketDataValidationError,\n    trim_to_common_cutoff,\n)\n\nBARS_PER_REGULAR_DAY_15M = 26\nLOOKBACK_1H = 4\nLOOKBACK_1D = BARS_PER_REGULAR_DAY_15M\nLOOKBACK_5D = 5 * BARS_PER_REGULAR_DAY_15M\nMIN_BARS = LOOKBACK_5D + 2\n\n\ndef _cross_section_rank_signal(series: pd.Series) -> pd.Series:\n    """Map cross-sectional ranks approximately to [-1, +1]."""\n    return 2.0 * series.rank(\n        method="average",\n        pct=True,\n    ) - 1.0\n\n\ndef compute_candidate_features(\n    frame: pd.DataFrame,\n) -> tuple[pd.DataFrame, pd.Timestamp]:\n    """Compute point-in-time features using one common information cutoff."""\n\n    trimmed, cutoff = trim_to_common_cutoff(frame)\n\n    counts = trimmed.groupby("symbol").size()\n    too_short = counts[counts < MIN_BARS]\n\n    if not too_short.empty:\n        raise MarketDataValidationError(\n            "Insufficient 15-minute regular-session history: "\n            + too_short.to_dict().__repr__()\n        )\n\n    rows: list[dict] = []\n\n    symbol_to_sector = {\n        symbol: sector\n        for sector, symbols in SECTORS.items()\n        for symbol in symbols\n    }\n\n    for symbol in OFFICIAL_SYMBOLS:\n        data = (\n            trimmed.loc[trimmed["symbol"] == symbol]\n            .sort_values("timestamp")\n            .reset_index(drop=True)\n        )\n\n        close = data["close"].astype(float)\n        one_bar_returns = close.pct_change(fill_method=None)\n\n        last_close = float(close.iloc[-1])\n        ret_1h = last_close / float(close.iloc[-1 - LOOKBACK_1H]) - 1.0\n        ret_1d = last_close / float(close.iloc[-1 - LOOKBACK_1D]) - 1.0\n        ret_5d = last_close / float(close.iloc[-1 - LOOKBACK_5D]) - 1.0\n\n        vol_1d = float(\n            one_bar_returns.tail(LOOKBACK_1D).std(ddof=1)\n        )\n\n        rows.append(\n            {\n                "symbol": symbol,\n                "sector": symbol_to_sector[symbol],\n                "asof": cutoff,\n                "close": last_close,\n                "ret_1h": ret_1h,\n                "ret_1d": ret_1d,\n                "ret_5d": ret_5d,\n                "vol_1d": vol_1d,\n            }\n        )\n\n    features = pd.DataFrame(rows).set_index("symbol")\n\n    if not np.isfinite(\n        features[["close", "ret_1h", "ret_1d", "ret_5d", "vol_1d"]]\n        .to_numpy(dtype=float)\n    ).all():\n        raise MarketDataValidationError(\n            "Candidate feature matrix contains non-finite values."\n        )\n\n    sector_mean_5d = features.groupby("sector")["ret_5d"].transform("mean")\n    features["sector_relative_5d"] = (\n        features["ret_5d"] - sector_mean_5d\n    )\n\n    features["momentum_rank"] = _cross_section_rank_signal(\n        features["ret_5d"]\n    )\n\n    features["reversal_rank"] = -_cross_section_rank_signal(\n        features["ret_1h"]\n    )\n\n    features["sector_relative_rank"] = _cross_section_rank_signal(\n        features["sector_relative_5d"]\n    )\n\n    # Hypothesis only. These coefficients are deliberately simple and\n    # must be validated on organizer data before being treated as strategy.\n    features["candidate_score"] = (\n        0.45 * features["momentum_rank"]\n        + 0.30 * features["reversal_rank"]\n        + 0.25 * features["sector_relative_rank"]\n    )\n\n    return (\n        features.sort_values(\n            "candidate_score",\n            ascending=False,\n        ),\n        cutoff,\n    )\n', 'our_agent/strategy/candidate_v0.py': '"""Candidate v0 decision engine.\n\nPurpose:\n- prove the end-to-end quantitative pipeline;\n- remain transparent and auditable;\n- produce NON-SUBMITTABLE dry-run weights.\n\nThis is not yet the selected Validation strategy.\n"""\n\nfrom __future__ import annotations\n\nimport pandas as pd\n\nfrom our_agent.config import OFFICIAL_SYMBOLS\nfrom our_agent.strategy.baselines import top_k_equal\nfrom our_agent.features.core import compute_candidate_features\n\n\nCANDIDATE_NAME = "candidate_v0_plumbing"\nTOP_K = 8\nGROSS_EXPOSURE = 0.80\n\n\ndef build_candidate_weights(\n    frame: pd.DataFrame,\n) -> tuple[dict[str, float], pd.DataFrame, pd.Timestamp]:\n    features, cutoff = compute_candidate_features(frame)\n\n    scores = {\n        symbol: float(features.loc[symbol, "candidate_score"])\n        for symbol in OFFICIAL_SYMBOLS\n    }\n\n    weights = top_k_equal(\n        scores,\n        k=TOP_K,\n        gross_exposure=GROSS_EXPOSURE,\n    )\n\n    return weights, features, cutoff\n', 'tools/dry_run_agent.py': '"""Fetch recent Alpaca IEX data and build a non-submittable candidate.\n\nThis tool never reads ICAIF team credentials and never calls a competition\nsubmission endpoint.\n"""\n\nfrom __future__ import annotations\n\nimport json\nfrom datetime import datetime, timedelta, timezone\nfrom pathlib import Path\n\nfrom our_agent.config import OFFICIAL_SYMBOLS\nfrom our_agent.data.alpaca import AlpacaMarketDataClient\nfrom our_agent.data.canonical import (\n    bars_payload_to_frame,\n    coverage_summary,\n    regular_session_only,\n)\nfrom our_agent.strategy.candidate_v0 import (\n    CANDIDATE_NAME,\n    build_candidate_weights,\n)\n\nROOT = Path(__file__).resolve().parents[1]\nCACHE_DIR = ROOT / "data" / "cache"\nARTIFACT_DIR = ROOT / "artifacts"\n\nTIMEFRAME = "15Min"\nLOOKBACK_DAYS = 90\n\n\ndef main() -> None:\n    now = datetime.now(timezone.utc)\n\n    # Basic historical access may exclude the most recent 15 minutes.\n    # End 20 minutes back so this research/plumbing fetch is deterministic.\n    end = now - timedelta(minutes=20)\n    start = end - timedelta(days=LOOKBACK_DAYS)\n\n    print("=== ICAIF CANDIDATE V0 DRY RUN ===")\n    print("Competition submission: DISABLED")\n    print("Data source: Alpaca")\n    print("Feed: IEX")\n    print(f"Historical timeframe: {TIMEFRAME}")\n    print(f"Requested window: {start.isoformat()} -> {end.isoformat()}")\n\n    with AlpacaMarketDataClient(feed="iex") as client:\n        payload = client.historical_bars(\n            symbols=OFFICIAL_SYMBOLS,\n            start=start.isoformat(),\n            end=end.isoformat(),\n            timeframe=TIMEFRAME,\n            adjustment="all",\n            limit=10_000,\n        )\n\n    frame = bars_payload_to_frame(payload)\n    frame = regular_session_only(frame)\n\n    CACHE_DIR.mkdir(parents=True, exist_ok=True)\n    cache_path = CACHE_DIR / "alpaca_iex_15min_recent.csv"\n    frame.to_csv(cache_path, index=False)\n\n    coverage = coverage_summary(frame)\n\n    weights, features, cutoff = build_candidate_weights(frame)\n\n    selected = [\n        symbol\n        for symbol, weight in weights.items()\n        if weight > 0\n    ]\n\n    top_features = (\n        features.loc[\n            selected,\n            [\n                "sector",\n                "ret_1h",\n                "ret_1d",\n                "ret_5d",\n                "vol_1d",\n                "sector_relative_5d",\n                "candidate_score",\n            ],\n        ]\n        .sort_values("candidate_score", ascending=False)\n        .reset_index()\n        .to_dict(orient="records")\n    )\n\n    artifact = {\n        "artifact_type": "DRY_RUN_ONLY_NOT_A_COMPETITION_SUBMISSION",\n        "candidate_name": CANDIDATE_NAME,\n        "generated_at_utc": now.isoformat(),\n        "data": {\n            "source": "alpaca",\n            "feed": "iex",\n            "timeframe": TIMEFRAME,\n            "adjustment": "all",\n            "requested_start": start.isoformat(),\n            "requested_end": end.isoformat(),\n            "common_information_cutoff": cutoff.isoformat(),\n            "cache_file": str(cache_path),\n        },\n        "portfolio": {\n            "gross_exposure": sum(weights.values()),\n            "selected_symbols": selected,\n            "weights": weights,\n        },\n        "top_features": top_features,\n        "notes": [\n            "Candidate coefficients are unvalidated hypotheses.",\n            "This artifact is intentionally not valid ICAIF decision.json format.",\n            "Do not submit this file to Codabench.",\n        ],\n    }\n\n    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)\n    artifact_path = ARTIFACT_DIR / "candidate_v0_dry_run.json"\n    artifact_path.write_text(\n        json.dumps(artifact, indent=2),\n        encoding="utf-8",\n    )\n\n    print()\n    print("=== DATA COVERAGE ===")\n    print(\n        f"Bars per symbol: min={int(coverage[\'bars\'].min())}, "\n        f"median={int(coverage[\'bars\'].median())}, "\n        f"max={int(coverage[\'bars\'].max())}"\n    )\n    print(\n        "Latest regular-session timestamp range:",\n        coverage["last_timestamp"].min(),\n        "to",\n        coverage["last_timestamp"].max(),\n    )\n    print("Common information cutoff:", cutoff)\n\n    print()\n    print("=== DRY-RUN PORTFOLIO ===")\n    print(f"Candidate: {CANDIDATE_NAME}")\n    print(f"Selected names: {len(selected)}")\n    print(f"Gross exposure: {sum(weights.values()):.4f}")\n\n    for symbol in selected:\n        print(\n            f"  {symbol}: weight={weights[symbol]:.4f}, "\n            f"score={features.loc[symbol, \'candidate_score\']:.4f}"\n        )\n\n    print()\n    print(f"Cache:    {cache_path}")\n    print(f"Artifact: {artifact_path}")\n    print()\n    print("DRY RUN COMPLETE. NO COMPETITION SUBMISSION WAS MADE.")\n\n\nif __name__ == "__main__":\n    main()\n', 'tests/test_features.py': 'import unittest\n\nimport numpy as np\nimport pandas as pd\n\nfrom our_agent.config import OFFICIAL_SYMBOLS\nfrom our_agent.features.core import compute_candidate_features\n\n\ndef synthetic_frame() -> pd.DataFrame:\n    timestamps = pd.date_range(\n        "2026-01-05 14:30:00+00:00",\n        periods=170,\n        freq="15min",\n    )\n\n    rows = []\n\n    for symbol_index, symbol in enumerate(OFFICIAL_SYMBOLS):\n        base = 100.0 + symbol_index\n        trend = 0.0002 * (symbol_index + 1)\n\n        for i, ts in enumerate(timestamps):\n            price = base * (1.0 + trend * i)\n            rows.append(\n                {\n                    "symbol": symbol,\n                    "timestamp": ts,\n                    "open": price * 0.999,\n                    "high": price * 1.001,\n                    "low": price * 0.998,\n                    "close": price,\n                    "volume": 1000 + i,\n                    "trade_count": 10,\n                    "vwap": price,\n                }\n            )\n\n    return pd.DataFrame(rows)\n\n\nclass CandidateFeatureTests(unittest.TestCase):\n    def test_feature_matrix_has_all_symbols(self):\n        features, cutoff = compute_candidate_features(\n            synthetic_frame()\n        )\n        self.assertEqual(set(features.index), set(OFFICIAL_SYMBOLS))\n        self.assertEqual(len(features), 30)\n        self.assertTrue(\n            np.isfinite(\n                features[\n                    [\n                        "ret_1h",\n                        "ret_1d",\n                        "ret_5d",\n                        "vol_1d",\n                        "candidate_score",\n                    ]\n                ].to_numpy(dtype=float)\n            ).all()\n        )\n\n\nif __name__ == "__main__":\n    unittest.main()\n', 'tests/test_candidate_v0.py': 'import unittest\n\nfrom our_agent.strategy.candidate_v0 import (\n    GROSS_EXPOSURE,\n    TOP_K,\n    build_candidate_weights,\n)\nfrom test_features import synthetic_frame\n\n\nclass CandidateV0Tests(unittest.TestCase):\n    def test_candidate_weights_are_safe(self):\n        weights, features, cutoff = build_candidate_weights(\n            synthetic_frame()\n        )\n\n        positive = [w for w in weights.values() if w > 0]\n\n        self.assertEqual(len(positive), TOP_K)\n        self.assertAlmostEqual(sum(weights.values()), GROSS_EXPOSURE)\n        self.assertLessEqual(max(weights.values()), 0.30)\n        self.assertTrue(all(w >= 0 for w in weights.values()))\n\n\nif __name__ == "__main__":\n    unittest.main()\n'}

for filename, content in files.items():
    path = ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote: {filename}")

print()
print("=== COMPLETE LOCAL TEST SUITE ===")
run(
    sys.executable,
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-v",
)

print()
print("=== STATIC CHECKS ===")
run(
    sys.executable,
    "-m",
    "compileall",
    "-q",
    "our_agent",
    "tests",
    "tools",
)
run("git", "diff", "--check")

print()
print("=== LIVE DATA / DRY-RUN TEST ===")
run(sys.executable, "tools/dry_run_agent.py")

changelog = ROOT / "CHANGELOG.md"
title = "## 2026-10-07 — Candidate v0 dry-run pipeline"

text = changelog.read_text(encoding="utf-8")

if title not in text:
    with changelog.open("a", encoding="utf-8") as f:
        f.write(
            "\n"
            + title
            + "\n\n"
            "- Added canonical Alpaca OHLCV validation and regular-session filtering.\n"
            "- Added a common-information-cutoff guard against cross-sectional look-ahead.\n"
            "- Added transparent momentum/reversal/sector-relative candidate features.\n"
            "- Added candidate v0 Top-8 equal-weight dry-run portfolio at 80% gross exposure.\n"
            "- Added recent Alpaca IEX history capture for plumbing tests only.\n"
            "- Candidate parameters remain unvalidated pending organizer historical data.\n"
            "- Dry-run artifact is intentionally not valid competition submission format.\n"
            "- No competition POST/submission logic was enabled.\n"
        )

print()
print("=== GIT CHECKPOINT ===")
run("git", "check-ignore", "-q", ".env")
run("git", "diff", "--check")

run(
    "git",
    "add",
    "CHANGELOG.md",
    "our_agent/data/canonical.py",
    "our_agent/features/core.py",
    "our_agent/strategy/candidate_v0.py",
    "tools/dry_run_agent.py",
    "tests/test_features.py",
    "tests/test_candidate_v0.py",
)

staged = output("git", "diff", "--cached", "--name-only")

if staged:
    run(
        "git",
        "commit",
        "-m",
        "Add dry-run candidate data and feature pipeline",
    )
else:
    print("No source changes needed committing.")

print()
print("=== FINAL STATUS ===")
print(f"Branch: {output('git', 'branch', '--show-current')}")
print(f"HEAD:   {output('git', 'rev-parse', 'HEAD')}")
run("git", "status", "--short")

print()
print("PHASE 4 COMPLETE.")
print("Recent data was fetched and a NON-SUBMITTABLE candidate was generated.")
print("No competition decision was uploaded.")
